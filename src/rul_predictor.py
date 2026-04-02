from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _PaperGRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


@dataclass
class _PaperGRUState:
    model: _PaperGRURegressor
    mean: np.ndarray
    std: np.ndarray
    total_life: float
    train_data_no: int


class RULPredictorWrapper:
    """
    Paper-aligned RUL predictor.

    Priority:
    1. Explicit external artifact mode if requested.
    2. Paper-style internal GRU baseline:
       - only Differential_pressure by default
       - train/validation on machine 5 (Data_No=18) with 80/20 chronological split
    3. Linear fallback for safety/debugging.
    """

    def __init__(self, artifact_dir: str | None, window: int, features: list[str], cfg: Optional[Any] = None):
        self.window = int(window)
        self.features = list(features)
        self.cfg = cfg
        self.use_artifact = False
        self.use_internal_gru = False
        self.p = None
        self.paper_gru_state: Optional[_PaperGRUState] = None
        self.mode = str(getattr(cfg, "RUL_PREDICTOR_MODE", "paper_gru")).lower() if cfg is not None else "paper_gru"

        if self.mode == "artifact" and artifact_dir and Path(artifact_dir).exists():
            try:
                from rul_model_artifact.example_usage import RULPredictor as ArtifactPredictor
                self.p = ArtifactPredictor(artifact_dir=artifact_dir)
                self.window = int(self.p.window)
                self.features = list(self.p.features)
                self.use_artifact = True
                self.mode = "artifact"
                return
            except Exception as exc:
                print(f"artifact RUL predictor unavailable, falling back to paper_gru: {exc}")

        try:
            self.paper_gru_state = self._train_paper_gru()
            self.use_internal_gru = self.paper_gru_state is not None
            if self.use_internal_gru:
                self.mode = "paper_gru"
        except Exception as exc:
            self.paper_gru_state = None
            self.use_internal_gru = False
            print(f"paper_gru RUL predictor unavailable, falling back to linear: {exc}")
            self.mode = "linear"

    def _build_windows(self, values: np.ndarray) -> np.ndarray:
        seq = np.asarray(values, dtype=np.float32)
        if seq.ndim == 1:
            seq = seq[:, None]
        n, f = seq.shape
        windows = np.zeros((n, self.window, f), dtype=np.float32)
        for idx in range(n):
            start = idx - self.window + 1
            if start < 0:
                pad = np.repeat(seq[0:1], repeats=-start, axis=0)
                core = seq[0:idx + 1]
                windows[idx] = np.concatenate([pad, core], axis=0)
            else:
                windows[idx] = seq[start:idx + 1]
        return windows

    def _train_paper_gru(self) -> Optional[_PaperGRUState]:
        if self.cfg is None:
            return None

        test_csv = Path(getattr(self.cfg, "TEST_CSV", "Test_Data_CSV.csv"))
        if not test_csv.exists():
            raise FileNotFoundError(f"paper_gru requires TEST_CSV: {test_csv}")

        train_data_no = int(getattr(self.cfg, "RUL_TRAIN_DATA_NO", 18))
        val_ratio = float(getattr(self.cfg, "RUL_VAL_RATIO", 0.2))
        batch_size = int(getattr(self.cfg, "RUL_GRU_BATCH_SIZE", 1024))
        epochs = int(getattr(self.cfg, "RUL_GRU_EPOCHS", 250))
        lr = float(getattr(self.cfg, "RUL_GRU_LR", 1e-3))
        hidden_dim = int(getattr(self.cfg, "RUL_GRU_HIDDEN_DIM", 40))
        num_layers = int(getattr(self.cfg, "RUL_GRU_NUM_LAYERS", 1))
        dropout = float(getattr(self.cfg, "RUL_GRU_DROPOUT", 0.25))
        seed = int(getattr(self.cfg, "SEED", 42))

        df = pd.read_csv(test_csv)
        required_cols = ["Data_No", "Time", "RUL", *self.features]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"paper_gru missing required columns in {test_csv}: {missing}")

        g = df[df["Data_No"] == train_data_no].copy()
        if g.empty:
            raise ValueError(f"paper_gru training Data_No={train_data_no} not found in {test_csv}")
        g = g.sort_values("Time").reset_index(drop=True)
        if len(g) < max(self.window + 8, 32):
            raise ValueError(f"paper_gru training sequence too short: len={len(g)}")

        split_idx = int(round(len(g) * (1.0 - val_ratio)))
        split_idx = min(max(split_idx, self.window), len(g) - 1)

        raw_x = g[self.features].to_numpy(dtype=np.float32)
        total_life = float((g["Time"] + g["RUL"]).iloc[0])
        y = np.clip(g["RUL"].to_numpy(dtype=np.float32) / max(total_life, 1e-6), 0.0, 1.0)

        mean = raw_x[:split_idx].mean(axis=0, keepdims=True).astype(np.float32)
        std = raw_x[:split_idx].std(axis=0, keepdims=True).astype(np.float32)
        std[std < 1e-6] = 1.0
        norm_x = (raw_x - mean) / std
        windows = self._build_windows(norm_x)

        x_train = torch.tensor(windows[:split_idx], dtype=torch.float32)
        y_train = torch.tensor(y[:split_idx], dtype=torch.float32)
        x_val = torch.tensor(windows[split_idx:], dtype=torch.float32)
        y_val = torch.tensor(y[split_idx:], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=min(batch_size, max(len(x_train), 1)),
            shuffle=True,
        )

        torch.manual_seed(seed)
        model = _PaperGRURegressor(len(self.features), hidden_dim, num_layers, dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_state = None
        best_val = float("inf")
        patience = max(5, epochs // 4)
        stale = 0

        for _ in range(max(epochs, 1)):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = float(criterion(val_pred, y_val).item())
            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        print(
            "paper_gru RUL predictor trained:",
            f"train_data_no={train_data_no}",
            f"features={self.features}",
            f"split={split_idx}/{len(g) - split_idx}",
            f"val_mse={best_val:.6f}",
        )

        return _PaperGRUState(
            model=model,
            mean=mean.squeeze(0),
            std=std.squeeze(0),
            total_life=total_life,
            train_data_no=train_data_no,
        )

    def _predict_internal_gru(self, window_array: np.ndarray) -> float:
        if self.paper_gru_state is None:
            raise RuntimeError("paper_gru state is not initialized")
        x = np.asarray(window_array, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[-1] != len(self.features):
            raise ValueError(
                "paper_gru input feature mismatch: "
                f"expected={len(self.features)} got={x.shape[-1]}"
            )
        norm_x = (x - self.paper_gru_state.mean[None, :]) / self.paper_gru_state.std[None, :]
        xt = torch.tensor(norm_x[None], dtype=torch.float32)
        with torch.no_grad():
            pred = float(self.paper_gru_state.model(xt).item())
        return float(np.clip(pred, 0.0, 1.0))

    def predict(self, data_no: int, window_array: np.ndarray, t_idx: int, lifespan: int) -> float:
        """
        Returns normalized RUL in [0,1].
        """
        if self.use_artifact:
            return float(self.p.predict(data_no=int(data_no), window_array=window_array))
        if self.use_internal_gru:
            return self._predict_internal_gru(window_array)
        denom = max(lifespan - 1, 1)
        return float(np.clip((denom - t_idx) / denom, 0.0, 1.0))
