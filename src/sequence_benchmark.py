from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.utils import set_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DUST_TO_FAMILY = {
    "ISO 12103-1, A2 Fine Test Dust": "A2",
    "ISO 12103-1, A3 Medium Test Dust": "A3",
    "ISO 12103-1, A4 Coarse Test Dust": "A4",
}

FAMILY_ORDER = ("A2", "A3", "A4")
FEATURE_NAMES = (
    "Differential_pressure",
    "dDp_dt",
    "Flow_rate",
    "Dust_feed",
)


@dataclass
class SequenceSample:
    sequence_id: str
    source_split: str
    data_no: int
    family: str
    dust_label: str
    dust_feed: float
    time: np.ndarray
    dp: np.ndarray
    ddp_dt: np.ndarray
    flow: np.ndarray
    feed: np.ndarray
    health: np.ndarray
    rul_true: np.ndarray

    @property
    def features(self) -> np.ndarray:
        return np.column_stack([self.dp, self.ddp_dt, self.flow, self.feed]).astype(np.float32)

    @property
    def length(self) -> int:
        return int(self.time.shape[0])


@dataclass
class FoldSplit:
    train_ids: Tuple[str, ...]
    val_ids: Tuple[str, ...]
    test_ids: Tuple[str, ...]


@dataclass
class SequenceBenchmarkConfig:
    train_csv: str = "Train_Data_CSV.csv"
    test_csv: str = "Test_Data_CSV.csv"
    models: Tuple[str, ...] = ("GRU", "LSTM", "TCN", "ATTENTION")
    families: Tuple[str, ...] = FAMILY_ORDER
    kfolds: int = 5
    window: int = 32
    batch_size: int = 64
    epochs: int = 100
    patience: int = 10
    seed: int = 42
    outdir: str = "outputs/sequence_model_benchmark"
    device: str = "auto"
    learning_rate: float = 1e-3
    val_fraction: float = 0.10
    hidden_dim: int = 32
    recurrent_layers: int = 1
    attention_heads: int = 2
    attention_layers: int = 2
    tcn_channels: Tuple[int, ...] = (32, 32)
    dropout: float = 0.10
    weight_decay: float = 1e-5
    max_sequences_per_family: Optional[int] = None
    max_windows_per_sequence: Optional[int] = None
    representative_fold: int = 0
    dt: float = 0.1
    health_clean_dp: float = 25.0
    health_failure_dp: float = 600.0
    rul_ewma_points: int = 5
    rul_ewma_alpha: float = 0.30


class WindowDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]


class FeatureStandardizer:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, windows: np.ndarray):
        self.mean = windows.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        std = windows.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        self.std = std
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("standardizer must be fit before transform")
        return ((windows - self.mean[None, None, :]) / self.std[None, None, :]).astype(np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        if self.mean is None or self.std is None:
            raise RuntimeError("standardizer must be fit before export")
        return {
            "feature_names": list(FEATURE_NAMES),
            "mean": [float(x) for x in self.mean],
            "std": [float(x) for x in self.std],
        }


class RecurrentRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, kind: str):
        super().__init__()
        rnn_cls = nn.GRU if kind.upper() == "GRU" else nn.LSTM
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _hidden = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.chomp = padding
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp <= 0:
            return x
        return x[:, :, :-self.chomp]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self._trim(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self._trim(y)
        y = self.activation(y)
        y = self.dropout(y)
        return y + self.residual(x)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim: int, channels: Sequence[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = input_dim
        for depth, out_channels in enumerate(channels):
            layers.append(CausalConvBlock(in_channels, int(out_channels), kernel_size=3, dilation=2**depth, dropout=dropout))
            in_channels = int(out_channels)
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.network(x.transpose(1, 2))
        return self.head(y[:, :, -1]).squeeze(-1)


class AttentionRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int, layers: int, dropout: float, max_length: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.shape[1], :]
        h = self.encoder(h)
        h = self.norm(h)
        return self.head(h[:, -1, :]).squeeze(-1)


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_model_names(models: Iterable[str]) -> Tuple[str, ...]:
    normalized = []
    for model_name in models:
        name = str(model_name).strip().upper()
        if not name:
            continue
        if name not in {"GRU", "LSTM", "TCN", "ATTENTION"}:
            raise ValueError(f"unsupported model: {model_name}")
        normalized.append(name)
    if not normalized:
        raise ValueError("at least one model must be provided")
    return tuple(dict.fromkeys(normalized))


def normalize_family_names(families: Iterable[str]) -> Tuple[str, ...]:
    normalized = []
    for family in families:
        name = str(family).strip().upper()
        if not name:
            continue
        if name not in FAMILY_ORDER:
            raise ValueError(f"unsupported family: {family}")
        normalized.append(name)
    if not normalized:
        raise ValueError("at least one family must be provided")
    return tuple(dict.fromkeys(normalized))


def compute_health(dp: np.ndarray, clean_dp: float, failure_dp: float) -> np.ndarray:
    health = (float(failure_dp) - dp.astype(np.float32)) / max(float(failure_dp) - float(clean_dp), 1e-6)
    return np.clip(health, 0.0, 1.0).astype(np.float32)


def canonical_family(dust_label: str) -> str:
    if dust_label not in DUST_TO_FAMILY:
        raise ValueError(f"unknown dust label: {dust_label}")
    return DUST_TO_FAMILY[dust_label]


def _load_split_sequences(path: str, split_name: str, cfg: SequenceBenchmarkConfig) -> List[SequenceSample]:
    df = pd.read_csv(path).sort_values(["Data_No", "Time"]).reset_index(drop=True)
    sequences: List[SequenceSample] = []
    for data_no, group in df.groupby("Data_No", sort=True):
        g = group.sort_values("Time").reset_index(drop=True)
        dust_label = str(g["Dust"].iloc[0])
        family = canonical_family(dust_label)
        time_arr = g["Time"].to_numpy(dtype=np.float32)
        dp = g["Differential_pressure"].to_numpy(dtype=np.float32)
        flow = g["Flow_rate"].to_numpy(dtype=np.float32)
        feed = g["Dust_feed"].to_numpy(dtype=np.float32)
        ddp_dt = np.zeros_like(dp, dtype=np.float32)
        if len(dp) > 1:
            dt = np.diff(time_arr)
            dt = np.clip(dt, 1e-6, None)
            ddp_dt[1:] = np.diff(dp) / dt
        health = compute_health(dp, clean_dp=cfg.health_clean_dp, failure_dp=cfg.health_failure_dp)
        if "RUL" in g.columns:
            rul_true = g["RUL"].to_numpy(dtype=np.float32)
        else:
            rul_true = np.full_like(dp, np.nan, dtype=np.float32)
        sequences.append(
            SequenceSample(
                sequence_id=f"{split_name.lower()}_{int(data_no):02d}",
                source_split=split_name.lower(),
                data_no=int(data_no),
                family=family,
                dust_label=dust_label,
                dust_feed=float(feed[0]),
                time=time_arr,
                dp=dp,
                ddp_dt=ddp_dt,
                flow=flow,
                feed=feed,
                health=health,
                rul_true=rul_true,
            )
        )
    return sequences


def load_sequence_pool(cfg: SequenceBenchmarkConfig) -> Tuple[Dict[str, SequenceSample], Dict[str, object]]:
    sequences = _load_split_sequences(cfg.train_csv, "train", cfg) + _load_split_sequences(cfg.test_csv, "test", cfg)
    sequence_map = {sample.sequence_id: sample for sample in sequences}
    summary_rows = []
    feed_sets_by_split: Dict[str, Dict[str, List[float]]] = {"train": {}, "test": {}}
    for split_name in ("train", "test"):
        split_sequences = [sample for sample in sequences if sample.source_split == split_name]
        for family in FAMILY_ORDER:
            family_sequences = [sample for sample in split_sequences if sample.family == family]
            feed_sets = sorted({round(float(sample.dust_feed), 6) for sample in family_sequences})
            feed_sets_by_split[split_name][family] = feed_sets
            summary_rows.append(
                {
                    "source_split": split_name,
                    "family": family,
                    "sequence_count": int(len(family_sequences)),
                    "row_count": int(sum(sample.length for sample in family_sequences)),
                    "feed_count": int(len(feed_sets)),
                    "feed_values": ",".join(f"{value:.6f}" for value in feed_sets),
                }
            )
    a2_train = set(feed_sets_by_split["train"]["A2"])
    a2_test = set(feed_sets_by_split["test"]["A2"])
    summary = {
        "sequence_count": int(len(sequences)),
        "sequence_ids_unique": int(len(sequence_map) == len(sequences)),
        "feature_names": list(FEATURE_NAMES),
        "health_formula": "clip((600 - Differential_pressure) / (600 - 25), 0, 1)",
        "feed_sets_by_split": feed_sets_by_split,
        "a2_feed_overlap": [float(x) for x in sorted(a2_train & a2_test)],
        "a2_feed_train_only": [float(x) for x in sorted(a2_train - a2_test)],
        "a2_feed_test_only": [float(x) for x in sorted(a2_test - a2_train)],
        "rows": summary_rows,
    }
    return sequence_map, summary


def _select_evenly_spaced_indices(length: int, limit: Optional[int]) -> np.ndarray:
    if limit is None or limit <= 0 or length <= limit:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, num=limit, dtype=np.int64))


def _limit_family_sequences(sequence_ids: Sequence[str], sequence_map: Dict[str, SequenceSample], limit: Optional[int], seed: int) -> List[str]:
    ids = list(sequence_ids)
    if limit is None or limit <= 0 or len(ids) <= limit:
        return sorted(ids)
    rng = np.random.default_rng(seed)
    train_ids = [sid for sid in ids if sequence_map[sid].source_split == "train"]
    test_ids = [sid for sid in ids if sequence_map[sid].source_split == "test"]
    rng.shuffle(train_ids)
    rng.shuffle(test_ids)
    total = len(ids)
    train_target = min(len(train_ids), max(1 if train_ids else 0, round(limit * len(train_ids) / total)))
    test_target = min(len(test_ids), max(1 if test_ids else 0, limit - train_target))
    selected = train_ids[:train_target] + test_ids[:test_target]
    remaining = [sid for sid in ids if sid not in set(selected)]
    if len(selected) < limit:
        selected.extend(remaining[: limit - len(selected)])
    return sorted(selected[:limit])


def build_grouped_folds(
    family: str,
    sequence_map: Dict[str, SequenceSample],
    cfg: SequenceBenchmarkConfig,
) -> Tuple[List[FoldSplit], List[str]]:
    family_ids = [sid for sid, sample in sequence_map.items() if sample.family == family]
    family_seed = cfg.seed + sum(ord(ch) for ch in family)
    selected_ids = _limit_family_sequences(family_ids, sequence_map, cfg.max_sequences_per_family, seed=family_seed)
    train_ids = [sid for sid in selected_ids if sequence_map[sid].source_split == "train"]
    test_ids = [sid for sid in selected_ids if sequence_map[sid].source_split == "test"]
    max_k = min(len(selected_ids), len(train_ids) if train_ids else len(selected_ids), len(test_ids) if test_ids else len(selected_ids))
    effective_k = max(2, min(int(cfg.kfolds), max_k))
    buckets = [[] for _ in range(effective_k)]
    for split_ids, seed_offset in ((train_ids, 11), (test_ids, 23)):
        rng = np.random.default_rng(cfg.seed + seed_offset + sum(ord(ch) for ch in family))
        shuffled = list(split_ids)
        rng.shuffle(shuffled)
        for idx, sid in enumerate(shuffled):
            buckets[idx % effective_k].append(sid)
    folds = []
    all_ids = set(selected_ids)
    for fold_index in range(effective_k):
        fold_test_ids = tuple(sorted(buckets[fold_index]))
        outer_train_ids = sorted(all_ids - set(fold_test_ids))
        val_ids = select_validation_ids(outer_train_ids, sequence_map, cfg.val_fraction, seed=cfg.seed + fold_index * 17)
        train_fold_ids = tuple(sorted(set(outer_train_ids) - set(val_ids)))
        folds.append(FoldSplit(train_ids=train_fold_ids, val_ids=tuple(sorted(val_ids)), test_ids=fold_test_ids))
    return folds, selected_ids


def select_validation_ids(
    train_pool_ids: Sequence[str],
    sequence_map: Dict[str, SequenceSample],
    val_fraction: float,
    seed: int,
) -> List[str]:
    if len(train_pool_ids) <= 1 or val_fraction <= 0.0:
        return []
    rng = np.random.default_rng(seed)
    buckets = {
        "train": [sid for sid in train_pool_ids if sequence_map[sid].source_split == "train"],
        "test": [sid for sid in train_pool_ids if sequence_map[sid].source_split == "test"],
    }
    val_ids: List[str] = []
    for split_name, split_ids in buckets.items():
        del split_name
        shuffled = list(split_ids)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            continue
        count = int(round(len(shuffled) * float(val_fraction)))
        count = max(1, count) if len(shuffled) >= 4 else (1 if len(shuffled) >= 2 else 0)
        count = min(count, len(shuffled) - 1)
        val_ids.extend(shuffled[:count])
    if not val_ids:
        shuffled = list(train_pool_ids)
        rng.shuffle(shuffled)
        val_ids = shuffled[:1]
    if len(val_ids) >= len(train_pool_ids):
        val_ids = val_ids[: max(0, len(train_pool_ids) - 1)]
    return sorted(dict.fromkeys(val_ids))


def build_window_arrays(
    sequence_ids: Sequence[str],
    sequence_map: Dict[str, SequenceSample],
    cfg: SequenceBenchmarkConfig,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    windows = []
    targets = []
    rows: List[Dict[str, object]] = []
    for sequence_id in sequence_ids:
        sample = sequence_map[sequence_id]
        features = sample.features
        indices = _select_evenly_spaced_indices(sample.length, cfg.max_windows_per_sequence)
        for end_idx in indices:
            start = int(end_idx) - int(cfg.window) + 1
            if start < 0:
                pad = np.repeat(features[0:1], repeats=-start, axis=0)
                core = features[0 : int(end_idx) + 1]
                window = np.concatenate([pad, core], axis=0)
            else:
                window = features[start : int(end_idx) + 1]
            windows.append(window.astype(np.float32))
            targets.append(float(sample.health[int(end_idx)]))
            rows.append(
                {
                    "sequence_id": sample.sequence_id,
                    "source_split": sample.source_split,
                    "data_no": int(sample.data_no),
                    "family": sample.family,
                    "dust_feed": float(sample.feed[int(end_idx)]),
                    "time": float(sample.time[int(end_idx)]),
                    "end_index": int(end_idx),
                    "health_true": float(sample.health[int(end_idx)]),
                    "rul_true": float(sample.rul_true[int(end_idx)]) if np.isfinite(sample.rul_true[int(end_idx)]) else np.nan,
                }
            )
    if not windows:
        raise ValueError("no windows were generated for the requested split")
    return np.stack(windows).astype(np.float32), np.asarray(targets, dtype=np.float32), pd.DataFrame(rows)


def build_model(model_name: str, cfg: SequenceBenchmarkConfig) -> nn.Module:
    model_name = model_name.upper()
    input_dim = len(FEATURE_NAMES)
    if model_name in {"GRU", "LSTM"}:
        return RecurrentRegressor(
            input_dim=input_dim,
            hidden_dim=int(cfg.hidden_dim),
            num_layers=int(cfg.recurrent_layers),
            dropout=float(cfg.dropout),
            kind=model_name,
        )
    if model_name == "TCN":
        return TCNRegressor(input_dim=input_dim, channels=cfg.tcn_channels, dropout=float(cfg.dropout))
    if model_name == "ATTENTION":
        return AttentionRegressor(
            input_dim=input_dim,
            hidden_dim=int(cfg.hidden_dim),
            heads=int(cfg.attention_heads),
            layers=int(cfg.attention_layers),
            dropout=float(cfg.dropout),
            max_length=int(cfg.window),
        )
    raise ValueError(f"unsupported model: {model_name}")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred), dtype=np.float64)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred), dtype=np.float64))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.square(y_true - float(np.mean(y_true))), dtype=np.float64))
    if denom <= 1e-12:
        return 0.0
    resid = float(np.sum(np.square(y_true - y_pred), dtype=np.float64))
    return float(1.0 - resid / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.clip(denom, 1e-6, None)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom, dtype=np.float64))


def compute_health_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "health_mae": mae(y_true, y_pred),
        "health_rmse": rmse(y_true, y_pred),
        "health_r2": r2_score(y_true, y_pred),
    }


def derive_rul_from_health(health_pred: np.ndarray, cfg: SequenceBenchmarkConfig) -> np.ndarray:
    rul_hat = np.zeros_like(health_pred, dtype=np.float32)
    alpha = float(cfg.rul_ewma_alpha)
    lookback = max(1, int(cfg.rul_ewma_points))
    positive_drops: List[float] = []
    for idx, health_value in enumerate(health_pred):
        if idx == 0:
            positive_drops.append(0.0)
        else:
            drop = max(0.0, float(health_pred[idx - 1] - health_pred[idx]))
            positive_drops.append(drop)
        recent = positive_drops[max(0, len(positive_drops) - lookback) :]
        weights = np.asarray([alpha * ((1.0 - alpha) ** (len(recent) - 1 - j)) for j in range(len(recent))], dtype=np.float32)
        if float(weights.sum()) <= 1e-12:
            decay = 0.0
        else:
            weights = weights / float(weights.sum())
            decay = float(np.dot(weights, np.asarray(recent, dtype=np.float32)))
        rul_hat[idx] = float(health_value) / max(decay, 1e-4) * float(cfg.dt)
    return rul_hat


def compute_rul_metrics(prediction_df: pd.DataFrame) -> Dict[str, float]:
    valid = prediction_df[(prediction_df["source_split"] == "test") & prediction_df["rul_true"].notna() & prediction_df["rul_pred"].notna()]
    if valid.empty:
        return {
            "rul_mae": math.nan,
            "rul_rmse": math.nan,
            "rul_smape": math.nan,
        }
    y_true = valid["rul_true"].to_numpy(dtype=np.float32)
    y_pred = valid["rul_pred"].to_numpy(dtype=np.float32)
    return {
        "rul_mae": mae(y_true, y_pred),
        "rul_rmse": rmse(y_true, y_pred),
        "rul_smape": smape(y_true, y_pred),
    }


def _predict_batches(model: nn.Module, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            pred = model(batch).detach().cpu().numpy()
            outputs.append(pred)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def train_fold_model(
    model_name: str,
    family: str,
    fold_index: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    standardizer: FeatureStandardizer,
    cfg: SequenceBenchmarkConfig,
    device: torch.device,
    artifact_dir: Path,
) -> Tuple[nn.Module, Dict[str, object]]:
    model = build_model(model_name, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(WindowDataset(train_x, train_y), batch_size=int(cfg.batch_size), shuffle=True, num_workers=0)
    use_validation = len(val_x) > 0
    best_state = None
    best_val_rmse = math.inf
    best_epoch = 0
    epochs_without_improve = 0
    history_rows = []
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        if use_validation:
            val_pred = _predict_batches(model, val_x, batch_size=int(cfg.batch_size), device=device)
            val_rmse = rmse(val_y, val_pred)
        else:
            train_pred = _predict_batches(model, train_x, batch_size=int(cfg.batch_size), device=device)
            val_rmse = rmse(train_y, train_pred)
        history_rows.append({"epoch": epoch, "train_mse": train_loss, "val_rmse": float(val_rmse)})
        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = float(val_rmse)
            best_epoch = int(epoch)
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if epochs_without_improve >= int(cfg.patience):
            break
    if best_state is None:
        best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(best_state)
    history_path = artifact_dir / f"fold_{fold_index:02d}_history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    checkpoint_path = artifact_dir / f"fold_{fold_index:02d}_best.pt"
    torch.save(
        {
            "model_name": model_name,
            "family": family,
            "fold_index": int(fold_index),
            "feature_names": list(FEATURE_NAMES),
            "state_dict": model.state_dict(),
            "standardizer": standardizer.to_dict(),
            "config": asdict(cfg),
            "best_epoch": int(best_epoch),
            "best_val_rmse": float(best_val_rmse),
        },
        checkpoint_path,
    )
    summary = {
        "model_name": model_name,
        "family": family,
        "fold_index": int(fold_index),
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
    }
    with (artifact_dir / f"fold_{fold_index:02d}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    return model, summary


def evaluate_fold_predictions(
    model: nn.Module,
    test_x: np.ndarray,
    test_meta: pd.DataFrame,
    cfg: SequenceBenchmarkConfig,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    pred = np.clip(_predict_batches(model, test_x, batch_size=int(cfg.batch_size), device=device), 0.0, 1.0)
    pred_df = test_meta.copy()
    pred_df["health_pred"] = pred
    pred_df["rul_pred"] = np.nan
    for sequence_id, group in pred_df.groupby("sequence_id", sort=False):
        ordered = group.sort_values("end_index")
        if ordered["source_split"].iloc[0] != "test":
            continue
        rul_hat = derive_rul_from_health(ordered["health_pred"].to_numpy(dtype=np.float32), cfg)
        pred_df.loc[ordered.index, "rul_pred"] = rul_hat.astype(np.float32)
        del sequence_id
    health_metrics = compute_health_metrics(
        pred_df["health_true"].to_numpy(dtype=np.float32),
        pred_df["health_pred"].to_numpy(dtype=np.float32),
    )
    rul_metrics = compute_rul_metrics(pred_df)
    return pred_df, health_metrics, rul_metrics


def _safe_metric_mean(values: Sequence[float]) -> float:
    valid = [float(value) for value in values if not math.isnan(float(value))]
    if not valid:
        return math.nan
    return float(np.mean(np.asarray(valid, dtype=np.float64)))


def _safe_metric_std(values: Sequence[float]) -> float:
    valid = [float(value) for value in values if not math.isnan(float(value))]
    if not valid:
        return math.nan
    return float(np.std(np.asarray(valid, dtype=np.float64)))


def aggregate_family_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, model_name), group in fold_df.groupby(["family", "model"], sort=True):
        row = {
            "family": family,
            "model": model_name,
            "fold_count": int(len(group)),
        }
        for metric in ("health_mae", "health_rmse", "health_r2", "rul_mae", "rul_rmse", "rul_smape"):
            row[f"{metric}_mean"] = _safe_metric_mean(group[metric].tolist())
            row[f"{metric}_std"] = _safe_metric_std(group[metric].tolist())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["family", "health_rmse_mean", "rul_rmse_mean", "model"]).reset_index(drop=True)


def aggregate_overall_metrics(family_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in family_df.groupby("model", sort=True):
        row = {
            "model": model_name,
            "family_count": int(len(group)),
        }
        for metric in ("health_mae_mean", "health_rmse_mean", "health_r2_mean", "rul_mae_mean", "rul_rmse_mean", "rul_smape_mean"):
            row[metric.replace("_mean", "")] = _safe_metric_mean(group[metric].tolist())
        rows.append(row)
    overall = pd.DataFrame(rows)
    if overall.empty:
        return overall
    return overall.sort_values(["health_rmse", "rul_rmse", "health_mae", "model"]).reset_index(drop=True)


def add_rankings(overall_df: pd.DataFrame) -> pd.DataFrame:
    ranked = overall_df.copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=np.int64)
    ranked["is_best_model"] = 0
    if not ranked.empty:
        ranked.loc[ranked.index[0], "is_best_model"] = 1
    return ranked


def _sample_scatter_points(df: pd.DataFrame, metric_col: str, target_col: str, seed: int, max_points: int = 4000) -> pd.DataFrame:
    subset = df[[metric_col, target_col, "model"]].dropna()
    if len(subset) <= max_points:
        return subset
    return subset.sample(n=max_points, random_state=seed)


def plot_metric_bars(overall_df: pd.DataFrame, outdir: Path):
    if overall_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(overall_df))
    axes[0].bar(x - 0.2, overall_df["health_rmse"], width=0.4, label="RMSE")
    axes[0].bar(x + 0.2, overall_df["health_mae"], width=0.4, label="MAE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(overall_df["model"])
    axes[0].set_title("Health Metrics by Model")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    axes[1].bar(x - 0.2, overall_df["rul_rmse"], width=0.4, label="RMSE")
    axes[1].bar(x + 0.2, overall_df["rul_mae"], width=0.4, label="MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(overall_df["model"])
    axes[1].set_title("Derived RUL Metrics by Model")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outdir / "metric_bars_overall.png", dpi=200)
    plt.close(fig)


def plot_scatter_grid(pred_df: pd.DataFrame, outdir: Path, value_col: str, pred_col: str, title_prefix: str, seed: int):
    models = sorted(pred_df["model"].unique().tolist())
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
    for ax, model_name in zip(axes[0], models):
        subset = pred_df[pred_df["model"] == model_name]
        sampled = _sample_scatter_points(subset, value_col, pred_col, seed=seed)
        ax.scatter(sampled[value_col], sampled[pred_col], s=10, alpha=0.35)
        lo = min(float(sampled[value_col].min()), float(sampled[pred_col].min()))
        hi = max(float(sampled[value_col].max()), float(sampled[pred_col].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
        ax.set_title(model_name)
        ax.set_xlabel(f"True {title_prefix}")
        ax.set_ylabel(f"Predicted {title_prefix}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"scatter_{pred_col}.png", dpi=200)
    plt.close(fig)


def plot_representative_sequences(pred_df: pd.DataFrame, outdir: Path):
    for (family, model_name), group in pred_df.groupby(["family", "model"], sort=True):
        health_seq = group.sort_values(["source_split", "sequence_id", "end_index"])
        if health_seq.empty:
            continue
        health_sequence_ids = health_seq["sequence_id"].unique().tolist()
        representative_id = None
        for sequence_id in health_sequence_ids:
            sequence_rows = health_seq[health_seq["sequence_id"] == sequence_id]
            if sequence_rows["source_split"].iloc[0] == "test":
                representative_id = sequence_id
                break
        if representative_id is None:
            representative_id = health_sequence_ids[0]
        rep = health_seq[health_seq["sequence_id"] == representative_id].sort_values("end_index")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rep["time"], rep["health_true"], label="true health", linewidth=2.0)
        ax.plot(rep["time"], rep["health_pred"], label="pred health", linewidth=2.0)
        ax.set_title(f"{family} {model_name} held-out health trajectory")
        ax.set_xlabel("Time")
        ax.set_ylabel("Normalized health")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"health_curve_{family}_{model_name}.png", dpi=200)
        plt.close(fig)
        if rep["source_split"].iloc[0] == "test" and rep["rul_true"].notna().any():
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(rep["time"], rep["rul_true"], label="true RUL", linewidth=2.0)
            ax.plot(rep["time"], rep["rul_pred"], label="derived RUL", linewidth=2.0)
            ax.set_title(f"{family} {model_name} held-out derived RUL")
            ax.set_xlabel("Time")
            ax.set_ylabel("RUL")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / f"rul_curve_{family}_{model_name}.png", dpi=200)
            plt.close(fig)


def print_rankings(family_df: pd.DataFrame, ranking_df: pd.DataFrame, out_root: Path, dataset_summary: Dict[str, object]):
    a2_train_only = dataset_summary.get("a2_feed_train_only", [])
    a2_test_only = dataset_summary.get("a2_feed_test_only", [])
    print("dataset summary:")
    print(f"  total sequences: {dataset_summary['sequence_count']}")
    print(f"  A2 train-only feeds: {a2_train_only}")
    print(f"  A2 test-only feeds: {a2_test_only}")
    for family, group in family_df.groupby("family", sort=True):
        print(f"{family} ranking:")
        ranked = group.sort_values(["health_rmse_mean", "rul_rmse_mean", "model"]).reset_index(drop=True)
        for idx, row in ranked.iterrows():
            print(
                f"  {idx + 1}. {row['model']} "
                f"health_rmse={float(row['health_rmse_mean']):.4f} "
                f"health_mae={float(row['health_mae_mean']):.4f} "
                f"rul_rmse={float(row['rul_rmse_mean']):.4f}"
            )
    if not ranking_df.empty:
        best = ranking_df.iloc[0]
        print("overall best model:")
        print(
            f"  {best['model']} "
            f"health_rmse={float(best['health_rmse']):.4f} "
            f"health_mae={float(best['health_mae']):.4f} "
            f"rul_rmse={float(best['rul_rmse']):.4f}"
        )
        print("overall summary:")
        for _, row in ranking_df.iterrows():
            print(
                f"  {row['model']}: "
                f"health_rmse={float(row['health_rmse']):.4f}, "
                f"health_mae={float(row['health_mae']):.4f}, "
                f"health_r2={float(row['health_r2']):.4f}, "
                f"rul_rmse={float(row['rul_rmse']):.4f}, "
                f"rul_mae={float(row['rul_mae']):.4f}"
            )
    print(f"output directory: {out_root}")


def run_benchmark(cfg: SequenceBenchmarkConfig) -> Dict[str, object]:
    cfg.models = normalize_model_names(cfg.models)
    cfg.families = normalize_family_names(cfg.families)
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    out_root = Path(cfg.outdir) / time.strftime("%Y%m%d_%H%M%S")
    artifact_root = out_root / "artifacts"
    plots_root = out_root / "plots"
    artifact_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    sequence_map, dataset_summary = load_sequence_pool(cfg)
    with (out_root / "benchmark_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=True, indent=2)
    with (out_root / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, ensure_ascii=True, indent=2)
    pd.DataFrame(dataset_summary["rows"]).to_csv(out_root / "dataset_summary.csv", index=False)

    fold_rows: List[Dict[str, object]] = []
    all_predictions: List[pd.DataFrame] = []
    for family in cfg.families:
        folds, selected_ids = build_grouped_folds(family, sequence_map, cfg)
        print(f"{family}: {len(selected_ids)} sequences, {len(folds)} folds")
        for model_name in cfg.models:
            family_model_dir = artifact_root / family / model_name.lower()
            family_model_dir.mkdir(parents=True, exist_ok=True)
            print(f"training {model_name} for {family}")
            for fold_index, fold in enumerate(folds):
                train_x_raw, train_y, _train_meta = build_window_arrays(fold.train_ids, sequence_map, cfg)
                if fold.val_ids:
                    val_x_raw, val_y, _val_meta = build_window_arrays(fold.val_ids, sequence_map, cfg)
                else:
                    val_x_raw = np.empty((0, cfg.window, len(FEATURE_NAMES)), dtype=np.float32)
                    val_y = np.empty((0,), dtype=np.float32)
                test_x_raw, _test_y, test_meta = build_window_arrays(fold.test_ids, sequence_map, cfg)
                standardizer = FeatureStandardizer().fit(train_x_raw)
                train_x = standardizer.transform(train_x_raw)
                val_x = standardizer.transform(val_x_raw) if len(val_x_raw) else val_x_raw
                test_x = standardizer.transform(test_x_raw)
                model, train_summary = train_fold_model(
                    model_name=model_name,
                    family=family,
                    fold_index=fold_index,
                    train_x=train_x,
                    train_y=train_y,
                    val_x=val_x,
                    val_y=val_y,
                    standardizer=standardizer,
                    cfg=cfg,
                    device=device,
                    artifact_dir=family_model_dir,
                )
                pred_df, health_metrics, rul_metrics = evaluate_fold_predictions(
                    model=model,
                    test_x=test_x,
                    test_meta=test_meta,
                    cfg=cfg,
                    device=device,
                )
                pred_df["model"] = model_name
                pred_df["fold"] = int(fold_index)
                pred_df.to_csv(family_model_dir / f"fold_{fold_index:02d}_predictions.csv", index=False)
                all_predictions.append(pred_df)
                fold_row = {
                    "family": family,
                    "model": model_name,
                    "fold": int(fold_index),
                    "train_sequence_count": int(len(fold.train_ids)),
                    "val_sequence_count": int(len(fold.val_ids)),
                    "test_sequence_count": int(len(fold.test_ids)),
                    "best_epoch": int(train_summary["best_epoch"]),
                    "best_val_rmse": float(train_summary["best_val_rmse"]),
                }
                fold_row.update(health_metrics)
                fold_row.update(rul_metrics)
                fold_rows.append(fold_row)
                print(
                    f"  fold {fold_index + 1}/{len(folds)} "
                    f"health_rmse={float(health_metrics['health_rmse']):.4f} "
                    f"rul_rmse={float(rul_metrics['rul_rmse']):.4f}"
                )

    fold_df = pd.DataFrame(fold_rows).sort_values(["family", "model", "fold"]).reset_index(drop=True)
    family_df = aggregate_family_metrics(fold_df)
    overall_df = aggregate_overall_metrics(family_df)
    ranking_df = add_rankings(overall_df)

    fold_df.to_csv(out_root / "metrics_by_fold.csv", index=False)
    family_df.to_csv(out_root / "metrics_by_family.csv", index=False)
    overall_df.to_csv(out_root / "metrics_overall.csv", index=False)
    ranking_df.to_csv(out_root / "model_rankings.csv", index=False)
    overall_payload = {
        "best_model": None if ranking_df.empty else str(ranking_df.iloc[0]["model"]),
        "models": [] if ranking_df.empty else ranking_df.to_dict(orient="records"),
        "dataset_summary_path": str(out_root / "dataset_summary.json"),
    }
    with (out_root / "metrics_overall.json").open("w", encoding="utf-8") as f:
        json.dump(overall_payload, f, ensure_ascii=True, indent=2)

    pred_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    if not pred_df.empty:
        plot_metric_bars(ranking_df, plots_root)
        plot_scatter_grid(pred_df, plots_root, value_col="health_true", pred_col="health_pred", title_prefix="health", seed=cfg.seed)
        rul_subset = pred_df[pred_df["rul_true"].notna() & pred_df["rul_pred"].notna()].copy()
        if not rul_subset.empty:
            plot_scatter_grid(rul_subset, plots_root, value_col="rul_true", pred_col="rul_pred", title_prefix="RUL", seed=cfg.seed + 13)
        plot_representative_sequences(pred_df, plots_root)

    print_rankings(family_df, ranking_df, out_root, dataset_summary)
    return {
        "out_root": str(out_root),
        "dataset_summary": dataset_summary,
        "metrics_by_fold": fold_df,
        "metrics_by_family": family_df,
        "metrics_overall": ranking_df,
    }
