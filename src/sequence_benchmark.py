from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
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
PRETRAIN_TARGET_NAMES = ("next_dp", "next_flow")


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
    dp_proxy_health: np.ndarray
    health_remaining_true: np.ndarray
    rul_true: np.ndarray
    total_life: np.ndarray

    @property
    def features(self) -> np.ndarray:
        return np.column_stack([self.dp, self.ddp_dt, self.flow, self.feed]).astype(np.float32)

    @property
    def length(self) -> int:
        return int(self.time.shape[0])

    @property
    def total_life_scalar(self) -> float:
        finite = self.total_life[np.isfinite(self.total_life)]
        return float(finite[0]) if finite.size else math.nan


@dataclass
class FoldSplit:
    train_ids: Tuple[str, ...]
    val_ids: Tuple[str, ...]
    test_ids: Tuple[str, ...]


@dataclass
class SequenceBenchmarkConfig:
    train_csv: str = "Train_Data_CSV.csv"
    test_csv: str = "Test_Data_CSV.csv"
    models: Tuple[str, ...] = ("LSTM",)
    families: Tuple[str, ...] = FAMILY_ORDER
    kfolds: int = 5
    window: int = 32
    batch_size: int = 64
    epochs: int = 100
    patience: int = 10
    pretrain_epochs: int = 20
    pretrain_patience: int = 5
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
    dt: float = 0.1
    health_clean_dp: float = 25.0
    health_failure_dp: float = 600.0
    health_target_type: str = "remaining_life_fraction"
    pretrain_task: str = "next_step_dp_flow"
    rul_derivation: str = "health_hat_times_total_life"
    add_tail_reference_rows: bool = True
    stage_weight_alpha_a2: float = 1.0
    stage_weight_alpha_a3: float = 1.5
    stage_weight_alpha_a4: float = 1.0
    a2_long_life_weight: float = 2.0
    enable_monotone_projection: bool = True
    enable_family_tuning: bool = True
    tuning_windows: Tuple[int, ...] = (32, 48)
    tuning_hidden_dims: Tuple[int, ...] = (32, 64)
    tuning_dropouts: Tuple[float, ...] = (0.05, 0.10)
    baseline_run_root: str = "outputs/sequence_model_benchmark/20260313_141821"


@dataclass(frozen=True)
class FamilyHyperParams:
    window: int
    hidden_dim: int
    dropout: float


class WindowDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, weights: Optional[np.ndarray] = None):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        if weights is None:
            weights = np.ones((len(features),), dtype=np.float32)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx], self.weights[idx]


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

    def transform_dp_flow_targets(self, targets: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("standardizer must be fit before transform")
        scaled = targets.copy().astype(np.float32)
        scaled[:, 0] = (scaled[:, 0] - self.mean[0]) / self.std[0]
        scaled[:, 1] = (scaled[:, 1] - self.mean[2]) / self.std[2]
        return scaled

    def to_dict(self) -> Dict[str, List[float]]:
        if self.mean is None or self.std is None:
            raise RuntimeError("standardizer must be fit before export")
        return {
            "feature_names": list(FEATURE_NAMES),
            "mean": [float(x) for x in self.mean],
            "std": [float(x) for x in self.std],
        }


class RecurrentEncoder(nn.Module):
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
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _hidden = self.rnn(x)
        return out[:, -1, :]


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


class TCNEncoder(nn.Module):
    def __init__(self, input_dim: int, channels: Sequence[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = input_dim
        for depth, out_channels in enumerate(channels):
            layers.append(CausalConvBlock(in_channels, int(out_channels), kernel_size=3, dilation=2**depth, dropout=dropout))
            in_channels = int(out_channels)
        self.network = nn.Sequential(*layers)
        self.output_dim = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.network(x.transpose(1, 2))
        return y[:, :, -1]


class AttentionEncoder(nn.Module):
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
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.shape[1], :]
        h = self.encoder(h)
        return self.norm(h[:, -1, :])


class SequenceBenchmarkModel(nn.Module):
    def __init__(self, model_name: str, cfg: SequenceBenchmarkConfig):
        super().__init__()
        input_dim = len(FEATURE_NAMES)
        model_name = model_name.upper()
        if model_name in {"GRU", "LSTM"}:
            self.encoder = RecurrentEncoder(
                input_dim=input_dim,
                hidden_dim=int(cfg.hidden_dim),
                num_layers=int(cfg.recurrent_layers),
                dropout=float(cfg.dropout),
                kind=model_name,
            )
        elif model_name == "TCN":
            self.encoder = TCNEncoder(input_dim=input_dim, channels=cfg.tcn_channels, dropout=float(cfg.dropout))
        elif model_name == "ATTENTION":
            self.encoder = AttentionEncoder(
                input_dim=input_dim,
                hidden_dim=int(cfg.hidden_dim),
                heads=int(cfg.attention_heads),
                layers=int(cfg.attention_layers),
                dropout=float(cfg.dropout),
                max_length=int(cfg.window),
            )
        else:
            raise ValueError(f"unsupported model: {model_name}")
        hidden_dim = int(self.encoder.output_dim)
        self.health_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pretrain_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(PRETRAIN_TARGET_NAMES)),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_health(self, x: torch.Tensor) -> torch.Tensor:
        return self.health_head(self.encode(x)).squeeze(-1)

    def forward_pretrain(self, x: torch.Tensor) -> torch.Tensor:
        return self.pretrain_head(self.encode(x))


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


def compute_dp_proxy_health(dp: np.ndarray, clean_dp: float, failure_dp: float) -> np.ndarray:
    health = (float(failure_dp) - dp.astype(np.float32)) / max(float(failure_dp) - float(clean_dp), 1e-6)
    return np.clip(health, 0.0, 1.0).astype(np.float32)


def compute_health(dp: np.ndarray, clean_dp: float, failure_dp: float) -> np.ndarray:
    return compute_dp_proxy_health(dp, clean_dp=clean_dp, failure_dp=failure_dp)


def family_stage_alpha(family: str, cfg: SequenceBenchmarkConfig) -> float:
    mapping = {
        "A2": float(cfg.stage_weight_alpha_a2),
        "A3": float(cfg.stage_weight_alpha_a3),
        "A4": float(cfg.stage_weight_alpha_a4),
    }
    return mapping[str(family).upper()]


def compute_total_life(time_arr: np.ndarray, rul_true: np.ndarray) -> np.ndarray:
    if np.all(np.isnan(rul_true)):
        return np.full_like(time_arr, np.nan, dtype=np.float32)
    total_life = time_arr.astype(np.float32) + rul_true.astype(np.float32)
    reference = float(np.nanmedian(total_life))
    return np.full_like(time_arr, reference, dtype=np.float32)


def compute_health_remaining(time_arr: np.ndarray, rul_true: np.ndarray) -> np.ndarray:
    total_life = compute_total_life(time_arr, rul_true)
    if np.all(np.isnan(total_life)):
        return np.full_like(time_arr, np.nan, dtype=np.float32)
    health_remaining = rul_true.astype(np.float32) / np.clip(total_life, 1e-6, None)
    return np.clip(health_remaining, 0.0, 1.0).astype(np.float32)


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
        dp_proxy_health = compute_dp_proxy_health(dp, clean_dp=cfg.health_clean_dp, failure_dp=cfg.health_failure_dp)
        if "RUL" in g.columns:
            rul_true = g["RUL"].to_numpy(dtype=np.float32)
            total_life = compute_total_life(time_arr, rul_true)
            health_remaining_true = compute_health_remaining(time_arr, rul_true)
        else:
            rul_true = np.full_like(dp, np.nan, dtype=np.float32)
            total_life = np.full_like(dp, np.nan, dtype=np.float32)
            health_remaining_true = np.full_like(dp, np.nan, dtype=np.float32)
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
                dp_proxy_health=dp_proxy_health,
                health_remaining_true=health_remaining_true,
                rul_true=rul_true,
                total_life=total_life,
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
            row = {
                "source_split": split_name,
                "family": family,
                "sequence_count": int(len(family_sequences)),
                "row_count": int(sum(sample.length for sample in family_sequences)),
                "feed_count": int(len(feed_sets)),
                "feed_values": ",".join(f"{value:.6f}" for value in feed_sets),
            }
            if split_name == "test" and family_sequences:
                terminal_health = [float(sample.health_remaining_true[-1]) for sample in family_sequences]
                total_life_values = [float(sample.total_life_scalar) for sample in family_sequences]
                row["terminal_health_remaining_min"] = float(np.min(terminal_health))
                row["terminal_health_remaining_max"] = float(np.max(terminal_health))
                row["total_life_median"] = float(np.median(np.asarray(total_life_values, dtype=np.float64)))
            summary_rows.append(row)
    a2_train = set(feed_sets_by_split["train"]["A2"])
    a2_test = set(feed_sets_by_split["test"]["A2"])
    summary = {
        "sequence_count": int(len(sequences)),
        "sequence_ids_unique": int(len(sequence_map) == len(sequences)),
        "feature_names": list(FEATURE_NAMES),
        "health_target_type": cfg.health_target_type,
        "pretrain_task": cfg.pretrain_task,
        "rul_derivation": cfg.rul_derivation,
        "dp_proxy_health_formula": "clip((600 - Differential_pressure) / (600 - 25), 0, 1)",
        "health_remaining_formula": "RUL / (Time + RUL)",
        "feed_sets_by_split": feed_sets_by_split,
        "a2_feed_overlap": [float(x) for x in sorted(a2_train & a2_test)],
        "a2_feed_train_only": [float(x) for x in sorted(a2_train - a2_test)],
        "a2_feed_test_only": [float(x) for x in sorted(a2_test - a2_train)],
        "rows": summary_rows,
    }
    return sequence_map, summary


def _select_evenly_spaced_indices(length: int, limit: Optional[int], max_end: Optional[int] = None) -> np.ndarray:
    end = length if max_end is None else max(0, min(length, int(max_end)))
    if end <= 0:
        return np.asarray([], dtype=np.int64)
    if limit is None or limit <= 0 or end <= limit:
        return np.arange(end, dtype=np.int64)
    return np.unique(np.linspace(0, end - 1, num=limit, dtype=np.int64))


def _limit_sequence_ids(sequence_ids: Sequence[str], limit: Optional[int], seed: int) -> List[str]:
    ids = sorted(sequence_ids)
    if limit is None or limit <= 0 or len(ids) <= limit:
        return ids
    rng = np.random.default_rng(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    return sorted(shuffled[:limit])


def build_grouped_folds(
    family: str,
    sequence_map: Dict[str, SequenceSample],
    cfg: SequenceBenchmarkConfig,
) -> Tuple[List[FoldSplit], List[str]]:
    test_ids_all = [sid for sid, sample in sequence_map.items() if sample.family == family and sample.source_split == "test"]
    family_seed = cfg.seed + 1009 + sum(ord(ch) for ch in family)
    test_ids = _limit_sequence_ids(test_ids_all, cfg.max_sequences_per_family, seed=family_seed)
    if len(test_ids) < 2:
        raise ValueError(f"family {family} requires at least two test sequences for grouped folds")
    effective_k = min(int(cfg.kfolds), len(test_ids))
    effective_k = max(2, effective_k)
    rng = np.random.default_rng(cfg.seed + 37 + sum(ord(ch) for ch in family))
    shuffled = list(test_ids)
    rng.shuffle(shuffled)
    buckets = [[] for _ in range(effective_k)]
    for idx, sid in enumerate(shuffled):
        buckets[idx % effective_k].append(sid)
    folds = []
    all_ids = set(test_ids)
    for fold_index in range(effective_k):
        fold_test_ids = tuple(sorted(buckets[fold_index]))
        outer_train_ids = sorted(all_ids - set(fold_test_ids))
        val_ids = select_validation_ids(outer_train_ids, cfg.val_fraction, seed=cfg.seed + fold_index * 17 + sum(ord(ch) for ch in family))
        train_fold_ids = tuple(sorted(set(outer_train_ids) - set(val_ids)))
        folds.append(FoldSplit(train_ids=train_fold_ids, val_ids=tuple(sorted(val_ids)), test_ids=fold_test_ids))
    return folds, test_ids


def select_validation_ids(train_pool_ids: Sequence[str], val_fraction: float, seed: int) -> List[str]:
    if len(train_pool_ids) <= 1 or val_fraction <= 0.0:
        return []
    rng = np.random.default_rng(seed)
    shuffled = list(train_pool_ids)
    rng.shuffle(shuffled)
    count = int(round(len(shuffled) * float(val_fraction)))
    count = max(1, count) if len(shuffled) >= 4 else (1 if len(shuffled) >= 2 else 0)
    count = min(count, len(shuffled) - 1)
    return sorted(shuffled[:count])


def family_train_sequence_ids(
    family: str,
    sequence_map: Dict[str, SequenceSample],
    cfg: SequenceBenchmarkConfig,
) -> List[str]:
    train_ids_all = [sid for sid, sample in sequence_map.items() if sample.family == family and sample.source_split == "train"]
    family_seed = cfg.seed + 2027 + sum(ord(ch) for ch in family)
    return _limit_sequence_ids(train_ids_all, cfg.max_sequences_per_family, seed=family_seed)


def pretrain_ids_for_fold(
    family: str,
    fold: FoldSplit,
    sequence_map: Dict[str, SequenceSample],
    cfg: SequenceBenchmarkConfig,
) -> List[str]:
    return sorted(dict.fromkeys(family_train_sequence_ids(family, sequence_map, cfg) + list(fold.train_ids)))


def build_pretrain_arrays(
    sequence_ids: Sequence[str],
    sequence_map: Dict[str, SequenceSample],
    cfg: SequenceBenchmarkConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    windows = []
    targets = []
    for sequence_id in sequence_ids:
        sample = sequence_map[sequence_id]
        features = sample.features
        end_indices = _select_evenly_spaced_indices(sample.length - 1, cfg.max_windows_per_sequence)
        for end_idx in end_indices:
            start = int(end_idx) - int(cfg.window) + 1
            if start < 0:
                pad = np.repeat(features[0:1], repeats=-start, axis=0)
                core = features[0 : int(end_idx) + 1]
                window = np.concatenate([pad, core], axis=0)
            else:
                window = features[start : int(end_idx) + 1]
            next_idx = int(end_idx) + 1
            target = np.asarray([sample.dp[next_idx], sample.flow[next_idx]], dtype=np.float32)
            windows.append(window.astype(np.float32))
            targets.append(target)
    if not windows:
        raise ValueError("no pretraining windows were generated")
    return np.stack(windows).astype(np.float32), np.stack(targets).astype(np.float32)


def build_supervised_arrays(
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
        end_indices = _select_evenly_spaced_indices(sample.length, cfg.max_windows_per_sequence)
        window_count = int(len(end_indices))
        for end_idx in end_indices:
            start = int(end_idx) - int(cfg.window) + 1
            if start < 0:
                pad = np.repeat(features[0:1], repeats=-start, axis=0)
                core = features[0 : int(end_idx) + 1]
                window = np.concatenate([pad, core], axis=0)
            else:
                window = features[start : int(end_idx) + 1]
            windows.append(window.astype(np.float32))
            targets.append(float(sample.health_remaining_true[int(end_idx)]))
            rows.append(
                {
                    "sequence_id": sample.sequence_id,
                    "family": sample.family,
                    "source_split": sample.source_split,
                    "data_no": int(sample.data_no),
                    "time": float(sample.time[int(end_idx)]),
                    "end_index": int(end_idx),
                    "dust_feed": float(sample.feed[int(end_idx)]),
                    "health_remaining_true": float(sample.health_remaining_true[int(end_idx)]),
                    "dp_proxy_health": float(sample.dp_proxy_health[int(end_idx)]),
                    "rul_true": float(sample.rul_true[int(end_idx)]),
                    "total_life": float(sample.total_life[int(end_idx)]),
                    "sequence_total_life": float(sample.total_life_scalar),
                    "window_count": window_count,
                    "is_observed": True,
                    "is_tail_reference": False,
                }
            )
    if not windows:
        raise ValueError("no supervised windows were generated")
    return np.stack(windows).astype(np.float32), np.asarray(targets, dtype=np.float32), pd.DataFrame(rows)


def build_supervised_sample_weights(
    family: str,
    meta: pd.DataFrame,
    train_ids: Sequence[str],
    sequence_map: Dict[str, SequenceSample],
    cfg: SequenceBenchmarkConfig,
) -> np.ndarray:
    if meta.empty:
        return np.empty((0,), dtype=np.float32)
    weights = 1.0 / np.clip(meta["window_count"].to_numpy(dtype=np.float32), 1.0, None)
    health_remaining = meta["health_remaining_true"].to_numpy(dtype=np.float32)
    weights *= 1.0 + family_stage_alpha(family, cfg) * (1.0 - health_remaining)
    if str(family).upper() == "A2" and train_ids:
        seq_lives = sorted(
            [(sequence_id, float(sequence_map[sequence_id].total_life_scalar)) for sequence_id in train_ids],
            key=lambda item: item[1],
        )
        long_count = max(1, int(math.ceil(len(seq_lives) / 3.0)))
        long_ids = {sequence_id for sequence_id, _life in seq_lives[-long_count:]}
        if long_ids:
            long_mask = meta["sequence_id"].isin(long_ids).to_numpy(dtype=bool)
            weights[long_mask] *= float(cfg.a2_long_life_weight)
    weights = weights.astype(np.float32)
    weight_mean = float(np.mean(weights))
    if weight_mean > 1e-8:
        weights /= weight_mean
    return weights.astype(np.float32)


def build_model(model_name: str, cfg: SequenceBenchmarkConfig) -> SequenceBenchmarkModel:
    return SequenceBenchmarkModel(model_name=model_name, cfg=cfg)


def cfg_with_family_hparams(cfg: SequenceBenchmarkConfig, override: Optional[FamilyHyperParams]) -> SequenceBenchmarkConfig:
    if override is None:
        return cfg
    return replace(
        cfg,
        window=int(override.window),
        hidden_dim=int(override.hidden_dim),
        dropout=float(override.dropout),
    )


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    sample_error = torch.square(pred - target)
    if sample_error.ndim > 1:
        sample_error = sample_error.mean(dim=1)
    sample_weights = weights.reshape(-1).to(sample_error.dtype)
    return torch.sum(sample_error * sample_weights) / torch.clamp(sample_weights.sum(), min=1e-6)


def _predict_health_batches(model: SequenceBenchmarkModel, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            pred = model.forward_health(batch).detach().cpu().numpy()
            outputs.append(pred)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _predict_pretrain_batches(model: SequenceBenchmarkModel, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            pred = model.forward_pretrain(batch).detach().cpu().numpy()
            outputs.append(pred)
    return np.concatenate(outputs, axis=0).astype(np.float32)


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


def compute_rul_metrics(prediction_df: pd.DataFrame) -> Dict[str, float]:
    valid = prediction_df[
        prediction_df["is_observed"]
        & prediction_df["rul_true"].notna()
        & prediction_df["rul_pred"].notna()
        & prediction_df["health_remaining_pred"].notna()
    ]
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


def train_pretrain_phase(
    model: SequenceBenchmarkModel,
    train_x: np.ndarray,
    train_targets: np.ndarray,
    cfg: SequenceBenchmarkConfig,
    device: torch.device,
    history_path: Path,
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    loss_fn = nn.MSELoss()
    loader = DataLoader(WindowDataset(train_x, train_targets), batch_size=int(cfg.batch_size), shuffle=True, num_workers=0)
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    best_rmse = math.inf
    best_epoch = 0
    epochs_without_improve = 0
    history_rows = []
    for epoch in range(1, int(cfg.pretrain_epochs) + 1):
        model.train()
        losses = []
        for batch_x, batch_y, _batch_w in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward_pretrain(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        full_pred = _predict_pretrain_batches(model, train_x, batch_size=int(cfg.batch_size), device=device)
        epoch_rmse = rmse(train_targets, full_pred)
        history_rows.append({"epoch": epoch, "train_mse": float(np.mean(losses)) if losses else 0.0, "train_rmse": epoch_rmse})
        if epoch_rmse < best_rmse - 1e-6:
            best_rmse = epoch_rmse
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if epochs_without_improve >= int(cfg.pretrain_patience):
            break
    model.load_state_dict(best_state)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    return {"pretrain_best_epoch": int(best_epoch), "pretrain_best_rmse": float(best_rmse)}


def train_finetune_phase(
    model: SequenceBenchmarkModel,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weights: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    cfg: SequenceBenchmarkConfig,
    device: torch.device,
    history_path: Path,
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    loader = DataLoader(WindowDataset(train_x, train_y, train_weights), batch_size=int(cfg.batch_size), shuffle=True, num_workers=0)
    use_validation = len(val_x) > 0
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    best_val_rmse = math.inf
    best_epoch = 0
    epochs_without_improve = 0
    history_rows = []
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        losses = []
        for batch_x, batch_y, batch_w in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward_health(batch_x)
            loss = weighted_mse_loss(pred, batch_y, batch_w)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if use_validation:
            val_pred = np.clip(_predict_health_batches(model, val_x, batch_size=int(cfg.batch_size), device=device), 0.0, 1.0)
            metric_rmse = rmse(val_y, val_pred)
        else:
            train_pred = np.clip(_predict_health_batches(model, train_x, batch_size=int(cfg.batch_size), device=device), 0.0, 1.0)
            metric_rmse = rmse(train_y, train_pred)
        history_rows.append({"epoch": epoch, "train_mse": float(np.mean(losses)) if losses else 0.0, "val_rmse": metric_rmse})
        if metric_rmse < best_val_rmse - 1e-6:
            best_val_rmse = metric_rmse
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if epochs_without_improve >= int(cfg.patience):
            break
    model.load_state_dict(best_state)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    return {"best_epoch": int(best_epoch), "best_val_rmse": float(best_val_rmse)}


def train_fold_model(
    model_name: str,
    family: str,
    fold_index: int,
    pretrain_x: np.ndarray,
    pretrain_targets: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weights: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    standardizer: FeatureStandardizer,
    cfg: SequenceBenchmarkConfig,
    device: torch.device,
    artifact_dir: Path,
) -> Tuple[SequenceBenchmarkModel, Dict[str, object]]:
    model = build_model(model_name, cfg).to(device)
    pretrain_summary = train_pretrain_phase(
        model=model,
        train_x=pretrain_x,
        train_targets=pretrain_targets,
        cfg=cfg,
        device=device,
        history_path=artifact_dir / f"fold_{fold_index:02d}_pretrain_history.csv",
    )
    finetune_summary = train_finetune_phase(
        model=model,
        train_x=train_x,
        train_y=train_y,
        train_weights=train_weights,
        val_x=val_x,
        val_y=val_y,
        cfg=cfg,
        device=device,
        history_path=artifact_dir / f"fold_{fold_index:02d}_finetune_history.csv",
    )
    checkpoint_path = artifact_dir / f"fold_{fold_index:02d}_best.pt"
    torch.save(
        {
            "model_name": model_name,
            "family": family,
            "fold_index": int(fold_index),
            "feature_names": list(FEATURE_NAMES),
            "pretrain_target_names": list(PRETRAIN_TARGET_NAMES),
            "state_dict": model.state_dict(),
            "standardizer": standardizer.to_dict(),
            "config": asdict(cfg),
            **pretrain_summary,
            **finetune_summary,
        },
        checkpoint_path,
    )
    summary = {
        "model_name": model_name,
        "family": family,
        "fold_index": int(fold_index),
        "checkpoint_path": str(checkpoint_path),
        **pretrain_summary,
        **finetune_summary,
    }
    with (artifact_dir / f"fold_{fold_index:02d}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    return model, summary


def add_tail_reference_rows(pred_df: pd.DataFrame, sequence_map: Dict[str, SequenceSample], cfg: SequenceBenchmarkConfig) -> pd.DataFrame:
    if not bool(cfg.add_tail_reference_rows):
        return pred_df
    tail_rows: List[Dict[str, object]] = []
    for sequence_id, group in pred_df[pred_df["is_observed"]].groupby("sequence_id", sort=False):
        sample = sequence_map[sequence_id]
        if sample.source_split != "test":
            continue
        total_life = float(sample.total_life_scalar)
        if not np.isfinite(total_life):
            continue
        last_time = float(group["time"].max())
        if total_life <= last_time + 1e-9:
            continue
        tail_times = np.arange(last_time + float(cfg.dt), total_life + 0.5 * float(cfg.dt), float(cfg.dt), dtype=np.float32)
        last_end_index = int(group["end_index"].max())
        for offset, tail_time in enumerate(tail_times, start=1):
            rul_true = max(total_life - float(tail_time), 0.0)
            health_remaining_true = rul_true / max(total_life, 1e-6)
            tail_rows.append(
                {
                    "sequence_id": sequence_id,
                    "family": sample.family,
                    "source_split": sample.source_split,
                    "data_no": int(sample.data_no),
                    "time": float(tail_time),
                    "end_index": int(last_end_index + offset),
                    "dust_feed": float(sample.dust_feed),
                    "health_remaining_true": float(health_remaining_true),
                    "health_remaining_pred": math.nan,
                    "dp_proxy_health": math.nan,
                    "rul_true": float(rul_true),
                    "rul_pred": math.nan,
                    "total_life": float(total_life),
                    "is_observed": False,
                    "is_tail_reference": True,
                }
            )
    if not tail_rows:
        return pred_df
    merged_rows = pred_df.to_dict(orient="records")
    merged_rows.extend(tail_rows)
    merged = pd.DataFrame.from_records(merged_rows, columns=list(pred_df.columns))
    return merged.sort_values(["sequence_id", "time", "is_tail_reference"]).reset_index(drop=True)


def apply_monotone_projection(pred_df: pd.DataFrame, cfg: SequenceBenchmarkConfig) -> pd.DataFrame:
    projected = pred_df.copy()
    projected["health_remaining_pred_raw"] = projected["health_remaining_pred"]
    projected["rul_pred_raw"] = projected["rul_pred"]
    if not bool(cfg.enable_monotone_projection):
        return projected
    observed = projected[projected["is_observed"]].copy()
    if observed.empty:
        return projected
    for sequence_id, group in observed.groupby("sequence_id", sort=False):
        seq = group.sort_values("time")
        raw = seq["health_remaining_pred_raw"].to_numpy(dtype=np.float32)
        monotone = np.minimum.accumulate(raw).astype(np.float32)
        projected.loc[seq.index, "health_remaining_pred"] = monotone
        projected.loc[seq.index, "rul_pred"] = monotone * seq["total_life"].to_numpy(dtype=np.float32)
    return projected


def evaluate_fold_predictions(
    model: SequenceBenchmarkModel,
    test_x: np.ndarray,
    test_meta: pd.DataFrame,
    sequence_map: Dict[str, SequenceSample],
    cfg: SequenceBenchmarkConfig,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    pred = np.clip(_predict_health_batches(model, test_x, batch_size=int(cfg.batch_size), device=device), 0.0, 1.0)
    pred_df = test_meta.copy()
    pred_df["health_remaining_pred"] = pred
    pred_df["rul_pred"] = pred_df["health_remaining_pred"] * pred_df["total_life"]
    pred_df = apply_monotone_projection(pred_df, cfg=cfg)
    pred_df = add_tail_reference_rows(pred_df, sequence_map=sequence_map, cfg=cfg)
    observed = pred_df[pred_df["is_observed"]].copy()
    health_metrics = compute_health_metrics(
        observed["health_remaining_true"].to_numpy(dtype=np.float32),
        observed["health_remaining_pred"].to_numpy(dtype=np.float32),
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


def _sample_scatter_points(df: pd.DataFrame, true_col: str, pred_col: str, seed: int, max_points: int = 4000) -> pd.DataFrame:
    subset = df[[true_col, pred_col, "model"]].dropna()
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
    axes[0].set_title("Remaining-Life Health Metrics by Model")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    axes[1].bar(x - 0.2, overall_df["rul_rmse"], width=0.4, label="RMSE")
    axes[1].bar(x + 0.2, overall_df["rul_mae"], width=0.4, label="MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(overall_df["model"])
    axes[1].set_title("Observed-Segment RUL Metrics by Model")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outdir / "metric_bars_overall.png", dpi=200)
    plt.close(fig)


def plot_scatter_grid(pred_df: pd.DataFrame, outdir: Path, true_col: str, pred_col: str, title_prefix: str, seed: int):
    observed = pred_df[pred_df["is_observed"]].copy()
    models = sorted(observed["model"].dropna().unique().tolist())
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
    for ax, model_name in zip(axes[0], models):
        subset = observed[observed["model"] == model_name]
        sampled = _sample_scatter_points(subset, true_col=true_col, pred_col=pred_col, seed=seed)
        ax.scatter(sampled[true_col], sampled[pred_col], s=10, alpha=0.35)
        lo = min(float(sampled[true_col].min()), float(sampled[pred_col].min()))
        hi = max(float(sampled[true_col].max()), float(sampled[pred_col].max()))
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
        if group.empty:
            continue
        sequence_ids = group["sequence_id"].unique().tolist()
        representative_id = sequence_ids[0]
        longest_len = -1
        for sequence_id in sequence_ids:
            candidate = group[group["sequence_id"] == sequence_id]
            observed_len = int(candidate["is_observed"].sum())
            if observed_len > longest_len:
                longest_len = observed_len
                representative_id = sequence_id
        rep = group[group["sequence_id"] == representative_id].sort_values("time")
        observed = rep[rep["is_observed"]].copy()
        true_line = rep.copy()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(true_line["time"], true_line["health_remaining_true"], label="true health_remaining", linewidth=2.0)
        ax.plot(observed["time"], observed["health_remaining_pred"], label="pred health_remaining", linewidth=2.0)
        if rep["is_tail_reference"].any():
            ax.axvline(float(observed["time"].max()), color="gray", linestyle=":", linewidth=1.0, label="observed end")
        ax.set_title(f"{family} {model_name} health_remaining trajectory")
        ax.set_xlabel("Time")
        ax.set_ylabel("Remaining-life health")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"health_curve_{family}_{model_name}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(true_line["time"], true_line["rul_true"], label="true RUL", linewidth=2.0)
        ax.plot(observed["time"], observed["rul_pred"], label="pred RUL", linewidth=2.0)
        if rep["is_tail_reference"].any():
            ax.axvline(float(observed["time"].max()), color="gray", linestyle=":", linewidth=1.0, label="observed end")
        ax.set_title(f"{family} {model_name} RUL trajectory")
        ax.set_xlabel("Time")
        ax.set_ylabel("RUL")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"rul_curve_{family}_{model_name}.png", dpi=200)
        plt.close(fig)


def print_rankings(family_df: pd.DataFrame, ranking_df: pd.DataFrame, out_root: Path, dataset_summary: Dict[str, object], cfg: SequenceBenchmarkConfig):
    print("benchmark reminders:")
    print(f"  health_target_type: {cfg.health_target_type}")
    print(f"  pretrain_task: {cfg.pretrain_task}")
    print(f"  rul_derivation: {cfg.rul_derivation}")
    print(f"  RUL uses true total_life from Test only; this is not simulator-side inference.")
    print("dataset summary:")
    print(f"  total sequences: {dataset_summary['sequence_count']}")
    print(f"  A2 train-only feeds: {dataset_summary.get('a2_feed_train_only', [])}")
    print(f"  A2 test-only feeds: {dataset_summary.get('a2_feed_test_only', [])}")
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


def save_run_metadata(
    out_root: Path,
    cfg: SequenceBenchmarkConfig,
    dataset_summary: Dict[str, object],
    family_overrides: Optional[Dict[str, FamilyHyperParams]] = None,
):
    config_payload = asdict(cfg)
    config_payload["family_hparam_overrides"] = {
        family: {
            "window": int(override.window),
            "hidden_dim": int(override.hidden_dim),
            "dropout": float(override.dropout),
        }
        for family, override in (family_overrides or {}).items()
    }
    with (out_root / "benchmark_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=True, indent=2)
    with (out_root / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, ensure_ascii=True, indent=2)
    pd.DataFrame(dataset_summary["rows"]).to_csv(out_root / "dataset_summary.csv", index=False)


def save_run_outputs(
    out_root: Path,
    cfg: SequenceBenchmarkConfig,
    dataset_summary: Dict[str, object],
    fold_df: pd.DataFrame,
    family_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    family_overrides: Optional[Dict[str, FamilyHyperParams]] = None,
    write_plots: bool = True,
):
    out_root.mkdir(parents=True, exist_ok=True)
    plots_root = out_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    save_run_metadata(out_root, cfg, dataset_summary, family_overrides=family_overrides)
    fold_df.to_csv(out_root / "metrics_by_fold.csv", index=False)
    family_df.to_csv(out_root / "metrics_by_family.csv", index=False)
    overall_df.to_csv(out_root / "metrics_overall.csv", index=False)
    ranking_df.to_csv(out_root / "model_rankings.csv", index=False)
    overall_payload = {
        "health_target_type": cfg.health_target_type,
        "pretrain_task": cfg.pretrain_task,
        "rul_derivation": cfg.rul_derivation,
        "best_model": None if ranking_df.empty else str(ranking_df.iloc[0]["model"]),
        "models": [] if ranking_df.empty else ranking_df.to_dict(orient="records"),
        "dataset_summary_path": str(out_root / "dataset_summary.json"),
        "family_hparam_overrides": {
            family: {
                "window": int(override.window),
                "hidden_dim": int(override.hidden_dim),
                "dropout": float(override.dropout),
            }
            for family, override in (family_overrides or {}).items()
        },
    }
    with (out_root / "metrics_overall.json").open("w", encoding="utf-8") as f:
        json.dump(overall_payload, f, ensure_ascii=True, indent=2)
    if write_plots and not pred_df.empty:
        plot_metric_bars(ranking_df, plots_root)
        plot_scatter_grid(
            pred_df,
            plots_root,
            true_col="health_remaining_true",
            pred_col="health_remaining_pred",
            title_prefix="health_remaining",
            seed=cfg.seed,
        )
        plot_scatter_grid(
            pred_df[pred_df["is_observed"] & pred_df["rul_pred"].notna()].copy(),
            plots_root,
            true_col="rul_true",
            pred_col="rul_pred",
            title_prefix="RUL",
            seed=cfg.seed + 13,
        )
        plot_representative_sequences(pred_df, plots_root)


def run_benchmark_pass(
    cfg: SequenceBenchmarkConfig,
    out_root: Path,
    sequence_map: Dict[str, SequenceSample],
    dataset_summary: Dict[str, object],
    device: torch.device,
    family_overrides: Optional[Dict[str, FamilyHyperParams]] = None,
    write_outputs: bool = True,
    write_plots: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    artifact_root = out_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    fold_rows: List[Dict[str, object]] = []
    all_predictions: List[pd.DataFrame] = []
    for family in cfg.families:
        family_override = (family_overrides or {}).get(family)
        family_cfg = cfg_with_family_hparams(cfg, family_override if "LSTM" in cfg.models else None)
        folds, selected_test_ids = build_grouped_folds(family, sequence_map, family_cfg)
        selected_train_ids = family_train_sequence_ids(family, sequence_map, family_cfg)
        if verbose:
            print(
                f"{family}: "
                f"train_pretrain_sequences={len(selected_train_ids)} "
                f"test_supervised_sequences={len(selected_test_ids)} "
                f"folds={len(folds)}"
            )
        for model_name in cfg.models:
            model_cfg = family_cfg if model_name == "LSTM" else cfg
            family_model_dir = artifact_root / family / model_name.lower()
            family_model_dir.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(
                    f"training {model_name} for {family} "
                    f"(window={int(model_cfg.window)}, hidden_dim={int(model_cfg.hidden_dim)}, dropout={float(model_cfg.dropout):.2f})"
                )
            for fold_index, fold in enumerate(folds):
                pretrain_ids = pretrain_ids_for_fold(family, fold, sequence_map, model_cfg)
                pretrain_x_raw, pretrain_targets_raw = build_pretrain_arrays(pretrain_ids, sequence_map, model_cfg)
                train_x_raw, train_y, train_meta = build_supervised_arrays(fold.train_ids, sequence_map, model_cfg)
                train_weights = build_supervised_sample_weights(
                    family=family,
                    meta=train_meta,
                    train_ids=fold.train_ids,
                    sequence_map=sequence_map,
                    cfg=model_cfg,
                )
                if fold.val_ids:
                    val_x_raw, val_y, _val_meta = build_supervised_arrays(fold.val_ids, sequence_map, model_cfg)
                else:
                    val_x_raw = np.empty((0, model_cfg.window, len(FEATURE_NAMES)), dtype=np.float32)
                    val_y = np.empty((0,), dtype=np.float32)
                test_x_raw, _test_y, test_meta = build_supervised_arrays(fold.test_ids, sequence_map, model_cfg)
                standardizer = FeatureStandardizer().fit(pretrain_x_raw)
                pretrain_x = standardizer.transform(pretrain_x_raw)
                pretrain_targets = standardizer.transform_dp_flow_targets(pretrain_targets_raw)
                train_x = standardizer.transform(train_x_raw)
                val_x = standardizer.transform(val_x_raw) if len(val_x_raw) else val_x_raw
                test_x = standardizer.transform(test_x_raw)
                model, train_summary = train_fold_model(
                    model_name=model_name,
                    family=family,
                    fold_index=fold_index,
                    pretrain_x=pretrain_x,
                    pretrain_targets=pretrain_targets,
                    train_x=train_x,
                    train_y=train_y,
                    train_weights=train_weights,
                    val_x=val_x,
                    val_y=val_y,
                    standardizer=standardizer,
                    cfg=model_cfg,
                    device=device,
                    artifact_dir=family_model_dir,
                )
                pred_df, health_metrics, rul_metrics = evaluate_fold_predictions(
                    model=model,
                    test_x=test_x,
                    test_meta=test_meta,
                    sequence_map=sequence_map,
                    cfg=model_cfg,
                    device=device,
                )
                pred_df["model"] = model_name
                pred_df["fold"] = int(fold_index)
                pred_df["window"] = int(model_cfg.window)
                pred_df["hidden_dim"] = int(model_cfg.hidden_dim)
                pred_df["dropout"] = float(model_cfg.dropout)
                pred_df.to_csv(family_model_dir / f"fold_{fold_index:02d}_predictions.csv", index=False)
                all_predictions.append(pred_df)
                fold_row = {
                    "family": family,
                    "model": model_name,
                    "fold": int(fold_index),
                    "window": int(model_cfg.window),
                    "hidden_dim": int(model_cfg.hidden_dim),
                    "dropout": float(model_cfg.dropout),
                    "pretrain_sequence_count": int(len(pretrain_ids)),
                    "train_sequence_count": int(len(fold.train_ids)),
                    "val_sequence_count": int(len(fold.val_ids)),
                    "test_sequence_count": int(len(fold.test_ids)),
                    "pretrain_best_epoch": int(train_summary["pretrain_best_epoch"]),
                    "pretrain_best_rmse": float(train_summary["pretrain_best_rmse"]),
                    "best_epoch": int(train_summary["best_epoch"]),
                    "best_val_rmse": float(train_summary["best_val_rmse"]),
                }
                fold_row.update(health_metrics)
                fold_row.update(rul_metrics)
                fold_rows.append(fold_row)
                if verbose:
                    print(
                        f"  fold {fold_index + 1}/{len(folds)} "
                        f"health_rmse={float(health_metrics['health_rmse']):.4f} "
                        f"rul_rmse={float(rul_metrics['rul_rmse']):.4f}"
                    )
    fold_df = pd.DataFrame(fold_rows).sort_values(["family", "model", "fold"]).reset_index(drop=True)
    family_df = aggregate_family_metrics(fold_df)
    overall_df = aggregate_overall_metrics(family_df)
    ranking_df = add_rankings(overall_df)
    pred_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    if write_outputs:
        save_run_outputs(
            out_root=out_root,
            cfg=cfg,
            dataset_summary=dataset_summary,
            fold_df=fold_df,
            family_df=family_df,
            overall_df=overall_df,
            ranking_df=ranking_df,
            pred_df=pred_df,
            family_overrides=family_overrides,
            write_plots=write_plots,
        )
    if verbose:
        print_rankings(family_df, ranking_df, out_root, dataset_summary, cfg)
    return {
        "out_root": str(out_root),
        "dataset_summary": dataset_summary,
        "metrics_by_fold": fold_df,
        "metrics_by_family": family_df,
        "metrics_overall": ranking_df,
        "predictions": pred_df,
    }


def iter_lstm_tuning_candidates(cfg: SequenceBenchmarkConfig) -> List[FamilyHyperParams]:
    candidates: List[FamilyHyperParams] = []
    for window in cfg.tuning_windows:
        for hidden_dim in cfg.tuning_hidden_dims:
            for dropout in cfg.tuning_dropouts:
                candidates.append(FamilyHyperParams(window=int(window), hidden_dim=int(hidden_dim), dropout=float(dropout)))
    return candidates


def select_best_family_candidate(search_df: pd.DataFrame, family: str) -> FamilyHyperParams:
    family_rows = search_df[search_df["family"] == family].copy()
    ranked = family_rows.sort_values(
        ["health_rmse_mean", "rul_rmse_mean", "health_r2_mean", "window", "hidden_dim", "dropout"],
        ascending=[True, True, False, True, True, True],
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    return FamilyHyperParams(window=int(best["window"]), hidden_dim=int(best["hidden_dim"]), dropout=float(best["dropout"]))


def run_lstm_family_tuning_search(
    cfg: SequenceBenchmarkConfig,
    out_root: Path,
    sequence_map: Dict[str, SequenceSample],
    dataset_summary: Dict[str, object],
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, FamilyHyperParams]]:
    search_root = out_root / "tuning_search"
    rows: List[Dict[str, object]] = []
    selected: Dict[str, FamilyHyperParams] = {}
    for family in cfg.families:
        print(f"tuning LSTM for {family}")
        for candidate in iter_lstm_tuning_candidates(cfg):
            candidate_cfg = replace(
                cfg,
                models=("LSTM",),
                families=(family,),
                window=int(candidate.window),
                hidden_dim=int(candidate.hidden_dim),
                dropout=float(candidate.dropout),
                enable_family_tuning=False,
            )
            config_tag = f"w{int(candidate.window)}_h{int(candidate.hidden_dim)}_d{str(float(candidate.dropout)).replace('.', 'p')}"
            candidate_root = search_root / family / config_tag
            result = run_benchmark_pass(
                cfg=candidate_cfg,
                out_root=candidate_root,
                sequence_map=sequence_map,
                dataset_summary=dataset_summary,
                device=device,
                family_overrides=None,
                write_outputs=True,
                write_plots=False,
                verbose=False,
            )
            family_row = result["metrics_by_family"].iloc[0].to_dict()
            family_row.update(
                {
                    "window": int(candidate.window),
                    "hidden_dim": int(candidate.hidden_dim),
                    "dropout": float(candidate.dropout),
                    "config_tag": config_tag,
                    "candidate_root": str(candidate_root),
                }
            )
            rows.append(family_row)
        family_df = pd.DataFrame(rows)
        selected[family] = select_best_family_candidate(family_df, family)
        best = selected[family]
        print(
            f"  selected for {family}: "
            f"window={best.window} hidden_dim={best.hidden_dim} dropout={best.dropout:.2f}"
        )
    search_df = pd.DataFrame(rows).sort_values(
        ["family", "health_rmse_mean", "rul_rmse_mean", "health_r2_mean", "window", "hidden_dim", "dropout"],
        ascending=[True, True, True, False, True, True, True],
    ).reset_index(drop=True)
    search_df.to_csv(out_root / "family_tuning_search.csv", index=False)
    selected_rows = [
        {
            "family": family,
            "window": int(params.window),
            "hidden_dim": int(params.hidden_dim),
            "dropout": float(params.dropout),
        }
        for family, params in selected.items()
    ]
    selected_df = pd.DataFrame(selected_rows).sort_values("family").reset_index(drop=True)
    selected_df.to_csv(out_root / "selected_lstm_family_configs.csv", index=False)
    with (out_root / "selected_lstm_family_configs.json").open("w", encoding="utf-8") as f:
        json.dump(selected_rows, f, ensure_ascii=True, indent=2)
    return search_df, selected


def compare_family_results(candidate_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    candidate = candidate_df[candidate_df["model"] == "LSTM"].copy()
    reference = reference_df[reference_df["model"] == "LSTM"].copy()
    merged = candidate.merge(
        reference,
        on="family",
        suffixes=("_candidate", "_reference"),
        how="outer",
    )
    for metric in ("health_rmse_mean", "rul_rmse_mean", "health_r2_mean"):
        merged[f"{metric}_delta"] = merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]
    return merged.sort_values("family").reset_index(drop=True)


def compare_against_baseline(
    baseline_root: Path,
    candidate_family_df: pd.DataFrame,
    candidate_fold_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    baseline_family_path = baseline_root / "metrics_by_family.csv"
    baseline_fold_path = baseline_root / "metrics_by_fold.csv"
    if not baseline_family_path.exists() or not baseline_fold_path.exists():
        return (
            pd.DataFrame(),
            {
                "baseline_root": str(baseline_root),
                "baseline_available": False,
                "accepted_as_default": False,
                "blocking_reasons": ["baseline reference files not found"],
            },
        )
    baseline_family_df = pd.read_csv(baseline_family_path)
    baseline_fold_df = pd.read_csv(baseline_fold_path)
    baseline_family = baseline_family_df[baseline_family_df["model"] == "LSTM"].copy()
    candidate_family = candidate_family_df[candidate_family_df["model"] == "LSTM"].copy()
    baseline_worst = (
        baseline_fold_df[baseline_fold_df["model"] == "LSTM"]
        .groupby("family", as_index=False)["rul_rmse"]
        .max()
        .rename(columns={"rul_rmse": "baseline_worst_fold_rul_rmse"})
    )
    candidate_worst = (
        candidate_fold_df[candidate_fold_df["model"] == "LSTM"]
        .groupby("family", as_index=False)["rul_rmse"]
        .max()
        .rename(columns={"rul_rmse": "candidate_worst_fold_rul_rmse"})
    )
    merged = candidate_family.merge(
        baseline_family[
            [
                "family",
                "health_rmse_mean",
                "rul_rmse_mean",
                "health_r2_mean",
            ]
        ].rename(
            columns={
                "health_rmse_mean": "baseline_health_rmse",
                "rul_rmse_mean": "baseline_rul_rmse",
                "health_r2_mean": "baseline_health_r2",
            }
        ),
        on="family",
        how="left",
    )
    merged = merged.merge(baseline_worst, on="family", how="left")
    merged = merged.merge(candidate_worst, on="family", how="left")
    merged = merged.rename(
        columns={
            "health_rmse_mean": "candidate_health_rmse",
            "rul_rmse_mean": "candidate_rul_rmse",
            "health_r2_mean": "candidate_health_r2",
        }
    )
    merged["health_rmse_delta"] = merged["candidate_health_rmse"] - merged["baseline_health_rmse"]
    merged["rul_rmse_delta"] = merged["candidate_rul_rmse"] - merged["baseline_rul_rmse"]
    merged["health_r2_delta"] = merged["candidate_health_r2"] - merged["baseline_health_r2"]
    merged["worst_fold_rul_rmse_delta"] = (
        merged["candidate_worst_fold_rul_rmse"] - merged["baseline_worst_fold_rul_rmse"]
    )
    merged["health_gate_pass"] = False
    merged["rul_gate_pass"] = False
    for family in merged["family"].tolist():
        family_mask = merged["family"] == family
        if family == "A3":
            merged.loc[family_mask, "health_gate_pass"] = (
                merged.loc[family_mask, "candidate_health_rmse"] < merged.loc[family_mask, "baseline_health_rmse"]
            )
            merged.loc[family_mask, "rul_gate_pass"] = (
                merged.loc[family_mask, "candidate_rul_rmse"] < merged.loc[family_mask, "baseline_rul_rmse"]
            )
        else:
            merged.loc[family_mask, "health_gate_pass"] = (
                merged.loc[family_mask, "candidate_health_rmse"] <= merged.loc[family_mask, "baseline_health_rmse"]
            )
            merged.loc[family_mask, "rul_gate_pass"] = (
                merged.loc[family_mask, "candidate_rul_rmse"] <= merged.loc[family_mask, "baseline_rul_rmse"]
            )
    merged["worst_fold_gate_pass"] = merged["candidate_worst_fold_rul_rmse"] <= merged["baseline_worst_fold_rul_rmse"]
    merged["family_gate_pass"] = merged["health_gate_pass"] & merged["rul_gate_pass"] & merged["worst_fold_gate_pass"]
    blocking_reasons: List[str] = []
    for row in merged.sort_values("family").to_dict(orient="records"):
        family_failures = []
        if not bool(row["health_gate_pass"]):
            family_failures.append("health_rmse gate failed")
        if not bool(row["rul_gate_pass"]):
            family_failures.append("rul_rmse gate failed")
        if not bool(row["worst_fold_gate_pass"]):
            family_failures.append("worst-fold rul_rmse gate failed")
        if family_failures:
            blocking_reasons.append(f"{row['family']}: {', '.join(family_failures)}")
    summary = {
        "baseline_root": str(baseline_root),
        "baseline_available": True,
        "accepted_as_default": bool(len(merged) > 0 and merged["family_gate_pass"].all()),
        "blocking_reasons": blocking_reasons,
    }
    return merged.sort_values("family").reset_index(drop=True), summary


def save_compare_outputs(out_root: Path, stem: str, compare_df: pd.DataFrame, summary: Dict[str, object]):
    if not compare_df.empty:
        compare_df.to_csv(out_root / f"{stem}.csv", index=False)
    with (out_root / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)


def run_benchmark(cfg: SequenceBenchmarkConfig) -> Dict[str, object]:
    cfg.models = normalize_model_names(cfg.models)
    cfg.families = normalize_family_names(cfg.families)
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    out_root = Path(cfg.outdir) / time.strftime("%Y%m%d_%H%M%S")
    sequence_map, dataset_summary = load_sequence_pool(cfg)
    tuning_active = bool(cfg.enable_family_tuning) and cfg.models == ("LSTM",)
    phase1_result = None
    if tuning_active:
        phase1_root = out_root / "phase1_protocol"
        phase1_result = run_benchmark_pass(
            cfg=cfg,
            out_root=phase1_root,
            sequence_map=sequence_map,
            dataset_summary=dataset_summary,
            device=device,
            family_overrides=None,
            write_outputs=True,
            write_plots=False,
            verbose=True,
        )
        _search_df, selected_overrides = run_lstm_family_tuning_search(
            cfg=cfg,
            out_root=out_root,
            sequence_map=sequence_map,
            dataset_summary=dataset_summary,
            device=device,
        )
        final_result = run_benchmark_pass(
            cfg=cfg,
            out_root=out_root,
            sequence_map=sequence_map,
            dataset_summary=dataset_summary,
            device=device,
            family_overrides=selected_overrides,
            write_outputs=True,
            write_plots=True,
            verbose=True,
        )
        protocol_compare_df = compare_family_results(
            candidate_df=final_result["metrics_by_family"],
            reference_df=phase1_result["metrics_by_family"],
        )
        protocol_compare_df.to_csv(out_root / "protocol_vs_tuned_compare.csv", index=False)
        baseline_root = Path(cfg.baseline_run_root)
        phase1_compare_df, phase1_summary = compare_against_baseline(
            baseline_root=baseline_root,
            candidate_family_df=phase1_result["metrics_by_family"],
            candidate_fold_df=phase1_result["metrics_by_fold"],
        )
        final_compare_df, final_summary = compare_against_baseline(
            baseline_root=baseline_root,
            candidate_family_df=final_result["metrics_by_family"],
            candidate_fold_df=final_result["metrics_by_fold"],
        )
        save_compare_outputs(out_root, "baseline_compare_phase1", phase1_compare_df, phase1_summary)
        save_compare_outputs(out_root, "baseline_compare_phase2", final_compare_df, final_summary)
        acceptance_summary = {
            "phase1_protocol": phase1_summary,
            "phase2_tuned": final_summary,
            "accepted_candidate": (
                "phase2_tuned"
                if final_summary.get("accepted_as_default")
                else ("phase1_protocol" if phase1_summary.get("accepted_as_default") else None)
            ),
        }
        with (out_root / "acceptance_summary.json").open("w", encoding="utf-8") as f:
            json.dump(acceptance_summary, f, ensure_ascii=True, indent=2)
        print("acceptance summary:")
        print(json.dumps(acceptance_summary, ensure_ascii=True, indent=2))
        final_result["phase1_protocol"] = phase1_result
        final_result["acceptance_summary"] = acceptance_summary
        return final_result

    result = run_benchmark_pass(
        cfg=cfg,
        out_root=out_root,
        sequence_map=sequence_map,
        dataset_summary=dataset_summary,
        device=device,
        family_overrides=None,
        write_outputs=True,
        write_plots=True,
        verbose=True,
    )
    if "LSTM" in cfg.models:
        baseline_compare_df, baseline_summary = compare_against_baseline(
            baseline_root=Path(cfg.baseline_run_root),
            candidate_family_df=result["metrics_by_family"],
            candidate_fold_df=result["metrics_by_fold"],
        )
        save_compare_outputs(out_root, "baseline_compare_phase1", baseline_compare_df, baseline_summary)
        result["acceptance_summary"] = baseline_summary
    return result
