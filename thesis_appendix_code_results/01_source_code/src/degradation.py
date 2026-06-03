from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

@dataclass
class DegradationProfile:
    data_no: int
    features: List[str]
    X: np.ndarray  # shape (T,F)

class DegradationReplay:
    """
    Replays real sensor trajectories (per Data_No) as the simulator's PHM stream.
    The simulator maps "machine operating time" -> index in the trajectory.
    """
    def __init__(self, train_csv: str, selected_data_nos: List[int], features: List[str], noise_std: float = 0.0):
        self.train_csv = train_csv
        self.selected_data_nos = selected_data_nos
        self.features = features
        self.noise_std = float(noise_std)

        df = pd.read_csv(train_csv).sort_values(["Data_No", "Time"]).reset_index(drop=True)
        self.profiles: Dict[int, DegradationProfile] = {}
        for dn in selected_data_nos:
            g = df[df["Data_No"] == dn].copy()
            if g.empty:
                raise ValueError(f"Data_No={dn} not found in {train_csv}")
            X = g[features].to_numpy(dtype=np.float32)
            self.profiles[dn] = DegradationProfile(dn, features, X)

        self.max_len = max(p.X.shape[0] for p in self.profiles.values())

    def lifespan(self, data_no: int) -> int:
        return int(self.profiles[data_no].X.shape[0])

    def sensor_at(self, data_no: int, idx: int) -> np.ndarray:
        prof = self.profiles[data_no]
        idx = int(np.clip(idx, 0, prof.X.shape[0]-1))
        x = prof.X[idx].copy()
        if self.noise_std > 0:
            x = x + np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
        return x

    def window(self, data_no: int, end_idx: int, W: int) -> np.ndarray:
        prof = self.profiles[data_no]
        end_idx = int(np.clip(end_idx, 0, prof.X.shape[0]-1))
        start = end_idx - W + 1
        if start < 0:
            pad = np.repeat(prof.X[0:1], repeats=-start, axis=0)
            core = prof.X[0:end_idx+1]
            Xw = np.concatenate([pad, core], axis=0)
        else:
            Xw = prof.X[start:end_idx+1]
        if self.noise_std > 0:
            Xw = Xw + np.random.normal(0.0, self.noise_std, size=Xw.shape).astype(np.float32)
        return Xw.astype(np.float32)
