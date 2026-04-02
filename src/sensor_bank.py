from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np


class SensorReplayBank:
    """
    Provides per-machine access to replay trajectories via a fixed mapping.
    """
    def __init__(self, degr, machine_curve: Dict[int, int]):
        self.degr = degr
        self.machine_curve = dict(machine_curve)

    def data_no(self, mid: int) -> int:
        return int(self.machine_curve[mid])

    def lifespan(self, mid: int) -> int:
        return int(self.degr.lifespan(self.data_no(mid)))

    def window(self, mid: int, end_idx: int, W: int) -> np.ndarray:
        return self.degr.window(self.data_no(mid), end_idx, W)


class GRUCache:
    """
    Precomputes RUL predictions over each machine's replay sequence for O(1) lookup.
    """
    def __init__(self, rul_predictor, bank: SensorReplayBank, window: int,
                 noise_std: float = 0.0, rng=None):
        self.rul = rul_predictor
        self.bank = bank
        self.window = int(window)
        self.noise_std = float(noise_std)
        self.rng = rng
        self.cache: Dict[int, np.ndarray] = {}

    def build(self):
        self.cache.clear()
        for mid in self.bank.machine_curve.keys():
            life = self.bank.lifespan(mid)
            vals = np.zeros(life, dtype=np.float32)
            data_no = self.bank.data_no(mid)
            for idx in range(life):
                Xw = self.bank.window(mid, idx, self.window)
                vals[idx] = self.rul.predict(data_no, Xw, t_idx=idx, lifespan=life)
            vals = np.minimum.accumulate(np.clip(vals, 0.0, 1.0))
            self.cache[mid] = vals.astype(np.float32)

    def get_h(self, mid: int, idx: int) -> float:
        arr = self.cache.get(mid)
        if arr is None or arr.size == 0:
            return 1.0
        i = min(max(int(idx), 0), int(arr.size - 1))
        return float(arr[i])

    def get_h_obs(self, mid: int, idx: int) -> float:
        h = self.get_h(mid, idx)
        if self.noise_std > 0.0:
            if self.rng is None:
                noise = np.random.normal(0.0, self.noise_std)
            else:
                noise = self.rng.normalvariate(0.0, self.noise_std)
            h = float(np.clip(h + noise, 0.0, 1.0))
        return float(h)
