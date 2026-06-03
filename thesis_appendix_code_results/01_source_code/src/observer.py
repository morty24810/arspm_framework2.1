from __future__ import annotations
from collections import deque
from typing import Deque, Tuple


class ObserverAgent:
    """
    Online estimator for lambda and DDT using rolling windows of observable events.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.arrivals: Deque[float] = deque()
        self.op_times: Deque[Tuple[float, float]] = deque()

    def reset(self):
        self.arrivals.clear()
        self.op_times.clear()

    def update_on_job_arrival(self, t: float):
        self.arrivals.append(float(t))

    def update_on_op_complete(self, t: float, op_duration: float):
        self.op_times.append((float(t), float(op_duration)))

    def _prune(self, t: float):
        window = float(self.cfg.OBS_WINDOW)
        t0 = t - window
        while self.arrivals and self.arrivals[0] < t0:
            self.arrivals.popleft()
        while self.op_times and self.op_times[0][0] < t0:
            self.op_times.popleft()

    def get_features(self, t: float, avg_slack: float, slack_pressure: float) -> Tuple[float, float, float, float, float, int]:
        self._prune(t)
        window = max(float(self.cfg.OBS_WINDOW), 1e-6)
        arrivals_in_window = len(self.arrivals)
        lam_values = list(getattr(self.cfg, "ARRIVAL_LAM_VALUES", []))
        lam_default = max(lam_values) if lam_values else window
        if arrivals_in_window >= 2:
            total = 0.0
            prev = None
            for ts in self.arrivals:
                if prev is not None:
                    total += (ts - prev)
                prev = ts
            count = max(arrivals_in_window - 1, 1)
            lambda_hat = total / count
        elif arrivals_in_window == 1:
            lambda_hat = window
        else:
            lambda_hat = lam_default

        if self.op_times:
            avg_proc = sum(d for _, d in self.op_times) / len(self.op_times)
        else:
            avg_proc = float(self.cfg.OBS_PROC_DEFAULT)
        denom = max(avg_proc, float(self.cfg.OBS_EPS))
        raw = 1.0 + (avg_slack / denom)
        ddt_values = list(getattr(self.cfg, "DDT_VALUES", (1.0,)))
        if not ddt_values:
            ddt_values = [1.0]
        ddt_min = min(ddt_values)
        ddt_max = max(ddt_values)
        raw = max(ddt_min, min(ddt_max, raw))
        ddt_hat = min(ddt_values, key=lambda v: (abs(v - raw), -v))
        if lam_values:
            lam_sorted = sorted(lam_values)
            lam_auto = lam_sorted[len(lam_sorted) // 2]
        else:
            lam_sorted = None
            lam_auto = lam_default
        lam_thresh = float(getattr(self.cfg, "RUSH_LAM_MEAN_THRESH", lam_auto))
        if lam_sorted and (lam_thresh < lam_sorted[0] or lam_thresh > lam_sorted[-1]):
            lam_thresh = lam_auto
        rush = 1 if (lambda_hat <= lam_thresh or slack_pressure >= self.cfg.RUSH_SLACK_THRESH) else 0
        return (float(arrivals_in_window), float(lambda_hat), float(avg_slack),
                float(slack_pressure), float(ddt_hat), int(rush))
