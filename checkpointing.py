from __future__ import annotations
import json
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def snapshot_config(cfg) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for key in dir(cfg):
        if not key.isupper():
            continue
        value = getattr(cfg, key, None)
        if callable(value):
            continue
        data[key] = _jsonable(value)
    return data


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _observer_state(observer) -> Optional[Dict[str, Any]]:
    if observer is None:
        return None
    try:
        return {
            "arrivals": list(getattr(observer, "arrivals", [])),
            "op_times": list(getattr(observer, "op_times", [])),
        }
    except Exception:
        return None


def _restore_observer(observer, state: Optional[Dict[str, Any]]):
    if observer is None or not state:
        return
    observer.reset()
    arrivals = state.get("arrivals", [])
    op_times = state.get("op_times", [])
    try:
        observer.arrivals.extend(arrivals)
        observer.op_times.extend(op_times)
    except Exception:
        return


def _state_dict_diff(model, state: Dict[str, Any], name: str) -> str:
    model_state = model.state_dict()
    missing = sorted(set(model_state.keys()) - set(state.keys()))
    extra = sorted(set(state.keys()) - set(model_state.keys()))
    shape_mismatch = []
    for key, val in model_state.items():
        if key in state and hasattr(val, "shape"):
            if val.shape != state[key].shape:
                shape_mismatch.append((key, tuple(val.shape), tuple(state[key].shape)))
    if not missing and not extra and not shape_mismatch:
        return ""
    parts = [f"{name} state_dict mismatch:"]
    if missing:
        parts.append(f"  missing={missing}")
    if extra:
        parts.append(f"  extra={extra}")
    if shape_mismatch:
        parts.append(f"  shape_mismatch={shape_mismatch}")
    return "\n".join(parts)


def _safe_load_model(model, state: Dict[str, Any], name: str):
    diff = _state_dict_diff(model, state, name)
    if diff:
        raise ValueError(diff)
    model.load_state_dict(state)


class CheckpointManager:
    def __init__(self, save_dir: str, cfg, device):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.device = str(device)
        self.best_score: Optional[float] = None
        self.latest_path = self.save_dir / "latest.pt"
        self.best_path = self.save_dir / "best.pt"
        self.metrics_path = self.save_dir / "metrics.csv"
        self.config_path = self.save_dir / "config_snapshot.json"
        self._config_snapshot = snapshot_config(cfg)
        if bool(getattr(cfg, "EXPORT_CONFIG_SNAPSHOT", True)):
            self._write_config_snapshot()

    def _write_config_snapshot(self):
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(self._config_snapshot, f, indent=2, ensure_ascii=True)

    def _build_payload(self, sched_agent, maint_agent, observer, env_state: Optional[Dict[str, Any]],
                       metrics: Dict[str, float], step_info: Optional[Dict[str, Any]]):
        models: Dict[str, Any] = {}
        if sched_agent is not None:
            models["high_q"] = sched_agent.q_high.state_dict()
            models["high_target"] = sched_agent.q_high_t.state_dict()
            models["low_q"] = sched_agent.q_low.state_dict()
            models["low_target"] = sched_agent.q_low_t.state_dict()
        if maint_agent is not None:
            models["maint_q"] = maint_agent.q.state_dict()
            models["maint_target"] = maint_agent.qt.state_dict()

        states: Dict[str, Any] = {}
        if sched_agent is not None:
            eps_val = float(sched_agent.eps(sched_agent.steps))
            states["high_extra"] = {"eps": eps_val, "steps": int(sched_agent.steps)}
            states["low_extra"] = {"eps": eps_val, "steps": int(sched_agent.steps)}
        if maint_agent is not None:
            eps_val = float(maint_agent.eps(maint_agent.steps))
            states["maint_extra"] = {"eps": eps_val, "steps": int(maint_agent.steps)}
        obs_state = _observer_state(observer)
        if obs_state is not None:
            states["observer"] = obs_state
        if env_state is not None:
            states["env_state"] = _jsonable(env_state)

        payload = {
            "version": 1,
            "meta": {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "git": _git_commit(),
                "seed": int(getattr(self.cfg, "SEED", 0)),
                "device": self.device,
                "metrics": _jsonable(metrics),
                "step_info": _jsonable(step_info or {}),
                "config": self._config_snapshot,
            },
            "models": models,
            "states": states,
        }
        return payload

    def save_latest(self, sched_agent, maint_agent, observer, env_state: Optional[Dict[str, Any]],
                    metrics: Dict[str, float], step_info: Optional[Dict[str, Any]] = None):
        payload = self._build_payload(sched_agent, maint_agent, observer, env_state, metrics, step_info)
        torch.save(payload, self.latest_path)

    def maybe_save_best(self, sched_agent, maint_agent, observer, env_state: Optional[Dict[str, Any]],
                        metrics: Dict[str, float], step_info: Optional[Dict[str, Any]] = None):
        if not bool(getattr(self.cfg, "SAVE_BEST", True)):
            return
        metric_name = str(getattr(self.cfg, "BEST_METRIC", "total_cost"))
        if metric_name in ("total_cost", "total"):
            score = float(metrics.get("total", metrics.get("tard", 0.0) + metrics.get("maint", 0.0)))
        else:
            score = float(metrics.get(metric_name, metrics.get("total", metrics.get("tard", 0.0) + metrics.get("maint", 0.0))))
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            payload = self._build_payload(sched_agent, maint_agent, observer, env_state, metrics, step_info)
            torch.save(payload, self.best_path)

    def append_metrics(self, metrics: Dict[str, float], seed: int):
        header = "episode,tard,maint,total,timestamp,seed\n"
        if not self.metrics_path.exists():
            self.metrics_path.write_text(header, encoding="utf-8")
        line = f"{metrics.get('episode', '')},{metrics.get('tard', 0.0):.6f},{metrics.get('maint', 0.0):.6f},{metrics.get('total', 0.0):.6f},{time.strftime('%Y-%m-%d %H:%M:%S')},{seed}\n"
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def ensure_best_exists(self):
        if self.best_path.exists() or not self.latest_path.exists():
            return
        self.best_path.write_bytes(self.latest_path.read_bytes())


def load_checkpoint(path: str, sched_agent=None, maint_agent=None, observer=None, map_location="cpu",
                    allow_missing_maint: bool = False):
    ckpt = torch.load(path, map_location=map_location)
    models = ckpt.get("models", {})

    if sched_agent is None and any(k in models for k in ("high_q", "low_q")):
        raise ValueError("Checkpoint contains scheduler weights, but sched_agent is None.")
    if sched_agent is not None:
        _safe_load_model(sched_agent.q_high, models.get("high_q", {}), "high_q")
        _safe_load_model(sched_agent.q_high_t, models.get("high_target", {}), "high_target")
        _safe_load_model(sched_agent.q_low, models.get("low_q", {}), "low_q")
        _safe_load_model(sched_agent.q_low_t, models.get("low_target", {}), "low_target")

    if maint_agent is None and any(k in models for k in ("maint_q", "maint_target")):
        raise ValueError("Checkpoint contains maintenance weights, but maint_agent is None.")
    if maint_agent is not None:
        if "maint_q" in models and "maint_target" in models:
            _safe_load_model(maint_agent.q, models.get("maint_q", {}), "maint_q")
            _safe_load_model(maint_agent.qt, models.get("maint_target", {}), "maint_target")
        elif not allow_missing_maint:
            raise ValueError("Checkpoint missing maintenance weights.")

    states = ckpt.get("states", {})
    if sched_agent is not None and "high_extra" in states:
        sched_agent.steps = int(states["high_extra"].get("steps", sched_agent.steps))
    if maint_agent is not None and "maint_extra" in states:
        maint_agent.steps = int(states["maint_extra"].get("steps", maint_agent.steps))
    _restore_observer(observer, states.get("observer"))

    return ckpt
