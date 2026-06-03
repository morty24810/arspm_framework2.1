from __future__ import annotations
import argparse
import copy
import math
import random
import time
from pathlib import Path
from typing import Any, Dict

import torch

from config import SimConfig
from src.utils import set_seed
from src.agents import HierMaintenanceAgentDDQN, MaintenanceAgentDDQN, PPOSchedulerAgent, THDQNAgent
from src.pomcp import POMCPPlanner
from src.compare import combo_dominant_maps, compare_mode_results, summarize_combo_conditioned_behavior, summarize_scheduling_strategy, write_mode_comparison_outputs
from checkpointing import load_checkpoint, load_maintenance_only
from run_experiment import (
    apply_experiment_profile,
    apply_machine_set_mode,
    build_degradation_and_rul,
    build_episode_combos,
    build_episode_scenario,
    build_scenario_bank,
    canonical_scheduler_mode,
    evaluate_once,
    build_policy_tag,
    build_policy_label,
    build_policy_context_label,
    build_maint_mode_tag,
    effective_base_degradation_rate,
    effective_degradation_rate_scale,
    experiment_result_metadata,
    maintenance_mode_family,
    maintenance_variant_for_context,
    make_rng,
    normalize_jobs_target_for_combo_mode,
    write_summary_files,
    validate_region_thresholds,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference/demo from a saved checkpoint.")
    parser.add_argument("--ckpt", default="checkpoints/latest.pt", help="Checkpoint path (.pt)")
    parser.add_argument("--seed", type=int, default=None) # random seed
    parser.add_argument("--episodes", type=int, default=1) # number of episodes to run
    parser.add_argument("--jobs_target", type=int, default=None) # how many jobs to schedule in each episode
    parser.add_argument("--machines", type=int, default=None) # override number of machines
    parser.add_argument("--randomize_combos", type=int, default=0) # randomize lambda/DDT combos
    parser.add_argument("--segment_jobs", type=int, default=None) # jobs per combo segment
    parser.add_argument("--combo_plan", type=str, default="") # lam:ddt[:segments],lam:ddt[:segments]
    parser.add_argument("--maint_mode", type=str, default="pomcp", choices=["flat_ddqn", "hier_ddqn", "pomcp", "OFF"]) # maintenance mode
    parser.add_argument("--compare_maint_modes", type=int, default=0, choices=[0, 1]) # compare pomcp vs explicit ddqn variant
    parser.add_argument("--dqn_ckpt", type=str, default="") # optional DQN maintenance checkpoint for compare mode
    parser.add_argument("--compare_unrestricted", type=int, default=1, choices=[0, 1]) # run constrained vs unrestricted side-by-side
    parser.add_argument("--outdir", type=str, default="outputs/demo_run_001") # output directory
    return parser.parse_args()


def parse_combo_plan(plan: str):
    combos = []
    seq = []
    for chunk in plan.split(","):
        item = chunk.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(":") if p.strip()]
        if len(parts) < 2:
            raise ValueError(f"invalid combo entry: '{chunk}'")
        lam = float(parts[0])
        ddt = float(parts[1])
        segments = int(parts[2]) if len(parts) > 2 else 1
        if segments < 1:
            raise ValueError(f"segments must be >= 1 in '{chunk}'")
        idx = len(combos)
        combos.append((lam, ddt))
        seq.extend([idx] * segments)
    if not combos:
        raise ValueError("combo_plan produced no combos")
    return combos, seq


def make_maint_agent(cfg: SimConfig, seed: int, device: torch.device) -> MaintenanceAgentDDQN:
    arch = str(getattr(cfg, "MAINT_AGENT_ARCH", "flat_ddqn")).strip().lower()
    agent_cls = HierMaintenanceAgentDDQN if arch == "hier_ddqn" else MaintenanceAgentDDQN
    return agent_cls(
        state_dim=int(getattr(cfg, "MAINTENANCE_STATE_DIM", 18)),
        cfg=cfg,
        rng=random.Random(seed),
        device=device,
    )


def make_sched_agent(cfg: SimConfig, seed: int, device: torch.device,
                     state_dim: int | None = None, low_state_dim: int | None = None,
                     low_state_mode: str | None = None):
    scheduler_mode = canonical_scheduler_mode(getattr(cfg, "SCHEDULER_MODE", "THDQN"))
    state_dim = int(getattr(cfg, "SCHEDULER_STATE_DIM", 15) if state_dim is None else state_dim)
    if scheduler_mode == "PPO":
        return PPOSchedulerAgent(state_dim=int(state_dim), cfg=cfg, rng=random.Random(seed), device=device)
    if low_state_mode is not None:
        cfg.THDQN_LOW_STATE_MODE = str(low_state_mode)
    return THDQNAgent(
        state_dim=int(state_dim),
        low_state_dim=int(getattr(cfg, "THDQN_LOW_STATE_DIM", 11) if low_state_dim is None else low_state_dim),
        cfg=cfg,
        rng=random.Random(seed),
        device=device,
    )


def build_infer_route_result(metrics: Dict[str, Any], env: Any, overdue_stats: Dict[str, Any],
                             policy_label: str, maint_mode_tag: str) -> Dict[str, Any]:
    cfg_env = getattr(env, "cfg", None)
    maint_variant = maint_mode_tag.replace("maint_", "").lower()
    return {
        "metrics": metrics,
        "env": env,
        "overdue": overdue_stats,
        "policy_label": policy_label,
        "decision_log": list(getattr(env, "last_decision_log", [])),
        "maint_mode_tag": maint_mode_tag,
        "maint_variant": maint_variant,
        "maint_arch": "pomcp" if maint_variant == "pomcp" else maint_variant,
        **experiment_result_metadata(
            cfg_env if cfg_env is not None else SimConfig(),
            maint_mode=maint_variant,
            compare_type="full_system",
            scheduler_mode=str(getattr(env, "last_scheduler_mode", "THDQN")).upper(),
            train_policy_tag=getattr(env, "last_train_policy_tag", None),
            eval_policy_tag=getattr(env, "last_eval_policy_tag", None),
        ),
        "scheduler_mode": str(getattr(env, "last_scheduler_mode", "THDQN")).upper(),
        "scheduler_mode_tag": f"sched_{str(getattr(env, 'last_scheduler_mode', 'THDQN')).lower()}",
        "sched_regime_feature_mode": str(getattr(env, "last_sched_regime_feature_mode", getattr(getattr(env, "cfg", None), "SCHED_REGIME_FEATURE_MODE", "observer"))).lower(),
    }


def build_infer_summary_row(ts: str, episode_idx: int, cfg_eval: SimConfig, seed: int,
                            jobs_target: int, policy_tag: str, policy_label: str,
                            maint_mode: str, maint_mode_tag: str, metrics: Dict[str, Any],
                            env: Any, overdue_stats: Dict[str, Any], *,
                            degradation_rate: float, degradation_rate_scale: float) -> Dict[str, Any]:
    decision_log = list(getattr(env, "last_decision_log", []))
    sched_summary = summarize_scheduling_strategy(decision_log, env=env)
    combo_behavior = summarize_combo_conditioned_behavior(decision_log)
    dominant_rule_by_combo, dominant_goal_by_combo = combo_dominant_maps(combo_behavior)
    maint_rows = [row for row in decision_log if row.get("event") == "maintenance"]
    return {
        "timestamp": ts,
        "episode": int(episode_idx),
        **experiment_result_metadata(
            cfg_eval,
            maint_mode=getattr(cfg_eval, "MAINT_VARIANT", maint_mode),
            compare_type="full_system",
            scheduler_mode=cfg_eval.SCHEDULER_MODE,
            train_policy_tag=policy_tag,
            eval_policy_tag=policy_tag,
        ),
        "policy_tag": policy_tag,
        "policy_label": policy_label,
        "maint_mode": maintenance_mode_family(maint_mode),
        "maint_variant": str(getattr(cfg_eval, "MAINT_VARIANT", maint_mode)),
        "maint_mode_tag": maint_mode_tag,
        "scheduler_mode": str(cfg_eval.SCHEDULER_MODE).upper(),
        "scheduler_mode_tag": f"sched_{str(cfg_eval.SCHEDULER_MODE).lower()}",
        "sched_regime_feature_mode": str(getattr(cfg_eval, "SCHED_REGIME_FEATURE_MODE", "observer")).lower(),
        "lam_ddt_mode": str(getattr(cfg_eval, "LAM_DDT_MODE", "variable")).lower(),
        "train_combo_mode": str(getattr(cfg_eval, "TRAIN_COMBO_MODE", getattr(cfg_eval, "LAM_DDT_MODE", "variable"))).lower(),
        "eval_combo_mode": str(getattr(cfg_eval, "EVAL_COMBO_MODE", getattr(cfg_eval, "LAM_DDT_MODE", "variable"))).lower(),
        "fixed_arrival_lam": float(getattr(cfg_eval, "FIXED_ARRIVAL_LAM", 40.0)),
        "fixed_ddt": float(getattr(cfg_eval, "FIXED_DDT", 1.5)),
        "enforce_region_policy": int(cfg_eval.ENFORCE_REGION_POLICY),
        "hx": float(cfg_eval.Hx),
        "hy": float(cfg_eval.Hy),
        "seed": int(seed),
        "jobs_target": int(jobs_target),
        "tard": float(metrics["tard"]),
        "maint": float(metrics["maint"]),
        "total": float(metrics["total"]),
        "overdue_ratio_ops": float(overdue_stats["ratio_ops"]),
        "overdue_ratio_time": float(overdue_stats["ratio_time"]),
        "overdue_ops": float(overdue_stats["overdue_ops"]),
        "total_ops": float(overdue_stats["total_ops"]),
        "breakdown_count": int(getattr(env, "breakdown_count", 0)),
        "breakdown_cost": float(getattr(env, "breakdown_cost_total", 0.0)),
        "requeued_op_count": int(getattr(env, "requeued_op_count", 0)),
        "interrupted_proc_time": float(getattr(env, "interrupted_proc_time", 0.0)),
        "hard_breakdown_count": int(getattr(env, "hard_breakdown_count", 0)),
        "stochastic_breakdown_count": int(getattr(env, "stochastic_breakdown_count", 0)),
        "current_stress_mean": float(sched_summary.get("current_stress_mean", 0.0)),
        "current_stress_max": float(sched_summary.get("current_stress_max", 0.0)),
        "degradation_rate": float(degradation_rate),
        "degradation_rate_scale": float(degradation_rate_scale),
        "machine_set_mode": str(getattr(cfg_eval, "MACHINE_SET_MODE", "current6")),
        "machine_curve_ids": list(getattr(cfg_eval, "MACHINE_CURVE_IDS", ())),
        "rul_life_clock_mode": str(getattr(cfg_eval, "RUL_LIFE_CLOCK_MODE", "label_driven_scaled")),
        "machine_replay_to_label_scale": {
            int(mid): float(scale)
            for mid, scale in getattr(env, "machine_replay_to_label_scale", {}).items()
        },
        "scenario_combos": [list(x) for x in getattr(getattr(env, "episode_scenario", None), "combos", [])],
        "scenario_combo_seq": list(getattr(getattr(env, "episode_scenario", None), "combo_seq", [])),
        "combo_behavior": combo_behavior,
        "dominant_rule_by_combo": dominant_rule_by_combo,
        "dominant_goal_by_combo": dominant_goal_by_combo,
        "im_invalid_filtered_count": int(sum(1 for row in maint_rows if bool(row.get("im_invalid_flag")))),
        "dn_veto_count": int(sum(1 for row in maint_rows if bool(row.get("dn_imminent_breakdown_veto")))),
        "cm_emergency_override_count": int(sum(1 for row in maint_rows if bool(row.get("cm_emergency_override")))),
        "cm_preference_penalty_count": int(sum(1 for row in maint_rows if bool(row.get("cm_preference_penalty_active")))),
        "gate_dn_count": int(sum(1 for row in maint_rows if str(row.get("maint_gate_action", "")).upper() == "DN")),
        "gate_maint_count": int(sum(1 for row in maint_rows if str(row.get("maint_gate_action", "")).upper() == "MAINT")),
        "gate_penalty_count": int(sum(1 for row in maint_rows if bool(row.get("gate_penalty_active")))),
        "type_penalty_count": int(sum(1 for row in maint_rows if bool(row.get("type_penalty_active")))),
        "high_health_cm_count": int(sum(
            1 for row in maint_rows
            if str(row.get("kind", "")).upper() == "CM"
            and float(row.get("h", -1.0)) > float(getattr(cfg_eval, "THDQN_DQN_HIGH_HEALTH_CM_H", 0.70))
        )),
    }


def should_use_formal_final_eval_scenario(args: argparse.Namespace) -> bool:
    return (
        args.jobs_target is None
        and args.segment_jobs is None
        and not bool(int(args.randomize_combos))
        and not args.combo_plan.strip()
    )


def infer_scheduler_state_dim(ckpt_path: Path, map_location: torch.device) -> int:
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    models = ckpt.get("models", {})
    weight = models.get("high_q", {}).get("net.0.weight")
    if hasattr(weight, "shape") and len(weight.shape) >= 2:
        return int(weight.shape[1])
    weight = models.get("sched_actor", {}).get("net.0.weight")
    if hasattr(weight, "shape") and len(weight.shape) >= 2:
        return int(weight.shape[1])
    return 15


def infer_thdqn_low_state_dim(ckpt_path: Path, map_location: torch.device) -> int:
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    meta_cfg = ckpt.get("meta", {}).get("config", {}) or {}
    if "THDQN_LOW_STATE_DIM" in meta_cfg:
        return int(meta_cfg["THDQN_LOW_STATE_DIM"])
    models = ckpt.get("models", {})
    weight = models.get("low_q", {}).get("net.0.weight")
    if hasattr(weight, "shape") and len(weight.shape) >= 2:
        return int(weight.shape[1] - 4)
    return 11


def infer_thdqn_low_state_mode(ckpt_path: Path, map_location: torch.device) -> str:
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    meta_cfg = ckpt.get("meta", {}).get("config", {}) or {}
    mode = str(meta_cfg.get("THDQN_LOW_STATE_MODE", "")).strip().lower()
    if mode in {"pruned", "full"}:
        return mode
    state_dim = infer_scheduler_state_dim(ckpt_path, map_location)
    low_state_dim = infer_thdqn_low_state_dim(ckpt_path, map_location)
    models = ckpt.get("models", {})
    weight = models.get("low_q", {}).get("net.0.weight")
    if not (hasattr(weight, "shape") and len(weight.shape) >= 2):
        raise ValueError("Unable to infer THDQN low-state mode from checkpoint.")
    low_input_dim = int(weight.shape[1] - 4)
    if low_input_dim == int(state_dim):
        return "full"
    if low_input_dim == int(low_state_dim):
        return "pruned"
    raise ValueError(
        f"Ambiguous THDQN low-state mode in checkpoint: low_input_dim={low_input_dim}, "
        f"state_dim={state_dim}, low_state_dim={low_state_dim}"
    )


def infer_maint_state_dim(ckpt_path: Path, map_location: torch.device) -> int:
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    meta_cfg = ckpt.get("meta", {}).get("config", {}) or {}
    if "MAINTENANCE_STATE_DIM" in meta_cfg:
        return int(meta_cfg["MAINTENANCE_STATE_DIM"])
    models = ckpt.get("models", {})
    weight = models.get("maint_gate_q", {}).get("net.0.weight")
    if hasattr(weight, "shape") and len(weight.shape) >= 2:
        return int(weight.shape[1])
    weight = models.get("maint_q", {}).get("net.0.weight")
    if hasattr(weight, "shape") and len(weight.shape) >= 2:
        return int(weight.shape[1])
    return 18


def infer_maint_agent_arch(ckpt_path: Path, map_location: torch.device) -> str:
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    meta_cfg = ckpt.get("meta", {}).get("config", {}) or {}
    arch = str(meta_cfg.get("MAINT_AGENT_ARCH", "")).strip().lower()
    if arch in {"flat_ddqn", "hier_ddqn"}:
        return arch
    models = ckpt.get("models", {})
    if any(k in models for k in ("maint_gate_q", "maint_gate_target", "maint_type_q", "maint_type_target")):
        return "hier_ddqn"
    return "flat_ddqn"


def infer_scheduler_mode(ckpt_path: Path, map_location: torch.device) -> str:
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    meta_cfg = ckpt.get("meta", {}).get("config", {}) or {}
    scheduler_mode = str(meta_cfg.get("SCHEDULER_MODE", "")).upper()
    if scheduler_mode:
        return canonical_scheduler_mode(scheduler_mode)
    models = ckpt.get("models", {})
    if "sched_actor" in models or "sched_critic" in models:
        return "PPO"
    return "THDQN"


def infer_checkpoint_config(ckpt_path: Path, map_location: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    return dict(ckpt.get("meta", {}).get("config", {}) or {})


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    cfg = SimConfig()
    meta_cfg = infer_checkpoint_config(ckpt_path, torch.device("cpu"))
    for key, value in meta_cfg.items():
        if key.isupper() and hasattr(cfg, key):
            try:
                setattr(cfg, key, value)
            except Exception:
                pass
    apply_experiment_profile(cfg)
    validate_region_thresholds(cfg)
    cfg.SEED = int(cfg.SEED if args.seed is None else args.seed)
    _, resolved_machine_curve_ids = apply_machine_set_mode(cfg)
    if args.machines is not None:
        cfg.NUM_MACHINES = int(args.machines)
    requested_maint_mode = str(args.maint_mode).strip()
    cfg.ENFORCE_REGION_POLICY = True
    checkpoint_enforce_region = bool(meta_cfg.get("ENFORCE_REGION_POLICY", True))
    cfg.COMBO_RANDOMIZE = bool(int(args.randomize_combos))
    if args.segment_jobs is not None:
        cfg.COMBO_SEGMENT_JOBS = max(1, int(args.segment_jobs))
    jobs_target = int(cfg.EVAL_JOBS_TARGET * 2 if args.jobs_target is None else args.jobs_target)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:", device)

    compare_maint_modes = bool(int(args.compare_maint_modes))
    dqn_ckpt_path = Path(args.dqn_ckpt) if args.dqn_ckpt else None
    if compare_maint_modes and dqn_ckpt_path is None:
        raise ValueError("--compare_maint_modes=1 requires --dqn_ckpt.")
    if dqn_ckpt_path is not None and not dqn_ckpt_path.exists():
        raise FileNotFoundError(f"DQN maintenance checkpoint not found: {dqn_ckpt_path}")

    machine_curve_ids = list(resolved_machine_curve_ids)
    _, degr, rul = build_degradation_and_rul(cfg, machine_curve_ids)

    cfg.SCHEDULER_MODE = infer_scheduler_mode(ckpt_path, device)
    sched_state_dim = infer_scheduler_state_dim(ckpt_path, device)
    thdqn_low_state_dim = infer_thdqn_low_state_dim(ckpt_path, device)
    thdqn_low_state_mode = infer_thdqn_low_state_mode(ckpt_path, device) if cfg.SCHEDULER_MODE == "THDQN" else None
    maint_agent_arch = infer_maint_agent_arch(ckpt_path, device)
    maint_state_dim = infer_maint_state_dim(ckpt_path, device)
    cfg.SCHEDULER_STATE_DIM = int(sched_state_dim)
    cfg.THDQN_LOW_STATE_DIM = int(thdqn_low_state_dim)
    if thdqn_low_state_mode is not None:
        cfg.THDQN_LOW_STATE_MODE = str(thdqn_low_state_mode)
    cfg.MAINT_AGENT_ARCH = str(maint_agent_arch)
    cfg.MAINTENANCE_STATE_DIM = int(maint_state_dim)
    cfg.MAINT_VARIANT = maintenance_variant_for_context(
        cfg,
        requested_maint_mode,
        enforce_region=True,
        scheduler_mode=cfg.SCHEDULER_MODE,
    )
    cfg.MAINT_MODE = maintenance_mode_family(cfg.MAINT_VARIANT)
    if cfg.MAINT_MODE == "DQN":
        cfg.MAINT_AGENT_ARCH = str(cfg.MAINT_VARIANT)
    if (
        str(cfg.SCHEDULER_MODE).upper() == "THDQN"
        and str(cfg.MAINT_VARIANT).lower() == "hier_ddqn"
        and not checkpoint_enforce_region
        and str(cfg.MAINT_AGENT_ARCH).lower() != "hier_ddqn"
    ):
        raise ValueError("Legacy flat THDQN + DQN + unrestricted maintenance checkpoints are not compatible with the hierarchical maintenance architecture.")
    if compare_maint_modes and cfg.MAINT_MODE == "OFF":
        raise ValueError("--compare_maint_modes=1 does not support maint_mode=OFF.")
    maint_agent = make_maint_agent(cfg, cfg.SEED, device)
    sched_agent = make_sched_agent(
        cfg,
        cfg.SEED,
        device,
        state_dim=sched_state_dim,
        low_state_dim=thdqn_low_state_dim,
        low_state_mode=thdqn_low_state_mode,
    )

    allow_missing_maint = cfg.MAINT_MODE != "DQN"
    ckpt = load_checkpoint(
        str(ckpt_path),
        sched_agent=sched_agent,
        maint_agent=maint_agent,
        observer=None,
        map_location=device,
        allow_missing_maint=allow_missing_maint,
    )
    ckpt_states = ckpt.get("states", {})
    observer_state = ckpt_states.get("observer")
    env_state = ckpt_states.get("env_state")

    if isinstance(sched_agent, THDQNAgent):
        sched_agent.q_high.eval()
        sched_agent.q_high_t.eval()
        sched_agent.q_low.eval()
        sched_agent.q_low_t.eval()
    else:
        sched_agent.actor.eval()
        sched_agent.critic.eval()
    if isinstance(maint_agent, HierMaintenanceAgentDDQN):
        maint_agent.q_gate.eval()
        maint_agent.q_gate_t.eval()
        maint_agent.q_type.eval()
        maint_agent.q_type_t.eval()
    else:
        maint_agent.q.eval()
        maint_agent.qt.eval()

    dqn_compare_agent = None
    compare_dqn_variant = None
    if compare_maint_modes:
        compare_cfg = copy.deepcopy(cfg)
        compare_cfg.MAINT_AGENT_ARCH = infer_maint_agent_arch(dqn_ckpt_path, device)
        compare_cfg.MAINTENANCE_STATE_DIM = infer_maint_state_dim(dqn_ckpt_path, device)
        compare_cfg.MAINT_VARIANT = str(compare_cfg.MAINT_AGENT_ARCH).lower()
        compare_cfg.MAINT_MODE = maintenance_mode_family(compare_cfg.MAINT_VARIANT)
        compare_dqn_variant = str(compare_cfg.MAINT_VARIANT)
        dqn_compare_agent = make_maint_agent(compare_cfg, cfg.SEED + 997, device)
        load_maintenance_only(str(dqn_ckpt_path), dqn_compare_agent, map_location=device, allow_missing_maint=False)
        if isinstance(dqn_compare_agent, HierMaintenanceAgentDDQN):
            dqn_compare_agent.q_gate.eval()
            dqn_compare_agent.q_gate_t.eval()
            dqn_compare_agent.q_type.eval()
            dqn_compare_agent.q_type_t.eval()
        else:
            dqn_compare_agent.q.eval()
            dqn_compare_agent.qt.eval()

    maint_modes = [str(cfg.MAINT_VARIANT)]
    if compare_maint_modes:
        if compare_dqn_variant and compare_dqn_variant not in maint_modes:
            maint_modes.append(compare_dqn_variant)
        if "pomcp" not in maint_modes:
            maint_modes.append("pomcp")

    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    combo_plan = args.combo_plan.strip()

    for ep in range(int(args.episodes)):
        seed = int(cfg.SEED) + ep
        set_seed(seed)
        cfg_eval_base = copy.deepcopy(cfg)
        cfg_eval_base.SEED = seed
        if should_use_formal_final_eval_scenario(args):
            scenario_bank = build_scenario_bank(seed, cfg_eval_base, degr, machine_curve_ids=machine_curve_ids)
            scenario = scenario_bank.final_eval_scenario
            jobs_target = int(scenario.jobs_target)
        elif combo_plan:
            combos, seq = parse_combo_plan(combo_plan)
            target_segments = max(1, int(math.ceil(int(jobs_target) / cfg_eval_base.COMBO_SEGMENT_JOBS)))
            if len(seq) < target_segments:
                seq.extend([seq[-1]] * (target_segments - len(seq)))
            elif len(seq) > target_segments:
                seq = seq[:target_segments]
            cfg_eval_base.COMBO_RANDOMIZE = False
            scenario = build_episode_scenario(
                cfg_eval_base,
                degr,
                jobs_target=int(jobs_target),
                scenario_rng=make_rng(seed, "infer", ep + 1, "scenario"),
                degradation_rate=float(effective_base_degradation_rate(cfg_eval_base)),
                episode_combos=combos,
                episode_combo_seq=seq,
                machine_curve_ids=machine_curve_ids,
            )
        else:
            combo_mode = str(getattr(cfg_eval_base, "EVAL_COMBO_MODE", getattr(cfg_eval_base, "LAM_DDT_MODE", "variable")))
            jobs_target = normalize_jobs_target_for_combo_mode(cfg_eval_base, int(jobs_target), combo_mode)
            combo_rng = random.Random(seed)
            combos, seq = build_episode_combos(
                cfg_eval_base,
                combo_rng,
                int(jobs_target),
                combo_mode=combo_mode,
            )
            scenario = build_episode_scenario(
                cfg_eval_base,
                degr,
                jobs_target=int(jobs_target),
                scenario_rng=make_rng(seed, "infer", ep + 1, "scenario"),
                degradation_rate=float(effective_base_degradation_rate(cfg_eval_base)),
                episode_combos=combos,
                episode_combo_seq=seq,
                machine_curve_ids=machine_curve_ids,
                combo_mode=combo_mode,
            )
        scenario_combos = list(getattr(scenario, "combos", []))
        scenario_combo_seq = list(getattr(scenario, "combo_seq", []))
        outdir = base_outdir if args.episodes == 1 else (base_outdir / f"episode_{ep+1:03d}")
        outdir.mkdir(parents=True, exist_ok=True)

        route_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        route_modes = [True, False] if int(args.compare_unrestricted) else [True]
        for enforce_region in route_modes:
            route_results.setdefault(build_policy_tag(cfg_eval_base, enforce_region), {})
            for maint_variant in maint_modes:
                cfg_eval = copy.deepcopy(cfg_eval_base)
                cfg_eval.ENFORCE_REGION_POLICY = enforce_region
                cfg_eval.MAINT_VARIANT = maintenance_variant_for_context(
                    cfg_eval,
                    maint_variant,
                    enforce_region=enforce_region,
                    scheduler_mode=cfg_eval.SCHEDULER_MODE,
                )
                cfg_eval.MAINT_MODE = maintenance_mode_family(cfg_eval.MAINT_VARIANT)
                if cfg_eval.MAINT_MODE == "DQN":
                    cfg_eval.MAINT_AGENT_ARCH = str(cfg_eval.MAINT_VARIANT)
                policy_tag = build_policy_tag(cfg_eval, enforce_region)
                maint_mode_tag = build_maint_mode_tag(cfg_eval.MAINT_VARIANT)
                policy_label = build_policy_context_label(cfg_eval, enforce_region, cfg_eval.MAINT_VARIANT)
                route_rng = random.Random(seed)
                pomcp_ep = (
                    POMCPPlanner(num_actions=3, gamma=cfg.GAMMA, c_ucb=cfg.POMCP_UCB_C, rng=route_rng)
                    if cfg_eval.MAINT_MODE == "POMCP" else None
                )
                if cfg_eval.MAINT_MODE == "DQN":
                    use_compare_agent = bool(compare_maint_modes and compare_dqn_variant == cfg_eval.MAINT_VARIANT and cfg_eval.MAINT_VARIANT != cfg.MAINT_VARIANT)
                    maint_agent_eval = dqn_compare_agent if use_compare_agent else maint_agent
                else:
                    maint_agent_eval = None

                stem_parts = [ts]
                if args.episodes != 1:
                    stem_parts.append(f"{ep+1:03d}")
                stem_parts.extend([maint_mode_tag, policy_tag])
                artifact_stem = "_".join(stem_parts)
                with torch.no_grad():
                    metrics, env, overdue_stats = evaluate_once(
                        cfg_eval,
                        route_rng,
                        degr,
                        rul,
                        sched_agent,
                        maint_agent_eval,
                        cfg_eval.MAINT_VARIANT,
                        pomcp_ep,
                        machine_curve_ids,
                        jobs_target=int(jobs_target),
                        episode_combos=scenario_combos,
                        episode_combo_seq=scenario_combo_seq,
                        generate_outputs=True,
                        outdir=outdir,
                        plot_prefix=artifact_stem,
                        decision_log_name=f"decision_log_{artifact_stem}.csv",
                        freeze_steps=True,
                        observer_state=observer_state,
                        env_state=env_state,
                        policy_label=policy_label,
                        threshold_enforced=enforce_region,
                        scenario=scenario,
                        env_rng=make_rng(seed, "infer", ep + 1, policy_tag, "env_breakdown"),
                        rul_obs_rng=make_rng(seed, "infer", ep + 1, policy_tag, "env_obs"),
                        belief_rng=make_rng(seed, cfg_eval.MAINT_VARIANT, "infer", ep + 1, policy_tag, "belief"),
                    )
                route_results[policy_tag][cfg_eval.MAINT_VARIANT] = build_infer_route_result(
                    metrics,
                    env,
                    overdue_stats,
                    policy_label,
                    maint_mode_tag,
                )
                summary_row = build_infer_summary_row(
                    ts,
                    ep + 1,
                    cfg_eval,
                    seed,
                    jobs_target,
                    policy_tag,
                    policy_label,
                    cfg_eval.MAINT_VARIANT,
                    maint_mode_tag,
                    metrics,
                    env,
                    overdue_stats,
                    degradation_rate=float(scenario.degradation_rate),
                    degradation_rate_scale=float(effective_degradation_rate_scale(cfg_eval_base)),
                )
                write_summary_files(outdir, f"summary_{artifact_stem}", summary_row)
                print(
                    f"episode {ep+1}/{args.episodes} [{policy_tag}][{maint_mode_tag}] "
                    f"tard={metrics['tard']:.2f} "
                    f"maint={metrics['maint']:.2f} "
                    f"total={metrics['total']:.2f} "
                    f"overdue_ratio={overdue_stats['ratio_ops']:.3f}"
                )

            if compare_maint_modes and "pomcp" in route_results[policy_tag]:
                primary_mode = str(cfg.MAINT_VARIANT) if str(cfg.MAINT_VARIANT) in route_results[policy_tag] else None
                compare_mode = "pomcp"
                if primary_mode == "pomcp":
                    primary_mode = compare_dqn_variant if compare_dqn_variant in route_results[policy_tag] else None
                elif primary_mode not in route_results[policy_tag]:
                    primary_mode = compare_dqn_variant if compare_dqn_variant in route_results[policy_tag] else None
                if primary_mode is None:
                    primary_mode = next((mode for mode in route_results[policy_tag].keys() if mode != "pomcp"), None)
                if primary_mode is None:
                    continue
                primary_result = route_results[policy_tag][primary_mode]
                compare_result = route_results[policy_tag][compare_mode]
                compare_summary, compare_rows = compare_mode_results(
                    primary_result,
                    compare_result,
                    primary_mode,
                    compare_mode,
                )
                compare_summary["policy_tag"] = policy_tag
                compare_summary["policy_label"] = build_policy_label(cfg_eval_base, enforce_region)
                compare_stem_parts = ["maint_compare", ts]
                if args.episodes != 1:
                    compare_stem_parts.append(f"{ep+1:03d}")
                compare_stem_parts.append(policy_tag)
                compare_label = (
                    f"{build_policy_label(cfg_eval_base, enforce_region)} | "
                    f"maintenance: {primary_mode} vs {compare_mode}"
                )
                write_mode_comparison_outputs(
                    outdir,
                    "_".join(compare_stem_parts),
                    compare_summary,
                    compare_rows,
                    policy_label=compare_label,
                )

        constrained_tag = build_policy_tag(cfg_eval_base, True)
        unrestricted_tag = build_policy_tag(cfg_eval_base, False)
        if constrained_tag in route_results and unrestricted_tag in route_results:
            shared_modes = [
                maint_variant for maint_variant in route_results[constrained_tag].keys()
                if maint_variant in route_results[unrestricted_tag]
            ]
            for maint_mode in shared_modes:
                if maint_mode not in route_results[constrained_tag] or maint_mode not in route_results[unrestricted_tag]:
                    continue
                c = route_results[constrained_tag][maint_mode]
                u = route_results[unrestricted_tag][maint_mode]
                print(
                    f"delta(unrestricted-constrained) [{build_maint_mode_tag(maint_mode)}]: "
                    f"tard={u['metrics']['tard'] - c['metrics']['tard']:.3f}, "
                    f"maint={u['metrics']['maint'] - c['metrics']['maint']:.3f}, "
                    f"total={u['metrics']['total'] - c['metrics']['total']:.3f}, "
                    f"overdue_ratio={u['overdue']['ratio_ops'] - c['overdue']['ratio_ops']:.3f}"
                )
                route_summary, route_rows = compare_mode_results(
                    c,
                    u,
                    "CONSTRAINED",
                    "UNRESTRICTED",
                    compare_type="full_system_route_compare",
                    train_policy_tag=constrained_tag,
                    eval_policy_tag=unrestricted_tag,
                    scheduler_anchor="none",
                )
                route_summary["policy_tag"] = f"{constrained_tag}__to__{unrestricted_tag}"
                route_summary["policy_label"] = "Route delta: unrestricted - constrained"
                route_summary["maint_mode"] = maintenance_mode_family(str(maint_mode))
                route_summary["maint_variant"] = str(maint_mode)
                route_summary["maint_mode_tag"] = build_maint_mode_tag(maint_mode)
                route_summary["route_delta_direction"] = "unrestricted_minus_constrained"
                route_summary["seed"] = int(seed)
                route_dir = outdir / "route_compare"
                route_dir.mkdir(parents=True, exist_ok=True)
                route_stem_parts = ["route_compare", "full_system", build_maint_mode_tag(maint_mode), ts]
                if args.episodes != 1:
                    route_stem_parts.append(f"{ep+1:03d}")
                write_mode_comparison_outputs(
                    route_dir,
                    "_".join(route_stem_parts),
                    route_summary,
                    route_rows,
                    policy_label=f"Route compare | Full system | maintenance: {maint_mode}",
                )
        print(f"outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
