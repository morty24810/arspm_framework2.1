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
from src.agents import MaintenanceAgentDDQN, PPOSchedulerAgent, THDQNAgent
from src.pomcp import POMCPPlanner
from src.compare import compare_mode_results, write_mode_comparison_outputs
from checkpointing import load_checkpoint, load_maintenance_only
from run_experiment import (
    build_degradation_and_rul,
    build_episode_combos,
    build_episode_scenario,
    evaluate_once,
    build_policy_tag,
    build_policy_label,
    build_policy_context_label,
    build_maint_mode_tag,
    make_rng,
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
    parser.add_argument("--randomize_combos", type=int, default=1) # randomize lambda/DDT combos
    parser.add_argument("--segment_jobs", type=int, default=None) # jobs per combo segment
    parser.add_argument("--combo_plan", type=str, default="") # lam:ddt[:segments],lam:ddt[:segments]
    parser.add_argument("--maint_mode", type=str, default="POMCP", choices=["DQN", "POMCP", "OFF"]) # maintenance mode
    parser.add_argument("--compare_maint_modes", type=int, default=0, choices=[0, 1]) # compare POMCP vs DQN
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
    return MaintenanceAgentDDQN(state_dim=17, cfg=cfg, rng=random.Random(seed), device=device)


def make_sched_agent(cfg: SimConfig, seed: int, device: torch.device, state_dim: int = 14):
    scheduler_mode = str(getattr(cfg, "SCHEDULER_MODE", "THDQN")).upper()
    if scheduler_mode == "PPO":
        return PPOSchedulerAgent(state_dim=int(state_dim), cfg=cfg, rng=random.Random(seed), device=device)
    return THDQNAgent(state_dim=int(state_dim), cfg=cfg, rng=random.Random(seed), device=device)


def build_infer_route_result(metrics: Dict[str, Any], env: Any, overdue_stats: Dict[str, Any],
                             policy_label: str, maint_mode_tag: str) -> Dict[str, Any]:
    return {
        "metrics": metrics,
        "env": env,
        "overdue": overdue_stats,
        "policy_label": policy_label,
        "decision_log": list(getattr(env, "last_decision_log", [])),
        "maint_mode_tag": maint_mode_tag,
        "scheduler_mode": str(getattr(env, "last_scheduler_mode", "THDQN")).upper(),
        "scheduler_mode_tag": f"sched_{str(getattr(env, 'last_scheduler_mode', 'THDQN')).lower()}",
    }


def infer_scheduler_state_dim(ckpt_path: Path, map_location: torch.device) -> int:
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    models = ckpt.get("models", {})
    weight = models.get("high_q", {}).get("net.0.weight")
    if hasattr(weight, "shape") and len(weight.shape) >= 2:
        return int(weight.shape[1])
    weight = models.get("sched_actor", {}).get("net.0.weight")
    if hasattr(weight, "shape") and len(weight.shape) >= 2:
        return int(weight.shape[1])
    return 14


def infer_scheduler_mode(ckpt_path: Path, map_location: torch.device) -> str:
    ckpt = torch.load(str(ckpt_path), map_location=map_location)
    meta_cfg = ckpt.get("meta", {}).get("config", {}) or {}
    scheduler_mode = str(meta_cfg.get("SCHEDULER_MODE", "")).upper()
    if scheduler_mode in ("THDQN", "PPO"):
        return scheduler_mode
    models = ckpt.get("models", {})
    if "sched_actor" in models or "sched_critic" in models:
        return "PPO"
    return "THDQN"


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    cfg = SimConfig()
    validate_region_thresholds(cfg)
    cfg.SEED = int(cfg.SEED if args.seed is None else args.seed)
    if args.machines is not None:
        cfg.NUM_MACHINES = int(args.machines)
    cfg.MAINT_MODE = str(args.maint_mode).upper()
    cfg.ENFORCE_REGION_POLICY = True
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
    if compare_maint_modes:
        if cfg.MAINT_MODE == "OFF":
            raise ValueError("--compare_maint_modes=1 does not support maint_mode=OFF.")
        if not args.dqn_ckpt:
            raise ValueError("--compare_maint_modes=1 requires --dqn_ckpt.")
        dqn_ckpt_path = Path(args.dqn_ckpt)
        if not dqn_ckpt_path.exists():
            raise FileNotFoundError(f"DQN maintenance checkpoint not found: {dqn_ckpt_path}")
        maint_modes = [cfg.MAINT_MODE]
        for mode in ("POMCP", "DQN"):
            if mode not in maint_modes:
                maint_modes.append(mode)
    else:
        dqn_ckpt_path = None
        maint_modes = [cfg.MAINT_MODE]

    machine_curve_ids = list(cfg.MACHINE_CURVE_IDS)
    _, degr, rul = build_degradation_and_rul(cfg, machine_curve_ids)

    maint_agent = make_maint_agent(cfg, cfg.SEED, device)
    cfg.SCHEDULER_MODE = infer_scheduler_mode(ckpt_path, device)
    sched_state_dim = infer_scheduler_state_dim(ckpt_path, device)
    sched_agent = make_sched_agent(cfg, cfg.SEED, device, state_dim=sched_state_dim)

    allow_missing_maint = cfg.MAINT_MODE != "DQN" or compare_maint_modes
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
    maint_agent.q.eval()
    maint_agent.qt.eval()

    dqn_compare_agent = None
    if compare_maint_modes:
        dqn_compare_agent = make_maint_agent(cfg, cfg.SEED + 997, device)
        load_maintenance_only(str(dqn_ckpt_path), dqn_compare_agent, map_location=device, allow_missing_maint=False)
        dqn_compare_agent.q.eval()
        dqn_compare_agent.qt.eval()

    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    combo_plan = args.combo_plan.strip()

    for ep in range(int(args.episodes)):
        seed = int(cfg.SEED) + ep
        set_seed(seed)
        combo_rng = random.Random(seed)
        cfg_eval_base = copy.deepcopy(cfg)
        cfg_eval_base.SEED = seed
        if combo_plan:
            combos, seq = parse_combo_plan(combo_plan)
            target_segments = max(1, int(math.ceil(int(jobs_target) / cfg_eval_base.COMBO_SEGMENT_JOBS)))
            if len(seq) < target_segments:
                seq.extend([seq[-1]] * (target_segments - len(seq)))
            elif len(seq) > target_segments:
                seq = seq[:target_segments]
            cfg_eval_base.COMBO_RANDOMIZE = False
        else:
            combos, seq = build_episode_combos(cfg_eval_base, combo_rng, int(jobs_target))
        scenario = build_episode_scenario(
            cfg_eval_base,
            degr,
            jobs_target=int(jobs_target),
            scenario_rng=make_rng(seed, "infer", ep + 1, "scenario"),
            degradation_rate=float(cfg_eval_base.BASE_DEGRADATION_RATE),
            episode_combos=combos,
            episode_combo_seq=seq,
            machine_curve_ids=machine_curve_ids,
        )
        outdir = base_outdir if args.episodes == 1 else (base_outdir / f"episode_{ep+1:03d}")
        outdir.mkdir(parents=True, exist_ok=True)

        route_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        route_modes = [True, False] if int(args.compare_unrestricted) else [True]
        for enforce_region in route_modes:
            route_results.setdefault(build_policy_tag(cfg_eval_base, enforce_region), {})
            for maint_mode in maint_modes:
                cfg_eval = copy.deepcopy(cfg_eval_base)
                cfg_eval.ENFORCE_REGION_POLICY = enforce_region
                cfg_eval.MAINT_MODE = maint_mode
                policy_tag = build_policy_tag(cfg_eval, enforce_region)
                maint_mode_tag = build_maint_mode_tag(maint_mode)
                policy_label = build_policy_context_label(cfg_eval, enforce_region, maint_mode)
                route_rng = random.Random(seed)
                pomcp_ep = (
                    POMCPPlanner(num_actions=3, gamma=cfg.GAMMA, c_ucb=cfg.POMCP_UCB_C, rng=route_rng)
                    if maint_mode == "POMCP" else None
                )
                if maint_mode == "DQN":
                    maint_agent_eval = dqn_compare_agent if compare_maint_modes else maint_agent
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
                        maint_mode,
                        pomcp_ep,
                        machine_curve_ids,
                        jobs_target=int(jobs_target),
                        episode_combos=combos,
                        episode_combo_seq=seq,
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
                        belief_rng=make_rng(seed, maint_mode, "infer", ep + 1, policy_tag, "belief"),
                    )
                route_results[policy_tag][maint_mode] = build_infer_route_result(
                    metrics,
                    env,
                    overdue_stats,
                    policy_label,
                    maint_mode_tag,
                )
                summary_row = {
                    "timestamp": ts,
                    "episode": int(ep + 1),
                    "policy_tag": policy_tag,
                    "policy_label": policy_label,
                    "maint_mode": str(maint_mode),
                    "maint_mode_tag": maint_mode_tag,
                    "scheduler_mode": str(cfg_eval.SCHEDULER_MODE).upper(),
                    "scheduler_mode_tag": f"sched_{str(cfg_eval.SCHEDULER_MODE).lower()}",
                    "enforce_region_policy": int(enforce_region),
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
                }
                write_summary_files(outdir, f"summary_{artifact_stem}", summary_row)
                print(
                    f"episode {ep+1}/{args.episodes} [{policy_tag}][{maint_mode_tag}] "
                    f"tard={metrics['tard']:.2f} "
                    f"maint={metrics['maint']:.2f} "
                    f"total={metrics['total']:.2f} "
                    f"overdue_ratio={overdue_stats['ratio_ops']:.3f}"
                )

            if compare_maint_modes and "POMCP" in route_results[policy_tag] and "DQN" in route_results[policy_tag]:
                primary_mode = cfg.MAINT_MODE if cfg.MAINT_MODE in route_results[policy_tag] else "POMCP"
                if "DQN" in route_results[policy_tag] and "POMCP" in route_results[policy_tag]:
                    primary_mode = "DQN"
                    compare_mode = "POMCP"
                else:
                    compare_mode = "DQN" if primary_mode != "DQN" else "POMCP"
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
                    f"Maintenance: {primary_mode} vs {compare_mode}"
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
            for maint_mode in maint_modes:
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
                route_summary["maint_mode"] = str(maint_mode)
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
                    policy_label=f"Route compare | Full system | Maintenance: {maint_mode}",
                )
        print(f"outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
