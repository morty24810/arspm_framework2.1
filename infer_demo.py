from __future__ import annotations
import argparse
import copy
import math
import random
import time
from pathlib import Path

import torch

from config import SimConfig
from src.utils import set_seed
from src.agents import MaintenanceAgentDDQN, THDQNAgent
from src.pomcp import POMCPPlanner
from checkpointing import load_checkpoint
from run_experiment import (
    build_degradation_and_rul,
    build_episode_combos,
    evaluate_once,
    build_policy_tag,
    build_policy_label,
    write_summary_files,
    validate_region_thresholds,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference/demo from a saved checkpoint.")
    parser.add_argument("--ckpt", default="checkpoints/latest.pt", help="Checkpoint path (.pt)")
    parser.add_argument("--seed", type=int, default=0) # random seed
    parser.add_argument("--episodes", type=int, default=1) # number of episodes to run
    parser.add_argument("--jobs_target", type=int, default=50) # how many jobs to schedule in each episode
    parser.add_argument("--machines", type=int, default=None) # override number of machines
    parser.add_argument("--randomize_combos", type=int, default=1) # randomize lambda/DDT combos
    parser.add_argument("--segment_jobs", type=int, default=None) # jobs per combo segment
    parser.add_argument("--combo_plan", type=str, default="") # lam:ddt[:segments],lam:ddt[:segments]
    parser.add_argument("--maint_mode", type=str, default="DQN", choices=["DQN", "POMCP", "OFF"]) # maintenance mode
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


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    cfg = SimConfig()
    validate_region_thresholds(cfg)
    cfg.SEED = int(args.seed)
    if args.machines is not None:
        cfg.NUM_MACHINES = int(args.machines)
    cfg.MAINT_MODE = str(args.maint_mode).upper()
    cfg.ENFORCE_REGION_POLICY = True
    cfg.COMBO_RANDOMIZE = bool(int(args.randomize_combos))
    if args.segment_jobs is not None:
        cfg.COMBO_SEGMENT_JOBS = max(1, int(args.segment_jobs))

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:", device)

    machine_curve_ids = list(cfg.MACHINE_CURVE_IDS)
    _, degr, rul = build_degradation_and_rul(cfg, machine_curve_ids)

    maint_agent = MaintenanceAgentDDQN(state_dim=17, cfg=cfg, rng=random.Random(cfg.SEED), device=device)
    sched_agent = THDQNAgent(state_dim=12, cfg=cfg, rng=random.Random(cfg.SEED), device=device)
    pomcp = None

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

    sched_agent.q_high.eval()
    sched_agent.q_high_t.eval()
    sched_agent.q_low.eval()
    sched_agent.q_low_t.eval()
    maint_agent.q.eval()
    maint_agent.qt.eval()

    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    combo_plan = args.combo_plan.strip()

    for ep in range(int(args.episodes)):
        seed = int(args.seed) + ep
        set_seed(seed)
        combo_rng = random.Random(seed)
        cfg_eval_base = copy.deepcopy(cfg)
        cfg_eval_base.SEED = seed
        if combo_plan:
            combos, seq = parse_combo_plan(combo_plan)
            target_segments = max(1, int(math.ceil(int(args.jobs_target) / cfg_eval_base.COMBO_SEGMENT_JOBS)))
            if len(seq) < target_segments:
                seq.extend([seq[-1]] * (target_segments - len(seq)))
            elif len(seq) > target_segments:
                seq = seq[:target_segments]
            cfg_eval_base.COMBO_RANDOMIZE = False
        else:
            combos, seq = build_episode_combos(cfg_eval_base, combo_rng, int(args.jobs_target))
        outdir = base_outdir if args.episodes == 1 else (base_outdir / f"episode_{ep+1:03d}")
        outdir.mkdir(parents=True, exist_ok=True)
        base_prefix = "" if args.episodes == 1 else f"ep{ep+1:03d}"

        route_results = {}
        route_modes = [True, False] if int(args.compare_unrestricted) else [True]
        for enforce_region in route_modes:
            cfg_eval = copy.deepcopy(cfg_eval_base)
            cfg_eval.ENFORCE_REGION_POLICY = enforce_region
            policy_tag = build_policy_tag(cfg_eval, enforce_region)
            policy_label = build_policy_label(cfg_eval, enforce_region)
            route_rng = random.Random(seed)
            pomcp_ep = POMCPPlanner(num_actions=3, gamma=cfg.GAMMA, c_ucb=cfg.POMCP_UCB_C, rng=route_rng) if cfg.MAINT_MODE == "POMCP" else None
            route_prefix = f"{base_prefix}_{policy_tag}" if base_prefix else policy_tag
            decision_log_name = (
                f"decision_log_{ts}_{policy_tag}.csv"
                if args.episodes == 1
                else f"decision_log_{ts}_{ep+1:03d}_{policy_tag}.csv"
            )
            with torch.no_grad():
                metrics, _, overdue_stats = evaluate_once(
                    cfg_eval, route_rng, degr, rul, sched_agent, maint_agent, cfg.MAINT_MODE, pomcp_ep, machine_curve_ids,
                    jobs_target=int(args.jobs_target),
                    episode_combos=combos,
                    episode_combo_seq=seq,
                    generate_outputs=True,
                    outdir=outdir,
                    plot_prefix=route_prefix,
                    decision_log_name=decision_log_name,
                    freeze_steps=True,
                    observer_state=observer_state,
                    env_state=env_state,
                    policy_label=policy_label,
                    threshold_enforced=enforce_region,
                )
            route_results[policy_tag] = {"metrics": metrics, "overdue": overdue_stats, "policy_label": policy_label}
            summary_row = {
                "timestamp": ts,
                "episode": int(ep + 1),
                "policy_tag": policy_tag,
                "policy_label": policy_label,
                "enforce_region_policy": int(enforce_region),
                "hx": float(cfg_eval.Hx),
                "hy": float(cfg_eval.Hy),
                "seed": int(seed),
                "jobs_target": int(args.jobs_target),
                "maint_mode": str(cfg.MAINT_MODE),
                "tard": float(metrics["tard"]),
                "maint": float(metrics["maint"]),
                "total": float(metrics["total"]),
                "overdue_ratio_ops": float(overdue_stats["ratio_ops"]),
                "overdue_ratio_time": float(overdue_stats["ratio_time"]),
                "overdue_ops": float(overdue_stats["overdue_ops"]),
                "total_ops": float(overdue_stats["total_ops"]),
            }
            summary_stem = (
                f"summary_{ts}_{policy_tag}"
                if args.episodes == 1
                else f"summary_{ts}_{ep+1:03d}_{policy_tag}"
            )
            write_summary_files(outdir, summary_stem, summary_row)
            print(
                f"episode {ep+1}/{args.episodes} [{policy_tag}] "
                f"tard={metrics['tard']:.2f} "
                f"maint={metrics['maint']:.2f} "
                f"total={metrics['total']:.2f} "
                f"overdue_ratio={overdue_stats['ratio_ops']:.3f}"
            )

        constrained_tag = build_policy_tag(cfg_eval_base, True)
        unrestricted_tag = build_policy_tag(cfg_eval_base, False)
        if constrained_tag in route_results and unrestricted_tag in route_results:
            c = route_results[constrained_tag]
            u = route_results[unrestricted_tag]
            print(
                "delta(unrestricted-constrained): "
                f"tard={u['metrics']['tard'] - c['metrics']['tard']:.3f}, "
                f"maint={u['metrics']['maint'] - c['metrics']['maint']:.3f}, "
                f"total={u['metrics']['total'] - c['metrics']['total']:.3f}, "
                f"overdue_ratio={u['overdue']['ratio_ops'] - c['overdue']['ratio_ops']:.3f}"
            )
        print(f"outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
