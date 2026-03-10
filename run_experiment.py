from __future__ import annotations
import csv
import json
import math
import hashlib
import os, random
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np
import torch

from config import SimConfig
from src.utils import set_seed
from src.degradation import DegradationReplay
from src.rul_predictor import RULPredictorWrapper
from src.env import EventDrivenShopEnv, EpisodeScenario, JobTemplate, OperationTemplate
from src.agents import MaintenanceAgentDDQN, THDQNAgent
from src.compare import (
    compare_mode_results,
    write_mode_comparison_outputs,
    extract_maintenance_rows,
    summarize_action_counts,
    summarize_scheduling_strategy,
)
from src.viz import (
    plot_gantt,
    plot_rul_curves,
    plot_rule_vs_features,
    plot_training_curves,
    plot_maint_action_rates,
    plot_maint_vs_slack,
)
from src.pomcp import POMCPPlanner
from checkpointing import CheckpointManager

def build_maintenance_state(h, dh, eta, slack_pressure, local_urgency, avg_slack,
                            lambda_hat, ddt_hat, risk_t, win_e, win_l,
                            rul_mu, rul_sigma, t_now, t_last_maint,
                            pf_dn, pf_im, pf_cm):
    dt_last = t_now - t_last_maint
    return np.array([
        h, dh, eta,
        slack_pressure, local_urgency, avg_slack,
        lambda_hat, ddt_hat,
        risk_t, win_e, win_l,
        rul_mu, rul_sigma,
        dt_last,
        pf_dn, pf_im, pf_cm
    ], dtype=np.float32)

def material_cost(action: int, cfg: SimConfig) -> float:
    if action == 1:
        return cfg.MAT_COST_IM
    if action == 2:
        return cfg.MAT_COST_CM
    return cfg.MAT_COST_DN

def validate_region_thresholds(cfg: SimConfig):
    hx = float(cfg.Hx)
    hy = float(cfg.Hy)
    if not (0.0 <= hy < hx <= 1.0):
        raise ValueError(f"invalid region thresholds: expect 0 <= Hy < Hx <= 1, got Hx={hx}, Hy={hy}")

def _format_tag_number(x: float) -> str:
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")

def build_policy_tag(cfg: SimConfig, enforce_region: bool) -> str:
    if enforce_region:
        return f"region_on_hx{_format_tag_number(cfg.Hx)}_hy{_format_tag_number(cfg.Hy)}"
    return "region_off_unrestricted"

def build_policy_label(cfg: SimConfig, enforce_region: bool) -> str:
    if enforce_region:
        return f"Policy: Region constrained (Hx={cfg.Hx:.1f}, Hy={cfg.Hy:.1f})"
    return "Policy: Unrestricted (no Hx/Hy enforcement)"

def build_maint_mode_tag(maint_mode: str) -> str:
    return f"maint_{str(maint_mode).lower()}"

def build_maint_mode_label(maint_mode: str) -> str:
    return f"Maintenance: {str(maint_mode).upper()}"

def build_policy_context_label(cfg: SimConfig, enforce_region: bool, maint_mode: str) -> str:
    return f"{build_policy_label(cfg, enforce_region)} | {build_maint_mode_label(maint_mode)}"

def allowed_actions_by_region(h_obs: float, cfg: SimConfig, enforce_region: bool) -> List[int]:
    if not enforce_region:
        return [0, 1, 2]
    if h_obs > cfg.Hx:
        return [0]
    if h_obs < cfg.Hy:
        return [2]
    return [0, 1, 2]

def enforce_action_by_region(action: int, h_obs: float, cfg: SimConfig, enforce_region: bool) -> int:
    if not enforce_region:
        return int(action)
    if h_obs > cfg.Hx:
        return 0
    if h_obs < cfg.Hy:
        return 2
    return int(action)

def compute_window(risk_t: float, local_urgency: float, t_now: float, cfg: SimConfig) -> Tuple[float, float]:
    # Stage-1 windowing: risk tightens t_l, urgency delays t_e.
    t_e = t_now + cfg.WINDOW_E_BASE * local_urgency
    t_l = t_now + max(cfg.WINDOW_L_MIN, cfg.WINDOW_L_BASE * math.exp(-cfg.WINDOW_RISK_K * risk_t))
    if t_l < t_e:
        t_l = t_e
    return t_e, t_l

def maintenance_reward(action: int, dur: float, local_urgency: float, risk_t: float,
                       window_violation: bool, cfg: SimConfig) -> float:
    time_cost = cfg.W_TIME * (dur * local_urgency)
    mat_cost = cfg.W_MAT * material_cost(action, cfg)
    risk_cost = cfg.W_RISK * risk_t
    violation = cfg.W_WINDOW_VIOLATION if window_violation else 0.0
    return -(time_cost + mat_cost + risk_cost + violation)

def scheduling_reward(goal: int, tard, maint, prev_tard, prev_maint):
    # incremental rewards, 4 goals
    dtard = tard - prev_tard
    dmaint = maint - prev_maint
    if goal == 0:   # tardiness-focused
        return -dtard
    if goal == 1:   # utilization proxy via penalizing maintenance (rough)
        return -dmaint
    if goal == 2:   # balanced (tardiness-heavy)
        return -(dtard + 0.3*dmaint)
    return -(0.3*dtard + dmaint)  # balanced (maint-heavy)

def summarize_overdue_ops(timeline_ops, jobs):
    total_ops = len(timeline_ops)
    total_proc_time = sum(t1 - t0 for _, t0, t1, _, _ in timeline_ops)
    overdue_ops = []
    overdue_proc_time = 0.0
    for mid, t0, t1, jid, oid in timeline_ops:
        job = jobs.get(jid)
        if job is None:
            continue
        if t1 > job.due:
            overdue_ops.append((jid, oid, mid, t0, t1, job.due))
            overdue_proc_time += (t1 - t0)

    overdue_count = len(overdue_ops)
    ratio_ops = (overdue_count / total_ops) if total_ops else 0.0
    ratio_time = (overdue_proc_time / total_proc_time) if total_proc_time else 0.0

    print("overdue operations:")
    if not overdue_ops:
        print("  none")
    else:
        for jid, oid, mid, t0, t1, due in overdue_ops:
            print(f"  J{jid}-O{oid} M{mid} [{t0:.1f}, {t1:.1f}] due={due:.1f}")
    print(f"overdue total time: {overdue_proc_time:.1f}")
    print(f"overdue ops ratio: {overdue_count}/{total_ops} ({ratio_ops:.3f})")
    print(f"overdue time ratio: {overdue_proc_time:.1f}/{total_proc_time:.1f} ({ratio_time:.3f})")

def bin_index(x: float, bins) -> int:
    if not bins or len(bins) < 2:
        return 0
    for i in range(len(bins) - 1):
        if bins[i] <= x < bins[i + 1]:
            return i
    return len(bins) - 2

def sample_degradation_rate(ep: int, cfg: SimConfig, rng: random.Random) -> float:
    curr_steps = max(1, int(cfg.CURR_FRAC * cfg.TRAIN_EPISODES))
    if ep < curr_steps:
        frac = ep / curr_steps
        lo = cfg.DEGRAD_HIGH - (cfg.DEGRAD_HIGH - cfg.DEGRAD_LOW) * frac
        hi = cfg.DEGRAD_HIGH
    else:
        lo, hi = cfg.DEGRAD_LOW, cfg.DEGRAD_HIGH
    return float(rng.uniform(lo, hi))

def build_episode_combos(cfg: SimConfig, rng: random.Random, jobs_target: int):
    segment_jobs = max(1, int(getattr(cfg, "COMBO_SEGMENT_JOBS", 1)))
    segment_count = max(1, int(math.ceil(jobs_target / segment_jobs)))

    if not bool(getattr(cfg, "COMBO_RANDOMIZE", True)):
        combos = list(getattr(cfg, "DEFAULT_COMBOS", []))
        if not combos:
            lam_values = list(getattr(cfg, "ARRIVAL_LAM_VALUES", [100.0]))
            ddt_values = list(getattr(cfg, "DDT_VALUES", (1.0,)))
            if not ddt_values:
                ddt_values = [1.0]
            combos = [(float(lam_values[0]), float(ddt_values[0]))]
        seq = [i % len(combos) for i in range(segment_count)]
        return combos, seq

    lam_values = list(getattr(cfg, "ARRIVAL_LAM_VALUES", [100.0]))
    if not lam_values:
        lam_values = [100.0]
    ddt_values = list(getattr(cfg, "DDT_VALUES", (1.0,)))
    if not ddt_values:
        ddt_values = [1.0]

    combos = []
    seq = []
    lam_cycle = list(lam_values)
    rng.shuffle(lam_cycle)
    for seg in range(segment_count):
        if not lam_cycle:
            lam_cycle = list(lam_values)
            rng.shuffle(lam_cycle)
        lam = float(lam_cycle.pop())
        ddt = float(rng.choice(ddt_values))
        combos.append((lam, ddt))
        seq.append(seg)
    return combos, seq


@dataclass
class ScenarioBank:
    train_scenarios: List[EpisodeScenario]
    periodic_eval_scenarios: Dict[int, EpisodeScenario]
    final_eval_scenario: EpisodeScenario


def derive_seed(base_seed: int, *parts: Any) -> int:
    payload = "::".join([str(base_seed), *[str(p) for p in parts]]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def make_rng(base_seed: int, *parts: Any) -> random.Random:
    return random.Random(derive_seed(base_seed, *parts))


def compute_train_jobs_target(cfg: SimConfig) -> int:
    avg_ops = 0.5 * (cfg.OPS_PER_JOB_MIN + cfg.OPS_PER_JOB_MAX)
    target_decisions = cfg.TARGET_MAINT_DECISIONS_PER_MACHINE * cfg.NUM_MACHINES
    auto_jobs = int(math.ceil(target_decisions / max(avg_ops, 1e-6)))
    return max(cfg.TRAIN_JOBS_TARGET, auto_jobs)


def compute_machine_time_scale(cfg: SimConfig, degr: DegradationReplay, machine_curve_ids: List[int]) -> Dict[int, float]:
    lifespans = [
        degr.lifespan(int(machine_curve_ids[i % len(machine_curve_ids)]))
        for i in range(cfg.NUM_MACHINES)
    ]
    avg_life = float(np.mean(lifespans)) if lifespans else 1.0
    machine_time_scale: Dict[int, float] = {}
    for i in range(cfg.NUM_MACHINES):
        scale = lifespans[i] / max(avg_life, 1e-6)
        scale = min(max(scale, cfg.MACHINE_PT_SCALE_MIN), cfg.MACHINE_PT_SCALE_MAX)
        machine_time_scale[i] = float(scale)
    return machine_time_scale


def build_episode_scenario(
    cfg: SimConfig,
    degr: DegradationReplay,
    jobs_target: int,
    scenario_rng: random.Random,
    degradation_rate: float,
    episode_combos: Optional[list[tuple[float, float]]] = None,
    episode_combo_seq: Optional[list[int]] = None,
    machine_curve_ids: Optional[List[int]] = None,
) -> EpisodeScenario:
    machine_curve_ids = list(machine_curve_ids or list(cfg.MACHINE_CURVE_IDS))
    if episode_combos is None or episode_combo_seq is None:
        combos, seq = build_episode_combos(cfg, scenario_rng, jobs_target)
    else:
        combos = [(float(lam), float(ddt)) for lam, ddt in episode_combos]
        seq = [int(x) for x in episode_combo_seq]
    segment_jobs = max(1, int(getattr(cfg, "COMBO_SEGMENT_JOBS", 1)))
    machine_time_scale = compute_machine_time_scale(cfg, degr, machine_curve_ids)

    def combo_for_job(job_id: int) -> Tuple[float, float]:
        seg = min(job_id // segment_jobs, len(seq) - 1)
        level_idx = seq[seg]
        lam, ddt = combos[level_idx]
        return float(lam), float(ddt)

    arrival_times: List[float] = []
    job_templates: List[JobTemplate] = []
    t_now = 0.0
    for job_id in range(jobs_target):
        lam, ddt = combo_for_job(job_id)
        if lam <= 0.0 or not math.isfinite(t_now):
            t_now = math.inf
        else:
            u = scenario_rng.random()
            t_now = t_now + (-math.log(max(u, 1e-12)) * lam)
        arrival = float(t_now)
        arrival_times.append(arrival)

        num_ops = scenario_rng.randint(cfg.OPS_PER_JOB_MIN, cfg.OPS_PER_JOB_MAX)
        ops: List[OperationTemplate] = []
        for _ in range(num_ops):
            k = scenario_rng.randint(cfg.FEASIBLE_M_MIN, min(cfg.FEASIBLE_M_MAX, cfg.NUM_MACHINES))
            feasible = scenario_rng.sample(list(range(cfg.NUM_MACHINES)), k=k)
            proc = {
                int(m): float(scenario_rng.uniform(cfg.PT_MIN, cfg.PT_MAX) * machine_time_scale.get(m, 1.0))
                for m in feasible
            }
            ops.append(OperationTemplate(feasible_machines=list(feasible), proc_times=proc))
        avg_sum = 0.0
        for op in ops:
            avg_sum += float(np.mean(list(op.proc_times.values())))
        due = arrival + avg_sum * ddt if math.isfinite(arrival) else math.inf
        urgency = float(scenario_rng.uniform(0.8, 1.2))
        job_templates.append(
            JobTemplate(arrival=arrival, due=float(due), urgency=urgency, ops=ops)
        )

    return EpisodeScenario(
        jobs_target=int(jobs_target),
        machine_curve_ids=machine_curve_ids,
        combos=combos,
        combo_seq=seq,
        degradation_rate=float(degradation_rate),
        arrival_times=arrival_times,
        job_templates=job_templates,
    )


def build_scenario_bank(base_seed: int, cfg: SimConfig, degr: DegradationReplay,
                        machine_curve_ids: Optional[List[int]] = None) -> ScenarioBank:
    if str(getattr(cfg, "SCENARIO_LOCK_SCOPE", "full")).lower() != "full":
        raise NotImplementedError("Only SCENARIO_LOCK_SCOPE='full' is implemented.")
    train_jobs_target = compute_train_jobs_target(cfg)
    train_scenarios: List[EpisodeScenario] = []
    periodic_eval_scenarios: Dict[int, EpisodeScenario] = {}
    machine_curve_ids = list(machine_curve_ids or list(cfg.MACHINE_CURVE_IDS))
    base_degrad = float(cfg.BASE_DEGRADATION_RATE)

    for ep in range(cfg.TRAIN_EPISODES):
        ep_num = ep + 1
        scenario_rng = make_rng(base_seed, "train", ep_num, "scenario")
        degrad_rng = make_rng(base_seed, "train", ep_num, "degradation")
        degrad_rate = sample_degradation_rate(ep, cfg, degrad_rng)
        scenario = build_episode_scenario(
            cfg,
            degr,
            jobs_target=train_jobs_target,
            scenario_rng=scenario_rng,
            degradation_rate=degrad_rate,
            machine_curve_ids=machine_curve_ids,
        )
        train_scenarios.append(scenario)
        if cfg.EVAL_EVERY > 0 and ep_num % cfg.EVAL_EVERY == 0:
            eval_jobs = cfg.EVAL_JOBS_TARGET * 2
            eval_rng = make_rng(base_seed, "periodic_eval", ep_num, "scenario")
            periodic_eval_scenarios[ep_num] = build_episode_scenario(
                cfg,
                degr,
                jobs_target=eval_jobs,
                scenario_rng=eval_rng,
                degradation_rate=base_degrad,
                machine_curve_ids=machine_curve_ids,
            )

    final_eval_scenario = build_episode_scenario(
        cfg,
        degr,
        jobs_target=cfg.EVAL_JOBS_TARGET * 2,
        scenario_rng=make_rng(base_seed, "final_eval", "scenario"),
        degradation_rate=base_degrad,
        machine_curve_ids=machine_curve_ids,
    )
    return ScenarioBank(
        train_scenarios=train_scenarios,
        periodic_eval_scenarios=periodic_eval_scenarios,
        final_eval_scenario=final_eval_scenario,
    )

def init_belief(h_obs: float, slack_pressure: float, cfg: SimConfig, rng: random.Random,
                baseline_rul: float = 1.0, region_b_elapsed: float = 0.0):
    particles = []
    baseline_rul = max(0.0, min(1.0, float(baseline_rul)))
    region_b_elapsed = max(0.0, float(region_b_elapsed))
    for _ in range(cfg.POMCP_PARTICLES):
        h = max(0.0, min(1.0, h_obs + rng.normalvariate(0.0, cfg.POMCP_OBS_NOISE)))
        stress = max(0.0, min(1.0, slack_pressure + rng.normalvariate(0.0, 0.1)))
        particles.append({
            "h_true": h,
            "stress": stress,
            "baseline_rul": baseline_rul,
            "region_b_elapsed": region_b_elapsed,
        })
    return particles

def update_belief(particles, h_obs: float, cfg: SimConfig, rng: random.Random):
    if not particles:
        return particles
    sigma = max(cfg.POMCP_OBS_NOISE, 1e-3)
    weights = []
    for p in particles:
        err = h_obs - p["h_true"]
        w = math.exp(-0.5 * (err / sigma) ** 2)
        weights.append(w)
    total = sum(weights)
    if total <= 0:
        return particles
    weights = [w / total for w in weights]
    resampled = []
    for _ in range(len(particles)):
        r = rng.random()
        acc = 0.0
        idx = 0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                idx = i
                break
        resampled.append(dict(particles[idx]))
    return resampled

def refresh_belief(pomcp_beliefs, mid: int, h_obs: float, slack_pressure: float,
                   cfg: SimConfig, rng: random.Random,
                   baseline_rul: Optional[float] = None,
                   region_b_elapsed: Optional[float] = None):
    if pomcp_beliefs is None:
        return []
    belief = pomcp_beliefs.get(mid, [])
    if not belief:
        belief = init_belief(
            h_obs,
            slack_pressure,
            cfg,
            rng,
            baseline_rul=1.0 if baseline_rul is None else baseline_rul,
            region_b_elapsed=0.0 if region_b_elapsed is None else region_b_elapsed,
        )
    else:
        belief = update_belief(belief, h_obs, cfg, rng)
    for p in belief:
        p["stress"] = float(slack_pressure)
        if baseline_rul is not None:
            p["baseline_rul"] = max(0.0, min(1.0, float(baseline_rul)))
        if region_b_elapsed is not None:
            p["region_b_elapsed"] = max(0.0, float(region_b_elapsed))
        elif h_obs > cfg.Hx:
            p["region_b_elapsed"] = 0.0
    pomcp_beliefs[mid] = belief
    return belief

def get_sensor_features(cfg: SimConfig) -> list[str]:
    import pandas as pd
    df0 = pd.read_csv(cfg.TRAIN_CSV)
    features = ["Differential_pressure"]
    if "Flow_rate" in df0.columns:
        features.append("Flow_rate")
    if "Dust_feed" in df0.columns:
        features.append("Dust_feed")
    return features

def build_degradation_and_rul(cfg: SimConfig, machine_curve_ids: list[int]):
    features = get_sensor_features(cfg)
    print("sensor features:", features)
    degr = DegradationReplay(cfg.TRAIN_CSV, selected_data_nos=machine_curve_ids, features=features, noise_std=cfg.DEGRAD_NOISE_STD)
    rul = RULPredictorWrapper(cfg.RUL_ARTIFACT_DIR, window=cfg.RUL_WINDOW, features=features)
    print("RUL predictor uses GRU artifact:", rul.use_artifact)
    if list(degr.features) != list(rul.features):
        raise ValueError(
            "GRU artifact features do not match replay features. "
            f"replay={degr.features} artifact={rul.features}"
        )
    return features, degr, rul

def compute_overdue_stats(timeline_ops, jobs) -> Dict[str, float]:
    total_ops = len(timeline_ops)
    total_proc_time = sum(t1 - t0 for _, t0, t1, _, _ in timeline_ops)
    overdue_count = 0
    overdue_proc_time = 0.0
    for _, t0, t1, jid, oid in timeline_ops:
        job = jobs.get(jid)
        if job is None:
            continue
        if t1 > job.due:
            overdue_count += 1
            overdue_proc_time += (t1 - t0)
    ratio_ops = (overdue_count / total_ops) if total_ops else 0.0
    ratio_time = (overdue_proc_time / total_proc_time) if total_proc_time else 0.0
    return {
        "overdue_ops": float(overdue_count),
        "total_ops": float(total_ops),
        "ratio_ops": float(ratio_ops),
        "ratio_time": float(ratio_time),
    }

def write_summary_files(outdir: Path, stem: str, summary: Dict[str, Any]):
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"{stem}.json"
    csv_path = outdir / f"{stem}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(summary.keys()))
        writer.writerow([summary[k] for k in summary.keys()])

def evaluate_once(cfg: SimConfig, rng: random.Random, degr: DegradationReplay, rul: RULPredictorWrapper,
                  sched_agent: THDQNAgent, maint_agent: Optional[MaintenanceAgentDDQN],
                  maint_mode: str, pomcp: Optional[POMCPPlanner], machine_curve_ids: list[int],
                  jobs_target: Optional[int] = None,
                  episode_combos: Optional[list[tuple[float, float]]] = None,
                  episode_combo_seq: Optional[list[int]] = None,
                  generate_outputs: bool = True,
                  outdir: Optional[Path] = None,
                  plot_prefix: str = "",
                  decision_log_name: str = "decision_log.csv",
                  freeze_steps: bool = False,
                  observer_state: Optional[Dict[str, Any]] = None,
                  env_state: Optional[Dict[str, Any]] = None,
                  policy_label: Optional[str] = None,
                  threshold_enforced: Optional[bool] = None,
                  scenario: Optional[EpisodeScenario] = None,
                  env_rng: Optional[random.Random] = None,
                  rul_obs_rng: Optional[random.Random] = None,
                  belief_rng: Optional[random.Random] = None):
    step_state = None
    if freeze_steps:
        step_state = {
            "sched_steps": int(getattr(sched_agent, "steps", 0)),
            "maint_steps": int(getattr(maint_agent, "steps", 0)) if maint_agent is not None else None,
        }

    try:
        enforce_region = bool(getattr(cfg, "ENFORCE_REGION_POLICY", True) if threshold_enforced is None else threshold_enforced)
        env_rng = env_rng if env_rng is not None else rng
        rul_obs_rng = rul_obs_rng if rul_obs_rng is not None else env_rng
        belief_rng = belief_rng if belief_rng is not None else rng
        env = EventDrivenShopEnv(cfg, env_rng, degr, rul, breakdown_rng=env_rng, obs_rng=rul_obs_rng)
        if episode_combos is not None:
            env.episode_combos = list(episode_combos)
        if episode_combo_seq is not None:
            env.episode_combo_seq = list(episode_combo_seq)
        if scenario is not None:
            env.cfg.JOBS_TARGET = int(scenario.jobs_target)
        elif jobs_target is None:
            env.cfg.JOBS_TARGET = cfg.EVAL_JOBS_TARGET * 2
        else:
            env.cfg.JOBS_TARGET = int(jobs_target)
        env.reset(machine_curve_ids=machine_curve_ids, scenario=scenario)
        if observer_state:
            env.observer.reset()
            env.observer.arrivals.extend(observer_state.get("arrivals", []))
            env.observer.op_times.extend(observer_state.get("op_times", []))
        if env_state:
            if "slack_scale" in env_state:
                env.slack_scale = env_state["slack_scale"]
            if "last_slack_pressure" in env_state:
                env.last_slack_pressure = env_state["last_slack_pressure"]

        prev_tard, prev_maint = 0.0, 0.0
        last_h = {m.mid: None for m in env.machines}
        decision_log = [] if generate_outputs else None
        p_fail_plot = [] if generate_outputs else None
        maint_scatter = [] if generate_outputs else None
        pending_maint = {}
        pomcp_beliefs = {m.mid: [] for m in env.machines}
        maint_seq_global = 0
        maint_seq_machine = {m.mid: 0 for m in env.machines}

        def append_decision_log(row: Dict[str, Any]):
            nonlocal maint_seq_global
            if decision_log is None:
                return
            rec = dict(row)
            rec.setdefault("maint_mode", str(maint_mode).upper())
            if rec.get("event") == "maintenance":
                mid_val = rec.get("mid")
                maint_seq_global += 1
                rec["maint_seq_global"] = maint_seq_global
                if mid_val is not None:
                    maint_seq_machine[mid_val] = maint_seq_machine.get(mid_val, 0) + 1
                    rec["maint_seq_machine"] = maint_seq_machine[mid_val]
                else:
                    rec["maint_seq_machine"] = None
            else:
                rec.setdefault("maint_seq_global", None)
                rec.setdefault("maint_seq_machine", None)
            decision_log.append(rec)

        while not env.done():
            done_flag, etype, payload = env.step_until_decision()
            if done_flag:
                break

            if etype == "MACHINE_IDLE":
                mid = payload["mid"]
                if not payload.get("from_maint", False):
                    if mid in pending_maint:
                        rec = pending_maint[mid]
                        m = env.machines[mid]
                        h_now = env.maintenance_decision_point(mid)
                        avg_slack, _, _, slack_pressure = env.compute_slack_stats()
                        local_urgency = env.get_local_urgency(mid)
                        arrivals, lambda_hat, _, _, ddt_hat, rush = env.get_obs_estimates(avg_slack, slack_pressure)
                        risk_now = env.failure_prob(h_now)
                        action_now = enforce_action_by_region(rec["action"], h_now, cfg, enforce_region)
                        execute_now = env.time >= rec["t_e"] or (enforce_region and h_now < cfg.Hy)
                        if action_now == 0:
                            if p_fail_plot is not None:
                                p_fail_plot.append((env.time, risk_now))
                            if maint_scatter is not None:
                                maint_scatter.append((slack_pressure, 0))
                            if decision_log is not None:
                                append_decision_log({
                                    "time": env.time,
                                    "event": "maintenance",
                                    "mid": m.mid,
                                    "state": rec["state"].tolist(),
                                    "action": 0,
                                    "kind": "DN",
                                    "duration": 0.0,
                                    "h": float(h_now),
                                    "dh": float(0.0),
                                    "eta": float(0.0),
                                    "slack_pressure": float(slack_pressure),
                                    "local_urgency": float(local_urgency),
                                    "im_count": 0,
                                    "im_damage": 0.0,
                                    "risk_h": float(risk_now),
                                    "risk_trend": 0.0,
                                    "im_longterm_penalty": 0.0,
                                    "opportunity_cost": 0.0,
                                    "p_fail": float(risk_now),
                                    "expected_fail_cost": 0.0,
                                    "downtime_cost": 0.0,
                                    "delta_t_since_last_maint": float(env.time - m.last_maint_end),
                                    "lambda_hat": float(lambda_hat),
                                    "ddt_hat": float(ddt_hat),
                                    "breakdown_flag": False,
                                    "breakdown_cost": 0.0,
                                    "t_e": float(rec["t_e"]),
                                    "t_l": float(rec["t_l"]),
                                    "window_violation": False,
                                    "risk_t": float(risk_now),
                                    "mat_cost": float(material_cost(0, cfg)),
                                    "scrap_part_cost": 0.0,
                                })
                            last_h[mid] = h_now
                            del pending_maint[mid]
                        elif execute_now:
                            window_violation = env.time > rec["t_l"]
                            dur, kind, post_rul = env.apply_maintenance(mid, action_now, h=h_now)
                            if p_fail_plot is not None:
                                p_fail_plot.append((env.time, risk_now))
                            h2 = post_rul if action_now != 0 else h_now
                            last_h[mid] = h2
                            if maint_scatter is not None:
                                maint_scatter.append((slack_pressure, action_now))
                            if decision_log is not None:
                                append_decision_log({
                                    "time": env.time,
                                    "event": "maintenance",
                                    "mid": m.mid,
                                    "state": rec["state"].tolist(),
                                    "action": int(action_now),
                                    "kind": kind,
                                    "duration": float(dur),
                                    "h": float(h_now),
                                    "dh": float(0.0),
                                    "eta": float(0.0),
                                    "slack_pressure": float(slack_pressure),
                                    "local_urgency": float(local_urgency),
                                    "im_count": 0,
                                    "im_damage": 0.0,
                                    "risk_h": float(risk_now),
                                    "risk_trend": 0.0,
                                    "im_longterm_penalty": 0.0,
                                    "opportunity_cost": 0.0,
                                    "p_fail": float(risk_now),
                                    "expected_fail_cost": 0.0,
                                    "downtime_cost": 0.0,
                                    "delta_t_since_last_maint": float(env.time - m.last_maint_end),
                                    "lambda_hat": float(lambda_hat),
                                    "ddt_hat": float(ddt_hat),
                                    "breakdown_flag": False,
                                    "breakdown_cost": 0.0,
                                    "t_e": float(rec["t_e"]),
                                    "t_l": float(rec["t_l"]),
                                    "window_violation": bool(window_violation),
                                    "risk_t": float(risk_now),
                                    "mat_cost": float(material_cost(action_now, cfg)),
                                    "scrap_part_cost": 0.0,
                                })
                            del pending_maint[mid]
                    elif mid not in pending_maint:
                        m = env.machines[mid]
                        h = env.maintenance_decision_point(mid)
                        avg_slack, _, _, slack_pressure = env.compute_slack_stats()
                        local_urgency = env.get_local_urgency(mid)
                        arrivals, lambda_hat, _, _, ddt_hat, rush = env.get_obs_estimates(avg_slack, slack_pressure)
                        prev_h = last_h.get(mid)
                        dh = 0.0 if prev_h is None else (h - prev_h)
                        if dh < -cfg.ETA_EPS:
                            eta = max(0.0, (h - cfg.Hy) / (-dh))
                        else:
                            eta = cfg.ETA_CAP
                        risk_t = env.failure_prob(h)
                        t_e, t_l = compute_window(risk_t, local_urgency, env.time, cfg)
                        win_e = max(t_e - env.time, 0.0)
                        win_l = max(t_l - env.time, 0.0)
                        rul_mu = h
                        rul_sigma = max(cfg.RUL_OBS_NOISE, 1e-6)
                        belief = refresh_belief(
                            pomcp_beliefs,
                            mid,
                            h,
                            slack_pressure,
                            cfg,
                            belief_rng,
                            baseline_rul=env.machines[mid].maint_rul_baseline,
                            region_b_elapsed=env.get_region_b_elapsed(mid, h),
                        )
                        pf_dn = estimate_p_fail_horizon(
                            env, mid, belief, slack_pressure, belief_rng, cfg,
                            first_action=0, num_sims=cfg.PFAIL_NUM_SIMS, horizon=cfg.PFAIL_HORIZON
                        )
                        pf_im = estimate_p_fail_horizon(
                            env, mid, belief, slack_pressure, belief_rng, cfg,
                            first_action=1, num_sims=cfg.PFAIL_NUM_SIMS, horizon=cfg.PFAIL_HORIZON
                        )
                        pf_cm = estimate_p_fail_horizon(
                            env, mid, belief, slack_pressure, belief_rng, cfg,
                            first_action=2, num_sims=cfg.PFAIL_NUM_SIMS, horizon=cfg.PFAIL_HORIZON
                        )
                        pf_dn = max(0.0, min(1.0, pf_dn))
                        pf_im = max(0.0, min(1.0, pf_im))
                        pf_cm = max(0.0, min(1.0, pf_cm))
                        s = build_maintenance_state(h, dh, eta, slack_pressure, local_urgency, avg_slack,
                                                    lambda_hat, ddt_hat, risk_t, win_e, win_l,
                                                    rul_mu, rul_sigma, env.time, m.last_maint_end,
                                                    pf_dn, pf_im, pf_cm)
                        a = select_maintenance_action(
                            maint_mode, maint_agent, pomcp, pomcp_beliefs, env, mid, h, s,
                            slack_pressure, local_urgency, cfg, belief_rng, explore=False
                        )
                        if a == 0:
                            if p_fail_plot is not None:
                                p_fail_plot.append((env.time, risk_t))
                            if maint_scatter is not None:
                                maint_scatter.append((slack_pressure, a))
                            if decision_log is not None:
                                append_decision_log({
                                    "time": env.time,
                                    "event": "maintenance",
                                    "mid": m.mid,
                                    "state": s.tolist(),
                                    "action": int(a),
                                    "kind": "DN",
                                    "duration": 0.0,
                                    "h": float(h),
                                    "dh": float(dh),
                                    "eta": float(eta),
                                    "slack_pressure": float(slack_pressure),
                                    "local_urgency": float(local_urgency),
                                    "im_count": 0,
                                    "im_damage": 0.0,
                                    "risk_h": float(risk_t),
                                    "risk_trend": 0.0,
                                    "im_longterm_penalty": 0.0,
                                    "opportunity_cost": 0.0,
                                    "p_fail": float(risk_t),
                                    "expected_fail_cost": 0.0,
                                    "downtime_cost": 0.0,
                                    "delta_t_since_last_maint": float(env.time - m.last_maint_end),
                                    "lambda_hat": float(lambda_hat),
                                    "ddt_hat": float(ddt_hat),
                                    "breakdown_flag": False,
                                    "breakdown_cost": 0.0,
                                    "t_e": float(t_e),
                                    "t_l": float(t_l),
                                    "window_violation": False,
                                    "risk_t": float(risk_t),
                                    "mat_cost": float(material_cost(a, cfg)),
                                    "scrap_part_cost": 0.0,
                                })
                            last_h[mid] = h
                        else:
                            pending_maint[mid] = {
                                "state": s,
                                "action": a,
                                "risk_t": risk_t,
                                "t_e": t_e,
                                "t_l": t_l,
                            }
                            if env.time >= t_e or (enforce_region and h < cfg.Hy):
                                rec = pending_maint[mid]
                                window_violation = env.time > rec["t_l"]
                                action_now = enforce_action_by_region(rec["action"], h, cfg, enforce_region)
                                dur, kind, post_rul = env.apply_maintenance(mid, action_now, h=h)
                                if p_fail_plot is not None:
                                    p_fail_plot.append((env.time, risk_t))
                                h2 = post_rul if action_now != 0 else h
                                last_h[mid] = h2
                                if maint_scatter is not None:
                                    maint_scatter.append((slack_pressure, action_now))
                                if decision_log is not None:
                                    append_decision_log({
                                        "time": env.time,
                                        "event": "maintenance",
                                        "mid": m.mid,
                                        "state": rec["state"].tolist(),
                                        "action": int(action_now),
                                        "kind": kind,
                                        "duration": float(dur),
                                        "h": float(h),
                                        "dh": float(dh),
                                        "eta": float(eta),
                                        "slack_pressure": float(slack_pressure),
                                        "local_urgency": float(local_urgency),
                                        "im_count": 0,
                                        "im_damage": 0.0,
                                        "risk_h": float(risk_t),
                                        "risk_trend": 0.0,
                                        "im_longterm_penalty": 0.0,
                                        "opportunity_cost": 0.0,
                                        "p_fail": float(risk_t),
                                        "expected_fail_cost": 0.0,
                                        "downtime_cost": 0.0,
                                        "delta_t_since_last_maint": float(env.time - m.last_maint_end),
                                        "lambda_hat": float(lambda_hat),
                                        "ddt_hat": float(ddt_hat),
                                        "breakdown_flag": False,
                                        "breakdown_cost": 0.0,
                                        "t_e": float(rec["t_e"]),
                                        "t_l": float(rec["t_l"]),
                                        "window_violation": bool(window_violation),
                                        "risk_t": float(risk_t),
                                        "mat_cost": float(material_cost(action_now, cfg)),
                                        "scrap_part_cost": 0.0,
                                    })
                                del pending_maint[mid]

            while env.has_idle_machine() and env.has_ready_ops():
                S = env.get_global_features()
                avg_slack, _, _, slack_pressure = env.compute_slack_stats()
                _, lambda_hat, _, _, ddt_hat, _ = env.get_obs_estimates(avg_slack, slack_pressure)
                g, rule = sched_agent.act(S, explore=False)
                env.rule_log.append((env.time, S.copy(), int(g), int(rule)))
                dispatched = env.dispatch(rule)
                if decision_log is not None:
                    op_info = None
                    overdue = None
                    job_due = None
                    breakdown = env.last_breakdown
                    breakdown_flag = bool(breakdown)
                    breakdown_cost = float(breakdown["cost"]) if breakdown else 0.0
                    if dispatched and env.timeline_ops:
                        mid, t0, t1, jid, oid = env.timeline_ops[-1]
                        job = env.jobs.get(jid)
                        job_due = float(job.due) if job is not None else None
                        overdue = (t1 > job_due) if job_due is not None else None
                        op_info = {"mid": mid, "t0": t0, "t1": t1, "jid": jid, "oid": oid}

                    append_decision_log({
                        "time": env.time,
                        "event": "scheduling",
                        "state": S.tolist(),
                        "goal": int(g),
                        "rule": int(rule),
                        "dispatched": bool(dispatched),
                        "op": op_info,
                        "job_due": job_due,
                        "overdue": overdue,
                        "lambda_hat": float(lambda_hat),
                        "ddt_hat": float(ddt_hat),
                        "local_urgency": None,
                        "breakdown_flag": breakdown_flag,
                        "breakdown_cost": breakdown_cost,
                        "t_e": None,
                        "t_l": None,
                        "window_violation": None,
                        "risk_t": None,
                        "mat_cost": None,
                        "scrap_part_cost": breakdown_cost,
                    })

        tard, maint = env.compute_costs()
        metrics = {"tard": float(tard), "maint": float(maint), "total": float(tard + maint)}
        overdue_stats = compute_overdue_stats(env.timeline_ops, env.jobs)
        env.last_decision_log = list(decision_log or [])
        env.last_policy_label = policy_label
        env.last_maint_mode = str(maint_mode).upper()

        if generate_outputs:
            outdir = Path(outdir or "outputs")
            outdir.mkdir(parents=True, exist_ok=True)
            suffix = f"_{plot_prefix}" if plot_prefix else ""
            policy_text = policy_label or build_policy_context_label(cfg, enforce_region, maint_mode)
            t_end = 0.0
            for _, t0, t1, _, _ in env.timeline_ops:
                t_end = max(t_end, t1)
            for _, t0, t1, _ in env.timeline_maint:
                t_end = max(t_end, t1)
            schedule = env.get_combo_time_log(t_end)
            plot_gantt(
                env.timeline_ops,
                env.timeline_maint,
                env.jobs,
                str(outdir / f"gantt{suffix}.png"),
                schedule=schedule,
                policy_label=policy_text,
            )
            plot_rul_curves(
                env.rul_log,
                env.timeline_maint,
                cfg.Hx,
                cfg.Hy,
                str(outdir / f"rul_curves{suffix}.png"),
                p_fail_log=p_fail_plot or [],
                policy_label=policy_text,
                threshold_enforced=enforce_region,
            )
            plot_rule_vs_features(env.rule_log, str(outdir / f"rule_vs_features{suffix}.png"), policy_label=policy_text)
            plot_maint_vs_slack(maint_scatter or [], str(outdir / f"maint_vs_slack{suffix}.png"), policy_label=policy_text)

            if decision_log is not None:
                csv_path = outdir / decision_log_name
                with csv_path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "time", "event", "maint_mode", "maint_seq_global", "maint_seq_machine", "mid", "state", "action", "kind", "duration", "h", "dh", "eta", "slack_pressure",
                        "im_count", "im_damage", "risk_h", "risk_trend", "im_longterm_penalty", "opportunity_cost",
                        "p_fail", "expected_fail_cost", "downtime_cost", "delta_t_since_last_maint",
                        "lambda_hat", "ddt_hat",
                        "goal", "rule", "dispatched", "op", "job_due", "overdue",
                        "local_urgency", "breakdown_flag", "breakdown_cost",
                        "t_e", "t_l", "window_violation", "risk_t", "mat_cost", "scrap_part_cost"
                    ])
                    for row in decision_log:
                        writer.writerow([
                            row.get("time"), row.get("event"), row.get("maint_mode"), row.get("maint_seq_global"), row.get("maint_seq_machine"), row.get("mid"),
                            json.dumps(row.get("state"), separators=(",", ":"), ensure_ascii=True) if row.get("state") is not None else "",
                            row.get("action"), row.get("kind"), row.get("duration"),
                            row.get("h"), row.get("dh"), row.get("eta"), row.get("slack_pressure"),
                            row.get("im_count"), row.get("im_damage"), row.get("risk_h"), row.get("risk_trend"),
                            row.get("im_longterm_penalty"), row.get("opportunity_cost"),
                            row.get("p_fail"), row.get("expected_fail_cost"),
                            row.get("downtime_cost"), row.get("delta_t_since_last_maint"),
                            row.get("lambda_hat"), row.get("ddt_hat"),
                            row.get("goal"), row.get("rule"), row.get("dispatched"),
                            json.dumps(row.get("op"), separators=(",", ":"), ensure_ascii=True) if row.get("op") is not None else "",
                            row.get("job_due"), row.get("overdue"),
                            row.get("local_urgency"), row.get("breakdown_flag"), row.get("breakdown_cost"),
                            row.get("t_e"), row.get("t_l"), row.get("window_violation"),
                            row.get("risk_t"), row.get("mat_cost"), row.get("scrap_part_cost"),
                        ])

            summarize_overdue_ops(env.timeline_ops, env.jobs)

        return metrics, env, overdue_stats
    finally:
        if freeze_steps and step_state is not None:
            sched_agent.steps = step_state["sched_steps"]
            if maint_agent is not None and step_state["maint_steps"] is not None:
                maint_agent.steps = step_state["maint_steps"]

def estimate_p_fail_horizon(env, mid, belief, slack_pressure, rng, cfg,
                            first_action: int = 0, num_sims: int = 64, horizon: int = 6) -> float:
    """
    估計：如果現在先做 first_action，接著用 DN rollout，
    在 horizon 步內的累積 failure 機率（用 1 - Π(1-p_fail) 近似）。
    """
    if not belief:
        return 0.0

    num_sims = int(num_sims)
    horizon = int(horizon)
    pf_list = []

    for _ in range(num_sims):
        p = rng.choice(belief)
        s = {
            "h_true": float(p.get("h_true", 1.0)),
            "stress": float(slack_pressure),
            "baseline_rul": float(p.get("baseline_rul", 1.0)),
            "region_b_elapsed": float(p.get("region_b_elapsed", 0.0)),
        }

        # 累積失敗機率：1 - ∏(1-p_fail_t)
        surv = 1.0
        a0 = int(first_action)

        for t in range(horizon):
            a = a0 if t == 0 else 0  # 第一步可測 DN/IM/CM，之後先用 DN rollout
            h_state = float(s.get("h_true", 1.0))
            a = enforce_action_by_region(a, h_state, cfg, bool(getattr(cfg, "ENFORCE_REGION_POLICY", True)))

            s, obs, info = env.generative_step(mid, s, a, slack_pressure, rng)
            p_fail_t = float(info.get("p_fail", env.failure_prob(float(s.get("h_true", 1.0)))))
            p_fail_t = max(0.0, min(1.0, p_fail_t))
            surv *= (1.0 - p_fail_t)

        pf_list.append(1.0 - surv)

    return float(sum(pf_list) / max(len(pf_list), 1))

def select_maintenance_action(mode: str, maint_agent, pomcp, pomcp_beliefs, env, mid: int,
                              h_obs: float, state: np.ndarray, slack_pressure: float,
                              local_urgency: float, cfg: SimConfig, rng: random.Random,
                              explore: bool) -> int:
    mode = mode.upper()
    enforce_region = bool(getattr(cfg, "ENFORCE_REGION_POLICY", True))
    allowed_actions = allowed_actions_by_region(h_obs, cfg, enforce_region)
    if mode == "OFF":
        return int(allowed_actions[0]) if len(allowed_actions) == 1 else 0
    if mode == "DQN":
        if maint_agent is None:
            return int(allowed_actions[0]) if len(allowed_actions) == 1 else 0
        action = int(maint_agent.act(state, explore=explore, allowed_actions=allowed_actions))
        return int(enforce_action_by_region(action, h_obs, cfg, enforce_region))

    if mode == "POMCP":
        if pomcp is None or pomcp_beliefs is None:
            raise NotImplementedError("MAINT_MODE=POMCP requires POMCP planner and belief tracking.")
        belief = pomcp_beliefs.get(mid, [])
        baseline_rul = float(env.machines[mid].maint_rul_baseline)
        region_b_elapsed = float(env.get_region_b_elapsed(mid, h_obs))
        if not belief:
            belief = init_belief(
                h_obs,
                slack_pressure,
                cfg,
                rng,
                baseline_rul=baseline_rul,
                region_b_elapsed=region_b_elapsed,
            )
        for p in belief:
            p["stress"] = float(slack_pressure)
            p["baseline_rul"] = baseline_rul
            p["region_b_elapsed"] = region_b_elapsed
        pomcp_beliefs[mid] = belief

        def model(state_p, action):
            h_state = float(state_p.get("h_true", 1.0))
            a = enforce_action_by_region(int(action), h_state, cfg, enforce_region)
            next_state, obs, info = env.generative_step(mid, state_p, a, slack_pressure, rng)
            risk_t = env.failure_prob(h_state)
            reward = maintenance_reward(a, info.get("dur", 0.0), local_urgency, risk_t, False, cfg)
            return next_state, obs, reward

        action = int(pomcp.plan(belief, model, cfg.POMCP_NUM_SIMS, cfg.POMCP_HORIZON))
        return int(enforce_action_by_region(action, h_obs, cfg, enforce_region))

    # fallback: conservative threshold rule
    fallback_action = 2 if h_obs < cfg.Hy else 0
    return int(enforce_action_by_region(fallback_action, h_obs, cfg, enforce_region))

def _make_scheduler_agent(cfg: SimConfig, seed: int, device) -> THDQNAgent:
    return THDQNAgent(state_dim=12, cfg=cfg, rng=make_rng(seed, "sched_agent"), device=device)


def _make_maint_agent(cfg: SimConfig, seed: int, device) -> MaintenanceAgentDDQN:
    return MaintenanceAgentDDQN(state_dim=17, cfg=cfg, rng=make_rng(seed, "maint_agent"), device=device)


def _build_run_record(seed: int, mode: str, policy_tag: str, final_result: Dict[str, Any]) -> Dict[str, Any]:
    decision_log = list(final_result.get("decision_log", []))
    maint_counts = summarize_action_counts(extract_maintenance_rows(decision_log), key="kind", values=["DN", "IM", "CM"])
    schedule_summary = summarize_scheduling_strategy(decision_log)
    return {
        "seed": int(seed),
        "maint_mode": str(mode),
        "maint_mode_tag": build_maint_mode_tag(mode),
        "policy_tag": policy_tag,
        "policy_label": final_result["policy_label"],
        "tard": float(final_result["metrics"]["tard"]),
        "maint": float(final_result["metrics"]["maint"]),
        "total": float(final_result["metrics"]["total"]),
        "overdue_ratio_ops": float(final_result["overdue"]["ratio_ops"]),
        "overdue_ratio_time": float(final_result["overdue"]["ratio_time"]),
        "dispatch_count": int(schedule_summary["dispatch_count"]),
        "scheduling_events": int(schedule_summary["scheduling_events"]),
        "breakdown_count": int(schedule_summary["breakdown_count"]),
        "makespan": float(schedule_summary["makespan"]),
        "maint_dn": int(maint_counts.get("DN", 0)),
        "maint_im": int(maint_counts.get("IM", 0)),
        "maint_cm": int(maint_counts.get("CM", 0)),
    }


def _calc_mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def write_aggregate_compare_outputs(outdir: Path, policy_tag: str,
                                    result_rows: List[Dict[str, Any]],
                                    compare_rows: List[Dict[str, Any]]):
    outdir.mkdir(parents=True, exist_ok=True)
    policy_rows = [row for row in result_rows if row["policy_tag"] == policy_tag]
    compare_policy_rows = [row for row in compare_rows if row["policy_tag"] == policy_tag]
    payload: Dict[str, Any] = {"policy_tag": policy_tag, "modes": {}, "delta_pomcp_minus_dqn": {}}
    csv_rows: List[Dict[str, Any]] = []
    metrics = ["tard", "maint", "total", "overdue_ratio_ops", "dispatch_count", "makespan"]
    for mode in ("DQN", "POMCP"):
        mode_rows = [row for row in policy_rows if row["maint_mode"] == mode]
        summary = {"mode": mode}
        for metric in metrics:
            stats = _calc_mean_std([float(row[metric]) for row in mode_rows])
            summary[f"{metric}_mean"] = stats["mean"]
            summary[f"{metric}_std"] = stats["std"]
        payload["modes"][mode] = summary
        csv_rows.append(summary)
    delta_summary = {"mode": "POMCP_MINUS_DQN"}
    delta_map = {
        "tard": "delta_tard",
        "maint": "delta_maint",
        "total": "delta_total",
        "overdue_ratio_ops": "delta_overdue_ratio",
        "dispatch_count": "delta_dispatch_count",
        "makespan": "delta_makespan",
    }
    for out_metric, source_key in delta_map.items():
        stats = _calc_mean_std([float(row[source_key]) for row in compare_policy_rows])
        delta_summary[f"{out_metric}_mean"] = stats["mean"]
        delta_summary[f"{out_metric}_std"] = stats["std"]
    payload["delta_pomcp_minus_dqn"] = delta_summary
    csv_rows.append(delta_summary)
    json_path = outdir / f"aggregate_compare_{policy_tag}.json"
    csv_path = outdir / f"aggregate_compare_{policy_tag}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    fieldnames = sorted({key for row in csv_rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def train_one_mode(
    base_cfg: SimConfig,
    mode: str,
    seed: int,
    device,
    degr: DegradationReplay,
    rul: RULPredictorWrapper,
    scenario_bank: ScenarioBank,
    outdir: Path,
    ckpt_dir: Path,
    ts: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.SEED = int(seed)
    cfg.MAINT_MODE = str(mode).upper()
    cfg.CKPT_DIR = str(ckpt_dir)
    cfg.ENFORCE_REGION_POLICY = True
    set_seed(derive_seed(seed, "global_init"))

    sched_agent = _make_scheduler_agent(cfg, seed, device)
    maint_agent = _make_maint_agent(cfg, seed, device) if cfg.MAINT_MODE == "DQN" else None
    ckpt_mgr = CheckpointManager(str(ckpt_dir), cfg, device)

    base_degrad = float(base_cfg.BASE_DEGRADATION_RATE)
    ep_tard: List[float] = []
    ep_maint: List[float] = []
    ep_dn_rate: List[float] = []
    ep_im_rate: List[float] = []
    ep_cm_rate: List[float] = []
    ep_avg_im: List[float] = []
    stable_count = 0
    stop_ep = None
    last_env = None

    for ep, scenario in enumerate(scenario_bank.train_scenarios):
        ep_num = ep + 1
        cfg.BASE_DEGRADATION_RATE = float(scenario.degradation_rate)
        env_breakdown_rng = make_rng(seed, "train", ep_num, "env_breakdown")
        env_obs_rng = make_rng(seed, "train", ep_num, "env_obs")
        belief_rng = make_rng(seed, mode, "train", ep_num, "belief")
        env = EventDrivenShopEnv(cfg, env_breakdown_rng, degr, rul, breakdown_rng=env_breakdown_rng, obs_rng=env_obs_rng)
        env.reset(machine_curve_ids=scenario.machine_curve_ids, scenario=scenario)
        episode_pomcp = (
            POMCPPlanner(num_actions=3, gamma=cfg.GAMMA, c_ucb=cfg.POMCP_UCB_C, rng=make_rng(seed, mode, "train", ep_num, "pomcp"))
            if cfg.MAINT_MODE == "POMCP" else None
        )

        prev_tard, prev_maint = 0.0, 0.0
        last_h = {m.mid: None for m in env.machines}
        slack_samples = []
        pressure_samples = []
        maint_counts = {0: 0, 1: 0, 2: 0}
        maint_intervals = []
        urgency_action_vals = {0: [], 1: [], 2: []}
        last_maint_time = None
        pending_maint = {}
        slack_action_counts = [[0, 0, 0] for _ in range(len(cfg.SLACK_PRESSURE_BINS) - 1)]
        eta_action_counts = [[0, 0, 0] for _ in range(len(cfg.ETA_BINS) - 1)]
        h_action_counts = [[0, 0, 0] for _ in range(len(cfg.H_BINS) - 1)]
        pomcp_beliefs = {m.mid: [] for m in env.machines}
        enforce_region_train = bool(getattr(cfg, "ENFORCE_REGION_POLICY", True))

        while not env.done():
            done_flag, etype, payload = env.step_until_decision()
            if done_flag:
                break

            if etype == "MACHINE_IDLE":
                mid = payload["mid"]
                if not payload.get("from_maint", False):
                    if mid in pending_maint:
                        rec = pending_maint[mid]
                        h_now = env.maintenance_decision_point(mid)
                        avg_slack, _, _, slack_pressure = env.compute_slack_stats()
                        local_urgency = env.get_local_urgency(mid)
                        arrivals, lambda_hat, _, _, ddt_hat, rush = env.get_obs_estimates(avg_slack, slack_pressure)
                        risk_now = env.failure_prob(h_now)
                        action_now = enforce_action_by_region(rec["action"], h_now, cfg, enforce_region_train)
                        execute_now = (
                            action_now == 0
                            or env.time >= rec["t_e"]
                            or (enforce_region_train and h_now < cfg.Hy)
                        )
                        if execute_now:
                            window_violation = (env.time > rec["t_l"]) if action_now != 0 else False
                            if action_now == 0:
                                dur, kind, post_rul = 0.0, "DN", None
                            else:
                                dur, kind, post_rul = env.apply_maintenance(mid, action_now, h=h_now)
                            r = maintenance_reward(action_now, dur, local_urgency, risk_now, window_violation, cfg)
                            slack_samples.append(avg_slack)
                            pressure_samples.append(slack_pressure)
                            maint_counts[action_now] = maint_counts.get(action_now, 0) + 1
                            urgency_action_vals[action_now].append(local_urgency)
                            if action_now in (1, 2):
                                if last_maint_time is not None:
                                    maint_intervals.append(env.time - last_maint_time)
                                last_maint_time = env.time

                            h2 = post_rul if action_now != 0 else h_now
                            dh2 = h2 - h_now
                            if dh2 < -cfg.ETA_EPS:
                                eta2 = max(0.0, (h2 - cfg.Hy) / (-dh2))
                            else:
                                eta2 = cfg.ETA_CAP
                            risk2 = env.failure_prob(h2)
                            t_e2, t_l2 = compute_window(risk2, local_urgency, env.time, cfg)
                            win_e2 = max(t_e2 - env.time, 0.0)
                            win_l2 = max(t_l2 - env.time, 0.0)
                            rul_mu2 = h2
                            rul_sigma = max(cfg.RUL_OBS_NOISE, 1e-6)
                            belief = refresh_belief(
                                pomcp_beliefs,
                                mid,
                                h2,
                                slack_pressure,
                                cfg,
                                belief_rng,
                                baseline_rul=env.machines[mid].maint_rul_baseline,
                                region_b_elapsed=env.get_region_b_elapsed(mid, h2),
                            )
                            pf_dn2 = estimate_p_fail_horizon(
                                env, mid, belief, slack_pressure, belief_rng, cfg,
                                first_action=0, num_sims=cfg.PFAIL_NUM_SIMS, horizon=cfg.PFAIL_HORIZON
                            )
                            pf_im2 = estimate_p_fail_horizon(
                                env, mid, belief, slack_pressure, belief_rng, cfg,
                                first_action=1, num_sims=cfg.PFAIL_NUM_SIMS, horizon=cfg.PFAIL_HORIZON
                            )
                            pf_cm2 = estimate_p_fail_horizon(
                                env, mid, belief, slack_pressure, belief_rng, cfg,
                                first_action=2, num_sims=cfg.PFAIL_NUM_SIMS, horizon=cfg.PFAIL_HORIZON
                            )
                            pf_dn2 = max(0.0, min(1.0, pf_dn2))
                            pf_im2 = max(0.0, min(1.0, pf_im2))
                            pf_cm2 = max(0.0, min(1.0, pf_cm2))
                            sp = build_maintenance_state(h2, dh2, eta2, slack_pressure, local_urgency, avg_slack,
                                                        lambda_hat, ddt_hat, risk2, win_e2, win_l2,
                                                        rul_mu2, rul_sigma, env.time, env.machines[mid].last_maint_end,
                                                        pf_dn2, pf_im2, pf_cm2)
                            if maint_agent is not None:
                                maint_agent.buf.add(rec["state"], action_now, r, sp, 0.0)
                                maint_agent.learn()
                            last_h[mid] = h2
                            del pending_maint[mid]
                    elif mid not in pending_maint:
                        m = env.machines[mid]
                        h = env.maintenance_decision_point(mid)
                        avg_slack, _, _, slack_pressure = env.compute_slack_stats()
                        local_urgency = env.get_local_urgency(mid)
                        arrivals, lambda_hat, _, _, ddt_hat, rush = env.get_obs_estimates(avg_slack, slack_pressure)
                        prev_h = last_h.get(mid)
                        dh = 0.0 if prev_h is None else (h - prev_h)
                        if dh < -cfg.ETA_EPS:
                            eta = max(0.0, (h - cfg.Hy) / (-dh))
                        else:
                            eta = cfg.ETA_CAP
                        risk_t = env.failure_prob(h)
                        t_e, t_l = compute_window(risk_t, local_urgency, env.time, cfg)
                        win_e = max(t_e - env.time, 0.0)
                        win_l = max(t_l - env.time, 0.0)
                        rul_mu = h
                        rul_sigma = max(cfg.RUL_OBS_NOISE, 1e-6)
                        belief = refresh_belief(
                            pomcp_beliefs,
                            mid,
                            h,
                            slack_pressure,
                            cfg,
                            belief_rng,
                            baseline_rul=env.machines[mid].maint_rul_baseline,
                            region_b_elapsed=env.get_region_b_elapsed(mid, h),
                        )
                        pf_dn = estimate_p_fail_horizon(
                            env, mid, belief, slack_pressure, belief_rng, cfg,
                            first_action=0, num_sims=cfg.PFAIL_NUM_SIMS, horizon=cfg.PFAIL_HORIZON
                        )
                        pf_im = estimate_p_fail_horizon(
                            env, mid, belief, slack_pressure, belief_rng, cfg,
                            first_action=1, num_sims=cfg.PFAIL_NUM_SIMS, horizon=cfg.PFAIL_HORIZON
                        )
                        pf_cm = estimate_p_fail_horizon(
                            env, mid, belief, slack_pressure, belief_rng, cfg,
                            first_action=2, num_sims=cfg.PFAIL_NUM_SIMS, horizon=cfg.PFAIL_HORIZON
                        )
                        pf_dn = max(0.0, min(1.0, pf_dn))
                        pf_im = max(0.0, min(1.0, pf_im))
                        pf_cm = max(0.0, min(1.0, pf_cm))
                        s = build_maintenance_state(h, dh, eta, slack_pressure, local_urgency, avg_slack,
                                                    lambda_hat, ddt_hat, risk_t, win_e, win_l,
                                                    rul_mu, rul_sigma, env.time, m.last_maint_end,
                                                    pf_dn, pf_im, pf_cm)
                        a = select_maintenance_action(
                            cfg.MAINT_MODE, maint_agent, episode_pomcp, pomcp_beliefs, env, mid, h, s,
                            slack_pressure, local_urgency, cfg, belief_rng, explore=True
                        )
                        sbin = bin_index(slack_pressure, cfg.SLACK_PRESSURE_BINS)
                        ebin = bin_index(eta, cfg.ETA_BINS)
                        hbin = bin_index(h, cfg.H_BINS)
                        if 0 <= sbin < len(slack_action_counts):
                            slack_action_counts[sbin][a] += 1
                        if 0 <= ebin < len(eta_action_counts):
                            eta_action_counts[ebin][a] += 1
                        if 0 <= hbin < len(h_action_counts):
                            h_action_counts[hbin][a] += 1
                        if a == 0:
                            r = maintenance_reward(a, 0.0, local_urgency, risk_t, False, cfg)
                            sp = build_maintenance_state(h, 0.0, eta, slack_pressure, local_urgency, avg_slack,
                                                        lambda_hat, ddt_hat, risk_t, win_e, win_l,
                                                        rul_mu, rul_sigma, env.time, m.last_maint_end,
                                                        pf_dn, pf_im, pf_cm)
                            if maint_agent is not None:
                                maint_agent.buf.add(s, a, r, sp, 0.0)
                                maint_agent.learn()
                            last_h[mid] = h
                            maint_counts[a] = maint_counts.get(a, 0) + 1
                            urgency_action_vals[a].append(local_urgency)
                        else:
                            pending_maint[mid] = {
                                "state": s,
                                "action": a,
                                "risk_t": risk_t,
                                "t_e": t_e,
                                "t_l": t_l,
                            }
                            if env.time >= t_e or (enforce_region_train and h < cfg.Hy):
                                rec = pending_maint[mid]
                                h_now = h
                                window_violation = env.time > rec["t_l"]
                                action_now = enforce_action_by_region(rec["action"], h_now, cfg, enforce_region_train)
                                dur, kind, post_rul = env.apply_maintenance(mid, action_now, h=h_now)
                                r = maintenance_reward(action_now, dur, local_urgency, risk_t, window_violation, cfg)
                                slack_samples.append(avg_slack)
                                pressure_samples.append(slack_pressure)
                                maint_counts[action_now] = maint_counts.get(action_now, 0) + 1
                                urgency_action_vals[action_now].append(local_urgency)
                                if action_now in (1, 2):
                                    if last_maint_time is not None:
                                        maint_intervals.append(env.time - last_maint_time)
                                    last_maint_time = env.time

                                h2 = post_rul if action_now != 0 else h_now
                                dh2 = h2 - h_now
                                if dh2 < -cfg.ETA_EPS:
                                    eta2 = max(0.0, (h2 - cfg.Hy) / (-dh2))
                                else:
                                    eta2 = cfg.ETA_CAP
                                risk2 = env.failure_prob(h2)
                                t_e2, t_l2 = compute_window(risk2, local_urgency, env.time, cfg)
                                win_e2 = max(t_e2 - env.time, 0.0)
                                win_l2 = max(t_l2 - env.time, 0.0)
                                rul_mu2 = h2
                                belief = refresh_belief(
                                    pomcp_beliefs,
                                    mid,
                                    h2,
                                    slack_pressure,
                                    cfg,
                                    belief_rng,
                                    baseline_rul=env.machines[mid].maint_rul_baseline,
                                    region_b_elapsed=env.get_region_b_elapsed(mid, h2),
                                )
                                pf_dn2 = estimate_p_fail_horizon(
                                    env, mid, belief, slack_pressure, belief_rng, cfg,
                                    first_action=0, num_sims=cfg.PFAIL_NUM_SIMS, horizon=cfg.PFAIL_HORIZON
                                )
                                pf_im2 = estimate_p_fail_horizon(
                                    env, mid, belief, slack_pressure, belief_rng, cfg,
                                    first_action=1, num_sims=cfg.PFAIL_NUM_SIMS, horizon=cfg.PFAIL_HORIZON
                                )
                                pf_cm2 = estimate_p_fail_horizon(
                                    env, mid, belief, slack_pressure, belief_rng, cfg,
                                    first_action=2, num_sims=cfg.PFAIL_NUM_SIMS, horizon=cfg.PFAIL_HORIZON
                                )
                                pf_dn2 = max(0.0, min(1.0, pf_dn2))
                                pf_im2 = max(0.0, min(1.0, pf_im2))
                                pf_cm2 = max(0.0, min(1.0, pf_cm2))
                                sp = build_maintenance_state(h2, dh2, eta2, slack_pressure, local_urgency, avg_slack,
                                                            lambda_hat, ddt_hat, risk2, win_e2, win_l2,
                                                            rul_mu2, rul_sigma, env.time, m.last_maint_end,
                                                            pf_dn2, pf_im2, pf_cm2)
                                if maint_agent is not None:
                                    maint_agent.buf.add(rec["state"], action_now, r, sp, 0.0)
                                    maint_agent.learn()
                                last_h[mid] = h2
                                del pending_maint[mid]

            while env.has_idle_machine() and env.has_ready_ops():
                S = env.get_global_features()
                slack_samples.append(float(S[6]))
                pressure_samples.append(float(S[8]))
                g, rule = sched_agent.act(S, explore=True)
                env.rule_log.append((env.time, S.copy(), int(g), int(rule)))
                dispatched = env.dispatch(rule)
                if dispatched:
                    tard, maint = env.compute_costs()
                    r_s = scheduling_reward(g, tard, maint, prev_tard, prev_maint)
                    prev_tard, prev_maint = tard, maint
                    Sp = env.get_global_features()
                    sched_agent.buf_h.add(S, g, r_s, Sp, 0.0)
                    sg = np.concatenate([S, np.eye(4, dtype=np.float32)[g]], axis=0)
                    sgp = np.concatenate([Sp, np.eye(4, dtype=np.float32)[g]], axis=0)
                    sched_agent.buf_l.add(sg, rule, r_s, sgp, 0.0)
                    sched_agent.learn()

        tard, maint = env.compute_costs()
        ep_tard.append(float(tard))
        ep_maint.append(float(maint))
        last_env = env
        if ep_num % 20 == 0:
            print(f"[seed {seed}][{build_maint_mode_tag(mode)}] ep {ep_num}/{cfg.TRAIN_EPISODES} tard={tard:.1f} maint={maint:.1f} events={len(env.timeline_ops)}")
        if ep_num % cfg.DIAG_EVERY == 0:
            if slack_samples:
                slack_q = np.quantile(slack_samples, [0.1, 0.5, 0.9])
                press_q = np.quantile(pressure_samples, [0.1, 0.5, 0.9]) if pressure_samples else [0.0, 0.0, 0.0]
                print(
                    f"diag [{build_maint_mode_tag(mode)}][seed {seed}] ep {ep_num}: avg_slack_q10/50/90="
                    f"{slack_q[0]:.1f}/{slack_q[1]:.1f}/{slack_q[2]:.1f} "
                    f"slack_pressure_q10/50/90="
                    f"{press_q[0]:.2f}/{press_q[1]:.2f}/{press_q[2]:.2f} "
                    f"maint DN/IM/CM={maint_counts.get(0,0)}/{maint_counts.get(1,0)}/{maint_counts.get(2,0)}"
                )
                if slack_action_counts:
                    for i, cnt in enumerate(slack_action_counts):
                        total = sum(cnt)
                        if total == 0:
                            continue
                        rates = [c / total for c in cnt]
                        lo = cfg.SLACK_PRESSURE_BINS[i]
                        hi = cfg.SLACK_PRESSURE_BINS[i + 1]
                        print(f"diag [{build_maint_mode_tag(mode)}][seed {seed}] ep {ep_num}: slack_bin[{lo:.2f},{hi:.2f}) DN/IM/CM={rates[0]:.2f}/{rates[1]:.2f}/{rates[2]:.2f}")
                if eta_action_counts:
                    for i, cnt in enumerate(eta_action_counts):
                        total = sum(cnt)
                        if total == 0:
                            continue
                        rates = [c / total for c in cnt]
                        lo = cfg.ETA_BINS[i]
                        hi = cfg.ETA_BINS[i + 1]
                        print(f"diag [{build_maint_mode_tag(mode)}][seed {seed}] ep {ep_num}: eta_bin[{lo:.1f},{hi:.1f}) DN/IM/CM={rates[0]:.2f}/{rates[1]:.2f}/{rates[2]:.2f}")
                if h_action_counts:
                    for i, cnt in enumerate(h_action_counts):
                        total = sum(cnt)
                        if total == 0:
                            continue
                        rates = [c / total for c in cnt]
                        lo = cfg.H_BINS[i]
                        hi = cfg.H_BINS[i + 1]
                        print(f"diag [{build_maint_mode_tag(mode)}][seed {seed}] ep {ep_num}: h_bin[{lo:.2f},{hi:.2f}) DN/IM/CM={rates[0]:.2f}/{rates[1]:.2f}/{rates[2]:.2f}")
                if maint_intervals:
                    avg_gap = float(np.mean(maint_intervals))
                    print(f"diag [{build_maint_mode_tag(mode)}][seed {seed}] ep {ep_num}: avg_time_between_maint={avg_gap:.1f}")
                if any(urgency_action_vals[a] for a in urgency_action_vals):
                    avg_urg = {a: (float(np.mean(v)) if v else 0.0) for a, v in urgency_action_vals.items()}
                    print(f"diag [{build_maint_mode_tag(mode)}][seed {seed}] ep {ep_num}: avg_local_urgency DN/IM/CM={avg_urg[0]:.2f}/{avg_urg[1]:.2f}/{avg_urg[2]:.2f}")
            else:
                print(f"diag [{build_maint_mode_tag(mode)}][seed {seed}] ep {ep_num}: no slack samples")

        total_actions = maint_counts.get(0, 0) + maint_counts.get(1, 0) + maint_counts.get(2, 0)
        if total_actions > 0:
            ep_dn_rate.append(maint_counts.get(0, 0) / total_actions)
            ep_im_rate.append(maint_counts.get(1, 0) / total_actions)
            ep_cm_rate.append(maint_counts.get(2, 0) / total_actions)
            ep_avg_im.append(0.0)
        else:
            ep_dn_rate.append(0.0)
            ep_im_rate.append(0.0)
            ep_cm_rate.append(0.0)
            ep_avg_im.append(0.0)

        saved_latest = False
        if cfg.EVAL_EVERY > 0 and ep_num in scenario_bank.periodic_eval_scenarios:
            eval_cfg = copy.deepcopy(cfg)
            eval_cfg.BASE_DEGRADATION_RATE = base_degrad
            eval_cfg.ENFORCE_REGION_POLICY = True
            eval_scenario = scenario_bank.periodic_eval_scenarios[ep_num]
            eval_pomcp = (
                POMCPPlanner(num_actions=3, gamma=cfg.GAMMA, c_ucb=cfg.POMCP_UCB_C, rng=make_rng(seed, mode, "periodic_eval", ep_num, "pomcp"))
                if cfg.MAINT_MODE == "POMCP" else None
            )
            eval_metrics, _, _ = evaluate_once(
                eval_cfg,
                make_rng(seed, mode, "periodic_eval", ep_num, "root"),
                degr,
                rul,
                sched_agent,
                maint_agent,
                cfg.MAINT_MODE,
                eval_pomcp,
                eval_scenario.machine_curve_ids,
                generate_outputs=False,
                freeze_steps=True,
                scenario=eval_scenario,
                env_rng=make_rng(seed, "periodic_eval", ep_num, "env_breakdown"),
                rul_obs_rng=make_rng(seed, "periodic_eval", ep_num, "env_obs"),
                belief_rng=make_rng(seed, mode, "periodic_eval", ep_num, "belief"),
            )
            eval_metrics["episode"] = ep_num
            ckpt_mgr.append_metrics(eval_metrics, derive_seed(seed, "periodic_eval", ep_num))
            env_state = {
                "slack_scale": env.slack_scale,
                "last_slack_pressure": env.last_slack_pressure,
            }
            ckpt_mgr.maybe_save_best(
                sched_agent, maint_agent, env.observer, env_state, eval_metrics,
                step_info={"episode": ep_num, "phase": "eval"},
            )
            if cfg.SAVE_EVERY > 0 and ep_num % cfg.SAVE_EVERY == 0:
                ckpt_mgr.save_latest(
                    sched_agent, maint_agent, env.observer, env_state, eval_metrics,
                    step_info={"episode": ep_num, "phase": "eval"},
                )
                saved_latest = True
        if cfg.SAVE_EVERY > 0 and ep_num % cfg.SAVE_EVERY == 0 and not saved_latest:
            train_metrics = {
                "episode": ep_num,
                "tard": float(tard),
                "maint": float(maint),
                "total": float(tard + maint),
            }
            env_state = {
                "slack_scale": env.slack_scale,
                "last_slack_pressure": env.last_slack_pressure,
            }
            ckpt_mgr.save_latest(
                sched_agent, maint_agent, env.observer, env_state, train_metrics,
                step_info={"episode": ep_num, "phase": "train"},
            )
        if cfg.EARLY_STOP_ENABLED and len(ep_tard) >= 2 * cfg.EARLY_STOP_WINDOW:
            w = cfg.EARLY_STOP_WINDOW
            total = np.array(ep_tard, dtype=np.float32) + np.array(ep_maint, dtype=np.float32)
            prev = float(np.mean(total[-2 * w:-w]))
            curr = float(np.mean(total[-w:]))
            rel = abs(curr - prev) / max(abs(prev), cfg.EARLY_STOP_EPS)
            if rel <= cfg.EARLY_STOP_REL_TOL:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= cfg.EARLY_STOP_PATIENCE:
                stop_ep = ep_num
                print(f"[seed {seed}][{build_maint_mode_tag(mode)}] early stop at ep {stop_ep}: rel_change={rel:.4f}, window={w}")
                break

    cfg.BASE_DEGRADATION_RATE = base_degrad
    final_results: Dict[str, Dict[str, Any]] = {}
    maint_mode_tag = build_maint_mode_tag(cfg.MAINT_MODE)
    final_scenario = scenario_bank.final_eval_scenario
    for enforce_region in (True, False):
        eval_cfg = copy.deepcopy(cfg)
        eval_cfg.BASE_DEGRADATION_RATE = float(final_scenario.degradation_rate)
        eval_cfg.ENFORCE_REGION_POLICY = enforce_region
        policy_tag = build_policy_tag(eval_cfg, enforce_region)
        policy_label = build_policy_context_label(eval_cfg, enforce_region, cfg.MAINT_MODE)
        eval_pomcp = (
            POMCPPlanner(num_actions=3, gamma=cfg.GAMMA, c_ucb=cfg.POMCP_UCB_C, rng=make_rng(seed, cfg.MAINT_MODE, "final_eval", policy_tag, "pomcp"))
            if cfg.MAINT_MODE == "POMCP" else None
        )
        eval_metrics, eval_env, overdue_stats = evaluate_once(
            eval_cfg,
            make_rng(seed, cfg.MAINT_MODE, "final_eval", policy_tag, "root"),
            degr,
            rul,
            sched_agent,
            maint_agent,
            cfg.MAINT_MODE,
            eval_pomcp,
            final_scenario.machine_curve_ids,
            generate_outputs=True,
            outdir=outdir,
            plot_prefix=f"{ts}_{maint_mode_tag}_{policy_tag}",
            decision_log_name=f"decision_log_{ts}_{maint_mode_tag}_{policy_tag}.csv",
            freeze_steps=True,
            policy_label=policy_label,
            threshold_enforced=enforce_region,
            scenario=final_scenario,
            env_rng=make_rng(seed, "final_eval", policy_tag, "env_breakdown"),
            rul_obs_rng=make_rng(seed, "final_eval", policy_tag, "env_obs"),
            belief_rng=make_rng(seed, cfg.MAINT_MODE, "final_eval", policy_tag, "belief"),
        )
        final_results[policy_tag] = {
            "metrics": eval_metrics,
            "env": eval_env,
            "overdue": overdue_stats,
            "policy_label": policy_label,
            "enforce_region": enforce_region,
            "decision_log": list(getattr(eval_env, "last_decision_log", [])),
        }
        summary_row = {
            "timestamp": ts,
            "seed": int(seed),
            "policy_tag": policy_tag,
            "policy_label": policy_label,
            "maint_mode_tag": maint_mode_tag,
            "enforce_region_policy": int(enforce_region),
            "hx": float(eval_cfg.Hx),
            "hy": float(eval_cfg.Hy),
            "jobs_target": int(final_scenario.jobs_target),
            "maint_mode": str(cfg.MAINT_MODE),
            "tard": float(eval_metrics["tard"]),
            "maint": float(eval_metrics["maint"]),
            "total": float(eval_metrics["total"]),
            "overdue_ratio_ops": float(overdue_stats["ratio_ops"]),
            "overdue_ratio_time": float(overdue_stats["ratio_time"]),
            "overdue_ops": float(overdue_stats["overdue_ops"]),
            "total_ops": float(overdue_stats["total_ops"]),
        }
        write_summary_files(outdir, f"summary_{ts}_{maint_mode_tag}_{policy_tag}", summary_row)

    plot_training_curves(ep_tard, ep_maint, str(outdir / f"training_curves_{ts}_{maint_mode_tag}.png"),
                         smooth_window=cfg.CURVE_SMOOTH_WINDOW, stop_ep=stop_ep)
    plot_maint_action_rates(ep_dn_rate, ep_im_rate, ep_cm_rate, str(outdir / f"maint_action_rates_{ts}_{maint_mode_tag}.png"),
                            avg_im_counts=ep_avg_im)

    constrained_tag = build_policy_tag(cfg, True)
    constrained = final_results[constrained_tag]
    final_metrics = dict(constrained["metrics"])
    final_metrics["episode"] = stop_ep or cfg.TRAIN_EPISODES
    ckpt_mgr.append_metrics(final_metrics, seed)
    env_state = {
        "slack_scale": constrained["env"].slack_scale,
        "last_slack_pressure": constrained["env"].last_slack_pressure,
    }
    ckpt_mgr.save_latest(
        sched_agent, maint_agent, constrained["env"].observer, env_state, final_metrics,
        step_info={"episode": final_metrics["episode"], "phase": "final_eval"},
    )
    ckpt_mgr.maybe_save_best(
        sched_agent, maint_agent, constrained["env"].observer, env_state, final_metrics,
        step_info={"episode": final_metrics["episode"], "phase": "final_eval"},
    )
    ckpt_mgr.ensure_best_exists()

    return {
        "seed": int(seed),
        "maint_mode": cfg.MAINT_MODE,
        "maint_mode_tag": maint_mode_tag,
        "outdir": outdir,
        "ckpt_dir": ckpt_dir,
        "final_results": final_results,
        "stop_ep": stop_ep or cfg.TRAIN_EPISODES,
        "last_env": last_env,
    }


def main():
    cfg = SimConfig()
    validate_region_thresholds(cfg)
    cfg.ENFORCE_REGION_POLICY = True
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:", device)

    base_machine_curve_ids = list(cfg.MACHINE_CURVE_IDS)
    _, degr, rul = build_degradation_and_rul(cfg, base_machine_curve_ids)
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path("outputs") / f"paired_{ts}"
    ckpt_root = Path(cfg.CKPT_DIR) / f"paired_{ts}"
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    experiment_seeds = [int(s) for s in getattr(cfg, "EXPERIMENT_SEEDS", (cfg.SEED,))]
    maint_modes = [str(m).upper() for m in getattr(cfg, "TRAIN_MAINT_MODES", ("DQN", "POMCP"))]
    all_result_rows: List[Dict[str, Any]] = []
    all_compare_rows: List[Dict[str, Any]] = []

    for seed in experiment_seeds:
        print(f"paired experiment seed={seed}")
        scenario_bank = build_scenario_bank(seed, cfg, degr, machine_curve_ids=base_machine_curve_ids)
        seed_results: Dict[str, Dict[str, Any]] = {}
        seed_root = output_root / f"seed_{seed:04d}"
        seed_root.mkdir(parents=True, exist_ok=True)
        for mode in maint_modes:
            mode_tag = build_maint_mode_tag(mode)
            mode_outdir = seed_root / mode_tag
            mode_ckpt_dir = ckpt_root / f"seed_{seed:04d}" / mode_tag
            mode_outdir.mkdir(parents=True, exist_ok=True)
            mode_ckpt_dir.mkdir(parents=True, exist_ok=True)
            run_result = train_one_mode(
                cfg,
                mode,
                seed,
                device,
                degr,
                rul,
                scenario_bank,
                mode_outdir,
                mode_ckpt_dir,
                ts,
            )
            seed_results[mode] = run_result
            for policy_tag, final_result in run_result["final_results"].items():
                all_result_rows.append(_build_run_record(seed, mode, policy_tag, final_result))

        if "DQN" in seed_results and "POMCP" in seed_results:
            compare_dir = seed_root / "compare"
            compare_dir.mkdir(parents=True, exist_ok=True)
            for enforce_region in (True, False):
                policy_tag = build_policy_tag(cfg, enforce_region)
                if policy_tag not in seed_results["DQN"]["final_results"] or policy_tag not in seed_results["POMCP"]["final_results"]:
                    continue
                compare_summary, compare_rows = compare_mode_results(
                    seed_results["DQN"]["final_results"][policy_tag],
                    seed_results["POMCP"]["final_results"][policy_tag],
                    "DQN",
                    "POMCP",
                )
                compare_summary["seed"] = int(seed)
                compare_summary["policy_tag"] = policy_tag
                compare_summary["policy_label"] = build_policy_label(cfg, enforce_region)
                compare_summary["delta_tard"] = float(compare_summary["delta_compare_minus_primary"]["tard"])
                compare_summary["delta_maint"] = float(compare_summary["delta_compare_minus_primary"]["maint"])
                compare_summary["delta_total"] = float(compare_summary["delta_compare_minus_primary"]["total"])
                compare_summary["delta_overdue_ratio"] = float(compare_summary["delta_compare_minus_primary"]["overdue_ratio"])
                compare_summary["delta_dispatch_count"] = int(compare_summary["delta_schedule_summary"]["dispatch_count"])
                compare_summary["delta_makespan"] = float(compare_summary["delta_schedule_summary"]["makespan"])
                all_compare_rows.append(dict(compare_summary))
                write_mode_comparison_outputs(
                    compare_dir,
                    f"maint_compare_{ts}_{policy_tag}",
                    compare_summary,
                    compare_rows,
                    policy_label=f"{build_policy_label(cfg, enforce_region)} | Maintenance: DQN vs POMCP",
                )
                print(
                    f"[seed {seed}][{policy_tag}] delta(POMCP-DQN): "
                    f"tard={compare_summary['delta_tard']:.3f}, "
                    f"maint={compare_summary['delta_maint']:.3f}, "
                    f"total={compare_summary['delta_total']:.3f}, "
                    f"overdue_ratio={compare_summary['delta_overdue_ratio']:.3f}, "
                    f"dispatch={compare_summary['delta_dispatch_count']}, "
                    f"makespan={compare_summary['delta_makespan']:.3f}"
                )

    paired_results_csv = output_root / "paired_final_eval_rows.csv"
    paired_results_json = output_root / "paired_final_eval_rows.json"
    with paired_results_json.open("w", encoding="utf-8") as f:
        json.dump(all_result_rows, f, ensure_ascii=True, indent=2)
    if all_result_rows:
        fieldnames = sorted({key for row in all_result_rows for key in row.keys()})
        with paired_results_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_result_rows)

    for policy_tag in {row["policy_tag"] for row in all_result_rows}:
        write_aggregate_compare_outputs(output_root, policy_tag, all_result_rows, all_compare_rows)

    print(f"paired training finished; outputs saved to {output_root}")

if __name__ == "__main__":
    main()
