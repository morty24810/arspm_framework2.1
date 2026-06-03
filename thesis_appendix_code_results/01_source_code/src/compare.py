from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

from .viz import plot_maint_mode_comparison


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _missing_status_label(mode: str) -> str:
    normalized = str(mode).strip().lower()
    return f"missing_on_{normalized}" if normalized else "missing_on_unknown"


def _normalize_op(op: Any) -> Dict[str, Any] | None:
    if op is None:
        return None
    if isinstance(op, dict):
        return op
    if isinstance(op, str) and op:
        try:
            parsed = json.loads(op)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def extract_maintenance_rows(decision_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in (decision_log or []) if row.get("event") == "maintenance"]


def extract_scheduling_rows(decision_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in (decision_log or []) if row.get("event") == "scheduling"]


def summarize_action_counts(rows: List[Dict[str, Any]], key: str, values: List[Any]) -> Dict[str, int]:
    counts = {str(v): 0 for v in values}
    for row in rows:
        raw = row.get(key)
        normalized = raw
        if isinstance(raw, float) and raw.is_integer():
            normalized = int(raw)
        label = str(normalized)
        if label in counts:
            counts[label] += 1
    return counts


def _combo_key_from_row(row: Dict[str, Any]) -> str | None:
    lam = _safe_float(row.get("lambda_true_segment"))
    ddt = _safe_float(row.get("ddt_true_segment"))
    if lam is None or ddt is None:
        return None
    return f"lam={lam:.1f}|ddt={ddt:.2f}"


def _dominant_from_counts(counts: Dict[str, int]) -> Dict[str, Any]:
    total = int(sum(int(v) for v in counts.values()))
    if total <= 0:
        return {"label": None, "share": 0.0}
    dominant_label, dominant_count = max(counts.items(), key=lambda kv: int(kv[1]))
    return {
        "label": dominant_label,
        "share": float(dominant_count / max(total, 1)),
    }


def _shares_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    total = int(sum(int(v) for v in counts.values()))
    if total <= 0:
        return {str(key): 0.0 for key in counts.keys()}
    return {
        str(key): float(int(value) / total)
        for key, value in counts.items()
    }


def summarize_combo_conditioned_behavior(decision_log: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    combo_summary: Dict[str, Dict[str, Any]] = {}
    for row in decision_log or []:
        combo_key = _combo_key_from_row(row)
        if combo_key is None:
            continue
        entry = combo_summary.setdefault(
            combo_key,
            {
                "rule_counts": {str(i): 0 for i in range(6)},
                "goal_counts": {str(i): 0 for i in range(4)},
                "maint_counts": {"DN": 0, "IM": 0, "CM": 0},
                "dispatch_count": 0,
                "maintenance_count": 0,
            },
        )
        event = str(row.get("event", "")).lower()
        if event == "scheduling":
            rule = row.get("rule")
            if isinstance(rule, float) and float(rule).is_integer():
                rule = int(rule)
            rule_key = str(rule)
            if rule_key in entry["rule_counts"]:
                entry["rule_counts"][rule_key] += 1
            goal = row.get("goal")
            if isinstance(goal, float) and float(goal).is_integer():
                goal = int(goal)
            goal_key = str(goal)
            if goal_key in entry["goal_counts"]:
                entry["goal_counts"][goal_key] += 1
            if bool(row.get("dispatched")):
                entry["dispatch_count"] += 1
        elif event == "maintenance":
            kind = str(row.get("kind", "")).upper()
            if kind in entry["maint_counts"]:
                entry["maint_counts"][kind] += 1
            entry["maintenance_count"] += 1

    for entry in combo_summary.values():
        entry["dominant_rule"] = _dominant_from_counts(entry["rule_counts"])
        entry["dominant_goal"] = _dominant_from_counts(entry["goal_counts"])
        entry["rule_shares"] = _shares_from_counts(entry["rule_counts"])
        entry["goal_shares"] = _shares_from_counts(entry["goal_counts"])
        entry["maint_shares"] = _shares_from_counts(entry["maint_counts"])
    return combo_summary


def combo_dominant_maps(combo_behavior: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dominant_rule_by_combo: Dict[str, Any] = {}
    dominant_goal_by_combo: Dict[str, Any] = {}
    for combo_key, entry in (combo_behavior or {}).items():
        dominant_rule_by_combo[str(combo_key)] = dict(entry.get("dominant_rule", {}))
        dominant_goal_by_combo[str(combo_key)] = dict(entry.get("dominant_goal", {}))
    return dominant_rule_by_combo, dominant_goal_by_combo


def combo_rule_diversity_metrics(combo_behavior: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    labels = [str(i) for i in range(6)]
    dominant_labels = []
    dominant_shares = []
    prob_vectors: List[List[float]] = []
    for entry in (combo_behavior or {}).values():
        dominant_rule = dict(entry.get("dominant_rule", {}) or {})
        label = str(dominant_rule.get("label", "")).strip()
        share = float(dominant_rule.get("share", 0.0) or 0.0)
        if label:
            dominant_labels.append(label)
        dominant_shares.append(share)
        shares = dict(entry.get("rule_shares", {}) or {})
        vec = [float(shares.get(label, 0.0) or 0.0) for label in labels]
        total = sum(vec)
        if total > 0.0:
            vec = [val / total for val in vec]
        prob_vectors.append(vec)

    def _kl(p: List[float], q: List[float]) -> float:
        total = 0.0
        for pi, qi in zip(p, q):
            if pi <= 0.0:
                continue
            total += float(pi) * math.log(float(pi) / max(float(qi), 1e-12), 2.0)
        return float(total)

    jsd_vals: List[float] = []
    for i in range(len(prob_vectors)):
        for j in range(i + 1, len(prob_vectors)):
            p = prob_vectors[i]
            q = prob_vectors[j]
            m = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]
            jsd_vals.append(0.5 * (_kl(p, m) + _kl(q, m)))

    return {
        "rule_distinct_count": float(len(set(dominant_labels))),
        "rule_avg_dominant_share": float(sum(dominant_shares) / len(dominant_shares)) if dominant_shares else 0.0,
        "rule_mean_pairwise_jsd": float(sum(jsd_vals) / len(jsd_vals)) if jsd_vals else 0.0,
    }


def compute_decision_log_makespan(decision_log: List[Dict[str, Any]]) -> float:
    t_end = 0.0
    for row in decision_log or []:
        if row.get("event") == "maintenance":
            t0 = _safe_float(row.get("time")) or 0.0
            dur = _safe_float(row.get("duration")) or 0.0
            t_end = max(t_end, t0 + dur)
            continue
        if row.get("event") != "scheduling":
            continue
        op = _normalize_op(row.get("op"))
        if op is not None:
            t_end = max(t_end, _safe_float(op.get("t1")) or 0.0)
        else:
            t_end = max(t_end, _safe_float(row.get("time")) or 0.0)
    return float(t_end)


def _env_makespan(env: Any) -> float:
    if env is None:
        return 0.0
    t_end = 0.0
    for _, t0, t1, _, _, *_ in getattr(env, "timeline_ops", []):
        t_end = max(t_end, float(t1))
    for _, t0, t1, _ in getattr(env, "timeline_maint", []):
        t_end = max(t_end, float(t1))
    return float(t_end)


def _env_final_health_summary(env: Any) -> Dict[str, float]:
    if env is None or not hasattr(env, "machines"):
        return {
            "final_health_mean": 0.0,
            "final_health_min": 0.0,
            "final_health_p25": 0.0,
            "final_low_health_count_h20": 0.0,
            "final_low_health_count_h10": 0.0,
        }
    machines = getattr(env, "machines", [])
    if isinstance(machines, dict):
        machine_iter = list(machines.values())
    else:
        machine_iter = list(machines)
    values = []
    for machine in machine_iter:
        mid = getattr(machine, "mid", None)
        if mid is None:
            continue
        try:
            if hasattr(env, "peek_rul_true"):
                values.append(float(env.peek_rul_true(mid)))
            elif hasattr(env, "maintenance_decision_point"):
                values.append(float(env.maintenance_decision_point(mid)))
        except Exception:
            continue
    if not values:
        return {
            "final_health_mean": 0.0,
            "final_health_min": 0.0,
            "final_health_p25": 0.0,
            "final_low_health_count_h20": 0.0,
            "final_low_health_count_h10": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "final_health_mean": float(np.mean(arr)),
        "final_health_min": float(np.min(arr)),
        "final_health_p25": float(np.percentile(arr, 25.0)),
        "final_low_health_count_h20": float(np.sum(arr < 0.20)),
        "final_low_health_count_h10": float(np.sum(arr < 0.10)),
    }


def summarize_scheduling_strategy(decision_log: List[Dict[str, Any]], env: Any = None) -> Dict[str, Any]:
    sched_rows = extract_scheduling_rows(decision_log)
    goal_counts = summarize_action_counts(sched_rows, key="goal", values=[0, 1, 2, 3])
    rule_counts = summarize_action_counts(sched_rows, key="rule", values=[0, 1, 2, 3, 4, 5])
    total_rule_count = sum(int(v) for v in rule_counts.values())
    rule_unique_count = sum(1 for v in rule_counts.values() if int(v) > 0)
    rule_top1_share = (max((int(v) for v in rule_counts.values()), default=0) / total_rule_count) if total_rule_count else 0.0
    dispatched_count = sum(1 for row in sched_rows if bool(row.get("dispatched")))
    breakdown_count = sum(1 for row in sched_rows if bool(row.get("breakdown_flag")))
    sched_safe_block_count = sum(1 for row in sched_rows if bool(row.get("sched_safe_blocked")))
    hard_breakdown_count = max((_safe_int(row.get("hard_breakdown_count")) or 0) for row in sched_rows) if sched_rows else 0
    stochastic_breakdown_count = max((_safe_int(row.get("stochastic_breakdown_count")) or 0) for row in sched_rows) if sched_rows else 0
    requeued_op_count = max((_safe_int(row.get("requeued_op_count")) or 0) for row in sched_rows) if sched_rows else 0
    interrupted_proc_time = max((_safe_float(row.get("interrupted_proc_time")) or 0.0) for row in sched_rows) if sched_rows else 0.0
    breakdown_cost = sum((_safe_float(row.get("breakdown_cost")) or 0.0) for row in sched_rows if bool(row.get("breakdown_flag")))
    stress_vals = [
        val for val in (_safe_float(row.get("current_stress")) for row in sched_rows)
        if val is not None
    ]
    return {
        "goal_counts": goal_counts,
        "rule_counts": rule_counts,
        "rule_unique_count": int(rule_unique_count),
        "rule_top1_share": float(rule_top1_share),
        "dispatch_count": int(dispatched_count),
        "scheduling_events": int(len(sched_rows)),
        "sched_safe_block_count": int(sched_safe_block_count),
        "breakdown_count": int(breakdown_count),
        "hard_breakdown_count": int(hard_breakdown_count),
        "stochastic_breakdown_count": int(stochastic_breakdown_count),
        "requeued_op_count": int(requeued_op_count),
        "interrupted_proc_time": float(interrupted_proc_time),
        "breakdown_cost": float(breakdown_cost),
        "current_stress_mean": float(sum(stress_vals) / len(stress_vals)) if stress_vals else 0.0,
        "current_stress_max": float(max(stress_vals)) if stress_vals else 0.0,
        "makespan": _env_makespan(env) if env is not None else compute_decision_log_makespan(decision_log),
    }


def compare_mode_results(
    primary_result: Dict[str, Any],
    compare_result: Dict[str, Any],
    primary_mode: str,
    compare_mode: str,
    *,
    compare_type: str = "full_system",
    train_policy_tag: str | None = None,
    eval_policy_tag: str | None = None,
    scheduler_anchor: str | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    primary_mode = str(primary_mode)
    compare_mode = str(compare_mode)
    primary_maint_variant = str(primary_result.get("maint_variant", primary_mode))
    compare_maint_variant = str(compare_result.get("maint_variant", compare_mode))
    primary_maint_arch = str(primary_result.get("maint_arch", primary_maint_variant))
    compare_maint_arch = str(compare_result.get("maint_arch", compare_maint_variant))
    primary_rows = extract_maintenance_rows(primary_result.get("decision_log", []))
    compare_rows = extract_maintenance_rows(compare_result.get("decision_log", []))
    primary_map = {
        (int(row.get("mid", -1)), int(row.get("maint_seq_machine", -1))): row
        for row in primary_rows
    }
    compare_map = {
        (int(row.get("mid", -1)), int(row.get("maint_seq_machine", -1))): row
        for row in compare_rows
    }
    union_keys = sorted(set(primary_map.keys()) | set(compare_map.keys()))
    missing_on_primary_label = _missing_status_label(primary_mode)
    missing_on_compare_label = _missing_status_label(compare_mode)

    rows: List[Dict[str, Any]] = []
    aligned_count = 0
    divergence_count = 0
    missing_on_primary = 0
    missing_on_compare = 0
    for mid, seq in union_keys:
        primary = primary_map.get((mid, seq))
        compare = compare_map.get((mid, seq))
        if primary is None:
            status = missing_on_primary_label
            missing_on_primary += 1
            divergence_count += 1
        elif compare is None:
            status = missing_on_compare_label
            missing_on_compare += 1
            divergence_count += 1
        else:
            aligned_count += 1
            same_action = str(primary.get("kind", "")).upper() == str(compare.get("kind", "")).upper()
            status = "same_action" if same_action else "different_action"
            if not same_action:
                divergence_count += 1
        rows.append({
            "mid": int(mid),
            "maint_seq_machine": int(seq),
            "status": status,
            "primary_mode": primary_mode,
            "compare_mode": compare_mode,
            "primary_action": primary.get("kind") if primary else None,
            "compare_action": compare.get("kind") if compare else None,
            "primary_time": _safe_float(primary.get("time")) if primary else None,
            "compare_time": _safe_float(compare.get("time")) if compare else None,
            "primary_h": _safe_float(primary.get("h")) if primary else None,
            "compare_h": _safe_float(compare.get("h")) if compare else None,
            "primary_duration": _safe_float(primary.get("duration")) if primary else None,
            "compare_duration": _safe_float(compare.get("duration")) if compare else None,
        })

    primary_schedule_summary = summarize_scheduling_strategy(primary_result.get("decision_log", []), env=primary_result.get("env"))
    compare_schedule_summary = summarize_scheduling_strategy(compare_result.get("decision_log", []), env=compare_result.get("env"))
    primary_combo_behavior = summarize_combo_conditioned_behavior(primary_result.get("decision_log", []))
    compare_combo_behavior = summarize_combo_conditioned_behavior(compare_result.get("decision_log", []))
    primary_dominant_rule_by_combo, primary_dominant_goal_by_combo = combo_dominant_maps(primary_combo_behavior)
    compare_dominant_rule_by_combo, compare_dominant_goal_by_combo = combo_dominant_maps(compare_combo_behavior)
    primary_rule_diversity = combo_rule_diversity_metrics(primary_combo_behavior)
    compare_rule_diversity = combo_rule_diversity_metrics(compare_combo_behavior)
    primary_scheduler_mode = str(primary_result.get("scheduler_mode", getattr(primary_result.get("env"), "last_scheduler_mode", "THDQN"))).upper()
    compare_scheduler_mode = str(compare_result.get("scheduler_mode", getattr(compare_result.get("env"), "last_scheduler_mode", "THDQN"))).upper()
    primary_sched_regime_feature_mode = str(
        primary_result.get(
            "sched_regime_feature_mode",
            getattr(getattr(primary_result.get("env"), "cfg", None), "SCHED_REGIME_FEATURE_MODE", "observer"),
        )
    ).lower()
    compare_sched_regime_feature_mode = str(
        compare_result.get(
            "sched_regime_feature_mode",
            getattr(getattr(compare_result.get("env"), "cfg", None), "SCHED_REGIME_FEATURE_MODE", "observer"),
        )
    ).lower()
    primary_metrics = primary_result["metrics"]
    compare_metrics = compare_result["metrics"]
    primary_overdue = primary_result["overdue"]
    compare_overdue = compare_result["overdue"]
    primary_final_health = _env_final_health_summary(primary_result.get("env"))
    compare_final_health = _env_final_health_summary(compare_result.get("env"))
    union_count = len(union_keys)
    summary = {
        "primary_mode": primary_mode,
        "compare_mode": compare_mode,
        "primary_maint_variant": primary_maint_variant,
        "compare_maint_variant": compare_maint_variant,
        "primary_maint_arch": primary_maint_arch,
        "compare_maint_arch": compare_maint_arch,
        "compare_type": compare_type,
        "train_policy_tag": train_policy_tag,
        "eval_policy_tag": eval_policy_tag,
        "scheduler_anchor": scheduler_anchor,
        "primary_scheduler_mode": primary_scheduler_mode,
        "compare_scheduler_mode": compare_scheduler_mode,
        "primary_sched_regime_feature_mode": primary_sched_regime_feature_mode,
        "compare_sched_regime_feature_mode": compare_sched_regime_feature_mode,
        "primary_action_counts": summarize_action_counts(primary_rows, key="kind", values=["DN", "IM", "CM"]),
        "compare_action_counts": summarize_action_counts(compare_rows, key="kind", values=["DN", "IM", "CM"]),
        "primary_combo_behavior": primary_combo_behavior,
        "compare_combo_behavior": compare_combo_behavior,
        "primary_dominant_rule_by_combo": primary_dominant_rule_by_combo,
        "compare_dominant_rule_by_combo": compare_dominant_rule_by_combo,
        "primary_dominant_goal_by_combo": primary_dominant_goal_by_combo,
        "compare_dominant_goal_by_combo": compare_dominant_goal_by_combo,
        "primary_rule_distinct_count": int(primary_rule_diversity["rule_distinct_count"]),
        "compare_rule_distinct_count": int(compare_rule_diversity["rule_distinct_count"]),
        "primary_rule_avg_dominant_share": float(primary_rule_diversity["rule_avg_dominant_share"]),
        "compare_rule_avg_dominant_share": float(compare_rule_diversity["rule_avg_dominant_share"]),
        "primary_rule_mean_pairwise_jsd": float(primary_rule_diversity["rule_mean_pairwise_jsd"]),
        "compare_rule_mean_pairwise_jsd": float(compare_rule_diversity["rule_mean_pairwise_jsd"]),
        "primary_im_invalid_filtered_count": int(sum(1 for row in primary_rows if bool(row.get("im_invalid_flag")))),
        "compare_im_invalid_filtered_count": int(sum(1 for row in compare_rows if bool(row.get("im_invalid_flag")))),
        "primary_dn_veto_count": int(sum(1 for row in primary_rows if bool(row.get("dn_imminent_breakdown_veto")))),
        "compare_dn_veto_count": int(sum(1 for row in compare_rows if bool(row.get("dn_imminent_breakdown_veto")))),
        "decision_union_count": int(union_count),
        "decision_aligned_count": int(aligned_count),
        "divergence_count": int(divergence_count),
        "divergence_rate": float(divergence_count / union_count) if union_count else 0.0,
        "missing_on_primary": int(missing_on_primary),
        "missing_on_compare": int(missing_on_compare),
        "missing_on_pomcp": int(missing_on_primary if primary_maint_variant == "pomcp" else missing_on_compare if compare_maint_variant == "pomcp" else 0),
        "missing_on_dqn": int(missing_on_primary if primary_maint_variant in {"flat_ddqn", "hier_ddqn"} else missing_on_compare if compare_maint_variant in {"flat_ddqn", "hier_ddqn"} else 0),
        "primary_metrics": {
            "tard": float(primary_metrics["tard"]),
            "maint": float(primary_metrics["maint"]),
            "total": float(primary_metrics["total"]),
            "overdue_ratio": float(primary_overdue["ratio_ops"]),
            "breakdown_count": float(getattr(primary_result.get("env"), "breakdown_count", 0)),
            "breakdown_cost": float(getattr(primary_result.get("env"), "breakdown_cost_total", 0.0)),
            "requeued_op_count": float(getattr(primary_result.get("env"), "requeued_op_count", 0)),
            "interrupted_proc_time": float(getattr(primary_result.get("env"), "interrupted_proc_time", 0.0)),
            "rule_unique_count": float(primary_schedule_summary.get("rule_unique_count", 0)),
            "rule_top1_share": float(primary_schedule_summary.get("rule_top1_share", 0.0)),
            "rule_distinct_count": float(primary_rule_diversity["rule_distinct_count"]),
            "rule_avg_dominant_share": float(primary_rule_diversity["rule_avg_dominant_share"]),
            "rule_mean_pairwise_jsd": float(primary_rule_diversity["rule_mean_pairwise_jsd"]),
            "flat_dqn_cm_remap_count": float(sum(1 for row in primary_rows if bool(row.get("flat_dqn_cm_remap")))),
            "pomcp_cm_block_count": float(sum(1 for row in primary_rows if bool(row.get("pomcp_cm_blocked")))),
            "pomcp_force_cm_count": float(sum(1 for row in primary_rows if bool(row.get("pomcp_force_cm")))),
            "pomcp_dn_conservative_veto_count": float(sum(1 for row in primary_rows if bool(row.get("pomcp_dn_conservative_veto")))),
            "sched_safe_block_count": float(primary_schedule_summary.get("sched_safe_block_count", 0)),
            **primary_final_health,
        },
        "compare_metrics": {
            "tard": float(compare_metrics["tard"]),
            "maint": float(compare_metrics["maint"]),
            "total": float(compare_metrics["total"]),
            "overdue_ratio": float(compare_overdue["ratio_ops"]),
            "breakdown_count": float(getattr(compare_result.get("env"), "breakdown_count", 0)),
            "breakdown_cost": float(getattr(compare_result.get("env"), "breakdown_cost_total", 0.0)),
            "requeued_op_count": float(getattr(compare_result.get("env"), "requeued_op_count", 0)),
            "interrupted_proc_time": float(getattr(compare_result.get("env"), "interrupted_proc_time", 0.0)),
            "rule_unique_count": float(compare_schedule_summary.get("rule_unique_count", 0)),
            "rule_top1_share": float(compare_schedule_summary.get("rule_top1_share", 0.0)),
            "rule_distinct_count": float(compare_rule_diversity["rule_distinct_count"]),
            "rule_avg_dominant_share": float(compare_rule_diversity["rule_avg_dominant_share"]),
            "rule_mean_pairwise_jsd": float(compare_rule_diversity["rule_mean_pairwise_jsd"]),
            "flat_dqn_cm_remap_count": float(sum(1 for row in compare_rows if bool(row.get("flat_dqn_cm_remap")))),
            "pomcp_cm_block_count": float(sum(1 for row in compare_rows if bool(row.get("pomcp_cm_blocked")))),
            "pomcp_force_cm_count": float(sum(1 for row in compare_rows if bool(row.get("pomcp_force_cm")))),
            "pomcp_dn_conservative_veto_count": float(sum(1 for row in compare_rows if bool(row.get("pomcp_dn_conservative_veto")))),
            "sched_safe_block_count": float(compare_schedule_summary.get("sched_safe_block_count", 0)),
            **compare_final_health,
        },
        "delta_compare_minus_primary": {
            "tard": float(compare_metrics["tard"] - primary_metrics["tard"]),
            "maint": float(compare_metrics["maint"] - primary_metrics["maint"]),
            "total": float(compare_metrics["total"] - primary_metrics["total"]),
            "overdue_ratio": float(compare_overdue["ratio_ops"] - primary_overdue["ratio_ops"]),
            "breakdown_count": float(getattr(compare_result.get("env"), "breakdown_count", 0) - getattr(primary_result.get("env"), "breakdown_count", 0)),
            "breakdown_cost": float(getattr(compare_result.get("env"), "breakdown_cost_total", 0.0) - getattr(primary_result.get("env"), "breakdown_cost_total", 0.0)),
            "requeued_op_count": float(getattr(compare_result.get("env"), "requeued_op_count", 0) - getattr(primary_result.get("env"), "requeued_op_count", 0)),
            "interrupted_proc_time": float(getattr(compare_result.get("env"), "interrupted_proc_time", 0.0) - getattr(primary_result.get("env"), "interrupted_proc_time", 0.0)),
            "sched_safe_block_count": float(compare_schedule_summary.get("sched_safe_block_count", 0) - primary_schedule_summary.get("sched_safe_block_count", 0)),
            "pomcp_dn_conservative_veto_count": float(
                sum(1 for row in compare_rows if bool(row.get("pomcp_dn_conservative_veto")))
                - sum(1 for row in primary_rows if bool(row.get("pomcp_dn_conservative_veto")))
            ),
            "rule_distinct_count": float(compare_rule_diversity["rule_distinct_count"] - primary_rule_diversity["rule_distinct_count"]),
            "rule_avg_dominant_share": float(compare_rule_diversity["rule_avg_dominant_share"] - primary_rule_diversity["rule_avg_dominant_share"]),
            "rule_mean_pairwise_jsd": float(compare_rule_diversity["rule_mean_pairwise_jsd"] - primary_rule_diversity["rule_mean_pairwise_jsd"]),
            "final_health_mean": float(compare_final_health["final_health_mean"] - primary_final_health["final_health_mean"]),
            "final_health_min": float(compare_final_health["final_health_min"] - primary_final_health["final_health_min"]),
            "final_health_p25": float(compare_final_health["final_health_p25"] - primary_final_health["final_health_p25"]),
            "final_low_health_count_h20": float(compare_final_health["final_low_health_count_h20"] - primary_final_health["final_low_health_count_h20"]),
            "final_low_health_count_h10": float(compare_final_health["final_low_health_count_h10"] - primary_final_health["final_low_health_count_h10"]),
        },
        "primary_schedule_summary": primary_schedule_summary,
        "compare_schedule_summary": compare_schedule_summary,
        "delta_schedule_summary": {
            "dispatch_count": int(compare_schedule_summary["dispatch_count"] - primary_schedule_summary["dispatch_count"]),
            "scheduling_events": int(compare_schedule_summary["scheduling_events"] - primary_schedule_summary["scheduling_events"]),
            "sched_safe_block_count": int(compare_schedule_summary["sched_safe_block_count"] - primary_schedule_summary["sched_safe_block_count"]),
            "breakdown_count": int(compare_schedule_summary["breakdown_count"] - primary_schedule_summary["breakdown_count"]),
            "hard_breakdown_count": int(compare_schedule_summary["hard_breakdown_count"] - primary_schedule_summary["hard_breakdown_count"]),
            "stochastic_breakdown_count": int(compare_schedule_summary["stochastic_breakdown_count"] - primary_schedule_summary["stochastic_breakdown_count"]),
            "requeued_op_count": int(compare_schedule_summary["requeued_op_count"] - primary_schedule_summary["requeued_op_count"]),
            "interrupted_proc_time": float(compare_schedule_summary["interrupted_proc_time"] - primary_schedule_summary["interrupted_proc_time"]),
            "breakdown_cost": float(compare_schedule_summary["breakdown_cost"] - primary_schedule_summary["breakdown_cost"]),
            "current_stress_mean": float(compare_schedule_summary["current_stress_mean"] - primary_schedule_summary["current_stress_mean"]),
            "current_stress_max": float(compare_schedule_summary["current_stress_max"] - primary_schedule_summary["current_stress_max"]),
            "makespan": float(compare_schedule_summary["makespan"] - primary_schedule_summary["makespan"]),
        },
    }
    return summary, rows


def write_mode_comparison_outputs(
    outdir: Path,
    stem: str,
    summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
    policy_label: str,
):
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows}
    json_path = outdir / f"{stem}.json"
    csv_path = outdir / f"{stem}.csv"
    summary_csv_path = outdir / f"{stem}_summary.csv"
    png_path = outdir / f"{stem}.png"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mid",
                "maint_seq_machine",
                "status",
                "primary_mode",
                "compare_mode",
                "primary_action",
                "compare_action",
                "primary_time",
                "compare_time",
                "primary_h",
                "compare_h",
                "primary_duration",
                "compare_duration",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    summary_row = {
        "primary_mode": summary.get("primary_mode"),
        "compare_mode": summary.get("compare_mode"),
        "primary_maint_variant": summary.get("primary_maint_variant"),
        "compare_maint_variant": summary.get("compare_maint_variant"),
        "primary_maint_arch": summary.get("primary_maint_arch"),
        "compare_maint_arch": summary.get("compare_maint_arch"),
        "compare_type": summary.get("compare_type"),
        "train_policy_tag": summary.get("train_policy_tag"),
        "eval_policy_tag": summary.get("eval_policy_tag"),
        "scheduler_anchor": summary.get("scheduler_anchor"),
        "primary_scheduler_mode": summary.get("primary_scheduler_mode"),
        "compare_scheduler_mode": summary.get("compare_scheduler_mode"),
        "primary_sched_regime_feature_mode": summary.get("primary_sched_regime_feature_mode"),
        "compare_sched_regime_feature_mode": summary.get("compare_sched_regime_feature_mode"),
        "divergence_count": summary.get("divergence_count"),
        "decision_union_count": summary.get("decision_union_count"),
        "divergence_rate": summary.get("divergence_rate"),
        "primary_im_invalid_filtered_count": summary.get("primary_im_invalid_filtered_count"),
        "compare_im_invalid_filtered_count": summary.get("compare_im_invalid_filtered_count"),
        "primary_dn_veto_count": summary.get("primary_dn_veto_count"),
        "compare_dn_veto_count": summary.get("compare_dn_veto_count"),
        "primary_tard": summary.get("primary_metrics", {}).get("tard"),
        "compare_tard": summary.get("compare_metrics", {}).get("tard"),
        "primary_maint": summary.get("primary_metrics", {}).get("maint"),
        "compare_maint": summary.get("compare_metrics", {}).get("maint"),
        "primary_total": summary.get("primary_metrics", {}).get("total"),
        "compare_total": summary.get("compare_metrics", {}).get("total"),
        "primary_overdue_ratio": summary.get("primary_metrics", {}).get("overdue_ratio"),
        "compare_overdue_ratio": summary.get("compare_metrics", {}).get("overdue_ratio"),
        "primary_dispatch_count": summary.get("primary_schedule_summary", {}).get("dispatch_count"),
        "compare_dispatch_count": summary.get("compare_schedule_summary", {}).get("dispatch_count"),
        "primary_sched_safe_block_count": summary.get("primary_schedule_summary", {}).get("sched_safe_block_count"),
        "compare_sched_safe_block_count": summary.get("compare_schedule_summary", {}).get("sched_safe_block_count"),
        "primary_current_stress_mean": summary.get("primary_schedule_summary", {}).get("current_stress_mean"),
        "compare_current_stress_mean": summary.get("compare_schedule_summary", {}).get("current_stress_mean"),
        "primary_current_stress_max": summary.get("primary_schedule_summary", {}).get("current_stress_max"),
        "compare_current_stress_max": summary.get("compare_schedule_summary", {}).get("current_stress_max"),
        "primary_makespan": summary.get("primary_schedule_summary", {}).get("makespan"),
        "compare_makespan": summary.get("compare_schedule_summary", {}).get("makespan"),
    }
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    plot_maint_mode_comparison(summary, rows, str(png_path), policy_label=policy_label)
