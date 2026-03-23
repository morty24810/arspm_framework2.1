from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

def _annotate_policy(ax, policy_label: Optional[str], extra_note: Optional[str] = None):
    if not policy_label and not extra_note:
        return
    lines = []
    if policy_label:
        lines.append(policy_label)
    if extra_note:
        lines.append(extra_note)
    ax.text(
        0.01,
        0.99,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

def plot_gantt(timeline_ops, timeline_maint, jobs, out_path: str,
               schedule: Optional[List[Tuple[float, float, float, float]]] = None,
               policy_label: Optional[str] = None):
    # machines on y axis
    mids = sorted(set([m for m, *_ in timeline_ops] + [m for m, *_ in timeline_maint]))
    if not mids:
        return
    fig, ax = plt.subplots(figsize=(12, 1 + 0.6*len(mids)))

    job_ids = sorted({jid for _, _, _, jid, _, *_ in timeline_ops})
    cmap = plt.get_cmap("tab20")
    job_color = {jid: cmap(i % cmap.N) for i, jid in enumerate(job_ids)}
    tardy_jobs = set()
    if jobs:
        for jid, job in jobs.items():
            if job.completed and job.completion_time is not None and job.completion_time > job.due:
                tardy_jobs.add(jid)

    ytick = []
    yticklabel = []

    t_max = 0.0
    for _, t0, t1, _, _, *_ in timeline_ops:
        t_max = max(t_max, t1)
    for _, t0, t1, _ in timeline_maint:
        t_max = max(t_max, t1)

    combo_colors = {}
    combo_order = []
    combo_cmap = plt.get_cmap("Pastel1")
    if schedule:
        for entry in schedule:
            if len(entry) == 5:
                t0, t1, lam, ddt, _level = entry
            else:
                t0, t1, lam, ddt = entry
            lam_val = float(lam)
            ddt_val = float(ddt)
            combo_key = (round(lam_val, 2), round(ddt_val, 2))
            if combo_key not in combo_colors:
                color = combo_cmap(len(combo_colors) % combo_cmap.N)
                combo_colors[combo_key] = color
                combo_order.append(combo_key)
            color = combo_colors[combo_key]
            if t0 >= t_max:
                continue
            t_end = min(t1, t_max)
            ax.axvspan(t0, t_end, facecolor=color, alpha=0.40, zorder=0)

    maint_colors = {
        "IM": "#BBDEFB",
        "CM": "#CFD8DC",
        "FAIL_CM": "#EF9A9A",
        "SCRAP": "#E57373",
        "BREAKDOWN": "#EF9A9A",
    }
    maint_kinds = set(k for _, _, _, k in timeline_maint)
    has_interrupted_ops = any((seg[5] if len(seg) > 5 else "DONE") == "INTERRUPTED" for seg in timeline_ops)

    for i, mid in enumerate(mids):
        y = i * 10
        ytick.append(y + 4)
        yticklabel.append(f"M{mid}")

        # operations
        for m, t0, t1, jid, oid, *rest in timeline_ops:
            if m != mid: 
                continue
            status = rest[0] if rest else "DONE"
            face = job_color.get(jid, (0.6, 0.6, 0.6, 1.0))
            edge = "red" if jid in tardy_jobs else "black"
            lw = 1.4 if jid in tardy_jobs else 0.8
            hatch = "xx" if status == "INTERRUPTED" else None
            alpha = 0.55 if status == "INTERRUPTED" else 1.0
            ax.broken_barh([(t0, t1-t0)], (y, 8), facecolors=face, edgecolors=edge, linewidth=lw, hatch=hatch, alpha=alpha)

        # maintenance blocks
        for m, t0, t1, kind in timeline_maint:
            if m != mid:
                continue
            face = maint_colors.get(kind, "#BDBDBD")
            ax.broken_barh([(t0, t1-t0)], (y, 8), facecolors=face, edgecolors="black",
                           hatch='//', linewidth=0.8)

    ax.set_yticks(ytick)
    ax.set_yticklabels(yticklabel)
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart (Operations + Maintenance)")
    ax.grid(True, axis="x", alpha=0.3)
    _annotate_policy(ax, policy_label)
    schedule_legend = None
    if schedule and combo_order:
        from matplotlib.patches import Patch
        legend_elems = [
            Patch(
                facecolor=combo_colors[key],
                edgecolor="none",
                label=f"lam={key[0]:.0f}, DDT={key[1]:.2f}"
            )
            for key in combo_order
        ]
        schedule_legend = ax.legend(handles=legend_elems, loc="upper right", ncol=3, frameon=False)
        ax.add_artist(schedule_legend)
    if maint_kinds:
        from matplotlib.patches import Patch
        maint_legend = [
            Patch(facecolor=maint_colors.get(kind, "#BDBDBD"), edgecolor="black", hatch='//', label=kind)
            for kind in sorted(maint_kinds)
        ]
        if has_interrupted_ops:
            maint_legend.append(Patch(facecolor="#9E9E9E", edgecolor="black", hatch="xx", alpha=0.55, label="INTERRUPTED_OP"))
        ax.legend(handles=maint_legend, loc="upper left", ncol=3, frameon=False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_rul_curves(rul_log: Dict[int, List[Tuple[float,float]]], maint: List[Tuple[int,float,float,str]],
                    Hx: float, Hy: float, out_path: str, p_fail_log: Optional[List[Tuple[float, float]]] = None,
                    policy_label: Optional[str] = None, threshold_enforced: bool = True):
    fig, ax = plt.subplots(figsize=(12,4))
    for mid, pts in rul_log.items():
        if not pts:
            continue
        t = [p[0] for p in pts]
        h = [p[1] for p in pts]
        ax.plot(t, h, label=f"M{mid}")
    ax.axhline(Hx, linestyle="--")
    ax.axhline(Hy, linestyle="--")

    # mark maintenance
    for mid, t0, t1, kind in maint:
        alert = kind in ("FAIL_CM", "SCRAP", "BREAKDOWN")
        color = "red" if alert else "gray"
        alpha = 0.3 if alert else 0.2
        ax.axvspan(t0, t1, alpha=alpha, color=color)
        if kind == "BREAKDOWN":
            ax.axvline(t0, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.8)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("RUL_norm")
    ax.set_title("RUL curves with maintenance windows")
    ax.grid(True, alpha=0.3)
    extra_note = None if threshold_enforced else "Hx/Hy lines shown as reference only (not enforced)."
    _annotate_policy(ax, policy_label, extra_note=extra_note)
    ax.legend()
    if p_fail_log:
        ax2 = ax.twinx()
        tpf = [x[0] for x in p_fail_log]
        pf = [x[1] for x in p_fail_log]
        ax2.plot(tpf, pf, color="#d62728", alpha=0.6, linestyle="--", label="p_fail")
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel("p_fail")
        ax2.legend(loc="upper right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_rule_vs_features(rule_log, out_path: str, policy_label: Optional[str] = None):
    # rule_log: list of (t, state_vec, goal, rule)
    if not rule_log:
        return
    t = np.array([x[0] for x in rule_log], dtype=np.float32)
    S = np.stack([x[1] for x in rule_log])
    goals = np.array([x[2] for x in rule_log], dtype=np.int64)
    rules = np.array([x[3] for x in rule_log], dtype=np.int64)

    # choose two interpretable features: arrivals_in_window and avg_slack
    arrivals = S[:,3]
    avg_slack = S[:,6]

    out_path = Path(out_path)

    fig, ax = plt.subplots(figsize=(10,4))
    sc = ax.scatter(arrivals, avg_slack, c=rules, s=18)
    ax.set_xlabel("arrivals_in_window")
    ax.set_ylabel("avg_slack")
    ax.set_title("Rule selection over (arrivals, avg_slack)")
    _annotate_policy(ax, policy_label)
    fig.colorbar(sc, ax=ax, label="rule_id")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,4))
    sc = ax.scatter(arrivals, avg_slack, c=goals, s=18)
    ax.set_xlabel("arrivals_in_window")
    ax.set_ylabel("avg_slack")
    ax.set_title("Goal selection over (arrivals, avg_slack)")
    _annotate_policy(ax, policy_label)
    fig.colorbar(sc, ax=ax, label="goal_id")
    fig.tight_layout()
    suffix = out_path.suffix or ".png"
    if out_path.name.startswith("rule_vs_features"):
        goal_name = "goal_vs_features" + out_path.name[len("rule_vs_features"):]
    else:
        goal_name = f"goal_vs_features{suffix}"
    goal_path = out_path.with_name(goal_name)
    fig.savefig(str(goal_path), dpi=200)
    plt.close(fig)

def plot_training_curves(tard_list, maint_list, out_path: str, smooth_window: int = 10, stop_ep: Optional[int] = None):
    if not tard_list:
        return
    ep = np.arange(1, len(tard_list) + 1, dtype=np.int32)
    total = np.array(tard_list, dtype=np.float32) + np.array(maint_list, dtype=np.float32)

    def moving_avg(x, w):
        if w <= 1:
            return x
        w = min(w, len(x))
        kernel = np.ones(w, dtype=np.float32) / float(w)
        return np.convolve(x, kernel, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep, tard_list, color="#1f77b4", alpha=0.35, label="tardiness")
    ax.plot(ep, maint_list, color="#ff7f0e", alpha=0.35, label="maintenance")
    ax.plot(ep, total, color="#2ca02c", alpha=0.35, label="total")

    total_ma = moving_avg(total, smooth_window)
    if len(total_ma) > 0:
        ax.plot(ep[len(ep) - len(total_ma):], total_ma, color="#2ca02c", linewidth=2.0, label="total (MA)")

    if stop_ep is not None and 1 <= stop_ep <= len(ep):
        ax.axvline(stop_ep, color="red", linestyle="--", linewidth=1.0, label="early stop")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cost")
    ax.set_title("Training Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_maint_action_rates(dn_rates, im_rates, cm_rates, out_path: str, avg_im_counts: Optional[List[float]] = None):
    if not dn_rates:
        return
    ep = np.arange(1, len(dn_rates) + 1, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep, dn_rates, label="DN", color="#1f77b4")
    ax.plot(ep, im_rates, label="IM", color="#ff7f0e")
    ax.plot(ep, cm_rates, label="CM", color="#2ca02c")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Action Rate")
    ax.set_title("Maintenance Action Rates per Episode")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if avg_im_counts:
        ax2 = ax.twinx()
        ax2.plot(ep, avg_im_counts, label="avg_im_count", color="#9467bd", alpha=0.7)
        ax2.set_ylabel("Avg IM Count")
        ax2.legend(loc="upper right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_maint_vs_slack(points, out_path: str, policy_label: Optional[str] = None):
    if not points:
        return
    slack = np.array([p[0] for p in points], dtype=np.float32)
    action = np.array([p[1] for p in points], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(slack, action, c=action, s=18, cmap="tab10")
    ax.set_xlabel("slack_pressure")
    ax.set_ylabel("action")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["DN", "IM", "CM"])
    ax.set_title("Maintenance Action vs Slack Pressure")
    _annotate_policy(ax, policy_label)
    fig.colorbar(sc, ax=ax, label="action")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_maint_mode_comparison(summary: Dict[str, object], compare_rows: List[Dict[str, object]],
                               out_path: str, policy_label: Optional[str] = None):
    if not compare_rows:
        compare_rows = []
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.4])
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    modes = [str(summary.get("primary_mode", "POMCP")), str(summary.get("compare_mode", "DQN"))]
    action_order = ["DN", "IM", "CM"]
    primary_counts = [float(summary.get("primary_action_counts", {}).get(a, 0.0)) for a in action_order]
    compare_counts = [float(summary.get("compare_action_counts", {}).get(a, 0.0)) for a in action_order]
    x = np.arange(len(action_order), dtype=np.float32)
    width = 0.35
    ax_top.bar(x - width / 2.0, primary_counts, width=width, label=modes[0], color="#1f77b4")
    ax_top.bar(x + width / 2.0, compare_counts, width=width, label=modes[1], color="#ff7f0e")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(action_order)
    ax_top.set_ylabel("Decision Count")
    ax_top.set_title("Maintenance Action Distribution")
    ax_top.grid(True, axis="y", alpha=0.3)
    ax_top.legend()
    subtitle = (
        f"divergence={int(summary.get('divergence_count', 0))}/"
        f"{int(summary.get('decision_union_count', 0))} "
        f"({float(summary.get('divergence_rate', 0.0)):.3f})"
    )
    p_sched = summary.get("primary_schedule_summary", {}) or {}
    c_sched = summary.get("compare_schedule_summary", {}) or {}
    if p_sched or c_sched:
        subtitle += (
            f" | dispatch={int(p_sched.get('dispatch_count', 0))}/"
            f"{int(c_sched.get('dispatch_count', 0))}"
            f" makespan={float(p_sched.get('makespan', 0.0)):.1f}/"
            f"{float(c_sched.get('makespan', 0.0)):.1f}"
        )
    _annotate_policy(ax_top, policy_label, extra_note=subtitle)

    ax_bottom.axis("off")
    ax_bottom.set_title("Decision Diff (aligned by machine + maintenance sequence)")
    rows_sorted = sorted(
        compare_rows,
        key=lambda r: (
            int(r.get("mid", -1)),
            int(r.get("maint_seq_machine", -1)),
        ),
    )
    lines = []
    max_rows = 18
    if p_sched or c_sched:
        lines.append(
            f"Scheduling: {modes[0]} dispatch={int(p_sched.get('dispatch_count', 0))}, makespan={float(p_sched.get('makespan', 0.0)):.1f}"
            f" | {modes[1]} dispatch={int(c_sched.get('dispatch_count', 0))}, makespan={float(c_sched.get('makespan', 0.0)):.1f}"
        )
        lines.append(
            f"Breakdown: {modes[0]} count={int(p_sched.get('breakdown_count', 0))}, hard={int(p_sched.get('hard_breakdown_count', 0))}, requeue={int(p_sched.get('requeued_op_count', 0))}"
            f" | {modes[1]} count={int(c_sched.get('breakdown_count', 0))}, hard={int(c_sched.get('hard_breakdown_count', 0))}, requeue={int(c_sched.get('requeued_op_count', 0))}"
        )
        lines.append(
            f"Rules: {modes[0]}={p_sched.get('rule_counts', {})} | {modes[1]}={c_sched.get('rule_counts', {})}"
        )
        lines.append(
            f"Goals: {modes[0]}={p_sched.get('goal_counts', {})} | {modes[1]}={c_sched.get('goal_counts', {})}"
        )
    for row in rows_sorted[:max_rows]:
        mid = int(row.get("mid", -1))
        seq = int(row.get("maint_seq_machine", -1))
        status = str(row.get("status", "unknown"))
        p_action = row.get("primary_action")
        c_action = row.get("compare_action")
        p_time = row.get("primary_time")
        c_time = row.get("compare_time")
        p_h = row.get("primary_h")
        c_h = row.get("compare_h")
        p_dur = row.get("primary_duration")
        c_dur = row.get("compare_duration")
        lines.append(
            f"M{mid}#{seq:02d} {status:<16} "
            f"{modes[0]}={p_action}@t={p_time},h={p_h},d={p_dur} | "
            f"{modes[1]}={c_action}@t={c_time},h={c_h},d={c_dur}"
        )
    if len(rows_sorted) > max_rows:
        lines.append(f"... {len(rows_sorted) - max_rows} more rows in CSV/JSON")
    if not lines:
        lines = ["No maintenance decisions recorded for comparison."]
    ax_bottom.text(
        0.01,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=9,
        transform=ax_bottom.transAxes,
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_im_damage(im_damage_log: Dict[int, List[Tuple[float, float]]], out_path: str):
    if not im_damage_log:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for mid, pts in im_damage_log.items():
        if not pts:
            continue
        t = [p[0] for p in pts]
        dmg = [p[1] for p in pts]
        ax.plot(t, dmg, label=f"M{mid}")
    ax.set_xlabel("Time")
    ax.set_ylabel("IM_damage")
    ax.set_title("IM Damage over Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
