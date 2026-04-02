from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch


CARD_BBOX = {
    "boxstyle": "round,pad=0.45,rounding_size=0.14",
    "facecolor": "#FFFFFF",
    "alpha": 0.97,
    "edgecolor": "#C9CCCF",
    "linewidth": 0.9,
}


def _compact_policy_parts(policy_label: Optional[str]) -> List[str]:
    if not policy_label:
        return []
    parts = [p.strip() for p in str(policy_label).split("|") if p.strip()]
    compact: List[str] = []
    for part in parts:
        if part.startswith("Policy:"):
            body = part.replace("Policy:", "", 1).strip()
            if "Region constrained" in body:
                compact.append("Restricted")
            elif "Unrestricted" in body:
                compact.append("Unrestricted")
            else:
                compact.append(body)
        elif part.startswith("Maintenance:"):
            compact.append(part.replace("Maintenance:", "", 1).strip())
        elif part.startswith("Compare:"):
            body = part.replace("Compare:", "", 1).strip().replace("_", " ")
            compact.append(body.title())
        elif part.startswith("Scheduler anchor:"):
            compact.append(part.replace("Scheduler anchor:", "", 1).strip())
        else:
            compact.append(part)
    return compact


def _add_card_axes(fig, bounds: Tuple[float, float, float, float]):
    ax = fig.add_axes(bounds, zorder=10)
    ax.set_axis_off()
    patch = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=0.9,
        edgecolor="#C9CCCF",
        facecolor="#FFFFFF",
        alpha=0.97,
    )
    ax.add_patch(patch)
    return ax


def _draw_badge(ax, x: float, y: float, text: str, *, face: str = "#EFF3F6", edge: str = "#CAD3DB"):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.7,
        color="#2E3A43",
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.16",
            "facecolor": face,
            "edgecolor": edge,
            "linewidth": 0.8,
        },
    )


def _draw_badge_small(ax, x: float, y: float, text: str, *, face: str = "#EFF3F6", edge: str = "#CAD3DB"):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.2,
        color="#2E3A43",
        bbox={
            "boxstyle": "round,pad=0.24,rounding_size=0.14",
            "facecolor": face,
            "edgecolor": edge,
            "linewidth": 0.75,
        },
    )


def _draw_info_card(fig, policy_label: Optional[str], extra_note: Optional[str] = None):
    parts = _compact_policy_parts(policy_label)
    notes = [chunk.strip() for chunk in str(extra_note or "").split("\n") if chunk.strip()]
    if not parts and not notes:
        return
    note_rows = max(1, len(notes)) if notes else 0
    height = 0.065 + 0.042 * note_rows
    ax = _add_card_axes(fig, (0.06, 0.05, 0.36, height))

    badge_specs = []
    palette = [
        ("#EEF4FB", "#C7D8EA"),
        ("#F2F4F7", "#D8DEE4"),
        ("#EDF6EC", "#C7DEC3"),
    ]
    for idx, part in enumerate(parts[:3]):
        face, edge = palette[idx % len(palette)]
        badge_specs.append((part, face, edge))

    x = 0.05
    for text, face, edge in badge_specs:
        _draw_badge(ax, x, 0.70 if notes else 0.52, text, face=face, edge=edge)
        x += 0.13 + min(0.26, 0.013 * len(text))

    if notes:
        y = 0.30
        for note in notes:
            ax.text(
                0.05,
                y,
                note,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=8.7,
                color="#5A6269",
            )
            y -= 0.24


def _draw_compare_card(fig, title: str, rows: List[Tuple[str, str]]):
    height = 0.16
    ax = _add_card_axes(fig, (0.06, 0.05, 0.58, height))
    parts = _compact_policy_parts(title)
    title_y = 0.82
    if parts:
        x = 0.05
        palette = [
            ("#EEF4FB", "#C7D8EA"),
            ("#F2F4F7", "#D8DEE4"),
            ("#EDF6EC", "#C7DEC3"),
        ]
        for idx, part in enumerate(parts[:3]):
            face, edge = palette[idx % len(palette)]
            _draw_badge_small(ax, x, title_y, part, face=face, edge=edge)
            x += 0.08 + min(0.20, 0.009 * len(part))
    else:
        ax.text(0.05, title_y, title, transform=ax.transAxes, ha="left", va="center",
                fontsize=9.2, color="#24323D", fontweight="semibold")

    columns = [
        (0.05, 0.17),
        (0.36, 0.49),
        (0.67, 0.82),
    ]
    y = 0.28
    for (key, value), (xk, xv) in zip(rows, columns):
        ax.text(xk, y, key, transform=ax.transAxes, ha="left", va="center",
                fontsize=8.3, color="#6A737B", fontweight="semibold")
        ax.text(xv, y, value, transform=ax.transAxes, ha="left", va="center",
                fontsize=8.7, color="#24323D")


def _draw_schedule_table(fig, combo_order, combo_colors):
    if not combo_order:
        return
    rows = len(combo_order)
    height = 0.16 + 0.078 * rows
    ax = _add_card_axes(fig, (0.79, 0.31, 0.18, min(height, 0.60)))
    ax.text(0.08, 0.90, "Load Settings", transform=ax.transAxes, ha="left", va="center",
            fontsize=9.0, color="#24323D", fontweight="semibold")
    ax.text(0.08, 0.74, "", transform=ax.transAxes)
    ax.text(0.23, 0.72, "λ", transform=ax.transAxes, ha="left", va="center",
            fontsize=8.6, color="#6A737B", fontweight="semibold")
    ax.text(0.39, 0.72, "DDT", transform=ax.transAxes, ha="left", va="center",
            fontsize=8.6, color="#6A737B", fontweight="semibold")
    y = 0.64
    for key in combo_order:
        ax.add_patch(
            FancyBboxPatch((0.08, y - 0.035), 0.06, 0.07, transform=ax.transAxes,
                           boxstyle="round,pad=0.01,rounding_size=0.01",
                           linewidth=0.0, facecolor=combo_colors[key], alpha=0.9)
        )
        ax.text(0.23, y, f"{key[0]:.0f}", transform=ax.transAxes, ha="left", va="center",
                fontsize=8.6, color="#24323D")
        ax.text(0.39, y, f"{key[1]:.2f}", transform=ax.transAxes, ha="left", va="center",
                fontsize=8.6, color="#24323D")
        y -= 0.11


def _draw_named_legend_card(fig, title: str, items: List[Tuple[str, str, Optional[str]]], bounds: Tuple[float, float, float, float]):
    if not items:
        return
    ax = _add_card_axes(fig, bounds)
    ax.text(0.08, 0.84, title, transform=ax.transAxes, ha="left", va="center",
            fontsize=9.0, color="#24323D", fontweight="semibold")
    y = 0.62
    for label, face, hatch in items:
        ax.add_patch(
            FancyBboxPatch((0.08, y - 0.045), 0.10, 0.09, transform=ax.transAxes,
                           boxstyle="round,pad=0.01,rounding_size=0.01",
                           linewidth=0.8, edgecolor="#313131", facecolor=face,
                           hatch=hatch or "")
        )
        ax.text(0.24, y, label, transform=ax.transAxes, ha="left", va="center",
                fontsize=8.5, color="#24323D")
        y -= 0.18


def _legend_outside(ax, handles, title: Optional[str] = None, *, anchor_y: float = 1.0):
    if not handles:
        return None
    return ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, anchor_y),
        borderaxespad=0.0,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        title=title,
    )


def _scatter_legend(ax, labels: List[str], colors: List[str], title: str, *, anchor_y: float = 1.0):
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="none",
               markerfacecolor=color, markeredgecolor="#4A4A4A", markersize=7, label=label)
        for label, color in zip(labels, colors)
    ]
    return _legend_outside(ax, handles, title=title, anchor_y=anchor_y)


def plot_gantt(timeline_ops, timeline_maint, jobs, out_path: str,
               schedule: Optional[List[Tuple[float, float, float, float]]] = None,
               policy_label: Optional[str] = None):
    mids = sorted(set([m for m, *_ in timeline_ops] + [m for m, *_ in timeline_maint]))
    if not mids:
        return
    fig, ax = plt.subplots(figsize=(14, 1 + 0.6 * len(mids)))

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
    for _, _, t1, _, _, *_ in timeline_ops:
        t_max = max(t_max, t1)
    for _, _, t1, _ in timeline_maint:
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
            combo_key = (round(float(lam), 2), round(float(ddt), 2))
            if combo_key not in combo_colors:
                combo_colors[combo_key] = combo_cmap(len(combo_colors) % combo_cmap.N)
                combo_order.append(combo_key)
            if t0 >= t_max:
                continue
            ax.axvspan(t0, min(t1, t_max), facecolor=combo_colors[combo_key], alpha=0.38, zorder=0)

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

        for m, t0, t1, jid, oid, *rest in timeline_ops:
            if m != mid:
                continue
            status = rest[0] if rest else "DONE"
            face = job_color.get(jid, (0.6, 0.6, 0.6, 1.0))
            edge = "red" if jid in tardy_jobs else "black"
            lw = 1.4 if jid in tardy_jobs else 0.8
            hatch = "xx" if status == "INTERRUPTED" else None
            alpha = 0.55 if status == "INTERRUPTED" else 1.0
            ax.broken_barh([(t0, t1 - t0)], (y, 8), facecolors=face, edgecolors=edge, linewidth=lw, hatch=hatch, alpha=alpha)

        for m, t0, t1, kind in timeline_maint:
            if m != mid:
                continue
            ax.broken_barh(
                [(t0, t1 - t0)],
                (y, 8),
                facecolors=maint_colors.get(kind, "#BDBDBD"),
                edgecolors="black",
                hatch="//",
                linewidth=0.8,
            )

    ax.set_yticks(ytick)
    ax.set_yticklabels(yticklabel)
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart (Operations + Maintenance)")
    ax.grid(True, axis="x", alpha=0.3)

    maint_items = [
        (kind, maint_colors.get(kind, "#BDBDBD"), "//")
        for kind in sorted(maint_kinds)
    ]
    if has_interrupted_ops:
        maint_items.append(("INTERRUPTED", "#9E9E9E", "xx"))

    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.22, top=0.90)
    _draw_schedule_table(fig, combo_order, combo_colors)
    _draw_named_legend_card(fig, "Maintenance", maint_items, bounds=(0.79, 0.24, 0.18, 0.17 + 0.06 * len(maint_items)))
    _draw_info_card(fig, policy_label)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rul_curves(rul_log: Dict[int, List[Tuple[float, float]]], maint: List[Tuple[int, float, float, str]],
                    Hx: float, Hy: float, out_path: str, p_fail_log: Optional[List[Tuple[float, float]]] = None,
                    policy_label: Optional[str] = None, threshold_enforced: bool = True,
                    rul_obs_log: Optional[Dict[int, List[Tuple[float, float]]]] = None,
                    hard_threshold: Optional[float] = None,
                    rul_segments: Optional[Dict[int, List[Tuple[float, float, float, float, str]]]] = None):
    fig, ax = plt.subplots(figsize=(12, 4))
    has_flat_processing_segments = False
    for mid in sorted(rul_log.keys()):
        pts = rul_log.get(mid) or []
        segs = [] if rul_segments is None else list(rul_segments.get(mid) or [])
        color = None
        if segs:
            ordered_segs = sorted(segs, key=lambda x: (x[0], x[1]))
            ordered_pts = sorted(pts, key=lambda x: x[0])
            maint_spans = sorted([m for m in maint if int(m[0]) == int(mid)], key=lambda x: (x[1], x[2]))

            def _post_maint_h(t1: float, fallback: float) -> float:
                for tp, hp in ordered_pts:
                    if abs(float(tp) - float(t1)) <= 1e-6:
                        return float(hp)
                for tp, hp in ordered_pts:
                    if float(tp) >= float(t1) - 1e-6:
                        return float(hp)
                return float(fallback)

            for idx, (t0, t1, h0, h1, _) in enumerate(ordered_segs):
                kwargs = {"color": color} if color is not None else {}
                line, = ax.plot([t0, t1], [h0, h1], label=f"M{mid}" if idx == 0 else None, **kwargs)
                color = line.get_color()
                if abs(float(h1) - float(h0)) <= 1e-9:
                    has_flat_processing_segments = True
                    t_mid = 0.5 * (float(t0) + float(t1))
                    ax.scatter(
                        [t_mid],
                        [float(h0)],
                        marker="o",
                        s=18,
                        facecolors="white",
                        edgecolors=color,
                        linewidths=0.9,
                        alpha=0.9,
                        zorder=4,
                    )
                if idx > 0:
                    _, pt1, _, ph1, _ = ordered_segs[idx - 1]
                    gap_maint = [
                        (float(mt0), float(mt1), str(mkind))
                        for _, mt0, mt1, mkind in maint_spans
                        if float(mt0) >= float(pt1) - 1e-6 and float(mt1) <= float(t0) + 1e-6
                    ]
                    if not gap_maint:
                        ax.plot([pt1, t0], [ph1, h0], linestyle="--", linewidth=0.9, alpha=0.45, color=color)
                    else:
                        cursor_t = float(pt1)
                        cursor_h = float(ph1)
                        for mt0, mt1, _ in gap_maint:
                            if mt0 > cursor_t + 1e-6:
                                ax.plot([cursor_t, mt0], [cursor_h, cursor_h], linestyle="--", linewidth=0.9, alpha=0.45, color=color)
                            ax.plot([mt0, mt1], [cursor_h, cursor_h], linestyle="--", linewidth=0.9, alpha=0.55, color=color)
                            next_h = _post_maint_h(mt1, h0)
                            if abs(next_h - cursor_h) > 1e-6:
                                ax.plot([mt1, mt1], [cursor_h, next_h], linestyle="--", linewidth=0.9, alpha=0.55, color=color)
                            cursor_t = float(mt1)
                            cursor_h = float(next_h)
                        if t0 > cursor_t + 1e-6:
                            ax.plot([cursor_t, t0], [cursor_h, cursor_h], linestyle="--", linewidth=0.9, alpha=0.45, color=color)
            if rul_obs_log is not None:
                obs_pts = rul_obs_log.get(mid) or []
                if obs_pts:
                    tobs = [p[0] for p in obs_pts]
                    hobs = [p[1] for p in obs_pts]
                    ax.plot(tobs, hobs, linestyle=":", linewidth=0.9, alpha=0.35, color=color)
            continue
        if not pts:
            continue
        t = [p[0] for p in pts]
        h = [p[1] for p in pts]
        line, = ax.plot(t, h, label=f"M{mid}")
        if rul_obs_log is not None:
            obs_pts = rul_obs_log.get(mid) or []
            if obs_pts:
                tobs = [p[0] for p in obs_pts]
                hobs = [p[1] for p in obs_pts]
                ax.plot(tobs, hobs, linestyle="--", linewidth=1.0, alpha=0.5, color=line.get_color())
    ax.axhline(Hx, linestyle="--", color="#7F8C8D", linewidth=1.0)
    ax.axhline(Hy, linestyle="--", color="#95A5A6", linewidth=1.0)
    if hard_threshold is not None:
        ax.axhline(float(hard_threshold), linestyle="-.", color="#d62728", linewidth=1.0, alpha=0.85)

    for _, t0, t1, kind in maint:
        alert = kind in ("FAIL_CM", "SCRAP", "BREAKDOWN")
        ax.axvspan(t0, t1, alpha=0.28 if alert else 0.18, color="#E57373" if alert else "#B0BEC5")
        if kind == "BREAKDOWN":
            ax.axvline(t0, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.8)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("RUL_norm")
    ax.set_title("RUL Curves with Maintenance Windows")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)

    notes = []
    if rul_obs_log is not None:
        notes.append("Observed trace shown as dotted overlay.")
    if has_flat_processing_segments:
        notes.append("White markers indicate processing with unchanged canonical RUL.")
    if not threshold_enforced:
        notes.append("Hx/Hy shown as reference only.")
    _draw_info_card(fig, policy_label, extra_note="\n".join(notes) if notes else None)
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.24, top=0.88)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_categorical_scatter(x, y, categories, color_map: Dict[int, str], labels: Dict[int, str],
                              out_path: Path, *, xlabel: str, ylabel: str, title: str,
                              policy_label: Optional[str] = None, y_ticks=None, y_ticklabels=None,
                              legend_title: str = "Category"):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = [color_map[int(cat)] for cat in categories]
    ax.scatter(x, y, c=colors, s=24, alpha=0.85, edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    ax.grid(True, alpha=0.25)
    _scatter_legend(
        ax,
        [labels[k] for k in labels.keys()],
        [color_map[k] for k in labels.keys()],
        legend_title,
        anchor_y=1.0,
    )
    _draw_info_card(fig, policy_label)
    fig.subplots_adjust(left=0.10, right=0.80, bottom=0.22, top=0.88)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def plot_rule_vs_features(rule_log, out_path: str, policy_label: Optional[str] = None):
    if not rule_log:
        return
    S = np.stack([x[1] for x in rule_log])
    rules = np.array([x[3] for x in rule_log], dtype=np.int64)
    raw_goals = [x[2] for x in rule_log]
    valid_goal_mask = np.array(
        [isinstance(g, (int, np.integer)) and 0 <= int(g) <= 3 for g in raw_goals],
        dtype=bool,
    )
    arrivals = S[:, 3]
    avg_slack = S[:, 6]
    out_path = Path(out_path)

    rule_colors = {
        0: "#4E79A7",
        1: "#F28E2B",
        2: "#E15759",
        3: "#76B7B2",
        4: "#59A14F",
        5: "#EDC948",
    }
    goal_colors = {
        0: "#4E79A7",
        1: "#E15759",
        2: "#59A14F",
        3: "#B07AA1",
    }
    _plot_categorical_scatter(
        arrivals,
        avg_slack,
        rules,
        rule_colors,
        {i: f"Rule {i}" for i in range(6)},
        out_path,
        xlabel="Arrivals in Window",
        ylabel="Average Slack",
        title="Dispatch Rule by Load and Slack",
        policy_label=policy_label,
        legend_title="Dispatch Rule",
    )

    if np.any(valid_goal_mask):
        goals = np.array([int(raw_goals[i]) for i in range(len(raw_goals)) if valid_goal_mask[i]], dtype=np.int64)
        suffix = out_path.suffix or ".png"
        goal_name = "goal_vs_features" + out_path.name[len("rule_vs_features"):] if out_path.name.startswith("rule_vs_features") else f"goal_vs_features{suffix}"
        goal_path = out_path.with_name(goal_name)
        _plot_categorical_scatter(
            arrivals[valid_goal_mask],
            avg_slack[valid_goal_mask],
            goals,
            goal_colors,
            {i: f"Goal {i}" for i in range(4)},
            goal_path,
            xlabel="Arrivals in Window",
            ylabel="Average Slack",
            title="Scheduler Goal by Load and Slack",
            policy_label=policy_label,
            legend_title="Goal",
        )


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
    color_map = {0: "#4E79A7", 1: "#F28E2B", 2: "#59A14F"}
    labels = {0: "DN", 1: "IM", 2: "CM"}
    _plot_categorical_scatter(
        slack,
        action,
        action,
        color_map,
        labels,
        Path(out_path),
        xlabel="Slack Pressure",
        ylabel="Maintenance Action",
        title="Maintenance Action by Slack Pressure",
        policy_label=policy_label,
        y_ticks=[0, 1, 2],
        y_ticklabels=["DN", "IM", "CM"],
        legend_title="Action",
    )


def plot_maint_mode_comparison(summary: Dict[str, object], compare_rows: List[Dict[str, object]],
                               out_path: str, policy_label: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(12, 5.2))

    modes = [str(summary.get("primary_mode", "POMCP")), str(summary.get("compare_mode", "DQN"))]
    action_order = ["DN", "IM", "CM"]
    primary_counts = [float(summary.get("primary_action_counts", {}).get(a, 0.0)) for a in action_order]
    compare_counts = [float(summary.get("compare_action_counts", {}).get(a, 0.0)) for a in action_order]
    x = np.arange(len(action_order), dtype=np.float32)
    width = 0.35
    ax.bar(x - width / 2.0, primary_counts, width=width, label=modes[0], color="#4E79A7")
    ax.bar(x + width / 2.0, compare_counts, width=width, label=modes[1], color="#F28E2B")
    ax.set_xticks(x)
    ax.set_xticklabels(action_order)
    ax.set_ylabel("Decision Count")
    ax.set_title("Maintenance Action Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=False)

    compare_type = str(summary.get("compare_type", "full_system"))
    title = policy_label or "Compare"
    if compare_type == "full_system_route_compare":
        title = policy_label or f"Route compare | Full system | Maintenance: {summary.get('maint_mode', '')}"

    p_sched = summary.get("primary_schedule_summary", {}) or {}
    c_sched = summary.get("compare_schedule_summary", {}) or {}
    rows = [
        ("divergence", f"{int(summary.get('divergence_count', 0))}/{int(summary.get('decision_union_count', 0))} ({float(summary.get('divergence_rate', 0.0)):.3f})"),
        ("dispatch", f"{int(p_sched.get('dispatch_count', 0))}/{int(c_sched.get('dispatch_count', 0))}"),
        ("makespan", f"{float(p_sched.get('makespan', 0.0)):.1f}/{float(c_sched.get('makespan', 0.0)):.1f}"),
    ]
    _draw_compare_card(fig, title, rows)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.26, top=0.88)
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
