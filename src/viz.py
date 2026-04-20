from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle


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


def _badge_width(text: str, *, small: bool = False) -> float:
    base = 0.14 if small else 0.18
    scale = 0.007 if small else 0.010
    cap = 0.32 if small else 0.40
    return min(cap, base + scale * len(str(text)))


def _draw_badge_rows(ax, items: List[Tuple[str, str, str]], *, start_y: float, max_width: float = 0.92,
                     left: float = 0.05, row_gap: float = 0.28, small: bool = False) -> int:
    x = left
    y = start_y
    rows = 1
    for text, face, edge in items:
        width = _badge_width(text, small=small)
        if x + width > max_width:
            rows += 1
            x = left
            y -= row_gap
        if small:
            _draw_badge_small(ax, x, y, text, face=face, edge=edge)
        else:
            _draw_badge(ax, x, y, text, face=face, edge=edge)
        x += width + 0.02
    return rows


def _draw_info_card(fig, policy_label: Optional[str], extra_note: Optional[str] = None,
                    *, bounds: Optional[Tuple[float, float, float, float]] = None):
    parts = _compact_policy_parts(policy_label)
    notes = [chunk.strip() for chunk in str(extra_note or "").split("\n") if chunk.strip()]
    if not parts and not notes:
        return

    palette = [
        ("#EEF4FB", "#C7D8EA"),
        ("#F2F4F7", "#D8DEE4"),
        ("#EDF6EC", "#C7DEC3"),
    ]
    badge_specs = []
    for idx, part in enumerate(parts[:3]):
        face, edge = palette[idx % len(palette)]
        badge_specs.append((part, face, edge))

    badge_rows = 0
    if badge_specs:
        row_width = 0.88
        used = 0.0
        badge_rows = 1
        for text, _, _ in badge_specs:
            width = _badge_width(text)
            if used > 0.0 and used + width > row_width:
                badge_rows += 1
                used = 0.0
            used += width + 0.02
    note_rows = len(notes)
    height = 0.06 + 0.048 * max(badge_rows, 1) + 0.042 * note_rows
    if bounds is None:
        bounds = (0.06, 0.03, 0.50, height)
    ax = _add_card_axes(fig, bounds)

    current_y = 0.70 if notes else 0.55
    if badge_specs:
        used_rows = _draw_badge_rows(ax, badge_specs, start_y=current_y, max_width=0.92, left=0.05, row_gap=0.30)
        current_y -= 0.30 * max(used_rows, 1)

    if notes:
        y = max(0.18, current_y + 0.02)
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
        palette = [
            ("#EEF4FB", "#C7D8EA"),
            ("#F2F4F7", "#D8DEE4"),
            ("#EDF6EC", "#C7DEC3"),
        ]
        badge_specs = []
        for idx, part in enumerate(parts[:3]):
            face, edge = palette[idx % len(palette)]
            badge_specs.append((part, face, edge))
        _draw_badge_rows(ax, badge_specs, start_y=title_y, max_width=0.94, left=0.05, row_gap=0.24, small=True)
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


def _draw_schedule_table(fig, combo_order, combo_colors, bounds: Tuple[float, float, float, float]):
    if not combo_order:
        return
    ax = _add_card_axes(fig, bounds)
    ax.text(0.08, 0.90, "Load Settings", transform=ax.transAxes, ha="left", va="center",
            fontsize=9.0, color="#24323D", fontweight="semibold")
    n_cols = 1 if len(combo_order) <= 5 else 2
    rows_per_col = int(np.ceil(len(combo_order) / n_cols))
    col_x = [0.08, 0.54]
    y_top = 0.74
    y_bottom = 0.18
    row_step = (y_top - y_bottom) / max(rows_per_col - 1, 1)
    box_h = max(0.05, min(0.08, row_step * 0.72 if rows_per_col > 1 else 0.08))
    for idx, key in enumerate(combo_order):
        col = min(idx // rows_per_col, n_cols - 1)
        row = idx % rows_per_col
        x0 = col_x[col]
        y = y_top - row * row_step
        ax.add_patch(
            FancyBboxPatch((x0, y - box_h / 2.0), 0.06, box_h, transform=ax.transAxes,
                           boxstyle="round,pad=0.01,rounding_size=0.01",
                           linewidth=0.0, facecolor=combo_colors[key], alpha=0.9)
        )
        ax.text(
            x0 + 0.09,
            y,
            f"λ {key[0]:.0f} | DDT {key[1]:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.4,
            color="#24323D",
        )


def _draw_named_legend_card(fig, title: str, items: List[Tuple[str, str, Optional[str]]], bounds: Tuple[float, float, float, float]):
    if not items:
        return
    ax = _add_card_axes(fig, bounds)
    ax.text(0.08, 0.84, title, transform=ax.transAxes, ha="left", va="center",
            fontsize=9.0, color="#24323D", fontweight="semibold")
    y_start, y_end = 0.62, 0.18
    step = (y_start - y_end) / max(len(items) - 1, 1)
    box_h = max(0.07, min(0.10, step * 0.72 if len(items) > 1 else 0.09))
    y = y_start
    for label, face, hatch in items:
        ax.add_patch(
            FancyBboxPatch((0.08, y - box_h / 2.0), 0.10, box_h, transform=ax.transAxes,
                           boxstyle="round,pad=0.01,rounding_size=0.01",
                           linewidth=0.8, edgecolor="#313131", facecolor=face,
                           hatch=hatch or "")
        )
        ax.text(0.24, y, label, transform=ax.transAxes, ha="left", va="center",
                fontsize=8.5, color="#24323D")
        y -= step


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
    fig_height = max(5.8, 2.4 + 0.72 * len(mids))
    fig, ax = plt.subplots(figsize=(14.5, fig_height))

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

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.30, top=0.90)
    _draw_schedule_table(fig, combo_order, combo_colors, bounds=(0.06, 0.06, 0.42, 0.18))
    _draw_named_legend_card(fig, "Maintenance", maint_items, bounds=(0.50, 0.06, 0.16, 0.18))
    _draw_info_card(fig, policy_label, bounds=(0.68, 0.06, 0.26, 0.18))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rul_curves(rul_log: Dict[int, List[Tuple[float, float]]], maint: List[Tuple[int, float, float, str]],
                    Hx: float, Hy: float, out_path: str, p_fail_log: Optional[List[Tuple[float, float]]] = None,
                    policy_label: Optional[str] = None, threshold_enforced: bool = True,
                    rul_obs_log: Optional[Dict[int, List[Tuple[float, float]]]] = None,
                    hard_threshold: Optional[float] = None,
                    rul_segments: Optional[Dict[int, List[Tuple[float, float, float, float, str]]]] = None):
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
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
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.30, top=0.88)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_categorical_scatter(x, y, categories, color_map: Dict[int, str], labels: Dict[int, str],
                              out_path: Path, *, xlabel: str, ylabel: str, title: str,
                              policy_label: Optional[str] = None, y_ticks=None, y_ticklabels=None,
                              legend_title: str = "Category"):
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
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
    fig.subplots_adjust(left=0.10, right=0.80, bottom=0.28, top=0.88)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def _combo_heatmap_annotations(values: List[int], allowed_values: List[int]) -> Tuple[Optional[int], float]:
    counts = {int(v): 0 for v in allowed_values}
    for value in values:
        if int(value) in counts:
            counts[int(value)] += 1
    total = sum(counts.values())
    if total <= 0:
        return None, 0.0
    dominant = max(counts.items(), key=lambda kv: kv[1])[0]
    share = counts[dominant] / max(total, 1)
    return int(dominant), float(share)


def _plot_combo_categorical_heatmap(
    lam_values: np.ndarray,
    ddt_values: np.ndarray,
    categories: np.ndarray,
    *,
    allowed_values: List[int],
    category_colors: Dict[int, str],
    category_labels: Dict[int, str],
    out_path: Path,
    title: str,
    legend_title: str,
    policy_label: Optional[str],
):
    uniq_lam = sorted({float(x) for x in lam_values.tolist()})
    uniq_ddt = sorted({float(x) for x in ddt_values.tolist()})
    if not uniq_lam or not uniq_ddt:
        return

    fig, ax = plt.subplots(figsize=(2.4 + 1.55 * len(uniq_ddt), 2.6 + 1.05 * len(uniq_lam)))
    ax.set_xlim(0, len(uniq_ddt))
    ax.set_ylim(0, len(uniq_lam))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(uniq_ddt)) + 0.5)
    ax.set_yticks(np.arange(len(uniq_lam)) + 0.5)
    ax.set_xticklabels([f"{x:.2f}" for x in uniq_ddt], fontsize=9)
    ax.set_yticklabels([f"{x:.0f}" for x in uniq_lam], fontsize=9)
    ax.set_xlabel("DDT", labelpad=6)
    ax.set_ylabel("λ")
    ax.set_title(title)
    ax.set_facecolor("#F7F8FA")

    for yi, lam in enumerate(uniq_lam):
        for xi, ddt in enumerate(uniq_ddt):
            mask = np.isclose(lam_values, lam) & np.isclose(ddt_values, ddt)
            cell_values = categories[mask]
            dominant, share = _combo_heatmap_annotations(cell_values.tolist(), allowed_values)
            if dominant is None:
                face = "#ECEFF2"
                label = "NA"
                share_text = ""
                alpha = 0.55
            else:
                face = category_colors[int(dominant)]
                label = category_labels[int(dominant)]
                share_text = f"{share * 100:.0f}%"
                alpha = 0.35 + 0.60 * share
            ax.add_patch(
                Rectangle(
                    (xi, yi),
                    1.0,
                    1.0,
                    facecolor=face,
                    edgecolor="#FFFFFF",
                    linewidth=1.5,
                    alpha=alpha,
                )
            )
            ax.text(
                xi + 0.5,
                yi + 0.42,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color="#1F2A33",
                fontweight="semibold",
            )
            if share_text:
                ax.text(
                    xi + 0.5,
                    yi + 0.70,
                    share_text,
                    ha="center",
                    va="center",
                    fontsize=8.6,
                    color="#30414D",
                )

    ax.set_aspect("equal")
    handles = [
        Patch(facecolor=category_colors[val], edgecolor="none", label=category_labels[val], alpha=0.75)
        for val in allowed_values
    ]
    _legend_outside(ax, handles, title=legend_title, anchor_y=1.0)
    fig.subplots_adjust(right=0.80, left=0.12, bottom=0.30, top=0.86)
    _draw_info_card(fig, policy_label, extra_note="Cells show dominant category and share by λ×DDT combo.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_rule_vs_features(rule_log, out_path: str, policy_label: Optional[str] = None):
    if not rule_log:
        return
    S = np.stack([x[1] for x in rule_log])
    sched_lambda = S[:, 4]
    sched_ddt = S[:, 5]
    rules = np.array([x[3] for x in rule_log], dtype=np.int64)
    raw_goals = [x[2] for x in rule_log]
    valid_goal_mask = np.array(
        [isinstance(g, (int, np.integer)) and 0 <= int(g) <= 3 for g in raw_goals],
        dtype=bool,
    )
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
    _plot_combo_categorical_heatmap(
        sched_lambda,
        sched_ddt,
        rules,
        allowed_values=list(range(6)),
        category_colors=rule_colors,
        category_labels={i: f"Rule {i}" for i in range(6)},
        out_path=out_path,
        title="Dispatch Rule by λ and DDT",
        legend_title="Dispatch Rule",
        policy_label=policy_label,
    )

    if np.any(valid_goal_mask):
        goals = np.array([int(raw_goals[i]) for i in range(len(raw_goals)) if valid_goal_mask[i]], dtype=np.int64)
        suffix = out_path.suffix or ".png"
        goal_name = "goal_vs_features" + out_path.name[len("rule_vs_features"):] if out_path.name.startswith("rule_vs_features") else f"goal_vs_features{suffix}"
        goal_path = out_path.with_name(goal_name)
        _plot_combo_categorical_heatmap(
            sched_lambda[valid_goal_mask],
            sched_ddt[valid_goal_mask],
            goals,
            allowed_values=list(range(4)),
            category_colors=goal_colors,
            category_labels={i: f"Goal {i}" for i in range(4)},
            out_path=goal_path,
            title="Scheduler Goal by λ and DDT",
            legend_title="Goal",
            policy_label=policy_label,
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

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
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
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_maint_action_rates(dn_rates, im_rates, cm_rates, out_path: str, avg_im_counts: Optional[List[float]] = None):
    if not dn_rates:
        return
    ep = np.arange(1, len(dn_rates) + 1, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.plot(ep, dn_rates, label="DN", color="#1f77b4")
    ax.plot(ep, im_rates, label="IM", color="#ff7f0e")
    ax.plot(ep, cm_rates, label="CM", color="#2ca02c")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Action Rate")
    ax.set_title("Maintenance Action Rates per Episode")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    if avg_im_counts:
        ax2 = ax.twinx()
        ax2.plot(ep, avg_im_counts, label="avg_im_count", color="#9467bd", alpha=0.7)
        ax2.set_ylabel("Avg IM Count")
        ax2.legend(loc="upper right", frameon=False)
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
    fig, ax = plt.subplots(figsize=(12.4, 5.6))

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
