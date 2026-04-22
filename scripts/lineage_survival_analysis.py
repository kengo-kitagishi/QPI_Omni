"""
lineage_survival_analysis.py — Mother fate analysis (Pos9/ch01 prototype).

Takes the outputs of central_cell_lineage_tracker.py (clist.csv + lineage_data3D.csv)
and classifies each in-tree cell by whether it resumed division in the recovery
epoch after a glucose-starvation switch. Writes:

  - survival_summary.csv (per-cell fate label + pre-starvation metrics)
  - figure 1: mother volume/mass trajectory with media epochs + division ticks
  - figure 2: fate-colored lineage tree
  - figure 3: pre-starvation volume/mass/RI vs fate (divided in recovery or not)
  - figure 4: recovery-lag histogram (cells that did divide in recovery)

Example:
    /opt/anaconda3/bin/python lineage_survival_analysis.py \\
        --indir /Volumes/2604/260405/ph_260405/Pos9/output_phase/channels/crop_sub_rawraw/ch01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_logger import save_figure

# =============================================================================
# Constants
# =============================================================================
# Experiment timing (Pos9/260405). Frames confirmed by user 2026-04-22:
# 575 -> starvation begins, 863 -> continued starvation (shown but unused for fate),
# 1439 -> 2% recovery.
FRAMES_PER_HOUR = 12.0
STARV_START_FRAME = 575    # 0% glucose begins
SWITCH_MID_FRAME = 863     # intermediate switch (drawn only, no fate effect)
REC_START_FRAME = 1439     # 2% glucose recovery begins

# Recovery slope fit: require at least this many non-outlier / non-border
# volume samples strictly after REC_START_FRAME.
REC_SLOPE_MIN_POINTS = 10

# Pre-starvation window (frames before STARV_START_FRAME used for baseline metrics)
PRE_WINDOW_FRAMES = 100    # ≈ 8.3 h

# Fate colors (Okabe-Ito). Survival is judged by recovery-phase elongation
# rate (µm/h). Interim threshold: ELONG_ALIVE_THR = 0.3 µm/h. This dataset
# does not contain clearly non-regrowing cells; the threshold is provisional
# and will be re-fit on future experiments that produce a clean bimodal
# distribution. "no_rec_data" = cell disappeared before enough recovery
# samples could be collected.
ELONG_ALIVE_THR = 0.3  # µm/h
FATE_COLORS = {
    "alive":        "#888888",   # gray (alive cells are not highlighted)
    "dead":         "#D55E00",   # vermillion (dead cells highlighted)
    "no_rec_data":  "#BBBBBB",   # lighter gray
}
# Tree color by mean_ri comparison at each event (birth or own division).
# Each cell's line is split into segments between its own division events,
# and each segment is colored by whether the cell has the higher or lower
# mean_ri side at the event that starts that segment.
TREE_RI_COLORS = {
    "higher": "#E69F00",        # orange — higher mean_ri at event
    "lower":  "#0072B2",        # blue   — lower mean_ri at event
}
MOTHER_COLOR = "#D55E00"          # mother lineage (pre-first-division)
DEAD_COLOR = "#CC0000"            # overrides RI color for cells judged dead
NO_REC_COLOR = "#BBBBBB"          # lighter gray for no_rec_data
RI_UNDETERMINED_COLOR = "#999999" # when mean_ri comparison not possible

VOLUME_YLIM = (50.0, 400.0)
MEAN_RI_YLIM = (1.345, 1.370)
MASS_YLIM = (0.0, 50.0)


# =============================================================================
# Epoch helper
# =============================================================================
def epoch_of(frame: int) -> str:
    """Return 'pre' / 'starv' / 'rec' for a given frame (or 'pre' for frame < 0)."""
    if frame < STARV_START_FRAME:
        return "pre"
    if frame < REC_START_FRAME:
        return "starv"
    return "rec"


def frame_to_h(frame: float) -> float:
    return float(frame) / FRAMES_PER_HOUR


# =============================================================================
# Load inputs
# =============================================================================
def load_inputs(channel_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    lineage_out = channel_dir / "inference_out" / "lineage_out"
    clist = pd.read_csv(lineage_out / "clist.csv")
    data3D = pd.read_csv(lineage_out / "lineage_data3D.csv")
    return clist, data3D


# =============================================================================
# Fate classification
# =============================================================================
def _recovery_elongation_rate(
    cell_data: pd.DataFrame,
    birth: int,
    death: int,
    rec_children_births: list[int],
) -> tuple[float, int, int]:
    """Mean per-segment elongation slope [µm/h] over recovery-era cell cycles.

    Segments are delimited by children's birth frames so each segment covers
    one cell cycle (birth -> next division) in recovery. This keeps the value
    meaningful for cells that divide frequently (mother) as well as for cells
    that never divide in recovery (quiescent/dead).

    Returns (mean_slope, total_n_points, n_segments).
    """
    segments: list[tuple[int, int]] = []
    seg_start = max(REC_START_FRAME, birth) + 1
    for b in sorted(rec_children_births):
        if b > seg_start:
            segments.append((seg_start, b))
        seg_start = b
    if death + 1 > seg_start:
        segments.append((seg_start, death + 1))

    slopes: list[float] = []
    total_n = 0
    for s0, s1 in segments:
        sub = cell_data[
            (cell_data["frame"] >= s0)
            & (cell_data["frame"] < s1)
            & (~cell_data["is_outlier"])
            & (~cell_data["touches_border"])
            & cell_data["long_axis_um"].notna()
        ]
        n = len(sub)
        if n >= 4:
            t = sub["time_h"].to_numpy()
            L = sub["long_axis_um"].to_numpy()
            slopes.append(float(np.polyfit(t, L, 1)[0]))
            total_n += n
    if not slopes or total_n < REC_SLOPE_MIN_POINTS:
        return (float("nan"), total_n, len(slopes))
    return (float(np.mean(slopes)), total_n, len(slopes))


def _find_false_births(clist: pd.DataFrame, data3D: pd.DataFrame) -> set[int]:
    """Cells whose birth coincides with, or directly follows, an outlier frame
    of the parent.

    The rank-pointer tracker detects divisions inline from frame-to-frame area
    ratios, *before* outlier detection runs. A segmentation blip that inflates
    the parent's area at frame F can therefore trigger a spurious division at
    frame F (curr drops relative to the bogus prev) or at frame F+1 (curr+next
    ~= inflated prev). Both symptoms are covered by checking whether the
    daughter's birth_frame falls on the parent's outlier set, or one frame
    after it.
    """
    outlier_frames_by_parent = {
        int(cid): set(grp[grp["is_outlier"]]["frame"].astype(int).tolist())
        for cid, grp in data3D.groupby("cell_id")
    }
    false_ids: set[int] = set()
    for _, row in clist.iterrows():
        parent = int(row["mother_id"])
        birth = int(row["birth_frame"])
        if parent < 0:
            continue
        outliers = outlier_frames_by_parent.get(parent, set())
        if birth in outliers or (birth - 1) in outliers:
            false_ids.add(int(row["cell_id"]))
    return false_ids


def build_fate_table(clist: pd.DataFrame, data3D: pd.DataFrame) -> pd.DataFrame:
    """One row per in-tree cell with fate label + pre-starvation metrics.

    Fate is elongation-rate based on long_axis_um after REC_START_FRAME.
    Provisional threshold ELONG_ALIVE_THR = 0.3 µm/h:
      slope >= 0.3  -> alive
      slope <  0.3  -> dead
      NaN (no data) -> no_rec_data
    Median/min of observed slopes are still stored in df.attrs for reference.
    """
    false_ids = _find_false_births(clist, data3D)
    if false_ids:
        print(f"[info] false-division cells excluded (birth on parent outlier frame): n={len(false_ids)}",
              file=sys.stderr)

    clist_real = clist[~clist["cell_id"].isin(false_ids)].copy()
    in_tree = clist_real[clist_real["in_tree"]].copy()

    children_map: dict[int, pd.DataFrame] = {
        cid: grp for cid, grp in clist_real.groupby("mother_id")
    }
    data3D_by_cid = {cid: grp for cid, grp in data3D.groupby("cell_id")}

    rows = []
    for _, cell in in_tree.iterrows():
        cid = int(cell["cell_id"])
        birth = int(cell["birth_frame"])
        death = int(cell["death_frame"])

        children = children_map.get(cid, pd.DataFrame())
        rec_children = children[children["birth_frame"] >= REC_START_FRAME] if not children.empty else children
        n_div_in_rec = int(len(rec_children))
        divided_in_rec = n_div_in_rec > 0

        if divided_in_rec:
            first_rec_div = int(rec_children["birth_frame"].min())
            anchor = max(REC_START_FRAME, birth)
            lag_frames = first_rec_div - anchor
        else:
            lag_frames = np.nan

        cell_data = data3D_by_cid.get(cid, data3D.iloc[0:0])
        rec_children_births = (
            rec_children["birth_frame"].astype(int).tolist()
            if not rec_children.empty else []
        )
        rec_elong, n_rec_points, n_rec_segments = _recovery_elongation_rate(
            cell_data, birth, death, rec_children_births
        )

        # pre-starvation window mean (drop outliers / border-touching frames)
        sub = cell_data[
            (cell_data["frame"] >= STARV_START_FRAME - PRE_WINDOW_FRAMES)
            & (cell_data["frame"] < STARV_START_FRAME)
            & (~cell_data["is_outlier"])
            & (~cell_data["touches_border"])
        ]
        pre_v = float(sub["volume_um3_rod"].mean()) if len(sub) else np.nan
        pre_m = float(sub["mass_pg"].mean()) if len(sub) else np.nan
        pre_ri = float(sub["mean_ri"].mean()) if len(sub) else np.nan

        rows.append({
            "cell_id": cid,
            "parent_id": int(cell["mother_id"]),
            "in_tree": bool(cell["in_tree"]),
            "generation": int(cell["generation"]),
            "birth_frame": birth,
            "death_frame": death,
            "birth_time_h": frame_to_h(birth) if birth >= 0 else 0.0,
            "birth_epoch": epoch_of(birth),
            "divided_in_rec": divided_in_rec,
            "n_divisions_in_rec": n_div_in_rec,
            "recovery_lag_frames": lag_frames,
            "recovery_lag_h": frame_to_h(lag_frames) if np.isfinite(lag_frames) else np.nan,
            "recovery_elong_um_per_h": rec_elong,
            "n_recovery_points": n_rec_points,
            "n_recovery_segments": n_rec_segments,
            "pre_starv_volume_um3": pre_v,
            "pre_starv_mass_pg": pre_m,
            "pre_starv_mean_ri": pre_ri,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df.attrs["elong_threshold_median"] = float("nan")
        df.attrs["elong_lower_bound"] = float("nan")
        df["fate"] = pd.Series(dtype=str)
        return df
    df = df.sort_values("cell_id").reset_index(drop=True)

    # determine fast/slow threshold from the elongation-rate distribution
    valid_slopes = df["recovery_elong_um_per_h"].dropna().to_numpy()
    if valid_slopes.size:
        thr_median = float(np.median(valid_slopes))
        thr_lower = float(np.min(valid_slopes))
    else:
        thr_median = float("nan")
        thr_lower = float("nan")

    def _label(slope: float) -> str:
        if not np.isfinite(slope):
            return "no_rec_data"
        return "alive" if slope >= ELONG_ALIVE_THR else "dead"

    df["fate"] = df["recovery_elong_um_per_h"].map(_label)
    df.attrs["elong_threshold_median"] = thr_median
    df.attrs["elong_lower_bound"] = thr_lower
    df.attrs["elong_alive_thr"] = ELONG_ALIVE_THR
    return df


# =============================================================================
# Tree layout (dendrogram, mother = column 0, branches to the right only)
# =============================================================================
def assign_tree_x(survival: pd.DataFrame) -> dict[int, float]:
    """Same idea as central_cell_lineage_tracker._assign_tree_x, but driven by the
    flat survival table rather than the LineageState object."""
    mother_rows = survival[(survival["in_tree"]) & (survival["parent_id"] == -1)]
    if mother_rows.empty:
        return {}
    mother_id = int(mother_rows.iloc[0]["cell_id"])

    children_map: dict[int, list[int]] = {}
    for _, r in survival.iterrows():
        p = int(r["parent_id"])
        if p >= 0:
            children_map.setdefault(p, []).append(int(r["cell_id"]))
    # Match the physical Mother Machine order: the newest daughter sits
    # immediately next to the mother (inner rank), older daughters get
    # pushed outward. Sort by birth_frame descending so the last-born
    # daughter is placed closest to the parent in the tree layout.
    for v in children_map.values():
        v.sort(key=lambda c: int(
            survival.loc[survival["cell_id"] == c, "birth_frame"].iloc[0]
        ), reverse=True)

    positions: dict[int, float] = {}

    def assign(cid: int, x: float) -> float:
        positions[cid] = x
        nxt = x + 1.0
        for ch in children_map.get(cid, []):
            nxt = assign(ch, nxt)
        return nxt

    assign(mother_id, 0.0)
    return positions


# =============================================================================
# Figure 1: mother trajectory with media epochs + division ticks
# =============================================================================
def plot_mother_trajectory(data3D: pd.DataFrame, survival: pd.DataFrame) -> plt.Figure:
    m = data3D[data3D["cell_id"] == 0].sort_values("frame")
    t = m["time_h"].to_numpy()
    # mask invalid frames with NaN for gap in the line
    bad = m["is_outlier"].to_numpy() | m["touches_border"].to_numpy()
    vol = np.where(bad, np.nan, m["volume_um3_rod"].to_numpy())
    mass = np.where(bad, np.nan, m["mass_pg"].to_numpy())

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True, constrained_layout=True)

    # media epoch shading
    t_starv = frame_to_h(STARV_START_FRAME)
    t_mid = frame_to_h(SWITCH_MID_FRAME)
    t_rec = frame_to_h(REC_START_FRAME)
    t_end = frame_to_h(data3D["frame"].max())
    for ax in axes:
        ax.axvspan(0, t_starv, color="#FAFAFA", zorder=0)
        ax.axvspan(t_starv, t_rec, color="#FFF2CC", alpha=0.55, zorder=0)  # 0%
        ax.axvspan(t_rec, t_end, color="#E6F4EA", alpha=0.55, zorder=0)   # recovery 2%

    # mother plot
    axes[0].plot(t, vol, color="#D55E00", linewidth=1.0, zorder=3)
    axes[1].plot(t, mass, color="#D55E00", linewidth=1.0, zorder=3)

    # division ticks (mother's daughters birth_time)
    mother_divs = survival[(survival["parent_id"] == 0)]["birth_time_h"].to_numpy()
    for td in mother_divs:
        for ax in axes:
            ax.axvline(td, color="#444444", linewidth=0.4, linestyle="--", alpha=0.55, zorder=2)

    # intermediate switch (drawn only, not used for fate)
    for ax in axes:
        ax.axvline(t_mid, color="#888888", linewidth=0.6, linestyle=":", zorder=4)
    # recovery start emphasis
    for ax in axes:
        ax.axvline(t_rec, color="#009E73", linewidth=1.2, zorder=4)

    axes[0].set_ylabel(r"volume [µm$^3$]")
    axes[0].set_ylim(VOLUME_YLIM)
    axes[1].set_ylabel("dry mass [pg]")
    axes[1].set_ylim(MASS_YLIM)
    axes[1].set_xlabel("time [h]")
    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    axes[0].set_title(
        f"Mother trajectory (pre | 0% starvation | 2% recovery),  "
        f"n_divisions = {len(mother_divs)}",
        fontsize=9,
    )
    return fig


# =============================================================================
# Figure 2: fate-colored lineage tree
# =============================================================================
def _compare_at_frame(self_row, other_row) -> str:
    """Return 'higher'/'lower'/'undetermined' based on mean_ri comparison."""
    if self_row is None or other_row is None:
        return "undetermined"
    if self_row["is_outlier"] or other_row["is_outlier"]:
        return "undetermined"
    if self_row["touches_border"] or other_row["touches_border"]:
        return "undetermined"
    s = self_row["mean_ri"]
    o = other_row["mean_ri"]
    if not (np.isfinite(s) and np.isfinite(o)):
        return "undetermined"
    return "higher" if s >= o else "lower"


def _segment_color(label: str, fate: str) -> str:
    if fate == "dead":
        return DEAD_COLOR
    if fate == "no_rec_data":
        return NO_REC_COLOR
    if label in TREE_RI_COLORS:
        return TREE_RI_COLORS[label]
    return RI_UNDETERMINED_COLOR


def plot_fate_colored_tree(survival: pd.DataFrame, data3D: pd.DataFrame) -> plt.Figure:
    positions = assign_tree_x(survival)
    n_cols = max(1, int(max(positions.values())) + 1) if positions else 1
    width_in = max(4.0, min(9.0, 0.045 * n_cols + 3.0))
    fig, ax = plt.subplots(figsize=(width_in, 5.2), constrained_layout=True)

    # fast (cell_id, frame) lookup
    d3_idx = data3D.set_index(["cell_id", "frame"], drop=False)

    def _row_at(cid: int, frame: int):
        try:
            r = d3_idx.loc[(cid, frame)]
        except KeyError:
            return None
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        return r

    # children_map from survival (sorted by birth_frame ascending for correct event order)
    children_map: dict[int, list[tuple[int, int]]] = {}  # parent -> [(child_id, child_birth)]
    for _, r in survival.iterrows():
        p = int(r["parent_id"])
        if p >= 0:
            children_map.setdefault(p, []).append((int(r["cell_id"]), int(r["birth_frame"])))
    for v in children_map.values():
        v.sort(key=lambda t: t[1])

    # horizontal connectors first
    for _, cell in survival.iterrows():
        p = int(cell["parent_id"])
        if p < 0:
            continue
        cid = int(cell["cell_id"])
        if cid not in positions or p not in positions:
            continue
        y = frame_to_h(cell["birth_frame"])
        ax.plot([positions[p], positions[cid]], [y, y],
                color="#888888", linewidth=0.3, alpha=0.55, zorder=1)

    # vertical cell lines, split by own-division events
    seg_counter = {"higher": 0, "lower": 0, "undetermined": 0, "mother": 0,
                   "dead": 0, "no_rec_data": 0}
    for _, cell in survival.iterrows():
        cid = int(cell["cell_id"])
        if cid not in positions:
            continue
        x = positions[cid]
        birth = int(cell["birth_frame"])
        death = int(cell["death_frame"])
        parent = int(cell["parent_id"])
        fate = cell["fate"]
        is_mother = parent == -1
        lw = 1.4 if is_mother else 0.7

        own_divs = [cb for (_, cb) in children_map.get(cid, []) if birth <= cb <= death]
        own_divs.sort()
        boundaries = [max(birth, 0)] + own_divs + [death]

        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]
            if seg_end <= seg_start:
                continue
            event_frame = boundaries[i]
            if i == 0:
                if is_mother:
                    color = MOTHER_COLOR
                    seg_counter["mother"] += 1
                else:
                    self_r = _row_at(cid, event_frame)
                    other_r = _row_at(parent, event_frame)
                    label = _compare_at_frame(self_r, other_r)
                    color = _segment_color(label, fate)
                    if fate == "dead":
                        seg_counter["dead"] += 1
                    elif fate == "no_rec_data":
                        seg_counter["no_rec_data"] += 1
                    else:
                        seg_counter[label] += 1
            else:
                # division event at event_frame: find the daughter born then
                daughter_id = next(
                    (c_id for (c_id, cb) in children_map.get(cid, [])
                     if cb == event_frame), None,
                )
                if daughter_id is None:
                    label = "undetermined"
                else:
                    self_r = _row_at(cid, event_frame)
                    other_r = _row_at(daughter_id, event_frame)
                    label = _compare_at_frame(self_r, other_r)
                color = _segment_color(label, fate)
                if fate == "dead":
                    seg_counter["dead"] += 1
                elif fate == "no_rec_data":
                    seg_counter["no_rec_data"] += 1
                else:
                    seg_counter[label] += 1

            ax.plot([x, x], [frame_to_h(seg_start), frame_to_h(seg_end)],
                    color=color, linewidth=lw, solid_capstyle="butt", zorder=2)

    # media epoch shading on the y axis
    t_starv = frame_to_h(STARV_START_FRAME)
    t_mid = frame_to_h(SWITCH_MID_FRAME)
    t_rec = frame_to_h(REC_START_FRAME)
    ax.axhspan(t_starv, t_rec, color="#FFF2CC", alpha=0.35, zorder=0)
    ax.axhline(t_mid, color="#888888", linewidth=0.4, linestyle=":", zorder=0)
    ax.axhline(t_rec, color="#009E73", linewidth=0.6, zorder=0)

    # mother label
    mother_rows = survival[(survival["parent_id"] == -1)]
    if not mother_rows.empty:
        mx = positions.get(int(mother_rows.iloc[0]["cell_id"]), 0.0)
        ax.text(mx, -0.4, "M", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.invert_yaxis()
    ax.set_xlabel("lineage (M = mother, branches to the right)")
    ax.set_ylabel("time [h]")
    ax.set_xticks([])
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=MOTHER_COLOR, lw=1.4,
                   label="mother (pre-first-division)"),
        plt.Line2D([0], [0], color=TREE_RI_COLORS["higher"], lw=1.0,
                   label=f"higher mean RI at event (n_segments={seg_counter['higher']})"),
        plt.Line2D([0], [0], color=TREE_RI_COLORS["lower"], lw=1.0,
                   label=f"lower mean RI at event (n_segments={seg_counter['lower']})"),
        plt.Line2D([0], [0], color=DEAD_COLOR, lw=1.0,
                   label=f"dead cells (n_segments={seg_counter['dead']})"),
        plt.Line2D([0], [0], color=NO_REC_COLOR, lw=1.0,
                   label=f"no_rec_data (n_segments={seg_counter['no_rec_data']})"),
    ]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="lower right")
    ax.set_title("Lineage tree: segment color updates at every division event", fontsize=9)
    return fig


# =============================================================================
# Figure 3: pre-starvation state vs recovery slope (continuous)
# =============================================================================
def plot_pre_starv_vs_slope(survival: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0), constrained_layout=True)
    metrics = [
        ("pre_starv_volume_um3", r"pre-starv volume [µm$^3$]", VOLUME_YLIM),
        ("pre_starv_mass_pg",     "pre-starv dry mass [pg]",    MASS_YLIM),
        ("pre_starv_mean_ri",     "pre-starv mean RI",          MEAN_RI_YLIM),
    ]

    valid = survival[
        survival["pre_starv_volume_um3"].notna()
        & survival["recovery_elong_um_per_h"].notna()
    ]
    thr_alive = survival.attrs.get("elong_alive_thr", ELONG_ALIVE_THR)

    for ax, (col, xlabel, xlim) in zip(axes, metrics):
        for fate, color in FATE_COLORS.items():
            sub = valid[valid["fate"] == fate]
            if sub.empty:
                continue
            ax.scatter(
                sub[col].to_numpy(),
                sub["recovery_elong_um_per_h"].to_numpy(),
                color=color, s=22, alpha=0.8,
                edgecolor="white", linewidth=0.4,
                label=f"{fate} (n={len(sub)})",
            )
        if np.isfinite(thr_alive):
            ax.axhline(thr_alive, color="#444444", linewidth=0.5,
                       linestyle="--", zorder=0)
        ax.axhline(0, color="#000000", linewidth=0.3, zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_xlim(xlim)
        ax.set_ylabel(r"recovery elongation rate [µm/h]")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    axes[0].legend(fontsize=6, frameon=False, loc="best")
    fig.suptitle(
        f"Pre-starvation state (mean over {PRE_WINDOW_FRAMES} frames) vs recovery-era elongation rate",
        fontsize=9,
    )
    return fig


# =============================================================================
# Figure 4: recovery-slope histogram with fast/slow threshold + lower bound
# =============================================================================
def plot_slope_hist(survival: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 3.0), constrained_layout=True)
    slopes_by_fate = {
        fate: survival[survival["fate"] == fate]["recovery_elong_um_per_h"].dropna().to_numpy()
        for fate in ["dead", "alive"]
    }
    all_slopes = np.concatenate([v for v in slopes_by_fate.values() if v.size])
    if all_slopes.size:
        edges = np.linspace(all_slopes.min() - 0.1, all_slopes.max() + 0.1, 24)
    else:
        edges = np.linspace(-1.0, 1.0, 24)

    ax.hist(
        [slopes_by_fate["dead"], slopes_by_fate["alive"]],
        bins=edges, stacked=True,
        color=[FATE_COLORS["dead"], FATE_COLORS["alive"]],
        edgecolor="white", linewidth=0.4,
        label=[
            f"dead (n={len(slopes_by_fate['dead'])})",
            f"alive (n={len(slopes_by_fate['alive'])})",
        ],
    )
    thr_alive = survival.attrs.get("elong_alive_thr", ELONG_ALIVE_THR)
    ax.axvline(thr_alive, color="#222222", linewidth=0.8, linestyle="--",
               label=f"alive threshold = {thr_alive:.2f}")
    ax.axvline(0, color="#000000", linewidth=0.3, zorder=0)

    ax.set_xlabel(r"recovery elongation rate [µm/h]")
    ax.set_ylabel("count")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title(f"Recovery-era per-cycle elongation rate (alive >= {thr_alive:.2f} µm/h)", fontsize=9)
    return fig


# =============================================================================
# Figure 5: recovery-lag histogram (first division after recovery)
# =============================================================================
def plot_recovery_lag_hist(survival: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.0, 3.0), constrained_layout=True)
    divided = survival[survival["divided_in_rec"]]
    mother_lag = divided[divided["parent_id"] == -1]["recovery_lag_h"].to_numpy()
    daughter_lag = divided[divided["parent_id"] != -1]["recovery_lag_h"].to_numpy()

    t_end = frame_to_h(2400)
    t_rec = frame_to_h(REC_START_FRAME)
    xmax = max(1.0, t_end - t_rec)
    bins = np.linspace(0, xmax, 21)

    ax.hist(
        [daughter_lag, mother_lag],
        bins=bins, stacked=True,
        color=["#888888", "#D55E00"],
        edgecolor="white", linewidth=0.4,
        label=[f"daughters (n={len(daughter_lag)})",
               f"mother (n={len(mother_lag)})"],
    )
    ax.set_xlabel("recovery lag until first division [h]")
    ax.set_ylabel("count")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("First division after 2% recovery", fontsize=9)
    return fig


# =============================================================================
# Main driver
# =============================================================================
def run(channel_dir: Path,
        starv_frame: int, rec_frame: int, pre_window: int) -> None:
    global STARV_START_FRAME, REC_START_FRAME, PRE_WINDOW_FRAMES
    STARV_START_FRAME = starv_frame
    REC_START_FRAME = rec_frame
    PRE_WINDOW_FRAMES = pre_window

    clist, data3D = load_inputs(channel_dir)
    print(f"[info] clist n={len(clist)}  in_tree={int(clist['in_tree'].sum())}  "
          f"data3D rows={len(data3D)}", file=sys.stderr)

    survival = build_fate_table(clist, data3D)
    out_path = channel_dir / "inference_out" / "lineage_out" / "survival_summary.csv"
    survival.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}  ({len(survival)} rows)", file=sys.stderr)

    counts = survival["fate"].value_counts().to_dict()
    print(f"[info] fate counts: {counts}", file=sys.stderr)
    thr_median = survival.attrs.get("elong_threshold_median", float("nan"))
    thr_lower = survival.attrs.get("elong_lower_bound", float("nan"))
    print(f"[info] alive threshold (elongation rate):       {ELONG_ALIVE_THR:.3f} um/h", file=sys.stderr)
    print(f"[info] recovery elongation median of observed:  {thr_median:.3f} um/h", file=sys.stderr)
    print(f"[info] recovery elongation lower bound (min):   {thr_lower:.3f} um/h", file=sys.stderr)
    mother_row = survival[survival["parent_id"] == -1]
    if not mother_row.empty:
        lag = mother_row.iloc[0]["recovery_lag_h"]
        elong = mother_row.iloc[0]["recovery_elong_um_per_h"]
        print(f"[info] mother recovery lag: {lag} h  elong: {elong:.3f} um/h", file=sys.stderr)

    params_common = {
        "channel_dir": str(channel_dir),
        "starvation_start_frame": starv_frame,
        "switch_mid_frame": SWITCH_MID_FRAME,
        "recovery_start_frame": rec_frame,
        "pre_window_frames": pre_window,
        "rec_slope_min_points": REC_SLOPE_MIN_POINTS,
        "frames_per_hour": FRAMES_PER_HOUR,
        "n_cells": len(survival),
        "n_alive": counts.get("alive", 0),
        "n_dead": counts.get("dead", 0),
        "n_no_rec_data": counts.get("no_rec_data", 0),
        "elong_alive_thr_um_per_h": ELONG_ALIVE_THR,
        "elong_threshold_median_um_per_h": thr_median,
        "elong_lower_bound_um_per_h": thr_lower,
    }
    data_source = {"raw_files": [
        str(channel_dir / "inference_out" / "lineage_out" / "clist.csv"),
        str(channel_dir / "inference_out" / "lineage_out" / "lineage_data3D.csv"),
    ]}

    fig1 = plot_mother_trajectory(data3D, survival)
    save_figure(fig1, params=params_common,
                description="mother volume/mass trajectory with media epochs and division ticks",
                fmt="pdf", data_source=data_source)
    plt.close(fig1)

    fig2 = plot_fate_colored_tree(survival, data3D)
    save_figure(fig2, params=params_common,
                description="lineage tree colored by alive/dead fate (elongation threshold)",
                fmt="pdf", data_source=data_source)
    plt.close(fig2)

    fig3 = plot_pre_starv_vs_slope(survival)
    save_figure(fig3, params=params_common,
                description="pre-starvation volume/mass/RI vs recovery elongation rate (scatter)",
                fmt="pdf", data_source=data_source)
    plt.close(fig3)

    fig4 = plot_slope_hist(survival)
    save_figure(fig4, params=params_common,
                description=f"histogram of recovery-era elongation rate with alive threshold = {ELONG_ALIVE_THR:.2f} um/h",
                fmt="pdf", data_source=data_source)
    plt.close(fig4)

    fig5 = plot_recovery_lag_hist(survival)
    save_figure(fig5, params=params_common,
                description="histogram of recovery lag until first division",
                fmt="pdf", data_source=data_source)
    plt.close(fig5)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indir", required=True, type=Path,
                   help="channel dir containing inference_out/lineage_out/{clist,lineage_data3D}.csv")
    p.add_argument("--starvation-start-frame", type=int, default=STARV_START_FRAME)
    p.add_argument("--recovery-start-frame", type=int, default=REC_START_FRAME)
    p.add_argument("--pre-window-frames", type=int, default=PRE_WINDOW_FRAMES)
    return p


def main() -> int:
    args = build_parser().parse_args()
    run(args.indir, args.starvation_start_frame, args.recovery_start_frame,
        args.pre_window_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
