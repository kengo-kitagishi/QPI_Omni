"""
per_channel_figures.py — Per-channel mother-cell plots.

Reads the CSV outputs of central_cell_lineage_tracker.py and emits three
separate single-quantity PDFs per channel (no multi-panel composites):

    mother_volume.pdf       mother cell volume vs time, with division ticks
                            and media-switch vlines
    mother_mean_ri.pdf      mother cell mean_RI vs time, same annotations
    lineage_tree.pdf        full lineage tree rooted at mother

Inputs (under <channel_dir>/inference_out/lineage_out/):
    lineage_data3D.csv      per-frame per-cell metrics (volume_um3_rod,
                            mean_ri, mass_pg, n_medium_used, ...)
    clist.csv               per-cell summary (birth_frame, death_frame,
                            mother_id, in_tree, ...)
    lineage_run_params.json optional — provides media_schedule and media_ri
                            for the media-switch vlines

Example:
    python per_channel_figures.py --indir /path/to/Pos9/ch01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_logger import save_figure

# Okabe-Ito palette
OI = {
    "vermilion": "#D55E00",  # mother
    "orange":    "#E69F00",  # higher mean_RI side at event
    "blue":      "#0072B2",  # lower mean_RI side at event
    "gray":      "#999999",  # undetermined / non-in-tree
    "skyblue":   "#56B4E9",
    "green":     "#009E73",
}


# =============================================================================
# I/O
# =============================================================================
def channel_label(channel_dir: Path) -> str:
    """Short label like 'Pos9_ch01' built from the directory path."""
    parts = channel_dir.resolve().parts
    pos = next((p for p in parts if p.startswith("Pos")), None)
    return f"{pos}_{channel_dir.name}" if pos else channel_dir.name


def load_tables(
    channel_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out = channel_dir / "inference_out" / "lineage_out"
    data3D = pd.read_csv(out / "lineage_data3D.csv")
    clist = pd.read_csv(out / "clist.csv")
    run_meta_path = out / "lineage_run_params.json"
    run_meta = {}
    if run_meta_path.exists():
        try:
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] failed to load {run_meta_path}: {e}", file=sys.stderr)
    return data3D, clist, run_meta


def parse_schedule_str(s: Optional[str]) -> list[tuple[int, str]]:
    if not s:
        return []
    out: list[tuple[int, str]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        f, name = tok.split(":", 1)
        try:
            out.append((int(f.strip()), name.strip()))
        except ValueError:
            continue
    out.sort(key=lambda x: x[0])
    return out


def mother_division_frames(clist: pd.DataFrame, mother_id: int = 0) -> np.ndarray:
    """Frame numbers at which the mother divided (= birth frames of her daughters)."""
    sub = clist[clist["mother_id"] == mother_id]
    if sub.empty:
        return np.array([], dtype=int)
    return np.sort(sub["birth_frame"].astype(int).to_numpy())


# =============================================================================
# Annotations shared by the two timeseries panels
# =============================================================================
def _add_media_vlines(
    ax,
    media_schedule: list[tuple[int, str]],
    media_ri: dict[str, float],
    time_interval_min: Optional[float],
) -> None:
    if not media_schedule or not media_ri:
        return
    to_x = (lambda f: f * (time_interval_min / 60.0)) if time_interval_min else float
    for f_switch, name in media_schedule:
        if f_switch <= 0:
            continue
        x = to_x(f_switch)
        ax.axvline(x, color="#888888", linestyle="--", linewidth=0.6, alpha=0.7, zorder=1)
        ri = media_ri.get(name, float("nan"))
        label = f"{name} (n={ri:.4f})" if np.isfinite(ri) else name
        ax.annotate(
            label, xy=(x, 1.0), xycoords=("data", "axes fraction"),
            xytext=(2, -2), textcoords="offset points",
            ha="left", va="top", fontsize=6, color="#555555",
        )


def _add_division_ticks(
    ax,
    division_frames: np.ndarray,
    time_interval_min: Optional[float],
) -> None:
    if division_frames.size == 0:
        return
    to_x = (lambda f: f * (time_interval_min / 60.0)) if time_interval_min else float
    for f in division_frames:
        x = to_x(int(f))
        ax.axvline(x, color="#444444", linewidth=0.35, linestyle=":", alpha=0.45, zorder=2)


# =============================================================================
# Figure 1: mother volume vs time
# =============================================================================
def fig_mother_volume(
    data3D: pd.DataFrame,
    division_frames: np.ndarray,
    media_schedule: list[tuple[int, str]],
    media_ri: dict[str, float],
    time_interval_min: Optional[float],
    label: str,
) -> plt.Figure:
    m = data3D[data3D["cell_id"] == 0].sort_values("frame")
    bad = m["is_outlier"].to_numpy(dtype=bool) | m["touches_border"].to_numpy(dtype=bool)
    t = m["time_h"].to_numpy() if "time_h" in m.columns else m["frame"].to_numpy()
    y = np.where(bad, np.nan, m["volume_um3_rod"].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(140/25.4, 70/25.4), constrained_layout=True)
    _add_division_ticks(ax, division_frames, time_interval_min)
    _add_media_vlines(ax, media_schedule, media_ri, time_interval_min)
    ax.plot(t, y, color=OI["vermilion"], linewidth=1.0, zorder=3)

    ax.set_xlabel("time [h]" if "time_h" in m.columns else "frame")
    ax.set_ylabel(r"volume [µm$^3$]")
    ax.set_title(f"{label}  mother volume (n_divisions={len(division_frames)})", fontsize=9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return fig


# =============================================================================
# Figure 2: mother mean_RI vs time
# =============================================================================
def fig_mother_mean_ri(
    data3D: pd.DataFrame,
    division_frames: np.ndarray,
    media_schedule: list[tuple[int, str]],
    media_ri: dict[str, float],
    time_interval_min: Optional[float],
    label: str,
) -> plt.Figure:
    m = data3D[data3D["cell_id"] == 0].sort_values("frame")
    bad = m["is_outlier"].to_numpy(dtype=bool) | m["touches_border"].to_numpy(dtype=bool)
    t = m["time_h"].to_numpy() if "time_h" in m.columns else m["frame"].to_numpy()
    y = np.where(bad, np.nan, m["mean_ri"].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(140/25.4, 70/25.4), constrained_layout=True)
    _add_division_ticks(ax, division_frames, time_interval_min)
    _add_media_vlines(ax, media_schedule, media_ri, time_interval_min)
    ax.plot(t, y, color=OI["vermilion"], linewidth=1.0, zorder=3)

    # If n_medium_used is recorded per frame, overlay it as a faint reference line.
    if "n_medium_used" in m.columns:
        nm = m["n_medium_used"].to_numpy(dtype=float)
        ax.plot(t, nm, color="#888888", linewidth=0.6, linestyle="-", alpha=0.6,
                zorder=2, label="n_medium")
        ax.legend(loc="best", fontsize=6, frameon=False)

    ax.set_xlabel("time [h]" if "time_h" in m.columns else "frame")
    ax.set_ylabel("mean RI")
    ax.set_title(f"{label}  mother mean RI", fontsize=9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return fig


# =============================================================================
# Figure 3: lineage tree (rooted at mother, branches strictly to the right)
# =============================================================================
def _assign_tree_x(clist: pd.DataFrame) -> dict[int, float]:
    """Mother at x=0, daughters branch right. Recently-born daughters sit
    immediately next to the parent (matches Mother Machine rank order)."""
    mother_rows = clist[(clist["in_tree"]) & (clist["mother_id"] == -1)]
    if mother_rows.empty:
        return {}
    mother_id = int(mother_rows.iloc[0]["cell_id"])

    children_map: dict[int, list[int]] = {}
    for _, r in clist.iterrows():
        p = int(r["mother_id"])
        if p >= 0 and bool(r["in_tree"]):
            children_map.setdefault(p, []).append(int(r["cell_id"]))
    for v in children_map.values():
        v.sort(key=lambda c: int(
            clist.loc[clist["cell_id"] == c, "birth_frame"].iloc[0]
        ), reverse=True)  # newest first -> sits next to parent

    positions: dict[int, float] = {}

    def assign(cid: int, x: float) -> float:
        positions[cid] = x
        nxt = x + 1.0
        for ch in children_map.get(cid, []):
            nxt = assign(ch, nxt)
        return nxt

    assign(mother_id, 0.0)
    return positions


def _compare_mean_ri(self_row, other_row) -> str:
    if self_row is None or other_row is None:
        return "undetermined"
    if bool(self_row["is_outlier"]) or bool(other_row["is_outlier"]):
        return "undetermined"
    if bool(self_row["touches_border"]) or bool(other_row["touches_border"]):
        return "undetermined"
    s, o = float(self_row["mean_ri"]), float(other_row["mean_ri"])
    if not (np.isfinite(s) and np.isfinite(o)):
        return "undetermined"
    return "higher" if s >= o else "lower"


def fig_lineage_tree(
    clist: pd.DataFrame,
    data3D: pd.DataFrame,
    time_interval_min: Optional[float],
    label: str,
) -> plt.Figure:
    positions = _assign_tree_x(clist)
    n_cols = max(1, int(max(positions.values())) + 1) if positions else 1
    width_in = max(4.0, min(9.0, 0.045 * n_cols + 3.0))
    fig, ax = plt.subplots(figsize=(width_in, 5.2), constrained_layout=True)

    to_h = (lambda f: f * (time_interval_min / 60.0)) if time_interval_min else float

    # fast lookup of per-frame mother / cell data
    d3_idx = data3D.set_index(["cell_id", "frame"], drop=False)

    def _row_at(cid: int, frame: int):
        try:
            r = d3_idx.loc[(cid, frame)]
        except KeyError:
            return None
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        return r

    children_map: dict[int, list[tuple[int, int]]] = {}
    for _, r in clist.iterrows():
        p = int(r["mother_id"])
        if p >= 0 and bool(r["in_tree"]):
            children_map.setdefault(p, []).append((int(r["cell_id"]), int(r["birth_frame"])))
    for v in children_map.values():
        v.sort(key=lambda t: t[1])

    # horizontal connectors first
    for _, cell in clist.iterrows():
        if not bool(cell["in_tree"]):
            continue
        p = int(cell["mother_id"])
        cid = int(cell["cell_id"])
        if p < 0 or p not in positions or cid not in positions:
            continue
        y = to_h(int(cell["birth_frame"]))
        ax.plot([positions[p], positions[cid]], [y, y],
                color="#888888", linewidth=0.3, alpha=0.55, zorder=1)

    # vertical cell lines, split by own-division events, colored per segment
    counter = {"higher": 0, "lower": 0, "undetermined": 0, "mother": 0}
    for _, cell in clist.iterrows():
        if not bool(cell["in_tree"]):
            continue
        cid = int(cell["cell_id"])
        if cid not in positions:
            continue
        x = positions[cid]
        birth = int(cell["birth_frame"])
        death = int(cell["death_frame"])
        parent = int(cell["mother_id"])
        is_mother = parent == -1
        lw = 1.4 if is_mother else 0.7

        own_divs = sorted(
            cb for (_, cb) in children_map.get(cid, []) if birth <= cb <= death
        )
        boundaries = [max(birth, 0)] + own_divs + [death]
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            if e <= s:
                continue
            event_frame = boundaries[i]
            if i == 0:
                if is_mother:
                    color = OI["vermilion"]
                    counter["mother"] += 1
                    label_tag = "mother"
                else:
                    self_r = _row_at(cid, event_frame)
                    other_r = _row_at(parent, event_frame)
                    tag = _compare_mean_ri(self_r, other_r)
                    color = OI["orange"] if tag == "higher" else OI["blue"] if tag == "lower" else OI["gray"]
                    counter[tag] += 1
                    label_tag = tag
            else:
                daughter_id = next(
                    (c_id for (c_id, cb) in children_map.get(cid, []) if cb == event_frame),
                    None,
                )
                if daughter_id is None:
                    tag = "undetermined"
                else:
                    self_r = _row_at(cid, event_frame)
                    other_r = _row_at(daughter_id, event_frame)
                    tag = _compare_mean_ri(self_r, other_r)
                color = OI["orange"] if tag == "higher" else OI["blue"] if tag == "lower" else OI["gray"]
                counter[tag] += 1
                label_tag = tag
            ax.plot([x, x], [to_h(s), to_h(e)],
                    color=color, linewidth=lw, solid_capstyle="butt", zorder=2)

    # mother label
    mother_rows = clist[(clist["in_tree"]) & (clist["mother_id"] == -1)]
    if not mother_rows.empty:
        mid = int(mother_rows.iloc[0]["cell_id"])
        if mid in positions:
            ax.text(positions[mid], -0.4, "M", ha="center", va="bottom",
                    fontsize=7, fontweight="bold")

    ax.invert_yaxis()
    ax.set_xlabel("lineage (M = mother, branches to the right)")
    ax.set_ylabel("time [h]" if time_interval_min else "frame")
    ax.set_xticks([])
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=OI["vermilion"], lw=1.4, label="mother"),
        plt.Line2D([0], [0], color=OI["orange"], lw=1.0,
                   label=f"higher mean RI at event (n={counter['higher']})"),
        plt.Line2D([0], [0], color=OI["blue"], lw=1.0,
                   label=f"lower mean RI at event (n={counter['lower']})"),
    ]
    if counter["undetermined"]:
        handles.append(
            plt.Line2D([0], [0], color=OI["gray"], lw=1.0,
                       label=f"undetermined (n={counter['undetermined']})")
        )
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="lower right")
    ax.set_title(f"{label}  lineage tree", fontsize=9)
    return fig


# =============================================================================
# Main
# =============================================================================
def run(channel_dir: Path) -> None:
    data3D, clist, run_meta = load_tables(channel_dir)
    label = channel_label(channel_dir)

    time_interval_min = run_meta.get("time_interval_min")
    media_schedule = parse_schedule_str(run_meta.get("media_schedule"))
    media_ri = run_meta.get("media_ri") or {}

    div_frames = mother_division_frames(clist)
    print(f"[info] {label}: n_divisions(mother)={len(div_frames)}", file=sys.stderr)

    params = {
        "channel_dir": str(channel_dir),
        "channel_label": label,
        "n_frames": int(data3D["frame"].max()) + 1 if len(data3D) else 0,
        "n_mother_divisions": int(len(div_frames)),
        "calibration_id": run_meta.get("calibration_id"),
        "media_schedule": run_meta.get("media_schedule"),
        "n_milliq": run_meta.get("n_milliq"),
        "time_interval_min": time_interval_min,
    }
    data_source = {"raw_files": [
        str(channel_dir / "inference_out" / "lineage_out" / "lineage_data3D.csv"),
        str(channel_dir / "inference_out" / "lineage_out" / "clist.csv"),
    ]}

    fig_v = fig_mother_volume(data3D, div_frames, media_schedule, media_ri,
                              time_interval_min, label)
    save_figure(fig_v, params=params, description=f"{label} mother volume vs time",
                fmt="pdf", data_source=data_source)
    plt.close(fig_v)

    fig_r = fig_mother_mean_ri(data3D, div_frames, media_schedule, media_ri,
                               time_interval_min, label)
    save_figure(fig_r, params=params, description=f"{label} mother mean RI vs time",
                fmt="pdf", data_source=data_source)
    plt.close(fig_r)

    fig_t = fig_lineage_tree(clist, data3D, time_interval_min, label)
    save_figure(fig_t, params=params, description=f"{label} lineage tree",
                fmt="pdf", data_source=data_source)
    plt.close(fig_t)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", type=Path, required=True,
                   help="Channel directory containing inference_out/lineage_out/")
    return p


def main() -> int:
    args = build_parser().parse_args()
    run(args.indir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
