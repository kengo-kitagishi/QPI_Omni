"""
mother_cell_cycle_stats.py — Cell-cycle homeostasis statistics for the mother cell.

Reads the lineage tracker output (masks.tif -> lineage tree):
    inference_out/lineage_out/clist.csv
    inference_out/lineage_out/lineage_data3D.csv

Filters to the mother cell (cell_id == 0, parent_id == -1) and uses its
own divisions (= birth_frames of its direct daughters) to segment the
timeseries into cell cycles. Outlier / border-touching frames are masked
via NaN in the lineage table, so statistics are computed on valid frames
only.

Generates qpi_fig_03-style figures, rewritten for a single mother lineage:
  - mother timeseries (volume / mass / mean_ri) with division ticks + media epochs
  - birth vs added homeostasis (volume / mass / RI), 3-panel scatter
  - division interval histogram
  - cycle-aligned trajectories (volume / mass / RI), mean ± SD
  - mother RI distribution + Gaussian fit
  - density homeostasis (birth_ri vs added_ri, with regression)
  - specific growth rate of volume and mass over relative cycle

Example:
    /opt/anaconda3/bin/python mother_cell_cycle_stats.py \\
        --indir /Volumes/2604/260405/ph_260405/Pos9/output_phase/channels/crop_sub_rawraw/ch01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from figure_logger import save_figure

# =============================================================================
# Constants (match lineage_survival_analysis.py defaults for Pos9/260405)
# =============================================================================
FRAMES_PER_HOUR = 12.0
STARV_START_FRAME = 575
SWITCH_MID_FRAME = 863
REC_START_FRAME = 1439
N_MEDIUM = 1.333
N_INTERP = 100

# Okabe-Ito palette (colorblind-safe)
OI = {
    "orange":    "#E69F00",
    "skyblue":   "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
    "black":     "#000000",
}
EPOCH_COLORS = {"pre": OI["blue"], "starv": OI["orange"], "rec": OI["green"]}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def frame_to_h(f: float) -> float:
    return float(f) / FRAMES_PER_HOUR


def epoch_of(frame: int) -> str:
    if frame < STARV_START_FRAME:
        return "pre"
    if frame < REC_START_FRAME:
        return "starv"
    return "rec"


# =============================================================================
# I/O + cycle extraction
# =============================================================================
def _find_false_births(clist: pd.DataFrame, data3D: pd.DataFrame) -> set[int]:
    """Same rule as lineage_survival_analysis.py: exclude daughters whose
    birth_frame lands on, or directly after, a parent's outlier frame.
    The rank-pointer tracker detects divisions from frame-to-frame area
    ratios before outlier flags are assigned, so a segmentation blip that
    inflates the parent's area can trigger a spurious division at that
    frame or the next.
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


def source_tag(channel_dir: Path) -> str:
    """Short identifier like 'Pos9/ch01' from a channel dir path."""
    parts = channel_dir.parts
    ch = channel_dir.name
    pos = next((p for p in parts if p.startswith("Pos")), "Pos?")
    return f"{pos}/{ch}"


def load_mother_cycles(channel_dir: Path,
                       max_frame: int | None = None,
                       ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Load lineage tables, filter to mother timeseries, extract cycles.

    A "cycle" spans [div_frames[i], div_frames[i+1] - 1]. Birth/division
    values are picked from the mother's own timeseries. Outlier / border
    frames are NaN in lineage_data3D; we tolerate that by falling back to
    the nearest valid frame within the cycle window.
    """
    out_dir = channel_dir / "inference_out" / "lineage_out"
    clist = pd.read_csv(out_dir / "clist.csv")
    data3D = pd.read_csv(out_dir / "lineage_data3D.csv")
    src = source_tag(channel_dir)

    false_ids = _find_false_births(clist, data3D)
    if false_ids:
        print(f"[info] false-division daughters excluded (birth on parent outlier frame): n={len(false_ids)}",
              file=sys.stderr)
    clist = clist[~clist["cell_id"].isin(false_ids)].copy()

    mother_rows = clist[clist["mother_id"] == -1]
    if mother_rows.empty:
        raise RuntimeError("no mother cell (mother_id == -1) in clist.csv")
    mother_id = int(mother_rows.iloc[0]["cell_id"])

    m_df = data3D[data3D["cell_id"] == mother_id].sort_values("frame").reset_index(drop=True)
    m_df = m_df.assign(source=src)

    daughters = clist[clist["mother_id"] == mother_id].sort_values("birth_frame")
    div_frames = daughters["birth_frame"].astype(int).tolist()

    # Index mother rows by frame for direct lookup
    m_idx = m_df.set_index("frame", drop=False)

    def _valid_row(f: int):
        if f not in m_idx.index:
            return None
        r = m_idx.loc[f]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        if bool(r["is_outlier"]) or bool(r["touches_border"]):
            return None
        return r

    cycles = []
    for i in range(len(div_frames) - 1):
        f_birth = div_frames[i]
        f_next = div_frames[i + 1]
        if max_frame is not None and f_next > max_frame:
            continue
        r_birth = _valid_row(f_birth)
        r_div = _valid_row(f_next)
        if r_birth is None or r_div is None:
            continue
        cycles.append({
            "source": src,
            "cycle_idx": i,
            "birth_frame": int(f_birth),
            "div_frame": int(f_next),
            "birth_time_h": frame_to_h(f_birth),
            "interval_h": frame_to_h(f_next - f_birth),
            "birth_epoch": epoch_of(f_birth),
            "birth_volume_um3":  float(r_birth["volume_um3_rod"]),
            "div_volume_um3":    float(r_div["volume_um3_rod"]),
            "added_volume_um3":  float(r_div["volume_um3_rod"] - r_birth["volume_um3_rod"]),
            "birth_mass_pg":     float(r_birth["mass_pg"]),
            "div_mass_pg":       float(r_div["mass_pg"]),
            "added_mass_pg":     float(r_div["mass_pg"] - r_birth["mass_pg"]),
            "birth_ri":          float(r_birth["mean_ri"]),
            "div_ri":            float(r_div["mean_ri"]),
            "added_ri":          float(r_div["mean_ri"] - r_birth["mean_ri"]),
        })
    return m_df, clist, cycles


def extract_cycle_traces(m_df: pd.DataFrame, cycles: list[dict],
                         n_interp: int = N_INTERP) -> list[dict]:
    """Interpolate volume/mass/RI onto relative cycle progression [0,1]."""
    rel = np.linspace(0, 1, n_interp)
    traces = []
    src = m_df["source"].iloc[0] if "source" in m_df.columns and len(m_df) else "?"
    for c in cycles:
        # Cycle spans [birth_frame, div_frame - 1] = [division frame, next division frame - 1]
        f0 = float(c["birth_frame"])
        f1 = float(c["div_frame"] - 1)
        if f1 <= f0:
            continue
        sub = m_df[(m_df["frame"] >= f0) & (m_df["frame"] <= f1)]
        sub = sub[~(sub["is_outlier"] | sub["touches_border"])]
        if len(sub) < 4:
            continue
        t = sub["frame"].to_numpy().astype(float)
        t_rel = (t - f0) / (f1 - f0)
        traces.append({
            "source":       src,
            "rel_progress": rel,
            "birth_epoch":  c["birth_epoch"],
            "volume":       np.interp(rel, t_rel, sub["volume_um3_rod"].to_numpy()),
            "mass":         np.interp(rel, t_rel, sub["mass_pg"].to_numpy()),
            "ri":           np.interp(rel, t_rel, sub["mean_ri"].to_numpy()),
        })
    return traces


# =============================================================================
# Figures
# =============================================================================
def _draw_epoch_shading(ax, t_max_h: float):
    t_starv = frame_to_h(STARV_START_FRAME)
    t_mid = frame_to_h(SWITCH_MID_FRAME)
    t_rec = frame_to_h(REC_START_FRAME)
    ax.axvspan(0,       t_starv, color="#FAFAFA", zorder=0)
    ax.axvspan(t_starv, t_rec,   color="#FFF2CC", alpha=0.55, zorder=0)
    ax.axvspan(t_rec,   t_max_h, color="#E6F4EA", alpha=0.55, zorder=0)
    ax.axvline(t_mid, color="#888", lw=0.5, ls=":", zorder=1)
    ax.axvline(t_rec, color=OI["green"], lw=0.8, zorder=1)


def fig_mother_timeseries(m_df: pd.DataFrame, cycles: list[dict]) -> plt.Figure:
    """All-mother overlay: each mother faint, population mean thick."""
    sources = sorted(m_df["source"].unique()) if "source" in m_df.columns else ["?"]
    t_max = float(m_df["frame"].max()) / FRAMES_PER_HOUR

    fig, axes = plt.subplots(3, 1, figsize=(183/25.4, 120/25.4), sharex=True,
                             constrained_layout=True)
    panels = [
        (axes[0], "volume_um3_rod", r"volume [µm$^3$]", OI["blue"]),
        (axes[1], "mass_pg",        "dry mass [pg]",     OI["orange"]),
        (axes[2], "mean_ri",        "mean RI",            OI["green"]),
    ]
    for ax, _, _, _ in panels:
        _draw_epoch_shading(ax, t_max)

    for src in sources:
        sub = m_df[m_df["source"] == src].sort_values("frame")
        bad = sub["is_outlier"].to_numpy() | sub["touches_border"].to_numpy()
        t = sub["frame"].to_numpy() / FRAMES_PER_HOUR
        for ax, col, _, color in panels:
            y = np.where(bad, np.nan, sub[col].to_numpy())
            ax.plot(t, y, color=color, lw=0.5, alpha=0.45, zorder=3)

    # population mean per frame (valid only)
    valid = m_df[~(m_df["is_outlier"] | m_df["touches_border"])]
    grp = valid.groupby("frame")
    for ax, col, ylab, _ in panels:
        mean = grp[col].mean()
        ax.plot(mean.index / FRAMES_PER_HOUR, mean.values,
                color="#222", lw=1.2, zorder=5, label=f"mean (n_mothers={len(sources)})")
        ax.set_ylabel(ylab)
        ax.legend(loc="upper right", frameon=False)
    axes[2].set_xlabel("time [h]")
    axes[0].set_title(f"Mother trajectories (n_mothers={len(sources)}, "
                      f"n_cycles={len(cycles)})")
    return fig


def _scatter_by_epoch(ax, df, xcol, ycol):
    for ep, color in EPOCH_COLORS.items():
        sub = df[df["birth_epoch"] == ep]
        if sub.empty:
            continue
        ax.scatter(sub[xcol], sub[ycol], color=color, alpha=0.75, s=18,
                   edgecolor="white", linewidth=0.4,
                   label=f"{ep} (n={len(sub)})")


def fig_homeostasis(cycles: list[dict]) -> plt.Figure | None:
    if len(cycles) < 3:
        print("  [homeostasis] n_cycles < 3, skipping", file=sys.stderr)
        return None
    df = pd.DataFrame(cycles).dropna(
        subset=["birth_volume_um3", "added_volume_um3",
                "birth_mass_pg", "added_mass_pg",
                "birth_ri", "added_ri"]
    )
    if len(df) < 3:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(183/25.4, 65/25.4),
                             constrained_layout=True)
    panels = [
        ("birth_volume_um3", "added_volume_um3",
         r"birth volume [µm$^3$]", r"added volume [µm$^3$]"),
        ("birth_mass_pg", "added_mass_pg",
         "birth dry mass [pg]", "added dry mass [pg]"),
        ("birth_ri", "added_ri",
         "birth mean RI", "Δ mean RI"),
    ]
    for ax, (xc, yc, xl, yl) in zip(axes, panels):
        _scatter_by_epoch(ax, df, xc, yc)
        r, p = pearsonr(df[xc], df[yc])
        z = np.polyfit(df[xc], df[yc], 1)
        xline = np.linspace(df[xc].min(), df[xc].max(), 50)
        ax.plot(xline, np.polyval(z, xline), color="#333", lw=0.8, ls="--",
                label=f"slope={z[0]:.2g}")
        ax.axhline(0, color="#000", lw=0.3, zorder=0)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"r={r:.2f}, p={p:.2e}")
    axes[0].legend(loc="best", frameon=False)
    return fig


def fig_interval_hist(cycles: list[dict]) -> plt.Figure | None:
    intervals = [c["interval_h"] for c in cycles]
    if len(intervals) < 3:
        return None
    df = pd.DataFrame(cycles)
    iv = np.asarray(intervals)
    mu, sd = float(np.mean(iv)), float(np.std(iv, ddof=1))
    bins = np.arange(0, iv.max() + 1.0, 0.5)

    fig, ax = plt.subplots(figsize=(89/25.4, 65/25.4), constrained_layout=True)
    # stacked by epoch
    stack = [df[df["birth_epoch"] == e]["interval_h"].to_numpy()
             for e in ("pre", "starv", "rec")]
    labels = [f"pre (n={len(stack[0])})",
              f"starv (n={len(stack[1])})",
              f"rec (n={len(stack[2])})"]
    ax.hist(stack, bins=bins, stacked=True,
            color=[EPOCH_COLORS["pre"], EPOCH_COLORS["starv"], EPOCH_COLORS["rec"]],
            edgecolor="white", linewidth=0.3, label=labels)
    ax.axvline(mu, color=OI["vermilion"], lw=0.8, ls="--",
               label=f"mean = {mu:.2f} h")
    ax.set_xlabel("division interval [h]")
    ax.set_ylabel("count")
    ax.set_title(f"n={len(iv)}, mean={mu:.2f} ± {sd:.2f} h")
    ax.legend(frameon=False)
    return fig


def fig_aligned_trajectories(traces: list[dict]) -> plt.Figure | None:
    if len(traces) < 3:
        return None
    rel = traces[0]["rel_progress"]
    fig, axes = plt.subplots(3, 1, figsize=(89/25.4, 140/25.4), sharex=True,
                             constrained_layout=True)
    for ax, key, ylab, color in [
        (axes[0], "volume", r"volume [µm$^3$]", OI["blue"]),
        (axes[1], "mass",   "dry mass [pg]",     OI["orange"]),
        (axes[2], "ri",     "mean RI",            OI["green"]),
    ]:
        stack = np.array([t[key] for t in traces])
        mean = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0)
        for row in stack:
            ax.plot(rel, row, color=color, alpha=0.08, lw=0.3)
        ax.plot(rel, mean, color=color, lw=1.2, label=f"mean (n={len(stack)})")
        ax.fill_between(rel, mean - sd, mean + sd, color=color, alpha=0.2,
                        label="±1 SD")
        ax.set_ylabel(ylab)
        ax.legend(loc="upper left", frameon=False)
    axes[2].set_xlabel("relative cycle progression")
    axes[0].set_title("Cycle-aligned mother trajectories")
    return fig


def _gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fig_ri_distribution(m_df: pd.DataFrame, max_frame: int | None = None) -> plt.Figure | None:
    sub = m_df[~(m_df["is_outlier"] | m_df["touches_border"])]
    if max_frame is not None:
        sub = sub[sub["frame"] <= max_frame]
    ri = sub["mean_ri"].dropna().to_numpy()
    if len(ri) < 20:
        return None
    mu, sd = float(np.mean(ri)), float(np.std(ri))
    cv = sd / mu

    fig, ax = plt.subplots(figsize=(89/25.4, 65/25.4), constrained_layout=True)
    counts, edges, _ = ax.hist(ri, bins=50, density=True,
                                color=OI["skyblue"], edgecolor="white",
                                linewidth=0.3, alpha=0.8)
    centers = 0.5 * (edges[:-1] + edges[1:])
    try:
        popt, _ = curve_fit(_gaussian, centers, counts,
                            p0=[counts.max(), mu, sd])
        xfit = np.linspace(ri.min(), ri.max(), 200)
        ax.plot(xfit, _gaussian(xfit, *popt), color=OI["vermilion"], lw=1.0,
                label=f"µ={popt[1]:.4f}, σ={popt[2]:.4f}")
    except RuntimeError:
        pass
    ax.set_xlabel("mean RI")
    ax.set_ylabel("probability density")
    ax.set_title(f"mother RI distribution (n={len(ri)}, CV={cv:.3f})")
    ax.legend(frameon=False)
    return fig


def fig_density_homeostasis(cycles: list[dict]) -> plt.Figure | None:
    df = pd.DataFrame(cycles).dropna(subset=["birth_ri", "added_ri"])
    if len(df) < 3:
        return None
    r, p = pearsonr(df["birth_ri"], df["added_ri"])
    z = np.polyfit(df["birth_ri"], df["added_ri"], 1)

    fig, ax = plt.subplots(figsize=(89/25.4, 70/25.4), constrained_layout=True)
    _scatter_by_epoch(ax, df, "birth_ri", "added_ri")
    xline = np.linspace(df["birth_ri"].min(), df["birth_ri"].max(), 50)
    ax.plot(xline, np.polyval(z, xline), color="#333", lw=0.8, ls="--",
            label=f"slope={z[0]:.2f}")
    ax.axhline(0, color="#000", lw=0.3, zorder=0)
    ax.set_xlabel("birth mean RI (initial density)")
    ax.set_ylabel("Δ mean RI over cycle")
    ax.set_title(f"density homeostasis (r={r:.2f}, p={p:.2e})")
    ax.legend(frameon=False)
    return fig


def fig_growth_rate(traces: list[dict]) -> plt.Figure | None:
    if len(traces) < 3:
        return None
    rel = traces[0]["rel_progress"]
    dr = rel[1] - rel[0]
    vol_rates, mass_rates = [], []
    for t in traces:
        v = np.asarray(t["volume"]); m = np.asarray(t["mass"])
        if np.any(v <= 0) or np.any(m <= 0):
            continue
        vol_rates.append(np.gradient(np.log(v), dr))
        mass_rates.append(np.gradient(np.log(m), dr))
    if not vol_rates:
        return None
    vol_rates = np.array(vol_rates)
    mass_rates = np.array(mass_rates)

    fig, axes = plt.subplots(2, 1, figsize=(89/25.4, 110/25.4), sharex=True,
                             constrained_layout=True)
    for ax, data, color, ylab, title in [
        (axes[0], vol_rates, OI["blue"],
         r"$d\ln V / d\tau$", "Volume specific growth rate"),
        (axes[1], mass_rates, OI["orange"],
         r"$d\ln M / d\tau$", "Dry-mass specific growth rate"),
    ]:
        mean = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        for row in data:
            ax.plot(rel, row, color=color, alpha=0.06, lw=0.3)
        ax.plot(rel, mean, color=color, lw=1.2)
        ax.fill_between(rel, mean - sd, mean + sd, color=color, alpha=0.2)
        ax.axhline(float(np.mean(mean)), color="#555", ls=":", lw=0.5)
        ax.set_ylabel(ylab)
        ax.set_title(title)
    axes[1].set_xlabel(r"relative cycle progression $\tau$")
    return fig


# =============================================================================
# Main
# =============================================================================
def run(channel_dirs: list[Path], starv_frame: int, rec_frame: int,
        max_frame: int | None = None) -> None:
    global STARV_START_FRAME, REC_START_FRAME
    STARV_START_FRAME = starv_frame
    REC_START_FRAME = rec_frame

    m_dfs: list[pd.DataFrame] = []
    all_cycles: list[dict] = []
    all_traces: list[dict] = []
    raw_files: list[str] = []
    for ch in channel_dirs:
        try:
            m_df, _clist, cycles = load_mother_cycles(ch, max_frame=max_frame)
        except Exception as e:
            print(f"[warn] skip {ch}: {e}", file=sys.stderr)
            continue
        if max_frame is not None:
            m_df = m_df[m_df["frame"] <= max_frame].reset_index(drop=True)
        traces = extract_cycle_traces(m_df, cycles)
        print(f"[info] {source_tag(ch)}  frames={len(m_df)}  cycles={len(cycles)}  traces={len(traces)}",
              file=sys.stderr)
        if not cycles:
            continue
        m_dfs.append(m_df)
        all_cycles.extend(cycles)
        all_traces.extend(traces)
        raw_files += [
            str(ch / "inference_out" / "lineage_out" / "clist.csv"),
            str(ch / "inference_out" / "lineage_out" / "lineage_data3D.csv"),
        ]

    if not all_cycles:
        print("[error] no cycles collected across sources", file=sys.stderr)
        return

    pooled_df = pd.concat(m_dfs, ignore_index=True)
    n_mothers = pooled_df["source"].nunique()
    print(f"[info] pooled: n_mothers={n_mothers}  n_cycles={len(all_cycles)}  "
          f"n_traces={len(all_traces)}", file=sys.stderr)

    params = {
        "channel_dirs": [str(c) for c in channel_dirs],
        "sources": sorted(pooled_df["source"].unique().tolist()),
        "n_mothers": int(n_mothers),
        "starvation_start_frame": STARV_START_FRAME,
        "recovery_start_frame": REC_START_FRAME,
        "frames_per_hour": FRAMES_PER_HOUR,
        "n_cycles": len(all_cycles),
        "n_traces": len(all_traces),
        "n_interp": N_INTERP,
        "max_frame": max_frame,
    }
    data_source = {"raw_files": raw_files}

    plots = [
        (fig_mother_timeseries(pooled_df, all_cycles),
         "pooled mother volume/mass/RI timeseries, all mothers overlaid + mean"),
        (fig_homeostasis(all_cycles),
         "pooled mother cell-cycle homeostasis: birth vs added (volume, mass, RI)"),
        (fig_interval_hist(all_cycles),
         "pooled mother division interval histogram, stacked by birth epoch"),
        (fig_aligned_trajectories(all_traces),
         "pooled cycle-aligned mother trajectories (volume/mass/RI, mean ± SD)"),
        (fig_ri_distribution(pooled_df, max_frame=max_frame),
         "pooled mother RI distribution with Gaussian fit"),
        (fig_density_homeostasis(all_cycles),
         "pooled density homeostasis: birth RI vs Δ RI"),
        (fig_growth_rate(all_traces),
         "pooled specific growth rate of volume and mass over relative cycle"),
    ]
    for fig, desc in plots:
        if fig is None:
            continue
        save_figure(fig, params=params, description=desc, fmt="pdf",
                    data_source=data_source)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indir", type=Path, action="append", default=None,
                   help="channel dir with inference_out/lineage_out/ (repeatable)")
    p.add_argument("--root", type=Path, default=None,
                   help="glob all Pos*/output_phase/channels/crop_sub_rawraw/ch*/inference_out/lineage_out/ under this root")
    p.add_argument("--exclude", action="append", default=[],
                   help="substring to exclude from discovered channel dirs (repeatable; e.g. 'ch10')")
    p.add_argument("--starvation-start-frame", type=int, default=STARV_START_FRAME)
    p.add_argument("--recovery-start-frame",   type=int, default=REC_START_FRAME)
    p.add_argument("--max-frame", type=int, default=None,
                   help="only include cycles whose division frame <= this (e.g. 575 for pre-starvation)")
    return p


def discover_channels(root: Path, exclude: list[str]) -> list[Path]:
    hits = sorted(root.glob("Pos*/output_phase/channels/crop_sub_rawraw/ch*"))
    channels = [p for p in hits
                if (p / "inference_out" / "lineage_out" / "clist.csv").is_file()
                and not any(ex in str(p) for ex in exclude)]
    return channels


def main() -> int:
    args = build_parser().parse_args()
    channel_dirs: list[Path] = list(args.indir or [])
    if args.root is not None:
        channel_dirs += discover_channels(args.root, args.exclude)
    if not channel_dirs:
        print("[error] no --indir and no --root produced any channel dirs", file=sys.stderr)
        return 1
    # apply --exclude to explicit --indir too
    channel_dirs = [c for c in channel_dirs
                    if not any(ex in str(c) for ex in args.exclude)]
    print(f"[info] processing {len(channel_dirs)} channel(s):", file=sys.stderr)
    for c in channel_dirs:
        print(f"       {source_tag(c)}  {c}", file=sys.stderr)
    run(channel_dirs, args.starvation_start_frame, args.recovery_start_frame,
        max_frame=args.max_frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
