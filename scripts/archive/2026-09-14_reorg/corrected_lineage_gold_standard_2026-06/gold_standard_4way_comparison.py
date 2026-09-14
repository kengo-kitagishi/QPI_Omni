"""gold_standard_4way_comparison.py — validate the rod-axis correction.

Cycle-aligned comparison of FOUR estimates of the mother's short axis, long
axis and rod volume across the gold-standard phase1 cohort:

  - skimage_raw          : the biased second-moment ellipse fit (current CSV)
  - rod_corrected        : theoretical correction (rod_axis_correction)
  - supersegger_adaptive : mask-direct, adaptive endcap trim (measured)
  - medial_axis          : mask-direct, Morphometrics-equivalent (measured)

If the correction is right, the two measured curves and the theoretical curve
overlap, and only skimage_raw drifts upward (~+6 % within-cycle on short axis).

Inputs:
  - results/260517/recomputed_axes/<pos>_<ch>.csv  (from batch_recompute_axes)
  - lineage_data3D.csv (for divisions + bad-frame masking)

Usage:
    python scripts/gold_standard_4way_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from qpi_paths import find_lineage_csv, find_clist_csv, results_dir  # noqa: E402
from gold_standard import select_gold_standard, rank1_division_frames  # noqa: E402
from rod_axis_correction import correction_factors  # noqa: E402
from figure_logger import save_figure  # noqa: E402

PHASE1_END_FRAME = 2018
N_REL = 100
LOW_MASS_PG = 10.0

# curve key -> (display label, color, linestyle)
CURVES = {
    "skimage_raw":          ("skimage raw (biased)", "#0072B2", "--"),
    "rod_corrected":        ("rod-corrected (theory)", "#1a1a1a", "-"),
    "supersegger_adaptive": ("supersegger adaptive (measured)", "#009E73", "-"),
    "medial_axis":          ("medial axis (measured)", "#D55E00", "-"),
    "efd":                  ("EFD contour-section (measured)", "#CC79A7", "-"),
}
PANELS = [
    ("short", "short axis [μm]"),
    ("long", "long axis [μm]"),
    ("volume", r"rod volume [μm$^3$]"),
]


def _rod_volume(L_um, w_um):
    r = w_um / 2.0
    h = np.maximum(L_um - 2.0 * r, 0.0)
    return (4.0 / 3.0) * np.pi * r ** 3 + np.pi * r ** 2 * h


def collect_cycles(channels):
    rel = np.linspace(0, 1, N_REL)
    # rows[curve][metric] = list of per-cycle interpolated arrays
    rows = {c: {m: [] for m, _ in PANELS} for c in CURVES}
    rec_dir = results_dir() / "recomputed_axes"
    rec_dir_efd = results_dir() / "recomputed_axes_efd"
    n_ch = 0
    for pos, ch in channels:
        rec_path = rec_dir / f"{pos}_{ch}.csv"
        lin_path = find_lineage_csv(pos, ch)
        if not rec_path.exists() or lin_path is None:
            continue
        rec = pd.read_csv(rec_path)
        lin = pd.read_csv(lin_path)
        clist_path = find_clist_csv(lin_path)
        clist = pd.read_csv(clist_path) if clist_path else None
        divs = rank1_division_frames(lin, clist)

        # bad-frame mask from the lineage (applied to all curves consistently)
        linm = lin[lin["rank"] == 1].sort_values("frame")
        bad = (linm["is_outlier"].astype(bool)
               | linm["touches_border"].astype(bool)
               | (linm["mass_pg"] < LOW_MASS_PG))
        bad_frames = set(linm["frame"][bad].astype(int))

        # build per-frame curve values
        sk = rec[rec["mode"] == "skimage_legacy"].set_index("frame")
        ad = rec[rec["mode"] == "supersegger_adaptive"].set_index("frame")
        md = rec[rec["mode"] == "medial_axis"].set_index("frame")
        if sk.empty or ad.empty or md.empty:
            continue
        # EFD contour-section recompute (optional 5th curve)
        efd = None
        efd_path = rec_dir_efd / f"{pos}_{ch}.csv"
        if efd_path.exists():
            edf = pd.read_csv(efd_path)
            edf = edf[edf["mode"] == "medial_axis"].set_index("frame")
            if not edf.empty and "volume_efd_um3" in edf.columns:
                efd = edf
        base = set(sk.index) & set(ad.index) & set(md.index)
        if efd is not None:
            base &= set(efd.index)
        frames = sorted(base)
        frames = [f for f in frames if f not in bad_frames]
        if not frames:
            continue
        fidx = np.array(frames)

        sk_short = sk.loc[frames, "short_axis_um"].to_numpy()
        sk_long = sk.loc[frames, "long_axis_um"].to_numpy()
        # theoretical correction from skimage axes
        L_corr, w_corr = correction_factors(sk_long, sk_short)
        series = {
            "skimage_raw": {
                "short": sk_short, "long": sk_long,
                "volume": sk.loc[frames, "volume_um3_rod"].to_numpy(),
            },
            "rod_corrected": {
                "short": w_corr, "long": L_corr, "volume": _rod_volume(L_corr, w_corr),
            },
            "supersegger_adaptive": {
                "short": ad.loc[frames, "short_axis_um"].to_numpy(),
                "long": ad.loc[frames, "long_axis_um"].to_numpy(),
                "volume": ad.loc[frames, "volume_um3_rod"].to_numpy(),
            },
            "medial_axis": {
                "short": md.loc[frames, "short_axis_um"].to_numpy(),
                "long": md.loc[frames, "long_axis_um"].to_numpy(),
                "volume": md.loc[frames, "volume_um3_rod"].to_numpy(),
            },
        }
        if efd is not None:
            series["efd"] = {
                "short": efd.loc[frames, "short_axis_efd_um"].to_numpy(),
                "long": efd.loc[frames, "long_axis_efd_um"].to_numpy(),
                "volume": efd.loc[frames, "volume_efd_um3"].to_numpy(),
            }
        # cycle-align each interval
        used = False
        for i in range(len(divs) - 1):
            f0, f1 = divs[i], divs[i + 1]
            if f1 > PHASE1_END_FRAME:
                continue
            sel = (fidx >= f0) & (fidx < f1)
            if sel.sum() < 4:
                continue
            t_rel = (fidx[sel] - f0) / (f1 - f0)
            for curve in CURVES:
                if curve not in series:
                    continue
                for metric, _ in PANELS:
                    y = series[curve][metric][sel]
                    rows[curve][metric].append(np.interp(rel, t_rel, y))
            used = True
        if used:
            n_ch += 1
    return rel, rows, n_ch


def plot(rel, rows, n_ch):
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(130 / 25.4, 175 / 25.4),
                             sharex=True, constrained_layout=True)
    n_cycles = 0
    for ax, (metric, ylabel) in zip(axes, PANELS):
        for curve, (label, color, ls) in CURVES.items():
            stack = rows[curve][metric]
            if not stack:
                continue
            M = np.vstack(stack)
            n_cycles = max(n_cycles, M.shape[0])
            mean = np.nanmean(M, axis=0)
            sd = np.nanstd(M, axis=0)
            var = 100 * (mean.max() - mean.min()) / mean.mean()
            ax.fill_between(rel, mean - sd, mean + sd, color=color,
                            alpha=0.10, linewidth=0)
            ax.plot(rel, mean, color=color, lw=1.3, ls=ls,
                    label=f"{label}  Δ={var:.1f}%")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="upper left", frameon=False, fontsize=5.6)
    axes[-1].set_xlabel("relative cell cycle progression", fontsize=8)
    fig.suptitle("Validation of rod-axis correction: theoretical vs measured\n"
                 f"gold-standard phase1 cohort (n={n_ch} ch, {n_cycles} cycles)",
                 fontsize=8.5)
    save_figure(
        fig,
        params={"n_channels": n_ch, "n_cycles_pooled": int(n_cycles),
                "phase1_end_frame": PHASE1_END_FRAME,
                "curves": "skimage_raw / rod_corrected / supersegger_adaptive / medial_axis"},
        description="gold-standard 4-way cycle-aligned axis/volume comparison",
    )
    plt.close(fig)
    return n_cycles


def main():
    gold = select_gold_standard(verbose=True)
    rel, rows, n_ch = collect_cycles(gold)
    print(f"channels with usable recomputed cycles: {n_ch}")
    # console summary of within-cycle variation
    print(f"\n{'metric':8s} {'curve':22s} {'rel=0':>8} {'rel=1':>8} {'within-cycle %':>14}")
    for metric, _ in PANELS:
        for curve in CURVES:
            stack = rows[curve][metric]
            if not stack:
                continue
            mean = np.nanmean(np.vstack(stack), axis=0)
            var = 100 * (mean.max() - mean.min()) / mean.mean()
            print(f"{metric:8s} {curve:22s} {mean[0]:8.3f} {mean[-1]:8.3f} {var:14.2f}")
    n_cycles = plot(rel, rows, n_ch)
    print(f"\npooled cycles: {n_cycles}")


if __name__ == "__main__":
    main()
