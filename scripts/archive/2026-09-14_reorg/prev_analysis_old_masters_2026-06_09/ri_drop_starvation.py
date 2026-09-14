"""Per-mother RI change across starvation, revived vs never_revived.

Two definitions (two figures):
  fig1: dRI = mean_ri(frame 2884, last 0% frame) - mean_ri(frame 2018, phase1 end)
  fig2: dRI = mean_ri over frames [2864, 2874] - mean_ri(frame 2018)
        (window average just before 2884, more robust than a single frame)

Reads the mother (rank=1) mean_ri via the corrected-volume-aware lineage loader,
so run with QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd to use the EFD lineage.
Bad frames (is_outlier / touches_border / EFD mass < 10 pg) are dropped.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overlay_gold_standard_and_phase1_dead import find_lineage_csv  # noqa: E402
from overlay_mean_sd_band_full_timecourse import (  # noqa: E402
    list_revived_mothers, list_never_revived_mothers,
)
from figure_logger import save_figure  # noqa: E402

F_PHASE1_END = 2018      # last phase1 (2%) frame; starvation 0.0055% starts 2019
F_0PER_END = 2884        # last 0% frame; 2% recovery starts 2885
WINDOW = (2864, 2874)
LOW_MASS_PG = 10.0
TOL = 8                  # accept nearest good frame within +/- TOL of the target


def ri_series(pos, ch):
    p = find_lineage_csv(pos, ch)
    if p is None:
        return None
    df = pd.read_csv(p)
    m = df[df["rank"] == 1].sort_values("frame")
    if m.empty:
        return None
    bad = (m["is_outlier"].astype(bool) | m["touches_border"].astype(bool)
           | (m["mass_pg"] < LOW_MASS_PG)).to_numpy()
    ri = m["mean_ri"].to_numpy(float)
    ri[bad] = np.nan
    return m["frame"].to_numpy(int), ri


def ri_at(frames, ri, target, tol=TOL):
    idx = np.where(np.isfinite(ri))[0]
    if idx.size == 0:
        return np.nan
    j = idx[np.argmin(np.abs(frames[idx] - target))]
    return ri[j] if abs(frames[j] - target) <= tol else np.nan


def ri_window(frames, ri, f0, f1):
    v = ri[(frames >= f0) & (frames <= f1)]
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else np.nan


def drops(channels):
    single, avg = [], []
    for pos, ch in channels:
        s = ri_series(pos, ch)
        if s is None:
            continue
        fr, ri = s
        r0 = ri_at(fr, ri, F_PHASE1_END)
        if not np.isfinite(r0):
            continue
        r_end = ri_at(fr, ri, F_0PER_END)
        r_win = ri_window(fr, ri, *WINDOW)
        if np.isfinite(r_end):
            single.append(r_end - r0)
        if np.isfinite(r_win):
            avg.append(r_win - r0)
    return np.array(single), np.array(avg)


def plot_hist(rvals, nvals, xlabel, description):
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    p_ks = float(ks_2samp(rvals, nvals).pvalue) if len(rvals) and len(nvals) else np.nan
    lo = min(rvals.min(), nvals.min())
    hi = max(rvals.max(), nvals.max())
    bins = np.linspace(lo, hi, 28)
    ax.hist(rvals, bins=bins, alpha=0.55, color="#4a4a4a", density=True,
            label=f"revived (n={len(rvals)}, med={np.median(rvals):+.4f})")
    ax.hist(nvals, bins=bins, alpha=0.55, color="#d62728", density=True,
            label=f"never_revived (n={len(nvals)}, med={np.median(nvals):+.4f})")
    ax.axvline(np.median(rvals), color="#1a1a1a", ls="--", lw=1)
    ax.axvline(np.median(nvals), color="#8c0000", ls="--", lw=1)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"Mother RI change across starvation  (KS p = {p_ks:.2g})", fontsize=9)
    fig.tight_layout()
    save_figure(fig, params={
        "definition": description, "n_revived": len(rvals),
        "n_never_revived": len(nvals),
        "median_revived": float(np.median(rvals)),
        "median_never_revived": float(np.median(nvals)),
        "ks_p": p_ks, "volume_method": "efd",
        "frame_phase1_end": F_PHASE1_END, "frame_0per_end": F_0PER_END,
        "window": list(WINDOW),
    }, description=description)
    plt.close(fig)
    return p_ks


def main():
    rev = list_revived_mothers()
    nr = list_never_revived_mothers()
    print(f"revived={len(rev)}, never_revived={len(nr)}")
    r_single, r_avg = drops(rev)
    n_single, n_avg = drops(nr)

    p1 = plot_hist(r_single, n_single,
                   "ΔRI = mean_ri(frame 2884) − mean_ri(frame 2018)",
                   "RI change across starvation: RI(2884)-RI(2018), revived vs never_revived (EFD)")
    print(f"[fig1 single 2884] revived med={np.median(r_single):+.5f} (n={len(r_single)})  "
          f"dead med={np.median(n_single):+.5f} (n={len(n_single)})  KS p={p1:.3g}")

    p2 = plot_hist(r_avg, n_avg,
                   "ΔRI = mean_ri(frames 2864–2874) − mean_ri(frame 2018)",
                   "RI change across starvation: mean RI(2864-2874)-RI(2018), revived vs never_revived (EFD)")
    print(f"[fig2 window 2864-2874] revived med={np.median(r_avg):+.5f} (n={len(r_avg)})  "
          f"dead med={np.median(n_avg):+.5f} (n={len(n_avg)})  KS p={p2:.3g}")


if __name__ == "__main__":
    main()
