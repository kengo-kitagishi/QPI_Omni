"""Mean RI at media-switch frames, EXACT frame value (no +/-window averaging).

Same 3-panel dead/alive histogram as
analyze_starvation_entry_cell_cycle.plot_ri_at_media_switches, but each mother's
value is mean_ri at EXACTLY the switch frame (2018/2306/2884). A mother is
dropped only if that exact frame is missing or bad (is_outlier / touches_border
/ mass < 10 pg). No RI_WINDOW averaging.

Run with QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd so the EFD lineage is read.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent))
from analyze_starvation_entry_cell_cycle import (  # noqa: E402
    find_lineage_csv, list_revived_mothers, list_never_revived_mothers,
    LOW_MASS_PG,
)
from figure_logger import save_figure  # noqa: E402

SWITCHES = [
    (2018, "2 % -> 0.0055 %  (frame 2018)"),
    (2306, "0.0055 % -> 0 %  (frame 2306)"),
    (2884, "0 % -> 2 % recovery  (frame 2884)"),
]


def ri_exact(pos, ch, frame):
    """mean_ri of the rank=1 mother at EXACTLY `frame`, or None if absent/bad."""
    p = find_lineage_csv(pos, ch)
    if p is None:
        return None
    df = pd.read_csv(p)
    m = df[(df["rank"] == 1) & (df["frame"] == frame)]
    if m.empty:
        return None
    row = m.iloc[0]
    if (bool(row["is_outlier"]) or bool(row["touches_border"])
            or row["mass_pg"] < LOW_MASS_PG):
        return None
    v = float(row["mean_ri"])
    return v if np.isfinite(v) else None


def vals(chs, frame):
    out = [ri_exact(p, c, frame) for p, c in chs]
    return np.array([v for v in out if v is not None])


def run():
    rev = list_revived_mothers()
    nr = list_never_revived_mothers()
    print(f"QPI_USE_CORRECTED={os.environ.get('QPI_USE_CORRECTED')} "
          f"QPI_VOLUME_VARIANT={os.environ.get('QPI_VOLUME_VARIANT')}")
    print(f"cohort: revived(alive)={len(rev)}, never_revived(dead)={len(nr)}")

    fig, axes = plt.subplots(len(SWITCHES), 1, figsize=(120 / 25.4, 130 / 25.4),
                             sharex=True, constrained_layout=True)
    bins = np.linspace(1.345, 1.385, 29)
    data = {}
    for row, (frame, label) in enumerate(SWITCHES):
        r = vals(rev, frame)
        n = vals(nr, frame)
        p = float(ks_2samp(r, n).pvalue) if len(r) and len(n) else float("nan")
        print(f"[frame {frame}] {label}")
        print(f"   alive(revived)   n={len(r):3d}  mean={np.mean(r):.5f}  "
              f"median={np.median(r):.5f}")
        print(f"   dead(never_rev)  n={len(n):3d}  mean={np.mean(n):.5f}  "
              f"median={np.median(n):.5f}")
        print(f"   KS p = {p:.4g}")

        ax = axes[row]
        ax.hist(r, bins=bins, color="#1a1a1a", alpha=0.55, edgecolor="white",
                linewidth=0.4,
                label=f"revived (n={len(r)}, mean={np.mean(r):.4f})")
        ax.hist(n, bins=bins, color="#d62728", alpha=0.55, edgecolor="white",
                linewidth=0.4,
                label=f"never_revived (n={len(n)}, mean={np.mean(n):.4f})")
        if len(r):
            ax.axvline(np.mean(r), color="#1a1a1a", lw=0.8, alpha=0.9)
        if len(n):
            ax.axvline(np.mean(n), color="#8c0000", lw=0.8, alpha=0.9)
        ax.set_title(f"mean RI AT frame {frame} (exact, no window)   {label}   "
                     f"KS p = {p:.3g}", fontsize=7)
        ax.set_ylabel("count", fontsize=8)
        ax.legend(loc="upper right", frameon=False, fontsize=6)
        ax.tick_params(labelsize=7)
        data[f"revived_f{frame}"] = r
        data[f"never_revived_f{frame}"] = n

    axes[-1].set_xlabel("mean RI", fontsize=8)
    fig.suptitle("Mean RI at medium-switch frames (exact frame, no window)",
                 fontsize=9)
    save_figure(
        fig,
        params={
            "media_switch_frames": [f for f, _ in SWITCHES],
            "ri_window_frames": 0,
            "low_mass_pg": LOW_MASS_PG,
            "volume_method": "efd",
            "value": "mean_ri at exact switch frame; missing/bad frame dropped",
        },
        description="mean RI distributions AT each medium-switch frame (exact "
                    "frame value, no +/-window averaging), revived vs "
                    "never_revived (EFD volume)",
        data=data,
    )
    plt.close(fig)


if __name__ == "__main__":
    run()
