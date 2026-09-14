"""Clean w=0.0020 figures (no 0 vertical line).

f001 = media-switch dead/alive histograms at the 3 switch frames (2018/2306/2884),
       exact frame / no window, vertical 3-panel, shared x.
f002 = RI-drop dead/alive histograms (single 2884, window 2864-2874), 2-panel.

Style: count y-axis, gray revived vs red dead, white edges, mean = solid line,
median = dashed line, legend with n/mean/median, KS in title. NO 0 line.

Run with QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent))
from _run_ri_at_switch_exact_efd import vals, SWITCHES  # noqa: E402
from ri_drop_starvation import (  # noqa: E402
    drops, list_revived_mothers, list_never_revived_mothers,
)
from figure_logger import save_figure  # noqa: E402

W = 0.0015
GRAY, RED, DRED = "#1a1a1a", "#d62728", "#8c0000"


def panel(ax, r, n, bins, title, xlabel=None):
    p = float(ks_2samp(r, n).pvalue) if len(r) and len(n) else float("nan")
    ax.hist(r, bins=bins, color=GRAY, alpha=0.55, edgecolor="white", linewidth=0.4,
            label=f"revived (n={len(r)}, mean={np.mean(r):+.4f}, "
                  f"med={np.median(r):+.4f})")
    ax.hist(n, bins=bins, color=RED, alpha=0.55, edgecolor="white", linewidth=0.4,
            label=f"never_revived (n={len(n)}, mean={np.mean(n):+.4f}, "
                  f"med={np.median(n):+.4f})")
    ax.axvline(np.mean(r), color=GRAY, lw=0.9)
    ax.axvline(np.mean(n), color=DRED, lw=0.9)
    ax.set_ylabel("count", fontsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    ax.set_title(f"{title}   KS p = {p:.3g}", fontsize=8)
    ax.tick_params(labelsize=7)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    return p


def fig_media_switch():
    rev, nr = list_revived_mothers(), list_never_revived_mothers()
    rows = [(f, lbl, vals(rev, f), vals(nr, f)) for f, lbl in SWITCHES]
    bins = np.arange(1.345, 1.385 + W / 2, W)
    fig, axes = plt.subplots(len(rows), 1, figsize=(6.4, 7.6), sharex=True,
                             constrained_layout=True)
    data = {}
    for i, (f, lbl, r, n) in enumerate(rows):
        xl = "mean RI" if i == len(rows) - 1 else None
        panel(axes[i], r, n, bins, f"frame {f}  {lbl}", xlabel=xl)
        data[f"revived_f{f}"] = r
        data[f"never_revived_f{f}"] = n
    data["bins"] = bins
    fig.suptitle("Mean RI at media-switch frames (exact, w=0.0020)", fontsize=9)
    save_figure(fig, params={"bin_width": W, "y_axis": "count",
                             "value": "exact frame, no window",
                             "zero_line": False, "volume_method": "efd"},
                description="Clean mean-RI-at-media-switch dead/alive histograms "
                            "(exact frame, w=0.0020, no 0 line), 3 frames",
                data=data)
    plt.close(fig)


def fig_ri_drop():
    rev, nr = list_revived_mothers(), list_never_revived_mothers()
    r_s, r_a = drops(rev)
    n_s, n_a = drops(nr)
    rows = [("single 2884", "dRI = RI(2884) - RI(2018)", r_s, n_s),
            ("window 2864-2874", "dRI = meanRI(2864-2874) - RI(2018)", r_a, n_a)]
    allv = np.concatenate([r_s, n_s, r_a, n_a])
    lo = np.floor(allv.min() / W) * W
    hi = np.ceil(allv.max() / W) * W
    bins = np.arange(lo, hi + W / 2, W)
    fig, axes = plt.subplots(len(rows), 1, figsize=(6.4, 6.2), sharex=True,
                             constrained_layout=True)
    for i, (key, xl, r, n) in enumerate(rows):
        panel(axes[i], r, n, bins, key,
              xlabel=xl if i == len(rows) - 1 else None)
    fig.suptitle("RI drop across starvation (w=0.0020, no 0 line)", fontsize=9)
    save_figure(fig, params={"bin_width": W, "y_axis": "count",
                             "zero_line": False, "volume_method": "efd"},
                description="Clean RI-drop dead/alive histograms (w=0.0020, no 0 "
                            "line), single 2884 + window 2864-2874",
                data={"revived_single": r_s, "never_revived_single": n_s,
                      "revived_window": r_a, "never_revived_window": n_a,
                      "bins": bins})
    plt.close(fig)


def main():
    fig_media_switch()
    fig_ri_drop()
    print("done")


if __name__ == "__main__":
    main()
