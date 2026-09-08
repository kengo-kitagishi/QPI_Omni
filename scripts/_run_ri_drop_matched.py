"""Re-render the 2 RI-drop histograms with style matched to the media-switch
(no-window) figure: y-axis = count (not density), bin width = 0.04/28 (same as
the 1.345..1.385 / 28-bin media-switch figure), dark-gray revived vs red dead,
white bar edges, mean vertical lines. Both figures share common bins and y-limit
so they are directly comparable.

Run with QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd (EFD lineage).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent))
from ri_drop_starvation import (  # noqa: E402
    drops, list_revived_mothers, list_never_revived_mothers,
)
from figure_logger import save_figure  # noqa: E402

W = 0.04 / 28  # media-switch figure bin width (1.345..1.385 over 28 bins)


def common_bins(*arrays):
    allv = np.concatenate(arrays)
    lo = np.floor(allv.min() / W) * W
    hi = np.ceil(allv.max() / W) * W
    return np.arange(lo, hi + W / 2, W)


def plot_one(r, n, bins, ylim, xlabel, title, description):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    p = float(ks_2samp(r, n).pvalue) if len(r) and len(n) else float("nan")
    ax.hist(r, bins=bins, color="#1a1a1a", alpha=0.55, edgecolor="white",
            linewidth=0.4,
            label=f"revived (n={len(r)}, mean={np.mean(r):+.4f}, "
                  f"med={np.median(r):+.4f})")
    ax.hist(n, bins=bins, color="#d62728", alpha=0.55, edgecolor="white",
            linewidth=0.4,
            label=f"never_revived (n={len(n)}, mean={np.mean(n):+.4f}, "
                  f"med={np.median(n):+.4f})")
    ax.axvline(np.mean(r), color="#1a1a1a", lw=0.8, alpha=0.9)
    ax.axvline(np.mean(n), color="#8c0000", lw=0.8, alpha=0.9)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.set_ylim(0, ylim)
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    ax.set_title(f"{title}   KS p = {p:.3g}", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_figure(fig, params={
        "bin_width": W, "y_axis": "count", "volume_method": "efd",
        "style_matched_to": "media-switch no-window figure",
        "n_revived": len(r), "n_never_revived": len(n),
        "mean_revived": float(np.mean(r)),
        "mean_never_revived": float(np.mean(n)),
        "median_revived": float(np.median(r)),
        "median_never_revived": float(np.median(n)),
        "ks_p": p,
    }, description=description,
        data={"revived": r, "never_revived": n, "bins": bins})
    plt.close(fig)
    return p


def main():
    rev = list_revived_mothers()
    nr = list_never_revived_mothers()
    print(f"revived={len(rev)}, never_revived={len(nr)}")
    r_single, r_avg = drops(rev)
    n_single, n_avg = drops(nr)

    bins = common_bins(r_single, n_single, r_avg, n_avg)

    def mc(vals):
        return int(np.histogram(vals, bins)[0].max())
    ylim = max(mc(r_single), mc(n_single), mc(r_avg), mc(n_avg)) + 1
    print(f"bins: {len(bins)-1} bins of width {W:.5f}  ylim={ylim}")

    p1 = plot_one(
        r_single, n_single, bins, ylim,
        "dRI = RI(2884) - RI(2018)",
        "RI drop across starvation (single frame 2884)",
        "RI drop RI(2884)-RI(2018), revived vs never_revived (EFD), "
        "style matched to media-switch figure (count, bin 0.00143)")
    print(f"[single 2884] revived med={np.median(r_single):+.5f} "
          f"(n={len(r_single)})  dead med={np.median(n_single):+.5f} "
          f"(n={len(n_single)})  KS p={p1:.4g}")

    p2 = plot_one(
        r_avg, n_avg, bins, ylim,
        "dRI = mean RI(2864-2874) - RI(2018)",
        "RI drop across starvation (window 2864-2874)",
        "RI drop meanRI(2864-2874)-RI(2018), revived vs never_revived (EFD), "
        "style matched to media-switch figure (count, bin 0.00143)")
    print(f"[window 2864-2874] revived med={np.median(r_avg):+.5f} "
          f"(n={len(r_avg)})  dead med={np.median(n_avg):+.5f} "
          f"(n={len(n_avg)})  KS p={p2:.4g}")


if __name__ == "__main__":
    main()
