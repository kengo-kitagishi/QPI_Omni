"""Bin-width sweep for the 3 figures, laid out as easy-to-compare grids.

Fig A (media-switch, exact frame / no window): rows = 3 switch frames
       (2018/2306/2884), cols = bin widths.
Fig B (RI drop): rows = 2 defs (single 2884, window 2864-2874), cols = bin widths.

Style matched to the current figures: y = count, dark-gray revived vs red dead,
white edges, mean vertical lines. Each panel autoscaled; column header = bin width.

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

WIDTHS = [0.0010, 0.0015, 0.0020, 0.0030]
GRAY, RED, DRED = "#1a1a1a", "#d62728", "#8c0000"


def fixed_bins(lo, hi, w):
    lo = np.floor(lo / w) * w
    hi = np.ceil(hi / w) * w
    return np.arange(lo, hi + w / 2, w)


def panel(ax, r, n, bins, title):
    p = float(ks_2samp(r, n).pvalue) if len(r) and len(n) else float("nan")
    ax.hist(r, bins=bins, color=GRAY, alpha=0.55, edgecolor="white", linewidth=0.3)
    ax.hist(n, bins=bins, color=RED, alpha=0.55, edgecolor="white", linewidth=0.3)
    ax.axvline(np.mean(r), color=GRAY, lw=0.8, alpha=0.9)
    ax.axvline(np.mean(n), color=DRED, lw=0.8, alpha=0.9)
    ax.set_title(f"{title}  KS={p:.3g}", fontsize=6.5)
    ax.tick_params(labelsize=6)


def fig_media_switch():
    rev = list_revived_mothers()
    nr = list_never_revived_mothers()
    rows = [(f, lbl, vals(rev, f), vals(nr, f)) for f, lbl in SWITCHES]
    lo = min(min(r.min(), n.min()) for _, _, r, n in rows)
    hi = max(max(r.max(), n.max()) for _, _, r, n in rows)

    nr_, nc = len(rows), len(WIDTHS)
    fig, axes = plt.subplots(nr_, nc, figsize=(3.0 * nc, 2.1 * nr_),
                             squeeze=False)
    for i, (f, lbl, r, n) in enumerate(rows):
        for j, w in enumerate(WIDTHS):
            bins = fixed_bins(lo, hi, w)
            ttl = (f"w={w:.4f} ({len(bins)-1}b)" if i == 0
                   else f"w={w:.4f}")
            panel(axes[i][j], r, n, bins, ttl)
            if j == 0:
                axes[i][j].set_ylabel(f"frame {f}\ncount", fontsize=7)
            if i == nr_ - 1:
                axes[i][j].set_xlabel("mean RI", fontsize=7)
    fig.suptitle("Mean RI at media switches (exact frame) - bin-width sweep  "
                 "[gray=revived, red=never_revived]", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    data = {}
    for f, _, r, n in rows:
        data[f"revived_f{f}"] = r
        data[f"never_revived_f{f}"] = n
    data["bin_widths"] = np.array(WIDTHS)
    save_figure(fig, params={"widths": WIDTHS, "y_axis": "count",
                             "value": "exact frame, no window",
                             "volume_method": "efd"},
                description="Bin-width sweep of mean-RI-at-media-switch dead/"
                            "alive histograms (exact frame), 3 frames x 4 widths",
                data=data)
    plt.close(fig)


def fig_ri_drop():
    rev = list_revived_mothers()
    nr = list_never_revived_mothers()
    r_s, r_a = drops(rev)
    n_s, n_a = drops(nr)
    rows = [("single 2884", "dRI=RI(2884)-RI(2018)", r_s, n_s),
            ("window 2864-2874", "dRI=meanRI(2864-2874)-RI(2018)", r_a, n_a)]
    lo = min(min(r.min(), n.min()) for _, _, r, n in rows)
    hi = max(max(r.max(), n.max()) for _, _, r, n in rows)

    nr_, nc = len(rows), len(WIDTHS)
    fig, axes = plt.subplots(nr_, nc, figsize=(3.0 * nc, 2.4 * nr_),
                             squeeze=False)
    for i, (key, xl, r, n) in enumerate(rows):
        for j, w in enumerate(WIDTHS):
            bins = fixed_bins(lo, hi, w)
            ttl = (f"w={w:.4f} ({len(bins)-1}b)" if i == 0
                   else f"w={w:.4f}")
            panel(axes[i][j], r, n, bins, ttl)
            axes[i][j].axvline(0, color="0.6", lw=0.5)
            if j == 0:
                axes[i][j].set_ylabel(f"{key}\ncount", fontsize=7)
            if i == nr_ - 1:
                axes[i][j].set_xlabel(xl, fontsize=7)
    fig.suptitle("RI drop across starvation - bin-width sweep  "
                 "[gray=revived, red=never_revived]", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    data = {"revived_single": r_s, "never_revived_single": n_s,
            "revived_window": r_a, "never_revived_window": n_a,
            "bin_widths": np.array(WIDTHS)}
    save_figure(fig, params={"widths": WIDTHS, "y_axis": "count",
                             "volume_method": "efd"},
                description="Bin-width sweep of RI-drop dead/alive histograms, "
                            "2 defs x 4 widths",
                data=data)
    plt.close(fig)


def main():
    print(f"widths={WIDTHS}")
    fig_media_switch()
    fig_ri_drop()
    print("done")


if __name__ == "__main__":
    main()
