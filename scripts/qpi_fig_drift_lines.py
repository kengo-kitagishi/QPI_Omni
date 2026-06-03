"""Publication figure pair (line style, no histogram):
   X and Y drift as two lines, full 200 h, 2026-04-16 plot_drift_summary data.

Before: cumulative_dx_um, cumulative_dy_um (uncorrected scene drift)
After : raw_tx_nm/1000, raw_ty_nm/1000     (closed-loop ECC residual)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

BASE = Path(
    "/Users/kitak/Library/CloudStorage/"
    "GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/共有ドライブ/"
    "wakamotolab_meeting/kitagishi/figure-hub/inbox/2026-04-16/"
    "plot_drift_summary/20260416T110350Z_a2dd4d"
)
F_BEFORE = BASE / "plot_drift_summary__Drift_correction_overview_cumulative_dri__20260416T110350Z_a2dd4d__f001_data.npz"
F_AFTER  = BASE / "plot_drift_summary__Kalman_filter_raw_ECC_vs_filtered_shift___20260416T110350Z_a2dd4d__f004_data.npz"

# Okabe-Ito: blue (1st), vermillion (4th) — colorblind-safe X/Y pair
COLOR_X = "#0072B2"
COLOR_Y = "#D55E00"
COLOR_GREY = "#666666"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
MM = 1 / 25.4


def line_panel(t, x, y, title, ylabel, y_lim, y_major, fig_tag, source_path):
    fig, ax = plt.subplots(figsize=(89 * MM, 42 * MM))
    ax.plot(t, x, color=COLOR_X, label="X")
    ax.plot(t, y, color=COLOR_Y, label="Y")
    ax.axhline(0, color=COLOR_GREY, lw=0.4, ls=":")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, t.max())
    ax.set_ylim(*y_lim)
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_major / 2))
    xspan = t.max()
    ax.xaxis.set_major_locator(MultipleLocator(50 if xspan > 80 else 20 if xspan > 40 else 4))
    ax.xaxis.set_minor_locator(MultipleLocator(10 if xspan > 80 else 5 if xspan > 40 else 1))
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.4,
              columnspacing=1.0, borderpad=0.2)
    ax.set_title(title, loc="left", pad=4)
    ax.tick_params(direction="in", top=True, right=True, which="both")
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)

    params = {
        "stage":        fig_tag,
        "t_max_h":      float(t.max()),
        "x_range_um":   [float(np.nanmin(x)), float(np.nanmax(x))],
        "y_range_um":   [float(np.nanmin(y)), float(np.nanmax(y))],
        "source_session": "2026-04-16_plot_drift_summary",
        "data_source":  str(source_path),
    }
    for fmt in ("pdf", "png"):
        save_figure(
            fig, params=params,
            description=f"Drift {fig_tag} XY lines (line style, no histogram)",
            fmt=fmt,
        )
    plt.close(fig)


d_b = np.load(F_BEFORE)
d_a = np.load(F_AFTER)
t = d_b["time_h"]
cum_x = d_b["cumulative_dx_um"]
cum_y = d_b["cumulative_dy_um"]
raw_x = d_a["raw_tx_nm"] / 1000.0
raw_y = d_a["raw_ty_nm"] / 1000.0

line_panel(
    t, cum_x, cum_y,
    title="a   Before: cumulative stage drift (no correction)",
    ylabel="Drift (μm)",
    y_lim=(-2.5, 7.5), y_major=2.0,
    fig_tag="before_cumulative_XY_200h",
    source_path=F_BEFORE,
)
line_panel(
    t, raw_x, raw_y,
    title="b   After: raw ECC residual (closed-loop)",
    ylabel="Residual (μm)",
    y_lim=(-0.35, 0.35), y_major=0.1,
    fig_tag="after_raw_residual_XY_200h",
    source_path=F_AFTER,
)
