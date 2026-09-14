"""Drift magnitude in an arbitrary 24-h window (default 100-124 h),
   plotted on a relative 0-24 h x-axis. Simple single panel, no histogram.

Data: 2026-04-16 plot_drift_summary (Pos1, 203 h acquisition).
Before: sqrt(cumulative_dx_um^2 + cumulative_dy_um^2)
After : sqrt(raw_tx_nm^2 + raw_ty_nm^2) / 1000
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

WINDOW_START_H = 100.0   # before-side default window
WINDOW_LEN_H   = 24.0
COLOR_LINE = "#0072B2"   # Okabe-Ito blue (1st) — for before
COLOR_GREY = "#666666"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
MM = 1 / 25.4


def make_panel(t_rel, y, title, ylabel, y_max, y_major, fig_tag, source_path):
    fig, ax = plt.subplots(figsize=(89 * MM, 42 * MM))
    ax.plot(t_rel, y, color=COLOR_LINE, lw=0.7)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, WINDOW_LEN_H)
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_major / 2))
    ax.tick_params(direction="in", top=True, right=True, which="both")
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)

    finite = np.isfinite(y)
    params = {
        "stage":           fig_tag,
        "window_start_h":  WINDOW_START_H,
        "window_len_h":    WINDOW_LEN_H,
        "median_um":       float(np.nanmedian(y[finite])),
        "max_um":          float(np.nanmax(y[finite])),
        "n_frames_used":   int(finite.sum()),
        "source_session":  "2026-04-16_plot_drift_summary",
        "data_source":     str(source_path),
    }
    for fmt in ("pdf", "png"):
        save_figure(fig, params=params,
                    description=f"Drift magnitude {fig_tag} window {WINDOW_START_H:g}-{WINDOW_START_H+WINDOW_LEN_H:g}h (rel x)",
                    fmt=fmt)
    plt.close(fig)


def slice_window(t, *arrays):
    mask = (t >= WINDOW_START_H) & (t < WINDOW_START_H + WINDOW_LEN_H)
    t_sel = t[mask] - WINDOW_START_H
    return (t_sel,) + tuple(a[mask] for a in arrays)


d_b = np.load(F_BEFORE)
d_a = np.load(F_AFTER)
t_b, cx, cy = slice_window(d_b["time_h"], d_b["cumulative_dx_um"], d_b["cumulative_dy_um"])
t_a, rx, ry = slice_window(d_a["time_h"], d_a["raw_tx_nm"], d_a["raw_ty_nm"])
mag_before = np.sqrt(cx**2 + cy**2)
mag_after  = np.sqrt(rx**2 + ry**2) / 1000.0
print(f"Before: n={len(t_b)}, max={mag_before.max():.3f} μm, median={np.median(mag_before):.3f} μm")
print(f"After : n={len(t_a)}, max={mag_after.max():.3f} μm, median={np.median(mag_after):.3f} μm")

# y-axis: round up to next nice number
def y_top(arr, candidates=(0.2, 0.35, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0)):
    m = float(np.nanmax(arr)) * 1.10
    for c in candidates:
        if c >= m:
            return c
    return m

y_b_max = y_top(mag_before)
y_a_max = y_top(mag_after)
y_b_major = 1.0 if y_b_max > 3 else 0.5
y_a_major = 0.1 if y_a_max <= 0.5 else 0.2

make_panel(t_b, mag_before,
           title=f"a   Before (t = {WINDOW_START_H:g}-{WINDOW_START_H+WINDOW_LEN_H:g} h)",
           ylabel="Drift magnitude (μm)",
           y_max=y_b_max, y_major=y_b_major,
           fig_tag="before_cumulative_window",
           source_path=F_BEFORE)
make_panel(t_a, mag_after,
           title=f"b   After (t = {WINDOW_START_H:g}-{WINDOW_START_H+WINDOW_LEN_H:g} h)",
           ylabel="Residual magnitude (μm)",
           y_max=y_a_max, y_major=y_a_major,
           fig_tag="after_residual_window",
           source_path=F_AFTER)
