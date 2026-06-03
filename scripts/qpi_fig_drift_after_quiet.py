"""Quietest 24-h after-correction residual found across all drift datasets.

Source : 2026-05-20 batch_pipeline_all_pos, Pos16, fine_ecc euclid2 (f105)
Window : t = 57 - 81 h  (24 h, x-axis re-based to 0-24)
Metric : euclid2 = distance from nearest grid in the 81-point lookup (µm)
Result : mean = 32 nm, median = 26 nm, max = 135 nm
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

NPZ = Path(
    "/Users/kitak/Library/CloudStorage/"
    "GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/共有ドライブ/"
    "wakamotolab_meeting/kitagishi/figure-hub/inbox/2026-05-20/"
    "batch_pipeline_all_pos/20260520T023102Z_13c09c/"
    "batch_pipeline_all_pos__shift_timeseries_fine_ecc_unknown__"
    "20260520T023102Z_13c09c__f105_data.npz"
)
WINDOW_START_H = 57.0
WINDOW_LEN_H   = 24.0
COLOR_LINE = "#009E73"   # Okabe-Ito green
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

d = np.load(NPZ)
t = d["x_values"]
mag = d["euclid2"]
mask = (t >= WINDOW_START_H) & (t < WINDOW_START_H + WINDOW_LEN_H)
t_rel = t[mask] - WINDOW_START_H
m_sel = mag[mask]
finite = np.isfinite(m_sel)
mean_v = float(np.nanmean(m_sel[finite]))
med_v  = float(np.nanmedian(m_sel[finite]))
max_v  = float(np.nanmax(m_sel[finite]))
print(f"window {WINDOW_START_H:g}-{WINDOW_START_H+WINDOW_LEN_H:g}h: mean={mean_v*1000:.0f} nm, median={med_v*1000:.0f} nm, max={max_v*1000:.0f} nm, n={finite.sum()}")


def make_panel(y_max, y_major, fig_tag):
    fig, ax = plt.subplots(figsize=(89 * MM, 42 * MM))
    ax.plot(t_rel, m_sel, color=COLOR_LINE, lw=0.7)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Residual magnitude (μm)")
    ax.set_xlim(0, WINDOW_LEN_H)
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_major / 2))
    ax.tick_params(direction="in", top=True, right=True, which="both")
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)

    params = {
        "y_scale_um":   y_max,
        "window_start_h": WINDOW_START_H,
        "window_len_h":   WINDOW_LEN_H,
        "mean_um":   mean_v,
        "median_um": med_v,
        "max_um":    max_v,
        "n_frames":  int(finite.sum()),
        "pos":       "Pos16 (260405_acute_z18_200h)",
        "source_session": "2026-05-20_batch_pipeline_all_pos",
        "data_source": str(NPZ),
    }
    for fmt in ("pdf", "png"):
        save_figure(fig, params=params,
                    description=f"Quietest 24-h after-correction residual (Pos16 fine_ecc, y={y_max} μm, {fig_tag})",
                    fmt=fmt)
    plt.close(fig)


make_panel(y_max=1.0, y_major=0.2, fig_tag="y_scale_0-1um")
make_panel(y_max=3.0, y_major=0.5, fig_tag="y_scale_0-3um")
