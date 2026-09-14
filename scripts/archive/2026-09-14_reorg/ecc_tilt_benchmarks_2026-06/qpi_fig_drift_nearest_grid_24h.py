"""Publication figure pair: stage drift before/after closed-loop correction (0-24 h).

Data: 2026-04-10 analyze_drift_control session (1325-frame timelapse, 110 h),
      which records both reconstructions on the SAME acquisition:
  Before (open-loop)  : ol_sx_nm, ol_sy_nm   -- drift reconstructed as if no
                                                stage correction was applied
  After  (closed-loop): z_sx_nm,  z_sy_nm    -- actual residual measured with
                                                real-time stage correction on

Magnitude = sqrt(dx^2 + dy^2) in micrometers. Same time axis (0-24 h), same color
(Okabe-Ito blue), so the figures differ only in y-axis scale of the underlying
drift, making the contraction directly visible.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

DRIFT_INBOX = Path(
    "/Users/kitak/Library/CloudStorage/"
    "GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/共有ドライブ/"
    "wakamotolab_meeting/kitagishi/figure-hub/inbox"
)
SESSIONS = [
    ("2026-03-27", "20260327T100000Z"),  # 14 h total — full session within 0-24h
    ("2026-03-30", None),                # 70 h
    ("2026-04-07", None),                # 47 h
    ("2026-04-10", None),                # 110 h
]
T_MAX_H = 24.0
COLOR_LINE = "#0072B2"   # Okabe-Ito blue (1st)
COLOR_GREY = "#666666"

def find_session_npz(date):
    import glob
    hits = sorted(glob.glob(str(DRIFT_INBOX / date / "analyze_drift_control" / "*" / "*_data.npz")))
    return Path(hits[0]) if hits else None

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


def make_panel(t, y, title, ylabel, y_max, y_major, fig_tag, source_session, data_path):
    fig, (ax_ts, ax_hist) = plt.subplots(
        1, 2, figsize=(89 * MM, 42 * MM),
        gridspec_kw={"width_ratios": [3, 1.2], "wspace": 0.32},
    )
    ax_ts.plot(t, y, color=COLOR_LINE, lw=0.7)
    ax_ts.set_xlabel("Time (h)")
    ax_ts.set_ylabel(ylabel)
    ax_ts.set_xlim(0, T_MAX_H)
    ax_ts.set_ylim(0, y_max)
    ax_ts.xaxis.set_major_locator(MultipleLocator(4))
    ax_ts.xaxis.set_minor_locator(MultipleLocator(1))
    ax_ts.yaxis.set_major_locator(MultipleLocator(y_major))
    ax_ts.yaxis.set_minor_locator(MultipleLocator(y_major / 2))
    ax_ts.set_title(title, loc="left", pad=4)

    finite = np.isfinite(y)
    y_f = y[finite]
    bins = np.linspace(0, y_max, 23)
    ax_hist.hist(y_f, bins=bins, orientation="horizontal",
                 color=COLOR_LINE, edgecolor="white", linewidth=0.3)
    ax_hist.set_ylim(0, y_max)
    ax_hist.set_xlabel("Count")
    ax_hist.yaxis.set_major_locator(MultipleLocator(y_major))
    ax_hist.yaxis.set_minor_locator(MultipleLocator(y_major / 2))
    ax_hist.tick_params(labelleft=False)
    med = float(np.nanmedian(y_f))
    mx  = float(np.nanmax(y_f))
    ax_hist.set_title(
        f"median {med*1000:.0f} nm\nmax {mx*1000:.0f} nm",
        fontsize=7, color=COLOR_GREY, loc="left", pad=4,
    )

    for ax in (ax_ts, ax_hist):
        ax.tick_params(direction="in", top=True, right=True, which="both")
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)

    params = {
        "stage":          fig_tag,
        "t_max_h":        T_MAX_H,
        "median_um":      med,
        "max_um":         mx,
        "n_frames_used":  int(finite.sum()),
        "source_session": source_session,
        "data_source":    str(data_path),
    }
    for fmt in ("pdf", "png"):
        save_figure(
            fig, params=params,
            description=f"Drift {fig_tag} (0-24h, single Pos, open-loop vs closed-loop)",
            fmt=fmt,
        )
    plt.close(fig)


for date, _ in SESSIONS:
    npz_path = find_session_npz(date)
    if npz_path is None:
        print(f"[skip] no analyze_drift_control session at {date}")
        continue
    d = np.load(npz_path)
    if "ol_sx_nm" not in d.files or "z_sx_nm" not in d.files:
        print(f"[skip] {date}: missing ol/z keys")
        continue
    t_full = d["time_h"]
    if t_full.max() < T_MAX_H:
        print(f"[skip {date}] only {t_full.max():.1f} h of data (< {T_MAX_H} h)")
        continue
    m = t_full <= T_MAX_H
    t = t_full[m]
    mag_before = np.sqrt(d["ol_sx_nm"][m]**2 + d["ol_sy_nm"][m]**2) / 1000.0
    mag_after  = np.sqrt(d["z_sx_nm"][m] **2 + d["z_sy_nm"][m] **2) / 1000.0
    ol_max = float(mag_before.max())
    y_before_max = max(0.5, round(ol_max * 1.15, 1))
    print(f"[{date}] ol_max@0-24h={ol_max:.2f} μm, cl_max@0-24h={mag_after.max():.3f} μm  -> y_before_max={y_before_max}")
    session_label = f"{date}_analyze_drift_control"
    make_panel(
        t, mag_before,
        title=f"a   Before ({date}): open-loop drift",
        ylabel="Drift magnitude (μm)",
        y_max=y_before_max, y_major=0.5 if y_before_max <= 3.5 else 1.0,
        fig_tag=f"before_open_loop_{date}",
        source_session=session_label, data_path=npz_path,
    )
    make_panel(
        t, mag_after,
        title=f"b   After ({date}): closed-loop residual",
        ylabel="Residual magnitude (μm)",
        y_max=0.7 if mag_after.max() > 0.35 else 0.35,
        y_major=0.1,
        fig_tag=f"after_closed_loop_{date}",
        source_session=session_label, data_path=npz_path,
    )


# 2026-04-16 plot_drift_summary: 200h acquisition with explicit cumulative drift + Kalman residual
P416 = (DRIFT_INBOX / "2026-04-16" / "plot_drift_summary" /
        "20260416T110350Z_a2dd4d" /
        "plot_drift_summary__Drift_correction_overview_cumulative_dri__"
        "20260416T110350Z_a2dd4d__f001_data.npz")
if P416.exists():
    d416 = np.load(P416)
    t416 = d416["time_h"]
    m416 = t416 <= T_MAX_H
    t_416  = t416[m416]
    before = np.sqrt(d416["cumulative_dx_um"][m416]**2 + d416["cumulative_dy_um"][m416]**2)
    P416_K = P416.parent / "plot_drift_summary__Kalman_filter_raw_ECC_vs_filtered_shift___20260416T110350Z_a2dd4d__f004_data.npz"
    d416k = np.load(P416_K)
    after = np.sqrt(d416k["raw_tx_nm"][m416]**2 + d416k["raw_ty_nm"][m416]**2) / 1000.0
    print(f"[2026-04-16 plot_drift_summary] cum_max@0-24h={before.max():.2f} μm, raw_resid_max@0-24h={after.max():.3f} μm")
    make_panel(
        t_416, before,
        title="a   Before (2026-04-16): cumulative stage drift",
        ylabel="Drift magnitude (μm)",
        y_max=2.8, y_major=0.5,
        fig_tag="before_2026-04-16_cumulative",
        source_session="2026-04-16_plot_drift_summary",
        data_path=P416,
    )
    make_panel(
        t_416, after,
        title="b   After (2026-04-16): raw ECC residual",
        ylabel="Residual magnitude (μm)",
        y_max=0.45, y_major=0.1,
        fig_tag="after_2026-04-16_raw_residual",
        source_session="2026-04-16_plot_drift_summary",
        data_path=P416_K,
    )

import glob as _glob
for vpath in sorted(_glob.glob(str(DRIFT_INBOX / "*" / "visualize_drift_log" / "*" / "*_data.npz"))):
    npz_path = Path(vpath)
    date = vpath.split("/inbox/")[1].split("/")[0]
    d = np.load(npz_path)
    if "cum_x" not in d.files or "cum_y" not in d.files:
        continue
    t_full = d["time_h"]
    if t_full.max() < T_MAX_H:
        print(f"[skip {date} visualize] only {t_full.max():.1f} h of data (< {T_MAX_H} h)")
        continue
    m = t_full <= T_MAX_H
    t = t_full[m]
    mag_cum = np.sqrt(d["cum_x"][m]**2 + d["cum_y"][m]**2)
    mx = float(mag_cum.max())
    y_max = max(0.5, round(mx * 1.15, 1))
    print(f"[{date} visualize] cum_max@0-24h={mx:.2f} μm  -> y_max={y_max}, n_frames={m.sum()}")
    make_panel(
        t, mag_cum,
        title=f"Drift ({date}, stage log): cumulative correction magnitude",
        ylabel="Drift magnitude (μm)",
        y_max=y_max, y_major=0.5 if y_max <= 3.5 else 1.0,
        fig_tag=f"stagelog_drift_{date}_{npz_path.parent.name}",
        source_session=f"{date}_visualize_drift_log", data_path=npz_path,
    )
