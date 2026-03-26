"""visualize_drift_log.py — drift log の可視化（cumulative drift / step drift / ECC）"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# figure_logger
sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

# --- load ---
DRIFT_LOG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log.json")
records = json.loads(DRIFT_LOG.read_text(encoding="utf-8"))

tp        = np.array([r["timepoint"]        for r in records])
cum_x     = np.array([r["cumulative_dx_um"] for r in records])
cum_y     = np.array([r["cumulative_dy_um"] for r in records])
step_x    = np.array([r["correction_stage_x_um"] for r in records])
step_y    = np.array([r["correction_stage_y_um"] for r in records])
ecc       = np.array([r["ecc_correlation"]  for r in records])
jump      = np.array([r["jump_detected"]    for r in records])
invalid   = np.array([not r["correction_valid"] for r in records])

interval_min = 300 / 60  # 5 min
time_h = tp * interval_min / 60

# --- figure ---
fig = plt.figure(figsize=(10, 9))
gs  = gridspec.GridSpec(3, 1, hspace=0.45)

# ---- panel 1: cumulative drift ----
ax1 = fig.add_subplot(gs[0])
ax1.plot(time_h, cum_x, color="#2196F3", lw=1.5, label="X (stage)")
ax1.plot(time_h, cum_y, color="#F44336", lw=1.5, label="Y (stage)")
ax1.axhline(0, color="gray", lw=0.5, ls="--")
ax1.set_ylabel("Cumulative drift (μm)")
ax1.set_xlabel("Time (h)")
ax1.set_title("Cumulative drift correction applied")
ax1.legend(frameon=False, fontsize=9)
ax1.set_xlim(time_h[0], time_h[-1])

# ---- panel 2: step shift per frame ----
ax2 = fig.add_subplot(gs[1])
ax2.plot(time_h, step_x, color="#2196F3", lw=1.0, alpha=0.8, label="X step")
ax2.plot(time_h, step_y, color="#F44336", lw=1.0, alpha=0.8, label="Y step")
ax2.axhline(0, color="gray", lw=0.5, ls="--")
# mark jumps / invalid
if jump.any():
    ax2.scatter(time_h[jump], step_x[jump], marker="x", color="orange", s=60, zorder=5, label="jump")
if invalid.any():
    ax2.scatter(time_h[invalid], step_x[invalid], marker="o", color="black", s=40, zorder=5, label="invalid")
ax2.set_ylabel("Step correction (μm)")
ax2.set_xlabel("Time (h)")
ax2.set_title("Per-frame drift step")
ax2.legend(frameon=False, fontsize=9)
ax2.set_xlim(time_h[0], time_h[-1])

# ---- panel 3: ECC correlation ----
ax3 = fig.add_subplot(gs[2])
ax3.plot(time_h, ecc, color="#4CAF50", lw=1.2, alpha=0.9)
ax3.axhline(0.95, color="gray", lw=0.8, ls="--", label="0.95")
ax3.set_ylim(0.80, 1.01)
ax3.set_ylabel("ECC correlation")
ax3.set_xlabel("Time (h)")
ax3.set_title("ECC registration quality")
ax3.legend(frameon=False, fontsize=9)
ax3.set_xlim(time_h[0], time_h[-1])

for ax in [ax1, ax2, ax3]:
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# --- summary stats ---
n_tp    = len(tp)
dur_h   = time_h[-1] - time_h[0]
final_dx = cum_x[-1]
final_dy = cum_y[-1]
ecc_mean = ecc.mean()
ecc_min  = ecc.min()
n_invalid = invalid.sum()

print(f"Timepoints : {n_tp} (TP {tp[0]}-{tp[-1]})")
print(f"Duration   : {dur_h:.1f} h")
print(f"Cumulative : X={final_dx:+.3f} μm, Y={final_dy:+.3f} μm")
print(f"ECC        : mean={ecc_mean:.4f}, min={ecc_min:.4f}")
print(f"Invalid    : {n_invalid}")

save_figure(
    fig,
    params={
        "n_timepoints": int(n_tp),
        "interval_min": interval_min,
        "final_cum_dx_um": float(final_dx),
        "final_cum_dy_um": float(final_dy),
        "ecc_mean": float(ecc_mean),
        "ecc_min": float(ecc_min),
        "n_invalid": int(n_invalid),
    },
    description="タイムラプスドリフト補正ログの可視化（cumulative drift / step / ECC correlation）",
    data={
        "time_h": time_h,
        "cum_x": cum_x,
        "cum_y": cum_y,
        "step_x": step_x,
        "step_y": step_y,
        "ecc": ecc,
        "tp": tp,
    },
)

plt.close(fig)
