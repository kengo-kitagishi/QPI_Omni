"""visualize_drift_log.py — drift log visualization (cumulative / step / ECC).

Supports both old flat format and new per-pos wrapper format.
When per-pos data is present, thin lines are drawn per position and
a bold line shows the cross-position mean.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np

# figure_logger
sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure


# --- helpers ---

def _load_drift_log(path):
    """Load drift_log.json; flatten per-pos wrapper if present."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw and isinstance(raw[0], dict) and raw[0].get("per_pos"):
        return [r for w in raw for r in w.get("positions", [w])]
    return raw


def _group_by_pos(records):
    """Group flat records by pos_label. Returns {label: [records]}."""
    groups = defaultdict(list)
    for r in records:
        groups[r.get("pos_label", "all")].append(r)
    return dict(groups)


def _avg_per_tp(records):
    """Average per-pos records into one record per timepoint."""
    by_tp = defaultdict(list)
    for r in records:
        by_tp[r["timepoint"]].append(r)
    out = []
    for t in sorted(by_tp):
        recs = by_tp[t]
        avg = {"timepoint": t}
        for k in ("tx_avg_px", "ty_avg_px", "ecc_correlation",
                   "correction_stage_x_um", "correction_stage_y_um",
                   "cumulative_dx_um", "cumulative_dy_um"):
            vals = [r[k] for r in recs if k in r]
            avg[k] = float(np.mean(vals)) if vals else 0.0
        avg["jump_detected"] = any(r.get("jump_detected") for r in recs)
        avg["correction_valid"] = all(r.get("correction_valid", True) for r in recs)
        out.append(avg)
    return out


# --- load ---
DRIFT_LOG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log.json")
flat_records = _load_drift_log(DRIFT_LOG)
pos_groups = _group_by_pos(flat_records)
pos_labels = sorted(pos_groups.keys())
is_per_pos = len(pos_labels) > 1
avg_records = _avg_per_tp(flat_records) if is_per_pos else flat_records

# Per-pos color map
pos_cmap = cm.get_cmap("tab10", max(len(pos_labels), 1))
pos_colors = {lbl: pos_cmap(i) for i, lbl in enumerate(pos_labels)}

# Average arrays (for bold lines and stats)
tp        = np.array([r["timepoint"]              for r in avg_records])
cum_x     = np.array([r["cumulative_dx_um"]       for r in avg_records])
cum_y     = np.array([r["cumulative_dy_um"]       for r in avg_records])
step_x    = np.array([r["correction_stage_x_um"]  for r in avg_records])
step_y    = np.array([r["correction_stage_y_um"]  for r in avg_records])
ecc       = np.array([r["ecc_correlation"]        for r in avg_records])
jump      = np.array([r["jump_detected"]          for r in avg_records])
invalid   = np.array([not r["correction_valid"]   for r in avg_records])

interval_min = 300 / 60  # 5 min
time_h = tp * interval_min / 60

print(f"Loaded {len(flat_records)} records  ({len(pos_labels)} positions)")

# --- figure ---
fig = plt.figure(figsize=(10, 9))
gs  = gridspec.GridSpec(3, 1, hspace=0.45)

# ---- panel 1: cumulative drift ----
ax1 = fig.add_subplot(gs[0])
if is_per_pos:
    for lbl in pos_labels:
        recs = pos_groups[lbl]
        t_h = np.array([r["timepoint"] for r in recs]) * interval_min / 60
        ax1.plot(t_h, [r["cumulative_dx_um"] for r in recs],
                 color=pos_colors[lbl], lw=0.5, alpha=0.35)
        ax1.plot(t_h, [r["cumulative_dy_um"] for r in recs],
                 color=pos_colors[lbl], lw=0.5, alpha=0.35, ls="--")
ax1.plot(time_h, cum_x, color="#2196F3", lw=1.5, label="X mean" if is_per_pos else "X (stage)")
ax1.plot(time_h, cum_y, color="#F44336", lw=1.5, label="Y mean" if is_per_pos else "Y (stage)")
ax1.axhline(0, color="gray", lw=0.5, ls="--")
ax1.set_ylabel("Cumulative drift (\u03bcm)")
ax1.set_xlabel("Time (h)")
title1 = "Cumulative drift correction applied"
if is_per_pos:
    title1 += f"  ({len(pos_labels)} positions, thin = per-pos)"
ax1.set_title(title1)
ax1.legend(frameon=False, fontsize=9)
ax1.set_xlim(time_h[0], time_h[-1])

# ---- panel 2: step shift per frame ----
ax2 = fig.add_subplot(gs[1])
if is_per_pos:
    for lbl in pos_labels:
        recs = pos_groups[lbl]
        t_h = np.array([r["timepoint"] for r in recs]) * interval_min / 60
        ax2.plot(t_h, [r["correction_stage_x_um"] for r in recs],
                 color=pos_colors[lbl], lw=0.4, alpha=0.3)
ax2.plot(time_h, step_x, color="#2196F3", lw=1.0, alpha=0.8, label="X step")
ax2.plot(time_h, step_y, color="#F44336", lw=1.0, alpha=0.8, label="Y step")
ax2.axhline(0, color="gray", lw=0.5, ls="--")
if jump.any():
    ax2.scatter(time_h[jump], step_x[jump], marker="x", color="orange", s=60, zorder=5, label="jump")
if invalid.any():
    ax2.scatter(time_h[invalid], step_x[invalid], marker="o", color="black", s=40, zorder=5, label="invalid")
ax2.set_ylabel("Step correction (\u03bcm)")
ax2.set_xlabel("Time (h)")
ax2.set_title("Per-frame drift step")
ax2.legend(frameon=False, fontsize=9)
ax2.set_xlim(time_h[0], time_h[-1])

# ---- panel 3: ECC correlation ----
ax3 = fig.add_subplot(gs[2])
if is_per_pos:
    for lbl in pos_labels:
        recs = pos_groups[lbl]
        t_h = np.array([r["timepoint"] for r in recs]) * interval_min / 60
        ax3.plot(t_h, [r["ecc_correlation"] for r in recs],
                 color=pos_colors[lbl], lw=0.4, alpha=0.3)
ax3.plot(time_h, ecc, color="#4CAF50", lw=1.2, alpha=0.9, label="mean" if is_per_pos else None)
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
print(f"Cumulative : X={final_dx:+.3f} \u03bcm, Y={final_dy:+.3f} \u03bcm")
print(f"ECC        : mean={ecc_mean:.4f}, min={ecc_min:.4f}")
print(f"Invalid    : {n_invalid}")

save_figure(
    fig,
    params={
        "n_timepoints": int(n_tp),
        "n_positions": len(pos_labels),
        "interval_min": interval_min,
        "final_cum_dx_um": float(final_dx),
        "final_cum_dy_um": float(final_dy),
        "ecc_mean": float(ecc_mean),
        "ecc_min": float(ecc_min),
        "n_invalid": int(n_invalid),
    },
    description="Timelapse drift correction log (cumulative drift / step / ECC correlation, per-pos overlay)",
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
