"""analyze_channel_drift.py -- Compare multiple aggregation methods using per-channel ECC data.

Targets TPs in drift_log.json that contain channel_details,
recomputes tx/ty time series with 4 aggregation methods, and compares noise levels.

Prerequisite: channel_details accumulated for at least 50 TPs.

Evaluation metric: residual std after removing low-frequency trend with Savitzky-Golay filter
  -> minimum = best reproduction of true drift
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

# ---- Settings ----
DRIFT_LOG      = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log.json")
INTERVAL_MIN   = 5.0
SMOOTH_WINDOW  = 51    # number of frames (~4 hour window). Must be odd
SMOOTH_ORDER   = 2
CORR_THRESHOLD = 0.98  # method B: exclude corr2 < this value
PIXEL_SCALE_UM = 0.3462  # μm/px

# ---- Data loading ----
records = json.loads(DRIFT_LOG.read_text(encoding="utf-8"))

# Use only records that have channel_details
detail_records = [r for r in records if r.get("channel_details")]
if len(detail_records) < SMOOTH_WINDOW:
    print(f"ERROR: channel_details has only {len(detail_records)} TPs (minimum {SMOOTH_WINDOW} TPs required)")
    sys.exit(1)

print(f"channel_details present: {len(detail_records)} TPs / total {len(records)} TPs")

tp_arr = np.array([r["timepoint"] for r in detail_records])
time_h = tp_arr * INTERVAL_MIN / 60


def _aggregate(method: str, records_: list) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate tx/ty from channel_details of each record using the specified method."""
    tx_out, ty_out = [], []

    for r in records_:
        details = r["channel_details"]
        tx_all   = np.array([d["tx2"] for d in details if "tx2" in d], dtype=np.float64)
        ty_all   = np.array([d["ty2"] for d in details if "ty2" in d], dtype=np.float64)
        corr_all = np.array([d.get("corr2", 0.0) for d in details if "tx2" in d], dtype=np.float64)

        if len(tx_all) == 0:
            tx_out.append(np.nan)
            ty_out.append(np.nan)
            continue

        if method == "A":
            # Current: MAD outlier removal -> simple mean
            mask = ~_mad_outlier(tx_all) & ~_mad_outlier(ty_all)
            if mask.sum() == 0:
                mask = np.ones(len(tx_all), dtype=bool)
            tx_out.append(tx_all[mask].mean())
            ty_out.append(ty_all[mask].mean())

        elif method == "B":
            # Correlation threshold exclusion -> simple mean
            mask = corr_all >= CORR_THRESHOLD
            if mask.sum() == 0:
                mask = np.ones(len(tx_all), dtype=bool)
            tx_out.append(tx_all[mask].mean())
            ty_out.append(ty_all[mask].mean())

        elif method == "C":
            # MAD retention + corr^2 weighted mean
            mask = ~_mad_outlier(tx_all) & ~_mad_outlier(ty_all)
            if mask.sum() == 0:
                mask = np.ones(len(tx_all), dtype=bool)
            w = corr_all[mask] ** 2
            if w.sum() == 0:
                w = np.ones(mask.sum())
            tx_out.append(np.average(tx_all[mask], weights=w))
            ty_out.append(np.average(ty_all[mask], weights=w))

        elif method == "D":
            # No outlier removal -> simple median
            tx_out.append(np.median(tx_all))
            ty_out.append(np.median(ty_all))

    return np.array(tx_out), np.array(ty_out)


def _mad_outlier(arr: np.ndarray, thresh: float = 5.0) -> np.ndarray:
    """MAD-based outlier flag (True = outlier)"""
    if len(arr) < 3:
        return np.zeros(len(arr), dtype=bool)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad == 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(arr - med) > thresh * mad


def noise_std(series: np.ndarray) -> float:
    """Std of residual after removing low-frequency component with Savitzky-Golay"""
    valid = ~np.isnan(series)
    if valid.sum() < SMOOTH_WINDOW:
        return np.nan
    wl = min(SMOOTH_WINDOW, valid.sum() - 1)
    if wl % 2 == 0:
        wl -= 1
    trend = savgol_filter(series[valid], window_length=wl, polyorder=SMOOTH_ORDER)
    return float(np.std(series[valid] - trend))


# ---- Aggregate with each method ----
methods = {
    "A: MAD->mean (current)": "A",
    f"B: corr>={CORR_THRESHOLD}->mean": "B",
    "C: MAD+corr2-weighted": "C",
    "D: median": "D",
}

results = {}
for label, key in methods.items():
    tx_px, ty_px = _aggregate(key, detail_records)
    # cumulative drift in μm
    cum_x = np.nancumsum(tx_px) * PIXEL_SCALE_UM
    cum_y = np.nancumsum(ty_px) * PIXEL_SCALE_UM
    results[label] = {
        "tx_px": tx_px, "ty_px": ty_px,
        "cum_x": cum_x, "cum_y": cum_y,
        "noise_x": noise_std(cum_x), "noise_y": noise_std(cum_y),
    }
    print(f"  {label:35s}  noise_x={results[label]['noise_x']:.4f} um  noise_y={results[label]['noise_y']:.4f} um")

# ---- Figure ----
n_methods = len(methods)
colors = ["#2196F3", "#F44336", "#FF9800", "#4CAF50"]

fig = plt.figure(figsize=(12, 4 * n_methods))
gs  = gridspec.GridSpec(n_methods, 2, hspace=0.5, wspace=0.35)

for row, (label, key) in enumerate(methods.items()):
    cum_x = results[label]["cum_x"]
    cum_y = results[label]["cum_y"]
    col = colors[row]

    for ax, series, axis_name in [
        (fig.add_subplot(gs[row, 0]), cum_x, "X"),
        (fig.add_subplot(gs[row, 1]), cum_y, "Y"),
    ]:
        noise_key = "noise_x" if axis_name == "X" else "noise_y"
        ax.plot(time_h, series, lw=0.8, alpha=0.6, color=col)
        valid = ~np.isnan(series)
        if valid.sum() >= SMOOTH_WINDOW:
            wl = min(SMOOTH_WINDOW, valid.sum() - 1)
            if wl % 2 == 0:
                wl -= 1
            trend = savgol_filter(series[valid], wl, SMOOTH_ORDER)
            ax.plot(time_h[valid], trend, lw=2.0, color=col,
                    label=f"trend (sigma_noise={results[label][noise_key]:.3f} um)")
        ax.set_ylabel("Cumulative drift (um)")
        ax.set_xlabel("Time (h)")
        ax.set_title(f"{label} — {axis_name}")
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

# ---- Summary display ----
print("\n=== Noise std summary (μm) ===")
print(f"{'Method':<35}  {'noise_x':>8}  {'noise_y':>8}")
for label in methods:
    print(f"{label:<35}  {results[label]['noise_x']:>8.4f}  {results[label]['noise_y']:>8.4f}")

best_x = min(methods, key=lambda l: results[l]["noise_x"] or 1e9)
best_y = min(methods, key=lambda l: results[l]["noise_y"] or 1e9)
print(f"\nBest X: {best_x}")
print(f"Best Y: {best_y}")

# ---- Save ----
save_figure(
    fig,
    params={
        "n_tp_with_channel_details": len(detail_records),
        "smooth_window": SMOOTH_WINDOW,
        "corr_threshold_B": CORR_THRESHOLD,
        **{f"noise_x_{k}": results[l]["noise_x"]
           for k, l in zip(["A", "B", "C", "D"], methods)},
        **{f"noise_y_{k}": results[l]["noise_y"]
           for k, l in zip(["A", "B", "C", "D"], methods)},
    },
    description="4 methods (MAD/corr-thresh/corr2-weighted/median) cumulative drift comparison",
    data={"time_h": time_h, "tp": tp_arr,
          **{f"cum_x_{k}": results[l]["cum_x"] for k, l in zip(["A","B","C","D"], methods)},
          **{f"cum_y_{k}": results[l]["cum_y"] for k, l in zip(["A","B","C","D"], methods)}},
)

plt.close(fig)
