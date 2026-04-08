"""analyze_channel_drift.py — チャネル別 ECC データを使って複数の集計手法を比較検証する。

drift_log.json に channel_details が含まれる TP を対象に、
4 種類の集計方法で tx/ty 時系列を再計算し、ノイズの大きさを比較する。

使用条件: channel_details が 50 TP 以上蓄積していること。

評価指標: Savitzky-Golay フィルタで低周波トレンドを除去した後の残差 std
  → 最小 = 真のドリフトを最もよく再現している
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

# ---- 設定 ----
DRIFT_LOG      = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log.json")
INTERVAL_MIN   = 5.0
SMOOTH_WINDOW  = 51    # フレーム数（~4 時間窓）。奇数であること
SMOOTH_ORDER   = 2
CORR_THRESHOLD = 0.98  # 手法 B: corr2 < この値を除外
PIXEL_SCALE_UM = 0.3462  # μm/px

# ---- データ読み込み ----
records = json.loads(DRIFT_LOG.read_text(encoding="utf-8"))

# channel_details があるレコードのみ使用
detail_records = [r for r in records if r.get("channel_details")]
if len(detail_records) < SMOOTH_WINDOW:
    print(f"ERROR: channel_details が {len(detail_records)} TP しかない（最低 {SMOOTH_WINDOW} TP 必要）")
    sys.exit(1)

print(f"channel_details あり: {len(detail_records)} TP / 全 {len(records)} TP")

tp_arr = np.array([r["timepoint"] for r in detail_records])
time_h = tp_arr * INTERVAL_MIN / 60


def _aggregate(method: str, records_: list) -> tuple[np.ndarray, np.ndarray]:
    """各レコードの channel_details から tx/ty を指定手法で集計する。"""
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
            # 現行: MAD 外れ値除外 → 単純平均
            mask = ~_mad_outlier(tx_all) & ~_mad_outlier(ty_all)
            if mask.sum() == 0:
                mask = np.ones(len(tx_all), dtype=bool)
            tx_out.append(tx_all[mask].mean())
            ty_out.append(ty_all[mask].mean())

        elif method == "B":
            # corr 閾値除外 → 単純平均
            mask = corr_all >= CORR_THRESHOLD
            if mask.sum() == 0:
                mask = np.ones(len(tx_all), dtype=bool)
            tx_out.append(tx_all[mask].mean())
            ty_out.append(ty_all[mask].mean())

        elif method == "C":
            # MAD 保持 + corr² 重み付き平均
            mask = ~_mad_outlier(tx_all) & ~_mad_outlier(ty_all)
            if mask.sum() == 0:
                mask = np.ones(len(tx_all), dtype=bool)
            w = corr_all[mask] ** 2
            if w.sum() == 0:
                w = np.ones(mask.sum())
            tx_out.append(np.average(tx_all[mask], weights=w))
            ty_out.append(np.average(ty_all[mask], weights=w))

        elif method == "D":
            # 外れ値除去なし → 単純 median
            tx_out.append(np.median(tx_all))
            ty_out.append(np.median(ty_all))

    return np.array(tx_out), np.array(ty_out)


def _mad_outlier(arr: np.ndarray, thresh: float = 5.0) -> np.ndarray:
    """MAD ベースの外れ値フラグ（True = 外れ値）"""
    if len(arr) < 3:
        return np.zeros(len(arr), dtype=bool)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad == 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(arr - med) > thresh * mad


def noise_std(series: np.ndarray) -> float:
    """Savitzky-Golay で低周波成分を除去した残差の std"""
    valid = ~np.isnan(series)
    if valid.sum() < SMOOTH_WINDOW:
        return np.nan
    wl = min(SMOOTH_WINDOW, valid.sum() - 1)
    if wl % 2 == 0:
        wl -= 1
    trend = savgol_filter(series[valid], window_length=wl, polyorder=SMOOTH_ORDER)
    return float(np.std(series[valid] - trend))


# ---- 各手法で集計 ----
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

# ---- 図 ----
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

# ---- サマリー表示 ----
print("\n=== Noise std summary (μm) ===")
print(f"{'Method':<35}  {'noise_x':>8}  {'noise_y':>8}")
for label in methods:
    print(f"{label:<35}  {results[label]['noise_x']:>8.4f}  {results[label]['noise_y']:>8.4f}")

best_x = min(methods, key=lambda l: results[l]["noise_x"] or 1e9)
best_y = min(methods, key=lambda l: results[l]["noise_y"] or 1e9)
print(f"\nBest X: {best_x}")
print(f"Best Y: {best_y}")

# ---- 保存 ----
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
