# %%
# ============================================================
# 41_42_from_phase_planes.py — 再構成済み位相マップからノイズ・ドリフト評価
# ============================================================
# 目的: すでに再構成された位相 TIFF（unwrap済み float32）を直接読み込み、
#       41_phase_diff_noise.py + 42_phase_drift_from_ref.py の
#       解析を再構成ステップなしで実行する。
#       shot noise の理論値はホログラムが不要なので手動入力
#       （SIGMA_SHOT_MRAD）で比較ラインを引く。
# ============================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from figure_logger import save_figure

# ============================================================
# 設定
# ============================================================

# %%

# --- データ設定 ---
DATA_DIR         = r"F:\timelapse_11day_exp200ms_1pos_EMM2\Pos1\output_phase_planesub"
FRAME_INTERVAL_S = 300.0    # [s] フレーム間隔（5分=300）
WAVELENGTH       = 658e-9   # [m] 波長（nm 変換用）

# --- ROI 設定（位相マップ座標） ---
ROI_SIZE   = 80             # [px]
ROI_CENTER = None           # None → 画像中央

# --- shot noise 参照ライン（手動入力、None で非表示）---
SIGMA_SHOT_MRAD  = None     # e.g. 4.5  ← 41.py で得た値を入れる

# ============================================================
# 初期化
# ============================================================

# %%

print("=== 41_42_from_phase_planes: 再構成済み位相ノイズ・ドリフト評価 ===")

_exts = {".tif", ".tiff"}
_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if os.path.splitext(f)[1].lower() in _exts
)
N_total = len(_files)
print(f"  フレーム数: {N_total}")
if N_total < 2:
    raise FileNotFoundError(f"フレームが不足: {DATA_DIR}")

# 画像サイズ確認
_probe = tifffile.imread(os.path.join(DATA_DIR, _files[0])).astype(np.float64)
H, W = _probe.shape[:2]
print(f"  画像サイズ: {H}x{W}  dtype: {tifffile.imread(os.path.join(DATA_DIR, _files[0])).dtype}")
print(f"  値域サンプル: min={_probe.min():.3f}  max={_probe.max():.3f} rad")

# ROI 設定
_half = ROI_SIZE // 2
if ROI_CENTER is None:
    cr, cc = H // 2, W // 2
else:
    cr, cc = ROI_CENTER
rs, re = cr - _half, cr + _half
cs, ce = cc - _half, cc + _half
print(f"  ROI: rows {rs}:{re}, cols {cs}:{ce}  ({ROI_SIZE}x{ROI_SIZE} px)")


def _load_phase(fname: str) -> np.ndarray:
    """位相 TIFF 1 枚を float64 で返す（すでに unwrap 済みと仮定）"""
    return tifffile.imread(os.path.join(DATA_DIR, fname)).astype(np.float64)


# ============================================================
# 41 相当: 隣接差分ノイズ
# ============================================================

# %%

print("\n[41] 隣接差分ノイズ計算中...")

n_pairs = N_total - 1

def _compute_pair_noise(i):
    ph0 = _load_phase(_files[i])
    ph1 = _load_phase(_files[i + 1])
    diff_roi = (ph1 - ph0)[rs:re, cs:ce]
    noise_rad = float(np.std(diff_roi) / np.sqrt(2))
    return noise_rad * 1e3, diff_roi.copy()

with ThreadPoolExecutor(max_workers=None) as pool:
    pair_results = list(pool.map(_compute_pair_noise, range(n_pairs)))

pair_noise_mrad = np.array([r[0] for r in pair_results])
diff_roi_stack  = np.stack([r[1] for r in pair_results], axis=0)
print(f"  {n_pairs} pairs computed (parallel)")

# 統計
noise_mean_mrad = float(pair_noise_mrad.mean())
noise_std_mrad  = float(pair_noise_mrad.std())
noise_mean_nm   = noise_mean_mrad * 1e-3 * WAVELENGTH / (2 * np.pi) * 1e9
noise_std_nm    = noise_std_mrad  * 1e-3 * WAVELENGTH / (2 * np.pi) * 1e9

print(f"  noise mean: {noise_mean_mrad:.2f} ± {noise_std_mrad:.2f} mrad/frame")
print(f"           = {noise_mean_nm:.4f} ± {noise_std_nm:.4f} nm OPD/frame")
if SIGMA_SHOT_MRAD is not None:
    ratio = noise_mean_mrad / SIGMA_SHOT_MRAD
    print(f"  σ_shot:    {SIGMA_SHOT_MRAD:.2f} mrad")
    print(f"  measured / σ_shot = {ratio:.2f}x")

# ============================================================
# 42 相当: 基準フレームからのドリフト
# ============================================================

# %%

print("\n[42] 基準フレームからのドリフト計算中...")

phase_ref = _load_phase(_files[0])

def _compute_drift(fname):
    ph = _load_phase(fname)
    diff_roi = (ph - phase_ref)[rs:re, cs:ce]
    return float(np.mean(diff_roi) * 1e3), float(np.std(diff_roi) * 1e3)

with ThreadPoolExecutor(max_workers=None) as pool:
    drift_results = list(pool.map(_compute_drift, _files))

roi_mean_mrad = np.array([r[0] for r in drift_results])
roi_std_mrad  = np.array([r[1] for r in drift_results])
print(f"  {N_total} frames computed (parallel)")
roi_mean_nm   = roi_mean_mrad * 1e-3 * WAVELENGTH / (2 * np.pi) * 1e9

# 2π ジャンプ補正
_diff_series   = np.diff(roi_mean_mrad * 1e-3)
_TWO_PI        = 2 * np.pi
_jump_corr     = np.round(_diff_series / _TWO_PI) * _TWO_PI
roi_mean_rad_corr = (roi_mean_mrad * 1e-3).copy()
roi_mean_rad_corr[1:] -= np.cumsum(_jump_corr)
roi_mean_mrad_corr = roi_mean_rad_corr * 1e3
roi_mean_nm_corr   = roi_mean_rad_corr * WAVELENGTH / (2 * np.pi) * 1e9
n_jumps = int(np.sum(_jump_corr != 0))

print(f"  2π ジャンプ検出: {n_jumps} 回")
print(f"  ROI mean drift range (raw)  [mrad]: {roi_mean_mrad.max()-roi_mean_mrad.min():.2f}")
print(f"  ROI mean drift range (corr) [mrad]: {roi_mean_mrad_corr.max()-roi_mean_mrad_corr.min():.2f}")

# ============================================================
# 時間軸
# ============================================================

pair_time_min  = np.arange(n_pairs) * FRAME_INTERVAL_S / 60
frame_time_min = np.arange(N_total) * FRAME_INTERVAL_S / 60

# ============================================================
# プロット
# ============================================================

# %%
# --- 図1: 隣接差分ノイズ [mrad] ---

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(pair_time_min, pair_noise_mrad, lw=0.7, color="steelblue", label="measured (adjacent diff)")
ax1.axhline(noise_mean_mrad, color="red", ls="--",
            label=f"mean = {noise_mean_mrad:.2f} mrad")
if SIGMA_SHOT_MRAD is not None:
    ax1.axhline(SIGMA_SHOT_MRAD, color="green", ls=":",
                label=f"σ_shot = {SIGMA_SHOT_MRAD:.2f} mrad  →  ratio = {noise_mean_mrad/SIGMA_SHOT_MRAD:.2f}x")
ax1.set_ylabel("Phase noise [mrad/frame]")
ax1.set_xlabel(f"Time [min]  (interval={FRAME_INTERVAL_S:.0f}s/frame, N={n_pairs} pairs)")
ax1.set_title(
    f"Adjacent-diff noise (from phase planes)  |  ROI {ROI_SIZE}x{ROI_SIZE} center=({cr},{cc})\n"
    f"mean = {noise_mean_mrad:.2f} mrad = {noise_mean_nm:.4f} nm OPD"
)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.4)
fig1.tight_layout()

_params1 = dict(
    data_dir=DATA_DIR,
    n_frames=N_total,
    n_pairs=n_pairs,
    roi_size=ROI_SIZE,
    roi_center=(int(cr), int(cc)),
    wavelength_nm=int(WAVELENGTH * 1e9),
    noise_mean_mrad=round(noise_mean_mrad, 3),
    noise_std_mrad=round(noise_std_mrad, 3),
    noise_mean_nm=round(noise_mean_nm, 4),
    sigma_shot_mrad=SIGMA_SHOT_MRAD,
    frame_interval_s=FRAME_INTERVAL_S,
)
save_figure(
    fig1,
    params=_params1,
    description=(
        f"隣接差分ノイズ（再構成済み位相マップから）: "
        f"mean={noise_mean_mrad:.2f}±{noise_std_mrad:.2f} mrad "
        f"({noise_mean_nm:.4f} nm OPD)  "
        + (f"σ_shot={SIGMA_SHOT_MRAD} mrad, ratio={noise_mean_mrad/SIGMA_SHOT_MRAD:.2f}x"
           if SIGMA_SHOT_MRAD else "σ_shot=None")
    ),
)

# %%
# --- 図2: ドリフト（基準フレームから）---

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

# 上左: ROI mean [mrad]
axes2[0, 0].plot(frame_time_min, roi_mean_mrad, lw=0.5, color="steelblue", alpha=0.4, label="raw")
axes2[0, 0].plot(frame_time_min, roi_mean_mrad_corr, lw=0.9, color="steelblue",
                 label=f"2π-corrected ({n_jumps} jumps)")
axes2[0, 0].axhline(0, color="gray", ls="--", lw=0.8)
axes2[0, 0].set_ylabel("ROI mean drift [mrad]")
axes2[0, 0].set_title("Phase drift (ROI mean) [mrad]")
axes2[0, 0].legend(fontsize=8)
axes2[0, 0].grid(True, alpha=0.4)

# 上右: ROI mean [nm]
axes2[0, 1].plot(frame_time_min, roi_mean_nm, lw=0.5, color="darkorange", alpha=0.4, label="raw")
axes2[0, 1].plot(frame_time_min, roi_mean_nm_corr, lw=0.9, color="darkorange",
                 label=f"2π-corrected ({n_jumps} jumps)")
axes2[0, 1].axhline(0, color="gray", ls="--", lw=0.8)
axes2[0, 1].set_ylabel("ROI mean drift [nm OPD]")
axes2[0, 1].set_title("Phase drift (ROI mean) [nm OPD]")
axes2[0, 1].legend(fontsize=8)
axes2[0, 1].grid(True, alpha=0.4)

# 下左: ROI std [mrad]
axes2[1, 0].plot(frame_time_min, roi_std_mrad, lw=0.8, color="steelblue")
axes2[1, 0].set_ylabel("ROI std [mrad]")
axes2[1, 0].set_xlabel(f"Time [min]  (interval={FRAME_INTERVAL_S:.0f}s/frame)")
axes2[1, 0].set_title("Spatial spread of drift (ROI std) [mrad]")
axes2[1, 0].grid(True, alpha=0.4)

# 下右: ROI std vs noise_mean 比較
axes2[1, 1].plot(frame_time_min, roi_std_mrad, lw=0.8, color="steelblue", label="ROI std (drift spread)")
axes2[1, 1].axhline(noise_mean_mrad, color="red", ls="--",
                    label=f"adj-diff noise = {noise_mean_mrad:.2f} mrad")
if SIGMA_SHOT_MRAD is not None:
    axes2[1, 1].axhline(SIGMA_SHOT_MRAD, color="green", ls=":",
                        label=f"σ_shot = {SIGMA_SHOT_MRAD:.2f} mrad")
axes2[1, 1].set_ylabel("mrad")
axes2[1, 1].set_xlabel(f"Time [min]")
axes2[1, 1].set_title("ROI std vs noise reference lines")
axes2[1, 1].legend(fontsize=8)
axes2[1, 1].grid(True, alpha=0.4)

plt.suptitle(
    f"Phase drift from frame[0]  |  {N_total} frames  |  "
    f"ROI {ROI_SIZE}x{ROI_SIZE} center=({cr},{cc})",
    fontsize=10,
)
plt.tight_layout()

_params2 = dict(
    data_dir=DATA_DIR,
    n_frames=N_total,
    roi_size=ROI_SIZE,
    roi_center=(int(cr), int(cc)),
    wavelength_nm=int(WAVELENGTH * 1e9),
    roi_mean_range_mrad=round(float(roi_mean_mrad.max() - roi_mean_mrad.min()), 3),
    roi_mean_range_corr_mrad=round(float(roi_mean_mrad_corr.max() - roi_mean_mrad_corr.min()), 3),
    roi_std_final_mrad=round(float(roi_std_mrad[-1]), 3),
    n_2pi_jumps=n_jumps,
    frame_interval_s=FRAME_INTERVAL_S,
)
save_figure(
    fig2,
    params=_params2,
    description=(
        f"基準フレームからのドリフト（再構成済み位相から）: "
        f"ROI mean range={roi_mean_mrad.max()-roi_mean_mrad.min():.2f} mrad (raw), "
        f"補正後={roi_mean_mrad_corr.max()-roi_mean_mrad_corr.min():.2f} mrad, "
        f"ROI std (final)={roi_std_mrad[-1]:.2f} mrad, "
        f"{N_total} frames, {n_jumps} 2π jumps"
    ),
)

# %%

print(f"\n完了: {N_total} frames, {n_pairs} adj-pairs")
plt.show()
