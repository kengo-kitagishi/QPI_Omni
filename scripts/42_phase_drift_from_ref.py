# %%
# ============================================================
# 42_phase_drift_from_ref.py — 基準フレームからの位相ドリフト評価
# ============================================================
# 目的: frame[0] を基準とし、各フレーム n の差分
#       diff[n] = phase[n] - phase[0] を計算して
#       ROI mean（系全体のドリフト）と ROI std（空間的ばらつき）を
#       フレーム番号の関数としてプロットする。
#
# 対比:
#   41_phase_diff_noise.py → 隣接ペア差分（フレーム間ノイズ）
#   42_phase_drift_from_ref.py → 基準フレームからの累積ドリフト
# ============================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage.restoration import unwrap_phase

sys.path.insert(0, os.path.dirname(__file__))
from qpi import QPIParameters, get_field
from figure_logger import save_figure

# ============================================================
# 設定
# ============================================================

# %%
# --- 光学パラメータ ---

WAVELENGTH     = 658e-9          # [m]
NA             = 0.95
PIXELSIZE      = 3.45e-6 / 40   # [m/px]  3.45 µm カメラ画素 / 40x 対物
OFFAXIS_CENTER = (1710, 644)     # (row, col) — FFT空間でのオフアクシス中心

# --- データ設定 ---

DATA_DIR          = r"F:\basler\exp200ms_int1000ms_300frame\Pos0"
FRAME_INTERVAL_S  = 1.0     # [s] フレーム間隔（時間軸変換用）。5分=300, 1s=1.0
CROP_SIDE   = None          # "right" / "left" / None
CROP_SIZE   = 2048
CROP_REGION = None          # None or (r0, r1, c0, c1)
ROI_SIZE    = 80            # ノイズ計測 ROI サイズ [px]
ROI_CENTER  = None          # None → 再構成画像の中央

# ============================================================
# 初期化
# ============================================================

# %%

print("=== 42_phase_drift_from_ref: 基準フレームからの位相ドリフト評価 ===")

_exts = {".tif", ".tiff", ".png"}
_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if os.path.splitext(f)[1].lower() in _exts
)
N_total = len(_files)
print(f"  全フレーム数: {N_total}")

if N_total < 2:
    raise FileNotFoundError(f"フレームが不足しています: {DATA_DIR}")

_probe_raw = tifffile.imread(os.path.join(DATA_DIR, _files[0])).astype(np.float64)
_H_full, _W_full = _probe_raw.shape[:2]

if CROP_SIDE == "right":
    CROP_REGION = (0, _H_full, _W_full - CROP_SIZE, _W_full)
elif CROP_SIDE == "left":
    CROP_REGION = (0, _H_full, 0, CROP_SIZE)

if CROP_REGION is not None:
    r0, r1, c0, c1 = CROP_REGION
    _probe_raw = _probe_raw[r0:r1, c0:c1]
    print(f"  クロップ: {CROP_SIDE or 'manual'}  rows {r0}:{r1}, cols {c0}:{c1}")
img_shape = _probe_raw.shape[:2]
print(f"  画像サイズ (after crop): {img_shape}")

params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=img_shape,
    pixelsize=PIXELSIZE,
    offaxis_center=OFFAXIS_CENTER,
)

aperturesize = params.aperturesize
print(f"  aperturesize: {aperturesize} px")

_half = ROI_SIZE // 2
if ROI_CENTER is None:
    cr, cc = aperturesize // 2, aperturesize // 2
else:
    cr, cc = ROI_CENTER
rs, re = cr - _half, cr + _half
cs, ce = cc - _half, cc + _half
print(f"  ROI: rows {rs}:{re}, cols {cs}:{ce}  ({ROI_SIZE}x{ROI_SIZE} px)")


def _load_holo(fname: str) -> np.ndarray:
    raw = tifffile.imread(os.path.join(DATA_DIR, fname)).astype(np.float64)
    if CROP_REGION is not None:
        r0, r1, c0, c1 = CROP_REGION
        raw = raw[r0:r1, c0:c1]
    return raw


# ============================================================
# 基準フレームの再構成
# ============================================================

# %%

print("\n  基準フレーム (frame[0]) を再構成中...")
holo_ref  = _load_holo(_files[0])
field_ref = get_field(holo_ref, params)
phase_ref = unwrap_phase(np.angle(field_ref))
print("  完了")

# ============================================================
# 全フレームのループ（frame[0] を含む frame[0] - frame[0] = 0 から始める）
# ============================================================

# %%

frame_nums   = []
roi_mean_rad = []   # ROI mean of diff[n]
roi_std_rad  = []   # ROI std  of diff[n]

for n, fname in enumerate(_files):
    if n % 20 == 0:
        print(f"  frame {n}/{N_total - 1} ...")

    holo  = _load_holo(fname)
    field = get_field(holo, params)
    phase = unwrap_phase(np.angle(field))

    diff     = phase - phase_ref
    diff_roi = diff[rs:re, cs:ce]

    frame_nums.append(n)
    roi_mean_rad.append(float(np.mean(diff_roi)))
    roi_std_rad.append(float(np.std(diff_roi)))

frame_nums   = np.array(frame_nums)
time_min     = frame_nums * FRAME_INTERVAL_S / 60
roi_mean_rad = np.array(roi_mean_rad)
roi_std_rad  = np.array(roi_std_rad)
roi_mean_mrad = roi_mean_rad * 1e3
roi_std_mrad  = roi_std_rad  * 1e3
roi_mean_nm   = roi_mean_rad * WAVELENGTH / (2 * np.pi) * 1e9
roi_std_nm    = roi_std_rad  * WAVELENGTH / (2 * np.pi) * 1e9

# ============================================================
# 2π ジャンプ補正
# ============================================================
_TWO_PI = 2 * np.pi
_diff_series = np.diff(roi_mean_rad)
_jump_corr   = np.round(_diff_series / _TWO_PI) * _TWO_PI  # 2π の整数倍のジャンプ量
roi_mean_rad_corr  = roi_mean_rad.copy()
roi_mean_rad_corr[1:] -= np.cumsum(_jump_corr)              # 累積補正を引く
roi_mean_mrad_corr = roi_mean_rad_corr * 1e3
roi_mean_nm_corr   = roi_mean_rad_corr * WAVELENGTH / (2 * np.pi) * 1e9

n_jumps = int(np.sum(_jump_corr != 0))
print(f"  2π ジャンプ検出: {n_jumps} 回")
print(f"  補正後 ROI mean ドリフト幅 [mrad]: "
      f"min={roi_mean_mrad_corr.min():.2f}, max={roi_mean_mrad_corr.max():.2f}, "
      f"range={roi_mean_mrad_corr.max()-roi_mean_mrad_corr.min():.2f}")

# ============================================================
# 集計・stdout
# ============================================================

# %%

print(f"\n--- 結果 ---")
print(f"  ROI mean ドリフト幅 [mrad]: min={roi_mean_mrad.min():.2f}, max={roi_mean_mrad.max():.2f}, "
      f"range={roi_mean_mrad.max() - roi_mean_mrad.min():.2f}")
print(f"  ROI mean ドリフト幅 [nm]:   min={roi_mean_nm.min():.3f}, max={roi_mean_nm.max():.3f}, "
      f"range={roi_mean_nm.max() - roi_mean_nm.min():.3f}")
print(f"  ROI std  最終値   [mrad]: {roi_std_mrad[-1]:.2f}")
print(f"  ROI std  最終値   [nm]:   {roi_std_nm[-1]:.3f}")

# ============================================================
# プロット
# ============================================================

# %%

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

# 上左: ROI mean [mrad]
axes[0, 0].plot(time_min, roi_mean_mrad,      lw=0.5, color="steelblue",
                alpha=0.4, label="raw")
axes[0, 0].plot(time_min, roi_mean_mrad_corr, lw=0.9, color="steelblue",
                label=f"2π-corrected ({n_jumps} jumps removed)")
axes[0, 0].axhline(0, color="gray", ls="--", lw=0.8)
axes[0, 0].set_ylabel("ROI mean drift [mrad]")
axes[0, 0].set_title("Phase drift (ROI mean) [mrad]")
axes[0, 0].grid(True, alpha=0.4)
axes[0, 0].legend(fontsize=8)

# 上右: ROI mean [nm]
axes[0, 1].plot(time_min, roi_mean_nm,      lw=0.5, color="darkorange",
                alpha=0.4, label="raw")
axes[0, 1].plot(time_min, roi_mean_nm_corr, lw=0.9, color="darkorange",
                label=f"2π-corrected ({n_jumps} jumps removed)")
axes[0, 1].axhline(0, color="gray", ls="--", lw=0.8)
axes[0, 1].set_ylabel("ROI mean drift [nm OPD]")
axes[0, 1].set_title("Phase drift (ROI mean) [nm OPD]")
axes[0, 1].grid(True, alpha=0.4)
axes[0, 1].legend(fontsize=8)

# 下左: ROI std [mrad]
axes[1, 0].plot(time_min, roi_std_mrad, lw=0.8, color="steelblue")
axes[1, 0].set_ylabel("ROI std [mrad]")
axes[1, 0].set_xlabel(f"Time [min]  (interval={FRAME_INTERVAL_S}s/frame)")
axes[1, 0].set_title("Spatial spread of drift (ROI std) [mrad]")
axes[1, 0].grid(True, alpha=0.4)

# 下右: ROI std [nm]
axes[1, 1].plot(time_min, roi_std_nm, lw=0.8, color="darkorange")
axes[1, 1].set_ylabel("ROI std [nm OPD]")
axes[1, 1].set_xlabel(f"Time [min]  (interval={FRAME_INTERVAL_S}s/frame)")
axes[1, 1].set_title("Spatial spread of drift (ROI std) [nm OPD]")
axes[1, 1].grid(True, alpha=0.4)

plt.suptitle(
    f"Phase drift from frame[0]  |  "
    f"lambda={WAVELENGTH*1e9:.0f} nm, NA={NA}, aperturesize={aperturesize}  |  "
    f"ROI {ROI_SIZE}x{ROI_SIZE} center=({cr},{cc})",
    fontsize=10,
)
plt.tight_layout()

_params_fig = dict(
    data_dir=DATA_DIR,
    n_frames=N_total,
    roi_size=ROI_SIZE,
    roi_center=(int(cr), int(cc)),
    wavelength_nm=int(WAVELENGTH * 1e9),
    NA=NA,
    aperturesize=aperturesize,
    roi_mean_range_mrad=round(float(roi_mean_mrad.max() - roi_mean_mrad.min()), 3),
    roi_mean_range_nm=round(float(roi_mean_nm.max() - roi_mean_nm.min()), 4),
    roi_std_final_mrad=round(float(roi_std_mrad[-1]), 3),
    roi_std_final_nm=round(float(roi_std_nm[-1]), 4),
    roi_mean_corr_range_mrad=round(float(roi_mean_mrad_corr.max() - roi_mean_mrad_corr.min()), 3),
    n_2pi_jumps=n_jumps,
    frame_interval_s=FRAME_INTERVAL_S,
)

save_figure(
    fig,
    params=_params_fig,
    description=(
        f"基準フレームからの位相ドリフト: "
        f"ROI mean range={roi_mean_mrad.max()-roi_mean_mrad.min():.2f} mrad (raw), "
        f"補正後 range={roi_mean_mrad_corr.max()-roi_mean_mrad_corr.min():.2f} mrad "
        f"({n_jumps} 2π jumps removed), "
        f"ROI std (final)={roi_std_mrad[-1]:.2f} mrad, "
        f"{N_total} frames"
    ),
    data={
        "frame_nums":         frame_nums,
        "roi_mean_mrad":      roi_mean_mrad,
        "roi_std_mrad":       roi_std_mrad,
        "roi_mean_nm":        roi_mean_nm,
        "roi_std_nm":         roi_std_nm,
        "roi_mean_mrad_corr": roi_mean_mrad_corr,
        "roi_mean_nm_corr":   roi_mean_nm_corr,
    },
)

print(f"\n完了: {N_total} フレーム処理")

# %%
