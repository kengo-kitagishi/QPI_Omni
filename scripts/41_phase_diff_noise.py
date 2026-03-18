# %%
# ============================================================
# 41_phase_diff_noise.py — 位相再構成後の時間ノイズ評価
# ============================================================
# 目的: ホログラム隣接ペアを位相再構成し、差分から temporal noise を
#       [mrad/frame] および [nm OPD/frame] で定量化する。
#       論文 eq. A.9 / A.12 / A.14 の理論限界と比較する。
#
# 対比:
#   40_qpi_noise_analysis.py UC_DIFF → 生画像 [ADU] の差分（センサーノイズのみ）
#   41_phase_diff_noise.py    → 位相再構成後 [rad] の差分（光学ノイズ全体）
# ============================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage.restoration import unwrap_phase

sys.path.insert(0, os.path.dirname(__file__))
from qpi import QPIParameters, get_field, _get_dc_ac, _get_visibility, _get_phase_noise
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

# --- センサーパラメータ（Basler acA2440-75um, EMVA1288 準拠）---
SENSOR_FULL_WELL_E  = 10340      # [e⁻] Full Well Capacity
SENSOR_BIT_DEPTH    = 12         # 有効ビット深度
SENSOR_GAIN         = 1.0        # ソフトウェアゲイン設定（0 dB = 1.0）
SENSOR_CONVERSION_GAIN = SENSOR_FULL_WELL_E / (2 ** SENSOR_BIT_DEPTH) / SENSOR_GAIN
# ≈ 2.52 e⁻/ADU

# 読み出しノイズ [e⁻]。EMVA1288 実測値がある場合に入力。
# None の場合はショットノイズのみ（σ_total = σ_shot）。
# Basler acA2440-75um 代表値: ~3 e⁻
SENSOR_READ_NOISE_E = None       # e.g. 3.0

# --- データ設定 ---

DATA_DIR          = r"F:\basler\exp200ms_int1000ms_300frame\Pos0"   # ホログラム TIFF が並ぶディレクトリ
FRAME_INTERVAL_S  = 1.0     # [s] フレーム間隔（時間軸変換用）。5分=300, 1s=1.0
CROP_SIDE         = None         # "right" → 右端, "left" → 左端, None → CROP_REGION を直接使う
CROP_SIZE         = 2048         # CROP_SIDE 使用時の正方形サイズ [px]
CROP_REGION       = None         # None or (r0, r1, c0, c1)（CROP_SIDE=None のときだけ参照）
ROI_SIZE          = 80           # ノイズ計測 ROI サイズ [px]（論文準拠 80×80）
ROI_CENTER        = None         # None → 再構成画像の中央。(row, col) で明示指定も可
PAIR_START_1BASED = 1            # 解析開始ペア番号（1 始まり）
PAIR_END_1BASED   = 150           # 解析終了ペア番号（1 始まり）

# ============================================================
# 初期化・パラメータ確認
# ============================================================

# %%

print("=== 41_phase_diff_noise: 位相再構成ノイズ評価 ===")
print(f"  Conversion gain: {SENSOR_CONVERSION_GAIN:.4f} e⁻/ADU")
if SENSOR_READ_NOISE_E is not None:
    sigma_s_adu = SENSOR_READ_NOISE_E / SENSOR_CONVERSION_GAIN
    print(f"  Read noise: {SENSOR_READ_NOISE_E:.1f} e⁻ = {sigma_s_adu:.3f} ADU")
else:
    sigma_s_adu = None
    print("  Read noise: not set (shot noise only)")

# ファイルリスト取得
_exts = {".tif", ".tiff", ".png"}
_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if os.path.splitext(f)[1].lower() in _exts
)
N_total = len(_files)
n_pairs_total = N_total // 2
print(f"  全フレーム数: {N_total} → 総ペア数: {n_pairs_total}")

if N_total == 0:
    raise FileNotFoundError(f"ホログラムファイルが見つかりません: {DATA_DIR}")

# 1 枚プローブ読み込みで画像サイズを確定、CROP_REGION を解決
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

# QPIParameters 生成（aperturesize は自動計算）
params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=img_shape,
    pixelsize=PIXELSIZE,
    offaxis_center=OFFAXIS_CENTER,
)

aperturesize = params.aperturesize
recon_shape  = (aperturesize, aperturesize)
A_aperture   = np.pi * (aperturesize / 2) ** 2
A_sensor     = img_shape[0] * img_shape[1]

print(f"  aperturesize: {aperturesize} px")
print(f"  recon_shape:  {recon_shape}")
print(f"  A_aperture:   {A_aperture:.0f} px²")
print(f"  A_sensor:     {A_sensor:.0f} px²")
print(f"  A_ap/A_sen:   {A_aperture/A_sensor:.4f}")

# ペア範囲チェック
pair_start_idx = PAIR_START_1BASED - 1
pair_end_idx   = PAIR_END_1BASED   - 1
if pair_start_idx < 0 or pair_end_idx < pair_start_idx:
    raise ValueError(f"ペア番号の範囲が無効です: {PAIR_START_1BASED}-{PAIR_END_1BASED}")
if pair_end_idx >= n_pairs_total:
    raise ValueError(
        f"ペア番号が範囲外: 要求 {PAIR_START_1BASED}-{PAIR_END_1BASED}, "
        f"利用可能 1-{n_pairs_total}"
    )

selected_pair_idx = np.arange(pair_start_idx, pair_end_idx + 1, dtype=int)
n_pairs = len(selected_pair_idx)
print(f"\n  解析対象ペア: {PAIR_START_1BASED}-{PAIR_END_1BASED} (N={n_pairs})")

# ROI 設定（再構成画像座標で定義）
_half = ROI_SIZE // 2
if ROI_CENTER is None:
    cr, cc = aperturesize // 2, aperturesize // 2
else:
    cr, cc = ROI_CENTER
rs, re = cr - _half, cr + _half
cs, ce = cc - _half, cc + _half
print(f"  ROI: rows {rs}:{re}, cols {cs}:{ce}  ({ROI_SIZE}×{ROI_SIZE} px, 再構成画像座標)")


def _load_holo(fname: str) -> np.ndarray:
    """ホログラム 1 枚を読み込み float64 で返す（CROP_REGION を適用）"""
    raw = tifffile.imread(os.path.join(DATA_DIR, fname)).astype(np.float64)
    if CROP_REGION is not None:
        r0, r1, c0, c1 = CROP_REGION
        raw = raw[r0:r1, c0:c1]
    return raw


# ============================================================
# ペアごとのループ
# ============================================================

# %%

pair_nums       = []
noise_rad_vals  = []   # [rad/frame]
noise_nm_vals   = []   # [nm OPD/frame]
diff_roi_stack  = []   # [pair, y, x] : temporal-axis diagnostics

# 理論値は最初の有効ペアから 1 回だけ計算する
sigma_shot_mrad   = None
sigma_sensor_mrad = None
sigma_total_mrad  = None
sigma_shot_nm     = None
sigma_sensor_nm   = None
sigma_total_nm    = None

for _i in selected_pair_idx:
    holo0 = _load_holo(_files[2 * _i    ])
    holo1 = _load_holo(_files[2 * _i + 1])

    # 位相再構成
    field0 = get_field(holo0, params)
    field1 = get_field(holo1, params)

    phase0 = unwrap_phase(np.angle(field0))
    phase1 = unwrap_phase(np.angle(field1))

    diff = phase1 - phase0

    # ROI 内の std / √2 = per-frame noise
    diff_roi = diff[rs:re, cs:ce]
    noise_rad = float(np.std(diff_roi) / np.sqrt(2))
    noise_nm  = noise_rad * WAVELENGTH / (2 * np.pi) * 1e9

    pair_nums.append(_i + 1)
    noise_rad_vals.append(noise_rad)
    noise_nm_vals.append(noise_nm)
    diff_roi_stack.append(diff_roi.copy())

    # 理論値計算（最初のペアのみ）
    if sigma_shot_mrad is None:
        dc, ac = _get_dc_ac(holo0, params)
        vis    = _get_visibility(dc, ac)
        vis_roi = vis[rs:re, cs:ce]
        dc_abs_roi = np.abs(dc)[rs:re, cs:ce]

        # σ_shot (eq. A.9): 実測ホログラムから得た visibility/DC を
        # 実測ノイズと同じ ROI (80x80) で評価する
        N_electron_map = dc_abs_roi * SENSOR_CONVERSION_GAIN  # ADU → e⁻
        sigma_shot_map = _get_phase_noise(
            vis_roi, aperturesize, N_electron_map, img_shape
        )
        sigma_shot_scalar = float(np.mean(sigma_shot_map))

        # σ_sensor (eq. A.12)
        if sigma_s_adu is not None:
            sigma_sensor_map = np.sqrt(
                2 * sigma_s_adu**2 * A_aperture
                / (vis_roi**2 * dc_abs_roi**2 * A_sensor)
            )
            sigma_sensor_scalar = float(np.mean(sigma_sensor_map))
        else:
            sigma_sensor_scalar = 0.0

        # σ_total (eq. A.14)
        sigma_total_scalar = float(np.sqrt(
            sigma_shot_scalar**2 + sigma_sensor_scalar**2
        ))

        # [rad] → [mrad], [nm]
        sigma_shot_mrad   = sigma_shot_scalar   * 1e3
        sigma_sensor_mrad = sigma_sensor_scalar * 1e3
        sigma_total_mrad  = sigma_total_scalar  * 1e3
        sigma_shot_nm     = sigma_shot_scalar   * WAVELENGTH / (2 * np.pi) * 1e9
        sigma_sensor_nm   = sigma_sensor_scalar * WAVELENGTH / (2 * np.pi) * 1e9
        sigma_total_nm    = sigma_total_scalar  * WAVELENGTH / (2 * np.pi) * 1e9

pair_nums      = np.array(pair_nums)
# pair k (1-based) uses frames 2k-2 and 2k-1; x-axis = start time of the pair
time_min       = (pair_nums - 1) * 2 * FRAME_INTERVAL_S / 60
noise_rad_vals = np.array(noise_rad_vals)
noise_nm_vals  = np.array(noise_nm_vals)
noise_mrad_vals = noise_rad_vals * 1e3
diff_roi_stack = np.stack(diff_roi_stack, axis=0)

# ============================================================
# 集計・stdout 出力
# ============================================================

# %%

noise_mean_mrad = float(noise_mrad_vals.mean())
noise_std_mrad  = float(noise_mrad_vals.std())
noise_mean_nm   = float(noise_nm_vals.mean())
noise_std_nm    = float(noise_nm_vals.std())

# 追加診断:
# 1) pixel-wise temporal std: 各画素で pair 軸 std を計算し、ROI 内平均
# 2) ROI-mean temporal std: 各 pair の ROI 平均値の時系列 std
temporal_std_map_rad = np.std(diff_roi_stack, axis=0) / np.sqrt(2)
temporal_std_map_mean_mrad = float(np.mean(temporal_std_map_rad) * 1e3)
temporal_std_map_std_mrad  = float(np.std(temporal_std_map_rad) * 1e3)
roi_mean_series_rad = np.mean(diff_roi_stack, axis=(1, 2))
roi_mean_temporal_std_mrad = float(np.std(roi_mean_series_rad) / np.sqrt(2) * 1e3)

print("\n--- 実測値の定義 ---")
print(
    "  main metric = mean_i[ std_xy( phi(2i+1)-phi(2i) ) / sqrt(2) ] "
    "(ROI内・空間stdをペア平均)"
)
print("\n--- 実測値 ---")
print(f"  main noise [mrad/frame]: {noise_mean_mrad:.2f} ± {noise_std_mrad:.2f}")
print(f"  main noise [nm OPD/frame]: {noise_mean_nm:.3f} ± {noise_std_nm:.3f}")
print(
    f"  temporal pixel std mean [mrad/frame]: {temporal_std_map_mean_mrad:.2f} "
    f"(ROI内pixel-wise temporal stdの平均, spatial std={temporal_std_map_std_mrad:.2f})"
)
print(
    f"  ROI-mean temporal std [mrad/frame]: {roi_mean_temporal_std_mrad:.3f} "
    f"(ROI平均位相差の時間ばらつき)"
)

print("\n--- 理論限界（最初のペアから算出）---")
print(f"  σ_shot   : {sigma_shot_mrad:.3f} mrad  /  {sigma_shot_nm:.4f} nm  (eq. A.9)")
if sigma_s_adu is not None:
    print(f"  σ_sensor : {sigma_sensor_mrad:.3f} mrad  /  {sigma_sensor_nm:.4f} nm  (eq. A.12)")
else:
    print("  σ_sensor : not computed (read noise not set)")
print(f"  σ_total  : {sigma_total_mrad:.3f} mrad  /  {sigma_total_nm:.4f} nm  (eq. A.14)")

ratio_shot  = noise_mean_mrad / sigma_shot_mrad  if sigma_shot_mrad  > 0 else np.nan
ratio_total = noise_mean_mrad / sigma_total_mrad if sigma_total_mrad > 0 else np.nan
print(f"\n  実測 / σ_shot  = {ratio_shot:.2f}x")
print(f"  実測 / σ_total = {ratio_total:.2f}x")

# ============================================================
# プロット
# ============================================================

# %%
# --- 図1: combined (mrad + nm) ---

fig_combined, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# 上段: mrad
axes[0].plot(time_min, noise_mrad_vals, lw=0.8, color="steelblue", label="measured")
axes[0].axhline(noise_mean_mrad, color="red",   ls="--",
                label=f"mean = {noise_mean_mrad:.2f} mrad")
axes[0].axhline(sigma_shot_mrad, color="green", ls=":",
                label=f"σ_shot = {sigma_shot_mrad:.2f} mrad  (eq. A.9)")
if sigma_s_adu is not None:
    axes[0].axhline(sigma_total_mrad, color="purple", ls="-.",
                    label=f"σ_total = {sigma_total_mrad:.2f} mrad  (eq. A.14)")
axes[0].set_ylabel("Phase noise [mrad/frame]")
axes[0].set_title(
    f"Phase diff noise  |  ROI {ROI_SIZE}×{ROI_SIZE} center=({cr},{cc})  |  "
    f"measured/σ_shot={ratio_shot:.2f}x"
)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.4)

# 下段: nm OPD
axes[1].plot(time_min, noise_nm_vals, lw=0.8, color="darkorange", label="measured")
axes[1].axhline(noise_mean_nm, color="red",   ls="--",
                label=f"mean = {noise_mean_nm:.3f} nm")
axes[1].axhline(sigma_shot_nm, color="green", ls=":",
                label=f"σ_shot = {sigma_shot_nm:.4f} nm  (eq. A.9)")
if sigma_s_adu is not None:
    axes[1].axhline(sigma_total_nm, color="purple", ls="-.",
                    label=f"σ_total = {sigma_total_nm:.4f} nm  (eq. A.14)")
axes[1].set_ylabel("OPD noise [nm/frame]")
axes[1].set_xlabel(
    f"Time [min]  (interval={FRAME_INTERVAL_S}s/frame, "
    f"N={n_pairs}, pair {PAIR_START_1BASED}-{PAIR_END_1BASED} / {n_pairs_total})"
)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.4)

plt.suptitle(
    f"Phase reconstruction temporal noise  |  "
    f"λ={WAVELENGTH*1e9:.0f} nm, NA={NA}, aperturesize={aperturesize}",
    fontsize=11
)
plt.tight_layout()

_params_common = dict(
    data_dir=DATA_DIR,
    n_frames=N_total,
    n_pairs_total=n_pairs_total,
    n_pairs=n_pairs,
    pair_start_1based=PAIR_START_1BASED,
    pair_end_1based=PAIR_END_1BASED,
    roi_size=ROI_SIZE,
    roi_center=(int(cr), int(cc)),
    wavelength_nm=int(WAVELENGTH * 1e9),
    NA=NA,
    aperturesize=aperturesize,
    A_ap_over_A_sen=round(A_aperture / A_sensor, 4),
    noise_mean_mrad=round(noise_mean_mrad, 3),
    noise_std_mrad=round(noise_std_mrad, 3),
    noise_mean_nm=round(noise_mean_nm, 4),
    noise_std_nm=round(noise_std_nm, 4),
    temporal_pixel_std_mean_mrad=round(temporal_std_map_mean_mrad, 3),
    temporal_pixel_std_spatial_std_mrad=round(temporal_std_map_std_mrad, 3),
    roi_mean_temporal_std_mrad=round(roi_mean_temporal_std_mrad, 3),
    sigma_shot_mrad=round(sigma_shot_mrad, 3),
    sigma_total_mrad=round(sigma_total_mrad, 3),
    sigma_shot_nm=round(sigma_shot_nm, 4),
    sigma_total_nm=round(sigma_total_nm, 4),
    measured_over_shot=round(float(ratio_shot), 3),
    measured_over_total=round(float(ratio_total), 3),
    read_noise_e=SENSOR_READ_NOISE_E,
    conversion_gain=round(SENSOR_CONVERSION_GAIN, 4),
    frame_interval_s=FRAME_INTERVAL_S,
)

_data_common = {
    "pair_nums":      pair_nums,
    "noise_mrad":     noise_mrad_vals,
    "noise_nm":       noise_nm_vals,
    "roi_mean_series_rad": roi_mean_series_rad,
    "temporal_std_map_rad": temporal_std_map_rad,
}

save_figure(
    fig_combined,
    params=_params_common,
    description=(
        f"位相再構成ノイズ評価（combined）: "
        f"measured={noise_mean_mrad:.2f}±{noise_std_mrad:.2f} mrad "
        f"({noise_mean_nm:.3f}±{noise_std_nm:.3f} nm), "
        f"σ_shot={sigma_shot_mrad:.2f} mrad, σ_total={sigma_total_mrad:.2f} mrad, "
        f"measured/shot={ratio_shot:.2f}x  "
        f"({n_pairs} pairs, {PAIR_START_1BASED}-{PAIR_END_1BASED})"
    ),
    data=_data_common,
)

# %%
# --- 図2: mrad のみ ---

fig_mrad, ax_mrad = plt.subplots(figsize=(10, 4))
ax_mrad.plot(time_min, noise_mrad_vals, lw=0.8, color="steelblue", label="measured")
ax_mrad.axhline(noise_mean_mrad, color="red",   ls="--",
                label=f"mean = {noise_mean_mrad:.2f} mrad")
ax_mrad.axhline(sigma_shot_mrad, color="green", ls=":",
                label=f"σ_shot = {sigma_shot_mrad:.2f} mrad  (eq. A.9)")
if sigma_s_adu is not None:
    ax_mrad.axhline(sigma_total_mrad, color="purple", ls="-.",
                    label=f"σ_total = {sigma_total_mrad:.2f} mrad  (eq. A.14)")
ax_mrad.set_ylabel("Phase noise [mrad/frame]")
ax_mrad.set_xlabel(
    f"Time [min]  (interval={FRAME_INTERVAL_S}s/frame, "
    f"N={n_pairs}, pair {PAIR_START_1BASED}-{PAIR_END_1BASED} / {n_pairs_total})"
)
ax_mrad.set_title(
    f"Phase reconstruction temporal noise [mrad]  |  "
    f"measured/σ_shot={ratio_shot:.2f}x"
)
ax_mrad.legend(fontsize=8)
ax_mrad.grid(True, alpha=0.4)
fig_mrad.tight_layout()

save_figure(
    fig_mrad,
    params={**_params_common, "panel": "mrad_only"},
    description=(
        f"位相再構成ノイズ（mrad only）: "
        f"measured={noise_mean_mrad:.2f} mrad, σ_shot={sigma_shot_mrad:.2f} mrad, "
        f"measured/shot={ratio_shot:.2f}x"
    ),
    data=_data_common,
)

# %%
# --- 図3: nm OPD のみ ---

fig_nm, ax_nm = plt.subplots(figsize=(10, 4))
ax_nm.plot(time_min, noise_nm_vals, lw=0.8, color="darkorange", label="measured")
ax_nm.axhline(noise_mean_nm, color="red",   ls="--",
              label=f"mean = {noise_mean_nm:.3f} nm")
ax_nm.axhline(sigma_shot_nm, color="green", ls=":",
              label=f"σ_shot = {sigma_shot_nm:.4f} nm  (eq. A.9)")
if sigma_s_adu is not None:
    ax_nm.axhline(sigma_total_nm, color="purple", ls="-.",
                  label=f"σ_total = {sigma_total_nm:.4f} nm  (eq. A.14)")
ax_nm.set_ylabel("OPD noise [nm/frame]")
ax_nm.set_xlabel(
    f"Time [min]  (interval={FRAME_INTERVAL_S}s/frame, "
    f"N={n_pairs}, pair {PAIR_START_1BASED}-{PAIR_END_1BASED} / {n_pairs_total})"
)
ax_nm.set_title(
    f"Phase reconstruction temporal noise [nm OPD]  |  "
    f"measured/σ_shot={ratio_shot:.2f}x"
)
ax_nm.legend(fontsize=8)
ax_nm.grid(True, alpha=0.4)
fig_nm.tight_layout()

save_figure(
    fig_nm,
    params={**_params_common, "panel": "nm_only"},
    description=(
        f"位相再構成ノイズ（nm OPD only）: "
        f"measured={noise_mean_nm:.3f} nm, σ_shot={sigma_shot_nm:.4f} nm, "
        f"measured/shot={ratio_shot:.2f}x"
    ),
    data=_data_common,
)

print(f"\n完了: {n_pairs} ペア測定 (pair {PAIR_START_1BASED}-{PAIR_END_1BASED})")

# %%
