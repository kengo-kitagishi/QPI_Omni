# %%
"""
with-wo焦点解析専用スクリプト（高速版）

- Crop領域の可視化
- Pos0をBGとして使用
- Gaussian背景引き
- with/wo画像のQPI再構成
- ECCアライメント
- 設定ファイル (focus_analysis_config.yaml) で管理
"""

import os
import yaml
import numpy as np
import tifffile
import cv2
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm
import matplotlib.pyplot as plt
from qpi_common import (create_qpi_params, to_uint8, visualize_crop_region,
                        gaussian_background_subtraction)

# %%
# ==================== 設定ファイル読み込み ====================

# 設定ファイルのパス（必要に応じて変更）
CONFIG_PATH = "/Users/kitak/QPI_Omni/scripts/focus_analysis_config.yaml"

print("=" * 80)
print("with-wo焦点解析スクリプト (Gaussian背景引き + ECC対応)")
print("=" * 80)
print(f"\n設定ファイルを読み込み: {CONFIG_PATH}")

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 設定を展開
BASE_DIR = config['base_dir']
WITH_DIR = config['with_dir']
WO_DIR = config['wo_dir']
BG_POS = config['bg_pos']
POS_START = config['pos_start']
POS_END = config['pos_end']
FILE_PATTERN = config.get('file_pattern', 'img_000000000_Default_000.tif')

# Crop設定
crop_config = config['crop']
CROP_Y_START = crop_config['y_start']
CROP_Y_END = crop_config['y_end']
CROP_X_START = crop_config['x_start']
CROP_X_END = crop_config['x_end']
CROP_COORDS = (CROP_Y_START, CROP_Y_END, CROP_X_START, CROP_X_END)

# QPI パラメータ
qpi_config = config['qpi']
WAVELENGTH = qpi_config['wavelength']
NA = qpi_config['NA']
PIXELSIZE = qpi_config['pixelsize']
OFFAXIS_CENTER = tuple(qpi_config['offaxis_center'])

# Gaussian背景引き設定
gauss_config = config['gaussian_backsub']
GAUSS_ENABLED = gauss_config['enabled']
HIST_MIN = gauss_config['hist_min']
HIST_MAX = gauss_config['hist_max']
N_BINS = gauss_config['n_bins']
SMOOTH_WINDOW = gauss_config['smooth_window']
MIN_PHASE = gauss_config['min_phase']

# アライメント設定
align_config = config['alignment']
WARP_MODE_STR = align_config.get('warp_mode', 'AFFINE')
WARP_MODE = getattr(cv2.MOTION_, WARP_MODE_STR, cv2.MOTION_AFFINE)
ECC_ITERATIONS = align_config.get('iterations', 5000)
ECC_EPS = align_config.get('eps', 1e-5)

# 出力設定
output_config = config['output']
DIFF_VMIN, DIFF_VMAX = output_config['diff_range']
COLORMAP = output_config.get('colormap', 'JET')
SAVE_ALIGNED = output_config.get('save_aligned', True)
SAVE_INDIVIDUAL_PHASE = output_config.get('save_individual_phase', False)
SAVE_CROP_VIZ = output_config.get('save_crop_visualization', True)

print("\n【設定内容】")
print(f"  BASE_DIR: {BASE_DIR}")
print(f"  WITH_DIR: {WITH_DIR}")
print(f"  WO_DIR: {WO_DIR}")
print(f"  BG_POS: Pos{BG_POS}")
print(f"  ポジション範囲: Pos{POS_START} ~ Pos{POS_END}")
print(f"  ファイル名: {FILE_PATTERN}")
print(f"  Crop領域: ({CROP_Y_START}:{CROP_Y_END}, {CROP_X_START}:{CROP_X_END})")
print(f"  Gaussian背景引き: {'有効' if GAUSS_ENABLED else '無効'}")
print(f"  アライメント: {WARP_MODE_STR} (iterations={ECC_ITERATIONS})")
print(f"  差分表示範囲: [{DIFF_VMIN}, {DIFF_VMAX}]")

# %%
# ==================== Crop領域の可視化 ====================

print("\n" + "=" * 80)
print("Crop領域の可視化")
print("=" * 80)

# サンプル画像を読み込んで可視化
sample_path = os.path.join(BASE_DIR, WITH_DIR, f"Pos{POS_START}", FILE_PATTERN)
if not os.path.exists(sample_path):
    print(f"⚠️ サンプル画像が見つかりません: {sample_path}")
    print("   Pos0で代用します...")
    sample_path = os.path.join(BASE_DIR, WITH_DIR, f"Pos{BG_POS}", FILE_PATTERN)

sample_img = np.array(Image.open(sample_path))
print(f"\n元画像サイズ: {sample_img.shape}")
print(f"Crop後サイズ: ({CROP_Y_END - CROP_Y_START}, {CROP_X_END - CROP_X_START})")

# Crop領域を可視化
fig = visualize_crop_region(sample_img, CROP_COORDS, 
                            title="Crop Region Visualization")
plt.show()

# 保存
output_base = os.path.join(BASE_DIR, WITH_DIR, "focus_analysis_output")
os.makedirs(output_base, exist_ok=True)

if SAVE_CROP_VIZ:
    crop_viz_path = os.path.join(output_base, "crop_region_visualization.png")
    fig.savefig(crop_viz_path, dpi=150, bbox_inches='tight')
    print(f"\nCrop領域の可視化を保存: {crop_viz_path}")
plt.close(fig)

# Crop後の画像も表示
cropped_img = sample_img[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]
plt.figure(figsize=(10, 8))
plt.imshow(cropped_img, cmap='gray')
plt.title('Cropped Image')
plt.colorbar()
plt.tight_layout()
plt.show()

print("✅ Crop領域を確認してください。問題なければ次に進みます。")

# %%
# ==================== Pos0（BG）画像の読み込みとパラメータ設定 ====================

print("\n" + "=" * 80)
print("Pos0（BG）画像の読み込み")
print("=" * 80)

# with側のPos0を読み込み
bg_path_with = os.path.join(BASE_DIR, WITH_DIR, f"Pos{BG_POS}", FILE_PATTERN)
print(f"\nBG画像パス: {bg_path_with}")

if not os.path.exists(bg_path_with):
    raise FileNotFoundError(f"BG画像が見つかりません: {bg_path_with}")

# BG画像読み込みとクロップ
bg_img = np.array(Image.open(bg_path_with))
bg_img = bg_img[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]

print(f"BG画像サイズ: {bg_img.shape}")

# QPIパラメータ設定
params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=bg_img.shape,
    pixelsize=PIXELSIZE,
    offaxis_center=OFFAXIS_CENTER
)

print(f"QPIパラメータ設定完了:")
print(f"  - 波長: {WAVELENGTH*1e9:.1f} nm")
print(f"  - NA: {NA}")
print(f"  - Offaxis center: {OFFAXIS_CENTER}")
print(f"  - Aperture size: {params.aperturesize}")

# BG画像のQPI再構成
field_bg = get_field(bg_img, params)
angle_bg = unwrap_phase(np.angle(field_bg))

print("BG位相画像の取得完了")

# %%
# ==================== 出力ディレクトリの準備 ====================

diff_dir = os.path.join(output_base, "diff_wo_minus_with")
cmap_dir = os.path.join(output_base, "diff_colormap")
os.makedirs(diff_dir, exist_ok=True)
os.makedirs(cmap_dir, exist_ok=True)

if SAVE_ALIGNED:
    aligned_with_dir = os.path.join(output_base, "aligned_with")
    aligned_wo_dir = os.path.join(output_base, "aligned_wo")
    os.makedirs(aligned_with_dir, exist_ok=True)
    os.makedirs(aligned_wo_dir, exist_ok=True)

if SAVE_INDIVIDUAL_PHASE:
    phase_with_dir = os.path.join(output_base, "phase_with")
    phase_wo_dir = os.path.join(output_base, "phase_wo")
    os.makedirs(phase_with_dir, exist_ok=True)
    os.makedirs(phase_wo_dir, exist_ok=True)

print(f"\n出力ディレクトリ: {output_base}")

# %%
# ==================== 全ポジションの処理 ====================

print("\n" + "=" * 80)
print("with-wo画像の処理開始")
print("=" * 80)

success_count = 0
fail_count = 0
failed_positions = []

# カラーマップ設定
if COLORMAP == "JET":
    cmap_cv = cv2.COLORMAP_JET
elif COLORMAP == "viridis":
    cmap_cv = cv2.COLORMAP_VIRIDIS
elif COLORMAP == "hot":
    cmap_cv = cv2.COLORMAP_HOT
else:
    cmap_cv = cv2.COLORMAP_JET

for pos_idx in tqdm(range(POS_START, POS_END + 1), desc="Processing positions"):
    pos_name = f"Pos{pos_idx}"
    
    try:
        # ==================== with画像の処理 ====================
        with_path = os.path.join(BASE_DIR, WITH_DIR, pos_name, FILE_PATTERN)
        
        if not os.path.exists(with_path):
            raise FileNotFoundError(f"with画像が見つかりません: {with_path}")
        
        # with画像読み込みとクロップ
        with_img = np.array(Image.open(with_path))
        with_img = with_img[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]
        
        # QPI再構成
        field_with = get_field(with_img, params)
        angle_with = unwrap_phase(np.angle(field_with))
        
        # 背景差分
        angle_with_nobg = angle_with - angle_bg
        
        # Gaussian背景引き
        if GAUSS_ENABLED:
            angle_with_nobg, corr_with = gaussian_background_subtraction(
                angle_with_nobg,
                hist_min=HIST_MIN,
                hist_max=HIST_MAX,
                n_bins=N_BINS,
                smooth_window=SMOOTH_WINDOW,
                min_phase=MIN_PHASE
            )
        
        # ==================== wo画像の処理 ====================
        wo_path = os.path.join(BASE_DIR, WO_DIR, pos_name, FILE_PATTERN)
        
        if not os.path.exists(wo_path):
            raise FileNotFoundError(f"wo画像が見つかりません: {wo_path}")
        
        # wo画像読み込みとクロップ
        wo_img = np.array(Image.open(wo_path))
        wo_img = wo_img[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]
        
        # QPI再構成
        field_wo = get_field(wo_img, params)
        angle_wo = unwrap_phase(np.angle(field_wo))
        
        # 背景差分
        angle_wo_nobg = angle_wo - angle_bg
        
        # Gaussian背景引き
        if GAUSS_ENABLED:
            angle_wo_nobg, corr_wo = gaussian_background_subtraction(
                angle_wo_nobg,
                hist_min=HIST_MIN,
                hist_max=HIST_MAX,
                n_bins=N_BINS,
                smooth_window=SMOOTH_WINDOW,
                min_phase=MIN_PHASE
            )
        
        # ==================== ECCアライメント（wo → with） ====================
        # uint8に変換
        with_uint8 = to_uint8(angle_with_nobg)
        wo_uint8 = to_uint8(angle_wo_nobg)
        
        # ECC アライメント
        if WARP_MODE == cv2.MOTION_AFFINE:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:  # TRANSLATION or EUCLIDEAN
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                   ECC_ITERATIONS, ECC_EPS)
        
        try:
            _, warp_matrix = cv2.findTransformECC(
                with_uint8, wo_uint8, warp_matrix, WARP_MODE,
                criteria, inputMask=None, gaussFiltSize=1
            )
        except cv2.error as e:
            # ECCが収束しない場合は単位行列のまま（アライメントなし）
            if pos_idx <= POS_START + 5:  # 最初の数個だけ警告表示
                print(f"\n⚠️  {pos_name}: ECC収束せず（アライメントなしで続行）")
        
        # wo画像をアライメント
        h, w = angle_with_nobg.shape
        angle_wo_aligned = cv2.warpAffine(
            angle_wo_nobg, warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        
        # ==================== 差分計算 ====================
        diff = angle_wo_aligned - angle_with_nobg
        
        # ==================== 保存 ====================
        # 差分TIF保存
        diff_path = os.path.join(diff_dir, f"{pos_name}_diff.tif")
        tifffile.imwrite(diff_path, diff.astype(np.float32))
        
        # カラーマップ画像保存
        diff_clipped = np.clip(diff, DIFF_VMIN, DIFF_VMAX)
        diff_norm = ((diff_clipped - DIFF_VMIN) / (DIFF_VMAX - DIFF_VMIN) * 255).astype(np.uint8)
        color_mapped = cv2.applyColorMap(diff_norm, cmap_cv)
        
        cmap_path = os.path.join(cmap_dir, f"{pos_name}_diff_cmap.png")
        cv2.imwrite(cmap_path, color_mapped)
        
        # アライメント済み画像保存（オプション）
        if SAVE_ALIGNED:
            tifffile.imwrite(
                os.path.join(aligned_with_dir, f"{pos_name}_with.tif"),
                angle_with_nobg.astype(np.float32)
            )
            tifffile.imwrite(
                os.path.join(aligned_wo_dir, f"{pos_name}_wo_aligned.tif"),
                angle_wo_aligned.astype(np.float32)
            )
        
        # 個別位相画像保存（オプション）
        if SAVE_INDIVIDUAL_PHASE:
            tifffile.imwrite(
                os.path.join(phase_with_dir, f"{pos_name}_phase.tif"),
                angle_with_nobg.astype(np.float32)
            )
            tifffile.imwrite(
                os.path.join(phase_wo_dir, f"{pos_name}_phase.tif"),
                angle_wo_nobg.astype(np.float32)
            )
        
        success_count += 1
        
    except Exception as e:
        fail_count += 1
        failed_positions.append((pos_name, str(e)))
        print(f"\n❌ {pos_name}: エラー - {e}")
        continue

# %%
# ==================== 結果サマリー ====================

print("\n" + "=" * 80)
print("処理完了")
print("=" * 80)

print(f"\n【処理結果】")
print(f"  成功: {success_count} ポジション")
print(f"  失敗: {fail_count} ポジション")
print(f"  合計: {POS_END - POS_START + 1} ポジション")

if failed_positions:
    print(f"\n【失敗したポジション】")
    for pos_name, error in failed_positions[:10]:  # 最初の10個まで表示
        print(f"  - {pos_name}: {error}")
    if len(failed_positions) > 10:
        print(f"  ... 他 {len(failed_positions) - 10} 件")

print(f"\n【出力ディレクトリ】")
print(f"  - 差分画像 (float32 TIF): {diff_dir}")
print(f"  - カラーマップ (PNG): {cmap_dir}")
if SAVE_ALIGNED:
    print(f"  - アライメント済みwith: {aligned_with_dir}")
    print(f"  - アライメント済みwo: {aligned_wo_dir}")
if SAVE_INDIVIDUAL_PHASE:
    print(f"  - with位相画像: {phase_with_dir}")
    print(f"  - wo位相画像: {phase_wo_dir}")

print(f"\n【設定】")
print(f"  - Crop領域: ({CROP_Y_START}:{CROP_Y_END}, {CROP_X_START}:{CROP_X_END})")
print(f"  - Gaussian背景引き: {'有効' if GAUSS_ENABLED else '無効'}")
print(f"  - 差分表示範囲: [{DIFF_VMIN}, {DIFF_VMAX}]")
print(f"  - カラーマップ: {COLORMAP}")
print(f"  - アライメント: {WARP_MODE_STR}")

print("\n✅ 全処理が完了しました！")

# %%

