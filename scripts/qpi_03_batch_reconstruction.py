# %%
"""
バッチQPI再構成スクリプト

複数ポジションやタイムラプス画像のQPI再構成を一括処理
"""

import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm
import matplotlib.pyplot as plt
from qpi_common import create_qpi_params, WAVELENGTH, NA, PIXELSIZE

# %%
# ==================== パラメータ設定 ====================

# 定数設定
OFFAXIS_CENTER = (1623, 1621)
CROP = [8, 2056, 416, 2464]  # [y_start, y_end, x_start, x_end]
MEAN_REGION = [1, 507, 254, 507]  # 平均0調整の領域

# ディレクトリ設定（使用時に変更）
BASE_DIR = "/Volumes/QPI3/251017"
BG_DIR = os.path.join(BASE_DIR, "Pos0")  # 背景画像があるディレクトリ
TARGET_DIR = os.path.join(BASE_DIR, "Pos1")  # 処理対象ディレクトリ

# ファイル名パターン
FILE_PATTERN = "*.tif"  # または特定のパターン

print("=" * 80)
print("バッチQPI再構成")
print("=" * 80)
print(f"\nBASE_DIR: {BASE_DIR}")
print(f"BG_DIR: {BG_DIR}")
print(f"TARGET_DIR: {TARGET_DIR}")

# %%
# ==================== セル1: 単一ディレクトリの処理（タイムラプス） ====================

print("\n" + "=" * 80)
print("単一ディレクトリのタイムラプス処理")
print("=" * 80)

# 出力ディレクトリ
OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_2, exist_ok=True)

# BG画像を読み込み
bg_files = sorted([f for f in os.listdir(BG_DIR) if f.endswith('.tif')])
if not bg_files:
    raise FileNotFoundError(f"BG画像が見つかりません: {BG_DIR}")

bg_file = bg_files[0]  # 最初のファイルを使用
bg_path = os.path.join(BG_DIR, bg_file)

print(f"\nBG画像: {bg_path}")

bg_img = np.array(Image.open(bg_path))
y_s, y_e, x_s, x_e = CROP
bg_img = bg_img[y_s:y_e, x_s:x_e]

print(f"BG画像サイズ: {bg_img.shape}")

# QPIパラメータ設定
params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=bg_img.shape,
    pixelsize=PIXELSIZE,
    offaxis_center=OFFAXIS_CENTER
)

field_bg = get_field(bg_img, params)
angle_bg = unwrap_phase(np.angle(field_bg))

print("BG位相画像の取得完了")

# ターゲットディレクトリの処理
target_files = sorted([f for f in os.listdir(TARGET_DIR) 
                      if f.endswith('.tif') and 'output' not in f])

print(f"\n処理対象ファイル数: {len(target_files)}")

for filename in tqdm(target_files, desc="Processing"):
    filepath = os.path.join(TARGET_DIR, filename)
    
    try:
        # 画像読み込みとクロップ
        img = np.array(Image.open(filepath))
        img = img[y_s:y_e, x_s:x_e]
        
        # QPI再構成
        field = get_field(img, params)
        angle = unwrap_phase(np.angle(field))
        
        # 背景差分と平均0調整
        angle_nobg = angle - angle_bg
        y1, y2, x1, x2 = MEAN_REGION
        angle_nobg -= np.mean(angle_nobg[y1:y2, x1:x2])
        
        # TIF保存
        outpath = os.path.join(OUTPUT_DIR, filename.replace(".tif", "_phase.tif"))
        tifffile.imwrite(outpath, angle_nobg.astype(np.float32))
        
        # PNG保存（カラーマップ付き）
        plt.figure(figsize=(6, 6))
        plt.imshow(angle_nobg, cmap='viridis', vmin=-4, vmax=2)
        plt.colorbar(label='Phase (rad)')
        plt.title(f"Phase: {filename}")
        plt.axis('off')
        plt.tight_layout()
        png_outpath = os.path.join(OUTPUT_DIR_2, filename.replace(".tif", "_colormap.png"))
        plt.savefig(png_outpath, dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"\n❌ エラー: {filename} - {e}")
        continue

print(f"\n✅ 処理完了")
print(f"  - 位相画像: {OUTPUT_DIR}")
print(f"  - カラーマップ: {OUTPUT_DIR_2}")

# %%
# ==================== セル2: 複数ポジションの一括処理 ====================

print("\n" + "=" * 80)
print("複数ポジションの一括処理")
print("=" * 80)

# ポジション範囲設定
POS_START = 1
POS_END = 10
FILE_NAME = "img_000000000_Default_000.tif"  # 特定のファイル名

# BG画像読み込み（Pos0）
bg_path = os.path.join(BASE_DIR, "Pos0", FILE_NAME)
print(f"\nBG画像: {bg_path}")

if not os.path.exists(bg_path):
    raise FileNotFoundError(f"BG画像が見つかりません: {bg_path}")

bg_img = np.array(Image.open(bg_path))
bg_img = bg_img[y_s:y_e, x_s:x_e]

# QPIパラメータ設定
params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=bg_img.shape,
    pixelsize=PIXELSIZE,
    offaxis_center=OFFAXIS_CENTER
)

field_bg = get_field(bg_img, params)
angle_bg = unwrap_phase(np.angle(field_bg))

print(f"ポジション範囲: Pos{POS_START} ~ Pos{POS_END}")

# 各ポジションを処理
for pos_idx in tqdm(range(POS_START, POS_END + 1), desc="Processing positions"):
    pos_name = f"Pos{pos_idx}"
    pos_dir = os.path.join(BASE_DIR, pos_name)
    
    if not os.path.exists(pos_dir):
        print(f"\n⚠️  {pos_name} が存在しません。スキップします。")
        continue
    
    # 出力ディレクトリ
    output_dir = os.path.join(pos_dir, "output_phase")
    output_dir_2 = os.path.join(pos_dir, "output_colormap")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_2, exist_ok=True)
    
    # ファイルパス
    filepath = os.path.join(pos_dir, FILE_NAME)
    
    if not os.path.exists(filepath):
        print(f"\n⚠️  {pos_name}/{FILE_NAME} が見つかりません。")
        continue
    
    try:
        # 画像読み込みとクロップ
        img = np.array(Image.open(filepath))
        img = img[y_s:y_e, x_s:x_e]
        
        # QPI再構成
        field = get_field(img, params)
        angle = unwrap_phase(np.angle(field))
        
        # 背景差分と平均0調整
        angle_nobg = angle - angle_bg
        y1, y2, x1, x2 = MEAN_REGION
        angle_nobg -= np.mean(angle_nobg[y1:y2, x1:x2])
        
        # TIF保存
        outpath = os.path.join(output_dir, FILE_NAME.replace(".tif", "_phase.tif"))
        tifffile.imwrite(outpath, angle_nobg.astype(np.float32))
        
        # PNG保存（カラーマップ付き）
        plt.figure(figsize=(6, 6))
        plt.imshow(angle_nobg, cmap='viridis', vmin=-4, vmax=2)
        plt.colorbar(label='Phase (rad)')
        plt.title(f"Phase: {pos_name}")
        plt.axis('off')
        plt.tight_layout()
        png_outpath = os.path.join(output_dir_2, FILE_NAME.replace(".tif", "_colormap.png"))
        plt.savefig(png_outpath, dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"\n❌ {pos_name}: エラー - {e}")
        continue

print(f"\n✅ 全ポジション処理完了")

# %%
# ==================== セル3: タイムラプス + 複数ポジション ====================

print("\n" + "=" * 80)
print("タイムラプス + 複数ポジション処理")
print("=" * 80)

# BG画像読み込み（各ポジションで対応するBG画像を使用）
bg_pos_dir = os.path.join(BASE_DIR, "Pos0")

for pos_idx in tqdm(range(POS_START, POS_END + 1), desc="Processing positions"):
    pos_name = f"Pos{pos_idx}"
    pos_dir = os.path.join(BASE_DIR, pos_name)
    
    if not os.path.exists(pos_dir):
        continue
    
    # 出力ディレクトリ
    output_dir = os.path.join(pos_dir, "output_phase")
    output_dir_2 = os.path.join(pos_dir, "output_colormap")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_2, exist_ok=True)
    
    # ターゲットファイルリスト
    target_files = sorted([f for f in os.listdir(pos_dir) 
                          if f.endswith('.tif') and 'output' not in f])
    
    if not target_files:
        continue
    
    for filename in target_files:
        filepath = os.path.join(pos_dir, filename)
        bg_filepath = os.path.join(bg_pos_dir, filename)
        
        if not os.path.exists(bg_filepath):
            continue
        
        try:
            # BG画像読み込み
            bg_img = np.array(Image.open(bg_filepath))
            bg_img = bg_img[y_s:y_e, x_s:x_e]
            
            # QPIパラメータ設定
            params = QPIParameters(
                wavelength=WAVELENGTH,
                NA=NA,
                img_shape=bg_img.shape,
                pixelsize=PIXELSIZE,
                offaxis_center=OFFAXIS_CENTER
            )
            
            field_bg = get_field(bg_img, params)
            angle_bg = unwrap_phase(np.angle(field_bg))
            
            # ターゲット画像読み込み
            img = np.array(Image.open(filepath))
            img = img[y_s:y_e, x_s:x_e]
            
            # QPI再構成
            field = get_field(img, params)
            angle = unwrap_phase(np.angle(field))
            
            # 背景差分と平均0調整
            angle_nobg = angle - angle_bg
            y1, y2, x1, x2 = MEAN_REGION
            angle_nobg -= np.mean(angle_nobg[y1:y2, x1:x2])
            
            # TIF保存
            outpath = os.path.join(output_dir, filename.replace(".tif", "_phase.tif"))
            tifffile.imwrite(outpath, angle_nobg.astype(np.float32))
            
            # PNG保存
            plt.figure(figsize=(6, 6))
            plt.imshow(angle_nobg, cmap='viridis', vmin=-4, vmax=2)
            plt.colorbar(label='Phase (rad)')
            plt.title(f"{pos_name}: {filename}")
            plt.axis('off')
            plt.tight_layout()
            png_outpath = os.path.join(output_dir_2, filename.replace(".tif", "_colormap.png"))
            plt.savefig(png_outpath, dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"\n❌ {pos_name}/{filename}: エラー - {e}")
            continue

print(f"\n✅ 全処理完了")

# %%
# ==================== 完了 ====================

print("\n" + "=" * 80)
print("バッチ処理スクリプト完了")
print("=" * 80)

print("\n【使い方】")
print("  1. パラメータ設定セルで、ディレクトリとパラメータを設定")
print("  2. 必要なセルを実行:")
print("     - セル1: 単一ディレクトリのタイムラプス処理")
print("     - セル2: 複数ポジションの特定ファイル処理")
print("     - セル3: 複数ポジション + タイムラプス")

print("\n【次のステップ】")
print("  - 位相画像の確認")
print("  - アライメント処理 (qpi_04_alignment_diff.py)")
print("  - 焦点解析 (qpi_05_focus_analysis.py)")

# %%

