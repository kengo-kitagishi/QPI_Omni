#!/usr/bin/env python3
"""
超シンプルなmean RI計算と図の出力

ellipse理論体積でtotal phaseを割ってmean RIを求める
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

# =============================================================================
# パラメータ設定
# =============================================================================
RESULTS_CSV = r"C:\Users\QPI\Documents\QPI_omni\data\align_demo\from_output\Results.csv"
IMAGE_DIR = r"C:\Users\QPI\Documents\QPI_omni\data\align_demo\from_outputphase\bg_corr\subtracted"
OUTPUT_FILE = "simple_mean_ri.png"

PIXEL_SIZE_UM = 0.348
WAVELENGTH_NM = 663
N_MEDIUM = 1.333
ALPHA_RI = 0.00018

# =============================================================================
# 関数定義
# =============================================================================

def calc_rod_volume(major_px, minor_px, pixel_size_um):
    """Rod shape体積計算 (カプセル型)"""
    length_um = major_px * pixel_size_um
    width_um = minor_px * pixel_size_um
    r = width_um / 2.0
    h = length_um - 2 * r
    
    if h < 0:
        # 球
        return (4.0 / 3.0) * np.pi * (r ** 3)
    else:
        # 球 + 円柱
        return (4.0 / 3.0) * np.pi * (r ** 3) + np.pi * (r ** 2) * h

def make_ellipse_mask(center_x, center_y, major, minor, angle_deg, image_shape):
    """楕円マスク作成"""
    height, width = image_shape
    y, x = np.ogrid[:height, :width]
    
    angle_rad = np.deg2rad(angle_deg)
    dx = x - center_x
    dy = y - center_y
    
    x_rot = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
    y_rot = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    
    a = major / 2.0
    b = minor / 2.0
    
    return ((x_rot / a) ** 2 + (y_rot / b) ** 2) <= 1.0

# =============================================================================
# メイン処理
# =============================================================================

print("="*60)
print("Simple Ellipse RI Analysis")
print("="*60)

# Results.csv読み込み
print(f"\nReading: {os.path.basename(RESULTS_CSV)}")
df = pd.read_csv(RESULTS_CSV)
print(f"  {len(df)} ROIs found")

# 画像リスト取得
image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.tif')))
print(f"  {len(image_files)} images found")

# 最初の画像でサイズ取得
first_img = Image.open(image_files[0])
image_shape = (first_img.height, first_img.width)
print(f"  Image size: {image_shape}")

# 結果格納
results = []

# 各ROIを処理
print("\nProcessing...")
for idx, row in df.iterrows():
    # フレーム番号
    if 'Label' in row:
        import re
        match = re.search(r'(\d{4})', row['Label'])
        frame_num = int(match.group(1)) if match else idx + 1
    elif 'Slice' in row:
        frame_num = int(row['Slice'])
    else:
        frame_num = idx + 1
    
    if frame_num > len(image_files):
        continue
    
    # 位相画像読み込み
    phase_img = np.array(Image.open(image_files[frame_num - 1])).astype(np.float64)
    
    # ROIパラメータ
    major = row.get('Major', row.get('Feret', 0))
    minor = row.get('Minor', row.get('MinFeret', 0))
    center_x = row['X']
    center_y = row['Y']
    angle = row.get('Angle', row.get('FeretAngle', 0))
    
    if major == 0 or minor == 0:
        continue
    
    # 体積計算
    volume_um3 = calc_rod_volume(major, minor, PIXEL_SIZE_UM)
    
    # マスク作成
    mask = make_ellipse_mask(center_x, center_y, major, minor, angle, image_shape)
    
    # total phase
    total_phase = np.sum(phase_img[mask])
    
    # mean RI計算
    wavelength_um = WAVELENGTH_NM * 1e-3
    pixel_area_um2 = PIXEL_SIZE_UM ** 2
    mean_ri = N_MEDIUM + (total_phase * wavelength_um * pixel_area_um2) / (2 * np.pi * volume_um3)
    
    # 質量計算
    mean_conc = (mean_ri - N_MEDIUM) / ALPHA_RI
    total_mass = mean_conc * volume_um3
    
    results.append({
        'frame': frame_num,
        'volume': volume_um3,
        'mean_ri': mean_ri,
        'mass': total_mass
    })
    
    if (idx + 1) % 20 == 0:
        print(f"  {idx + 1}/{len(df)}")

# DataFrame化
results_df = pd.DataFrame(results).sort_values('frame')

print(f"\nProcessed {len(results_df)} ROIs")
print(f"  Volume range: {results_df['volume'].min():.1f} - {results_df['volume'].max():.1f} µm³")
print(f"  Mean RI range: {results_df['mean_ri'].min():.4f} - {results_df['mean_ri'].max():.4f}")
print(f"  Mass range: {results_df['mass'].min():.1f} - {results_df['mass'].max():.1f} pg")

# =============================================================================
# プロット作成
# =============================================================================

print(f"\nCreating plot...")

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle('Simple Ellipse RI Analysis', fontsize=14, fontweight='bold')

# Volume
axes[0].plot(results_df['frame'], results_df['volume'], 'o-', color='#1f77b4', linewidth=2, markersize=5)
axes[0].set_ylabel('Volume (µm³)', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Mean RI
axes[1].plot(results_df['frame'], results_df['mean_ri'], 's-', color='#ff7f0e', linewidth=2, markersize=5)
axes[1].set_ylabel('Mean RI', fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Mass
axes[2].plot(results_df['frame'], results_df['mass'], '^-', color='#2ca02c', linewidth=2, markersize=5)
axes[2].set_xlabel('Frame', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Total Mass (pg)', fontsize=11, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_FILE}")

plt.show()

print("\nDone!")



