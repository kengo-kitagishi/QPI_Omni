# %%
"""
単一画像のQPI再構成と確認スクリプト

1枚の画像とバックグラウンドからQPI再構成を行い、結果を詳細に確認
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from qpi_common import create_qpi_params, WAVELENGTH, NA, PIXELSIZE

# %%
# ==================== パラメータ設定 ====================

# 画像パス（使用時に変更）
path = "/Volumes/QPI3/250910_kk/ph_1/Pos1/img_000000000_Default_000.tif"
path_bg = "/Volumes/QPI3/250910_kk/ph_1/Pos1/img_000000000_Default_000.tif"

# クロップ領域
crop_slice = np.s_[8:2056, 208:2256]

# QPIパラメータ
offaxis_center = (1504, 1708)

# 保存設定
SAVE_OUTPUT = False  # Trueにすると結果を保存
output_path = "/Users/kitak/QPI/output/angle_nobg.tif"

print("=" * 80)
print("単一画像のQPI再構成")
print("=" * 80)
print(f"\nサンプル画像: {path}")
print(f"BG画像: {path_bg}")

# %%
# ==================== 画像読み込み ====================

print("\n画像を読み込み中...")

img = Image.open(path)
img = np.array(img)
img = img[crop_slice]

img_bg = Image.open(path_bg)
img_bg = np.array(img_bg)
img_bg = img_bg[crop_slice]

print(f"画像サイズ: {img.shape}")

# 画像表示
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Sample Image')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(img_bg, cmap='gray')
plt.title('Background Image')
plt.colorbar()
plt.tight_layout()
plt.show()

# %%
# ==================== QPIパラメータ設定 ====================

params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=img.shape,
    pixelsize=PIXELSIZE,
    offaxis_center=offaxis_center,
)

print("\nQPIパラメータ:")
print(f"  - 波長: {WAVELENGTH*1e9:.1f} nm")
print(f"  - NA: {NA}")
print(f"  - ピクセルサイズ: {PIXELSIZE*1e6:.3f} µm")
print(f"  - オフ軸中心: {offaxis_center}")
print(f"  - 開口サイズ: {params.aperturesize} pixels")

# %%
# ==================== QPI再構成 ====================

print("\nQPI再構成を実行中...")

# 複素場の取得
field = get_field(img, params)
field_bg = get_field(img_bg, params)

# 位相アンラッピング
angle = unwrap_phase(np.angle(field))
angle_bg = unwrap_phase(np.angle(field_bg))

print("再構成完了")

# %%
# ==================== 位相画像の表示 ====================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].imshow(angle, cmap='viridis')
axes[0].set_title('Sample Phase')
plt.colorbar(im1, ax=axes[0], label='Phase (rad)')

im2 = axes[1].imshow(angle_bg, cmap='viridis')
axes[1].set_title('Background Phase')
plt.colorbar(im2, ax=axes[1], label='Phase (rad)')

plt.tight_layout()
plt.show()

# %%
# ==================== 背景差分 ====================

print("\n背景差分を計算...")

angle_nobg = angle - angle_bg

# 平均0調整（背景領域を指定）
bg_region = np.s_[1:100, 1:254]
mean_value = np.mean(angle_nobg[bg_region])
angle_nobg = angle_nobg - mean_value

print(f"背景領域の平均値: {mean_value:.6f} rad")
print(f"調整後の平均値: {np.mean(angle_nobg):.6e} rad")

# 背景差分後の位相画像
plt.figure(figsize=(10, 8))
plt.imshow(angle_nobg, vmin=-0.1, vmax=0.1, cmap='viridis')
plt.colorbar(label='Phase (rad)')
plt.title('Background-subtracted Phase Map')
plt.tight_layout()
plt.show()

# %%
# ==================== 統計情報 ====================

print("\n" + "=" * 80)
print("位相画像の統計情報")
print("=" * 80)

print(f"\n【全体】")
print(f"  平均: {np.mean(angle_nobg):.6e} rad")
print(f"  標準偏差: {np.std(angle_nobg):.6f} rad")
print(f"  最小値: {np.min(angle_nobg):.6f} rad")
print(f"  最大値: {np.max(angle_nobg):.6f} rad")

# ヒストグラム
plt.figure(figsize=(10, 5))
plt.hist(angle_nobg.flatten(), bins=100, alpha=0.7, edgecolor='black')
plt.xlabel('Phase (rad)')
plt.ylabel('Frequency')
plt.title('Phase Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# ==================== 位相プロファイル ====================

print("\n位相プロファイルを表示...")

# 水平方向のプロファイル（画像中央）
y_coord = img.shape[0] // 2
profile_horizontal = angle_nobg[y_coord, :]

# 垂直方向のプロファイル（画像中央）
x_coord = img.shape[1] // 2
profile_vertical = angle_nobg[:, x_coord]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 水平プロファイル
axes[0].plot(profile_horizontal, linewidth=2)
axes[0].set_title(f'Horizontal Profile (y = {y_coord})')
axes[0].set_xlabel('X coordinate (pixels)')
axes[0].set_ylabel('Phase (rad)')
axes[0].grid(True, alpha=0.3)

# 垂直プロファイル
axes[1].plot(profile_vertical, linewidth=2)
axes[1].set_title(f'Vertical Profile (x = {x_coord})')
axes[1].set_xlabel('Y coordinate (pixels)')
axes[1].set_ylabel('Phase (rad)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# ==================== カラーマップ比較 ====================

print("\n様々なカラーマップで表示...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
cmaps = ['viridis', 'plasma', 'inferno', 'RdBu_r', 'seismic', 'coolwarm']

for idx, cmap in enumerate(cmaps):
    ax = axes[idx // 3, idx % 3]
    im = ax.imshow(angle_nobg, vmin=-0.1, vmax=0.1, cmap=cmap)
    ax.set_title(f'Colormap: {cmap}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.show()

# %%
# ==================== 振幅画像（オプション） ====================

print("\n振幅画像を表示...")

amplitude = np.abs(field)
amplitude_bg = np.abs(field_bg)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].imshow(amplitude, cmap='gray')
axes[0].set_title('Sample Amplitude')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(amplitude_bg, cmap='gray')
axes[1].set_title('Background Amplitude')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# %%
# ==================== 保存（オプション） ====================

if SAVE_OUTPUT:
    print(f"\n結果を保存: {output_path}")
    tifffile.imwrite(output_path, angle_nobg.astype(np.float32))
    
    # カラーマップ付きPNGも保存
    png_path = output_path.replace('.tif', '_colormap.png')
    plt.figure(figsize=(10, 8))
    plt.imshow(angle_nobg, cmap='viridis', vmin=-0.1, vmax=0.1)
    plt.colorbar(label='Phase (rad)')
    plt.title('Background-subtracted Phase Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  - TIF: {output_path}")
    print(f"  - PNG: {png_path}")
else:
    print("\n※結果を保存する場合は、SAVE_OUTPUT = True に設定してください")

# %%
# ==================== 完了 ====================

print("\n" + "=" * 80)
print("QPI再構成完了")
print("=" * 80)

print("\n【確認項目】")
print("  ✓ 位相画像が適切に再構成されているか")
print("  ✓ 背景差分が正しく行われているか")
print("  ✓ プロファイルが滑らかか")
print("  ✓ 異常値やアーティファクトがないか")

print("\n【次のステップ】")
print("  - 問題なければ → バッチ処理へ (qpi_03_batch_reconstruction.py)")
print("  - 結果が不適切 → 焦点調整へ (qpi_01_focus_setup.py)")

print("\n✅ 単一画像処理スクリプト完了")

# %%

