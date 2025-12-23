# %%
"""
焦点合わせ・光学系調整スクリプト

顕微鏡撮影前の焦点位置調整、オフ軸中心確認、visibility評価などに使用
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from qpi_common import visibility, create_qpi_params, WAVELENGTH, NA, PIXELSIZE
from CursorVisualizer import CursorVisualizer

# %%
# ==================== パラメータ設定 ====================

# 画像パス（使用時に変更）
path = "/Volumes/QPI3/250910_kk/ph_1/Pos1/img_000000000_Default_000.tif"
path_bg = "/Volumes/QPI3/250910_kk/ph_1/Pos1/img_000000000_Default_000.tif"

# クロップ領域（必要に応じて調整）
crop_slice = np.s_[8:2056, 208:2256]
# 例: crop_slice = np.s_[8:2056, 416:2464]
# 例: crop_slice = np.s_[516:1540, 500:1524]

# オフ軸中心の初期値（FFTで確認後に調整）
offaxis_center = (1504, 1708)

print("=" * 80)
print("焦点合わせ・光学系調整")
print("=" * 80)
print(f"\n画像パス: {path}")
print(f"BG画像パス: {path_bg}")
print(f"クロップ領域: {crop_slice}")

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
# ==================== FFT表示とオフ軸中心確認 ====================

print("\nFFTを計算中...")

img_fft = np.fft.fftshift(np.fft.fft2(img_bg))

plt.figure(figsize=(10, 8))
plt.imshow(np.log(np.abs(img_fft)))
plt.title('FFT (log scale)')
plt.colorbar()
plt.show()

# %%
# ==================== CursorVisualizerで詳細確認 ====================

print("\nCursorVisualizerを起動...")
print("  - マウスをFFT画像上で動かすと座標が表示されます")
print("  - オフ軸ピークの座標を確認してください")

cb = CursorVisualizer(np.log(np.abs(img_fft)))
cb.run()

# CursorVisualizerで確認した座標をここに入力
# offaxis_center = (y座標, x座標)
# 例: offaxis_center = (1504, 1708)

print(f"\n現在のオフ軸中心: {offaxis_center}")
print("  ※必要に応じて上記の値を変更してください")

# %%
# ==================== QPIパラメータ設定と開口確認 ====================

IMG_SHAPE = img.shape

params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=IMG_SHAPE,
    pixelsize=PIXELSIZE,
    offaxis_center=offaxis_center,
)

print("\nQPIパラメータ:")
print(f"  - 波長: {WAVELENGTH*1e9:.1f} nm")
print(f"  - NA: {NA}")
print(f"  - ピクセルサイズ: {PIXELSIZE*1e6:.3f} µm")
print(f"  - オフ軸中心: {offaxis_center}")
print(f"  - 開口サイズ: {params.aperturesize} pixels")

# FFT画像に開口円を重ねて表示
fft_image = np.log(np.abs(img_fft))

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(fft_image)

# 開口円の描画
aperture_size = params.aperturesize
radius = aperture_size // 2
circle_center = (offaxis_center[1], offaxis_center[0])  # (x, y)に変換
circle = plt.Circle(circle_center, radius, color='red', fill=False, linewidth=2)
ax.add_patch(circle)

ax.set_title("FFT with Aperture Circle (red)")
plt.tight_layout()
plt.show()

print("\n✅ 赤い円がオフ軸ピークを適切にカバーしているか確認してください")

# %%
# ==================== 位相画像の再構成 ====================

print("\n位相画像を再構成中...")

field = get_field(img, params)
field_bg = get_field(img_bg, params)

angle = unwrap_phase(np.angle(field))
angle_bg = unwrap_phase(np.angle(field_bg))

# 位相画像表示
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

print("\n背景差分を計算中...")

angle_nobg = angle - angle_bg

# 平均0調整（背景領域を指定）
# 使用時に適切な領域を指定してください
bg_region = np.s_[1:100, 1:254]
angle_nobg = angle_nobg - np.mean(angle_nobg[bg_region])

plt.figure(figsize=(10, 8))
plt.imshow(angle_nobg, vmin=-0.1, vmax=0.1, cmap='viridis')
plt.colorbar(label='Phase (rad)')
plt.title('Background-subtracted Phase Map')
plt.tight_layout()
plt.show()

print(f"位相画像の統計:")
print(f"  - 平均: {np.mean(angle_nobg):.6f} rad")
print(f"  - 標準偏差: {np.std(angle_nobg):.6f} rad")
print(f"  - 最小値: {np.min(angle_nobg):.6f} rad")
print(f"  - 最大値: {np.max(angle_nobg):.6f} rad")

# %%
# ==================== 位相プロファイル確認 ====================

print("\n位相プロファイルを確認...")

# 任意のx座標でプロファイルを取得
x_coord = img.shape[1] // 2  # 中央
profile = angle_nobg[:, x_coord]

plt.figure(figsize=(10, 6))
plt.plot(profile, linewidth=2)
plt.title(f'Phase Profile at x = {x_coord}')
plt.xlabel('Y coordinate (pixels)')
plt.ylabel('Phase (rad)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"  x = {x_coord} での位相プロファイルを表示")
print("  ※焦点が合っていると、プロファイルが滑らかになります")

# %%
# ==================== Visibility（可視性）評価 ====================

print("\nVisibility（可視性）を計算中...")

vis_map = visibility(img, params)

plt.figure(figsize=(10, 8))
plt.imshow(vis_map, vmin=0.3, vmax=0.95, cmap='viridis')
plt.colorbar(label='Visibility')
plt.title('Visibility Map')
plt.tight_layout()
plt.show()

# Visibilityのヒストグラム
plt.figure(figsize=(10, 5))
plt.hist(vis_map.flatten(), bins=100, range=(0.3, 1.0), alpha=0.7, edgecolor='black')
plt.xlabel('Visibility')
plt.ylabel('Frequency')
plt.title('Visibility Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

vis_mean = vis_map.mean()
print(f"\nVisibility統計:")
print(f"  - 平均: {vis_mean:.4f}")
print(f"  - 標準偏差: {vis_map.std():.4f}")
print(f"  - 最小値: {vis_map.min():.4f}")
print(f"  - 最大値: {vis_map.max():.4f}")

if vis_mean > 0.7:
    print("\n✅ Visibilityが良好です（> 0.7）")
elif vis_mean > 0.5:
    print("\n⚠️  Visibilityがやや低めです（0.5-0.7）- 調整を検討してください")
else:
    print("\n❌ Visibilityが低いです（< 0.5）- 光学系の調整が必要です")

# %%
# ==================== 焦点確認サマリー ====================

print("\n" + "=" * 80)
print("焦点確認サマリー")
print("=" * 80)

print("\n【確認項目】")
print(f"  1. オフ軸中心: {offaxis_center}")
print(f"  2. 開口サイズ: {params.aperturesize} pixels")
print(f"  3. Visibility平均: {vis_mean:.4f}")
print(f"  4. 位相画像の標準偏差: {np.std(angle_nobg):.6f} rad")

print("\n【次のステップ】")
print("  - Visibilityが低い場合 → 焦点位置を調整")
print("  - オフ軸ピークが不明瞭 → 光軸を調整")
print("  - 問題なければ → 撮影を開始")

print("\n✅ 焦点調整スクリプト完了")

# %%

