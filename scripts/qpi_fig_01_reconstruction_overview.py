# %%
"""
qpi_fig_01_reconstruction_overview.py

vistest_5 の単一フレームを用いたQPI再構成概要図生成スクリプト。

出力 (5枚):
  1. 元画像（グレースケール）
  2. FFT（ログスケール、グレースケール）
  3. FFT + フィルタ円（実線:r, 破線:2r, 点線:DC）
  4. オフ軸ピーク切り出し・中心化（円外をゼロ埋め）
  5. 位相再構成（グレースケール）
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage.restoration import unwrap_phase

from qpi import QPIParameters, get_field, make_disk
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE, CROP_REGION
from figure_logger import save_figure

# ============================================================
# 設定
# ============================================================
PATH = r"E:\Acuisition\kitagishi\260301\movetest_3\Pos2\img_000000000_Default_000.tif"

# ============================================================
# 画像読み込み・クロップ
# ============================================================
img_raw = tifffile.imread(PATH)
r0, r1, c0, c1 = CROP_REGION
img = img_raw[r0:r1, c0:c1].astype(float)

params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=img.shape,
    pixelsize=PIXELSIZE,
    offaxis_center=OFFAXIS_CENTER,
)
radius = params.aperturesize // 2
center_row, center_col = params.img_center  # FFT中心 (row, col)

oa_row, oa_col = OFFAXIS_CENTER
sym_row = 2 * center_row - oa_row  # 点対称位置
sym_col = 2 * center_col - oa_col

# FFT
img_fft = np.fft.fftshift(np.fft.fft2(img))
fft_log = np.log(np.abs(img_fft) + 1)

# ============================================================
# 図1: 元画像
# ============================================================
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.imshow(img, cmap="gray")
ax1.axis("off")
save_figure(fig1, params={"crop": str(CROP_REGION)}, description="元画像（グレースケール）")
plt.close(fig1)

# ============================================================
# 図2: FFT（ログスケール）
# ============================================================
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.imshow(fft_log, cmap="gray")
ax2.axis("off")
save_figure(fig2, params={"offaxis_center": str(OFFAXIS_CENTER)},
            description="FFT（ログスケール、グレースケール）")
plt.close(fig2)

# ============================================================
# 図3: FFT + フィルタ円（赤・白の2枚）
# 3つの円: +1次(r), DC中心(2r), -1次(r) — すべて実線
# matplotlib circle center は (x, y) = (col, row)
# ============================================================
def _make_circle_patches(color):
    return [
        plt.Circle((oa_col, oa_row),       radius,     color=color, fill=False, lw=1.5),  # +1次
        plt.Circle((center_col, center_row), radius*2, color=color, fill=False, lw=1.5),  # DC (2r)
        plt.Circle((sym_col, sym_row),     radius,     color=color, fill=False, lw=1.5),  # -1次
    ]

for color, label in [("red", "赤"), ("white", "白")]:
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.imshow(fft_log, cmap="gray")
    for p in _make_circle_patches(color):
        ax3.add_patch(p)
    ax3.axis("off")
    save_figure(fig3,
                params={"offaxis_center": str(OFFAXIS_CENTER), "radius": radius, "circle_color": color},
                description=f"FFT + フィルタ円（{label}）: +1次(r), DC(2r), -1次(r)")
    plt.close(fig3)

# ============================================================
# 図4: オフ軸ピーク切り出し・中心化（円外ゼロ埋め）
# ============================================================
mask = make_disk(OFFAXIS_CENTER, radius, img.shape)
fft_masked = img_fft * mask

shift_row = center_row - oa_row
shift_col = center_col - oa_col
fft_shifted = np.roll(fft_masked, (shift_row, shift_col), axis=(0, 1))

center_mask = make_disk(params.img_center, radius, img.shape)
fft_centered = fft_shifted * center_mask

fig4, ax4 = plt.subplots(figsize=(6, 6))
ax4.imshow(np.log(np.abs(fft_centered) + 1), cmap="gray")
ax4.axis("off")
save_figure(fig4,
            params={"offaxis_center": str(OFFAXIS_CENTER), "radius": radius},
            description="オフ軸ピーク切り出し・中心化（円外ゼロ埋め）")
plt.close(fig4)

# ============================================================
# 図5: 位相再構成
# ============================================================
field = get_field(img, params)
phase = unwrap_phase(np.angle(field))

fig5, ax5 = plt.subplots(figsize=(6, 6))
ax5.imshow(phase, cmap="gray")
ax5.axis("off")
save_figure(fig5,
            params={"offaxis_center": str(OFFAXIS_CENTER)},
            description="位相再構成（グレースケール）")
plt.close(fig5)

print("完了: 5枚保存しました")
