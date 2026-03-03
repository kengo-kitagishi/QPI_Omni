# %%
"""
qpi_fig_02_visibility.py

ホログラムから Visibility を計算する手順を示す6パネル図（修論用）。

パネル構成（2行×3列）:
  上段: a(Hologram) | c(Interferometric Amp) | e(Interferometric OPD)
  下段: b(2D FFT)   | d(Non-interferometric Amp) | f(Visibility = 2β/α)

矢印:
  a → b: FFT (下向き)
  b → c: IFFT (干渉項)
  b → d: IFFT (非干渉項)
  c,d → f: ratio (Visibility)

参照: Park et al., Fig. 7.1 スタイル
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import tifffile
from skimage.restoration import unwrap_phase

sys.path.insert(0, "/Users/kitak/QPI_Omni/scripts")
from qpi import QPIParameters, get_field, make_disk, crop_array
from figure_logger import save_figure

# ============================================================
# 設定 — 実データに合わせて変更
# ============================================================

HOLOGRAM_PATH = (
    "/Volumes/QPI_0_.01_r/251211/sequence shot/"
    "Basler_acA2440-75um__25176370__20251211_152604439_0000.tiff"
)

WAVELENGTH     = 658e-9          # [m]
NA             = 0.95
PIXELSIZE      = 3.45e-6 / 40   # [m/px]
OFFAXIS_CENTER = (1664, 485)     # (row, col) — FFT空間でのサイドバンド位置
CROP           = (8, 2056, 208, 2256)  # (row_start, row_end, col_start, col_end)

# ============================================================
# ホログラム読み込み
# ============================================================

def load_hologram(path, crop):
    r0, r1, c0, c1 = crop
    try:
        img = tifffile.imread(path).astype(np.float64)
    except Exception:
        img = np.array(Image.open(path)).astype(np.float64)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img[r0:r1, c0:c1]


if os.path.exists(HOLOGRAM_PATH):
    holo = load_hologram(HOLOGRAM_PATH, CROP)
    print(f"Loaded hologram: shape={holo.shape}, dtype={holo.dtype}")
else:
    print(f"[WARNING] File not found: {HOLOGRAM_PATH}")
    print("  → プレースホルダー（ランダムフリンジ）を使用します")
    rng = np.random.default_rng(42)
    H, W = 2048, 2048
    yy, xx = np.mgrid[:H, :W]
    kx, ky = 0.05, 0.12   # fringe spatial frequency [rad/px]
    holo = (
        1000
        + 300 * np.cos(kx * xx + ky * yy)
        + 50 * rng.standard_normal((H, W))
    )
    holo = np.clip(holo, 0, 4095)

H, W = holo.shape

# ============================================================
# QPIパラメータ
# ============================================================

params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=(H, W),
    pixelsize=PIXELSIZE,
    offaxis_center=OFFAXIS_CENTER,
)
ap = params.aperturesize
img_center = params.img_center
print(f"  aperturesize = {ap} px")
print(f"  img_center   = {img_center}")

# ============================================================
# FFT
# ============================================================

fft_full = np.fft.fftshift(np.fft.fft2(holo))      # (H, W) complex
fft_log  = np.log1p(np.abs(fft_full))               # 可視化用 log magnitude

# ============================================================
# 干渉項（サイドバンド / 1次光）抽出 → β
# ============================================================

sb_mask    = make_disk(OFFAXIS_CENTER, ap // 2, (H, W))
sb_cropped = crop_array(fft_full * sb_mask, OFFAXIS_CENTER, ap)   # (ap, ap)
sb_field   = np.fft.ifft2(np.fft.ifftshift(sb_cropped))

beta       = np.abs(sb_field)                        # 干渉項振幅
opd        = unwrap_phase(np.angle(sb_field))        # 位相 [rad]

# ============================================================
# 非干渉項（DC / 0次光）抽出 → α
# ============================================================

dc_mask    = make_disk(img_center, ap // 2, (H, W))
dc_cropped = crop_array(fft_full * dc_mask, img_center, ap)       # (ap, ap)
dc_field   = np.fft.ifft2(np.fft.ifftshift(dc_cropped))

alpha      = np.abs(dc_field)                        # 非干渉項振幅

# ============================================================
# Visibility V = 2β / α
# ============================================================

with np.errstate(divide="ignore", invalid="ignore"):
    visibility = 2.0 * beta / alpha
    visibility[~np.isfinite(visibility)] = 0.0

print(f"  β (mean, center 80%) = {np.percentile(beta, [10, 90])}")
print(f"  α (mean, center 80%) = {np.percentile(alpha, [10, 90])}")
print(f"  V (mean)             = {np.mean(visibility[visibility > 0]):.3f}")

# ============================================================
# 描画
# ============================================================

FONT = {"fontsize": 9, "fontweight": "bold"}

fig = plt.figure(figsize=(12, 7))
gs  = GridSpec(
    2, 3,
    figure=fig,
    left=0.04, right=0.97,
    top=0.95, bottom=0.06,
    wspace=0.35, hspace=0.35,
)

ax_a = fig.add_subplot(gs[0, 0])   # Hologram
ax_b = fig.add_subplot(gs[1, 0])   # FFT
ax_c = fig.add_subplot(gs[0, 1])   # Interferometric Amp
ax_d = fig.add_subplot(gs[1, 1])   # Non-interferometric Amp
ax_e = fig.add_subplot(gs[0, 2])   # OPD
ax_f = fig.add_subplot(gs[1, 2])   # Visibility

# --- a: Hologram ---
im_a = ax_a.imshow(holo, cmap="viridis", origin="upper")
ax_a.set_title("Hologram", **FONT)
cb_a = plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
cb_a.set_label("(ADU)", fontsize=7)
ax_a.set_axis_off()

# --- b: FFT (log magnitude) + circles ---
im_b = ax_b.imshow(fft_log, cmap="viridis", origin="upper")
ax_b.set_title("2D FFT", **FONT)
plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04).set_label("log|FFT|", fontsize=7)
ax_b.set_axis_off()

# 2つの円: +1次(r), DC中心(2r)
radius = ap // 2
circle_specs = [
    ((OFFAXIS_CENTER[1], OFFAXIS_CENTER[0]), radius),       # +1次
    ((img_center[1], img_center[0]), radius * 2),           # DC (2r)
]
for (cx, cy), rr in circle_specs:
    ax_b.add_patch(
        mpatches.Circle(
            (cx, cy),
            radius=rr,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
    )

# --- c: Interferometric Amplitude ---
vmax_c = np.percentile(beta, 99)
im_c = ax_c.imshow(beta, cmap="viridis", vmin=0, vmax=vmax_c, origin="upper")
ax_c.set_title("Interferometric term\nAmplitude", **FONT)
plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04).set_label("(ADU)", fontsize=7)
ax_c.set_axis_off()

# --- d: Non-interferometric Amplitude ---
vmax_d = np.percentile(alpha, 99)
im_d = ax_d.imshow(alpha, cmap="viridis", vmin=0, vmax=vmax_d, origin="upper")
ax_d.set_title("Non-interferometric term\nAmplitude", **FONT)
plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04).set_label("(ADU)", fontsize=7)
ax_d.set_axis_off()

# --- e: OPD ---
opd_centered = opd - np.median(opd)
vlim_e = np.percentile(np.abs(opd_centered), 98)
im_e = ax_e.imshow(opd_centered, cmap="viridis", vmin=-vlim_e, vmax=vlim_e, origin="upper")
ax_e.set_title("Interferometric term\nOPD", **FONT)
plt.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04).set_label("(rad)", fontsize=7)
ax_e.set_axis_off()

# --- f: Visibility ---
vmin_f = max(0, np.percentile(visibility[visibility > 0], 1))
vmax_f = min(1, np.percentile(visibility, 99))
im_f = ax_f.imshow(visibility, cmap="viridis", vmin=vmin_f, vmax=vmax_f, origin="upper")
ax_f.set_title("Visibility", **FONT)
plt.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)
ax_f.set_axis_off()

# ============================================================
# 保存
# ============================================================

save_figure(
    fig,
    params={
        "hologram_path": HOLOGRAM_PATH,
        "wavelength_nm": WAVELENGTH * 1e9,
        "NA": NA,
        "offaxis_center": OFFAXIS_CENTER,
        "aperturesize": int(ap),
        "crop": CROP,
        "visibility_mean": float(np.mean(visibility[visibility > 0])),
    },
    description=(
        "Visibility calculation procedure: "
        "Hologram → FFT → IFFT(sideband/DC) → Amplitude & OPD → Visibility"
    ),
)

plt.show()
print("Done.")
