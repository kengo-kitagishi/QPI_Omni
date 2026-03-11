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
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
import tifffile
from skimage.restoration import unwrap_phase

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from qpi import QPIParameters, get_field, make_disk, crop_array
from figure_logger import save_figure

# ============================================================
# 設定 — 実データに合わせて変更
# ============================================================

DEFAULT_HOLOGRAM_PATH = (
    r"E:\Acuisition\kitagishi\260301\movetest_3\Pos2"
    r"\img_000000000_Default_000.tif"
)

WAVELENGTH     = 658e-9          # [m]
NA             = 0.95
PIXELSIZE      = 3.45e-6 / 40   # [m/px]
OFFAXIS_CENTER = (1712,  532)     # (row, col) — FFT空間でのサイドバンド位置
CROP           = (8, 2056, 208, 2256)  # (row_start, row_end, col_start, col_end)
DEFAULT_EXPORT_SINGLE_PANELS = True
DEFAULT_SINGLE_PANEL_TIF_BASE = (
    Path(__file__).resolve().parents[1] / "results" / "figures" / "visibility_single_panels"
)
HOLOGRAM_VMIN = float(np.percentile(holo, 1))
HOLOGRAM_VMAX = float(np.percentile(holo, 99))
INTERFERO_AMP_VMIN = 0.0
INTERFERO_AMP_VMAX = 250.0
NON_INTERFERO_AMP_VMIN = 0.0
NON_INTERFERO_AMP_VMAX = 500.0
VISIBILITY_VMIN = 0.3
VISIBILITY_VMAX = 1.0


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Create visibility reconstruction 6-panel figure with optional single-panel exports."
    )
    parser.add_argument(
        "--hologram-path",
        default=DEFAULT_HOLOGRAM_PATH,
        help="Input hologram .tif path.",
    )
    parser.add_argument(
        "--single-panel-out-dir",
        default=str(DEFAULT_SINGLE_PANEL_TIF_BASE),
        help="Base directory for single panel TIF/PNG outputs.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional fixed run tag for output folder name (e.g., 20260308T230000).",
    )
    parser.add_argument(
        "--no-single-panel-export",
        action="store_true",
        help="Disable single panel TIF/PNG export.",
    )
    return parser.parse_args()


_args = _parse_args()
HOLOGRAM_PATH = _args.hologram_path
SINGLE_PANEL_TIF_BASE = Path(_args.single_panel_out_dir).expanduser()
EXPORT_SINGLE_PANELS = DEFAULT_EXPORT_SINGLE_PANELS and (not _args.no_single_panel_export)
RUN_TAG_OVERRIDE = _args.run_tag

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
    holo = np.clip(holo, 0, 4096)

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


def _finite_percentile(arr, q, fallback):
    finite = np.asarray(arr)[np.isfinite(arr)]
    if finite.size == 0:
        return float(fallback)
    return float(np.percentile(finite, q))


def _safe_visibility_limits(vis):
    positive = np.asarray(vis)[(vis > 0) & np.isfinite(vis)]
    if positive.size == 0:
        return 0.0, 1.0
    vmin = max(0.0, float(np.percentile(positive, 1)))
    vmax = min(1.0, _finite_percentile(vis, 99, 1.0))
    if vmax <= vmin:
        vmax = min(1.0, vmin + 1e-6)
    return vmin, vmax


def _save_single_panel_preview(name, image, cmap, vmin, vmax, cbar_label, export_png_path=None):
    panel_fig, panel_ax = plt.subplots(figsize=(4.2, 4.2))
    im = panel_ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    panel_ax.set_title(name, **FONT)
    panel_ax.set_axis_off()
    if cbar_label:
        cbar = plt.colorbar(im, ax=panel_ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=7)
    saved_path = save_figure(
        panel_fig,
        params={
            "panel_name": name,
            "cmap": cmap,
            "vmin": None if vmin is None else float(vmin),
            "vmax": None if vmax is None else float(vmax),
        },
        description=f"Single panel preview for range check: {name}",
        publish=False,
    )
    if export_png_path is not None:
        export_png_path = Path(export_png_path)
        export_png_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved_path, export_png_path)
    plt.close(panel_fig)

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
im_a = ax_a.imshow(holo, cmap="viridis", vmin=HOLOGRAM_VMIN, vmax=HOLOGRAM_VMAX, origin="upper")
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
vmin_c, vmax_c = INTERFERO_AMP_VMIN, INTERFERO_AMP_VMAX
im_c = ax_c.imshow(beta, cmap="viridis", vmin=vmin_c, vmax=vmax_c, origin="upper")
ax_c.set_title("Interferometric term\nAmplitude", **FONT)
plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04).set_label("(ADU)", fontsize=7)
ax_c.set_axis_off()

# --- d: Non-interferometric Amplitude ---
vmin_d, vmax_d = NON_INTERFERO_AMP_VMIN, NON_INTERFERO_AMP_VMAX
im_d = ax_d.imshow(alpha, cmap="viridis", vmin=vmin_d, vmax=vmax_d, origin="upper")
ax_d.set_title("Non-interferometric term\nAmplitude", **FONT)
plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04).set_label("(ADU)", fontsize=7)
ax_d.set_axis_off()

# --- e: OPD ---
opd_centered = opd - np.median(opd)
vlim_e = _finite_percentile(np.abs(opd_centered), 98, 1.0)
im_e = ax_e.imshow(opd_centered, cmap="viridis", vmin=-vlim_e, vmax=vlim_e, origin="upper")
ax_e.set_title("Interferometric term\nOPD", **FONT)
plt.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04).set_label("(rad)", fontsize=7)
ax_e.set_axis_off()

# --- f: Visibility ---
vmin_f, vmax_f = VISIBILITY_VMIN, VISIBILITY_VMAX
im_f = ax_f.imshow(visibility, cmap="viridis", vmin=vmin_f, vmax=vmax_f, origin="upper")
ax_f.set_title("Visibility", **FONT)
plt.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)
ax_f.set_axis_off()

# ============================================================
# 単一パネル保存（レンジ確認用）
# ============================================================
if EXPORT_SINGLE_PANELS:
    run_tag = RUN_TAG_OVERRIDE or datetime.now().strftime("%Y%m%dT%H%M%S")
    single_tif_dir = SINGLE_PANEL_TIF_BASE / run_tag
    single_tif_dir.mkdir(parents=True, exist_ok=True)

    raw_tif_specs = [
        ("panel_a_hologram", holo),
        ("panel_b_fft_log", fft_log),
        ("panel_c_interferometric_amp_beta", beta),
        ("panel_d_non_interferometric_amp_alpha", alpha),
        ("panel_e_opd_centered", opd_centered),
        ("panel_f_visibility", visibility),
    ]
    for panel_name, panel_data in raw_tif_specs:
        out_tif = single_tif_dir / f"{panel_name}.tif"
        tifffile.imwrite(out_tif, np.asarray(panel_data, dtype=np.float32))

    _save_single_panel_preview(
        "a: Hologram", holo, "viridis", HOLOGRAM_VMIN, HOLOGRAM_VMAX, "(ADU)",
        export_png_path=single_tif_dir / "panel_a_hologram_viridis.png",
    )
    _save_single_panel_preview(
        "b: 2D FFT (log)", fft_log, "viridis", None, None, "log|FFT|",
        export_png_path=single_tif_dir / "panel_b_fft_log_viridis.png",
    )
    _save_single_panel_preview(
        "c: Interferometric Amp", beta, "viridis", vmin_c, vmax_c, "(ADU)",
        export_png_path=single_tif_dir / "panel_c_interferometric_amp_beta_viridis.png",
    )
    _save_single_panel_preview(
        "d: Non-interferometric Amp", alpha, "viridis", vmin_d, vmax_d, "(ADU)",
        export_png_path=single_tif_dir / "panel_d_non_interferometric_amp_alpha_viridis.png",
    )
    _save_single_panel_preview(
        "e: OPD", opd_centered, "viridis", -vlim_e, vlim_e, "(rad)",
        export_png_path=single_tif_dir / "panel_e_opd_centered_viridis.png",
    )
    _save_single_panel_preview(
        "f: Visibility", visibility, "viridis", vmin_f, vmax_f, "",
        export_png_path=single_tif_dir / "panel_f_visibility_viridis.png",
    )

    print(f"Single-panel raw TIFs: {single_tif_dir}")
    print(f"Single-panel viridis PNGs: {single_tif_dir}")
    print(
        "Current preview ranges: "
        f"c=[{vmin_c:.6g},{vmax_c:.6g}], "
        f"d=[{vmin_d:.6g},{vmax_d:.6g}], "
        f"e=[{-vlim_e:.6g},{vlim_e:.6g}], "
        f"f=[{vmin_f:.6g},{vmax_f:.6g}]"
    )

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

# %%
