# %%
"""
qpi_fig_01_generate_panels.py

Generate intermediate panels from 260321 _cal_grid_0pergluc_60ms_1 hologram TIF,
to be loaded by qpi_fig_01_reconstruction_procedure.py.

Panel layout:
  f001 : raw hologram (panel a)
  f003 : 2D FFT + circles, RGB (panel b)
  f005 : sideband-centered FFT log-mag (panel c)
  f004 : LP-filtered FFT log-mag (panel d)
  f006 : after 2D IFT - amplitude (panel e)
  f007 : final OPD after BG subtraction, viridis RGB (panel f)

Output:
  PNG (uint8, for display)  <- loaded by qpi_fig_01_reconstruction_procedure.py
  TIF (float32, raw data)
  Save location: results/figures/qpi_fig_01_panels/YYYYMMDDTHHMMSS/
"""

import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase

import argparse

sys.path.insert(0, str(Path(__file__).parent))
from qpi import QPIParameters, get_field, make_disk, crop_array
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

# ============================================================
# CLI
# ============================================================
_parser = argparse.ArgumentParser()
_parser.add_argument("--hologram-path", default=None)
_parser.add_argument("--background-path", default=None)
_args = _parser.parse_args()

# ============================================================
# Settings
# ============================================================
POS1_PATH = (
    r"D:\AquisitionData\Kitagishi\260321\_cal_grid_0pergluc_60ms_1"
    r"\Pos1\img_000000000_ph_000.tif"
)
POS0_PATH = (
    r"D:\AquisitionData\Kitagishi\260321\_cal_grid_0pergluc_60ms_1"
    r"\Pos0\img_000000000_ph_000.tif"
)

if _args.hologram_path:
    POS1_PATH = _args.hologram_path
if _args.background_path:
    POS0_PATH = _args.background_path

# Right channel (260321 Pos < 31): col 400-2448
# CROP_REGION (208-2256) in optical_config.py is a generic value, not used here
CROP = (0, 2048, 400, 2448)

# ============================================================
# Output directory
# ============================================================
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")
OUT_DIR = _REPO_ROOT / "results" / "figures" / "qpi_fig_01_panels" / _TIMESTAMP
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}")


# ============================================================
# Utilities
# ============================================================
def _load_hologram(path: str) -> np.ndarray:
    r0, r1, c0, c1 = CROP
    try:
        img = tifffile.imread(path).astype(np.float64)
    except Exception:
        img = np.array(Image.open(path)).astype(np.float64)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img[r0:r1, c0:c1]


def _to_uint8(arr: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """float array -> uint8 (grayscale)"""
    v0 = float(np.percentile(arr, 1)) if vmin is None else vmin
    v1 = float(np.percentile(arr, 99)) if vmax is None else vmax
    clipped = np.clip(arr, v0, v1)
    scaled = (clipped - v0) / max(v1 - v0, 1e-12) * 255.0
    return scaled.astype(np.uint8)


def _save(name: str, png_arr: np.ndarray, tif_arr: np.ndarray):
    """Save PNG (uint8 or uint8-RGB) and TIF (float32)"""
    png_path = OUT_DIR / f"{name}.png"
    tif_path = OUT_DIR / f"{name}.tif"
    Image.fromarray(png_arr).save(png_path)
    tifffile.imwrite(tif_path, np.asarray(tif_arr, dtype=np.float32))
    print(f"  saved {name}.png / .tif")


# ============================================================
# Data loading
# ============================================================
print("Loading holograms...")
holo   = _load_hologram(POS1_PATH)   # sample (Pos1)
holo_bg = _load_hologram(POS0_PATH)  # background (Pos0)
H, W = holo.shape
print(f"  holo shape: {holo.shape}")

# QPI parameters
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
# FFT (common)
# ============================================================
fft_s  = np.fft.fftshift(np.fft.fft2(holo))
fft_bg = np.fft.fftshift(np.fft.fft2(holo_bg))

# ============================================================
# f001: raw hologram (panel a)
# ============================================================
print("\n[f001] raw hologram")
f001_png = _to_uint8(holo)
_save("f001", f001_png, holo.astype(np.float32))

# ============================================================
# f003: 2D FFT + circles, RGB (panel b)
# ============================================================
print("[f003] 2D FFT + circles")
fft_log = np.log1p(np.abs(fft_s))
fft_log_vmin = float(fft_log.min())   # matplotlib auto = min/max, unified with c,d
fft_log_vmax = float(fft_log.max())
radius = ap // 2

fig_b, ax_b = plt.subplots(figsize=(8, 8))
ax_b.imshow(fft_log, cmap="gray", origin="upper", vmin=fft_log_vmin, vmax=fft_log_vmax)
ax_b.axis("off")
for (cr, cc), rr in [
    (OFFAXIS_CENTER, radius),    # +1st order
    (img_center,    radius * 2), # DC (0th order)
]:
    ax_b.add_patch(mpatches.Circle(
        (cc, cr), radius=rr, fill=False, edgecolor="red", linewidth=1.5,
    ))
_tmp_b = OUT_DIR / "_tmp_f003.png"
fig_b.savefig(_tmp_b, bbox_inches="tight", pad_inches=0, dpi=150)
plt.close(fig_b)
f003_png = np.array(Image.open(_tmp_b).convert("RGB"))
_tmp_b.unlink()
_save("f003", f003_png, fft_log.astype(np.float32))

# ============================================================
# f005: sideband-centered FFT - sideband region crop (panel c)
#   crop_array extracts ap x ap region centered on offaxis_center
# ============================================================
print("[f005] sideband-centered FFT (cropped sideband region)")
sb_raw = crop_array(fft_s, OFFAXIS_CENTER, ap)   # ap x ap complex
sb_log = np.log1p(np.abs(sb_raw))
f005_png = _to_uint8(sb_log, vmin=fft_log_vmin, vmax=fft_log_vmax)  # same scale as panel b
_save("f005", f005_png, sb_log.astype(np.float32))

# ============================================================
# f004: LP-filtered - apply disk mask to sideband crop (panel d)
# ============================================================
print("[f004] LP-filtered sideband")
sb_h, sb_w = sb_raw.shape
lp_mask_ap = make_disk((sb_h // 2, sb_w // 2), radius, (sb_h, sb_w))
sb_lp = sb_raw * lp_mask_ap
lp_log = np.log1p(np.abs(sb_lp))
f004_png = _to_uint8(lp_log, vmin=fft_log_vmin, vmax=fft_log_vmax)  # same scale as panel b
_save("f004", f004_png, lp_log.astype(np.float32))

# ============================================================
# f006: Pos1 OPD without BG subtraction (panel e)
#   Phase immediately after 2D IFT (unwrapped) - shows state before BG subtraction (e->f)
# ============================================================
print("[f006] Pos1 OPD (no BG subtraction)")
field_ap = get_field(holo, params)      # ap x ap complex field (Pos1)
opd_ap_raw = unwrap_phase(np.angle(field_ap))
opd_ap_raw -= np.median(opd_ap_raw)

# ============================================================
# f007: final OPD after BG subtraction (panel f)
# ============================================================
_has_bg = _args.background_path is not None or not _args.hologram_path
if _has_bg:
    print("[f007] final OPD (BG subtracted: Pos0)")
    field_ap_bg = get_field(holo_bg, params)
    opd_ap = unwrap_phase(np.angle(field_ap) - np.angle(field_ap_bg))
else:
    print("[f007] final OPD (no BG subtraction)")
    opd_ap = unwrap_phase(np.angle(field_ap))
opd_ap -= np.median(opd_ap)

# Display e and f with common scale
vlim = float(np.percentile(np.abs(np.concatenate([opd_ap_raw.ravel(), opd_ap.ravel()])), 98))
f006_png = _to_uint8(opd_ap_raw, vmin=-vlim, vmax=vlim)
_save("f006", f006_png, opd_ap_raw.astype(np.float32))
f007_gray = _to_uint8(opd_ap, vmin=-vlim, vmax=vlim)   # grayscale
_save("f007", f007_gray, opd_ap.astype(np.float32))

# ============================================================
# Pos0 / Pos1 reconstruction TIF
#   2048x2048 hologram -> FFT -> disk mask + shift+crop(ap x ap) -> IFFT
#   Save amplitude and wrapped phase per Pos
# ============================================================
print("\n[reconstruction TIFs] Pos0 & Pos1")

def _save_recon_tifs(hologram: np.ndarray, label: str):
    """Execute get_field pipeline and save amplitude / phase TIF"""
    field = get_field(hologram, params)   # ap x ap complex
    amp   = np.abs(field).astype(np.float32)
    phase = np.angle(field).astype(np.float32)
    tifffile.imwrite(OUT_DIR / f"{label}_amplitude.tif", amp)
    tifffile.imwrite(OUT_DIR / f"{label}_phase.tif",     phase)
    print(f"  {label}: amp={amp.shape} phase={phase.shape}  "
          f"amp=[{amp.min():.1f}, {amp.max():.1f}]  "
          f"phase=[{phase.min():.3f}, {phase.max():.3f}] rad")

_save_recon_tifs(holo_bg, "Pos0")   # already loaded
_save_recon_tifs(holo,    "Pos1")   # could reuse field_ap but explicitly recompute

# ============================================================
# Completion report
# ============================================================
print("\n" + "=" * 60)
print("Panel generation complete.")
print(f"Output dir: {OUT_DIR}")
print()
print("-> Update qpi_fig_01_reconstruction_procedure.py:")
print(f'  INBOX = r"{OUT_DIR}"')
print(f'  PRE   = ""')
print("=" * 60)

# %%
