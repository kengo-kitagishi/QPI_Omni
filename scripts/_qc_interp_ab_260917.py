"""Same frame, same grid, different sub-pixel interpolation of the residual shift.

The production kernel warps the timelapse frame by the residual (r < 0.145 px) with
cv2.warpAffine INTER_LINEAR before subtracting the grid reference. This swaps only that one
function and writes the resulting crops, so the arms can be compared as images:

  bilinear   cv2.warpAffine INTER_LINEAR            -- production
  spline5    scipy.ndimage.shift(order=5)           -- quintic spline on the unwrapped phase
  fourier    phase shifted by the Fourier shift theorem
  none       no residual warp at all                -- the size of the shift itself

Everything else (crop, tilt fit, grid point, calibration) is the production path unchanged.
Reads the kept output_phase_raw of ph_zstack_test_3; writes crops to D:.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import shift as ndshift

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config_zstack.json")
ap.add_argument("--log", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log_zstack.json")
ap.add_argument("--frame", type=int, default=400)
ap.add_argument("--pos-end", type=int, default=99)
ap.add_argument("--out-root", default=r"D:\AquisitionData\Kitagishi\260917\_interp_ab")
a = ap.parse_args()

PROD_WARP = gs.apply_inverse_shift_warp      # keep the production implementation to restore


def only_image(fn):
    """Swap the interpolator for the image only.

    process_single_frame warps two things with this same function: the phase frame (float64)
    and an all-ones float32 array whose warped values define the out-of-bounds mask. cv2 fills
    outside with 0, which is what that mask relies on; a spline with mode='nearest' or a
    Fourier shift of a step would change the MASK as well as the interpolation, so the arms
    would differ for two reasons at once. The mask keeps the production warp in every arm.
    """
    def wrapped(img, shift_x, shift_y):
        if np.asarray(img).dtype == np.float32:      # the ones-array validity mask
            return PROD_WARP(img, shift_x, shift_y)
        return fn(img, shift_x, shift_y)
    return wrapped


def warp_spline5(img, shift_x, shift_y):
    # ndimage works in (row, col) = (y, x); production shifts by (-shift_x, -shift_y)
    return ndshift(img.astype(np.float64), (-shift_y, -shift_x), order=5,
                   mode="nearest", prefilter=True)


def warp_fourier(img, shift_x, shift_y):
    f = np.fft.fft2(img.astype(np.float64))
    ky = np.fft.fftfreq(img.shape[0])[:, None]
    kx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(f * np.exp(-2j * np.pi * (ky * -shift_y + kx * -shift_x))))


def warp_none(img, shift_x, shift_y):
    return np.asarray(img, dtype=np.float64)


ARMS = {"bilinear": PROD_WARP,
        "spline5": only_image(warp_spline5),
        "fourier": only_image(warp_fourier),
        "none": only_image(warp_none)}

cfg = json.load(open(a.config, encoding="utf-8"))
log = [e for e in json.load(open(a.log, encoding="utf-8")) if e.get("per_pos")]
entry = next((e for e in log if e["timepoint"] == a.frame), None)
if entry is None:
    sys.exit(f"frame {a.frame} is not in the drift log")
tilt_h = cfg.get("tilt_crop_h_raw", 270)
out_h = cfg.get("crop_sub_output_crop_h") or tilt_h
pos_split = int(cfg["pos_split"])
grid_dir = Path(cfg["grid_dir"])
grid_z = int(cfg["raw_grid_z_index"])

residuals = []
n_pos = 0
for p in entry["positions"]:
    pos = int(re.match(r"Pos(\d+)", p["pos_label"]).group(1))
    if pos > a.pos_end:
        continue
    label, sx, sy = p["pos_label"], p["tx_avg_px"], p["ty_avg_px"]
    fit_right = pos >= pos_split
    tl_path = Path(cfg["save_dir"]) / label / "z000" / "output_phase_raw" / \
        f"img_{a.frame:09d}_ph_000_phase.tif"
    if not tl_path.exists():
        continue
    pos_map = gs.scan_grid_positions(grid_dir, label)
    cal_path = grid_dir / f"grid_calibration_{label}.json"
    if not pos_map or not cal_path.exists():
        continue
    xi, yi, _, _, _, cal_dx, cal_dy, rx, ry = gs.select_grid(
        sx, sy, pos_map, gs.load_grid_calibration(str(cal_path)),
        pixel_scale_um=cfg["pixel_scale_um"],
        x_step=cfg.get("crop_sub_x_step_um", 0.05), y_step=cfg.get("crop_sub_y_step_um", 0.05),
        shift_sign_x=-1, shift_sign_y=-1)
    g_path = pos_map[(xi, yi)] / "output_phase_raw" / f"img_000000000_ph_{grid_z:03d}_phase.tif"
    if not g_path.exists():
        continue
    tl_img = tifffile.imread(str(tl_path)).astype(np.float64)
    grid_img = tifffile.imread(str(g_path)).astype(np.float64)
    rois = json.load(open(grid_dir / f"{label}_x+0_y+0" / "output_phase" / "channels" /
                          "channel_rois.json", encoding="utf-8"))
    residuals.append((rx, ry))

    for arm, fn in ARMS.items():
        gs.apply_inverse_shift_warp = fn          # swap only the interpolation
        crops, _ = gs.process_single_frame(
            tl_img, sx, sy, rois, cal_dx, cal_dy, rx, ry, grid_img,
            output_crop_h_override=out_h, tilt_crop_h_raw=tilt_h, use_raw_phase=True,
            apply_subpixel_correction=True, fit_right=fit_right, apply_inverse_shift=False)
        for ch, crop in enumerate(crops):
            d = Path(a.out_root) / arm / label / "output_phase" / "channels" / \
                "crop_sub_rawraw" / "z000" / f"ch{ch:02d}"
            d.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(d / f"img_{a.frame:09d}_ph_000.tif"), crop.astype(np.float32))
    gs.apply_inverse_shift_warp = PROD_WARP
    n_pos += 1

r = np.array(residuals)
print(f"frame {a.frame}: {n_pos} Pos, arms {list(ARMS)}")
print(f"residual shift actually applied: |rx| median {np.median(np.abs(r[:, 0])):.3f} px, "
      f"max {np.abs(r[:, 0]).max():.3f};  |ry| median {np.median(np.abs(r[:, 1])):.3f} px, "
      f"max {np.abs(r[:, 1]).max():.3f}")
print(f"crops under {a.out_root}\\<arm>\\")
