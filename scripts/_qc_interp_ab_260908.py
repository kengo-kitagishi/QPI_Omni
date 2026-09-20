"""260908 (cells in the traps): does the sub-pixel interpolation choice change the result?

Same question as the 260917 version, on the real experiment with cells. One frame is
processed three ways and the pairwise differences are written as images:

  bilinear  cv2.warpAffine INTER_LINEAR   -- what produced the data we analyse today
  spline5   scipy.ndimage.shift(order=5)
  fourier   shift by the Fourier shift theorem

A difference image is "same frame, two methods, subtracted": how many rad the final phase
changes per pixel if the method is swapped. fourier - spline5 is the control: those two
compute nearly the same thing, so structure there means an implementation problem, not a
real difference between methods.

260908 has no live drift log; the per-Pos pos_shifts_cal_online.json written by the online
run carries the same per-frame grid node and residual, so it is read instead.
Reads D:\...\260908\ph_zstack_2 phase + E:\260908\ye_grid_0p05_2 grid; writes crops to D:.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import shift as ndshift

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frame", type=int, default=700)
ap.add_argument("--crop-sub", default=r"D:\AquisitionData\Kitagishi\260908\online_crop_sub_zstack_2")
ap.add_argument("--tl-root", default=r"D:\AquisitionData\Kitagishi\260908\ph_zstack_2")
ap.add_argument("--grid-dir", default=r"E:\260908\ye_grid_0p05_2")
ap.add_argument("--grid-z", type=int, default=5)
ap.add_argument("--pos", type=int, nargs="+", default=None)
ap.add_argument("--pos-end", type=int, default=98)
ap.add_argument("--pos-split", type=int, default=52)
ap.add_argument("--tilt-h", type=int, default=270)
ap.add_argument("--out-h", type=int, default=240)
ap.add_argument("--out-root", default=r"D:\AquisitionData\Kitagishi\260908\_interp_ab")
a = ap.parse_args()

PROD_WARP = gs.apply_inverse_shift_warp


def only_image(fn):
    """Swap the interpolator for the phase image only, never for the validity mask.

    process_single_frame warps an all-ones float32 array with the same function to mark
    out-of-bounds pixels; cv2 fills outside with 0 and the mask depends on that, so the mask
    always keeps the production warp and the arms differ by interpolation alone.
    """
    def wrapped(img, shift_x, shift_y):
        if np.asarray(img).dtype == np.float32:
            return PROD_WARP(img, shift_x, shift_y)
        return fn(img, shift_x, shift_y)
    return wrapped


def warp_spline5(img, shift_x, shift_y):
    return ndshift(img.astype(np.float64), (-shift_y, -shift_x), order=5,
                   mode="nearest", prefilter=True)


def warp_fourier(img, shift_x, shift_y):
    f = np.fft.fft2(img.astype(np.float64))
    ky = np.fft.fftfreq(img.shape[0])[:, None]
    kx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(f * np.exp(-2j * np.pi * (ky * -shift_y + kx * -shift_x))))


ARMS = {"bilinear": PROD_WARP,
        "spline5": only_image(warp_spline5),
        "fourier": only_image(warp_fourier)}

grid_dir = Path(a.grid_dir)
n_pos = 0
res = []
for pos in range(1, a.pos_end + 1):
    if a.pos and pos not in a.pos:
        continue
    label = f"Pos{pos}"
    ch_dir = Path(a.crop_sub) / label / "output_phase" / "channels"
    shifts_json = ch_dir / "pos_shifts_cal_online.json"
    rois_json = ch_dir / "channel_rois.json"
    tl_path = Path(a.tl_root) / label / "z000" / "output_phase_raw" / \
        f"img_{a.frame:09d}_ph_000_phase.tif"
    if not (shifts_json.exists() and rois_json.exists() and tl_path.exists()):
        continue
    shifts = json.loads(shifts_json.read_text(encoding="utf-8"))
    frames = shifts.get("frame_results") or shifts.get("alignment_results")
    entry = next((f for f in frames if f and f.get("frame_index") == a.frame), None)
    if entry is None:
        continue
    xi, yi = int(entry["grid_xi"]), int(entry["grid_yi"])
    rx, ry = float(entry["residual_x_px"]), float(entry["residual_y_px"])
    sx, sy = float(entry["shift_x_avg"]), float(entry["shift_y_avg"])

    cal_path = grid_dir / f"grid_calibration_{label}.json"
    grid_cal = gs.load_grid_calibration(str(cal_path)) if cal_path.exists() else {}
    if (xi, yi) in grid_cal:
        cal_dx, cal_dy = grid_cal[(xi, yi)]
    else:
        pixel_scale_um = (gs.SENSOR_PIXEL_SIZE / gs.MAGNIFICATION
                          * gs.ORIGINAL_DIM / gs.RECONSTRUCTED_DIM * 1e6)
        cal_dx = shifts.get("shift_sign_y", 1) * yi * shifts.get("y_step_um", 0.05) / pixel_scale_um
        cal_dy = shifts.get("shift_sign_x", 1) * xi * shifts.get("x_step_um", 0.05) / pixel_scale_um

    g_path = grid_dir / f"{label}_x{xi:+d}_y{yi:+d}" / "output_phase_raw" / \
        f"img_000000000_ph_{a.grid_z:03d}_phase.tif"
    if not g_path.exists():
        continue
    tl_img = tifffile.imread(str(tl_path)).astype(np.float64)
    grid_img = tifffile.imread(str(g_path)).astype(np.float64)
    rois = json.loads(rois_json.read_text(encoding="utf-8"))
    fit_right = pos >= a.pos_split
    res.append((rx, ry))

    for arm, fn in ARMS.items():
        gs.apply_inverse_shift_warp = fn
        crops, _ = gs.process_single_frame(
            tl_img, sx, sy, rois, cal_dx, cal_dy, rx, ry, grid_img,
            output_crop_h_override=a.out_h, tilt_crop_h_raw=a.tilt_h, use_raw_phase=True,
            apply_subpixel_correction=True, fit_right=fit_right, apply_inverse_shift=False)
        for ch, crop in enumerate(crops):
            d = Path(a.out_root) / arm / label / "output_phase" / "channels" / \
                "crop_sub_rawraw" / "z000" / f"ch{ch:02d}"
            d.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(d / f"img_{a.frame:09d}_ph_000.tif"), crop.astype(np.float32))
    gs.apply_inverse_shift_warp = PROD_WARP
    n_pos += 1

r = np.array(res)
print(f"260908 frame {a.frame}: {n_pos} Pos, arms {list(ARMS)}")
if len(r):
    print(f"residual shift applied: |rx| median {np.median(np.abs(r[:, 0])):.3f} px "
          f"max {np.abs(r[:, 0]).max():.3f};  |ry| median {np.median(np.abs(r[:, 1])):.3f} px "
          f"max {np.abs(r[:, 1]).max():.3f}")
print(f"crops under {a.out_root}\\<arm>\\")
