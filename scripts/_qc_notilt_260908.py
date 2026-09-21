"""crop_sub output with the tilt correction switched off.

Production's per-channel step is apply_2pi_tilt_crop: a 2pi offset from the end third, then a
straight line fitted on that same end third and subtracted, then the centre crop. This
produces the same crops with that middle step removed:

  tilt      2pi offset + line fit on the end third + centre crop   -- production
  notilt    2pi offset + centre crop                               -- no line subtracted
  nothing   centre crop only                                       -- not even the 2pi offset

Everything before it (grid node, residual warp with the production cv2 bilinear, subtraction)
is untouched, so the arms differ only in the background step. Works on either session.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frame", type=int, default=200)
ap.add_argument("--crop-sub", default=r"D:\AquisitionData\Kitagishi\260908\online_crop_sub_zstack_2")
ap.add_argument("--tl-root", default=r"D:\AquisitionData\Kitagishi\260908\ph_zstack_2")
ap.add_argument("--grid-dir", default=r"E:\260908\ye_grid_0p05_2")
ap.add_argument("--grid-z", type=int, default=5)
ap.add_argument("--pos-end", type=int, default=98)
ap.add_argument("--pos-split", type=int, default=52)
ap.add_argument("--tilt-h", type=int, default=270)
ap.add_argument("--out-h", type=int, default=240)
ap.add_argument("--out-root", default=r"D:\AquisitionData\Kitagishi\260908\_notilt_ab")
a = ap.parse_args()

PROD_CORRECT = gs._raw_subtract_correct


def centre_crop(sub_large, out_crop_h):
    start = (sub_large.shape[1] - out_crop_h) // 2
    return sub_large[:, start:start + out_crop_h]


def correct_notilt(sub_large, out_crop_h, fit_right=False, tilt_crop_h_raw=None):
    """2pi offset from the end third, then crop. No line fitted, no line subtracted."""
    th = tilt_crop_h_raw if tilt_crop_h_raw is not None else a.tilt_h
    fn = max(1, th // 3)
    bg = sub_large[:, -fn:] if fit_right else sub_large[:, :fn]
    k = int(round(float(np.mean(bg)) / (2.0 * np.pi)))
    return centre_crop(sub_large - k * 2.0 * np.pi if k else sub_large, out_crop_h)


def correct_nothing(sub_large, out_crop_h, fit_right=False, tilt_crop_h_raw=None):
    return centre_crop(sub_large, out_crop_h)


ARMS = {"tilt": PROD_CORRECT, "notilt": correct_notilt, "nothing": correct_nothing}

grid_dir = Path(a.grid_dir)
n_pos = 0
for pos in range(1, a.pos_end + 1):
    label = f"Pos{pos}"
    ch_dir = Path(a.crop_sub) / label / "output_phase" / "channels"
    shifts_json, rois_json = ch_dir / "pos_shifts_cal_online.json", ch_dir / "channel_rois.json"
    tl_path = Path(a.tl_root) / label / "z000" / "output_phase_raw" / \
        f"img_{a.frame:09d}_ph_000_phase.tif"
    if not (shifts_json.exists() and rois_json.exists() and tl_path.exists()):
        continue
    shifts = json.loads(shifts_json.read_text(encoding="utf-8"))
    frames = shifts.get("frame_results") or shifts.get("alignment_results") or []
    e = next((f for f in frames if f and f.get("frame_index") == a.frame), None)
    if e is None:
        continue
    xi, yi = int(e["grid_xi"]), int(e["grid_yi"])
    rx, ry = float(e["residual_x_px"]), float(e["residual_y_px"])
    sx, sy = float(e["shift_x_avg"]), float(e["shift_y_avg"])
    cal_path = grid_dir / f"grid_calibration_{label}.json"
    grid_cal = gs.load_grid_calibration(str(cal_path)) if cal_path.exists() else {}
    if (xi, yi) in grid_cal:
        cal_dx, cal_dy = grid_cal[(xi, yi)]
    else:
        px = gs.SENSOR_PIXEL_SIZE / gs.MAGNIFICATION * gs.ORIGINAL_DIM / gs.RECONSTRUCTED_DIM * 1e6
        cal_dx = shifts.get("shift_sign_y", 1) * yi * shifts.get("y_step_um", 0.05) / px
        cal_dy = shifts.get("shift_sign_x", 1) * xi * shifts.get("x_step_um", 0.05) / px
    g_path = grid_dir / f"{label}_x{xi:+d}_y{yi:+d}" / "output_phase_raw" / \
        f"img_000000000_ph_{a.grid_z:03d}_phase.tif"
    if not g_path.exists():
        continue
    tl_img = tifffile.imread(str(tl_path)).astype(np.float64)
    grid_img = tifffile.imread(str(g_path)).astype(np.float64)
    rois = json.loads(rois_json.read_text(encoding="utf-8"))
    fit_right = pos >= a.pos_split

    for arm, fn in ARMS.items():
        gs._raw_subtract_correct = fn
        crops, _ = gs.process_single_frame(
            tl_img, sx, sy, rois, cal_dx, cal_dy, rx, ry, grid_img,
            output_crop_h_override=a.out_h, tilt_crop_h_raw=a.tilt_h, use_raw_phase=True,
            apply_subpixel_correction=True, fit_right=fit_right, apply_inverse_shift=False)
        gs._raw_subtract_correct = PROD_CORRECT
        for ch, crop in enumerate(crops):
            d = Path(a.out_root) / arm / label / "output_phase" / "channels" / \
                "crop_sub_rawraw" / "z000" / f"ch{ch:02d}"
            d.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(d / f"img_{a.frame:09d}_ph_000.tif"), crop.astype(np.float32))
    n_pos += 1

print(f"frame {a.frame}: {n_pos} Pos, arms {list(ARMS)} -> {a.out_root}")
