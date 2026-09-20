"""260908 (cells): crop_sub with the tilt correct replaced by a quadratic fitted outside the channel.

Same as the 260917 version, with the mask source as an explicit arm, because with cells it
matters which image defines "the channel":

  mask=grid  the channel comes from the GRID's output_phase, which holds no cells. The mask is
             pure geometry, so a cell can never be counted as background.
  mask=tl    the channel comes from the timelapse's own output_phase. A cell has POSITIVE
             phase, so its pixels sit above the threshold and get labelled "outside the
             channel" -- the fit then sees the cell. This arm exists to show that.

Everything else is production: same grid node, same residual warp, same 2pi offset, same
output width. Out-of-image and warp-invalid pixels are excluded from the fit and zeroed in the
output, as production does with its validity mask.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs  # noqa: E402
from ecc_utils import extract_rect_roi  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frame", type=int, default=700)
ap.add_argument("--crop-sub", default=r"D:\AquisitionData\Kitagishi\260908\online_crop_sub_zstack_2")
ap.add_argument("--tl-root", default=r"D:\AquisitionData\Kitagishi\260908\ph_zstack_2")
ap.add_argument("--grid-dir", default=r"E:\260908\ye_grid_0p05_2")
ap.add_argument("--grid-z", type=int, default=5)
ap.add_argument("--mask-from", choices=["grid", "tl"], default="grid")
ap.add_argument("--ch-thresh", type=float, default=-1.0)
ap.add_argument("--dilate", type=int, default=2)
ap.add_argument("--min-bg", type=int, default=500)
ap.add_argument("--valid-erode", type=int, default=1)
ap.add_argument("--pos-end", type=int, default=98)
ap.add_argument("--pos-split", type=int, default=52)
ap.add_argument("--tilt-h", type=int, default=270)
ap.add_argument("--out-h", type=int, default=240)
ap.add_argument("--out-root", default=r"D:\AquisitionData\Kitagishi\260908\_outsidech_quadfit")
a = ap.parse_args()

grid_dir = Path(a.grid_dir)
out_root = Path(a.out_root) / f"mask_{a.mask_from}"


def quad_fit_outside(delta, outside):
    h, w = delta.shape
    c, r = np.meshgrid(np.arange(w) / w, np.arange(h) / h)
    terms = [np.ones_like(c), c, r, c * c, r * r, c * r]
    A = np.stack([t[outside] for t in terms], axis=1)
    co, *_ = np.linalg.lstsq(A, delta[outside], rcond=None)
    return delta - sum(k * t for k, t in zip(co, terms))


n_pos = n_crop = n_fallback = 0
bg_frac, oob_frac = [], []
for pos in range(1, a.pos_end + 1):
    label = f"Pos{pos}"
    ch_dir = Path(a.crop_sub) / label / "output_phase" / "channels"
    shifts_json, rois_json = ch_dir / "pos_shifts_cal_online.json", ch_dir / "channel_rois.json"
    tl_raw = Path(a.tl_root) / label / "z000" / "output_phase_raw" / \
        f"img_{a.frame:09d}_ph_000_phase.tif"
    tl_ph = Path(a.tl_root) / label / "z000" / "output_phase" / \
        f"img_{a.frame:09d}_ph_000_phase.tif"
    if not (shifts_json.exists() and rois_json.exists() and tl_raw.exists()):
        continue
    shifts = json.loads(shifts_json.read_text(encoding="utf-8"))
    frames = shifts.get("frame_results") or shifts.get("alignment_results") or []
    e = next((f for f in frames if f and f.get("frame_index") == a.frame), None)
    if e is None:
        continue
    xi, yi = int(e["grid_xi"]), int(e["grid_yi"])
    rx, ry = float(e["residual_x_px"]), float(e["residual_y_px"])
    cal_path = grid_dir / f"grid_calibration_{label}.json"
    grid_cal = gs.load_grid_calibration(str(cal_path)) if cal_path.exists() else {}
    if (xi, yi) in grid_cal:
        cal_dx, cal_dy = grid_cal[(xi, yi)]
    else:
        px = gs.SENSOR_PIXEL_SIZE / gs.MAGNIFICATION * gs.ORIGINAL_DIM / gs.RECONSTRUCTED_DIM * 1e6
        cal_dx = shifts.get("shift_sign_y", 1) * yi * shifts.get("y_step_um", 0.05) / px
        cal_dy = shifts.get("shift_sign_x", 1) * xi * shifts.get("x_step_um", 0.05) / px

    node = grid_dir / f"{label}_x{xi:+d}_y{yi:+d}"
    g_raw = node / "output_phase_raw" / f"img_000000000_ph_{a.grid_z:03d}_phase.tif"
    g_ph = node / "output_phase" / f"img_000000000_ph_{a.grid_z:03d}_phase.tif"
    if not g_raw.exists():
        continue
    mask_src = g_ph if a.mask_from == "grid" else tl_ph
    if not mask_src.exists():
        continue

    tl_img = tifffile.imread(str(tl_raw)).astype(np.float64)
    valid_full = np.ones_like(tl_img, dtype=np.float32)
    if rx or ry:
        tl_img = gs.apply_inverse_shift_warp(tl_img, rx, ry)
        valid_full = gs.apply_inverse_shift_warp(valid_full, rx, ry)
    grid_img = tifffile.imread(str(g_raw)).astype(np.float64)
    mask_img = tifffile.imread(str(mask_src)).astype(np.float64)
    full = tl_img - grid_img
    fit_right = pos >= a.pos_split
    ones = np.ones_like(tl_img, dtype=np.float64)

    for ch, roi in enumerate(json.loads(rois_json.read_text(encoding="utf-8"))):
        cy, cx = int(round(roi["cy"] + cal_dy)), int(round(roi["cx"] + cal_dx))
        delta = extract_rect_roi(full, cy, cx, roi["crop_w"], a.tilt_h)
        mask_w = extract_rect_roi(mask_img, cy, cx, roi["crop_w"], a.tilt_h)
        inside = mask_w < a.ch_thresh
        if a.dilate:
            inside = binary_dilation(inside, iterations=a.dilate)
        valid_w = extract_rect_roi(valid_full.astype(np.float64), cy, cx,
                                   roi["crop_w"], a.tilt_h) > 0.999
        if a.valid_erode:
            valid_w = binary_erosion(valid_w, iterations=a.valid_erode, border_value=1)
        valid_w &= extract_rect_roi(ones, cy, cx, roi["crop_w"], a.tilt_h) > 0.5
        outside = (~inside) & valid_w

        fn = max(1, a.tilt_h // 3)
        bg = delta[:, -fn:] if fit_right else delta[:, :fn]
        k = int(round(float(np.mean(bg)) / (2.0 * np.pi)))
        if k:
            delta = delta - k * 2.0 * np.pi

        bg_frac.append(outside.mean())
        oob_frac.append(1.0 - valid_w.mean())
        if outside.sum() >= a.min_bg:
            flat = quad_fit_outside(delta, outside)
        else:
            flat = delta
            n_fallback += 1
        start = (a.tilt_h - a.out_h) // 2
        crop = flat[:, start:start + a.out_h] * valid_w[:, start:start + a.out_h]
        d = (out_root / label / "output_phase" / "channels" /
             "crop_sub_rawraw" / "z000" / f"ch{ch:02d}")
        d.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(d / f"img_{a.frame:09d}_ph_000.tif"), crop.astype(np.float32))
        n_crop += 1
    n_pos += 1

print(f"260908 frame {a.frame}, mask from {a.mask_from}: {n_pos} Pos, {n_crop} crops -> {out_root}")
print(f"  fit area (outside channel, valid): mean {100*np.mean(bg_frac):.1f} % "
      f"(min {100*np.min(bg_frac):.1f} %, max {100*np.max(bg_frac):.1f} %)")
print(f"  invalid area excluded: mean {100*np.mean(oob_frac):.1f} %, max {100*np.max(oob_frac):.1f} %")
print(f"  channels with too little background ({a.min_bg} px): {n_fallback}")
