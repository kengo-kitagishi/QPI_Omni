"""crop_sub with the tilt correct replaced by a quadratic fitted OUTSIDE the channel.

Production per-channel step on the subtracted delta is: 2pi offset from the aperture-end
third, a straight line fitted on that same third, then the centre crop. The line is fitted on
one end only and extrapolated across the rest.

Here the background region is defined by the channel itself instead. In output_phase (the
image the online run writes), pixels below CH_THRESH rad are inside the channel; everything
above it is outside the channel, and outside the channel there can be no cell whatever the
trap holds. So:

  1. delta = timelapse raw phase - grid raw phase, in the same wide window as production
  2. mask = (output_phase of the same frame, same window) < CH_THRESH, grown by DILATE px
  3. 2pi offset, as production does
  4. fit a 2D quadratic (1, c, r, c^2, r^2, c*r) on the pixels OUTSIDE the mask, subtract it
  5. centre crop to the same output width

Both ends and both long sides of the window feed the fit, so nothing is extrapolated, and the
cell can never enter the fit region by construction.
"""
import argparse
import json
import re
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
ap.add_argument("--config", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config_zstack.json")
ap.add_argument("--log", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log_zstack.json")
ap.add_argument("--frame", type=int, default=4)
ap.add_argument("--ch-thresh", type=float, default=-1.0,
                help="output_phase below this is inside the channel [rad]")
ap.add_argument("--dilate", type=int, default=2, help="grow the channel mask by this many px")
ap.add_argument("--min-bg", type=int, default=500, help="minimum outside-channel pixels to fit")
ap.add_argument("--valid-erode", type=int, default=1,
                help="erode the validity mask, as production does")
ap.add_argument("--out-root", default=r"E:\260917\outsidech_quadfit_test_6")
a = ap.parse_args()

cfg = json.load(open(a.config, encoding="utf-8"))
log = [e for e in json.load(open(a.log, encoding="utf-8")) if e.get("per_pos")]
entry = next((e for e in log if e["timepoint"] == a.frame), None)
if entry is None:
    sys.exit(f"frame {a.frame} is not in the drift log")
grid_dir = Path(cfg["grid_dir"])
grid_z = int(cfg["raw_grid_z_index"])
pos_split = int(cfg["pos_split"])
tilt_h = cfg.get("tilt_crop_h_raw", 270)
out_h = cfg.get("crop_sub_output_crop_h") or tilt_h


def quad_fit_outside(delta, outside):
    h, w = delta.shape
    c, r = np.meshgrid(np.arange(w) / w, np.arange(h) / h)
    terms = [np.ones_like(c), c, r, c * c, r * r, c * r]
    A = np.stack([t[outside] for t in terms], axis=1)
    co, *_ = np.linalg.lstsq(A, delta[outside], rcond=None)
    return delta - sum(k * t for k, t in zip(co, terms))


n_pos = n_crop = n_fallback = 0
bg_frac = []
oob_frac = []
for p in entry["positions"]:
    pos = int(re.match(r"Pos(\d+)", p["pos_label"]).group(1))
    label, sx, sy = p["pos_label"], p["tx_avg_px"], p["ty_avg_px"]
    fit_right = pos >= pos_split
    tl_raw = Path(cfg["save_dir"]) / label / "z000" / "output_phase_raw" / \
        f"img_{a.frame:09d}_ph_000_phase.tif"
    ph = Path(cfg["save_dir"]) / label / "z000" / "output_phase" / \
        f"img_{a.frame:09d}_ph_000_phase.tif"
    rois_path = grid_dir / f"{label}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json"
    cal_path = grid_dir / f"grid_calibration_{label}.json"
    if not (tl_raw.exists() and ph.exists() and rois_path.exists() and cal_path.exists()):
        continue
    pos_map = gs.scan_grid_positions(grid_dir, label)
    if not pos_map:
        continue
    xi, yi, _, _, _, cal_dx, cal_dy, rx, ry = gs.select_grid(
        sx, sy, pos_map, gs.load_grid_calibration(str(cal_path)),
        pixel_scale_um=cfg["pixel_scale_um"], x_step=cfg.get("crop_sub_x_step_um", 0.05),
        y_step=cfg.get("crop_sub_y_step_um", 0.05), shift_sign_x=-1, shift_sign_y=-1)
    g_path = pos_map[(xi, yi)] / "output_phase_raw" / f"img_000000000_ph_{grid_z:03d}_phase.tif"
    if not g_path.exists():
        continue

    tl_img = tifffile.imread(str(tl_raw)).astype(np.float64)
    valid_full = np.ones_like(tl_img, dtype=np.float32)
    if rx or ry:                                   # the production residual warp, unchanged
        tl_img = gs.apply_inverse_shift_warp(tl_img, rx, ry)
        valid_full = gs.apply_inverse_shift_warp(valid_full, rx, ry)
    grid_img = tifffile.imread(str(g_path)).astype(np.float64)
    ph_img = tifffile.imread(str(ph)).astype(np.float64)
    full = tl_img - grid_img

    for ch, roi in enumerate(json.loads(rois_path.read_text(encoding="utf-8"))):
        cy = int(round(roi["cy"] + cal_dy))
        cx = int(round(roi["cx"] + cal_dx))
        delta = extract_rect_roi(full, cy, cx, roi["crop_w"], tilt_h)
        ph_w = extract_rect_roi(ph_img, cy, cx, roi["crop_w"], tilt_h)
        inside = ph_w < a.ch_thresh
        if a.dilate:
            inside = binary_dilation(inside, iterations=a.dilate)
        # Pixels the window took from outside the 511 px reconstruction are zero-padded by
        # extract_rect_roi, and zero is "above the threshold", so without this they would be
        # counted as outside-channel background and pull the fit. The warp leaves the same
        # kind of invalid border. Production tracks exactly this with the ones-array mask.
        valid_w = extract_rect_roi(valid_full.astype(np.float64), cy, cx,
                                   roi["crop_w"], tilt_h) > 0.999
        if a.valid_erode:
            valid_w = binary_erosion(valid_w, iterations=a.valid_erode, border_value=1)
        in_image = extract_rect_roi(np.ones_like(tl_img, dtype=np.float64), cy, cx,
                                    roi["crop_w"], tilt_h) > 0.5
        valid_w &= in_image
        outside = (~inside) & valid_w

        fn = max(1, tilt_h // 3)                   # 2pi offset, same as production
        bg = delta[:, -fn:] if fit_right else delta[:, :fn]
        k = int(round(float(np.mean(bg)) / (2.0 * np.pi)))
        if k:
            delta = delta - k * 2.0 * np.pi

        bg_frac.append(outside.mean())
        oob_frac.append(1.0 - valid_w.mean())
        if outside.sum() >= a.min_bg:
            flat = quad_fit_outside(delta, outside)
        else:                                      # too little background: leave it, and count it
            flat = delta
            n_fallback += 1
        start = (tilt_h - out_h) // 2
        crop = flat[:, start:start + out_h] * valid_w[:, start:start + out_h]
        d = (Path(a.out_root) / label / "output_phase" / "channels" /
             "crop_sub_rawraw" / "z000" / f"ch{ch:02d}")
        d.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(d / f"img_{a.frame:09d}_ph_000.tif"), crop.astype(np.float32))
        n_crop += 1
    n_pos += 1

print(f"frame {a.frame}: {n_pos} Pos, {n_crop} crops -> {a.out_root}")
print(f"outside-channel area in the {roi['crop_w']}x{tilt_h} window: mean {100*np.mean(bg_frac):.1f} %"
      f" (min {100*np.min(bg_frac):.1f} %, max {100*np.max(bg_frac):.1f} %)")
print(f"channels with too little background to fit ({a.min_bg} px): {n_fallback}")
print(f"invalid (out of image / warped-in) area excluded from the fit: mean "
      f"{100*np.mean(oob_frac):.1f} %, max {100*np.max(oob_frac):.1f} %")
