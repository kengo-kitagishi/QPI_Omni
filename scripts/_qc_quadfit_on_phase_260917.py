"""Background from the phase image itself: mask the walls, fit a quadratic to the rest.

No grid subtraction at all. For each Pos of one frame:

  1. read output_phase (the 511x511 phase the online run writes)
  2. mask every pixel with phase <= THRESH rad (the trap walls; 6 % of the frame at -2 rad)
     and grow that mask by DILATE px
  3. fit a 2D quadratic surface (1, c, r, c^2, r^2, c*r) on everything OUTSIDE the mask
  4. subtract it, then cut the same per-channel windows the pipeline uses, so the result can
     be looked at next to crop_sub in the same contact sheets

Channel windows come from the grid's channel_rois.json plus that frame's grid calibration
offset, the same centres process_single_frame uses. No residual warp is applied: nothing is
being aligned to a reference here.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs  # noqa: E402
from ecc_utils import extract_rect_roi  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config_zstack.json")
ap.add_argument("--log", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log_zstack.json")
ap.add_argument("--frame", type=int, default=4)
ap.add_argument("--thresh", type=float, default=-2.0, help="mask pixels at or below this phase [rad]")
ap.add_argument("--dilate", type=int, default=2, help="grow the mask by this many px")
ap.add_argument("--out-root", default=r"E:\260917\quadfit_test_6")
a = ap.parse_args()

cfg = json.load(open(a.config, encoding="utf-8"))
log = [e for e in json.load(open(a.log, encoding="utf-8")) if e.get("per_pos")]
entry = next((e for e in log if e["timepoint"] == a.frame), None)
if entry is None:
    sys.exit(f"frame {a.frame} is not in the drift log")
grid_dir = Path(cfg["grid_dir"])
tilt_h = cfg.get("tilt_crop_h_raw", 270)
out_h = cfg.get("crop_sub_output_crop_h") or tilt_h


def quad_background(img, mask):
    crop_w, out_w = img.shape
    c, r = np.meshgrid(np.arange(out_w) / out_w, np.arange(crop_w) / crop_w)
    terms = [np.ones_like(c), c, r, c * c, r * r, c * r]
    bg = ~mask
    A = np.stack([t[bg] for t in terms], axis=1)
    co, *_ = np.linalg.lstsq(A, img[bg], rcond=None)
    return sum(k * t for k, t in zip(co, terms))


n_pos = n_crop = 0
frac = []
for p in entry["positions"]:
    pos = int(re.match(r"Pos(\d+)", p["pos_label"]).group(1))
    label = p["pos_label"]
    ph_path = Path(cfg["save_dir"]) / label / "z000" / "output_phase" / \
        f"img_{a.frame:09d}_ph_000_phase.tif"
    rois_path = grid_dir / f"{label}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json"
    cal_path = grid_dir / f"grid_calibration_{label}.json"
    if not (ph_path.exists() and rois_path.exists()):
        continue
    img = tifffile.imread(str(ph_path)).astype(np.float64)
    mask = img <= a.thresh
    if a.dilate:
        mask = binary_dilation(mask, iterations=a.dilate)
    frac.append(mask.mean())
    if (~mask).sum() < 1000:
        print(f"{label}: only {int((~mask).sum())} background px, skipped")
        continue
    flat = img - quad_background(img, mask)

    grid_cal = gs.load_grid_calibration(str(cal_path)) if cal_path.exists() else {}
    pos_map = gs.scan_grid_positions(grid_dir, label)
    xi, yi, _, _, _, cal_dx, cal_dy, _, _ = gs.select_grid(
        p["tx_avg_px"], p["ty_avg_px"], pos_map, grid_cal,
        pixel_scale_um=cfg["pixel_scale_um"], x_step=cfg.get("crop_sub_x_step_um", 0.05),
        y_step=cfg.get("crop_sub_y_step_um", 0.05), shift_sign_x=-1, shift_sign_y=-1)

    for ch, roi in enumerate(json.loads(rois_path.read_text(encoding="utf-8"))):
        crop = extract_rect_roi(flat, int(round(roi["cy"] + cal_dy)),
                                int(round(roi["cx"] + cal_dx)), roi["crop_w"], out_h)
        d = (Path(a.out_root) / label / "output_phase" / "channels" /
             "crop_sub_rawraw" / "z000" / f"ch{ch:02d}")
        d.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(d / f"img_{a.frame:09d}_ph_000.tif"), crop.astype(np.float32))
        n_crop += 1
    n_pos += 1

print(f"frame {a.frame}: {n_pos} Pos, {n_crop} crops -> {a.out_root}")
print(f"mask (phase <= {a.thresh} rad, dilated {a.dilate} px) covers "
      f"{100*np.mean(frac):.1f} % of the frame on average "
      f"({100*np.min(frac):.1f}-{100*np.max(frac):.1f} %)")
