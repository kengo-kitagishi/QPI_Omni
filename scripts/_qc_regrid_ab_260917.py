"""Same timelapse frame, two different grid references. Which half of the improvement is the grid?

test_2 looks better than test_1, but two things changed at once: the grid is minutes old
instead of 14 hours, and it is a single-z acquisition that took 1.1 h instead of an 11-z one
that took 12 h. This holds the TIMELAPSE fixed and swaps only the reference:

  new : test_2 frame - new single-z grid (z index 0)     = what the online run wrote
  old : test_2 frame - old 11-z merged grid (z index 5)  = same physical plane, 14 h older

If 'old' lands back at test_1's level, the reference is what matters. If 'old' stays as good
as 'new', the improvement came from the timelapse side, not from the grid.

Uses the production kernel (grid_subtract.process_single_frame) unchanged. Writes the
re-subtracted crops to D: so the acquisition disk is only read, never written.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frame", type=int, default=6)
ap.add_argument("--config", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config_zstack.json")
ap.add_argument("--log", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log_zstack.json")
ap.add_argument("--old-grid", default=r"D:\AquisitionData\Kitagishi\260917\grid_hologram_0p05")
ap.add_argument("--old-grid-z", type=int, default=5)
ap.add_argument("--out-root", default=r"D:\AquisitionData\Kitagishi\260917\_regrid_ab")
ap.add_argument("--pos", type=int, nargs="+", default=None)
ap.add_argument("--save-dir", default=None,
                help="timelapse save_dir to read frames from (default: the one in the config)")
ap.add_argument("--tag", default="", help="suffix for the output arm directories")
a = ap.parse_args()

cfg = json.load(open(a.config, encoding="utf-8"))
log = [e for e in json.load(open(a.log, encoding="utf-8")) if e.get("per_pos")]
entry = next((e for e in log if e["timepoint"] == a.frame), None)
if entry is None:
    sys.exit(f"frame {a.frame} is not in the drift log yet")

tilt_h = cfg.get("tilt_crop_h_raw", 270)
out_h = cfg.get("crop_sub_output_crop_h") or tilt_h
pos_split = int(cfg["pos_split"])
arms = {"new": (Path(cfg["grid_dir"]), int(cfg["raw_grid_z_index"])),
        "old": (Path(a.old_grid), a.old_grid_z)}

stats = {k: [] for k in arms}
for p in entry["positions"]:
    pos = int(re.match(r"Pos(\d+)", p["pos_label"]).group(1))
    if a.pos and pos not in a.pos:
        continue
    label, sx, sy = p["pos_label"], p["tx_avg_px"], p["ty_avg_px"]
    fit_right = pos >= pos_split
    tl_path = Path(a.save_dir or cfg["save_dir"]) / label / "z000" / "output_phase_raw" / \
        f"img_{a.frame:09d}_ph_000_phase.tif"
    if not tl_path.exists():
        continue
    tl_img = tifffile.imread(str(tl_path)).astype(np.float64)

    for arm, (grid_dir, grid_z) in arms.items():
        pos_map = gs.scan_grid_positions(grid_dir, label)
        cal_path = grid_dir / f"grid_calibration_{label}.json"
        if not pos_map or not cal_path.exists():
            continue
        grid_cal = gs.load_grid_calibration(str(cal_path))
        xi, yi, _, _, _, cal_dx, cal_dy, rx, ry = gs.select_grid(
            sx, sy, pos_map, grid_cal, pixel_scale_um=cfg["pixel_scale_um"],
            x_step=cfg.get("crop_sub_x_step_um", 0.05),
            y_step=cfg.get("crop_sub_y_step_um", 0.05),
            shift_sign_x=-1, shift_sign_y=-1)
        g_path = pos_map[(xi, yi)] / "output_phase_raw" / f"img_000000000_ph_{grid_z:03d}_phase.tif"
        if not g_path.exists():
            continue
        grid_img = tifffile.imread(str(g_path)).astype(np.float64)
        rois = json.load(open(grid_dir / f"{label}_x+0_y+0" / "output_phase" / "channels" /
                              "channel_rois.json", encoding="utf-8"))
        crops, _ = gs.process_single_frame(
            tl_img, sx, sy, rois, cal_dx, cal_dy, rx, ry, grid_img,
            output_crop_h_override=out_h, tilt_crop_h_raw=tilt_h, use_raw_phase=True,
            apply_subpixel_correction=True, fit_right=fit_right, apply_inverse_shift=False)
        for ch, crop in enumerate(crops):
            prof = crop.mean(axis=0)
            stats[arm].append(abs(prof[:20].mean() if fit_right else prof[-20:].mean()))
            d = Path(a.out_root) / (arm + a.tag) / label / "output_phase" / "channels" / \
                "crop_sub_rawraw" / "z000" / f"ch{ch:02d}"
            d.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(d / f"img_{a.frame:09d}_ph_000.tif"), crop.astype(np.float32))

print(f"frame {a.frame}: same timelapse frame, two references")
for arm in arms:
    v = np.array(stats[arm])
    if not len(v):
        print(f"  {arm}: nothing processed")
        continue
    print(f"  {arm:3s} grid {arms[arm][0].name} z{arms[arm][1]}: n={len(v)}  "
          f"|far-end| median {np.median(v):.4f}  p90 {np.percentile(v, 90):.4f}  "
          f"max {v.max():.3f}  >0.1 rad: {(v > 0.1).sum()}")
print(f"crops written under {a.out_root}\\<arm>\\PosN\\... (D:, so E: is only read)")
