"""Keep a spline5 copy of the running timelapse's crop_sub, frame by frame.

Same output as the online crop_sub_rawraw (same grid node, same residual, same tilt fit); the
only change is that the residual sub-pixel warp uses scipy.ndimage.shift(order=5) instead of
the production cv2 bilinear. Written to its own tree so it can be opened as a timelapse:

    <out-root>/PosN/output_phase/channels/crop_sub_rawraw/z000/chNN/img_*.tif

The validity mask keeps the production cv2 warp: process_single_frame warps an all-ones
float32 array with the same function and relies on cv2 filling outside with 0, so swapping
that too would change the mask as well as the interpolation.

Follows the acquisition: frames already written are skipped, new ones are picked up as they
appear. One frame of 99 Pos takes well under the ~2.5 min cycle, so it keeps up live.

    python run_spline5_live_260917.py --follow
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import shift as ndshift

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs  # noqa: E402

PROD_WARP = gs.apply_inverse_shift_warp


def warp_spline5(img, shift_x, shift_y):
    if np.asarray(img).dtype == np.float32:      # the ones-array validity mask
        return PROD_WARP(img, shift_x, shift_y)
    return ndshift(img.astype(np.float64), (-shift_y, -shift_x), order=5,
                   mode="nearest", prefilter=True)


def process_frame(cfg, entry, out_root, tilt_h, out_h, pos_split, grid_dir, grid_z, grid_cache):
    written = 0
    for p in entry["positions"]:
        pos = int(re.match(r"Pos(\d+)", p["pos_label"]).group(1))
        label, sx, sy = p["pos_label"], p["tx_avg_px"], p["ty_avg_px"]
        frame = entry["timepoint"]
        name = f"img_{frame:09d}_ph_000.tif"
        probe = (Path(out_root) / label / "output_phase" / "channels" /
                 "crop_sub_rawraw" / "z000" / "ch00" / name)
        if probe.exists():
            continue
        tl_path = Path(cfg["save_dir"]) / label / "z000" / "output_phase_raw" / \
            f"img_{frame:09d}_ph_000_phase.tif"
        if not tl_path.exists():
            continue
        if label not in grid_cache:
            pos_map = gs.scan_grid_positions(grid_dir, label)
            cal_path = grid_dir / f"grid_calibration_{label}.json"
            rois_path = grid_dir / f"{label}_x+0_y+0" / "output_phase" / "channels" / \
                "channel_rois.json"
            if not pos_map or not cal_path.exists() or not rois_path.exists():
                grid_cache[label] = None
            else:
                grid_cache[label] = (pos_map, gs.load_grid_calibration(str(cal_path)),
                                     json.loads(rois_path.read_text(encoding="utf-8")), {})
        if grid_cache[label] is None:
            continue
        pos_map, grid_cal, rois, img_cache = grid_cache[label]
        xi, yi, _, _, _, cal_dx, cal_dy, rx, ry = gs.select_grid(
            sx, sy, pos_map, grid_cal, pixel_scale_um=cfg["pixel_scale_um"],
            x_step=cfg.get("crop_sub_x_step_um", 0.05),
            y_step=cfg.get("crop_sub_y_step_um", 0.05), shift_sign_x=-1, shift_sign_y=-1)
        if (xi, yi) not in img_cache:
            g = pos_map[(xi, yi)] / "output_phase_raw" / f"img_000000000_ph_{grid_z:03d}_phase.tif"
            img_cache[(xi, yi)] = tifffile.imread(str(g)).astype(np.float64) if g.exists() else None
        grid_img = img_cache[(xi, yi)]
        if grid_img is None:
            continue

        tl_img = tifffile.imread(str(tl_path)).astype(np.float64)
        gs.apply_inverse_shift_warp = warp_spline5
        crops, _ = gs.process_single_frame(
            tl_img, sx, sy, rois, cal_dx, cal_dy, rx, ry, grid_img,
            output_crop_h_override=out_h, tilt_crop_h_raw=tilt_h, use_raw_phase=True,
            apply_subpixel_correction=True, fit_right=pos >= pos_split,
            apply_inverse_shift=False)
        gs.apply_inverse_shift_warp = PROD_WARP
        for ch, crop in enumerate(crops):
            d = (Path(out_root) / label / "output_phase" / "channels" /
                 "crop_sub_rawraw" / "z000" / f"ch{ch:02d}")
            d.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(d / name), crop.astype(np.float32))
            written += 1
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config_zstack.json")
    ap.add_argument("--log", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log_zstack.json")
    ap.add_argument("--out-root", default=r"E:\260917\online_crop_sub_zstack_test_6_spline5")
    ap.add_argument("--follow", action="store_true", help="keep picking up new frames")
    ap.add_argument("--poll-sec", type=int, default=60)
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    tilt_h = cfg.get("tilt_crop_h_raw", 270)
    out_h = cfg.get("crop_sub_output_crop_h") or tilt_h
    pos_split = int(cfg["pos_split"])
    grid_dir = Path(cfg["grid_dir"])
    grid_z = int(cfg["raw_grid_z_index"])
    grid_cache = {}
    print(f"spline5 copy of {cfg['save_dir']} -> {a.out_root}", flush=True)

    done = set()
    while True:
        log = [e for e in json.load(open(a.log, encoding="utf-8")) if e.get("per_pos")]
        todo = [e for e in log if e["timepoint"] not in done]
        for entry in todo:
            t0 = time.time()
            n = process_frame(cfg, entry, a.out_root, tilt_h, out_h, pos_split,
                              grid_dir, grid_z, grid_cache)
            done.add(entry["timepoint"])
            if n:
                print(f"T={entry['timepoint']}: {n} crops in {time.time()-t0:.0f} s", flush=True)
        if not a.follow:
            break
        time.sleep(a.poll_sec)


if __name__ == "__main__":
    main()
