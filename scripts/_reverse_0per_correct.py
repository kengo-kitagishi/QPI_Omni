"""
_reverse_0per_correct.py
------------------------
Reverse the 0per glucose correction for specified Pos directories.
Reads the saved correct_0pergluc_log.json and delta_full.tif,
then adds back the delta that was subtracted (img += delta_corrected).

Usage:
  python scripts/_reverse_0per_correct.py
"""
import numpy as np
import tifffile
import json
import re
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

from grid_subtract import (
    extract_rect_roi,
    apply_inverse_shift_warp,
    load_grid_calibration,
)
from correct_0pergluc import (
    _load_frame_log,
    _cal_dx_dy,
    POS_SPLIT,
    TILT_CROP_H_RAW,
    N_PARALLEL_FRAMES,
    GRID_POINTS_BASE_LABEL,
)
from ecc_utils import apply_2pi_tilt_crop

PH_SESSION_ROOT = r"D:\AquisitionData\Kitagishi\260508\online_crop_sub_zstack"
CHANNEL_OUTPUT_SUBDIR = "crop_sub_rawraw"
GRID_2PER_DIR = r"E:\260504\grid_2pergluc_1"

POS_TO_REVERSE = [1, 2, 3]


def reverse_one_pos(pos_num):
    label = f"Pos{pos_num}"
    session = Path(PH_SESSION_ROOT)
    out_dir = session / label / "output_phase" / "channels" / CHANNEL_OUTPUT_SUBDIR
    log_path = out_dir / "correct_0pergluc_log.json"

    if not log_path.exists():
        print(f"  SKIP {label}: no correct_0pergluc_log.json")
        return

    clog = json.loads(log_path.read_text(encoding="utf-8"))
    print(f"\n{'='*60}")
    print(f"  Reversing {label}")
    print(f"{'='*60}")

    delta_tif = out_dir / "delta_full.tif"
    if not delta_tif.exists():
        print(f"  ERROR: delta_full.tif not found: {delta_tif}")
        return
    delta_full = tifffile.imread(str(delta_tif)).astype(np.float64)
    print(f"  delta_full: shape={delta_full.shape}, mean={delta_full.mean():.6f} rad")

    glog_path = session / label / "output_phase" / "channels" / "grid_subtract_log.json"
    if not glog_path.exists():
        glog_path = session / label / "output_phase" / "channels" / "pos_shifts_cal_online.json"
    log_data = _load_frame_log(glog_path)
    frame_log = log_data["frame_log"]

    rois_path = session / label / "output_phase" / "channels" / "channel_rois.json"
    rois = json.loads(rois_path.read_text(encoding="utf-8"))
    n_channels = len(rois)

    gcal_path = Path(GRID_2PER_DIR) / f"grid_calibration_{label}.json"
    grid_cal = {}
    if gcal_path.exists():
        grid_cal = load_grid_calibration(str(gcal_path))

    g0_start = clog["glucose_0_start"]
    g0_end = clog["glucose_0_end"]
    fit_right_val = clog["fit_right"]

    pixel_scale_um = log_data.get("pixel_scale_um")
    x_step_um = log_data.get("x_step_um", 0.1)
    y_step_um = log_data.get("y_step_um", 0.1)
    shift_sign_x = log_data.get("shift_sign_x", -1)
    shift_sign_y = log_data.get("shift_sign_y", -1)

    cal_dx_00, cal_dy_00 = _cal_dx_dy(
        0, 0, grid_cal, pixel_scale_um, x_step_um, y_step_um,
        shift_sign_x, shift_sign_y,
    )

    target_indices = []
    for idx, entry in enumerate(frame_log):
        fi = entry["frame_index"]
        if g0_start <= fi < g0_end:
            target_indices.append(idx)

    ch0_files = sorted((out_dir / "ch00").glob("*.tif"))
    filenames = [f.name for f in ch0_files]

    print(f"  Frames to reverse: {len(target_indices)}")

    _cal_dxdy_by_key = {}
    delta_warp_cache = {}
    for idx in target_indices:
        entry = frame_log[idx]
        xi, yi = entry["grid_xi"], entry["grid_yi"]
        key = (xi, yi)
        if key not in _cal_dxdy_by_key:
            _cal_dxdy_by_key[key] = _cal_dx_dy(
                xi, yi, grid_cal, pixel_scale_um, x_step_um, y_step_um,
                shift_sign_x, shift_sign_y,
            )
    for key in set((frame_log[i]["grid_xi"], frame_log[i]["grid_yi"]) for i in target_indices):
        if key in delta_warp_cache:
            continue
        cal_dx, cal_dy = _cal_dxdy_by_key[key]
        d_warp_x = cal_dx - cal_dx_00
        d_warp_y = cal_dy - cal_dy_00
        delta_warp_cache[key] = apply_inverse_shift_warp(delta_full, d_warp_x, d_warp_y)

    reversed_count = [0]

    def _process_one(pos_in_targets):
        idx = target_indices[pos_in_targets]
        entry = frame_log[idx]
        fi = entry["frame_index"]
        xi, yi = entry["grid_xi"], entry["grid_yi"]
        cal_dx, cal_dy = _cal_dxdy_by_key[(xi, yi)]
        delta_warped = delta_warp_cache[(xi, yi)]

        for ch in range(n_channels):
            roi = rois[ch] if ch < len(rois) else rois[-1]
            crop_cx = int(round(roi["cx"] + cal_dx))
            crop_cy = int(round(roi["cy"] + cal_dy))

            tif_path = out_dir / f"ch{ch:02d}" / filenames[idx]
            if not tif_path.exists():
                continue

            img = tifffile.imread(str(tif_path)).astype(np.float64)
            if img.ndim != 2:
                continue

            out_crop_h = img.shape[1]
            delta_large = extract_rect_roi(
                delta_warped, crop_cy, crop_cx, roi["crop_w"], TILT_CROP_H_RAW,
            )
            delta_corrected = apply_2pi_tilt_crop(
                delta_large.copy(), out_crop_h, TILT_CROP_H_RAW, fit_right=fit_right_val,
            )
            if img.shape != delta_corrected.shape:
                continue

            img += delta_corrected
            tifffile.imwrite(str(tif_path), img.astype(np.float32))

        reversed_count[0] += 1

    with ThreadPoolExecutor(max_workers=N_PARALLEL_FRAMES) as ex:
        futures = [ex.submit(_process_one, p) for p in range(len(target_indices))]
        for f in tqdm(as_completed(futures), total=len(target_indices),
                      desc=f"reverse {label}"):
            f.result()

    print(f"  Reversed {reversed_count[0]} frames for {label}")

    log_path.unlink()
    print(f"  Removed: {log_path.name}")


def main():
    for n in POS_TO_REVERSE:
        reverse_one_pos(n)
    print("\nDone. All specified Pos reversed.")


if __name__ == "__main__":
    main()
