"""
generate_aligned_raw.py
-----------------------
Generate and save per-Pos aligned_raw stacks, which are the internal
intermediate products of grid_subtract.py (warped, before grid subtraction).

Pos1: Reuse existing grid_subtract_log.json, execute warp+crop only.
Pos0: Use Pos1's pos_shifts.json, run same grid selection+warp+crop.

Output: {pos}/output_phase/channels/grid_subtracted/channel_{ch:02d}_aligned_raw.tif
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import grid_subtract as gs

# ============================================================
# Settings
# ============================================================
BASE_DIR   = Path(r"E:\Acuisition\kitagishi\260301\movetest_9")
POS1_LABEL = "Pos1"

GRID_DIR   = r"E:\Acuisition\kitagishi\260301\multipos_test_1"
BASE_LABEL = "Pos4"

X_STEP = 0.05   # um
Y_STEP = 0.05   # um

SENSOR_PIXEL_SIZE = 3.45e-6
MAGNIFICATION     = 40
ORIGINAL_DIM      = 2048
RECONSTRUCTED_DIM = 511
SHIFT_SIGN_X      = 1
SHIFT_SIGN_Y      = 1
TL_Z_INDEX        = 0

# Target channel (None for all channels)
CHANNEL_INDEX = 1

MAX_FRAMES = None  # For testing: None for all frames

TARGET_POS = [
    {"pos": "Pos1", "use_log": True},   # Restore from grid_subtract_log.json
    {"pos": "Pos0", "use_log": False},  # Recalculate using Pos1 shifts
]
# ============================================================


def apply_warp(img: np.ndarray, rx: float, ry: float) -> np.ndarray:
    """Apply (-rx, -ry) affine warp (same as grid_subtract)."""
    h, w = img.shape
    M = np.array([[1.0, 0.0, -rx], [0.0, 1.0, -ry]], dtype=np.float32)
    return cv2.warpAffine(
        img.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR
    ).astype(np.float64)


def process_pos(
    pos_label: str,
    use_log: bool,
    pos1_channels: Path,
    grid_dir: Path,
    pixel_scale_um: float,
    channel_index,
    max_frames,
):
    pos_dir       = BASE_DIR / pos_label
    tl_phase_dir  = pos_dir / "output_phase"
    own_channels  = pos_dir / "output_phase" / "channels"

    # Use Pos-specific ROI (Pos0 also has channel_rois.json)
    rois_json = own_channels / "channel_rois.json"
    if not rois_json.exists():
        print(f"SKIP {pos_label}: channel_rois.json not found")
        return
    with open(rois_json, encoding="utf-8") as f:
        rois = json.load(f)

    ch_indices = (
        [channel_index] if channel_index is not None else list(range(len(rois)))
    )

    # List of timelapse frames
    pattern   = f"img_*_ph_{TL_Z_INDEX:03d}_phase.tif"
    tl_frames = sorted(tl_phase_dir.glob(pattern))
    if not tl_frames:
        print(f"SKIP {pos_label}: no frames in {tl_phase_dir}")
        return

    n_total = len(tl_frames)
    if max_frames:
        n_total = min(max_frames, n_total)

    # Output destination
    out_dir = own_channels / "grid_subtracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_stacks = {ch: [] for ch in ch_indices}

    if use_log:
        # -- Pos1: Restore residual and crop position from grid_subtract_log.json --
        log_path = pos1_channels / "grid_subtract_log.json"
        with open(log_path, encoding="utf-8") as f:
            log_data = json.load(f)

        ssX    = log_data.get("shift_sign_x", SHIFT_SIGN_X)
        ssY    = log_data.get("shift_sign_y", SHIFT_SIGN_Y)
        x_step = log_data.get("x_step_um",    X_STEP)
        y_step = log_data.get("y_step_um",    Y_STEP)
        pscale = log_data.get("pixel_scale_um", pixel_scale_um)

        frame_log = {e["frame_index"]: e for e in log_data["frame_log"]}

        for t in tqdm(range(n_total), desc=f"{pos_label} (log)"):
            entry = frame_log.get(t)
            if entry is None:
                for ch in ch_indices:
                    roi = rois[ch]
                    out_stacks[ch].append(
                        np.zeros((roi["crop_w"], roi["crop_h"]), dtype=np.float32)
                    )
                continue

            rx = entry["residual_x_px"]
            ry = entry["residual_y_px"]
            xi = entry["grid_xi"]
            yi = entry["grid_yi"]
            cal_dx = ssY * yi * y_step / pscale
            cal_dy = ssX * xi * x_step / pscale

            tl_img = tifffile.imread(str(tl_frames[t])).astype(np.float64)
            tl_warped = apply_warp(tl_img, rx, ry) if (rx or ry) else tl_img

            for ch in ch_indices:
                roi = rois[ch]
                crop_cx = int(round(roi["cx"] + cal_dx))
                crop_cy = int(round(roi["cy"] + cal_dy))
                crop = gs.extract_rect_roi(
                    tl_warped, crop_cy, crop_cx, roi["crop_w"], roi["crop_h"]
                )
                out_stacks[ch].append(crop.astype(np.float32))

    else:
        # -- Pos0: Use Pos1's pos_shifts.json for grid selection+warp+crop --
        shifts_json = pos1_channels / "pos_shifts.json"
        if not shifts_json.exists():
            print(f"SKIP {pos_label}: pos_shifts.json not found at {shifts_json}")
            return
        with open(shifts_json, encoding="utf-8") as f:
            shifts_data = json.load(f)
        frame_results = shifts_data.get("frame_results") or shifts_data.get(
            "alignment_results"
        )
        if max_frames:
            frame_results = frame_results[:max_frames]

        pos_map = gs.scan_grid_positions(grid_dir, BASE_LABEL)
        if not pos_map:
            print(f"SKIP {pos_label}: no grid positions found")
            return

        for t, r in tqdm(
            enumerate(frame_results), total=len(frame_results), desc=f"{pos_label} (shifts)"
        ):
            if t >= n_total:
                break

            sx = float(r.get("shift_x_avg") or r.get("shift_x") or 0.0)
            sy = float(r.get("shift_y_avg") or r.get("shift_y") or 0.0)

            dx_um = SHIFT_SIGN_X * sy * pixel_scale_um
            dy_um = SHIFT_SIGN_Y * sx * pixel_scale_um
            (xi, yi), _ = gs.find_nearest_grid(pos_map, dx_um, dy_um, X_STEP, Y_STEP)

            cal_dx     = SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um
            cal_dy     = SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um
            residual_x = SHIFT_SIGN_Y * sx - yi * Y_STEP / pixel_scale_um
            residual_y = SHIFT_SIGN_X * sy - xi * X_STEP / pixel_scale_um

            tl_img    = tifffile.imread(str(tl_frames[t])).astype(np.float64)
            tl_warped = (
                apply_warp(tl_img, residual_x, residual_y)
                if (residual_x or residual_y)
                else tl_img
            )

            for ch in ch_indices:
                roi = rois[ch]
                crop_cx = int(round(roi["cx"] + cal_dx))
                crop_cy = int(round(roi["cy"] + cal_dy))
                crop = gs.extract_rect_roi(
                    tl_warped, crop_cy, crop_cx, roi["crop_w"], roi["crop_h"]
                )
                out_stacks[ch].append(crop.astype(np.float32))

    # Save
    for ch in ch_indices:
        arr = np.array(out_stacks[ch], dtype=np.float32)
        out_path = out_dir / f"channel_{ch:02d}_aligned_raw.tif"
        tifffile.imwrite(str(out_path), arr, imagej=True)
        print(f"Saved: {out_path}  shape={arr.shape}")


def main():
    pixel_scale_um = (
        SENSOR_PIXEL_SIZE / MAGNIFICATION * ORIGINAL_DIM / RECONSTRUCTED_DIM * 1e6
    )
    print(f"Pixel scale: {pixel_scale_um:.4f} um/px")

    pos1_channels = BASE_DIR / POS1_LABEL / "output_phase" / "channels"

    for cfg in TARGET_POS:
        print(f"\n{'='*60}\n{cfg['pos']}\n{'='*60}")
        process_pos(
            pos_label=cfg["pos"],
            use_log=cfg["use_log"],
            pos1_channels=pos1_channels,
            grid_dir=Path(GRID_DIR),
            pixel_scale_um=pixel_scale_um,
            channel_index=CHANNEL_INDEX,
            max_frames=MAX_FRAMES,
        )

    print("\nDone")


if __name__ == "__main__":
    main()
