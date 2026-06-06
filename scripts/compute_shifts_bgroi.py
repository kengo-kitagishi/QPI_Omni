"""
compute_shifts_bgroi.py
-----------------------
Extract BG ROI from output_phase frames and compute shift amounts via 3-pass ECC.

3-pass ECC (same approach as compute_drift_online.py v2):
  Pass 1: grid(0,0) BG ROI vs TL BG ROI -> coarse shift (sx1, sy1)
  Pass 2: (sx1,sy1) -> nearest grid(xi2,yi2) -> its BG ROI vs TL BG ROI -> relative shift -> absolute shift
  Pass 3: Pass2 absolute shift -> nearest grid(xi3,yi3) -> ECC -> final absolute shift

Output: TL_DIR/output_phase/channels/pos_shifts.json

This file can be read directly by timelapse_plane_bgsub.py.
"""
import sys
import re
import json
import numpy as np
import tifffile
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.optimize import curve_fit

# ============================================================
# Configuration parameters
# ============================================================
TL_DIR    = Path(r"F:\timelapse_11day_exp200ms_1pos_EMM2\Pos1")
GRID_DIR  = Path(r"F:\grid_0p5_0p5_0p1_exp200ms_1pos_EMM2_1")
BASE_LABEL = "Pos1"
TL_Z_INDEX   = 0   # Timelapse z index
GRID_Z_INDEX = 5   # Grid reference z index

# ---- BG ROI (extract_rect_roi convention: crop_w=y-direction size, crop_h=x-direction size) ----
BG_CY     = 359   # y center (ch8: dark channel stripe with ECC contrast)
BG_CX     = 260   # x center (same as channel_rois.json)
BG_CROP_W = 40    # y-direction size [px] (same as channel_rois)
BG_CROP_H = 440   # x-direction size [px] (full channel length)

# ---- ECC parameters ----
ECC_VMIN     = -5.0
ECC_VMAX     =  2.0
ECC_MAX_ITER = 100000
ECC_EPSILON  = 1e-8

# ---- Coordinate conversion (for nearest grid selection) ----
# Image X(shift_x) <-> Stage Y(dy_um), Image Y(shift_y) <-> Stage X(dx_um)
SHIFT_SIGN_X = 1   # Adjust after verifying with real data (1 or -1)
SHIFT_SIGN_Y = 1
SENSOR_PIXEL_SIZE = 3.45e-6
MAGNIFICATION     = 40
ORIGINAL_DIM      = 2048
RECONSTRUCTED_DIM = 511
X_STEP = 0.1   # Grid step [um] (MicroManager setting, 0p1 in grid_0p5_0p5_0p1)
Y_STEP = 0.1

# ---- Backsub parameters ----
BACKSUB_MIN_PHASE    = -1.1
BACKSUB_HIST_MIN     = -1.1
BACKSUB_HIST_MAX     =  1.5
BACKSUB_N_BINS       = 512
BACKSUB_SMOOTH_WINDOW = 20

# ---- Timeseries outlier removal ----
OUTLIER_TIMESERIES_WINDOW = 11
OUTLIER_TIMESERIES_THRESH = 3.0   # 0 to disable

MAX_FRAMES = None   # For testing: None = all frames, integer = first N frames only
# ============================================================


# ---- Utilities ----

from ecc_utils import (
    extract_rect_roi,
    # Float ECC input (clipped float32, no 8-bit quantisation); local to_uint8
    # wrapper keeps its name but now returns float32 for ecc_align.
    to_ecc_input as _to_uint8_shared,
    ecc_align,
)


def compute_backsub_offset(img: np.ndarray) -> float:
    bin_edges = np.linspace(BACKSUB_HIST_MIN, BACKSUB_HIST_MAX, BACKSUB_N_BINS + 1)
    hist_counts, _ = np.histogram(img.flatten(), bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0]
    smoothed = uniform_filter1d(hist_counts.astype(float), size=BACKSUB_SMOOTH_WINDOW, mode='nearest')
    smoothed = uniform_filter1d(smoothed, size=BACKSUB_SMOOTH_WINDOW, mode='nearest')
    valid_idx = np.where(bin_centers >= BACKSUB_MIN_PHASE)[0]
    search_idx = valid_idx[valid_idx < int(len(bin_centers) * 0.95)]
    if len(search_idx) == 0:
        return 0.0
    peak_idx = search_idx[np.argmax(smoothed[search_idx])]
    peak_value = bin_centers[peak_idx]
    s = max(0, peak_idx - 300)
    e = min(len(bin_centers), peak_idx + 300)
    try:
        popt, _ = curve_fit(
            lambda x, a, m, s_: a * np.exp(-((x - m)**2) / (2 * s_**2)),
            bin_centers[s:e], smoothed[s:e],
            p0=[float(np.max(smoothed[s:e])), peak_value, bin_width * 20],
            maxfev=5000,
        )
        return float(-popt[1])
    except Exception:
        return float(-peak_value)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return _to_uint8_shared(img, ECC_VMIN, ECC_VMAX)


def crop_and_backsub(img: np.ndarray) -> np.ndarray:
    crop = extract_rect_roi(img, BG_CY, BG_CX, BG_CROP_W, BG_CROP_H).astype(np.float64)
    offset = compute_backsub_offset(crop)
    return crop + offset


def detect_timeseries_outliers(shifts: list, window: int, thresh: float) -> list:
    arr = np.array(shifts, dtype=np.float64)
    if thresh <= 0 or len(arr) < window:
        return [False] * len(arr)
    smoothed = median_filter(arr, size=window, mode='reflect')
    residual = arr - smoothed
    mad = float(np.median(np.abs(residual - np.median(residual))))
    if mad == 0:
        return [False] * len(arr)
    return [bool(abs(r) > thresh * mad) for r in residual]


# ---- Grid utilities ----

def scan_grid_positions(grid_dir: Path, base_label: str) -> dict:
    pattern = re.compile(rf"^{re.escape(base_label)}_x([+-]?\d+)_y([+-]?\d+)$")
    pos_map = {}
    for d in grid_dir.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                pos_map[(int(m.group(1)), int(m.group(2)))] = d
    return pos_map


def find_nearest_grid(pos_map: dict, dx_um: float, dy_um: float) -> tuple:
    best_key, best_dist = None, float('inf')
    for (xi, yi) in pos_map:
        dist = ((xi * X_STEP - dx_um) ** 2 + (yi * Y_STEP - dy_um) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_key = (xi, yi)
    return best_key


def shift_to_um(sx: float, sy: float, pixel_scale_um: float) -> tuple:
    """Image shift [px] -> stage coordinates (dx_um, dy_um)"""
    dx_um = SHIFT_SIGN_X * sy * pixel_scale_um   # Image Y -> Stage X
    dy_um = SHIFT_SIGN_Y * sx * pixel_scale_um   # Image X -> Stage Y
    return dx_um, dy_um


def grid_offset_px(xi: int, yi: int, pixel_scale_um: float) -> tuple:
    """Offset of grid(xi,yi) relative to grid(0,0) [image px]"""
    offset_x = SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um   # Stage Y -> Image X
    offset_y = SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um   # Stage X -> Image Y
    return offset_x, offset_y


def load_grid_crop(pos_map: dict, xi: int, yi: int, cache: dict) -> np.ndarray:
    """Return BG ROI (backsub applied) for grid(xi,yi) with caching"""
    key = (xi, yi)
    if key not in cache:
        pos_dir = pos_map[key]
        fname = f"img_000000000_ph_{GRID_Z_INDEX:03d}_phase.tif"
        img = tifffile.imread(str(pos_dir / "output_phase" / fname)).astype(np.float64)
        cache[key] = crop_and_backsub(img)
    return cache[key]


# ---- Main ----

def main():
    tl_dir = TL_DIR
    out_json = tl_dir / "output_phase" / "channels" / "pos_shifts.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Timelapse frame list
    tl_frames = sorted((tl_dir / "output_phase").glob(f"img_*_ph_{TL_Z_INDEX:03d}_phase.tif"))
    if not tl_frames:
        print(f"ERROR: No TL frames found"); sys.exit(1)
    n_frames = len(tl_frames)
    if MAX_FRAMES is not None:
        n_frames = min(MAX_FRAMES, n_frames)
        tl_frames = tl_frames[:n_frames]
        print(f"[TEST] Limiting to {n_frames} frames")
    print(f"TL frames: {n_frames}")

    # Grid scan
    pos_map = scan_grid_positions(GRID_DIR, BASE_LABEL)
    if (0, 0) not in pos_map:
        print(f"ERROR: grid(0,0) not found"); sys.exit(1)
    print(f"Grid positions: {len(pos_map)}")

    pixel_scale_um = SENSOR_PIXEL_SIZE / MAGNIFICATION * ORIGINAL_DIM / RECONSTRUCTED_DIM * 1e6
    print(f"Pixel scale: {pixel_scale_um:.4f} μm/px")

    # BG ROI info display
    print(f"BG ROI: cy={BG_CY}, cx={BG_CX}, y:{BG_CY-BG_CROP_W//2}-{BG_CY+BG_CROP_W//2}, x:{BG_CX-BG_CROP_H//2}-{BG_CX+BG_CROP_H//2}")

    # Grid BG ROI cache
    grid_cache: dict = {}

    # 3-pass ECC loop
    alignment_results = []
    shift_x_list = []
    shift_y_list = []

    for fp in tqdm(tl_frames, desc="3-pass ECC"):
        tl_img = tifffile.imread(str(fp)).astype(np.float64)
        tl_crop = crop_and_backsub(tl_img)
        tl_u8 = to_uint8(tl_crop)

        # ---- ECC: single pass using grid(0,0) BG ROI as reference ----
        # BG ROI is a flat background that looks the same at any grid position
        # -> relative shift from grid(0,0) directly gives the absolute shift
        ref = load_grid_crop(pos_map, 0, 0, grid_cache)
        res = ecc_align(to_uint8(ref), tl_u8)
        if res is None:
            alignment_results.append({"filename": fp.stem, "shift_x": None, "shift_y": None,
                                       "correlation": 0.0})
            shift_x_list.append(None); shift_y_list.append(None)
            continue
        sx_final, sy_final, corr_final = res

        alignment_results.append({
            "filename": fp.stem,
            "shift_x": sx_final,
            "shift_y": sy_final,
            "correlation": corr_final,
        })
        shift_x_list.append(sx_final)
        shift_y_list.append(sy_final)

    # ---- Timeseries outlier removal ----
    outlier_x = detect_timeseries_outliers(
        [v if v is not None else 0.0 for v in shift_x_list],
        OUTLIER_TIMESERIES_WINDOW, OUTLIER_TIMESERIES_THRESH
    )
    outlier_y = detect_timeseries_outliers(
        [v if v is not None else 0.0 for v in shift_y_list],
        OUTLIER_TIMESERIES_WINDOW, OUTLIER_TIMESERIES_THRESH
    )

    # ---- Generate frame_results ----
    frame_results = []
    for i, (sx, sy) in enumerate(zip(shift_x_list, shift_y_list)):
        is_outlier = outlier_x[i] or outlier_y[i] or sx is None
        frame_results.append({
            "shift_x_avg": sx if not is_outlier else None,
            "shift_y_avg": sy if not is_outlier else None,
            "shift_x_raw": sx,
            "shift_y_raw": sy,
            "is_outlier_timeseries": bool(is_outlier),
        })

    # ---- Statistics ----
    sx_arr = np.array([r["shift_x_avg"] for r in frame_results if r["shift_x_avg"] is not None])
    sy_arr = np.array([r["shift_y_avg"] for r in frame_results if r["shift_y_avg"] is not None])
    n_outliers = sum(1 for r in frame_results if r["is_outlier_timeseries"])
    print(f"\n--- Results ---")
    print(f"Frames: {n_frames}  Outliers: {n_outliers}")
    if len(sx_arr):
        print(f"shift_x: mean={sx_arr.mean():.3f}  std={sx_arr.std():.3f}  [{sx_arr.min():.3f}, {sx_arr.max():.3f}]")
        print(f"shift_y: mean={sy_arr.mean():.3f}  std={sy_arr.std():.3f}  [{sy_arr.min():.3f}, {sy_arr.max():.3f}]")

    # ---- Save JSON ----
    out_data = {
        "method": "ecc_bgroi_3pass",
        "n_frames": n_frames,
        "bg_roi": {
            "cy": BG_CY, "cx": BG_CX,
            "crop_w": BG_CROP_W, "crop_h": BG_CROP_H,
            "y_range": [BG_CY - BG_CROP_W // 2, BG_CY + BG_CROP_W // 2],
            "x_range": [BG_CX - BG_CROP_H // 2, BG_CX + BG_CROP_H // 2],
        },
        "shift_sign_x": SHIFT_SIGN_X,
        "shift_sign_y": SHIFT_SIGN_Y,
        "pixel_scale_um": pixel_scale_um,
        "outlier_timeseries_window": OUTLIER_TIMESERIES_WINDOW,
        "outlier_timeseries_thresh": OUTLIER_TIMESERIES_THRESH,
        "alignment_results": alignment_results,
        "frame_results": frame_results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"JSON saved: {out_json}")


if __name__ == "__main__":
    main()
