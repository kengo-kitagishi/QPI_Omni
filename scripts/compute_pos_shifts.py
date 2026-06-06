# %%
"""
compute_pos_shifts.py
---------------------
Compute per-frame shift amounts for a single Pos's channel stacks
(channel_XX*.tif) using ECC or phase_correlation, then average
across channels with outlier removal and save to pos_shifts.json.
"""
import numpy as np
import tifffile
import cv2
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import median_filter
import concurrent.futures
import threading
import re as _re

from ecc_utils import (
    tilt_fit_crop, extract_rect_roi, ecc_align, mad, remove_outliers_mad,
    ECC_MIN_CORR,
    # Float ECC input: clipped float32, no 8-bit quantisation (removes the
    # uint8 X bias). The local to_uint8 wrapper and *_u8 names are kept, but
    # the data flowing into ecc_align is now float32.
    to_ecc_input as _to_uint8_fixed,
)

# ============================================================
# Configuration parameters
# ============================================================
CHANNELS_DIR = r"F:\260405\ph_260405\Pos1\output_phase\channels"
CHANNEL_PATTERN = "channel_*.tif"      # use "channel_*_bg_corr.tif" if backsub is done

# --- Reference image selection ---
# USE_GRID_REFERENCE = True  : Crop grid x+0_y+0 image as reference for each channel (recommended)
# USE_GRID_REFERENCE = False : Use REFERENCE_FRAME-th frame from timelapse as reference (legacy)
USE_GRID_REFERENCE  = True
GRID_DIR            = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1"
GRID_BASE_LABEL     = "Pos1"           # PosX part of PosX_x+0_y+0
POS_SPLIT           = 53              # Must match pos_split in drift_config
GRID_Z_INDEX        = 18              # img_000000000_ph_{Z_INDEX:03d}.tif
CHANNEL_ROIS_JSON   = r"F:\260405\ph_260405\Pos1\output_phase\channels\channel_rois.json"

REFERENCE_FRAME = 150                  # Used only when USE_GRID_REFERENCE=False (1-based)

ALIGNMENT_METHOD = 'ecc'              # 'ecc' or 'phase_correlation'
VMIN = -5.0
VMAX = 2.0                            # Normalization range for to_uint8 (affects ECC accuracy)
USE_PERCENTILE_NORM = False
PERCENTILE_LO       = 5
PERCENTILE_HI       = 95
OUTLIER_MAD_THRESH = 5.0              # MAD threshold for inter-channel outlier removal
OUTLIER_TIMESERIES_WINDOW = 11        # Median filter width for timeseries outlier detection (odd)
OUTLIER_TIMESERIES_THRESH = 0.0       # Timeseries MAD threshold (0 to disable)
# ECC_MIN_CORR imported from ecc_utils (single source = 0.99). Exclude channels
# with ECC score below this -> cell-free average, removing the glucose-dependent
# cell-channel ECC bias. Override via cps.ECC_MIN_CORR = ... in batch drivers.
OUTPUT_JSON = "pos_shifts_cal.json"

# --- Apply gaussian_backsub to grid reference image ---
# When True, apply the same backsub as timelapse to grid reference images
APPLY_BACKSUB_TO_GRID_REF = True
BACKSUB_MIN_PHASE   = -1.1
BACKSUB_HIST_MIN    = -1.1
BACKSUB_HIST_MAX    =  1.5
BACKSUB_N_BINS      = 512
BACKSUB_SMOOTH_WINDOW = 20

# --- Incremental tracking mode ---
# When True, select nearest grid(xi,yi) as reference based on previous frame's shift
USE_INCREMENTAL_TRACKING   = True    # Default True (overridden by pipeline later)
X_STEP                     = 0.1    # Grid step [um]
Y_STEP                     = 0.1    # Grid step [um]
SHIFT_SIGN_X               = 1      # Shift sign (1 or -1)
SHIFT_SIGN_Y               = 1
JUMP_THRESH_UM             = 1.0   # Outlier if shift diff from previous frame exceeds this [um] (0 to disable)
MAX_FRAMES                 = None # For test runs: None for all frames, integer for first N frames only
# Optical parameters (for pixel scale calculation)
SENSOR_PIXEL_SIZE          = 3.45e-6  # [m]
MAGNIFICATION              = 40
ORIGINAL_DIM               = 2048
RECONSTRUCTED_DIM          = 511
# Grid calibration (output JSON from calibrate_grid_positions.py)
# None -> use nominal values (xi*X_STEP/pixel_scale_um)
GRID_CALIBRATION_JSON      = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1\grid_calibration_Pos1.json"
# --- 2-stage ECC (active only when USE_INCREMENTAL_TRACKING=True) ---
USE_SECOND_PASS_ECC    = True    # True to enable 2nd ECC pass
FIRST_PASS_HALF        = False   # (disabled: all passes use full crop)
SECOND_PASS_HALF       = 'right' # (disabled: unified to full crop)
USE_THIRD_PASS_ECC     = True    # True for 3rd ECC pass (re-select nearest grid from pass2 result -> half ECC)
# Save corr/shift data as NPZ + CSV (True recommended: for verifying correspondence with subtracted images)
SAVE_CORR_DATA         = True
# Save shift_visualize figures (several uploads to the shared Drive -> slow).
# Set False for multi-Pos batch runs; regenerate cumulative-drift plots later
# from pos_shifts_cal.json / *_corr_data.npz.
SAVE_SHIFT_FIGURES     = True
# When SAVE_SHIFT_FIGURES is True: save ONLY the fine_ecc figure per Pos
# (skip shift_timeseries / trajectory / pass1 / pass2 / pass1_vs_pass2 /
# exclusion). Batch runs use this to keep the one useful figure but stay fast.
SHIFT_FIGURES_FINE_ONLY = False
# --- Parallel processing ---
N_WORKERS = None               # None = os.cpu_count(). 1 = serial (for debugging)

# Timelapse z-index (used in slope correction mode)
TL_Z_INDEX = 0                 # img_*_ph_{Z:03d}_phase.tif

# ============================================================
# X-tilt correction (alternative to gaussian_backsub)
# When True, instead of reading channel stacks, take a TILT_CROP_H px wide
# crop from full phase images in output_phase/, fit slope+intercept on the
# left 1/3, and use the corrected central crop_h px for ECC.
# STEP_GAUSSIAN_BACKSUB becomes unnecessary.
# ============================================================
USE_SLOPE_CORRECTION = True    # True: no bg_corr needed, use full phase images directly
TILT_CROP_H          = 270     # Width for correction (left 1/3 or right 1/3 is the background fit region)
ECC_CROP_H           = 80      # X width of crop used for ECC (cut from center of TILT_CROP_H)
# ============================================================
_m           = _re.match(r"Pos(\d+)", GRID_BASE_LABEL)
_POS_NUM     = int(_m.group(1)) if _m else 1
TILT_FIT_RIGHT = _POS_NUM >= POS_SPLIT  # Pos<POS_SPLIT: left 1/3 fit, Pos>=POS_SPLIT: right 1/3 fit


def _tilt_correct(img_f64, cy, cx, crop_w, crop_h_out, fit_right: bool = False):
    """Thin wrapper around tilt_utils.tilt_fit_crop using module-level TILT_CROP_H.

    Returns None when the wide crop would go out of bounds. Callers must
    treat None as "this channel has no valid tilt-corrected crop" and skip
    it from any ECC median (no silent fallback).
    """
    return tilt_fit_crop(img_f64, cy, cx, crop_w, crop_h_out, TILT_CROP_H,
                         fit_right=fit_right)


def to_uint8(img, vmin=VMIN, vmax=VMAX):
    if USE_PERCENTILE_NORM:
        vmin = float(np.percentile(img, PERCENTILE_LO))
        vmax = float(np.percentile(img, PERCENTILE_HI))
    return _to_uint8_fixed(img, vmin, vmax)


def compute_backsub_offset(img: np.ndarray) -> float:
    """
    Gaussian-fit the background peak using the same method as gaussian_backsub,
    and return the correction offset (= -peak_mean). No file saving.
    img: float array (any shape)
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.optimize import curve_fit

    bin_edges = np.linspace(BACKSUB_HIST_MIN, BACKSUB_HIST_MAX, BACKSUB_N_BINS + 1)
    hist_counts, _ = np.histogram(img.flatten(), bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0]

    smoothed = uniform_filter1d(hist_counts, size=BACKSUB_SMOOTH_WINDOW, mode='nearest')
    smoothed = uniform_filter1d(smoothed, size=BACKSUB_SMOOTH_WINDOW, mode='nearest')

    valid_idx = np.where(bin_centers >= BACKSUB_MIN_PHASE)[0]
    max_search_idx = int(len(bin_centers) * 0.95)
    search_idx = valid_idx[valid_idx < max_search_idx]
    if len(search_idx) == 0:
        return 0.0

    peak_idx = search_idx[np.argmax(smoothed[search_idx])]
    peak_value = bin_centers[peak_idx]

    fit_width = 300
    s = max(0, peak_idx - fit_width)
    e = min(len(bin_centers), peak_idx + fit_width)
    x_data = bin_centers[s:e]
    y_data = smoothed[s:e]

    def gaussian(x, amp, mean, std):
        return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    try:
        p0 = [float(np.max(y_data)), peak_value, bin_width * 20]
        popt, _ = curve_fit(gaussian, x_data, y_data, p0=p0, maxfev=5000)
        _, mean_fit, _ = popt
        return float(-mean_fit)
    except Exception as ex:
        print(f"    [backsub] Gaussian fit failed ({ex}), using peak value as fallback")
        return float(-peak_value)


# ecc_align is imported from ecc_utils


def phase_align(ref_img, tl_img):
    """Return (shift_x, shift_y, correlation) using phase_cross_correlation. Returns None on failure."""
    from skimage import registration
    try:
        shift, error, _ = registration.phase_cross_correlation(
            ref_img, tl_img, upsample_factor=10
        )
        return float(shift[1]), float(shift[0]), float(1.0 - error)
    except Exception:
        return None


# mad and remove_outliers_mad are imported from ecc_utils


def detect_timeseries_outliers(shift_avg, window, thresh):
    """
    Rolling median-based outlier detection for timeseries shifts.
    window: median filter width (odd recommended)
    thresh: MAD multiplier (0 to flag all as false)
    Returns: bool array, shape=(n_frames,)
    """
    if thresh <= 0:
        return np.zeros(len(shift_avg), dtype=bool)
    arr = np.array(shift_avg, dtype=np.float64)
    smoothed = median_filter(arr, size=window, mode='reflect')
    residual = arr - smoothed
    md = mad(residual)
    if md == 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(residual) > thresh * md


def load_grid_refs(channels_dir, n_channels):
    """
    Load grid x+0_y+0 image, crop with each channel's ROI,
    and return a list of per-channel reference images.
    """
    # extract_rect_roi imported at top level from ecc_utils

    # Try candidates in priority order:
    #   1. output_phase/*_ph_ZZZ_phase.tif  (reconstructed by pipeline_full.py)
    #   2. output_phase/*_ph_ZZZ.tif        (legacy naming)
    #   3. *_ph_ZZZ.tif                     (unreconstructed raw image fallback)
    base_dir = Path(GRID_DIR) / f"{GRID_BASE_LABEL}_x+0_y+0"
    z_str = f"ph_{GRID_Z_INDEX:03d}"

    candidates = [
        base_dir / "output_phase" / f"img_000000000_{z_str}_phase.tif",
        base_dir / "output_phase" / f"img_000000000_{z_str}.tif",
        base_dir / f"img_000000000_{z_str}.tif",
    ]
    grid_ref_path = next((p for p in candidates if p.exists()), None)
    if grid_ref_path is None:
        raise FileNotFoundError(
            f"Grid reference image not found: {base_dir}\n"
            f"  Tried paths:\n" + "\n".join(f"    {p}" for p in candidates)
        )

    rois_path = Path(CHANNEL_ROIS_JSON)
    if not rois_path.exists():
        raise FileNotFoundError(f"channel_rois.json not found: {rois_path}")

    grid_img = tifffile.imread(str(grid_ref_path)).astype(np.float64)
    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)

    print(f"Grid reference image: {grid_ref_path}")
    print(f"  Grid image size: {grid_img.shape}")

    refs = []
    for ch in range(n_channels):
        roi = rois[ch] if ch < len(rois) else rois[-1]
        if USE_SLOPE_CORRECTION:
            cropped = _tilt_correct(grid_img, roi["cy"], roi["cx"], roi["crop_w"], ECC_CROP_H,
                                    fit_right=TILT_FIT_RIGHT)
            if cropped is None:
                print(f"  ch{ch:02d} tilt bounds NG (cx={roi['cx']}): skip from ECC")
                refs.append(None)
                continue
            print(f"  ch{ch:02d} tilt-corrected crop: {cropped.shape}")
        else:
            cropped = extract_rect_roi(grid_img, roi["cy"], roi["cx"], roi["crop_w"], ECC_CROP_H)
            if APPLY_BACKSUB_TO_GRID_REF:
                offset = compute_backsub_offset(cropped)
                cropped = cropped + offset
                print(f"  ch{ch:02d} ROI crop (full): {cropped.shape}  backsub offset={offset:+.4f} rad")
            else:
                print(f"  ch{ch:02d} ROI crop (full): {cropped.shape}")
        refs.append(cropped)

    return refs, str(grid_ref_path)


def load_grid_calibration(json_path):
    """
    Load grid_calibration.json and return a dict of (xi, yi) -> (cal_dx_px, cal_dy_px).

    calibrate_grid_positions.py saves actual_dx_px = -tx (content displacement),
    but compute_pos_shifts.py uses shift_x = +tx (raw ECC warp_matrix value).
    Sign is inverted on load to match conventions.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    cal = {}
    for entry in data.get("positions", []):
        cal[(entry["xi"], entry["yi"])] = (
            -entry["actual_dx_px"],   # actual_dx = -tx -> cal_dx = +tx (match shift_x convention)
            -entry["actual_dy_px"],
        )
    return cal


def scan_grid_positions(grid_dir, base_label):
    """Return a map of (xi, yi) -> folder_path."""
    import re
    grid_dir = Path(grid_dir)
    pattern = re.compile(rf"^{re.escape(base_label)}_x([+-]?\d+)_y([+-]?\d+)$")
    pos_map = {}
    for d in grid_dir.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            pos_map[(int(m.group(1)), int(m.group(2)))] = d
    return pos_map


def find_nearest_grid(pos_map, dx_um, dy_um, x_step, y_step):
    """Return the nearest (xi, yi) and distance."""
    best_key, best_dist = None, float('inf')
    for (xi, yi) in pos_map:
        dist = ((xi * x_step - dx_um) ** 2 + (yi * y_step - dy_um) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_key = (xi, yi)
    return best_key, best_dist


def load_grid_ref_mn(pos_map, xi, yi, rois, n_channels):
    """
    Return per-channel ROI crops from grid(xi, yi).
    Uses fixed (cx, cy) for cropping so results are directly comparable with pre-cropped stacks.
    """
    # extract_rect_roi imported at top level from ecc_utils
    pos_dir = pos_map[(xi, yi)]
    fname = f"img_000000000_ph_{GRID_Z_INDEX:03d}_phase.tif"
    path = pos_dir / "output_phase" / fname
    if not path.exists():
        raise FileNotFoundError(f"Grid image not found: {path}")
    grid_img = tifffile.imread(str(path)).astype(np.float64)
    refs_out = []
    for ch in range(n_channels):
        roi = rois[ch] if ch < len(rois) else rois[-1]
        if USE_SLOPE_CORRECTION:
            cropped = _tilt_correct(grid_img, roi["cy"], roi["cx"], roi["crop_w"], ECC_CROP_H,
                                    fit_right=TILT_FIT_RIGHT)
            # cropped is None => tilt bounds NG; keep placeholder so callers can skip.
        else:
            cropped = extract_rect_roi(grid_img, roi["cy"], roi["cx"], roi["crop_w"], ECC_CROP_H)
            if APPLY_BACKSUB_TO_GRID_REF:
                cropped = cropped + compute_backsub_offset(cropped)
        refs_out.append(cropped)
    return refs_out


def load_grid_ref_mn_half(pos_map, xi, yi, rois, n_channels):
    """Return per-channel ROI full crops from grid(xi, yi) (for 2-stage ECC)."""
    # extract_rect_roi imported at top level from ecc_utils
    pos_dir = pos_map[(xi, yi)]
    fname = f"img_000000000_ph_{GRID_Z_INDEX:03d}_phase.tif"
    path = pos_dir / "output_phase" / fname
    if not path.exists():
        raise FileNotFoundError(f"Grid image not found: {path}")
    grid_img = tifffile.imread(str(path)).astype(np.float64)
    refs_out = []
    for ch in range(n_channels):
        roi = rois[ch] if ch < len(rois) else rois[-1]
        if USE_SLOPE_CORRECTION:
            cropped = _tilt_correct(grid_img, roi["cy"], roi["cx"], roi["crop_w"], ECC_CROP_H,
                                    fit_right=TILT_FIT_RIGHT)
            # cropped is None => tilt bounds NG; keep placeholder so callers can skip.
        else:
            cropped = extract_rect_roi(grid_img, roi["cy"], roi["cx"], roi["crop_w"], ECC_CROP_H)
            if APPLY_BACKSUB_TO_GRID_REF:
                offset = compute_backsub_offset(cropped)
                cropped = cropped + offset
                print(f"  ch{ch:02d} ROI crop (full): {cropped.shape}  backsub offset={offset:+.4f} rad")
            else:
                print(f"  ch{ch:02d} ROI crop (full): {cropped.shape}")
        refs_out.append(cropped)
    return refs_out


def _select_nearest_grid(shift_x, shift_y, grid_cal, pos_map, pixel_scale_um):
    """Return nearest (xi, yi) from shift_x/y [px]."""
    if grid_cal:
        best_key, best_dist = None, float('inf')
        for key, (adx, ady) in grid_cal.items():
            if key not in pos_map:
                continue
            dist = ((adx - shift_x) ** 2 + (ady - shift_y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_key = key
        return best_key
    else:
        dx_um = SHIFT_SIGN_X * shift_y * pixel_scale_um
        dy_um = SHIFT_SIGN_Y * shift_x * pixel_scale_um
        (xi, yi), _ = find_nearest_grid(pos_map, dx_um, dy_um, X_STEP, Y_STEP)
        return xi, yi


def _get_grid_offset(xi, yi, grid_cal, pixel_scale_um):
    """Return content offset [px] of grid(xi, yi) relative to grid(0,0)."""
    if grid_cal and (xi, yi) in grid_cal:
        return grid_cal[(xi, yi)]
    return (SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um,
            SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um)


def _save_corr_npz_csv(records, channels_dir):
    """
    Save corr data as NPZ (all data) + CSV (per-frame summary).
    For verifying correspondence between subtracted images and corr values.
    """
    import csv as _csv
    channels_dir = Path(channels_dir)

    def _get(key):
        return np.array(
            [r.get(key) if r.get(key) is not None else np.nan for r in records],
            dtype=np.float32)

    save_dict = {
        "t":       np.array([r["t"]  for r in records], dtype=np.int32),
        "ch":      np.array([r["ch"] for r in records], dtype=np.int32),
        "shift_x": _get("shift_x"),
        "shift_y": _get("shift_y"),
        "corr":    _get("corr"),
        "grid_xi": np.array([r.get("grid_xi", 0) for r in records], dtype=np.int32),
        "grid_yi": np.array([r.get("grid_yi", 0) for r in records], dtype=np.int32),
        "failed":  np.array([r.get("failed", False) for r in records], dtype=bool),
    }
    has_2pass = any("pass1_corr" in r for r in records)
    has_3pass = any("pass3_corr" in r for r in records)
    if has_2pass:
        save_dict.update({
            "pass1_corr":    _get("pass1_corr"),
            "pass1_shift_x": _get("pass1_shift_x"),
            "pass1_shift_y": _get("pass1_shift_y"),
            "pass1_grid_xi": np.array([r.get("pass1_grid_xi", 0) for r in records], dtype=np.int32),
            "pass1_grid_yi": np.array([r.get("pass1_grid_yi", 0) for r in records], dtype=np.int32),
            "pass2_corr":    _get("pass2_corr"),
            "pass2_shift_x": _get("pass2_shift_x"),
            "pass2_shift_y": _get("pass2_shift_y"),
            "pass2_grid_xi": np.array([r.get("pass2_grid_xi", 0) for r in records], dtype=np.int32),
            "pass2_grid_yi": np.array([r.get("pass2_grid_yi", 0) for r in records], dtype=np.int32),
        })
    if has_3pass:
        save_dict.update({
            "pass3_corr":    _get("pass3_corr"),
            "pass3_shift_x": _get("pass3_shift_x"),
            "pass3_shift_y": _get("pass3_shift_y"),
            "pass3_grid_xi": np.array([r.get("pass3_grid_xi", 0) for r in records], dtype=np.int32),
            "pass3_grid_yi": np.array([r.get("pass3_grid_yi", 0) for r in records], dtype=np.int32),
        })

    npz_path = channels_dir / "pos_shifts_corr_data.npz"
    np.savez_compressed(str(npz_path), **save_dict)
    print(f"  [corr_data] NPZ saved: {npz_path}")

    # per-frame summary CSV
    csv_path = channels_dir / "pos_shifts_corr_summary.csv"
    frame_indices = sorted(set(r["t"] for r in records))
    cols = ["frame_index", "n_channels", "corr_mean", "corr_min", "corr_max", "grid_xi", "grid_yi"]
    if has_2pass:
        cols += ["pass1_corr_mean", "pass2_corr_mean"]
    if has_3pass:
        cols += ["pass3_corr_mean"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for t_idx in frame_indices:
            recs_t = [r for r in records
                      if r["t"] == t_idx and not r.get("failed", True) and r.get("corr") is not None]
            if not recs_t:
                continue
            corrs = [r["corr"] for r in recs_t]
            from collections import Counter
            xi_counts = Counter(r.get("grid_xi", 0) for r in recs_t)
            yi_counts = Counter(r.get("grid_yi", 0) for r in recs_t)
            row = {
                "frame_index": t_idx,
                "n_channels":  len(recs_t),
                "corr_mean":   round(float(np.mean(corrs)), 6),
                "corr_min":    round(float(np.min(corrs)), 6),
                "corr_max":    round(float(np.max(corrs)), 6),
                "grid_xi":     xi_counts.most_common(1)[0][0],
                "grid_yi":     yi_counts.most_common(1)[0][0],
            }
            if has_2pass:
                p1 = [r["pass1_corr"] for r in recs_t if r.get("pass1_corr") is not None]
                p2 = [r["pass2_corr"] for r in recs_t if r.get("pass2_corr") is not None]
                row["pass1_corr_mean"] = round(float(np.mean(p1)), 6) if p1 else ""
                row["pass2_corr_mean"] = round(float(np.mean(p2)), 6) if p2 else ""
            if has_3pass:
                p3 = [r["pass3_corr"] for r in recs_t if r.get("pass3_corr") is not None]
                row["pass3_corr_mean"] = round(float(np.mean(p3)), 6) if p3 else ""
            writer.writerow(row)
    print(f"  [corr_data] CSV saved: {csv_path}")


def _save_exclusion_summary_csv(frame_results, channels_dir):
    """
    Save per-frame excluded channel breakdown as CSV.
    columns: frame_index, n_total, n_excl_failed, n_excl_low_ecc, n_excl_mad, n_used
    """
    import csv as _csv
    channels_dir = Path(channels_dir)
    csv_path = channels_dir / "pos_shifts_exclusion_summary.csv"
    cols = ["frame_index", "n_total", "n_excl_failed", "n_excl_low_ecc", "n_excl_mad", "n_used"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in frame_results:
            pc = r.get("per_channel", [])
            n_total = len(pc)
            n_failed  = sum(1 for c in pc if c.get("exclude_reason") == "alignment_failed")
            n_low_ecc = sum(1 for c in pc if c.get("exclude_reason") == "low_ecc_score")
            n_mad     = sum(1 for c in pc if c.get("exclude_reason") == "channel_outlier_mad")
            n_used    = r.get("n_channels_used", 0)
            writer.writerow({
                "frame_index":    r["frame_index"],
                "n_total":        n_total,
                "n_excl_failed":  n_failed,
                "n_excl_low_ecc": n_low_ecc,
                "n_excl_mad":     n_mad,
                "n_used":         n_used,
            })
    print(f"  [exclusion_summary] CSV saved: {csv_path}")


def _frame_result_from_per_channel(t, per_channel):
    """
    Compute outlier removal and averaging from per_channel list, returning
    (frame_result, sx_avg, sy_avg). sx_avg=sy_avg=None when all channels fail.
    """
    valid = [c for c in per_channel if not c["excluded"]]
    if len(valid) == 0:
        return {
            "frame_index": t,
            "shift_x_avg": None,
            "shift_y_avg": None,
            "n_channels_used": 0,
            "n_channels_excluded_outlier": 0,
            "is_outlier_timeseries": False,
            "per_channel": per_channel
        }, None, None

    xs = np.array([c["shift_x"] for c in valid])
    ys = np.array([c["shift_y"] for c in valid])

    if len(valid) >= 3:
        outlier_x = remove_outliers_mad(xs.tolist(), OUTLIER_MAD_THRESH)
        outlier_y = remove_outliers_mad(ys.tolist(), OUTLIER_MAD_THRESH)
        is_outlier = outlier_x | outlier_y
    else:
        is_outlier = np.zeros(len(valid), dtype=bool)

    for i, c in enumerate(valid):
        if is_outlier[i]:
            c["excluded"] = True
            c["exclude_reason"] = "channel_outlier_mad"

    used = [c for c in valid if not c["excluded"]]
    n_excluded = int(np.sum(is_outlier))

    if len(used) == 0:
        sx_avg = float(np.mean(xs))
        sy_avg = float(np.mean(ys))
        n_used = len(valid)
        n_excluded = 0
        for c in valid:
            c["excluded"] = False
            c["exclude_reason"] = None
    else:
        sx_avg = float(np.mean([c["shift_x"] for c in used]))
        sy_avg = float(np.mean([c["shift_y"] for c in used]))
        n_used = len(used)

    return {
        "frame_index": t,
        "shift_x_avg": sx_avg,
        "shift_y_avg": sy_avg,
        "n_channels_used": n_used,
        "n_channels_excluded_outlier": n_excluded,
        "is_outlier_timeseries": False,
        "per_channel": per_channel
    }, sx_avg, sy_avg


# ============================================================
# Frame-parallel incremental 3-pass ECC (ProcessPoolExecutor)
# ============================================================
_wp = {}  # worker-process shared data (set by _init_incr_worker)


def _init_incr_worker(stacks, p1_refs_u8, grid_refs_u8, grid_cal,
                      n_channels, ecc_min_corr, use_third_pass,
                      invalid_chs=None):
    """Initialize worker process with shared data."""
    global _wp
    _wp = {
        'stacks': stacks, 'p1': p1_refs_u8, 'grefs': grid_refs_u8,
        'gcal': grid_cal, 'avail': set(grid_refs_u8.keys()),
        'nch': n_channels, 'ecc_min_corr': ecc_min_corr,
        'use_third': use_third_pass,
        'invalid_chs': set(invalid_chs) if invalid_chs else set(),
    }


def _nearest_from_preloaded(shift_x, shift_y, grid_cal, available_keys):
    """Select nearest grid position from pre-loaded calibration data."""
    best_key, best_dist = None, float('inf')
    for key, (adx, ady) in grid_cal.items():
        if key not in available_keys:
            continue
        dist = ((adx - shift_x) ** 2 + (ady - shift_y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_key = key
    return best_key if best_key is not None else (0, 0)


def _incr_compute_frame(t):
    """Process one frame: all channels, 3-pass ECC.
    Returns (t, per_channel, corr_records)."""
    S = _wp
    per_channel = []
    corr_records = []

    for ch in range(S['nch']):
        if ch in S['invalid_chs']:
            per_channel.append({
                "channel": ch, "shift_x": None, "shift_y": None,
                "correlation": None, "excluded": True,
                "exclude_reason": "tilt_bounds_ng",
                "grid_xi": 0, "grid_yi": 0})
            corr_records.append({
                "t": t, "ch": ch, "shift_x": None, "shift_y": None,
                "corr": None, "grid_xi": 0, "grid_yi": 0, "failed": True})
            continue
        frame_u8 = to_uint8(S['stacks'][ch][t])

        # ---- Pass 1: grid(0,0) fixed ----
        result1 = ecc_align(S['p1'][ch], frame_u8)
        if result1 is None:
            per_channel.append({
                "channel": ch, "shift_x": None, "shift_y": None,
                "correlation": None, "excluded": True,
                "exclude_reason": "alignment_failed",
                "grid_xi": 0, "grid_yi": 0})
            corr_records.append({
                "t": t, "ch": ch, "shift_x": None, "shift_y": None,
                "corr": None, "grid_xi": 0, "grid_yi": 0, "failed": True})
            continue

        fine1_x, fine1_y, corr1 = result1
        shift1_x, shift1_y = fine1_x, fine1_y  # grid(0,0) offset is (0,0)

        # ---- Pass 2: nearest grid from pass1 ----
        xi2, yi2 = _nearest_from_preloaded(
            shift1_x, shift1_y, S['gcal'], S['avail'])
        gox2, goy2 = S['gcal'].get((xi2, yi2), (0.0, 0.0))

        if (xi2, yi2) not in S['grefs']:
            low_ecc = S['ecc_min_corr'] > 0 and corr1 < S['ecc_min_corr']
            per_channel.append({
                "channel": ch,
                "shift_x": shift1_x, "shift_y": shift1_y, "correlation": corr1,
                "excluded": low_ecc,
                "exclude_reason": "low_ecc_score" if low_ecc else None,
                "grid_xi": 0, "grid_yi": 0,
                "pass1_shift_x": shift1_x, "pass1_shift_y": shift1_y,
                "pass1_fine_x": fine1_x, "pass1_fine_y": fine1_y,
                "pass1_grid_offset_x": 0.0, "pass1_grid_offset_y": 0.0,
                "pass1_corr": corr1, "pass1_grid_xi": 0, "pass1_grid_yi": 0,
                "pass2_shift_x": None, "pass2_shift_y": None, "pass2_corr": None,
                "pass2_fine_x": None, "pass2_fine_y": None,
                "pass2_grid_offset_x": None, "pass2_grid_offset_y": None,
                "pass2_grid_xi": xi2, "pass2_grid_yi": yi2})
            corr_records.append({
                "t": t, "ch": ch,
                "shift_x": shift1_x, "shift_y": shift1_y, "corr": corr1,
                "grid_xi": 0, "grid_yi": 0, "failed": False,
                "pass1_corr": corr1, "pass1_grid_xi": 0, "pass1_grid_yi": 0,
                "pass2_corr": None, "pass2_grid_xi": xi2, "pass2_grid_yi": yi2})
            continue

        result2 = ecc_align(S['grefs'][(xi2, yi2)][ch], frame_u8)
        if result2 is None:
            low_ecc = S['ecc_min_corr'] > 0 and corr1 < S['ecc_min_corr']
            per_channel.append({
                "channel": ch,
                "shift_x": shift1_x, "shift_y": shift1_y, "correlation": corr1,
                "excluded": low_ecc,
                "exclude_reason": "low_ecc_score" if low_ecc else None,
                "grid_xi": 0, "grid_yi": 0,
                "pass1_shift_x": shift1_x, "pass1_shift_y": shift1_y,
                "pass1_fine_x": fine1_x, "pass1_fine_y": fine1_y,
                "pass1_grid_offset_x": 0.0, "pass1_grid_offset_y": 0.0,
                "pass1_corr": corr1, "pass1_grid_xi": 0, "pass1_grid_yi": 0,
                "pass2_shift_x": None, "pass2_shift_y": None, "pass2_corr": None,
                "pass2_fine_x": None, "pass2_fine_y": None,
                "pass2_grid_offset_x": gox2, "pass2_grid_offset_y": goy2,
                "pass2_grid_xi": xi2, "pass2_grid_yi": yi2})
            corr_records.append({
                "t": t, "ch": ch,
                "shift_x": shift1_x, "shift_y": shift1_y, "corr": corr1,
                "grid_xi": 0, "grid_yi": 0, "failed": False,
                "pass1_corr": corr1, "pass1_grid_xi": 0, "pass1_grid_yi": 0,
                "pass2_corr": None, "pass2_grid_xi": xi2, "pass2_grid_yi": yi2})
            continue

        fine2_x, fine2_y, corr2 = result2
        shift2_x = fine2_x + gox2
        shift2_y = fine2_y + goy2
        final_sx, final_sy = shift2_x, shift2_y
        final_corr = corr2
        final_xi, final_yi = xi2, yi2

        # ---- Pass 3 ----
        fine3_x = fine3_y = corr3 = None
        xi3, yi3 = xi2, yi2
        gox3, goy3 = gox2, goy2
        if S['use_third']:
            xi3, yi3 = _nearest_from_preloaded(
                shift2_x, shift2_y, S['gcal'], S['avail'])
            gox3, goy3 = S['gcal'].get((xi3, yi3), (0.0, 0.0))
            if (xi3, yi3) in S['grefs']:
                result3 = ecc_align(S['grefs'][(xi3, yi3)][ch], frame_u8)
                if result3 is not None:
                    fine3_x, fine3_y, corr3 = result3
                    final_sx = fine3_x + gox3
                    final_sy = fine3_y + goy3
                    final_corr = corr3
                    final_xi, final_yi = xi3, yi3

        low_ecc_corrs = [corr1, corr2]
        if S['use_third'] and corr3 is not None:
            low_ecc_corrs.append(corr3)
        low_ecc = (S['ecc_min_corr'] > 0
                   and any(c < S['ecc_min_corr'] for c in low_ecc_corrs))

        pc_entry = {
            "channel": ch,
            "shift_x": final_sx, "shift_y": final_sy,
            "correlation": final_corr,
            "excluded": low_ecc,
            "exclude_reason": "low_ecc_score" if low_ecc else None,
            "grid_xi": final_xi, "grid_yi": final_yi,
            "pass1_shift_x": shift1_x, "pass1_shift_y": shift1_y,
            "pass1_fine_x": fine1_x, "pass1_fine_y": fine1_y,
            "pass1_grid_offset_x": 0.0, "pass1_grid_offset_y": 0.0,
            "pass1_corr": corr1, "pass1_grid_xi": 0, "pass1_grid_yi": 0,
            "pass2_shift_x": shift2_x, "pass2_shift_y": shift2_y,
            "pass2_fine_x": fine2_x, "pass2_fine_y": fine2_y,
            "pass2_grid_offset_x": gox2, "pass2_grid_offset_y": goy2,
            "pass2_corr": corr2, "pass2_grid_xi": xi2, "pass2_grid_yi": yi2,
        }
        cr_entry = {
            "t": t, "ch": ch,
            "shift_x": final_sx, "shift_y": final_sy, "corr": final_corr,
            "grid_xi": final_xi, "grid_yi": final_yi, "failed": False,
            "pass1_corr": corr1, "pass1_grid_xi": 0, "pass1_grid_yi": 0,
            "pass2_corr": corr2, "pass2_grid_xi": xi2, "pass2_grid_yi": yi2,
        }
        if S['use_third']:
            pc_entry.update({
                "pass3_shift_x": final_sx if corr3 is not None else None,
                "pass3_shift_y": final_sy if corr3 is not None else None,
                "pass3_fine_x": fine3_x, "pass3_fine_y": fine3_y,
                "pass3_grid_offset_x": gox3, "pass3_grid_offset_y": goy3,
                "pass3_corr": corr3, "pass3_grid_xi": xi3, "pass3_grid_yi": yi3,
            })
            cr_entry.update({
                "pass3_corr": corr3,
                "pass3_grid_xi": xi3, "pass3_grid_yi": yi3,
            })

        per_channel.append(pc_entry)
        corr_records.append(cr_entry)

    return t, per_channel, corr_records


def main():
    channels_dir = Path(CHANNELS_DIR)
    if not channels_dir.exists():
        print(f"ERROR: CHANNELS_DIR not found: {channels_dir}")
        sys.exit(1)

    tilt_invalid_chs = set()  # channels whose tilt wide-crop goes OOB; skip from ECC.
    if USE_SLOPE_CORRECTION:
        # ===== Slope correction mode: build directly from full phase images in output_phase/ =====
        tl_dir = channels_dir.parent
        phase_paths = sorted(tl_dir.glob(f"img_*_ph_{TL_Z_INDEX:03d}_phase.tif"))
        if not phase_paths:
            print(f"ERROR: Phase images not found: {tl_dir}")
            sys.exit(1)
        rois_path_sc = Path(CHANNEL_ROIS_JSON)
        if not rois_path_sc.exists():
            print(f"ERROR: CHANNEL_ROIS_JSON not found: {rois_path_sc}")
            sys.exit(1)
        with open(rois_path_sc, encoding="utf-8") as f:
            rois_sc = json.load(f)
        n_channels = len(rois_sc)
        if MAX_FRAMES is not None:
            phase_paths = phase_paths[:MAX_FRAMES]
        n_frames = len(phase_paths)
        print(f"Channels: {n_channels}  Frames: {n_frames}  [slope correction mode / TILT_CROP_H={TILT_CROP_H}]")
        stacks = [np.zeros((n_frames, roi["crop_w"], ECC_CROP_H), dtype=np.float64) for roi in rois_sc]
        for t, pp in enumerate(tqdm(phase_paths, desc="slope-correct")):
            full_img = tifffile.imread(str(pp)).astype(np.float64)
            for ch, roi in enumerate(rois_sc):
                if ch in tilt_invalid_chs:
                    continue
                crop = _tilt_correct(full_img, roi["cy"], roi["cx"], roi["crop_w"], ECC_CROP_H,
                                     fit_right=TILT_FIT_RIGHT)
                if crop is None:
                    tilt_invalid_chs.add(ch)
                    print(f"  [tilt bounds NG] ch{ch:02d} cx={roi['cx']} tilt_crop_h={TILT_CROP_H} -> skip from ECC")
                    continue
                stacks[ch][t] = crop
    else:
        # ===== Legacy mode: read from channel stacks (bg_corr etc.) =====
        stacks_paths = sorted(channels_dir.glob(CHANNEL_PATTERN))
        if not stacks_paths:
            print(f"ERROR: No files matching {CHANNEL_PATTERN} found: {channels_dir}")
            sys.exit(1)
        print(f"Channels: {len(stacks_paths)}")
        for p in stacks_paths:
            print(f"  {p.name}")
        stacks = []
        for p in stacks_paths:
            arr = tifffile.imread(str(p))
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            stacks.append(arr.astype(np.float64))
            print(f"  {p.name}: shape={arr.shape}")
        n_frames = stacks[0].shape[0]
        n_channels = len(stacks)
        if MAX_FRAMES is not None and MAX_FRAMES < n_frames:
            stacks = [s[:MAX_FRAMES] for s in stacks]
            n_frames = MAX_FRAMES
            print(f"[TEST] Limiting frames to {n_frames}")
    print(f"\nFrames: {n_frames}")
    print(f"Alignment method: {ALIGNMENT_METHOD}")
    print(f"Outlier MAD threshold: {OUTLIER_MAD_THRESH}")
    if OUTLIER_TIMESERIES_THRESH > 0:
        print(f"Timeseries outlier: window={OUTLIER_TIMESERIES_WINDOW}, thresh={OUTLIER_TIMESERIES_THRESH}")
    else:
        print("Timeseries outlier detection: disabled")

    # Build reference images
    reference_info = {}
    if USE_GRID_REFERENCE or USE_INCREMENTAL_TRACKING:
        print(f"\nReference: grid x+0_y+0  ({GRID_BASE_LABEL})")
        try:
            refs, grid_ref_path_str = load_grid_refs(channels_dir, n_channels)
            reference_info = {
                "reference_type": "grid",
                "grid_dir": GRID_DIR,
                "grid_base_label": GRID_BASE_LABEL,
                "grid_z_index": GRID_Z_INDEX,
                "grid_reference_path": grid_ref_path_str,
            }
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        ref_idx = REFERENCE_FRAME - 1
        if ref_idx < 0 or ref_idx >= n_frames:
            print(f"ERROR: REFERENCE_FRAME={REFERENCE_FRAME} out of range (1~{n_frames})")
            sys.exit(1)
        print(f"\nReference: timelapse frame {REFERENCE_FRAME} (0-indexed: {ref_idx})")
        refs = [stacks[ch][ref_idx] for ch in range(n_channels)]
        reference_info = {
            "reference_type": "timelapse_frame",
            "reference_frame": REFERENCE_FRAME,
        }

    # Channels whose tilt wide-crop went out of bounds: exclude from all ECC averaging.
    invalid_chs = tilt_invalid_chs | {ch for ch, r in enumerate(refs) if r is None}
    if invalid_chs:
        print(f"[tilt bounds NG] Channels excluded from ECC: {sorted(invalid_chs)}")

    if ALIGNMENT_METHOD == 'ecc':
        refs_u8 = [None if r is None else to_uint8(r) for r in refs]

    # channel_rois.json: required for incremental grid-ref loading
    rois_for_incremental = None
    if USE_INCREMENTAL_TRACKING:
        rois_path = Path(CHANNEL_ROIS_JSON)
        if not rois_path.exists():
            print(f"ERROR: CHANNEL_ROIS_JSON not found: {rois_path}")
            sys.exit(1)
        with open(rois_path, encoding="utf-8") as f:
            rois_for_incremental = json.load(f)

    # Compute alignment per frame
    frame_results = []
    corr_records  = []   # Full record of corr/shift (for NPZ/CSV saving)
    pixel_scale_um = SENSOR_PIXEL_SIZE / MAGNIFICATION * ORIGINAL_DIM / RECONSTRUCTED_DIM * 1e6

    if USE_INCREMENTAL_TRACKING:
        pos_map = scan_grid_positions(GRID_DIR, GRID_BASE_LABEL)
        if not pos_map:
            print("ERROR: Grid Pos not found")
            sys.exit(1)
        print(f"[incremental] grid positions: {len(pos_map)}")

        # Load grid calibration
        grid_cal = {}
        if GRID_CALIBRATION_JSON:
            cal_path = Path(GRID_CALIBRATION_JSON)
            if cal_path.exists():
                grid_cal = load_grid_calibration(str(cal_path))
                print(f"[calibration] Loaded {len(grid_cal)} measured offsets: {cal_path}")
            else:
                print(f"[calibration] JSON not found: {cal_path}  -> using nominal values")

        _n_workers = N_WORKERS or os.cpu_count()

        # For pass1: grid(0,0) half or full crop (always fixed to grid(0,0) when USE_SECOND_PASS_ECC)
        if USE_SECOND_PASS_ECC:
            if FIRST_PASS_HALF:
                print(f"[pass1] Using grid(0,0) HALF crop ({SECOND_PASS_HALF} side)")
                p1_refs    = load_grid_ref_mn_half(pos_map, 0, 0, rois_for_incremental, n_channels)
                p1_refs_u8 = ([None if r is None else to_uint8(r) for r in p1_refs]
                              if ALIGNMENT_METHOD == 'ecc' else None)
            else:
                print(f"[pass1] Using grid(0,0) FULL crop (backsub offset already applied)")
                p1_refs    = refs
                p1_refs_u8 = refs_u8 if ALIGNMENT_METHOD == 'ecc' else None

        # Pre-load all grid references for frame-parallel processing
        print(f"[preload] Loading {len(pos_map)} grid positions...")
        all_grid_refs_u8 = {}
        for key in sorted(pos_map.keys()):
            try:
                refs_mn = load_grid_ref_mn_half(
                    pos_map, key[0], key[1], rois_for_incremental, n_channels)
                all_grid_refs_u8[key] = [None if r is None else to_uint8(r) for r in refs_mn]
            except FileNotFoundError:
                pass
        print(f"[preload] {len(all_grid_refs_u8)} / {len(pos_map)} loaded")

        # Frame-parallel ECC with ProcessPoolExecutor
        from concurrent.futures import ProcessPoolExecutor as _PPE
        print(f"[parallel] {n_frames} frames x {_n_workers} processes")

        with _PPE(
            max_workers=_n_workers,
            initializer=_init_incr_worker,
            initargs=(stacks, p1_refs_u8, all_grid_refs_u8, grid_cal,
                      n_channels, ECC_MIN_CORR, USE_THIRD_PASS_ECC,
                      invalid_chs),
        ) as pool:
            results_iter = pool.map(_incr_compute_frame, range(n_frames),
                                    chunksize=4)
            all_results = list(tqdm(results_iter, total=n_frames,
                                    desc="Processing frames"))

        # Collect results (pool.map returns in order)
        for t_idx, pc_list, cr_list in all_results:
            fr, _, _ = _frame_result_from_per_channel(t_idx, pc_list)
            frame_results.append(fr)
            corr_records.extend(cr_list)

        # Post-process: jump detection
        if JUMP_THRESH_UM > 0:
            prev_sx, prev_sy = 0.0, 0.0
            for fr in frame_results:
                sx = fr.get("shift_x_avg")
                sy = fr.get("shift_y_avg")
                if sx is not None and sy is not None:
                    jump = (((sx - prev_sx) * pixel_scale_um) ** 2 +
                            ((sy - prev_sy) * pixel_scale_um) ** 2) ** 0.5
                    if jump > JUMP_THRESH_UM:
                        fr["is_outlier_timeseries"] = True
                    else:
                        prev_sx, prev_sy = sx, sy

        # Save corr data
        if SAVE_CORR_DATA and corr_records:
            _save_corr_npz_csv(corr_records, channels_dir)

    else:
        # ---- Non-incremental: 1-pass ECC against grid(0,0), frame-parallel ----
        _n_workers = N_WORKERS or os.cpu_count()
        print(f"Parallel workers: {_n_workers} (non-incremental / frame-parallel, 1-pass)")

        def _run_frame(t):
            pc, cr = [], []
            for ch in range(n_channels):
                if ch in invalid_chs:
                    pc.append({"channel": ch, "shift_x": None, "shift_y": None,
                               "correlation": None, "excluded": True,
                               "exclude_reason": "tilt_bounds_ng"})
                    cr.append({"t": t, "ch": ch, "shift_x": None, "shift_y": None,
                               "corr": None, "failed": True})
                    continue
                frame = stacks[ch][t]

                if ALIGNMENT_METHOD == 'ecc':
                    res = ecc_align(refs_u8[ch], to_uint8(frame))
                else:
                    res = phase_align(refs[ch], frame)
                if res is None:
                    pc.append({"channel": ch, "shift_x": None, "shift_y": None,
                               "correlation": None, "excluded": True,
                               "exclude_reason": "alignment_failed"})
                    cr.append({"t": t, "ch": ch, "shift_x": None, "shift_y": None,
                               "corr": None, "failed": True})
                else:
                    sx, sy, corr = res
                    low_ecc = ECC_MIN_CORR > 0 and corr < ECC_MIN_CORR
                    pc.append({"channel": ch, "shift_x": sx, "shift_y": sy,
                               "correlation": corr, "excluded": low_ecc,
                               "exclude_reason": "low_ecc_score" if low_ecc else None})
                    cr.append({"t": t, "ch": ch, "shift_x": sx, "shift_y": sy,
                               "corr": corr, "failed": False})

            return pc, cr

        _frame_results_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=_n_workers) as ex:
            fs = {ex.submit(_run_frame, t): t for t in range(n_frames)}
            for fut in tqdm(concurrent.futures.as_completed(fs), total=n_frames, desc="Processing frames"):
                t = fs[fut]
                _frame_results_map[t] = fut.result()

        for t in range(n_frames):
            pc, cr = _frame_results_map[t]
            frame_result, _, _ = _frame_result_from_per_channel(t, pc)
            frame_results.append(frame_result)
            corr_records.extend(cr)

        # ---- Save corr data ----
        if SAVE_CORR_DATA and corr_records:
            _save_corr_npz_csv(corr_records, channels_dir)

    # Timeseries outlier detection
    avg_x = [r["shift_x_avg"] for r in frame_results]
    avg_y = [r["shift_y_avg"] for r in frame_results]
    # Interpolate linearly if None values are present before detection
    def fill_none(arr):
        arr = np.array([np.nan if v is None else v for v in arr], dtype=np.float64)
        nans = np.isnan(arr)
        if nans.all():
            return arr
        xp = np.where(~nans)[0]
        arr[nans] = np.interp(np.where(nans)[0], xp, arr[xp])
        return arr

    avg_x_filled = fill_none(avg_x)
    avg_y_filled = fill_none(avg_y)
    ts_outlier_x = detect_timeseries_outliers(avg_x_filled, OUTLIER_TIMESERIES_WINDOW, OUTLIER_TIMESERIES_THRESH)
    ts_outlier_y = detect_timeseries_outliers(avg_y_filled, OUTLIER_TIMESERIES_WINDOW, OUTLIER_TIMESERIES_THRESH)
    ts_outlier = ts_outlier_x | ts_outlier_y

    for i, r in enumerate(frame_results):
        r["is_outlier_timeseries"] = bool(ts_outlier[i])

    n_ts_outlier = int(np.sum(ts_outlier))
    print(f"\nTimeseries outlier frames: {n_ts_outlier} / {n_frames}")

    # Shift statistics
    valid_avg_x = [r["shift_x_avg"] for r in frame_results if r["shift_x_avg"] is not None]
    valid_avg_y = [r["shift_y_avg"] for r in frame_results if r["shift_y_avg"] is not None]
    if valid_avg_x:
        print(f"shift_x: mean={np.mean(valid_avg_x):.3f}, range=[{np.min(valid_avg_x):.3f}, {np.max(valid_avg_x):.3f}]")
        print(f"shift_y: mean={np.mean(valid_avg_y):.3f}, range=[{np.min(valid_avg_y):.3f}, {np.max(valid_avg_y):.3f}]")

    # Save JSON
    out = {
        "method": ALIGNMENT_METHOD,
        "n_channels": n_channels,
        "n_frames": n_frames,
        "outlier_mad_thresh": OUTLIER_MAD_THRESH,
        "outlier_timeseries_window": OUTLIER_TIMESERIES_WINDOW,
        "outlier_timeseries_thresh": OUTLIER_TIMESERIES_THRESH,
        "channels_dir": str(channels_dir),
        "channel_pattern": CHANNEL_PATTERN,
        **reference_info,
        # shift_visualize.py compatible fields (also hold average shift amounts in alignment_results format)
        "alignment_results": [
            {
                "filename": f"frame_{r['frame_index']:06d}",
                "shift_x": r["shift_x_avg"] if r["shift_x_avg"] is not None else 0.0,
                "shift_y": r["shift_y_avg"] if r["shift_y_avg"] is not None else 0.0,
                "correlation": None,
                "warp_matrix": [[1.0, 0.0, r["shift_x_avg"] or 0.0],
                                 [0.0, 1.0, r["shift_y_avg"] or 0.0]],
                "is_outlier_timeseries": r["is_outlier_timeseries"]
            }
            for r in frame_results
        ],
        "frame_results": frame_results
    }

    out_path = channels_dir / OUTPUT_JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # Save exclusion summary CSV
    _save_exclusion_summary_csv(frame_results, channels_dir)

    # Visualize with shift_visualize (slow: uploads several figures to the
    # shared Drive). Skipped in batch runs via SAVE_SHIFT_FIGURES=False.
    if SAVE_SHIFT_FIGURES:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            import shift_visualize as _sv
            from shift_visualize import visualize_shifts, visualize_2pass_shifts, visualize_exclusion_summary
            if SHIFT_FIGURES_FINE_ONLY:
                # Save only the fine_ecc figure (the one useful summary).
                _sv.SAVE_REDUNDANT_PASS_FIGS = False
                if USE_INCREMENTAL_TRACKING and USE_SECOND_PASS_ECC:
                    visualize_2pass_shifts(str(out_path))
                else:
                    visualize_shifts(str(out_path))
            else:
                visualize_shifts(str(out_path))
                if USE_INCREMENTAL_TRACKING and USE_SECOND_PASS_ECC:
                    visualize_2pass_shifts(str(out_path))
                excl_csv = channels_dir / "pos_shifts_exclusion_summary.csv"
                if excl_csv.exists():
                    visualize_exclusion_summary(str(excl_csv), str(out_path))
        except Exception as e:
            print(f"[shift_visualize] Skipped: {e}")
    else:
        print("[shift_visualize] SAVE_SHIFT_FIGURES=False -> skipping figures")



if __name__ == "__main__":
    main()

# %%
