# %%
"""
correct_0pergluc.py
-------------------
Post-processing script for 0% glucose correction.

Runs AFTER grid_subtract.py. Reads the grid-subtracted output,
computes delta_full = aligned(grid_0per) - grid_2per for the
(x+0, y+0) grid point, and subtracts the tilt-corrected delta
from every frame in the 0% glucose period.

Per frame, delta_full (2per grid(0,0) coordinates) is warped by
apply_inverse_shift_warp with (cal(xi,yi) - cal(0,0)) from the same
grid calibration as grid_subtract — not timelapse residual_x_px.
Then the same cal(xi,yi) crop as grid_subtract raw-raw, tilt, subtract.

The delta represents the medium RI difference between 0% and 2%
glucose conditions. Since grid_subtract.py uses grid_2pergluc
as reference, frames acquired under 0% glucose retain this
medium RI offset, which this script removes.

Usage:
  1. Run grid_subtract.py as usual (no changes needed).
  2. Set GLUCOSE_0_START / GLUCOSE_0_END below.
  3. 複数 Pos: PH_SESSION_ROOT + POS_NUMBERS_TO_RUN（空でなければ一括）。
     単一 Pos: POS_NUMBERS_TO_RUN = [] とし OUTPUT_DIR 等を指定。
  4. Run:  python scripts/correct_0pergluc.py
"""
import numpy as np
import tifffile
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from grid_subtract import (
    extract_rect_roi,
    apply_inverse_shift_warp,
    load_grid_calibration,
    scan_grid_positions,
)
from compute_pos_shifts import compute_backsub_offset, to_uint8, ecc_align


def _cal_dx_dy(xi, yi, grid_cal, pixel_scale_um, x_step_um, y_step_um,
               shift_sign_x, shift_sign_y):
    """grid_subtract.py と同じ規則で (cal_dx, cal_dy) を返す。"""
    if grid_cal and (xi, yi) in grid_cal:
        return grid_cal[(xi, yi)]
    if pixel_scale_um:
        cal_dx = shift_sign_y * yi * y_step_um / pixel_scale_um
        cal_dy = shift_sign_x * xi * x_step_um / pixel_scale_um
        return (cal_dx, cal_dy)
    return (0.0, 0.0)


# ============================================================
# Parameters
# ============================================================

# grid_subtract.py output directory (contains ch00/, ch01/, ...)
OUTPUT_DIR = r"D:\AquisitionData\Kitagishi\260405\ph_260405\Pos1\output_phase\channels\crop_sub_rawraw_0per_test"

# grid_subtract_log.json (for per-frame grid_xi, grid_yi)
GRID_SUB_LOG = r"D:\AquisitionData\Kitagishi\260405\ph_260405\Pos1\output_phase\channels\grid_subtract_log.json"

# channel_rois.json (for cx, cy, crop_w, crop_h)
CHANNEL_ROIS_JSON = r"D:\AquisitionData\Kitagishi\260405\ph_260405\Pos1\output_phase\channels\channel_rois.json"

# Grid directories
GRID_2PER_DIR = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1"
GRID_0PER_DIR = r"C:\grid_0pergluc_60ms_1"
BASE_LABEL    = "Pos1"

# Grid calibration JSON (for (xi,yi) -> (cal_dx, cal_dy) mapping)
GRID_CALIBRATION_JSON = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1\grid_calibration_Pos1.json"

# z-index デフォルト（grid_subtract_log に grid_z_index があればそちらを優先）
GRID_Z_INDEX = 18

# 0% glucose frame range [inclusive, exclusive)
GLUCOSE_0_START = 575    # frame index (inclusive)
GLUCOSE_0_END   = 1151   # frame index (exclusive)

# Tilt correction parameters
TILT_CROP_H_RAW = 270
ECC_CROP_H      = 80       # must match compute_pos_shifts.py
POS_SPLIT       = 33       # must match compute_pos_shifts.py

# Output crop height override (None -> use channel_rois.json crop_h)
OUTPUT_CROP_H = None

# --- 複数 Pos 一括（空リストなら下の単一路径設定のみ使用）---
PH_SESSION_ROOT = r"D:\AquisitionData\Kitagishi\260405\ph_260405"
# grid_subtract の出力サブフォルダ名（Pos1 と同じ構成を想定）
# Pos ごとに grid_subtract の出力フォルダ名が違う場合は実行前に合わせる
CHANNEL_OUTPUT_SUBDIR = "crop_sub_rawraw"
POS_NUMBERS_TO_RUN = [6]

# ECC の背景フィット側（compute_pos_shifts と揃える）
FIT_RIGHT = False

# grid_subtract_log に base_label が無いときのフォルダ名プレフィックス（例: Pos0_x*_y*）
GRID_POINTS_BASE_LABEL = "Pos0"

# ============================================================


def _load_grid_output_phase(pos_dir, z_index):
    """Read Pos0-subtracted phase from output_phase/. Error if not found."""
    path = Path(pos_dir) / "output_phase" / f"img_000000000_ph_{z_index:03d}_phase.tif"
    if not path.exists():
        raise FileNotFoundError(f"output_phase not found: {path}")
    return tifffile.imread(str(path)).astype(np.float64)


def _tilt_correct(img_f64, cy, cx, crop_w, crop_h_out, tilt_crop_h, fit_right=False):
    """
    Identical to compute_pos_shifts._tilt_correct.
    Extract tilt_crop_h-wide crop, fit slope on background 1/3,
    subtract linear trend, return center crop_h_out pixels.
    """
    h, w = img_f64.shape
    if (cx - tilt_crop_h // 2) < 0 or (cx + tilt_crop_h // 2) > w:
        raise ValueError(
            f"tilt crop out of bounds: cx={cx}, tilt_crop_h={tilt_crop_h}, w={w}"
        )
    big = extract_rect_roi(img_f64, cy, cx, crop_w, tilt_crop_h).astype(np.float64)
    x = np.arange(tilt_crop_h, dtype=np.float64)
    prof = big.mean(axis=0)
    fit_n = max(1, tilt_crop_h // 3)
    if fit_right:
        a, b = np.polyfit(x[-fit_n:], prof[-fit_n:], 1)
    else:
        a, b = np.polyfit(x[:fit_n], prof[:fit_n], 1)
    corrected = big - (a * x + b)[np.newaxis, :]
    start = (tilt_crop_h - crop_h_out) // 2
    return corrected[:, start : start + crop_h_out]


def tilt_correct_and_crop(img_large, out_crop_h, tilt_crop_h, fit_right=False):
    """
    Same pipeline as grid_subtract._raw_subtract_correct:
    2pi correction -> tilt correction -> center crop.

    Applied to delta_large so the correction matches
    what was done to the grid-subtracted output.
    """
    fit_n = max(1, tilt_crop_h // 3)

    # 1. 2pi correction
    if fit_right:
        bg_mean = float(np.mean(img_large[:, -fit_n:]))
    else:
        bg_mean = float(np.mean(img_large[:, :fit_n]))
    k = int(round(bg_mean / (2.0 * np.pi)))
    if k != 0:
        img_large = img_large - k * 2.0 * np.pi

    # 2. tilt correction
    x = np.arange(tilt_crop_h, dtype=np.float64)
    prof = img_large.mean(axis=0)
    if fit_right:
        a, b = np.polyfit(x[-fit_n:], prof[-fit_n:], 1)
    else:
        a, b = np.polyfit(x[:fit_n], prof[:fit_n], 1)
    img_large = img_large - (a * x + b)[np.newaxis, :]

    # 3. center crop
    start = (tilt_crop_h - out_crop_h) // 2
    return img_large[:, start : start + out_crop_h]


def _ecc_per_channel(g0_full, g2_full, rois, fit_right):
    """
    Per-channel ECC between two Pos0-subtracted full images.
    Pipeline matches compute_pos_shifts.py exactly:
      _tilt_correct -> compute_backsub_offset -> to_uint8 -> ecc_align
    Returns (median_tx, median_ty, median_corr, per_ch_details) or None.
    """
    tilt_h = TILT_CROP_H_RAW
    all_tx, all_ty, all_corr = [], [], []
    for ch, roi in enumerate(rois):
        g0_crop = _tilt_correct(g0_full, roi["cy"], roi["cx"],
                                roi["crop_w"], ECC_CROP_H, tilt_h, fit_right)
        g2_crop = _tilt_correct(g2_full, roi["cy"], roi["cx"],
                                roi["crop_w"], ECC_CROP_H, tilt_h, fit_right)
        g0_crop_bc = g0_crop + compute_backsub_offset(g0_crop)
        g2_crop_bc = g2_crop + compute_backsub_offset(g2_crop)
        result = ecc_align(to_uint8(g2_crop_bc), to_uint8(g0_crop_bc))
        if result:
            all_tx.append(result[0])
            all_ty.append(result[1])
            all_corr.append(result[2])
    if not all_tx:
        return None
    return (float(np.median(all_tx)), float(np.median(all_ty)),
            float(np.median(all_corr)),
            {"tx": all_tx, "ty": all_ty, "corr": all_corr})


def compute_delta_full(grid_0per_dir, grid_2per_dir, grid_points_label,
                       z_index, rois, fit_right):
    """
    Compute delta_full = aligned(grid_0per_best) - grid_2per (511x511).
    Uses output_phase (Pos0-subtracted) images — matching compute_pos_shifts.py.

    1. Read grid_2per output_phase {grid_points_label}_x+0_y+0.
    2. Coarse ECC grid_0per(0,0) vs grid_2per(0,0) to estimate displacement.
    3. Convert pixel displacement to grid coordinates -> find nearest grid_0per point.
    4. Fine ECC on the nearest grid_0per point (residual should be sub-pixel).
    5. Warp and subtract.

    Returns: (delta_full, ecc_info_dict)
    """
    # --- Read grid_2per reference (Pos0-subtracted) ---
    g2_pos = Path(grid_2per_dir) / f"{grid_points_label}_x+0_y+0"
    g2_full = _load_grid_output_phase(g2_pos, z_index)
    print(f"[2per] output_phase loaded: {g2_pos.name}  shape={g2_full.shape}")

    # --- Scan available grid_0per positions ---
    g0_pos_map = scan_grid_positions(grid_0per_dir, grid_points_label)
    if not g0_pos_map:
        raise FileNotFoundError(
            f"No grid_0per positions found: {grid_0per_dir}/{grid_points_label}_x*_y*"
        )
    print(f"[0per] grid positions available: {len(g0_pos_map)}")

    # --- Coarse ECC: grid_0per(0,0) vs grid_2per(0,0) ---
    if (0, 0) not in g0_pos_map:
        raise FileNotFoundError(
            f"grid_0per (0,0) not found in {grid_0per_dir}"
        )
    g0_00_full = _load_grid_output_phase(g0_pos_map[(0, 0)], z_index)

    coarse = _ecc_per_channel(g0_00_full, g2_full, rois, fit_right)
    if coarse is None:
        raise RuntimeError(
            "Coarse ECC failed: all channels returned None for grid_0per(0,0) vs grid_2per(0,0)"
        )
    coarse_tx, coarse_ty, coarse_corr, _ = coarse
    print(f"[coarse ECC] (0,0): tx={coarse_tx:.3f} ty={coarse_ty:.3f} "
          f"corr={coarse_corr:.4f}")

    # --- Convert pixel shift to grid indices ---
    from grid_subtract import SHIFT_SIGN_X, SHIFT_SIGN_Y, X_STEP, Y_STEP
    pixel_scale_um = (3.45e-6 / 40 * 2048 / 511) * 1e6  # sensor/mag/dim
    step_x_px = Y_STEP / pixel_scale_um  # grid yi -> image x
    step_y_px = X_STEP / pixel_scale_um  # grid xi -> image y

    best_yi = int(round(SHIFT_SIGN_Y * coarse_tx / step_x_px))
    best_xi = int(round(SHIFT_SIGN_X * coarse_ty / step_y_px))
    print(f"[grid search] coarse displacement -> best candidate: "
          f"({best_xi:+d}, {best_yi:+d})")

    # --- Search nearby grid points for best match ---
    search_radius = 2
    best_key = (best_xi, best_yi)
    best_residual = abs(coarse_tx) + abs(coarse_ty)
    best_ecc_result = None

    candidates = []
    for dxi in range(-search_radius, search_radius + 1):
        for dyi in range(-search_radius, search_radius + 1):
            key = (best_xi + dxi, best_yi + dyi)
            if key in g0_pos_map:
                candidates.append(key)

    if not candidates:
        raise RuntimeError(
            f"No grid_0per candidates found around ({best_xi:+d}, {best_yi:+d})"
        )

    print(f"[grid search] testing {len(candidates)} candidates "
          f"around ({best_xi:+d}, {best_yi:+d})...")

    for key in candidates:
        xi, yi = key
        g0_cand = _load_grid_output_phase(g0_pos_map[key], z_index)
        result = _ecc_per_channel(g0_cand, g2_full, rois, fit_right)
        if result:
            tx, ty, corr, details = result
            residual = abs(tx) + abs(ty)
            print(f"  ({xi:+d},{yi:+d}): tx={tx:+.3f} ty={ty:+.3f} "
                  f"residual={residual:.3f} corr={corr:.4f}")
            if residual < best_residual:
                best_residual = residual
                best_key = key
                best_ecc_result = (tx, ty, corr, details, g0_cand)

    # --- Use best grid point (no fallback) ---
    if best_ecc_result is None:
        raise RuntimeError(
            f"ECC failed for all {len(candidates)} grid_0per candidates "
            f"around ({best_xi:+d}, {best_yi:+d})"
        )

    tx, ty, corr, details, g0_best = best_ecc_result
    aligned_0per = apply_inverse_shift_warp(g0_best, tx, ty)

    ecc_info = {
        "tx": tx, "ty": ty, "corr": corr, "success": True,
        "grid_0per_point": list(best_key),
        "coarse_tx": coarse_tx, "coarse_ty": coarse_ty,
        "per_channel_tx": details["tx"],
        "per_channel_ty": details["ty"],
        "per_channel_corr": details["corr"],
    }
    print(f"[BEST] grid_0per({best_key[0]:+d},{best_key[1]:+d}): "
          f"tx={tx:.3f} ty={ty:.3f} corr={corr:.4f}")

    delta_full = aligned_0per - g2_full
    return delta_full, ecc_info


def run_correct_0pergluc(
    out_dir: Path,
    grid_sub_log: Path,
    channel_rois_json: Path,
    base_label: str,
    grid_calibration_json: Path,
):
    # --- Validate inputs ---
    if GLUCOSE_0_START is None or GLUCOSE_0_END is None:
        print("ERROR: GLUCOSE_0_START and GLUCOSE_0_END must be set.")
        sys.exit(1)
    if GLUCOSE_0_START >= GLUCOSE_0_END:
        print("ERROR: GLUCOSE_0_START must be < GLUCOSE_0_END.")
        sys.exit(1)

    if not out_dir.exists():
        print(f"ERROR: output dir not found: {out_dir}")
        sys.exit(1)

    log_path = Path(grid_sub_log)
    if not log_path.exists():
        print(f"ERROR: grid_subtract_log not found: {log_path}")
        sys.exit(1)

    rois_path = Path(channel_rois_json)
    if not rois_path.exists():
        print(f"ERROR: channel_rois.json not found: {rois_path}")
        sys.exit(1)

    # --- Load grid_subtract log and channel ROIs ---
    log_data = json.loads(log_path.read_text(encoding="utf-8"))
    frame_log = log_data["frame_log"]
    rois = json.loads(rois_path.read_text(encoding="utf-8"))
    n_channels = len(rois)

    print(f"Loaded grid_subtract_log: {len(frame_log)} frames, {n_channels} channels")
    grid_z_index = int(log_data.get("grid_z_index", GRID_Z_INDEX))
    print(f"[z] grid_z_index={grid_z_index} (from log, else GRID_Z_INDEX={GRID_Z_INDEX})")
    grid_points_label = log_data.get("base_label") or GRID_POINTS_BASE_LABEL
    print(
        f"[grid] folder prefix={grid_points_label} "
        f"(grid_subtract_log base_label, else GRID_POINTS_BASE_LABEL={GRID_POINTS_BASE_LABEL})"
    )
    print(f"0%% glucose range: [{GLUCOSE_0_START}, {GLUCOSE_0_END})")

    # --- Load grid calibration ---
    grid_cal = {}
    cal_path = Path(grid_calibration_json)
    if cal_path.exists():
        grid_cal = load_grid_calibration(str(cal_path))
        print(f"[calibration] Loaded {len(grid_cal)} grid positions")
    else:
        print(f"[WARNING] Grid calibration not found: {cal_path}")
        print("  -> Will use nominal offsets from grid_subtract_log")

    # --- Build file list per channel ---
    # grid_subtract.py saves files as ch{ch:02d}/{original_holo_name}.tif
    # frame_log order matches the output file order (sorted by filename).
    ch0_files = sorted((out_dir / "ch00").glob("*.tif"))
    if len(ch0_files) != len(frame_log):
        print(f"WARNING: file count ({len(ch0_files)}) != frame_log count ({len(frame_log)})")

    # Build filename list (same filenames across all channels)
    filenames = [f.name for f in ch0_files]

    # --- Compute delta_full ---
    delta_full, ecc_info = compute_delta_full(
        GRID_0PER_DIR,
        GRID_2PER_DIR,
        grid_points_label,
        grid_z_index,
        rois,
        FIT_RIGHT,
    )
    print(f"delta_full: shape={delta_full.shape}, mean={delta_full.mean():.6f} rad")

    # --- Identify target frames (0% glucose period) ---
    target_indices = []
    for idx, entry in enumerate(frame_log):
        fi = entry["frame_index"]
        if GLUCOSE_0_START <= fi < GLUCOSE_0_END:
            target_indices.append(idx)

    print(f"Frames to correct: {len(target_indices)}")
    if not target_indices:
        print("No frames in the specified range. Exiting.")
        sys.exit(0)

    # --- Pixel scale (for nominal offset fallback) ---
    pixel_scale_um = log_data.get("pixel_scale_um")
    x_step_um = log_data.get("x_step_um", 0.1)
    y_step_um = log_data.get("y_step_um", 0.1)
    shift_sign_x = log_data.get("shift_sign_x", -1)
    shift_sign_y = log_data.get("shift_sign_y", -1)

    cal_dx_00, cal_dy_00 = _cal_dx_dy(
        0, 0, grid_cal, pixel_scale_um, x_step_um, y_step_um,
        shift_sign_x, shift_sign_y,
    )
    print(f"[cal] grid(0,0): cal_dx={cal_dx_00:.4f} cal_dy={cal_dy_00:.4f} "
          f"(delta warp uses cal(xi,yi) - this)")

    # --- Save "before" snapshot for profile comparison ---
    sample_idx = target_indices[0]
    sample_fi = frame_log[sample_idx]["frame_index"]
    sample_ch = n_channels // 2
    before_imgs = {}
    for ch in range(n_channels):
        p = out_dir / f"ch{ch:02d}" / filenames[sample_idx]
        if p.exists():
            before_imgs[ch] = tifffile.imread(str(p)).astype(np.float64)
    print(f"[profile] 'before' snapshot: frame {sample_fi}, {len(before_imgs)} channels")

    # --- Main correction loop ---
    corrected_frames = []
    delta_warp_cache = {}

    for idx in tqdm(target_indices, desc="0per correction"):
        entry = frame_log[idx]
        fi = entry["frame_index"]
        xi = entry["grid_xi"]
        yi = entry["grid_yi"]

        cal_dx, cal_dy = _cal_dx_dy(
            xi, yi, grid_cal, pixel_scale_um, x_step_um, y_step_um,
            shift_sign_x, shift_sign_y,
        )

        gw_key = (xi, yi)
        if gw_key not in delta_warp_cache:
            d_warp_x = cal_dx - cal_dx_00
            d_warp_y = cal_dy - cal_dy_00
            delta_warp_cache[gw_key] = apply_inverse_shift_warp(
                delta_full, d_warp_x, d_warp_y,
            )
        delta_warped = delta_warp_cache[gw_key]

        for ch in range(n_channels):
            roi = rois[ch] if ch < len(rois) else rois[-1]
            cx, cy = roi["cx"], roi["cy"]
            crop_w = roi["crop_w"]
            crop_h = roi["crop_h"]
            out_crop_h = OUTPUT_CROP_H if OUTPUT_CROP_H is not None else crop_h

            # Same crop position as grid_subtract.py (L481-482)
            crop_cx = int(round(cx + cal_dx))
            crop_cy = int(round(cy + cal_dy))

            # Crop warped delta at the same position with TILT_CROP_H_RAW
            delta_large = extract_rect_roi(
                delta_warped, crop_cy, crop_cx, crop_w, TILT_CROP_H_RAW,
            )

            # Apply the same tilt correction pipeline
            delta_corrected = tilt_correct_and_crop(
                delta_large.copy(), out_crop_h, TILT_CROP_H_RAW,
            )

            # Read existing output, subtract delta, overwrite
            ch_dir = out_dir / f"ch{ch:02d}"
            tif_path = ch_dir / filenames[idx]
            if not tif_path.exists():
                print(f"  [WARNING] File not found: {tif_path}")
                continue

            img = tifffile.imread(str(tif_path)).astype(np.float64)

            if img.shape != delta_corrected.shape:
                print(f"  [WARNING] Shape mismatch at frame {fi} ch{ch:02d}: "
                      f"img={img.shape} delta={delta_corrected.shape}")
                continue

            img -= delta_corrected
            tifffile.imwrite(str(tif_path), img.astype(np.float32))

        corrected_frames.append(fi)

    # --- Save log ---
    correction_log = {
        "grid_0per_dir": str(GRID_0PER_DIR),
        "grid_2per_dir": str(GRID_2PER_DIR),
        "session_base_label": base_label,
        "grid_points_base_label": grid_points_label,
        "grid_z_index": grid_z_index,
        "grid_calibration_json": str(cal_path),
        "glucose_0_start": GLUCOSE_0_START,
        "glucose_0_end": GLUCOSE_0_END,
        "delta_warp_mode": "cal_diff_vs_grid00",
        "cal_grid00": {"cal_dx": cal_dx_00, "cal_dy": cal_dy_00},
        "note": "delta_full warped with apply_inverse_shift_warp(cal(xi,yi)-cal(0,0)); "
                "not timelapse residual_x_px",
        "unique_grid_cells_warped": len(delta_warp_cache),
        "ecc_tx": ecc_info["tx"],
        "ecc_ty": ecc_info["ty"],
        "ecc_corr": ecc_info["corr"],
        "ecc_success": ecc_info["success"],
        "delta_mean_rad": float(delta_full.mean()),
        "delta_std_rad": float(delta_full.std()),
        "tilt_crop_h_raw": TILT_CROP_H_RAW,
        "n_corrected_frames": len(corrected_frames),
        "corrected_frame_indices": corrected_frames,
    }

    log_out = out_dir / "correct_0pergluc_log.json"
    with open(log_out, "w", encoding="utf-8") as f:
        json.dump(correction_log, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Corrected {len(corrected_frames)} frames.")
    print(f"Log saved: {log_out}")

    # ==========================================================
    # Save delta TIF and center horizontal line profile
    # ==========================================================

    # --- Save delta_full as TIF ---
    delta_tif_path = out_dir / "delta_full.tif"
    tifffile.imwrite(str(delta_tif_path), delta_full.astype(np.float32))
    print(f"Saved: {delta_tif_path}")

    # --- Per-channel delta crops (at first 0% frame's grid position) ---
    entry0 = frame_log[sample_idx]
    xi0, yi0 = entry0["grid_xi"], entry0["grid_yi"]
    cal_dx0, cal_dy0 = _cal_dx_dy(
        xi0, yi0, grid_cal, pixel_scale_um, x_step_um, y_step_um,
        shift_sign_x, shift_sign_y,
    )
    delta_warped_sample = delta_warp_cache[(xi0, yi0)]

    delta_dir = out_dir / "delta_per_ch"
    delta_dir.mkdir(exist_ok=True)

    delta_crops = {}
    for ch in range(n_channels):
        roi = rois[ch]
        crop_cx = int(round(roi["cx"] + cal_dx0))
        crop_cy = int(round(roi["cy"] + cal_dy0))
        out_h = OUTPUT_CROP_H if OUTPUT_CROP_H is not None else roi["crop_h"]
        d_large = extract_rect_roi(
            delta_warped_sample, crop_cy, crop_cx, roi["crop_w"], TILT_CROP_H_RAW,
        )
        d_crop = tilt_correct_and_crop(d_large.copy(), out_h, TILT_CROP_H_RAW)
        tifffile.imwrite(
            str(delta_dir / f"delta_ch{ch:02d}.tif"), d_crop.astype(np.float32),
        )
        delta_crops[ch] = d_crop
    print(f"Saved per-channel delta TIFs: {delta_dir}")

    # --- Center horizontal line profile (before vs after) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figure_logger import save_figure

    after_imgs = {}
    for ch in before_imgs:
        p = out_dir / f"ch{ch:02d}" / filenames[sample_idx]
        after_imgs[ch] = tifffile.imread(str(p)).astype(np.float64)

    n_plot = min(n_channels, 6)
    plot_chs = np.linspace(0, n_channels - 1, n_plot, dtype=int)
    fig, axes = plt.subplots(2, n_plot, figsize=(4 * n_plot, 7),
                             squeeze=False)

    for j, ch in enumerate(plot_chs):
        mid = before_imgs[ch].shape[0] // 2
        # top: before / after profiles
        ax = axes[0, j]
        ax.plot(before_imgs[ch][mid, :], lw=0.8, label="Before")
        ax.plot(after_imgs[ch][mid, :], lw=0.8, label="After")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Phase (rad)")
        ax.set_title(f"ch{ch:02d}  frame {sample_fi}")
        ax.legend(fontsize=7)
        # bottom: delta profile
        ax2 = axes[1, j]
        d_mid = delta_crops[ch].shape[0] // 2
        ax2.plot(delta_crops[ch][d_mid, :], lw=0.8, color="C2")
        ax2.axhline(0, ls="--", lw=0.5, color="gray")
        ax2.set_xlabel("X (px)")
        ax2.set_ylabel("Delta (rad)")
        ax2.set_title(f"Delta ch{ch:02d}")

    fig.suptitle(
        f"0%% glucose correction — grid_0per{ecc_info['grid_0per_point']}, "
        f"ECC tx={ecc_info['tx']:.3f} ty={ecc_info['ty']:.3f}",
        fontsize=12,
    )
    fig.tight_layout()

    save_figure(
        fig,
        params={
            "sample_fi": int(sample_fi),
            "grid_0per_point": ecc_info["grid_0per_point"],
            "ecc_tx": ecc_info["tx"],
            "ecc_ty": ecc_info["ty"],
            "ecc_corr": ecc_info["corr"],
            "delta_mean_rad": float(delta_full.mean()),
        },
        description="0per correction: before/after center horizontal profile and delta per channel",
        data={
            "before_profiles": np.array(
                [before_imgs[ch][before_imgs[ch].shape[0] // 2, :] for ch in plot_chs]
            ),
            "after_profiles": np.array(
                [after_imgs[ch][after_imgs[ch].shape[0] // 2, :] for ch in plot_chs]
            ),
            "delta_profiles": np.array(
                [delta_crops[ch][delta_crops[ch].shape[0] // 2, :] for ch in plot_chs]
            ),
            "plot_channels": np.array(plot_chs),
        },
    )
    plt.close(fig)
    print("[profile] figure saved via figure_logger")


def main():
    session = Path(PH_SESSION_ROOT)
    grid2 = Path(GRID_2PER_DIR)

    if POS_NUMBERS_TO_RUN:
        for n in POS_NUMBERS_TO_RUN:
            label = f"Pos{n}"
            out = session / label / "output_phase" / "channels" / CHANNEL_OUTPUT_SUBDIR
            glog = session / label / "output_phase" / "channels" / "grid_subtract_log.json"
            crois = session / label / "output_phase" / "channels" / "channel_rois.json"
            gcal = grid2 / f"grid_calibration_{label}.json"
            print("\n" + "=" * 60)
            print(f"  correct_0pergluc - {label}")
            print("=" * 60)
            run_correct_0pergluc(out, glog, crois, label, gcal)
    else:
        run_correct_0pergluc(
            Path(OUTPUT_DIR),
            Path(GRID_SUB_LOG),
            Path(CHANNEL_ROIS_JSON),
            BASE_LABEL,
            Path(GRID_CALIBRATION_JSON),
        )


if __name__ == "__main__":
    main()

# %%
