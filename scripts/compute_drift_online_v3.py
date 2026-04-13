"""
compute_drift_online_v3.py
--------------------------
Per-position drift correction with parallel processing.

Changes from v2:
  1. Each position computes drift independently against its own grid ref
  2. ThreadPoolExecutor processes positions concurrently
  3. drift_sample_interval groups positions; leader's correction is shared
  4. drift_state.txt has per-pos keys (CUMULATIVE_DX_UM_{pos_idx})

New config keys (add to drift_config.json):
  per_pos_correction: true     -- enable per-pos mode (required)
  drift_sample_interval: 1     -- 1 = all pos, N = every Nth pos
  max_drift_workers: 0         -- 0 = auto (cpu_count - 4)

Usage:
    python compute_drift_online_v3.py \\
        --timepoint 5 \\
        --config "drift_config.json"
"""

import sys
import os
import re
import json
import csv
import argparse
import numpy as np
import tifffile
import cv2
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Import shared utilities from v2 (same directory)
from compute_drift_online import (
    load_config,
    kf_step_posonly_nm,
    compute_backsub_offset,
    extract_rect_roi,
    _tilt_correct,
    to_uint8,
    ecc_align,
    _mad,
    _remove_outliers_mad,
    scan_grid_positions,
    _select_nearest_grid,
    _get_grid_offset_px,
    _load_grid_ref_full,
)


# ================================================================
# Argument parsing
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Per-position drift correction v3")
    p.add_argument("--timepoint", type=int, required=True)
    p.add_argument("--sample-raw", default=None,
                   help="Ignored in per-pos mode (v2 compat)")
    p.add_argument("--bg-raw", default="none",
                   help="Ignored in per-pos mode (v2 compat)")
    p.add_argument("--config", required=True)
    return p.parse_args()


# ================================================================
# Position / path helpers
# ================================================================

def load_positions_csv(csv_path):
    """Load positions.csv -> list of dicts with index, label, x, y."""
    positions = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            positions.append({
                "index": int(row[0]),
                "label": row[1].strip(),
                "x": float(row[2]),
                "y": float(row[3]),
            })
    return positions


def get_raw_path(save_dir, label, timepoint):
    return Path(save_dir) / label / f"img_{timepoint:09d}_ph_000.tif"


def _pos_index_from_label(label):
    """Extract numeric index from label, e.g. 'Pos5' -> 5."""
    m = re.search(r"\d+", label)
    return int(m.group()) if m else 0


# ================================================================
# Phase reconstruction (per-pos crop selection)
# ================================================================

def reconstruct_phase_for_pos(raw_path, cfg, bg_path=None, pos_index=0):
    """QPI phase reconstruction. Uses pos_index for crop selection."""
    script_dir = Path(cfg["script_dir"])
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from PIL import Image
    from qpi import QPIParameters, get_field
    from skimage.restoration import unwrap_phase
    from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

    pos_split = cfg.get("pos_split", 3)
    crop = tuple(cfg["crop_before"]) if pos_index < pos_split else tuple(cfg["crop_after"])
    rs, re_, cs, ce = crop

    def _recon(path):
        img = np.array(Image.open(str(path)))
        img_crop = img[rs:re_, cs:ce]
        qpi_params = QPIParameters(
            wavelength=WAVELENGTH, NA=NA,
            img_shape=img_crop.shape, pixelsize=PIXELSIZE,
            offaxis_center=OFFAXIS_CENTER,
        )
        return unwrap_phase(np.angle(get_field(img_crop, qpi_params)))

    phase = _recon(raw_path)
    if bg_path is not None and Path(bg_path).exists():
        phase = phase - _recon(bg_path)
    return phase


# ================================================================
# Grid reference management
# ================================================================

def load_grid_ref_crops_for_pos(pos_label, cfg, rois):
    """Load grid(0,0) reference crops for a position.
    Returns (float64_crops, uint8_crops) lists.
    """
    grid_dir = Path(cfg["grid_dir"])
    z_index = cfg.get("grid_z_index", 0)
    tilt_crop_h = cfg.get("tilt_crop_h", 0)
    ecc_crop_h = cfg.get("ecc_crop_h", 0)
    use_tilt = tilt_crop_h > 0 and ecc_crop_h > 0
    vmin = cfg.get("ecc_vmin", -5.0)
    vmax = cfg.get("ecc_vmax", 2.0)
    pos_split = cfg.get("pos_split", 3)
    pos_index = _pos_index_from_label(pos_label)
    fit_right = pos_index >= pos_split

    grid_ref_path = (grid_dir / f"{pos_label}_x+0_y+0" / "output_phase"
                     / f"img_000000000_ph_{z_index:03d}_phase.tif")
    if not grid_ref_path.exists():
        raise FileNotFoundError(f"Grid ref not found: {grid_ref_path}")

    grid_img = tifffile.imread(str(grid_ref_path)).astype(np.float64)

    gradient_sigma = cfg.get("gradient_sigma", 0)
    if gradient_sigma > 0:
        from scipy.ndimage import gaussian_filter
        grid_img = grid_img - gaussian_filter(grid_img, sigma=gradient_sigma, mode="nearest")

    f64_crops, u8_crops = [], []
    for roi in rois:
        if use_tilt:
            crop = _tilt_correct(grid_img, roi["cy"], roi["cx"], roi["crop_w"],
                                 tilt_crop_h, ecc_crop_h, fit_right=fit_right)
        else:
            crop = extract_rect_roi(grid_img, roi["cy"], roi["cx"],
                                    roi["crop_w"], roi["crop_h"])
            offset = compute_backsub_offset(crop, cfg)
            crop = crop + offset
        f64_crops.append(crop.astype(np.float64))
        u8_crops.append(to_uint8(crop, vmin, vmax))
    return f64_crops, u8_crops


def load_grid_cal_for_pos(pos_label, cfg):
    """Load grid calibration for a position. Returns dict or {}."""
    grid_dir = Path(cfg["grid_dir"])
    cal_path = grid_dir / f"grid_calibration_{pos_label}.json"
    if not cal_path.exists():
        return {}
    try:
        with open(cal_path, encoding="utf-8") as f:
            data = json.load(f)
        return {
            (e["xi"], e["yi"]): (-e["actual_dx_px"], -e["actual_dy_px"])
            for e in data.get("positions", [])
        }
    except Exception:
        return {}


# ================================================================
# Per-pos state management
# ================================================================

def read_per_pos_state(state_path, pos_idx):
    """Read previous per-pos state. Falls back to global keys for v2->v3."""
    result = {"cumulative_dx_um": 0.0, "cumulative_dy_um": 0.0,
              "ema_tx_px": None, "ema_ty_px": None}
    try:
        with open(state_path, encoding="utf-8") as f:
            per_pos_found = False
            g_dx = g_dy = 0.0
            g_ema_tx = g_ema_ty = None
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k == f"CUMULATIVE_DX_UM_{pos_idx}":
                    result["cumulative_dx_um"] = float(v); per_pos_found = True
                elif k == f"CUMULATIVE_DY_UM_{pos_idx}":
                    result["cumulative_dy_um"] = float(v)
                elif k == f"EMA_TX_PX_{pos_idx}":
                    result["ema_tx_px"] = float(v)
                elif k == f"EMA_TY_PX_{pos_idx}":
                    result["ema_ty_px"] = float(v)
                elif k == "CUMULATIVE_DX_UM":
                    g_dx = float(v)
                elif k == "CUMULATIVE_DY_UM":
                    g_dy = float(v)
                elif k == "EMA_TX_PX":
                    g_ema_tx = float(v)
                elif k == "EMA_TY_PX":
                    g_ema_ty = float(v)
            if not per_pos_found:
                result["cumulative_dx_um"] = g_dx
                result["cumulative_dy_um"] = g_dy
                result["ema_tx_px"] = g_ema_tx
                result["ema_ty_px"] = g_ema_ty
    except Exception:
        pass
    return result


def write_per_pos_state(state_path, t, pos_results, bg_pos_index):
    """Write drift_state.txt with per-position entries."""
    valid_results = [r for r in pos_results if r.get("valid") and not r.get("jump")]
    any_jump = any(r.get("jump", False) for r in pos_results)

    if valid_results:
        avg_dx = float(np.mean([r["cumulative_dx_um"] for r in valid_results]))
        avg_dy = float(np.mean([r["cumulative_dy_um"] for r in valid_results]))
    else:
        avg_dx = avg_dy = 0.0

    lines = [
        "# drift_state.txt - written by compute_drift_online_v3.py",
        f"STATUS={'correction_ready' if valid_results else 'correction_skipped'}",
        f"TIMEPOINT={t}",
        f"PER_POS=true",
        f"CUMULATIVE_DX_UM={avg_dx:.6f}",
        f"CUMULATIVE_DY_UM={avg_dy:.6f}",
        f"CORRECTION_VALID={'true' if valid_results and not any_jump else 'false'}",
        f"JUMP_DETECTED={'true' if any_jump else 'false'}",
    ]
    for r in pos_results:
        i = r["pos_idx"]
        lines.extend([
            f"CUMULATIVE_DX_UM_{i}={r['cumulative_dx_um']:.6f}",
            f"CUMULATIVE_DY_UM_{i}={r['cumulative_dy_um']:.6f}",
            f"DX_UM_{i}={r['dx_um']:.6f}",
            f"DY_UM_{i}={r['dy_um']:.6f}",
            f"EMA_TX_PX_{i}={r['ema_tx']:.6f}",
            f"EMA_TY_PX_{i}={r['ema_ty']:.6f}",
            f"CORRECTION_VALID_{i}={'true' if r['valid'] and not r['jump'] else 'false'}",
            f"ECC_CORRELATION_{i}={r['corr']:.6f}",
        ])
    lines.append(f"TIMESTAMP={datetime.now().isoformat()}")
    with open(state_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def load_per_pos_kf_state(kf_path, pos_label, R_nm2):
    """Load per-pos KF state from nested JSON."""
    default = {"kf_pos_tx_nm": 0.0, "kf_P_tx": R_nm2,
               "kf_pos_ty_nm": 0.0, "kf_P_ty": R_nm2}
    try:
        with open(kf_path, encoding="utf-8") as f:
            data = json.load(f)
        if pos_label in data and isinstance(data[pos_label], dict):
            return {k: data[pos_label].get(k, default[k]) for k in default}
        # v2 flat format fallback
        if "kf_pos_tx_nm" in data:
            for key in ("kf_P_tx", "kf_P_ty"):
                if key in data and isinstance(data[key], list):
                    data[key] = float(np.array(data[key])[0, 0])
            return {k: data.get(k, default[k]) for k in default}
        return default
    except Exception:
        return default


def save_all_kf_states(kf_path, kf_updates):
    """Save per-pos KF states. kf_updates: {pos_label: {...}}."""
    try:
        with open(kf_path, encoding="utf-8") as f:
            existing = json.load(f)
        if "kf_pos_tx_nm" in existing and not any(isinstance(v, dict) for v in existing.values()):
            existing = {}  # discard v2 flat format
    except Exception:
        existing = {}
    existing.update(kf_updates)
    with open(kf_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


# ================================================================
# Core: process one position
# ================================================================

def _process_one_position(pos_idx, pos_label, raw_path, bg_path,
                          cfg, rois, grid_ref_u8, pos_map, grid_cal,
                          prev_state, kf_state):
    """Full drift pipeline for one position.

    Returns dict with:
        pos_idx, pos_label, dx_um, dy_um, cumulative_dx_um, cumulative_dy_um,
        valid, jump, corr, ema_tx, ema_ty, kf_update, channel_details
    """
    t_start = datetime.now()
    n_channels = len(rois)
    vmin = cfg.get("ecc_vmin", -5.0)
    vmax = cfg.get("ecc_vmax", 2.0)
    ecc_min_corr = cfg.get("ecc_min_corr", 0.0)
    jump_thresh = cfg.get("jump_thresh_um", 1.0)
    max_total = cfg.get("max_total_corr_um", 15.0)
    pixel_scale_um = cfg.get("pixel_scale_um", 0.3462)
    sx_sign = cfg.get("shift_sign_x", 1)
    sy_sign = cfg.get("shift_sign_y", 1)
    x_step = cfg.get("x_step_um", 0.1)
    y_step = cfg.get("y_step_um", 0.1)
    grid_z_index = cfg.get("grid_z_index", 0)
    enable_third_pass = cfg.get("enable_third_pass", True)
    tilt_crop_h = cfg.get("tilt_crop_h", 0)
    ecc_crop_h = cfg.get("ecc_crop_h", 0)
    use_tilt = tilt_crop_h > 0 and ecc_crop_h > 0
    pos_split = cfg.get("pos_split", 3)
    fit_right = pos_idx >= pos_split
    ema_alpha = cfg.get("correction_ema_alpha", 1.0)
    use_kalman = cfg.get("use_kalman_filter", False)

    fail_result = {
        "pos_idx": pos_idx, "pos_label": pos_label,
        "dx_um": 0.0, "dy_um": 0.0,
        "cumulative_dx_um": prev_state["cumulative_dx_um"],
        "cumulative_dy_um": prev_state["cumulative_dy_um"],
        "valid": False, "jump": False, "corr": 0.0,
        "ema_tx": prev_state.get("ema_tx_px", 0.0) or 0.0,
        "ema_ty": prev_state.get("ema_ty_px", 0.0) or 0.0,
        "kf_update": None, "channel_details": [],
    }

    # ---- Phase reconstruction ----
    if not Path(raw_path).exists():
        print(f"  [{pos_label}] ERROR: raw image not found: {raw_path}")
        return fail_result

    try:
        phase = reconstruct_phase_for_pos(raw_path, cfg, bg_path, pos_idx)
    except Exception as ex:
        print(f"  [{pos_label}] ERROR: phase reconstruction failed: {ex}")
        return fail_result

    # Mean removal
    h_p, w_p = phase.shape
    region = phase[1:h_p - 1, 1:w_p // 2]
    if region.size > 0:
        phase -= np.mean(region)

    # Gradient removal
    gradient_sigma = cfg.get("gradient_sigma", 0)
    if gradient_sigma > 0:
        from scipy.ndimage import gaussian_filter
        phase = phase - gaussian_filter(phase, sigma=gradient_sigma, mode="nearest")

    # Save reconstructed phase
    out_dir = Path(raw_path).parent / "output_phase"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (Path(raw_path).stem + "_phase.tif")
    tifffile.imwrite(str(out_path), phase.astype(np.float32))

    # ---- Channel crops (tilt or backsub) ----
    current_crops = []
    for roi in rois:
        if use_tilt:
            crop = _tilt_correct(phase, roi["cy"], roi["cx"], roi["crop_w"],
                                 tilt_crop_h, ecc_crop_h, fit_right=fit_right)
        else:
            crop = extract_rect_roi(phase, roi["cy"], roi["cx"],
                                    roi["crop_w"], roi["crop_h"])
            offset = compute_backsub_offset(crop, cfg)
            crop = crop + offset
        current_crops.append(crop)

    # ---- Multi-pass ECC per channel ----
    # Grid cache local to this position (no threading issues)
    grid_half_cache = {}
    grid_half_u8_cache = {}

    tx_list, ty_list, corr_list = [], [], []
    valid_ch_indices = []
    channel_details = []

    for ch_idx in range(min(n_channels, len(current_crops))):
        ref_u8_p1 = grid_ref_u8[ch_idx] if ch_idx < len(grid_ref_u8) else grid_ref_u8[-1]
        cur_crop = current_crops[ch_idx]

        # Pass 1: grid(0,0)
        result1 = ecc_align(ref_u8_p1, to_uint8(cur_crop, vmin, vmax))
        if result1 is None:
            channel_details.append({"ch": ch_idx, "outlier": True, "status": "pass1_failed"})
            continue

        fine1_x, fine1_y, corr1 = result1
        shift1_x, shift1_y = fine1_x, fine1_y
        detail = {"tx1": shift1_x, "ty1": shift1_y, "corr1": corr1,
                  "xi": 0, "yi": 0}

        if not pos_map:
            # 1-pass only
            detail.update({"tx2": shift1_x, "ty2": shift1_y, "corr2": corr1, "status": "pass1_only"})
            tx_list.append(shift1_x); ty_list.append(shift1_y); corr_list.append(corr1)
            valid_ch_indices.append(ch_idx)
            channel_details.append({"ch": ch_idx, "outlier": False, **detail})
            continue

        # Pass 2: nearest grid
        xi2, yi2 = _select_nearest_grid(
            shift1_x, shift1_y, pos_map,
            sx_sign, sy_sign, x_step, y_step, pixel_scale_um, grid_cal=grid_cal)
        offset_x2, offset_y2 = _get_grid_offset_px(
            xi2, yi2, sx_sign, sy_sign, x_step, y_step, pixel_scale_um, grid_cal=grid_cal)

        if (xi2, yi2) not in grid_half_cache:
            try:
                halves = _load_grid_ref_full(
                    pos_map, xi2, yi2, rois, n_channels,
                    grid_z_index, cfg,
                    tilt_crop_h=tilt_crop_h, ecc_crop_h=ecc_crop_h,
                    fit_right=fit_right)
                grid_half_cache[(xi2, yi2)] = halves
                grid_half_u8_cache[(xi2, yi2)] = [to_uint8(h, vmin, vmax) for h in halves]
            except FileNotFoundError:
                # Fall back to pass 1
                detail.update({"xi": xi2, "yi": yi2, "tx2": shift1_x, "ty2": shift1_y,
                               "corr2": corr1, "status": "pass2_load_failed"})
                tx_list.append(shift1_x); ty_list.append(shift1_y); corr_list.append(corr1)
                valid_ch_indices.append(ch_idx)
                channel_details.append({"ch": ch_idx, "outlier": False, **detail})
                continue

        ref_u8_p2 = grid_half_u8_cache[(xi2, yi2)][ch_idx]
        result2 = ecc_align(ref_u8_p2, to_uint8(cur_crop, vmin, vmax))
        if result2 is None:
            detail.update({"xi": xi2, "yi": yi2, "tx2": shift1_x, "ty2": shift1_y,
                           "corr2": corr1, "status": "pass2_ecc_failed"})
            tx_list.append(shift1_x); ty_list.append(shift1_y); corr_list.append(corr1)
            valid_ch_indices.append(ch_idx)
            channel_details.append({"ch": ch_idx, "outlier": False, **detail})
            continue

        fine2_x, fine2_y, corr2 = result2
        shift2_x = fine2_x + offset_x2
        shift2_y = fine2_y + offset_y2
        detail.update({"xi": xi2, "yi": yi2, "tx2": shift2_x, "ty2": shift2_y, "corr2": corr2})

        # Pass 3
        final_shift_x, final_shift_y, final_corr = shift2_x, shift2_y, corr2
        final_status = "pass2_ok"

        if enable_third_pass:
            xi3, yi3 = _select_nearest_grid(
                shift2_x, shift2_y, pos_map,
                sx_sign, sy_sign, x_step, y_step, pixel_scale_um, grid_cal=grid_cal)

            if (xi3, yi3) != (xi2, yi2):
                offset_x3, offset_y3 = _get_grid_offset_px(
                    xi3, yi3, sx_sign, sy_sign, x_step, y_step, pixel_scale_um, grid_cal=grid_cal)

                if (xi3, yi3) not in grid_half_cache:
                    try:
                        halves3 = _load_grid_ref_full(
                            pos_map, xi3, yi3, rois, n_channels,
                            grid_z_index, cfg,
                            tilt_crop_h=tilt_crop_h, ecc_crop_h=ecc_crop_h,
                            fit_right=fit_right)
                        grid_half_cache[(xi3, yi3)] = halves3
                        grid_half_u8_cache[(xi3, yi3)] = [to_uint8(h, vmin, vmax) for h in halves3]
                    except FileNotFoundError:
                        pass

                if (xi3, yi3) in grid_half_u8_cache:
                    ref_u8_p3 = grid_half_u8_cache[(xi3, yi3)][ch_idx]
                    result3 = ecc_align(ref_u8_p3, to_uint8(cur_crop, vmin, vmax))
                    if result3 is not None:
                        fine3_x, fine3_y, corr3 = result3
                        final_shift_x = fine3_x + offset_x3
                        final_shift_y = fine3_y + offset_y3
                        final_corr = corr3
                        final_status = "pass3_ok"
                        detail.update({"xi3": xi3, "yi3": yi3,
                                       "tx3": final_shift_x, "ty3": final_shift_y, "corr3": corr3})

        detail["status"] = final_status
        tx_list.append(final_shift_x)
        ty_list.append(final_shift_y)
        corr_list.append(final_corr)
        valid_ch_indices.append(ch_idx)
        channel_details.append({"ch": ch_idx, "outlier": False, **detail})

    # ---- All channels failed ----
    if not tx_list:
        print(f"  [{pos_label}] ERROR: ECC failed on all channels")
        return fail_result

    # ---- Channel averaging (outlier removal) ----
    n_ch_raw = len(tx_list)
    low_corr_mask = np.zeros(n_ch_raw, dtype=bool)
    if ecc_min_corr > 0:
        low_corr_mask = np.array([c < ecc_min_corr for c in corr_list])

    if n_ch_raw >= 3:
        is_out = _remove_outliers_mad(tx_list) | _remove_outliers_mad(ty_list) | low_corr_mask
        used_idx = [i for i, o in enumerate(is_out) if not o]
        if not used_idx:
            used_idx = list(range(n_ch_raw))
    else:
        is_out = low_corr_mask
        used_idx = [i for i, o in enumerate(is_out) if not o]
        if not used_idx:
            used_idx = list(range(n_ch_raw))

    # Mark outliers in channel_details
    detail_idx = 0
    for cd in channel_details:
        if cd.get("status") != "pass1_failed":
            if detail_idx < len(is_out):
                cd["outlier"] = bool(is_out[detail_idx])
            detail_idx += 1

    tx_arr = np.array(tx_list)
    ty_arr = np.array(ty_list)
    corr_arr = np.array(corr_list)
    tx_avg = float(np.mean(tx_arr[used_idx]))
    ty_avg = float(np.mean(ty_arr[used_idx]))
    corr_avg = float(np.mean(corr_arr[used_idx]))

    # ---- EMA filter ----
    prev_ema_tx = prev_state["ema_tx_px"]
    prev_ema_ty = prev_state["ema_ty_px"]
    if prev_ema_tx is None:
        tx_filt, ty_filt = tx_avg, ty_avg
    else:
        tx_filt = ema_alpha * tx_avg + (1.0 - ema_alpha) * prev_ema_tx
        ty_filt = ema_alpha * ty_avg + (1.0 - ema_alpha) * prev_ema_ty

    # ---- Sign/scale conversion (pixel -> um, image -> stage) ----
    correction_stage_x_um = sx_sign * ty_filt * pixel_scale_um
    correction_stage_y_um = sy_sign * tx_filt * pixel_scale_um

    # ---- Kalman filter ----
    kf_update = None
    if use_kalman:
        kf_Q_ty = cfg.get("kf_Q_ty_nm2", cfg.get("kf_Q_pos_nm2", 548.0))
        kf_Q_tx = cfg.get("kf_Q_tx_nm2", cfg.get("kf_Q_pos_nm2", 548.0))
        kf_R_ty = cfg.get("kf_R_ty_nm2", cfg.get("kf_R_nm2", 454.0))
        kf_R_tx = cfg.get("kf_R_tx_nm2", cfg.get("kf_R_nm2", 454.0))
        px_scale_nm = pixel_scale_um * 1000.0

        z_ty_nm = tx_avg * px_scale_nm * sy_sign
        z_tx_nm = ty_avg * px_scale_nm * sx_sign
        ol_pos_ty_nm = prev_state["cumulative_dy_um"] * 1000.0 + z_ty_nm
        ol_pos_tx_nm = prev_state["cumulative_dx_um"] * 1000.0 + z_tx_nm

        pos_ty_new, P_ty_new, K_ty = kf_step_posonly_nm(
            ol_pos_ty_nm, kf_state["kf_pos_ty_nm"], kf_state["kf_P_ty"], kf_Q_ty, kf_R_ty)
        pos_tx_new, P_tx_new, K_tx = kf_step_posonly_nm(
            ol_pos_tx_nm, kf_state["kf_pos_tx_nm"], kf_state["kf_P_tx"], kf_Q_tx, kf_R_tx)

        correction_stage_y_um = pos_ty_new / 1000.0 - prev_state["cumulative_dy_um"]
        correction_stage_x_um = pos_tx_new / 1000.0 - prev_state["cumulative_dx_um"]

        kf_update = {
            "kf_pos_tx_nm": float(pos_tx_new), "kf_P_tx": float(P_tx_new),
            "kf_pos_ty_nm": float(pos_ty_new), "kf_P_ty": float(P_ty_new),
        }

    # ---- Cumulative drift ----
    is_first_frame = prev_state["ema_tx_px"] is None
    if is_first_frame:
        cum_dx = cum_dy = 0.0
    else:
        cum_dx = prev_state["cumulative_dx_um"] + correction_stage_x_um
        cum_dy = prev_state["cumulative_dy_um"] + correction_stage_y_um

    # ---- Jump detection ----
    step_um = (correction_stage_x_um**2 + correction_stage_y_um**2) ** 0.5
    total_um = (cum_dx**2 + cum_dy**2) ** 0.5
    if jump_thresh is None:
        jump = total_um > max_total
    else:
        jump = (step_um > jump_thresh) or (total_um > max_total)

    if jump:
        cum_dx = prev_state["cumulative_dx_um"]
        cum_dy = prev_state["cumulative_dy_um"]

    elapsed = (datetime.now() - t_start).total_seconds()
    status_str = "JUMP" if jump else "OK"
    print(f"  [{pos_label}] {status_str}  "
          f"dx={correction_stage_x_um:+.4f}um dy={correction_stage_y_um:+.4f}um  "
          f"cum=({cum_dx:+.3f},{cum_dy:+.3f})um  "
          f"corr={corr_avg:.3f}  {len(used_idx)}/{n_ch_raw}ch  {elapsed:.1f}s")

    return {
        "pos_idx": pos_idx,
        "pos_label": pos_label,
        "dx_um": correction_stage_x_um,
        "dy_um": correction_stage_y_um,
        "cumulative_dx_um": cum_dx,
        "cumulative_dy_um": cum_dy,
        "valid": True,
        "jump": jump,
        "corr": corr_avg,
        "ema_tx": tx_filt,
        "ema_ty": ty_filt,
        "kf_update": kf_update,
        "channel_details": sorted(channel_details, key=lambda x: x["ch"]),
        "n_channels_used": len(used_idx),
        "n_channels_raw": n_ch_raw,
        "tx_avg_px": tx_avg,
        "ty_avg_px": ty_avg,
        "raw_path": str(raw_path),
    }


# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()
    cfg = load_config(args.config)
    t = args.timepoint

    # ---- Load positions ----
    positions = load_positions_csv(cfg["positions_csv"])
    bg_pos_index = cfg["bg_pos_index"]
    bg_label = positions[bg_pos_index]["label"]
    sample_positions = [p for p in positions if p["index"] != bg_pos_index]

    # ---- Grouping ----
    interval = cfg.get("drift_sample_interval", 1)
    group_leaders = sample_positions[::interval] if interval > 1 else sample_positions
    group_map = {}
    for i, pos in enumerate(sample_positions):
        leader_idx = min((i // interval) * interval, len(sample_positions) - 1) if interval > 1 else i
        group_map[pos["index"]] = sample_positions[leader_idx]["index"]

    print(f"[T={t}] compute_drift_online_v3.py  "
          f"{len(group_leaders)} leaders / {len(sample_positions)} positions  "
          f"interval={interval}")

    # ---- Load channel ROIs ----
    with open(cfg["channel_rois_json"], encoding="utf-8") as f:
        rois = json.load(f)
    n_channels = len(rois)

    # ---- BG image path ----
    bg_raw = get_raw_path(cfg["save_dir"], bg_label, t)
    if not bg_raw.exists():
        print(f"WARNING: BG image not found: {bg_raw}")
        bg_raw = None

    # ---- Load grid references for leaders ----
    grid_dir = Path(cfg["grid_dir"])
    leader_data = {}
    for leader in group_leaders:
        label = leader["label"]
        try:
            _, u8_crops = load_grid_ref_crops_for_pos(label, cfg, rois)
            pos_map = scan_grid_positions(str(grid_dir), label)
            grid_cal = load_grid_cal_for_pos(label, cfg)
            leader_data[leader["index"]] = {
                "u8_crops": u8_crops,
                "pos_map": pos_map,
                "grid_cal": grid_cal,
            }
            cal_str = f"{len(grid_cal)} cal" if grid_cal else "nominal"
            print(f"  {label}: {len(pos_map)} grid pos, {cal_str}")
        except FileNotFoundError as e:
            print(f"  WARNING: skipping {label}: {e}")

    if not leader_data:
        print("ERROR: no leaders have grid data")
        sys.exit(1)

    # ---- Load previous states and KF states ----
    state_path = cfg["state_file"]
    kf_path = cfg.get("kf_state_file",
                       str(Path(cfg["session_dir"]) / "drift_kf_state.json"))
    kf_R = max(cfg.get("kf_R_ty_nm2", 454.0), cfg.get("kf_R_tx_nm2", 454.0))

    # ---- Process leaders in parallel ----
    def _process_leader(leader):
        idx = leader["index"]
        label = leader["label"]
        if idx not in leader_data:
            return None
        raw_path = get_raw_path(cfg["save_dir"], label, t)
        prev = read_per_pos_state(state_path, idx)
        kf_st = load_per_pos_kf_state(kf_path, label, kf_R)
        ld = leader_data[idx]
        return _process_one_position(
            idx, label, str(raw_path), str(bg_raw) if bg_raw else None,
            cfg, rois, ld["u8_crops"], ld["pos_map"], ld["grid_cal"],
            prev, kf_st)

    max_workers = cfg.get("max_drift_workers", 0)
    if max_workers <= 0:
        max_workers = max(1, os.cpu_count() - 4)
    max_workers = min(max_workers, len(group_leaders))

    print(f"  Processing with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        leader_results = list(executor.map(_process_leader, group_leaders))

    # Filter out None results (skipped leaders)
    leader_results = [r for r in leader_results if r is not None]

    # ---- Apply group mapping ----
    leader_result_map = {r["pos_idx"]: r for r in leader_results}
    all_results = []
    for pos in sample_positions:
        leader_idx = group_map[pos["index"]]
        if leader_idx in leader_result_map:
            if pos["index"] == leader_idx:
                all_results.append(leader_result_map[leader_idx])
            else:
                # Copy leader's correction for group member
                lr = leader_result_map[leader_idx]
                all_results.append({
                    "pos_idx": pos["index"],
                    "pos_label": pos["label"],
                    "dx_um": lr["dx_um"],
                    "dy_um": lr["dy_um"],
                    "cumulative_dx_um": lr["cumulative_dx_um"],
                    "cumulative_dy_um": lr["cumulative_dy_um"],
                    "valid": lr["valid"],
                    "jump": lr["jump"],
                    "corr": lr["corr"],
                    "ema_tx": lr["ema_tx"],
                    "ema_ty": lr["ema_ty"],
                    "kf_update": None,
                    "channel_details": [],
                })

    # ---- Write per-pos state ----
    write_per_pos_state(state_path, t, all_results, bg_pos_index)

    # ---- Save KF states ----
    kf_updates = {}
    for r in leader_results:
        if r.get("kf_update"):
            kf_updates[r["pos_label"]] = r["kf_update"]
    if kf_updates:
        save_all_kf_states(kf_path, kf_updates)

    # ---- Write log ----
    log_path = cfg["log_file"]
    log_entries = []
    for r in leader_results:
        log_entries.append({
            "timepoint": t,
            "timestamp": datetime.now().isoformat(),
            "pos_idx": r["pos_idx"],
            "pos_label": r["pos_label"],
            "raw_path": r.get("raw_path"),
            "n_channels_used": r.get("n_channels_used", 0),
            "n_channels_raw": r.get("n_channels_raw", 0),
            "tx_avg_px": r.get("tx_avg_px", 0.0),
            "ty_avg_px": r.get("ty_avg_px", 0.0),
            "ecc_correlation": r["corr"],
            "correction_stage_x_um": r["dx_um"],
            "correction_stage_y_um": r["dy_um"],
            "cumulative_dx_um": r["cumulative_dx_um"],
            "cumulative_dy_um": r["cumulative_dy_um"],
            "jump_detected": r["jump"],
            "correction_valid": r["valid"] and not r["jump"],
            "channel_details": r.get("channel_details", []),
        })

    try:
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
    except Exception:
        log = []
    log.append({"timepoint": t, "per_pos": True, "positions": log_entries})
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    n_valid = sum(1 for r in all_results if r["valid"] and not r["jump"])
    n_jump = sum(1 for r in all_results if r["jump"])
    print(f"[T={t}] done  {n_valid} valid, {n_jump} jump, "
          f"{len(all_results) - n_valid - n_jump} failed")


if __name__ == "__main__":
    main()
