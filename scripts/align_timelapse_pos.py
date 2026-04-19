# %%
"""
align_timelapse_pos.py
----------------------
Before starting a timelapse, align each base pos in timelapse.pos
to its corresponding grid x+0_y+0 using ECC.

Usage:
  Run after determining z with focus_check_subtract.py and before running
  prepare_drift_session.
  Use the output timelapse_aligned.pos directly for the timelapse so that
  each PosN starts acquisition near grid x+0_y+0.

  During the timelapse, compute_drift_online.py measures Pos1 drift in real-time
  and applies stage correction to all pos uniformly, so pre-correcting each PosN
  individually keeps each PosN near grid x+0_y+0 throughout the entire timelapse.

Processing:
  For each PosN (excluding keys in BG_COPY_FROM):
    1. Auto-reconstruct CALIB_GRID/PosN_x+0_y+0 if not yet reconstructed (x+0_y+0 only)
    2. ECC( REF_GRID/PosN_x+0_y+0, CALIB_GRID/PosN_x+0_y+0 )
       - _tilt_correct + fixed ecc_vmin/ecc_vmax
       - MAD outlier removal (thresh=5.0)
    3. Apply correction to PosN XYStage coordinates
  For BG pos (e.g. Pos0), copy correction from the designated sample pos (e.g. Pos1).

Output:
  OUTPUT_POS (default: BASE_DIR/timelapse_aligned.pos)
  BASE_DIR/align_timelapse_log.json

Sign convention -- identical to calibrate_grid_pos.py L389-394:
  findTransformECC(ref, sample) -> (tx, ty)
    tx = warp[0,2] : image X (col) direction shift [px]
    ty = warp[1,2] : image Y (row) direction shift [px]
  drift_stage_x_um = sx_sign * ty * pixel_scale_um   (image Y -> stage X)
  drift_stage_y_um = sy_sign * tx * pixel_scale_um   (image X -> stage Y)
  pos_correct = -drift
"""

import json
import copy
import re
import sys
import numpy as np
import tifffile
import cv2
from pathlib import Path

# ============================================================
# Configuration parameters -- edit here before each run
# ============================================================

# timelapse.pos to correct (base positions only)
TIMELAPSE_POS = r"D:\AquisitionData\Kitagishi\260331\timelapse.pos"

# Output pos file path
# None generates timelapse_aligned.pos in the same directory as timelapse.pos
OUTPUT_POS = None

# Log output directory (None -> same directory as TIMELAPSE_POS)
LOG_DIR = None

# Today's calibration grid folder
# Contains raw .tif for PosN_x+0_y+0 (auto-reconstructs if output_phase/ is missing)
CALIB_GRID_DIR = r"D:\AquisitionData\Kitagishi\260331\grid_0p00525pergluc_60ms_1"

# Day-1 fixed reference grid folder
# PosN_x+0_y+0/output_phase/*_phase.tif must be pre-reconstructed
REF_GRID_DIR = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1"

# z index (img_000000000_ph_{Z:03d}_phase.tif)
REF_Z_INDEX   = 10   # REF side
CALIB_Z_INDEX = 10   # CALIB side (z used for reconstruction + ECC)

# BG base label for CALIB_GRID (used for BG subtraction during reconstruction)
CALIB_BG_BASE_LABEL = "Pos0"

# drift_config.json (optical parameters, signs, ECC vmin/vmax, crop settings)
DRIFT_CONFIG = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json"

# ECC / tilt correction parameters
TILT_CROP_H = 270   # X width for tilt correction [px]
ECC_CROP_H  = 80    # Crop X width used for ECC [px]
MAD_THRESH  = 5.0   # Inter-channel MAD outlier threshold

# BG pos correction copy settings
# Key: BG LABEL, Value: sample pos LABEL to copy correction from
BG_COPY_FROM = {"Pos0": "Pos1"}

# ============================================================


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


from ecc_utils import (
    tilt_fit_crop, extract_rect_roi, to_uint8, ecc_align,
    mad, remove_outliers_mad,
)


def get_crops_u8(img_f64, rois, n_channels, vmin, vmax, fit_right: bool = False):
    """Return tilt-corrected uint8 crops for all channels (None if OOB)."""
    crops = []
    for ch in range(n_channels):
        tc = tilt_fit_crop(img_f64, rois[ch]["cy"], rois[ch]["cx"],
                           rois[ch]["crop_w"], ECC_CROP_H, TILT_CROP_H,
                           fit_right=fit_right)
        crops.append(to_uint8(tc, vmin, vmax) if tc is not None else None)
    return crops


def load_phase_image(grid_dir, label, z_index):
    path = (Path(grid_dir) / label / "output_phase"
            / f"img_000000000_ph_{z_index:03d}_phase.tif")
    if not path.exists():
        raise FileNotFoundError(str(path))
    return tifffile.imread(str(path)).astype(np.float64)


def reconstruct_phase(raw_path: Path, cfg: dict, bg_path: Path = None,
                      pos_num: int = 1) -> np.ndarray:
    script_dir = Path(cfg["script_dir"])
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from PIL import Image
    from qpi import QPIParameters, get_field
    from skimage.restoration import unwrap_phase
    from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

    pos_split = cfg.get("pos_split", 3)
    crop = tuple(cfg["crop_before"]) if pos_num < pos_split else tuple(cfg["crop_after"])
    rs, re_, cs, ce = crop

    def _recon(path):
        img = np.array(Image.open(str(path)))
        img_crop = img[rs:re_, cs:ce]
        qpi_params = QPIParameters(
            wavelength=WAVELENGTH, NA=NA,
            img_shape=img_crop.shape, pixelsize=PIXELSIZE,
            offaxis_center=OFFAXIS_CENTER,
        )
        field = get_field(img_crop, qpi_params)
        return unwrap_phase(np.angle(field))

    phase = _recon(raw_path)
    if bg_path is not None and bg_path.exists():
        phase = phase - _recon(bg_path)
        print(f"    BG: {bg_path.parent.name}/{bg_path.name}")
    else:
        print("    No BG")
    return phase


def postprocess_phase(phase: np.ndarray, cfg: dict) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    h_p, w_p = phase.shape
    region = phase[1:h_p - 1, 1:w_p // 2]
    if region.size > 0:
        phase = phase - np.mean(region)
    sigma = cfg.get("gradient_sigma", 0)
    if sigma > 0:
        phase = phase - gaussian_filter(phase, sigma=sigma, mode="nearest")
    return phase


def ensure_center_reconstructed(calib_grid_dir: str, label: str, z_index: int,
                                 bg_base_label: str, cfg: dict) -> Path:
    """Reconstruct if CALIB_GRID/label/output_phase/*_phase.tif is not yet created."""
    out_path = (Path(calib_grid_dir) / label / "output_phase"
                / f"img_000000000_ph_{z_index:03d}_phase.tif")
    if out_path.exists():
        return out_path

    print(f"  [reconstruction] {label} -> auto-reconstructing")
    raw_path = Path(calib_grid_dir) / label / f"img_000000000_ph_{z_index:03d}.tif"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw image not found: {raw_path}")

    bg_raw = (Path(calib_grid_dir) / f"{bg_base_label}_x+0_y+0"
              / f"img_000000000_ph_{z_index:03d}.tif")
    if not bg_raw.exists():
        print(f"    WARNING: BG raw not found -> no BG")
        bg_raw = None

    m = re.match(r"Pos(\d+)", label)
    pos_num = int(m.group(1)) if m else 1

    phase = reconstruct_phase(raw_path, cfg, bg_raw, pos_num)
    phase = postprocess_phase(phase, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), phase.astype(np.float32))
    print(f"    Saved: {out_path}")
    return out_path


def center_ecc(ref_img, calib_img, rois, n_channels, vmin, vmax):
    """ECC -> MAD outlier removal -> average. Returns (tx_avg, ty_avg, per_ch)."""
    ref_crops   = get_crops_u8(ref_img,   rois, n_channels, vmin, vmax)
    calib_crops = get_crops_u8(calib_img, rois, n_channels, vmin, vmax)

    tx_list, ty_list, corr_list = [], [], []
    per_ch = []
    for ch in range(n_channels):
        if ref_crops[ch] is None or calib_crops[ch] is None:
            per_ch.append({"ch": ch, "tx": None, "ty": None,
                           "corr": None, "excluded": True, "reason": "tilt_oob"})
            continue
        res = ecc_align(ref_crops[ch], calib_crops[ch])
        if res is None:
            per_ch.append({"ch": ch, "tx": None, "ty": None,
                           "corr": None, "excluded": True, "reason": "ecc_failed"})
        else:
            tx, ty, corr = res
            tx_list.append(tx); ty_list.append(ty); corr_list.append(corr)
            per_ch.append({"ch": ch, "tx": tx, "ty": ty,
                           "corr": corr, "excluded": False, "reason": None})

    if not tx_list:
        return None, None, per_ch

    n_raw = len(tx_list)
    if n_raw >= 3:
        is_out = (remove_outliers_mad(tx_list, MAD_THRESH)
                  | remove_outliers_mad(ty_list, MAD_THRESH))
    else:
        is_out = np.zeros(n_raw, dtype=bool)

    valid_indices = [i for i, c in enumerate(per_ch) if not c["excluded"]]
    for k, idx in enumerate(valid_indices):
        if is_out[k]:
            per_ch[idx]["excluded"] = True
            per_ch[idx]["reason"] = "mad_outlier"

    used_mask = ~is_out
    if not np.any(used_mask):
        used_mask = np.ones(n_raw, dtype=bool)
        for idx in valid_indices:
            per_ch[idx]["excluded"] = False
            per_ch[idx]["reason"] = None

    tx_avg   = float(np.mean(np.array(tx_list)[used_mask]))
    ty_avg   = float(np.mean(np.array(ty_list)[used_mask]))
    n_used   = int(np.sum(used_mask))
    corr_avg = float(np.mean(np.array(corr_list)[used_mask]))
    print(f"  ECC: tx={tx_avg:+.3f}px  ty={ty_avg:+.3f}px  "
          f"used {n_used}/{n_raw}ch  corr={corr_avg:.4f}")
    return tx_avg, ty_avg, per_ch


# ============================================================
# Main
# ============================================================

def find_channel_rois(grid_dir: str, label: str):
    """Try both x+0_y+0 and x-0_y-0 naming conventions."""
    for center_name in [f"{label}_x+0_y+0", f"{label}_x-0_y-0"]:
        p = (Path(grid_dir) / center_name / "output_phase" / "channels" / "channel_rois.json")
        if p.exists():
            return p
    return None


def main():
    cfg            = load_config(DRIFT_CONFIG)
    vmin           = cfg.get("ecc_vmin", -5.0)
    vmax           = cfg.get("ecc_vmax",  2.0)
    pixel_scale_um = cfg["pixel_scale_um"]
    sx_sign        = cfg.get("shift_sign_x", 1)
    sy_sign        = cfg.get("shift_sign_y", 1)

    print(f"pixel_scale: {pixel_scale_um:.5f} μm/px  sx_sign={sx_sign}  sy_sign={sy_sign}")

    with open(TIMELAPSE_POS, "r") as f:
        pos_data = json.load(f)
    positions = pos_data["POSITIONS"]
    print(f"timelapse.pos: {len(positions)} positions")

    pos_by_label       = {p["LABEL"]: p for p in positions}
    correction_by_label: dict[str, dict] = {}
    bg_labels          = set(BG_COPY_FROM.keys())
    log_entries        = []

    # ---- Center correction for sample pos ----
    for pos in positions:
        base_label   = pos["LABEL"]
        center_label = f"{base_label}_x+0_y+0"

        if base_label in bg_labels:
            continue

        print(f"\n=== {base_label} ===")

        # Step 0: Load per-PosN channel_rois
        channel_rois_path = find_channel_rois(REF_GRID_DIR, base_label)
        if channel_rois_path is None:
            print(f"  ERROR: channel_rois.json not found in {REF_GRID_DIR} -> skip")
            log_entries.append({"base_label": base_label, "center_correction": None,
                                 "error": "channel_rois_not_found"})
            continue
        with open(channel_rois_path, encoding="utf-8") as f:
            rois = json.load(f)
        n_channels = len(rois)
        print(f"  channel_rois: {n_channels} ch  vmin={vmin}  vmax={vmax}")

        # Step 1: Reconstruct x+0_y+0 only (if needed)
        try:
            ensure_center_reconstructed(
                CALIB_GRID_DIR, center_label, CALIB_Z_INDEX,
                CALIB_BG_BASE_LABEL, cfg,
            )
        except FileNotFoundError as e:
            print(f"  [reconstruction] Failed -> skipping: {e}")
            log_entries.append({"base_label": base_label, "center_correction": None,
                                 "error": f"reconstruction failed: {e}"})
            continue

        # Step 2: Center correction ECC
        try:
            ref_img   = load_phase_image(REF_GRID_DIR,   center_label, REF_Z_INDEX)
            calib_img = load_phase_image(CALIB_GRID_DIR, center_label, CALIB_Z_INDEX)
        except FileNotFoundError as e:
            print(f"  [center correction] Image load failed -> skipping: {e}")
            log_entries.append({"base_label": base_label, "center_correction": None,
                                 "error": str(e)})
            continue

        tx_avg, ty_avg, per_ch = center_ecc(
            ref_img, calib_img, rois, n_channels, vmin, vmax)

        if tx_avg is None:
            print(f"  [center correction] All channels ECC failed -> skipping")
            log_entries.append({"base_label": base_label, "center_correction": None,
                                 "error": "all_ecc_failed", "per_channel": per_ch})
            continue

        # Sign convention: identical to calibrate_grid_pos.py L389-394
        drift_stage_x_um = sx_sign * ty_avg * pixel_scale_um
        drift_stage_y_um = sy_sign * tx_avg * pixel_scale_um
        pos_correct_x_um = -drift_stage_x_um
        pos_correct_y_um = -drift_stage_y_um

        print(f"  Correction: stage_X={pos_correct_x_um:+.4f}um  stage_Y={pos_correct_y_um:+.4f}um")

        for dev in pos["DEVICES"]:
            if dev["DEVICE"] == "XYStage":
                dev["X"] += pos_correct_x_um
                dev["Y"] += pos_correct_y_um

        correction = {
            "tx_avg_px": tx_avg, "ty_avg_px": ty_avg,
            "drift_stage_x_um": drift_stage_x_um,
            "drift_stage_y_um": drift_stage_y_um,
            "pos_correct_x_um": pos_correct_x_um,
            "pos_correct_y_um": pos_correct_y_um,
            "n_channels_total": n_channels,
            "per_channel": per_ch,
        }
        correction_by_label[base_label] = correction
        log_entries.append({"base_label": base_label, "center_correction": correction})

    # ---- Copy corresponding sample correction to BG pos ----
    for bg_label, src_label in BG_COPY_FROM.items():
        print(f"\n=== {bg_label} (BG: copying correction from {src_label}) ===")
        if src_label not in correction_by_label:
            print(f"  WARNING: Correction for {src_label} not computed -> skipping {bg_label}")
            log_entries.append({"base_label": bg_label, "copied_from": src_label,
                                 "center_correction": None,
                                 "error": "source correction not available"})
            continue

        src_corr = correction_by_label[src_label]
        pos = pos_by_label.get(bg_label)
        if pos is None:
            print(f"  WARNING: {bg_label} not found in timelapse.pos")
            continue

        for dev in pos["DEVICES"]:
            if dev["DEVICE"] == "XYStage":
                dev["X"] += src_corr["pos_correct_x_um"]
                dev["Y"] += src_corr["pos_correct_y_um"]

        print(f"  Correction: stage_X={src_corr['pos_correct_x_um']:+.4f}um"
              f"  stage_Y={src_corr['pos_correct_y_um']:+.4f}um  (copied from: {src_label})")
        log_entries.append({
            "base_label": bg_label,
            "copied_from": src_label,
            "center_correction": {k: v for k, v in src_corr.items() if k != "per_channel"},
        })

    # ---- Determine output path ----
    tl_path = Path(TIMELAPSE_POS)
    out_pos = Path(OUTPUT_POS) if OUTPUT_POS else tl_path.parent / "timelapse_aligned.pos"
    log_dir = Path(LOG_DIR) if LOG_DIR else tl_path.parent

    # ---- Save corrected pos ----
    with open(out_pos, "w") as f:
        json.dump(pos_data, f, indent=3)
    print(f"\nCorrected pos saved: {out_pos}  ({len(positions)} positions)")

    # ---- Save log ----
    out_log = log_dir / "align_timelapse_log.json"
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump({
            "timelapse_pos":    str(TIMELAPSE_POS),
            "output_pos":       str(out_pos),
            "calib_grid_dir":   str(CALIB_GRID_DIR),
            "ref_grid_dir":     str(REF_GRID_DIR),
            "ref_z_index":      REF_Z_INDEX,
            "calib_z_index":    CALIB_Z_INDEX,
            "ecc_vmin":         vmin,
            "ecc_vmax":         vmax,
            "tilt_crop_h":      TILT_CROP_H,
            "ecc_crop_h":       ECC_CROP_H,
            "mad_thresh":       MAD_THRESH,
            "bg_copy_from":     BG_COPY_FROM,
            "per_pos":          log_entries,
        }, f, indent=2, ensure_ascii=False)
    print(f"Log: {out_log}")
    print("\n--- Done ---")
    print(f"Next step: Load {out_pos.name} into MicroManager")
    print(f"  Verify z with focus_check_subtract.py -> run prepare_drift_session")


if __name__ == "__main__":
    main()

# %%
