# %%
"""
calibrate_grid_pos_per_pos.py
-----------------------------
For each base pos (PosN), perform ECC center correction by referencing
the corresponding PosN_x+0_y+0 individually, then expand a 0.1 um nominal grid.

Workflow:
  For each PosN in TIMELAPSE_POS (excluding keys in BG_COPY_FROM):
    1. Center correction reconstruction:
       Auto-reconstruct CALIB_GRID/PosN_x+0_y+0 output_phase if not yet created.
       (Only x+0_y+0; does not reconstruct all grid points.)
    2. Center correction: ECC( REF_GRID/PosN_x+0_y+0, CALIB_GRID/PosN_x+0_y+0 )
       - _tilt_correct + fixed ecc_vmin/ecc_vmax
       - MAD outlier removal (thresh=5.0)
       - Apply correction to PosN coordinates (same sign convention as calibrate_grid_pos.py)

  For BG pos specified by BG_COPY_FROM (e.g. Pos0),
  the correction from the corresponding sample pos (e.g. Pos1) is copied as-is.

  Output:
    BASE_DIR/_per_pos_ecc_corrected.pos  : corrected base pos
    BASE_DIR/_per_pos_ecc_grid.pos       : nominal 0.1 um grid expansion (snake scan)
    BASE_DIR/per_pos_ecc_log.json        : correction log

Sign convention:
  findTransformECC(ref, sample) -> (tx, ty)
    tx = warp[0,2] : image X (col) direction shift [px]
    ty = warp[1,2] : image Y (row) direction shift [px]
  drift_stage_x_um = sx_sign * ty * pixel_scale_um   (image Y -> stage X)
  drift_stage_y_um = sy_sign * tx * pixel_scale_um   (image X -> stage Y)
  pos_correct = -drift  (direction to cancel drift)
  Identical to calibrate_grid_pos.py L389-394
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

# Output directory
BASE_DIR = r"C:\260416"

# timelapse.pos to correct (contains only base positions)
TIMELAPSE_POS = r"D:\AquisitionData\Kitagishi\260416\timelapse.pos"

# Today's calibration data folder
# For grid data (Pos1_x+0_y+0): CALIB_SUFFIX = "x+0_y+0"
# For focus/timelapse data (Pos1): CALIB_SUFFIX = ""
CALIB_SUFFIX   = ""
CALIB_GRID_DIR = r"C:\260416\_for_lowper_gridgluc_1"

# Day-1 fixed reference grid folder
# PosN_x+0_y+0/output_phase/*_phase.tif must be pre-reconstructed
REF_GRID_DIR   = r"C:\260416\2per_gridgluc_1"

# z index (img_000000000_ph_{Z:03d}_phase.tif)
REF_Z_INDEX   = 5    # REF side (260416: 11-frame z-stack, z=0=index5)
CALIB_Z_INDEX = 0    # CALIB side (single z acquisition)

# BG base label for CALIB_GRID (Pos0_x+0_y+0 used for BG subtraction during reconstruction)
CALIB_BG_BASE_LABEL = "Pos0"

# If channel_rois.json does not exist per-pos, channel_crop.py --detect is auto-executed

# drift_config.json (optical parameters, signs, ECC vmin/vmax, crop settings)
DRIFT_CONFIG = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json"

# ECC / tilt correction parameters
TILT_CROP_H = 270   # X width for tilt correction [px] (same as compute_pos_shifts.py)
ECC_CROP_H  = 80    # Crop X width used for ECC [px]
MAD_THRESH  = 5.0   # Inter-channel MAD outlier threshold

# Grid expansion parameters (must match generate_grid_pos.py)
X_STEP = 0.1   # [um]  stage X -> image Y
Y_STEP = 0.1   # [um]  stage Y -> image X
X_HALF = 4     # -> total 9 points/axis -> 81 points/Pos
Y_HALF = 4

# BG pos correction copy settings
# Key: BG LABEL, Value: sample pos LABEL to copy correction from
# Example: apply Pos1's correction to Pos0 as-is
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
    """Read output_phase/img_000000000_ph_{z_index:03d}_phase.tif."""
    path = (Path(grid_dir) / label / "output_phase"
            / f"img_000000000_ph_{z_index:03d}_phase.tif")
    if not path.exists():
        raise FileNotFoundError(str(path))
    return tifffile.imread(str(path)).astype(np.float64)


# ============================================================
# Phase reconstruction (ported from calibrate_grid_pos.py, with per-Pos crop support)
# ============================================================

def reconstruct_phase(raw_path: Path, cfg: dict, bg_path: Path = None,
                      pos_num: int = 1) -> np.ndarray:
    """QPI phase reconstruction. Switches crop_before / crop_after based on pos_num."""
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
    """Same as calibrate_grid_pos.py. Mean removal + Gaussian gradient removal."""
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
                                 bg_base_label: str, cfg: dict,
                                 bg_suffix: str = "x+0_y+0") -> Path:
    """
    Reconstruct if CALIB_GRID/label/output_phase/img_..._phase.tif does not exist.
    label example: "Pos1_x+0_y+0" (grid) or "Pos1" (focus/timelapse). Returns path of reconstructed file.
    bg_suffix: BG directory suffix (grid: "x+0_y+0", focus: "")
    """
    out_path = (Path(calib_grid_dir) / label / "output_phase"
                / f"img_000000000_ph_{z_index:03d}_phase.tif")
    if out_path.exists():
        return out_path

    print(f"  [reconstruction] {label} -> auto-reconstructing")
    raw_path = (Path(calib_grid_dir) / label
                / f"img_000000000_ph_{z_index:03d}.tif")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw image not found: {raw_path}")

    bg_label = f"{bg_base_label}_{bg_suffix}" if bg_suffix else bg_base_label
    bg_raw   = Path(calib_grid_dir) / bg_label / f"img_000000000_ph_{z_index:03d}.tif"
    if not bg_raw.exists():
        print(f"    WARNING: BG raw not found: {bg_raw} -> no BG")
        bg_raw = None

    m = re.match(r"Pos(\d+)", label)
    pos_num = int(m.group(1)) if m else 1

    phase = reconstruct_phase(raw_path, cfg, bg_raw, pos_num)
    phase = postprocess_phase(phase, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), phase.astype(np.float32))
    print(f"    Saved: {out_path}")
    return out_path


# ============================================================
# Center correction ECC
# ============================================================

def center_ecc(ref_img, calib_img, rois, n_channels, vmin, vmax, fit_right: bool = False):
    """
    ECC -> MAD outlier removal -> average.
    Returns: (tx_avg, ty_avg, per_ch_list) | (None, None, per_ch_list) when all fail.
    tx: image X (col) [px], ty: image Y (row) [px]
    """
    ref_crops   = get_crops_u8(ref_img,   rois, n_channels, vmin, vmax, fit_right)
    calib_crops = get_crops_u8(calib_img, rois, n_channels, vmin, vmax, fit_right)

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
        is_out = remove_outliers_mad(tx_list, MAD_THRESH) | remove_outliers_mad(ty_list, MAD_THRESH)
    else:
        is_out = np.zeros(n_raw, dtype=bool)

    valid_indices = [i for i, c in enumerate(per_ch) if not c["excluded"]]
    for k, idx in enumerate(valid_indices):
        if is_out[k]:
            per_ch[idx]["excluded"] = True
            per_ch[idx]["reason"] = "mad_outlier"

    used_mask = ~is_out
    if not np.any(used_mask):
        # All are outliers -> use all (equivalent to calibrate_grid_pos.py L364-365)
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

def _run_channel_detect(pos_dir: Path, z_index: int = 0):
    """Run channel_crop.py --detect on the phase images in output_phase/.
    Since out_dir = img_dir / "channels", passing output_phase/ to --dir
    outputs to output_phase/channels/channel_rois.json.
    """
    import subprocess
    script = Path(__file__).resolve().parent / "channel_crop.py"
    output_phase_dir = pos_dir / "output_phase"
    pattern = f"img_*_ph_{z_index:03d}_phase.tif"
    cmd = [sys.executable, str(script),
           "--dir", str(output_phase_dir),
           "--pattern", pattern,
           "--detect"]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [!] channel_crop.py --detect failed (returncode={result.returncode})")


def _plot_corrections(log_entries, base_dir):
    """Plot per-pos ECC correction summary and save via figure_logger."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figure_logger import save_figure

    labels, corr_x, corr_y = [], [], []
    for entry in log_entries:
        cc = entry.get("center_correction")
        if cc is None:
            continue
        labels.append(entry["base_label"])
        corr_x.append(cc["pos_correct_x_um"])
        corr_y.append(cc["pos_correct_y_um"])

    if not labels:
        print("  [figure] no valid corrections to plot")
        return

    pos_idx = [int(re.match(r"Pos(\d+)", lb).group(1)) for lb in labels]
    corr_x = np.array(corr_x)
    corr_y = np.array(corr_y)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].bar(pos_idx, corr_x * 1000, width=0.8, color="tab:blue", alpha=0.8)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("Stage X correction (nm)")
    axes[0].set_title("Per-position ECC correction (REF grid -> CALIB)")

    axes[1].bar(pos_idx, corr_y * 1000, width=0.8, color="tab:orange", alpha=0.8)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("Stage Y correction (nm)")
    axes[1].set_xlabel("Position index")

    for ax in axes:
        ax.tick_params(labelsize=9)

    fig.tight_layout()
    save_figure(
        fig,
        params={"ref_grid": str(REF_GRID_DIR), "calib_grid": str(CALIB_GRID_DIR),
                "ref_z": REF_Z_INDEX, "calib_z": CALIB_Z_INDEX,
                "n_positions": len(labels)},
        description="Per-position ECC correction for grid pos calibration",
        data={"pos_idx": np.array(pos_idx),
              "corr_x_um": corr_x, "corr_y_um": corr_y},
    )
    plt.close(fig)


def main():
    cfg            = load_config(DRIFT_CONFIG)
    vmin           = cfg.get("ecc_vmin", -5.0)
    vmax           = cfg.get("ecc_vmax",  2.0)
    pixel_scale_um = cfg["pixel_scale_um"]
    sx_sign        = cfg.get("shift_sign_x", 1)
    sy_sign        = cfg.get("shift_sign_y", 1)
    pos_split      = cfg.get("pos_split", 33)

    print(f"vmin={vmin}  vmax={vmax}")
    print(f"pixel_scale: {pixel_scale_um:.5f} μm/px  sx_sign={sx_sign}  sy_sign={sy_sign}")

    with open(TIMELAPSE_POS, "r") as f:
        pos_data = json.load(f)
    positions = pos_data["POSITIONS"]
    print(f"timelapse.pos: {len(positions)} positions")

    # label -> pos dict (for updating coordinates during BG copy)
    pos_by_label = {p["LABEL"]: p for p in positions}

    # label -> correction dict (recorded for BG copy)
    correction_by_label: dict[str, dict] = {}

    bg_labels = set(BG_COPY_FROM.keys())

    log_entries = []

    # ---- Center correction for sample pos ----
    for pos in positions:
        base_label = pos["LABEL"]
        if base_label in bg_labels:
            continue   # BG will be copied later

        ref_label   = f"{base_label}_x+0_y+0"
        calib_label = f"{base_label}_{CALIB_SUFFIX}" if CALIB_SUFFIX else base_label
        print(f"\n=== {base_label} ===")

        # Step 1: Reconstruct CALIB side only if needed
        try:
            ensure_center_reconstructed(
                CALIB_GRID_DIR, calib_label, CALIB_Z_INDEX,
                CALIB_BG_BASE_LABEL, cfg, bg_suffix=CALIB_SUFFIX,
            )
        except FileNotFoundError as e:
            print(f"  [reconstruction] Failed -> skipping: {e}")
            log_entries.append({"base_label": base_label, "center_correction": None,
                                 "error": f"reconstruction failed: {e}"})
            continue

        # Step 2: Center correction ECC
        try:
            ref_img   = load_phase_image(REF_GRID_DIR,   ref_label,   REF_Z_INDEX)
            calib_img = load_phase_image(CALIB_GRID_DIR, calib_label, CALIB_Z_INDEX)
        except FileNotFoundError as e:
            print(f"  [center correction] Image load failed -> skipping: {e}")
            log_entries.append({"base_label": base_label, "center_correction": None,
                                 "error": str(e)})
            continue

        pos_num   = int(re.match(r"Pos(\d+)", base_label).group(1))
        fit_right = pos_num >= pos_split

        # Load per-pos channel_rois.json from REF_GRID_DIR
        # If not found, auto-run channel_crop.py --detect to generate it
        perpos_rois_path = (Path(REF_GRID_DIR) / ref_label
                            / "output_phase" / "channels" / "channel_rois.json")
        if not perpos_rois_path.exists():
            print(f"  [channel_detect] No channel_rois.json -> auto-detecting: {ref_label}")
            _run_channel_detect(Path(REF_GRID_DIR) / ref_label, z_index=REF_Z_INDEX)
        if not perpos_rois_path.exists():
            raise FileNotFoundError(
                f"Failed to generate channel_rois.json: {perpos_rois_path}\n"
                f"Run manually: python scripts/channel_crop.py --dir \"{Path(REF_GRID_DIR) / ref_label}\" --detect"
            )
        with open(perpos_rois_path, encoding="utf-8") as f:
            rois = json.load(f)
        print(f"  channel_rois: {len(rois)} ch ({ref_label})")
        n_ch = len(rois)

        tx_avg, ty_avg, per_ch = center_ecc(ref_img, calib_img, rois, n_ch, vmin, vmax,
                                            fit_right=fit_right)

        if tx_avg is None:
            print(f"  [center correction] All channels ECC failed -> skipping")
            log_entries.append({"base_label": base_label, "center_correction": None,
                                 "error": "all_ecc_failed", "per_channel": per_ch})
            continue

        # Sign convention: identical to calibrate_grid_pos.py L389-394
        drift_stage_x_um = sx_sign * ty_avg * pixel_scale_um   # image Y -> stage X
        drift_stage_y_um = sy_sign * tx_avg * pixel_scale_um   # image X -> stage Y
        pos_correct_x_um = -drift_stage_x_um
        pos_correct_y_um = -drift_stage_y_um

        print(f"  Correction: stage_X={pos_correct_x_um:+.4f}um  stage_Y={pos_correct_y_um:+.4f}um")

        for dev in pos["DEVICES"]:
            if dev["DEVICE"] == "XYStage":
                dev["X"] += pos_correct_x_um
                dev["Y"] += pos_correct_y_um

        correction = {
            "tx_avg_px": tx_avg,
            "ty_avg_px": ty_avg,
            "drift_stage_x_um": drift_stage_x_um,
            "drift_stage_y_um": drift_stage_y_um,
            "pos_correct_x_um": pos_correct_x_um,
            "pos_correct_y_um": pos_correct_y_um,
            "n_channels_total": n_ch,
            "per_channel": per_ch,
        }
        correction_by_label[base_label] = correction
        log_entries.append({"base_label": base_label, "center_correction": correction})

    # ---- Copy corresponding sample correction to BG pos ----
    for bg_label, src_label in BG_COPY_FROM.items():
        print(f"\n=== {bg_label} (BG: copying correction from {src_label}) ===")
        if src_label not in correction_by_label:
            print(f"  WARNING: Correction for {src_label} not computed -> skipping {bg_label}")
            log_entries.append({"base_label": bg_label,
                                 "center_correction": None,
                                 "copied_from": src_label,
                                 "error": "source correction not available"})
            continue

        src_corr = correction_by_label[src_label]
        pos = pos_by_label.get(bg_label)
        if pos is None:
            print(f"  WARNING: {bg_label} not found in timelapse.pos -> skipping")
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
            "center_correction": {
                k: v for k, v in src_corr.items() if k != "per_channel"
            },
        })

    # ---- Save corrected base pos ----
    base_dir = Path(BASE_DIR)
    out_corrected = base_dir / "_per_pos_ecc_corrected.pos"
    with open(out_corrected, "w") as f:
        json.dump(pos_data, f, indent=3)
    print(f"\nCorrected base pos: {out_corrected}  ({len(positions)} positions)")

    # ---- Grid expansion (snake scan: identical to generate_grid_pos.py) ----
    new_positions = []
    for pos in positions:
        base_x = base_y = base_z = 0.0
        for dev in pos["DEVICES"]:
            if dev["DEVICE"] == "XYStage":
                base_x = dev["X"]
                base_y = dev["Y"]
            elif dev["DEVICE"] == "TIPFSOffset":
                base_z = dev["X"]

        for xi in range(-X_HALF, X_HALF + 1):
            row = xi + X_HALF   # 0-indexed
            yi_range = (range(-Y_HALF, Y_HALF + 1) if row % 2 == 0
                        else range(Y_HALF, -Y_HALF - 1, -1))
            for yi in yi_range:
                new_pos = copy.deepcopy(pos)
                new_pos["LABEL"] = f"{pos['LABEL']}_x{xi:+d}_y{yi:+d}"
                for dev in new_pos["DEVICES"]:
                    if dev["DEVICE"] == "XYStage":
                        dev["X"] = base_x + xi * X_STEP
                        dev["Y"] = base_y + yi * Y_STEP
                    elif dev["DEVICE"] == "TIPFSOffset":
                        dev["X"] = base_z
                new_positions.append(new_pos)

    pos_data["POSITIONS"] = new_positions
    out_grid = base_dir / "_per_pos_ecc_grid.pos"
    with open(out_grid, "w") as f:
        json.dump(pos_data, f, indent=3)
    n_base = len(positions)
    print(f"After grid expansion: {out_grid}  "
          f"({len(new_positions)} = {n_base} x {(2*X_HALF+1)*(2*Y_HALF+1)})")

    # ---- Save log ----
    out_log = base_dir / "per_pos_ecc_log.json"
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump({
            "timelapse_pos":   str(TIMELAPSE_POS),
            "calib_grid_dir":  str(CALIB_GRID_DIR),
            "ref_grid_dir":    str(REF_GRID_DIR),
            "ref_z_index":     REF_Z_INDEX,
            "calib_z_index":   CALIB_Z_INDEX,
            "ecc_vmin":        vmin,
            "ecc_vmax":        vmax,
            "tilt_crop_h":     TILT_CROP_H,
            "ecc_crop_h":      ECC_CROP_H,
            "mad_thresh":      MAD_THRESH,
            "x_step_um":       X_STEP,
            "y_step_um":       Y_STEP,
            "x_half":          X_HALF,
            "y_half":          Y_HALF,
            "bg_copy_from":    BG_COPY_FROM,
            "per_pos":         log_entries,
        }, f, indent=2, ensure_ascii=False)
    print(f"Log: {out_log}")

    # ---- correction summary figure ----
    _plot_corrections(log_entries, base_dir)

    print("\n--- Done ---")


if __name__ == "__main__":
    main()

# %%
