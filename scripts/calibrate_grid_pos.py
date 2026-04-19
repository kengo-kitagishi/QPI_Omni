"""
calibrate_grid_pos.py
---------------------
Correct sample position drift caused by temperature changes
and generate grid pos files for each glucose condition.

Acquisition pattern (common for each run):
  - Calibration acquisition: use timelapse.pos with MDA for a single time point
    -> Pos0 (BG), Pos1 (sample) each yield one frame at ph_000
  - grid2per reference: use Pos1_x+0_y+0 from grid_2pergluc_60ms_1 at ph_005 (middle z)
    -> auto-reconstructed if not yet done

Processing workflow:
  Step 0. Auto-reconstruct grid2per reference image (ph_005) if not yet reconstructed
  Step 1. Phase-reconstruct calibration Pos1 at ph_000 (on-the-fly)
  Step 2. Load grid2per reference image (pre-reconstructed ph_005)
  Step 3. ECC alignment per channel (all channels from channel_rois.json)
  Step 4. Channel average (MAD outlier removal)
  Step 5. Image shift -> stage correction (sign identical to compute_drift_online.py)
  Step 6. Apply correction to all positions in timelapse.pos -> save corrected pos
  Step 7. Grid expansion logic from generate_grid_pos.py -> generate per-condition grid pos

Usage:
  Only change CALIB_DIR and CONDITION for each new condition.
  GRID2PER_* is fixed from Day 1 (do not change).

Sign convention (identical to compute_drift_online.py):
  findTransformECC(ref, sample) -> (tx, ty)  [image shift in ref->sample direction]
  drift_stage_x_um = sx_sign * ty * pixel_scale_um  (image Y -> stage X)
  drift_stage_y_um = sy_sign * tx * pixel_scale_um  (image X -> stage Y)
  pos correction = -drift  (direction to cancel drift)
"""

import sys
import json
import copy
import numpy as np
import tifffile
import cv2
from pathlib import Path
from scipy.ndimage import uniform_filter1d, gaussian_filter
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import curve_fit


# ============================================================
# Configuration parameters -- edit here before each run
# ============================================================

# Calibration acquisition folder (single time-point shot with timelapse.pos)
CALIB_DIR      = r"E:\Acuisition\kitagishi\_calib_grid_0p00525pergluc_60ms_1"
CALIB_LABEL    = "Pos1"   # Position used for ECC (for temperature drift calculation)
CALIB_BG_LABEL = "Pos0"   # BG (used during reconstruction, "none" to skip)
# Calibration is acquired with timelapse.pos -> no z-stack -> always a single ph_000 frame
CALIB_Z_INDEX  = 0

# ---- Output -- only change CONDITION for repeated use ----
CONDITION = "0p00525pergluc"
BASE_DIR  = r"D:\AquisitionData\Kitagishi\260321"
# Corrected timelapse.pos (small file before grid expansion)
OUTPUT_POS_CORRECTED = rf"{BASE_DIR}\timelapse_{CONDITION}.pos"
# Grid-expanded pos file (to be loaded into MicroManager MDA)
OUTPUT_GRID_POS      = rf"{BASE_DIR}\timelapse_grid_{CONDITION}_60ms_1.pos"

# ============================================================
# Fixed from Day 1 -- normally do not change
# ============================================================

# grid2per raw data folder (used for reference reconstruction)
GRID2PER_DIR      = r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1"
GRID2PER_LABEL    = "Pos1_x+0_y+0"   # Grid center label
GRID2PER_BG_LABEL = "Pos0_x+0_y+0"   # BG
# z-stack 11 frames (ph_000 to ph_010) -> middle = ph_005
# At the grid acquisition stage, focus is not yet determined so the middle z is used as reference
GRID2PER_Z_INDEX  = 5

# channel_rois.json (ROI definitions used for ECC)
CHANNEL_ROIS_JSON = (
    r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1"
    r"\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"
)

# Pos file to correct (same file as generate_grid_pos.py input)
TIMELAPSE_POS = r"D:\AquisitionData\Kitagishi\260321\timelapse.pos"

# drift_config.json (optical parameters, signs, backsub settings loaded from here)
DRIFT_CONFIG = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json"

# ---- ECC crop ----
ECC_CROP_H = 80   # X-direction crop width for ECC [px] (unified with compute_pos_shifts.py)

# ---- generate_grid_pos parameters (must match generate_grid_pos.py) ----
X_STEP = 0.1   # um (stage X direction, image Y direction)
Y_STEP = 0.1   # um (stage Y direction, image X direction)
X_HALF = 3     # Half grid count -> total 2*X_HALF+1 points
Y_HALF = 3     # Half grid count -> total 2*Y_HALF+1 points

# ============================================================


# ============================================================
# Utility functions (verbatim copy from compute_drift_online.py)
# ============================================================

def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def extract_rect_roi(img, cy, cx, crop_w, crop_h):
    h, w = img.shape
    y1 = cy - crop_w // 2; y2 = y1 + crop_w
    x1 = cx - crop_h // 2; x2 = x1 + crop_h
    pad_y0 = max(0, -y1); y1 = max(0, y1)
    pad_y1 = max(0, y2 - h); y2 = min(h, y2)
    pad_x0 = max(0, -x1); x1 = max(0, x1)
    pad_x1 = max(0, x2 - w); x2 = min(w, x2)
    crop = img[y1:y2, x1:x2]
    if any([pad_y0, pad_y1, pad_x0, pad_x1]):
        crop = np.pad(crop, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant")
    return crop


def to_uint8(img, vmin, vmax):
    clipped = np.clip(img, vmin, vmax)
    return ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def compute_backsub_offset(img, cfg) -> float:
    min_phase = cfg.get("backsub_min_phase", -1.1)
    hist_min  = cfg.get("backsub_hist_min", -1.1)
    hist_max  = cfg.get("backsub_hist_max",  1.5)
    n_bins    = cfg.get("backsub_n_bins", 512)
    smooth_w  = cfg.get("backsub_smooth_window", 20)

    bin_edges = np.linspace(hist_min, hist_max, n_bins + 1)
    hist_counts, _ = np.histogram(img.flatten(), bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0]
    smoothed = uniform_filter1d(hist_counts.astype(float), size=smooth_w, mode='nearest')
    smoothed = uniform_filter1d(smoothed, size=smooth_w, mode='nearest')
    valid_idx = np.where(bin_centers >= min_phase)[0]
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
            maxfev=5000
        )
        return float(-popt[1])
    except Exception:
        return float(-peak_value)


def ecc_align(ref_u8, tl_u8):
    """ECC alignment. Returns (tx, ty, correlation). None on failure."""
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-8)
    try:
        corr, warp_matrix = cv2.findTransformECC(
            ref_u8, tl_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria
        )
        return float(warp_matrix[0, 2]), float(warp_matrix[1, 2]), float(corr)
    except Exception:
        return None


def _mad(arr):
    m = np.median(arr)
    return float(np.median(np.abs(arr - m)))


def _remove_outliers_mad(values, thresh=5.0):
    arr = np.array(values, dtype=np.float64)
    md = _mad(arr)
    if md == 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(arr - np.median(arr)) > thresh * md


def reconstruct_phase(raw_path: Path, cfg: dict, bg_path: Path = None) -> np.ndarray:
    """QPI phase reconstruction. Returns phase difference if bg_path is provided."""
    script_dir = Path(cfg["script_dir"])
    sys.path.insert(0, str(script_dir))

    from PIL import Image
    from qpi import QPIParameters, get_field
    from skimage.restoration import unwrap_phase
    from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

    ref_idx   = cfg["ref_pos_index"]
    pos_split = cfg.get("pos_split", 3)
    if ref_idx < pos_split:
        crop = tuple(cfg["crop_before"])
    else:
        crop = tuple(cfg["crop_after"])
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
        print(f"  BG subtraction: {bg_path.name}")
    else:
        print("  No BG (backsub only)")

    return phase


def postprocess_phase(phase: np.ndarray, cfg: dict) -> np.ndarray:
    """Post-processing after phase reconstruction (mean removal + Gaussian gradient removal).
    Same procedure as compute_drift_online.py main()."""
    h_p, w_p = phase.shape
    region = phase[1:h_p - 1, 1:w_p // 2]
    if region.size > 0:
        phase = phase - np.mean(region)

    gradient_sigma = cfg.get("gradient_sigma", 0)
    if gradient_sigma > 0:
        bg_gauss = gaussian_filter(phase, sigma=gradient_sigma, mode='nearest')
        phase = phase - bg_gauss
        print(f"  Gaussian gradient removal: sigma={gradient_sigma}px")

    return phase


# ============================================================
# Main
# ============================================================

def main():
    cfg = load_config(DRIFT_CONFIG)

    # ---- channel_rois.json ----
    rois_path = Path(CHANNEL_ROIS_JSON)
    if not rois_path.exists():
        print(f"ERROR: channel_rois.json not found: {rois_path}")
        sys.exit(1)
    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)
    n_channels = len(rois)
    print(f"channel_rois: {n_channels} channels")
    print(f"CONDITION: {CONDITION}")

    # ---- Step 0: Auto-reconstruct grid2per reference image (ph_{GRID2PER_Z_INDEX:03d}) if not yet done ----
    grid2per_ref_path = (
        Path(GRID2PER_DIR) / GRID2PER_LABEL / "output_phase"
        / f"img_000000000_ph_{GRID2PER_Z_INDEX:03d}_phase.tif"
    )
    if not grid2per_ref_path.exists():
        print(f"\n[Step 0] grid2per reference image not reconstructed -> auto-reconstructing")
        grid2per_raw = (
            Path(GRID2PER_DIR) / GRID2PER_LABEL
            / f"img_000000000_ph_{GRID2PER_Z_INDEX:03d}.tif"
        )
        grid2per_bg = (
            Path(GRID2PER_DIR) / GRID2PER_BG_LABEL
            / f"img_000000000_ph_{GRID2PER_Z_INDEX:03d}.tif"
        )
        if not grid2per_raw.exists():
            print(f"ERROR: grid2per raw image not found: {grid2per_raw}")
            sys.exit(1)
        if not grid2per_bg.exists():
            print(f"  [WARNING] grid2per BG not found: {grid2per_bg}  -> no BG")
            grid2per_bg = None

        print(f"  raw: {grid2per_raw.name}")
        phase_grid2per = reconstruct_phase(grid2per_raw, cfg, grid2per_bg)
        phase_grid2per = postprocess_phase(phase_grid2per, cfg)

        grid2per_ref_path.parent.mkdir(exist_ok=True)
        tifffile.imwrite(str(grid2per_ref_path), phase_grid2per.astype(np.float32))
        print(f"  grid2per reference image saved: {grid2per_ref_path}")
    else:
        print(f"\n[Step 0] grid2per reference image: already reconstructed -> skipping")
        print(f"  {grid2per_ref_path}")

    # ---- Step 1: Calibration phase reconstruction ----
    raw_path = Path(CALIB_DIR) / CALIB_LABEL / f"img_000000000_ph_{CALIB_Z_INDEX:03d}.tif"
    if CALIB_BG_LABEL.strip().lower() == "none":
        bg_path = None
    else:
        bg_path = (
            Path(CALIB_DIR) / CALIB_BG_LABEL
            / f"img_000000000_ph_{CALIB_Z_INDEX:03d}.tif"
        )
        if not bg_path.exists():
            print(f"  [WARNING] BG not found: {bg_path}  -> reconstructing without BG")
            bg_path = None

    if not raw_path.exists():
        print(f"ERROR: Calibration image not found: {raw_path}")
        sys.exit(1)

    print(f"\n[Step 1] Calibration phase reconstruction: {raw_path}")
    phase_calib = reconstruct_phase(raw_path, cfg, bg_path)
    phase_calib = postprocess_phase(phase_calib, cfg)

    out_phase_dir = raw_path.parent / "output_phase"
    out_phase_dir.mkdir(exist_ok=True)
    out_phase_path = out_phase_dir / (raw_path.stem + "_phase.tif")
    tifffile.imwrite(str(out_phase_path), phase_calib.astype(np.float32))
    print(f"  Phase saved: {out_phase_path}")

    # ---- Step 2: Load grid2per reference image ----
    phase_ref = tifffile.imread(str(grid2per_ref_path)).astype(np.float64)
    print(f"\n[Step 2] grid2per reference image: {grid2per_ref_path.name}  shape={phase_ref.shape}")

    # ---- Step 3: ECC alignment per channel ----
    print(f"\n[Step 3] ECC alignment ({n_channels} channels)...")
    vmin = cfg.get("ecc_vmin", -5.0)
    vmax = cfg.get("ecc_vmax",  2.0)

    def _ecc_one_channel(ch_idx):
        roi = rois[ch_idx]
        ref_crop = extract_rect_roi(
            phase_ref, roi["cy"], roi["cx"], roi["crop_w"], ECC_CROP_H
        ).copy().astype(np.float64)
        cal_crop = extract_rect_roi(
            phase_calib, roi["cy"], roi["cx"], roi["crop_w"], ECC_CROP_H
        ).copy().astype(np.float64)
        ref_crop += compute_backsub_offset(ref_crop, cfg)
        cal_crop += compute_backsub_offset(cal_crop, cfg)
        result = ecc_align(
            to_uint8(ref_crop, vmin, vmax),
            to_uint8(cal_crop, vmin, vmax),
        )
        return ch_idx, result

    tx_list, ty_list, corr_list = [], [], []
    with ThreadPoolExecutor(max_workers=None) as pool:
        for ch_idx, result in pool.map(_ecc_one_channel, range(n_channels)):
            if result is None:
                print(f"  ch{ch_idx:02d}: ECC failed → skip")
                continue
            tx, ty, corr = result
            tx_list.append(tx)
            ty_list.append(ty)
            corr_list.append(corr)
            print(f"  ch{ch_idx:02d}: tx={tx:+.3f}px  ty={ty:+.3f}px  corr={corr:.4f}")

    if not tx_list:
        print("ERROR: ECC failed on all channels")
        sys.exit(1)

    # ---- Step 4: Channel average (MAD outlier removal) ----
    n_raw = len(tx_list)
    if n_raw >= 3:
        out_x  = _remove_outliers_mad(tx_list)
        out_y  = _remove_outliers_mad(ty_list)
        is_out = out_x | out_y
        used   = [i for i, o in enumerate(is_out) if not o]
        excl   = [i for i in range(n_raw) if is_out[i]]
        if not used:
            used = list(range(n_raw))
        if excl:
            print(f"  [outlier removal] idx={excl}: {len(excl)}ch excluded")
    else:
        used = list(range(n_raw))

    tx_arr   = np.array(tx_list)
    ty_arr   = np.array(ty_list)
    corr_arr = np.array(corr_list)
    tx_avg   = float(np.mean(tx_arr[used]))
    ty_avg   = float(np.mean(ty_arr[used]))
    corr_avg = float(np.mean(corr_arr[used]))
    print(f"  ECC average: tx={tx_avg:+.4f}px  ty={ty_avg:+.4f}px  corr={corr_avg:.4f}"
          f"  (used {len(used)}/{n_raw}ch)")

    # ---- Step 5: Image shift -> stage correction ----
    # Identical formula to compute_drift_online.py L505-506
    sx_sign        = cfg.get("shift_sign_x", 1)
    sy_sign        = cfg.get("shift_sign_y", 1)
    pixel_scale_um = cfg["pixel_scale_um"]

    shift_x = tx_avg   # Image X (col) direction shift [px]
    shift_y = ty_avg   # Image Y (row) direction shift [px]

    drift_stage_x_um = sx_sign * shift_y * pixel_scale_um   # Image Y -> stage X
    drift_stage_y_um = sy_sign * shift_x * pixel_scale_um   # Image X -> stage Y

    # pos correction = -drift (direction to cancel drift)
    pos_correct_x_um = -drift_stage_x_um
    pos_correct_y_um = -drift_stage_y_um

    print(f"\n[Step 5] Shift -> stage correction")
    print(f"  drift:   stage_X={drift_stage_x_um:+.4f}μm  stage_Y={drift_stage_y_um:+.4f}μm")
    print(f"  correct: stage_X={pos_correct_x_um:+.4f}μm  stage_Y={pos_correct_y_um:+.4f}μm")

    # ---- Step 6: Apply correction to all positions in timelapse.pos ----
    with open(TIMELAPSE_POS, "r") as f:
        pos_data = json.load(f)

    n_pos = len(pos_data["POSITIONS"])
    for pos in pos_data["POSITIONS"]:
        for dev in pos["DEVICES"]:
            if dev["DEVICE"] == "XYStage":
                dev["X"] += pos_correct_x_um
                dev["Y"] += pos_correct_y_um

    out_corrected = Path(OUTPUT_POS_CORRECTED)
    with open(out_corrected, "w") as f:
        json.dump(pos_data, f, indent=3)
    print(f"\n[Step 6] Corrected pos saved: {out_corrected}  ({n_pos} positions)")

    # ---- Step 7: Grid expansion using generate_grid_pos logic ----
    # Inline reuse of expansion logic from generate_grid_pos.py
    new_positions = []
    for orig in pos_data["POSITIONS"]:
        base_label = orig["LABEL"]
        base_x, base_y, base_z_offset = 0.0, 0.0, 0.0
        for dev in orig["DEVICES"]:
            if dev["DEVICE"] == "XYStage":
                base_x = dev["X"]
                base_y = dev["Y"]
            elif dev["DEVICE"] == "TIPFSOffset":
                base_z_offset = dev["X"]

        for xi in range(-X_HALF, X_HALF + 1):
            for yi in range(-Y_HALF, Y_HALF + 1):
                new_pos = copy.deepcopy(orig)
                new_pos["LABEL"] = f"{base_label}_x{xi:+d}_y{yi:+d}"
                for dev in new_pos["DEVICES"]:
                    if dev["DEVICE"] == "XYStage":
                        dev["X"] = base_x + xi * X_STEP
                        dev["Y"] = base_y + yi * Y_STEP
                    elif dev["DEVICE"] == "TIPFSOffset":
                        dev["X"] = base_z_offset
                new_positions.append(new_pos)

    pos_data["POSITIONS"] = new_positions
    out_grid = Path(OUTPUT_GRID_POS)
    with open(out_grid, "w") as f:
        json.dump(pos_data, f, indent=3)
    n_new = len(new_positions)
    print(f"\n[Step 7] Grid-expanded pos saved: {out_grid}")
    print(f"  Original positions: {n_pos}  "
          f"After grid expansion: {n_new}  "
          f"({n_pos} x {2*X_HALF+1} x {2*Y_HALF+1})")

    print(f"\n--- Done ---")
    print(f"  Load {out_grid.name} into MicroManager,")
    print(f"  and visually verify Pos1_x+0_y+0 (should match grid2per Pos1_x+0_y+0).")


if __name__ == "__main__":
    main()
