"""
prepare_drift_session.py
------------------------
Pre-flight setup for a real-time drift-corrected QPI time-lapse session.
Run once before launching ``realtime_drift_mda.bsh`` from Micro-Manager.

What this script does:
  1. Read the .pos file used by MM and emit ``positions.csv`` (the simple
     format BeanShell consumes).
  2. Verify that ``grid_calibration_{pos_label}.json`` exists for every
     sample position (BG excluded). Missing files abort the setup -- the
     online drift correction has no nominal-step fallback any more.
  3. Write ``drift_config.json`` with all parameters needed by
     ``compute_drift_online.py`` and ``realtime_drift_mda.bsh``.
  4. Initialise ``drift_state.txt`` and ``drift_log.json``.

Usage:
  python prepare_drift_session.py
"""

import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

# ============================================================
# Edit per experiment
# ============================================================

# .pos file consumed by Micro-Manager (the actual time-lapse position list)
POSITIONS_FILE   = r"D:\AquisitionData\Kitagishi\260331\_focus_check6_ecc_corrected.pos"

# Grid acquisition directory (small grid is fine)
GRID_DIR         = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1"
GRID_Z_INDEX     = 18       # z-slice of the grid TIFFs to use as reference

# channel_rois.json (built beforehand by pipeline_full.py / channel_crop.py)
CHANNEL_ROIS_JSON = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"

# Session working directory (config + state files land here)
SESSION_DIR      = r"C:\Users\QPI\Documents\QPI_Omni\drift_session"

# Time-lapse image save directory (Micro-Manager output)
SAVE_DIR         = r"D:\AquisitionData\Kitagishi\260405\ph_260405"

# Index of the BG position inside the .pos file (0-based; cell-free Pos)
BG_POS_INDEX     = 0

# Micro-Manager acquisition parameters
N_TIMEPOINTS     = 2592
INTERVAL_SEC     = 300        # Time-lapse interval [s]
EXPOSURE_MS      = 60.0
SETTLE_MS        = 150        # Stage settle time after move [ms]
PFS_SETTLE_MS    = 200        # Nikon PFS lock time [ms] (0 to disable)

# Micro-Manager device names
XY_STAGE_DEVICE  = "XYStage"
Z_OFFSET_DEVICE  = "TIPFSOffset"
CHANNEL_NAME     = "ph"
PYTHON_EXE       = r"C:\Users\QPI\AppData\Local\Programs\Python\Python311\python.exe"

# Drift-correction parameters
JUMP_THRESH_UM       = None    # Per-step jump cutoff [um]; None disables
MAX_TOTAL_CORR_UM    = 15.0    # Cumulative correction safety cap [um]
SHIFT_SIGN_X         = 1
SHIFT_SIGN_Y         = 1

# EMA / Kalman filter
CORRECTION_EMA_ALPHA = 0.3
USE_KALMAN_FILTER    = True
# Measured (2026-04-03): stage sigma_y=49.8nm, sigma_x=93.9nm
#                        ECC sigma_ty=9.5nm, sigma_tx=16.6nm
# Q tuned for K~0.80 with beta=0.24 overshoot already absorbed.
KF_Q_TY_NM2          = 291.0
KF_Q_TX_NM2          = 877.0
KF_R_TY_NM2          = 91.0
KF_R_TX_NM2          = 274.0

# Per-position scheduling and ECC controls (consumed by compute_drift_online.py)
DRIFT_SAMPLE_INTERVAL = 1      # 1 = every position; N = every Nth (group leader)
MAX_DRIFT_WORKERS     = 0      # 0 = auto (cpu_count - 4)
ENABLE_THIRD_PASS     = True   # Run pass 3 (re-select grid after pass 2)
ECC_MIN_CORR          = 0.0    # ECC correlation threshold (0 disables filter)

# Optical parameters
SENSOR_PIXEL_SIZE    = 3.45e-6
MAGNIFICATION        = 40
ORIGINAL_DIM         = 2048
RECONSTRUCTED_DIM    = 511

# Position-dependent crop (matches pipeline_full.py)
POS_SPLIT    = 33
CROP_BEFORE  = (0, 2048, 400, 2448)
CROP_AFTER   = (0, 2048,   0, 2048)

# Histogram-based background subtraction
BACKSUB_MIN_PHASE     = -1.1
BACKSUB_HIST_MIN      = -1.1
BACKSUB_HIST_MAX      =  1.5
BACKSUB_N_BINS        = 512
BACKSUB_SMOOTH_WINDOW = 20

# Spatial gradient removal applied before channel cropping (0 = disabled)
GRADIENT_SIGMA = 0

# ECC normalisation range
VMIN = -5.0
VMAX =  2.0

# Tilt-correction crop sizes (must match calibrate_grid_pos_per_pos.py /
# compute_pos_shifts.py); set both to 0 to fall back to plain backsub.
TILT_CROP_H = 270
ECC_CROP_H  = 80

# Crop-subtract / raw-phase Phase B (online crop_sub_rawraw save)
# Step values are nominal fallback only; grid_calibration_*.json (measured)
# wins when present.
RAW_TL_Z_INDEX        = 0
CROP_SUB_X_STEP_UM    = 0.1
CROP_SUB_Y_STEP_UM    = 0.1
ENABLE_CROP_SUB_SAVE  = False
CROP_SUB_ROOT         = r"E:\Acuisition\kitagishi\260405\online_crop_sub"
CROP_SUB_MAX_SECONDS  = 200.0
CROP_SUB_MAX_WORKERS  = 0
CROP_SUB_MIN_FREE_GB  = 2.0

# ============================================================


def main():
    session_dir = Path(SESSION_DIR)
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session directory: {session_dir}")

    # ---- 1. Read .pos -> positions.csv ----
    pos_file = Path(POSITIONS_FILE)
    if not pos_file.exists():
        print(f"ERROR: .pos file not found: {pos_file}")
        sys.exit(1)

    with open(pos_file, encoding="utf-8") as f:
        pos_data = json.load(f)

    positions = []
    for i, pos_entry in enumerate(pos_data["POSITIONS"]):
        label = pos_entry["LABEL"]
        x, y, z_offset = 0.0, 0.0, 0.0
        for dev in pos_entry["DEVICES"]:
            if dev["DEVICE"] == XY_STAGE_DEVICE:
                x = float(dev["X"])
                y = float(dev["Y"])
            elif dev["DEVICE"] == Z_OFFSET_DEVICE:
                z_offset = float(dev["X"])
        positions.append({"index": i, "label": label, "x": x, "y": y, "z_offset": z_offset})
        print(f"  Pos[{i}] {label}: x={x:.3f}, y={y:.3f}, z_offset={z_offset:.4f}")

    n_positions = len(positions)
    print(f"Positions: {n_positions}")

    if BG_POS_INDEX >= n_positions:
        print(f"ERROR: BG_POS_INDEX={BG_POS_INDEX} out of range (0..{n_positions-1})")
        sys.exit(1)

    csv_path = session_dir / "positions.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,label,x,y,z_offset\n")
        for p in positions:
            f.write(f"{p['index']},{p['label']},{p['x']:.6f},{p['y']:.6f},{p['z_offset']:.6f}\n")
    print(f"positions.csv written: {csv_path}")

    # ---- 2. channel_rois.json ----
    rois_path = Path(CHANNEL_ROIS_JSON)
    if not rois_path.exists():
        print(f"ERROR: channel_rois.json not found: {rois_path}")
        sys.exit(1)
    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)
    n_channels = len(rois)
    print(f"Channels: {n_channels}")
    if n_channels == 0:
        print(f"ERROR: channel_rois.json has zero channels: {rois_path}")
        print("  Run channel_crop.py --detect to populate it.")
        sys.exit(1)

    # ---- 3. grid_calibration_{label}.json must exist for every sample Pos ----
    grid_dir = Path(GRID_DIR)
    missing = []
    for p in positions:
        if p["index"] == BG_POS_INDEX:
            continue
        cal_path = grid_dir / f"grid_calibration_{p['label']}.json"
        if not cal_path.exists():
            missing.append(cal_path)
    if missing:
        print("ERROR: grid_calibration JSON missing for the following positions:")
        for m in missing:
            print(f"  - {m}")
        print("Run calibrate_grid_pos_per_pos.py first.")
        sys.exit(1)
    print(f"grid calibration: verified for {n_positions - 1} sample positions")

    # ---- 4. drift_config.json ----
    pixel_scale_um = SENSOR_PIXEL_SIZE / MAGNIFICATION * ORIGINAL_DIM / RECONSTRUCTED_DIM * 1e6
    config = {
        "python_exe":         PYTHON_EXE,
        "script_dir":         str(_script_dir),
        "session_dir":        str(session_dir),
        "save_dir":           SAVE_DIR,
        "positions_csv":      str(csv_path),
        "state_file":         str(session_dir / "drift_state.txt"),
        "log_file":           str(session_dir / "drift_log.json"),
        "channel_rois_json":  str(rois_path),
        "kf_state_file":      str(session_dir / "drift_kf_state.json"),

        # Acquisition
        "n_timepoints":       N_TIMEPOINTS,
        "interval_sec":       INTERVAL_SEC,
        "exposure_ms":        EXPOSURE_MS,
        "settle_ms":          SETTLE_MS,
        "pfs_settle_ms":      PFS_SETTLE_MS,
        "bg_pos_index":       BG_POS_INDEX,
        "n_channels":         n_channels,
        "xy_stage_device":    XY_STAGE_DEVICE,
        "z_offset_device":    Z_OFFSET_DEVICE,
        "channel_name":       CHANNEL_NAME,

        # Drift correction
        "jump_thresh_um":     JUMP_THRESH_UM,
        "max_total_corr_um":  MAX_TOTAL_CORR_UM,
        "shift_sign_x":       SHIFT_SIGN_X,
        "shift_sign_y":       SHIFT_SIGN_Y,
        "pixel_scale_um":     pixel_scale_um,
        "drift_sample_interval": DRIFT_SAMPLE_INTERVAL,
        "max_drift_workers":  MAX_DRIFT_WORKERS,
        "enable_third_pass":  ENABLE_THIRD_PASS,
        "ecc_min_corr":       ECC_MIN_CORR,

        # Reconstruction
        "pos_split":          POS_SPLIT,
        "crop_before":        list(CROP_BEFORE),
        "crop_after":         list(CROP_AFTER),
        "sensor_pixel_size":  SENSOR_PIXEL_SIZE,
        "magnification":      MAGNIFICATION,
        "original_dim":       ORIGINAL_DIM,
        "reconstructed_dim":  RECONSTRUCTED_DIM,

        # Backsub / gradient / ECC normalisation
        "backsub_min_phase":  BACKSUB_MIN_PHASE,
        "backsub_hist_min":   BACKSUB_HIST_MIN,
        "backsub_hist_max":   BACKSUB_HIST_MAX,
        "backsub_n_bins":     BACKSUB_N_BINS,
        "backsub_smooth_window": BACKSUB_SMOOTH_WINDOW,
        "gradient_sigma":     GRADIENT_SIGMA,
        "tilt_crop_h":        TILT_CROP_H,
        "ecc_crop_h":         ECC_CROP_H,
        "ecc_vmin":           VMIN,
        "ecc_vmax":           VMAX,

        # Grid references
        "grid_dir":           GRID_DIR,
        "grid_z_index":       GRID_Z_INDEX,

        # Crop-subtract / raw-phase Phase B (consumed by compute_drift_online.py
        # and recorded in pos_shifts_cal_online.json metadata)
        "raw_grid_z_index":   GRID_Z_INDEX,
        "raw_tl_z_index":     RAW_TL_Z_INDEX,
        "crop_sub_x_step_um": CROP_SUB_X_STEP_UM,
        "crop_sub_y_step_um": CROP_SUB_Y_STEP_UM,
        "tilt_crop_h_raw":    TILT_CROP_H,
        "enable_crop_sub_save": ENABLE_CROP_SUB_SAVE,
        "crop_sub_root":      CROP_SUB_ROOT,
        "crop_sub_max_seconds": CROP_SUB_MAX_SECONDS,
        "crop_sub_max_workers": CROP_SUB_MAX_WORKERS,
        "crop_sub_min_free_gb": CROP_SUB_MIN_FREE_GB,

        # EMA / Kalman
        "correction_ema_alpha": CORRECTION_EMA_ALPHA,
        "use_kalman_filter":  USE_KALMAN_FILTER,
        "kf_Q_ty_nm2":        KF_Q_TY_NM2,
        "kf_Q_tx_nm2":        KF_Q_TX_NM2,
        "kf_R_ty_nm2":        KF_R_TY_NM2,
        "kf_R_tx_nm2":        KF_R_TX_NM2,
    }
    config_path = session_dir / "drift_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"drift_config.json written: {config_path}")

    # ---- 5. drift_state.txt (per-pos format; populated by compute_drift_online.py) ----
    state_path = session_dir / "drift_state.txt"
    with open(state_path, "w", encoding="utf-8") as f:
        f.write("# drift_state.txt - written by compute_drift_online.py\n")
        f.write("STATUS=idle\n")
        f.write("TIMEPOINT=-1\n")
    print(f"drift_state.txt initialised: {state_path}")

    # ---- 6. drift_log.json (archive previous run) ----
    log_path = session_dir / "drift_log.json"
    if log_path.exists():
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        archive_path = session_dir / f"drift_log_{timestamp}.json"
        shutil.copy2(log_path, archive_path)
        print(f"Previous drift_log.json archived: {archive_path}")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump([], f)
    print(f"drift_log.json initialised: {log_path}")

    # ---- 6b. Reset Kalman state ----
    kf_state_path = session_dir / "drift_kf_state.json"
    if kf_state_path.exists():
        kf_state_path.unlink()
        print(f"drift_kf_state.json removed (Kalman state reset): {kf_state_path}")

    # ---- 7. Create per-position save directories ----
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    for p in positions:
        (save_dir / p["label"]).mkdir(exist_ok=True)
    print(f"Save directories created under: {save_dir}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Setup complete. Next steps:")
    print(f"  1. Open MM1.4 Script Panel.")
    print(f"  2. Load realtime_drift_mda.bsh.")
    print(f"  3. Set CONFIG_FILE in the script to:")
    print(f"     {config_path}")
    print(f"  4. Press Run.")
    print(f"\nSave dir:   {SAVE_DIR}")
    print(f"Timepoints: {N_TIMEPOINTS}  Interval: {INTERVAL_SEC}s")
    print(f"BG Pos:     [{BG_POS_INDEX}] {positions[BG_POS_INDEX]['label']}")
    print(f"Pixel scale: {pixel_scale_um:.4f} um/px")


if __name__ == "__main__":
    main()
