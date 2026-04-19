"""
run_260310_grid_and_prepare.py
------------------------------
Batch-execute setup steps before starting a timelapse.

Execution order:
  Step 1. Grid reconstruction   : Reconstruct all Pos in GRID_DIR
  Step 2. Channel crop          : Detect & apply on GRID_ORIGIN_DIR/output_phase/
  Step 3. prepare_drift_session : Generate drift_config.json / positions.csv / grid_ref_crops.tif

Usage:
  python run_260310_grid_and_prepare.py

Next step:
  Open realtime_drift_mda.bsh in the MM1.4 Script Panel and click Run.
"""
# %%
import sys
import json
from pathlib import Path

_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

# ============================================================
# *** Modify here for each experiment ***
# ============================================================

# Grid acquisition directory
GRID_DIR        = r"E:\Acuisition\kitagishi\260317_0p0055\grid_0p5_0p5_0p1_exp60ms_allpos_EMM2_1"
GRID_BASE_LABEL = "Pos1"   # Pos label used for drift estimation & channel_crop
GRID_Z_INDEX    = 0        # z-slice index used for grid_ref_crops

# Channel crop settings (overwritten automatically after detection)
CHANNEL_CROP_W  = 40   # y-direction size [px] (channel height)
CHANNEL_CX      = 340  # x center [px] (center of peak position)
CHANNEL_CROP_H  = 80   # x-direction size [px] (peak width)

# Timelapse acquisition settings
POSITIONS_FILE   = r"D:\AquisitionData\Kitagishi\260310\movetest.pos"
SAVE_DIR         = r"C:\ph_1"
REF_POS_INDEX    = 1    # Pos for drift estimation (Pos with sample)
BG_POS_INDEX     = 0    # BG Pos (no cells, for phase correction)
N_TIMEPOINTS     = 3168  # 11 days x 5 min interval (11*24*60/5)
INTERVAL_SEC     = 300   # Interval between timepoints [sec]
EXPOSURE_MS      = 60.0
SETTLE_MS        = 150
PFS_SETTLE_MS    = 200

# Session file output directory
SESSION_DIR      = r"C:\Users\QPI\Documents\QPI_Omni\drift_session"

# ============================================================

if __name__ == "__main__":
    GRID_ORIGIN_DIR = f"{GRID_BASE_LABEL}_x+0_y+0"

    # ============================================================
    # Step 1: Grid reconstruction
    # ============================================================
    print("=" * 60)
    print("Step 1: Grid reconstruction")
    print("=" * 60)

    import pipeline_full as pf

    pf.GRID_DIR                      = GRID_DIR
    pf.STEP_GRID_RECONSTRUCTION      = True
    pf.STEP_TIMELAPSE_RECONSTRUCTION = False
    pf.STEP_CHANNEL_CROP             = False
    pf.STEP_GAUSSIAN_BACKSUB         = False
    pf.STEP_ALIGN_SIMPLE             = False
    pf.STEP_COMPUTE_SHIFTS           = False
    pf.STEP_GRID_SUBTRACT            = False
    pf.GRID_SKIP_IF_EXISTS           = True   # Skip if already reconstructed

    pf.step_grid_reconstruction()
    print("\nStep 1 done\n")


    # ============================================================
    # Step 2: Channel crop (origin only)
    # ============================================================
    print("=" * 60)
    print(f"Step 2: Channel crop ({GRID_ORIGIN_DIR})")
    print("=" * 60)

    phase_dir = Path(GRID_DIR) / GRID_ORIGIN_DIR / "output_phase"
    if not phase_dir.exists():
        print(f"ERROR: output_phase not found: {phase_dir}")
        sys.exit(1)

    channel_rois_json = phase_dir / "channels" / "channel_rois.json"

    # Detect (auto-acquire cy) -> overwrite cx/crop_h with configured values
    pf.CROP_W               = CHANNEL_CROP_W
    pf.CROP_H               = 120
    pf.CROP_FORCE_RECOMPUTE = False
    pf.CROP_FORCE_DETECT    = True   # Always re-detect cy to get the latest position
    pf.CROP_DETECT          = True
    pf.CROP_APPLY           = False  # apply is done after cx/crop_h override

    ok = pf.step_channel_crop(phase_dir)
    if not ok:
        print("ERROR: channel_crop failed")
        sys.exit(1)

    # Keep cy as-is, overwrite cx and crop_h with configured values
    with open(channel_rois_json, encoding="utf-8") as f:
        rois = json.load(f)
    for roi in rois:
        roi["cx"]     = CHANNEL_CX
        roi["crop_w"] = CHANNEL_CROP_W
        roi["crop_h"] = CHANNEL_CROP_H
    with open(channel_rois_json, "w", encoding="utf-8") as f:
        json.dump(rois, f, indent=2)
    print(f"  ROI override: cx={CHANNEL_CX}, crop_w={CHANNEL_CROP_W}, crop_h={CHANNEL_CROP_H}")

    # apply (execute with the overridden ROIs)
    pf.CROP_FORCE_DETECT = False
    pf.CROP_DETECT       = False
    pf.CROP_APPLY        = True
    pf.step_channel_crop(phase_dir)

    print(f"channel_rois.json: {channel_rois_json}")
    print("\nStep 2 done\n")


    # ============================================================
    # Step 3: prepare_drift_session
    # ============================================================
    print("=" * 60)
    print("Step 3: prepare_drift_session")
    print("=" * 60)

    import prepare_drift_session as pds

    pds.POSITIONS_FILE    = POSITIONS_FILE
    pds.GRID_DIR          = GRID_DIR
    pds.GRID_BASE_LABEL   = GRID_BASE_LABEL
    pds.GRID_Z_INDEX      = GRID_Z_INDEX
    pds.CHANNEL_ROIS_JSON = str(channel_rois_json)
    pds.SESSION_DIR       = SESSION_DIR
    pds.SAVE_DIR          = SAVE_DIR
    pds.REF_POS_INDEX     = REF_POS_INDEX
    pds.BG_POS_INDEX      = BG_POS_INDEX
    pds.N_TIMEPOINTS      = N_TIMEPOINTS
    pds.INTERVAL_SEC      = INTERVAL_SEC
    pds.EXPOSURE_MS       = EXPOSURE_MS
    pds.SETTLE_MS         = SETTLE_MS
    pds.PFS_SETTLE_MS     = PFS_SETTLE_MS

    pds.main()
    print("\nStep 3 done\n")


    print("=" * 60)
    print("Setup complete.")
    print(f"  drift_config.json: {SESSION_DIR}/drift_config.json")
    print(f"  Save dir:          {SAVE_DIR}")
    print(f"  N timepoints:      {N_TIMEPOINTS}  interval: {INTERVAL_SEC}s")
    print("")
    print("Next: open realtime_drift_mda.bsh in MM Script Panel and click Run.")
    print("=" * 60)

# %%
