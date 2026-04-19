# %%
"""
run_pipeline.py
---------------
Batch-execute the following steps for all Pos* directories in ROOT_DIR:
  1. channel_crop     : Channel crop -> channels/channel_XX.tif
  2. gaussian_backsub : Background correction -> channels/channel_XX_bg_corr.tif
  3. align_simple     : Verification alignment (optional)
  4. compute_shifts   : Inter-channel mean shift calculation -> channels/pos_shifts.json
  5. grid_subtract    : Grid image subtraction -> channels/grid_subtracted/
"""
import sys
import json
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

# ============================================================
# Target
# ============================================================
ROOT_DIR = r"E:\Acuisition\kitagishi\260301\movetest_8"

# Pos filter: None for all Pos, or specify e.g. ["Pos4", "Pos5"]
POS_FILTER = ["Pos1"]

# Steps to execute (set False to skip)
STEP_CHANNEL_CROP_DETECT = True   # auto-detect if channel_rois.json is missing
STEP_CHANNEL_CROP_APPLY  = True
STEP_GAUSSIAN_BACKSUB    = True
STEP_ALIGN_SIMPLE        = False  # for verification (time-consuming)
STEP_COMPUTE_SHIFTS      = True
STEP_GRID_SUBTRACT       = True

# ============================================================
# channel_crop parameters
# ============================================================
CROP_PATTERN   = "img_*_ph_000.tif"
CROP_W         = 40
CROP_H         = 270
MIN_DIST       = 35
PROMINENCE     = 0.3
X_START        = None   # set to int to skip automatic edge detection
X_END          = None

# ============================================================
# gaussian_backsub parameters
# ============================================================
BACKSUB_MIN_PHASE    = -1.1
BACKSUB_HIST_MIN     = -1.1
BACKSUB_HIST_MAX     = 1.5
BACKSUB_N_BINS       = 512
BACKSUB_SMOOTH_WINDOW = 20
BACKSUB_SAVE_PNG     = False
BACKSUB_PNG_DPI      = 150

# ============================================================
# align_and_subtract_simple parameters (used only when STEP_ALIGN_SIMPLE=True)
# ============================================================
ALIGN_REFERENCE_FRAME  = 150     # 1始まり
ALIGN_METHOD           = 'ecc'
ALIGN_SAVE_PNG         = True
ALIGN_PNG_SAMPLE       = 5
ALIGN_VMIN             = -0.1
ALIGN_VMAX             = 1.7

# ============================================================
# compute_pos_shifts parameters
# ============================================================
SHIFTS_CHANNEL_PATTERN      = "channel_*_bg_corr.tif"  # pattern after backsub

# USE_GRID_REFERENCE=True: use grid x+0_y+0 as reference (recommended)
# USE_GRID_REFERENCE=False: use timelapse frame at SHIFTS_REFERENCE_FRAME as reference
SHIFTS_USE_GRID_REFERENCE   = True
SHIFTS_REFERENCE_FRAME      = 150    # used only when USE_GRID_REFERENCE=False

SHIFTS_METHOD               = 'ecc'
SHIFTS_VMIN                 = -5.0
SHIFTS_VMAX                 = 2.0
SHIFTS_OUTLIER_MAD_THRESH   = 5.0
SHIFTS_TIMESERIES_WINDOW    = 11
SHIFTS_TIMESERIES_THRESH    = 3.0

# ============================================================
# grid_subtract parameters
# ============================================================
GRID_DIR             = r"E:\Acuisition\kitagishi\260301\multipos_test_1"
Z_INDEX              = 2
X_STEP               = 0.1
Y_STEP               = 0.1
SENSOR_PIXEL_SIZE    = 3.45e-6
MAGNIFICATION        = 40
ORIGINAL_DIM         = 2048
RECONSTRUCTED_DIM    = 511
SHIFT_SIGN_X         = 1
SHIFT_SIGN_Y         = 1
APPLY_INVERSE_SHIFT  = False
# ============================================================


# ---- Import functions from each module ----
_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))


def _print_step(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ===========================================================
# Step 1: channel_crop
# ===========================================================
def run_channel_crop(pos_dir: Path):
    from channel_crop import run_detect, run_apply

    img_files = sorted(pos_dir.glob(CROP_PATTERN))
    if not img_files:
        print(f"  [SKIP] No images found: {CROP_PATTERN}")
        return False

    out_dir = pos_dir / "channels"
    out_dir.mkdir(exist_ok=True)
    roi_path = out_dir / "channel_rois.json"

    rois = None
    if STEP_CHANNEL_CROP_DETECT and not roi_path.exists():
        print(f"  [detect] Detecting channels from {img_files[0].name}")
        rois = run_detect(
            img_files[0], CROP_W, CROP_H, out_dir,
            min_dist=MIN_DIST,
            prominence_sigma=PROMINENCE,
            x_start=X_START,
            x_end=X_END,
        )
    elif not roi_path.exists():
        print(f"  [ERROR] channel_rois.json not found and detect is disabled")
        return False

    if STEP_CHANNEL_CROP_APPLY:
        if rois is None:
            with open(roi_path, encoding="utf-8") as f:
                rois = json.load(f)
        print(f"  [apply] {len(img_files)} frames x {len(rois)} channels")
        run_apply(pos_dir, CROP_PATTERN, rois, out_dir)

    return True


# ===========================================================
# Step 2: gaussian_backsub
# ===========================================================
def run_gaussian_backsub(pos_dir: Path):
    """Process each channel_XX.tif (stack) in channels/ one file at a time."""
    import importlib
    import sys as _sys

    # Dynamically import 19_gaussian_backsub.py (starts with a digit)
    import importlib.util
    backsub_path = _script_dir / "19_gaussian_backsub.py"
    spec = importlib.util.spec_from_file_location("gaussian_backsub", backsub_path)
    backsub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backsub)

    # Override parameters
    backsub.minPhase      = BACKSUB_MIN_PHASE
    backsub.hist_min      = BACKSUB_HIST_MIN
    backsub.hist_max      = BACKSUB_HIST_MAX
    backsub.n_bins        = BACKSUB_N_BINS
    backsub.smooth_window = BACKSUB_SMOOTH_WINDOW

    channels_dir = pos_dir / "channels"
    stack_files = sorted(channels_dir.glob("channel_*.tif"))
    # Skip _bg_corr.tif files
    stack_files = [p for p in stack_files if "_bg_corr" not in p.name]

    if not stack_files:
        print(f"  [SKIP] No channel_*.tif found: {channels_dir}")
        return

    for tif_path in stack_files:
        out_path = channels_dir / f"{tif_path.stem}_bg_corr.tif"
        if out_path.exists():
            print(f"  [SKIP already] {tif_path.name}")
            continue
        print(f"  Processing: {tif_path.name}")
        backsub.process_image(tif_path, channels_dir, save_png_data=False)
        if BACKSUB_SAVE_PNG:
            result = backsub.process_image(tif_path, channels_dir, save_png_data=True)
            if isinstance(result, dict):
                backsub.save_png_plots(result, channels_dir, BACKSUB_PNG_DPI)


# ===========================================================
# Step 3: align_and_subtract_simple (for verification)
# ===========================================================
def run_align_simple(pos_dir: Path):
    from align_and_subtract_simple import (
        get_tif_files, load_tif_image, process_timelapse
    )
    channels_dir = pos_dir / "channels"
    stack_files = sorted(channels_dir.glob(SHIFTS_CHANNEL_PATTERN))
    if not stack_files:
        print(f"  [SKIP] {SHIFTS_CHANNEL_PATTERN} not found")
        return

    for stack_path in stack_files:
        print(f"  align_simple: {stack_path.name}")
        # Treat stack as individual frames
        arr = tifffile.imread(str(stack_path))
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        ref_idx = min(ALIGN_REFERENCE_FRAME - 1, arr.shape[0] - 1)
        reference_img = arr[ref_idx].astype(np.float64)

        # Write individual frames to a temp directory, then call process_timelapse
        # -> create a temporary tif list matching the process_timelapse signature
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frame_paths = []
            for i in range(arr.shape[0]):
                p = tmpdir / f"frame_{i:06d}.tif"
                tifffile.imwrite(str(p), arr[i].astype(np.float32))
                frame_paths.append(str(p))

            out_base = channels_dir / stack_path.stem
            process_timelapse(
                str(out_base), reference_img, frame_paths,
                method=ALIGN_METHOD,
                save_png=ALIGN_SAVE_PNG,
                vmin=ALIGN_VMIN, vmax=ALIGN_VMAX,
                png_dpi=150,
                png_sample_interval=ALIGN_PNG_SAMPLE,
            )


# ===========================================================
# Step 4: compute_pos_shifts
# ===========================================================
def run_compute_shifts(pos_dir: Path):
    import compute_pos_shifts as cps

    channels_dir = pos_dir / "channels"

    base_label = pos_dir.name  # e.g. "Pos4"

    # Inject parameters into module-level variables
    cps.CHANNELS_DIR              = str(channels_dir)
    cps.CHANNEL_PATTERN           = SHIFTS_CHANNEL_PATTERN
    cps.USE_GRID_REFERENCE        = SHIFTS_USE_GRID_REFERENCE
    cps.GRID_DIR                  = GRID_DIR
    cps.GRID_BASE_LABEL           = base_label
    cps.GRID_Z_INDEX              = Z_INDEX
    cps.CHANNEL_ROIS_JSON         = str(channels_dir / "channel_rois.json")
    cps.REFERENCE_FRAME           = SHIFTS_REFERENCE_FRAME
    cps.ALIGNMENT_METHOD          = SHIFTS_METHOD
    cps.VMIN                      = SHIFTS_VMIN
    cps.VMAX                      = SHIFTS_VMAX
    cps.OUTLIER_MAD_THRESH        = SHIFTS_OUTLIER_MAD_THRESH
    cps.OUTLIER_TIMESERIES_WINDOW = SHIFTS_TIMESERIES_WINDOW
    cps.OUTLIER_TIMESERIES_THRESH = SHIFTS_TIMESERIES_THRESH

    cps.main()


# ===========================================================
# Step 5: grid_subtract
# ===========================================================
def run_grid_subtract(pos_dir: Path, base_label: str):
    import grid_subtract as gs

    channels_dir = pos_dir / "channels"
    shifts_json  = channels_dir / "pos_shifts.json"
    rois_json    = channels_dir / "channel_rois.json"

    if not shifts_json.exists():
        print(f"  [SKIP] pos_shifts.json not found: {shifts_json}")
        return
    if not rois_json.exists():
        print(f"  [SKIP] channel_rois.json not found: {rois_json}")
        return

    gs.CHANNELS_DIR       = str(channels_dir)
    gs.CHANNEL_PATTERN    = SHIFTS_CHANNEL_PATTERN
    gs.SHIFTS_JSON        = str(shifts_json)
    gs.CHANNEL_ROIS_JSON  = str(rois_json)
    gs.GRID_DIR           = GRID_DIR
    gs.BASE_LABEL         = base_label
    gs.Z_INDEX            = Z_INDEX
    gs.X_STEP             = X_STEP
    gs.Y_STEP             = Y_STEP
    gs.SENSOR_PIXEL_SIZE  = SENSOR_PIXEL_SIZE
    gs.MAGNIFICATION      = MAGNIFICATION
    gs.ORIGINAL_DIM       = ORIGINAL_DIM
    gs.RECONSTRUCTED_DIM  = RECONSTRUCTED_DIM
    gs.SHIFT_SIGN_X       = SHIFT_SIGN_X
    gs.SHIFT_SIGN_Y       = SHIFT_SIGN_Y
    gs.APPLY_INVERSE_SHIFT = APPLY_INVERSE_SHIFT

    gs.main()


# ===========================================================
# Main
# ===========================================================
def main():
    root = Path(ROOT_DIR)
    if not root.exists():
        print(f"ERROR: ROOT_DIR not found: {root}")
        sys.exit(1)

    # List Pos directories
    pos_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("Pos")])
    if POS_FILTER:
        pos_dirs = [d for d in pos_dirs if d.name in POS_FILTER]

    if not pos_dirs:
        print(f"ERROR: No Pos* directories found: {root}")
        sys.exit(1)

    print(f"Target Pos: {[d.name for d in pos_dirs]}")

    errors = []

    for pos_dir in pos_dirs:
        print(f"\n{'#'*60}")
        print(f"  {pos_dir.name}  ({pos_dir})")
        print(f"{'#'*60}")

        base_label = pos_dir.name  # e.g. "Pos4"

        # Step 1: channel_crop
        if STEP_CHANNEL_CROP_DETECT or STEP_CHANNEL_CROP_APPLY:
            _print_step(f"[1] channel_crop  ({pos_dir.name})")
            try:
                ok = run_channel_crop(pos_dir)
                if not ok:
                    errors.append(f"{pos_dir.name}: channel_crop failed")
                    continue
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                errors.append(f"{pos_dir.name}: channel_crop ERROR: {e}")
                continue

        # Step 2: gaussian_backsub
        if STEP_GAUSSIAN_BACKSUB:
            _print_step(f"[2] gaussian_backsub  ({pos_dir.name})")
            try:
                run_gaussian_backsub(pos_dir)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                errors.append(f"{pos_dir.name}: gaussian_backsub ERROR: {e}")
                continue

        # Step 3: align_simple (for verification)
        if STEP_ALIGN_SIMPLE:
            _print_step(f"[3] align_and_subtract_simple  ({pos_dir.name})")
            try:
                run_align_simple(pos_dir)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                errors.append(f"{pos_dir.name}: align_simple ERROR: {e}")

        # Step 4: compute_shifts
        if STEP_COMPUTE_SHIFTS:
            _print_step(f"[4] compute_pos_shifts  ({pos_dir.name})")
            try:
                run_compute_shifts(pos_dir)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                errors.append(f"{pos_dir.name}: compute_shifts ERROR: {e}")
                continue

        # Step 5: grid_subtract
        if STEP_GRID_SUBTRACT:
            _print_step(f"[5] grid_subtract  ({pos_dir.name})")
            try:
                run_grid_subtract(pos_dir, base_label)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                errors.append(f"{pos_dir.name}: grid_subtract ERROR: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Done: {len(pos_dirs)} Pos processed")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All Pos completed successfully")


if __name__ == "__main__":
    main()

# %%
