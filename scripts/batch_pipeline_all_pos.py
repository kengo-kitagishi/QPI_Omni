"""
batch_pipeline_all_pos.py
-------------------------
Full analysis pipeline for ALL Pos in a timelapse dataset.

Steps per Pos (skipped if already done):
  0. Reconstruction  (raw holo -> phase, with Pos0 BG subtraction)
  1. Copy channel_rois.json from grid data
  2. compute_pos_shifts.py  (ECC shift calculation)
  3. grid_subtract.py       (raw-raw grid subtraction)
  4. correct_0pergluc.py    (0% glucose correction)

Parameters are sourced from drift_config.json and confirmed settings.
Do NOT copy params from pipeline_full.py (stale).

Usage:
    python scripts/batch_pipeline_all_pos.py
"""
import json
import sys
import re
import time
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ============================================================
# Configuration
# ============================================================
TIMELAPSE_ROOT = Path(r"F:\260405\ph_260405")
GRID_2PER_DIR  = Path(r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1")
GRID_0PER_DIR  = Path(r"C:\grid_0pergluc_60ms_1")

# Pos range (inclusive). Pos0 is BG, skip it.
POS_START = 2
POS_END   = 64

# 0% glucose frame range
GLUCOSE_0_START = 575
GLUCOSE_0_END   = 1440   # exclusive

# Reconstruction workers (28 logical cores available)
N_WORKERS_RECON = 8

# compute_pos_shifts workers
N_WORKERS_ECC = 16

# Progress log (for resume)
PROGRESS_LOG = TIMELAPSE_ROOT / "batch_pipeline_progress.json"

# ============================================================
# Confirmed parameters (from drift_config.json + memory)
# ============================================================
POS_SPLIT   = 33
CROP_BEFORE = (0, 2048, 400, 2448)   # Pos < POS_SPLIT
CROP_AFTER  = (0, 2048, 0, 2048)     # Pos >= POS_SPLIT

# Reconstruction
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

# ECC / compute_pos_shifts (confirmed 2026-04-13)
ECC_CROP_H           = 80
TILT_CROP_H          = 270
GRID_Z_INDEX         = 18
OUTLIER_MAD_THRESH   = 5.0
OUTLIER_TS_THRESH    = 0.0    # always disabled
ECC_MIN_CORR         = 0.96
VMIN, VMAX           = -5.0, 2.0

# grid_subtract (confirmed)
TILT_CROP_H_RAW      = 270
RAW_TL_Z_INDEX       = 0
RAW_GRID_Z_INDEX     = 18
SHIFT_SIGN_X         = -1
SHIFT_SIGN_Y         = -1

# ============================================================


def get_crop(pos_num):
    return CROP_BEFORE if pos_num < POS_SPLIT else CROP_AFTER


def load_progress():
    if PROGRESS_LOG.exists():
        return json.loads(PROGRESS_LOG.read_text(encoding="utf-8"))
    return {}


def save_progress(prog):
    PROGRESS_LOG.write_text(
        json.dumps(prog, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def is_step_done(prog, pos_label, step):
    return prog.get(pos_label, {}).get(step, False)


def mark_done(prog, pos_label, step):
    prog.setdefault(pos_label, {})[step] = True
    save_progress(prog)


# ============================================================
# Step 0: Reconstruction
# ============================================================
def _reconstruct_one(args):
    """Reconstruct a single frame (worker function).

    Saves two outputs:
      - output_phase/     : BG-subtracted + region mean subtracted (for ECC)
      - output_phase_raw/ : raw phase, no BG subtraction (for grid_subtract)
    """
    tgt_path_str, bg_path_str, crop, out_path_str, raw_out_path_str, pos_num, pos_split = args
    from PIL import Image
    from qpi import QPIParameters, get_field
    from skimage.restoration import unwrap_phase

    tgt_path = Path(tgt_path_str)
    bg_path  = Path(bg_path_str)
    out_path = Path(out_path_str)
    raw_out_path = Path(raw_out_path_str)
    if out_path.exists() and raw_out_path.exists():
        return True

    rs, re_, cs, ce = crop

    def _recon(p):
        img = np.array(Image.open(str(p)))[rs:re_, cs:ce]
        qp = QPIParameters(
            wavelength=WAVELENGTH, NA=NA,
            img_shape=img.shape, pixelsize=PIXELSIZE,
            offaxis_center=OFFAXIS_CENTER,
        )
        return unwrap_phase(np.angle(get_field(img, qp)))

    try:
        tgt_phase = _recon(tgt_path)

        # Save raw phase (for grid_subtract - no BG subtraction)
        if not raw_out_path.exists():
            raw_out_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(raw_out_path), tgt_phase.astype(np.float32))

        # Save BG-subtracted phase (for ECC alignment)
        if not out_path.exists():
            phase = tgt_phase - _recon(bg_path)
            h, w = phase.shape
            if pos_num < pos_split:
                region = phase[1:h-1, 1:w//2]
            else:
                region = phase[1:h-1, w//2:w-1]
            if region.size > 0:
                phase -= np.mean(region)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(out_path), phase.astype(np.float32))
        return True
    except Exception as e:
        print(f"  [RECON ERROR] {tgt_path.name}: {e}")
        return False


def step0_reconstruct(pos_num, pos_dir):
    """Reconstruct all frames for a Pos.

    Saves both BG-subtracted (output_phase/) and raw (output_phase_raw/) phase.
    """
    crop = get_crop(pos_num)
    bg_dir = TIMELAPSE_ROOT / "Pos0"
    out_dir = pos_dir / "output_phase"
    raw_out_dir = pos_dir / "output_phase_raw"
    out_dir.mkdir(exist_ok=True)
    raw_out_dir.mkdir(exist_ok=True)

    # Find raw holos
    raw_files = sorted(pos_dir.glob("img_*_ph_000.tif"))
    if not raw_files:
        print(f"  [SKIP] No raw holos found")
        return False

    # Check how many are already done (both outputs needed)
    existing_phase = len(list(out_dir.glob("*_phase.tif")))
    existing_raw = len(list(raw_out_dir.glob("*_phase.tif")))
    existing = min(existing_phase, existing_raw)
    if existing >= len(raw_files):
        print(f"  [SKIP] Reconstruction done ({existing}/{len(raw_files)})")
        return True

    print(f"  Reconstructing {len(raw_files)} frames "
          f"({existing_phase} phase, {existing_raw} raw existing, crop={crop})...")

    tasks = []
    for raw in raw_files:
        out_path = out_dir / (raw.stem + "_phase.tif")
        raw_out_path = raw_out_dir / (raw.stem + "_phase.tif")
        bg_path = bg_dir / raw.name
        if not bg_path.exists():
            continue
        if out_path.exists() and raw_out_path.exists():
            continue
        tasks.append((str(raw), str(bg_path), crop, str(out_path),
                      str(raw_out_path), pos_num, POS_SPLIT))

    if not tasks:
        return True

    done = 0
    if N_WORKERS_RECON == 1:
        for t in tqdm(tasks, desc="  recon"):
            if _reconstruct_one(t):
                done += 1
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS_RECON) as ex:
            futures = {ex.submit(_reconstruct_one, t): t for t in tasks}
            for f in tqdm(as_completed(futures), total=len(futures), desc="  recon"):
                if f.result():
                    done += 1

    print(f"  Reconstructed {done}/{len(tasks)} new frames")
    return True


# ============================================================
# Step 1: Copy channel_rois.json from grid data
# ============================================================
def step1_channel_rois(pos_num, pos_dir):
    """Copy channel_rois.json from grid_2per data."""
    channels_dir = pos_dir / "output_phase" / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    dst = channels_dir / "channel_rois.json"

    if dst.exists():
        import json as _json
        _rois = _json.loads(dst.read_text())
        _bad = any(r.get("crop_w") != 40 for r in _rois)
        if not _bad:
            print(f"  [SKIP] channel_rois.json exists (crop_w=40 OK)")
            return True
        print(f"  [OVERWRITE] channel_rois.json has wrong crop_w, replacing")

    src = (GRID_2PER_DIR / f"Pos{pos_num}_x+0_y+0"
           / "output_phase" / "channels" / "channel_rois.json")
    if not src.exists():
        print(f"  [ERROR] Grid channel_rois not found: {src}")
        return False

    import shutil
    shutil.copy2(str(src), str(dst))
    print(f"  Copied channel_rois.json from grid")
    return True


# ============================================================
# Step 2: compute_pos_shifts
# ============================================================
def step2_compute_pos_shifts(pos_num, pos_dir):
    """Run compute_pos_shifts with confirmed parameters."""
    channels_dir = pos_dir / "output_phase" / "channels"
    out_json = channels_dir / "pos_shifts_cal.json"

    # Check if already done with correct frame count
    n_holos = len(list(pos_dir.glob("img_*_ph_000.tif")))
    if out_json.exists():
        try:
            d = json.loads(out_json.read_text(encoding="utf-8"))
            if d.get("n_frames", 0) >= n_holos:
                print(f"  [SKIP] pos_shifts_cal.json exists "
                      f"({d['n_frames']} frames)")
                return True
            print(f"  pos_shifts_cal.json has {d.get('n_frames', 0)} frames, "
                  f"need {n_holos}. Re-running...")
        except Exception:
            pass

    import compute_pos_shifts as cps

    # Override globals
    cps.CHANNELS_DIR = str(channels_dir)
    cps.CHANNEL_ROIS_JSON = str(channels_dir / "channel_rois.json")
    cps.GRID_DIR = str(GRID_2PER_DIR)
    cps.GRID_BASE_LABEL = f"Pos{pos_num}"
    cps.POS_SPLIT = POS_SPLIT
    cps.GRID_Z_INDEX = GRID_Z_INDEX
    cps.GRID_CALIBRATION_JSON = str(
        GRID_2PER_DIR / f"grid_calibration_Pos{pos_num}.json"
    )

    # Confirmed params
    cps.ECC_CROP_H = ECC_CROP_H
    cps.TILT_CROP_H = TILT_CROP_H
    cps.USE_SLOPE_CORRECTION = True
    cps.USE_GRID_REFERENCE = True
    cps.USE_INCREMENTAL_TRACKING = True
    cps.USE_SECOND_PASS_ECC = True
    cps.USE_THIRD_PASS_ECC = True
    cps.OUTLIER_MAD_THRESH = OUTLIER_MAD_THRESH
    cps.OUTLIER_TIMESERIES_THRESH = OUTLIER_TS_THRESH
    cps.ECC_MIN_CORR = ECC_MIN_CORR
    cps.VMIN = VMIN
    cps.VMAX = VMAX
    cps.MAX_FRAMES = None
    cps.N_WORKERS = N_WORKERS_ECC
    cps.TL_Z_INDEX = 0
    cps.SHIFT_SIGN_X = 1
    cps.SHIFT_SIGN_Y = 1
    cps.OUTPUT_JSON = "pos_shifts_cal.json"
    cps.SAVE_CORR_DATA = True
    cps.APPLY_BACKSUB_TO_GRID_REF = True

    # Recompute derived
    cps.TILT_FIT_RIGHT = pos_num >= POS_SPLIT

    print(f"  Running compute_pos_shifts ({n_holos} frames, "
          f"workers={N_WORKERS_ECC})...")
    try:
        cps.main()
        return True
    except Exception as e:
        print(f"  [ERROR] compute_pos_shifts: {e}")
        return False


# ============================================================
# Step 3: grid_subtract (raw-raw)
# ============================================================
def step3_grid_subtract(pos_num, pos_dir):
    """Run grid_subtract in raw-raw mode."""
    channels_dir = pos_dir / "output_phase" / "channels"
    out_dir = channels_dir / "crop_sub_rawraw"

    # Check if already done
    if out_dir.exists():
        ch0_count = len(list((out_dir / "ch00").glob("*.tif"))) if (out_dir / "ch00").exists() else 0
        n_holos = len(list(pos_dir.glob("img_*_ph_000.tif")))
        if ch0_count >= n_holos:
            print(f"  [SKIP] grid_subtract done ({ch0_count} frames)")
            return True
        if ch0_count > 0:
            print(f"  grid_subtract partial ({ch0_count}/{n_holos}). Re-running...")

    import grid_subtract as gs

    crop = get_crop(pos_num)

    gs.TIMELAPSE_DIR = str(pos_dir)
    gs.SHIFTS_JSON = str(channels_dir / "pos_shifts_cal.json")
    gs.CHANNEL_ROIS_JSON = str(channels_dir / "channel_rois.json")
    gs.GRID_DIR = str(GRID_2PER_DIR)
    gs.BASE_LABEL = f"Pos{pos_num}"
    gs.GRID_CALIBRATION_JSON = str(
        GRID_2PER_DIR / f"grid_calibration_Pos{pos_num}.json"
    )
    gs.OUTPUT_DIR = str(out_dir)

    gs.TL_Z_INDEX = 0
    gs.GRID_Z_INDEX = GRID_Z_INDEX
    gs.MAX_FRAMES = None
    gs.PICK_FRAMES = None
    gs.APPLY_SUBPIXEL_CORRECTION = True
    gs.APPLY_INVERSE_SHIFT = False
    gs.OUTPUT_CROP_H = TILT_CROP_H_RAW   # 270: save full tilt-corrected strip
    gs.OUTPUT_SAVE_FULL_FRAME = False

    # raw-raw mode with pre-reconstructed phase from output_phase_raw/
    gs.USE_RAW_PHASE = True
    raw_phase_dir = pos_dir / "output_phase_raw"
    if raw_phase_dir.exists() and len(list(raw_phase_dir.glob("*_phase.tif"))) > 0:
        gs.TL_PHASE_DIR = str(raw_phase_dir)
    else:
        gs.TL_PHASE_DIR = None
    gs.RAW_CROP = crop
    gs.RAW_TL_Z_INDEX = RAW_TL_Z_INDEX
    gs.RAW_GRID_Z_INDEX = RAW_GRID_Z_INDEX
    gs.TILT_CROP_H_RAW = TILT_CROP_H_RAW

    gs.SHIFT_SIGN_X = SHIFT_SIGN_X
    gs.SHIFT_SIGN_Y = SHIFT_SIGN_Y
    gs.X_STEP = 0.1
    gs.Y_STEP = 0.1

    print(f"  Running grid_subtract (raw-raw, crop={crop})...")
    try:
        gs.main()
        return True
    except Exception as e:
        print(f"  [ERROR] grid_subtract: {e}")
        return False


# ============================================================
# Step 4: correct_0pergluc
# ============================================================
def step4_correct_0pergluc(pos_num, pos_dir):
    """Run 0% glucose correction."""
    channels_dir = pos_dir / "output_phase" / "channels"
    out_dir = channels_dir / "crop_sub_rawraw"
    log_file = out_dir / "correct_0pergluc_log.json"

    if log_file.exists():
        print(f"  [SKIP] 0per correction done")
        return True

    if not out_dir.exists():
        print(f"  [ERROR] grid_subtract output not found")
        return False

    import correct_0pergluc as c0

    crop = get_crop(pos_num)

    c0.OUTPUT_DIR = str(out_dir)
    c0.GRID_SUB_LOG = str(channels_dir / "grid_subtract_log.json")
    c0.CHANNEL_ROIS_JSON = str(channels_dir / "channel_rois.json")
    c0.GRID_2PER_DIR = str(GRID_2PER_DIR)
    c0.GRID_0PER_DIR = str(GRID_0PER_DIR)
    c0.BASE_LABEL = f"Pos{pos_num}"
    c0.GRID_CALIBRATION_JSON = str(
        GRID_2PER_DIR / f"grid_calibration_Pos{pos_num}.json"
    )
    c0.GRID_Z_INDEX = GRID_Z_INDEX
    c0.GLUCOSE_0_START = GLUCOSE_0_START
    c0.GLUCOSE_0_END = GLUCOSE_0_END
    c0.RAW_CROP = crop
    c0.TILT_CROP_H_RAW = TILT_CROP_H_RAW
    c0.OUTPUT_CROP_H = None

    print(f"  Running correct_0pergluc (frames {GLUCOSE_0_START}-{GLUCOSE_0_END-1})...")
    try:
        c0.main()
        return True
    except Exception as e:
        print(f"  [ERROR] correct_0pergluc: {e}")
        return False


# ============================================================
# Main
# ============================================================
def main():
    prog = load_progress()
    total = POS_END - POS_START + 1
    t_start = time.time()

    print("=" * 60)
    print(f"Batch pipeline: Pos{POS_START} - Pos{POS_END} ({total} Pos)")
    print(f"  Timelapse: {TIMELAPSE_ROOT}")
    print(f"  Grid 2per: {GRID_2PER_DIR}")
    print(f"  Grid 0per: {GRID_0PER_DIR}")
    print(f"  POS_SPLIT: {POS_SPLIT}")
    print(f"  GLUCOSE_0: [{GLUCOSE_0_START}, {GLUCOSE_0_END})")
    print(f"  Workers: recon={N_WORKERS_RECON}, ecc={N_WORKERS_ECC}")
    print(f"  Progress log: {PROGRESS_LOG}")
    print("=" * 60)

    for pos_num in range(POS_START, POS_END + 1):
        pos_label = f"Pos{pos_num}"
        pos_dir = TIMELAPSE_ROOT / pos_label

        if not pos_dir.exists():
            print(f"\n[{pos_label}] Directory not found, skipping")
            continue

        elapsed = time.time() - t_start
        print(f"\n{'=' * 60}")
        print(f"[{pos_label}] ({pos_num - POS_START + 1}/{total}) "
              f"elapsed={elapsed/60:.0f}min")
        print(f"{'=' * 60}")

        # Step 0: Reconstruction
        if not is_step_done(prog, pos_label, "recon"):
            if step0_reconstruct(pos_num, pos_dir):
                mark_done(prog, pos_label, "recon")
            else:
                print(f"  [FAIL] Reconstruction failed, skipping remaining steps")
                continue

        # Step 1: channel_rois.json
        if not is_step_done(prog, pos_label, "rois"):
            if step1_channel_rois(pos_num, pos_dir):
                mark_done(prog, pos_label, "rois")
            else:
                continue

        # Step 2: compute_pos_shifts
        if not is_step_done(prog, pos_label, "shifts"):
            if step2_compute_pos_shifts(pos_num, pos_dir):
                mark_done(prog, pos_label, "shifts")
            else:
                continue

        # Step 3: grid_subtract
        if not is_step_done(prog, pos_label, "gridsub"):
            if step3_grid_subtract(pos_num, pos_dir):
                mark_done(prog, pos_label, "gridsub")
            else:
                continue

        # Step 4: correct_0pergluc
        if not is_step_done(prog, pos_label, "0per"):
            if step4_correct_0pergluc(pos_num, pos_dir):
                mark_done(prog, pos_label, "0per")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Batch pipeline complete. Total time: {elapsed/3600:.1f} hours")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
