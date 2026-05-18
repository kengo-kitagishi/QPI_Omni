"""
batch_pipeline_all_pos.py
-------------------------
Full analysis pipeline for ALL Pos in a timelapse dataset.

On every launch (batch_pipeline_progress.json is NOT used for resumption):
  - First: Grid 0%% reconstruction (batch_reconstruction_grid.py)
  - Delete batch_pipeline_progress.json, remove output_phase / output_phase_raw
    for each Pos in the POS_START–POS_END range, then run reconstruction through step4

Steps per Pos:
  0. Reconstruction  (raw holo -> phase, with Pos0 BG subtraction)
  1. Copy channel_rois.json from grid data
  2. compute_pos_shifts.py  (ECC shift calculation)
  3. grid_subtract.py       (raw-raw grid subtraction)
  4. correct_0pergluc.py    (0% glucose correction)

Parameters are sourced from drift_config.json and confirmed settings.
Do NOT copy params from pipeline_full.py (stale).

Usage:
    python scripts/batch_pipeline_all_pos.py
    python scripts/batch_pipeline_all_pos.py --skip-grid-0per   # skip only 0%% grid reconstruction
"""
import argparse
import json
import shutil
import subprocess
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
TIMELAPSE_ROOT = Path(r"F:\260405_acute_z18_200h\ph_260405")
GRID_2PER_DIR  = Path(r"F:\260405_acute_z18_200h\grid_2pergluc_60ms_1")
GRID_0PER_DIR  = Path(r"F:\260405_acute_z18_200h\grid_0pergluc_60ms_1")

# Pos range (inclusive). Pos0 is BG, skip it. Pos3 and Pos5 are missing on disk
# and will be skipped automatically by the existence check.
POS_START = 1
POS_END   = 64

# 0% glucose frame range (user 2026-05-19: 575..1440 → END=1440 exclusive,
# i.e. last 0% frame is 1439, recovery starts at 1440)
GLUCOSE_0_START = 575
GLUCOSE_0_END   = 1440   # exclusive

# Reconstruction workers (28 logical cores available)
N_WORKERS_RECON = 24

# compute_pos_shifts workers
N_WORKERS_ECC = 24

# Progress log for the current run (not read on next launch; deleted at startup)
PROGRESS_LOG = TIMELAPSE_ROOT / "batch_pipeline_progress.json"

# BG (Pos0) phase cache dirs — precomputed once per crop, reused across all Pos
BG_CACHE_BEFORE = TIMELAPSE_ROOT / "Pos0" / "bg_phase_before"
BG_CACHE_AFTER  = TIMELAPSE_ROOT / "Pos0" / "bg_phase_after"

# ============================================================
# Confirmed parameters (from drift_config.json + memory)
# ============================================================
POS_SPLIT   = 33
CROP_BEFORE = (0, 2048, 400, 2448)   # Pos < POS_SPLIT
CROP_AFTER  = (0, 2048, 0, 2048)     # Pos >= POS_SPLIT

# Reconstruction — canonical functions from batch_reconstruction_grid
from batch_reconstruction_grid import reconstruct_from_holo, reconstruct_image, make_qpi_params
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


def save_progress(prog):
    PROGRESS_LOG.write_text(
        json.dumps(prog, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def mark_done(prog, pos_label, step):
    prog.setdefault(pos_label, {})[step] = True
    save_progress(prog)


def step_grid_0per_reconstruction(grid_dir: Path) -> bool:
    """Run batch_reconstruction_grid on GRID_0PER_DIR (reference for correct_0pergluc)."""
    script = _SCRIPT_DIR / "batch_reconstruction_grid.py"
    if not script.is_file():
        print(f"  [ERROR] Not found: {script}")
        return False
    if not grid_dir.is_dir():
        print(f"  [ERROR] Grid 0per directory not found: {grid_dir}")
        return False
    print("\n" + "=" * 60)
    print("Grid 0% reconstruction (batch_reconstruction_grid.py)")
    print(f"  {grid_dir}")
    print("=" * 60)
    r = subprocess.run(
        [sys.executable, str(script), "--grid-dir", str(grid_dir)],
        cwd=str(_SCRIPT_DIR),
    )
    if r.returncode != 0:
        print(f"  [ERROR] batch_reconstruction_grid exited with {r.returncode}")
        return False
    print("Grid 0% reconstruction finished.\n")
    return True


def _reset_timelapse_for_full_run():
    """Ignore progress and start from scratch: delete log + remove reconstruction outputs for each Pos."""
    if PROGRESS_LOG.exists():
        PROGRESS_LOG.unlink()
    for pos_num in range(POS_START, POS_END + 1):
        pos_dir = TIMELAPSE_ROOT / f"Pos{pos_num}"
        if not pos_dir.is_dir():
            continue
        for sub in ("output_phase", "output_phase_raw"):
            p = pos_dir / sub
            if p.is_dir():
                shutil.rmtree(p)


# ============================================================
# Step 0: Reconstruction
# ============================================================
# ============================================================
# BG phase shared_memory cache (worker-side globals)
# ============================================================
_BG_SHM = None      # SharedMemory instance attached in the worker (kept alive)
_BG_CACHE = {}      # stem -> ndarray view into shared_memory


def _init_recon_worker(shm_name, shape, dtype_str, stems):
    """ProcessPool initializer: attach to BG shared_memory and build stem -> view dict.

    Main process creates a single shm segment with all BG phase frames stacked
    along axis 0. Each worker attaches once and exposes a read-only dict keyed
    by the cache TIF stem (e.g. 'img_000000000_ph_000_phase').
    """
    global _BG_SHM, _BG_CACHE
    from multiprocessing import shared_memory
    _BG_SHM = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_BG_SHM.buf)
    _BG_CACHE = {stem: arr[i] for i, stem in enumerate(stems)}


def _build_bg_shm(cache_dir, label):
    """Load all BG cache TIFs from `cache_dir` into one shared_memory segment.

    Returns (shm, (shm_name, shape, dtype_str, stems)) — the second tuple is
    what gets passed to _init_recon_worker via `initargs`.
    Caller is responsible for shm.close() + shm.unlink() when done.
    """
    from multiprocessing import shared_memory
    cache_files = sorted(cache_dir.glob("*_phase.tif"))
    if not cache_files:
        raise FileNotFoundError(f"No BG cache TIFs in {cache_dir}")
    first = tifffile.imread(str(cache_files[0]))
    dtype = first.dtype
    shape = (len(cache_files), first.shape[0], first.shape[1])
    nbytes = int(np.prod(shape)) * dtype.itemsize
    print(f"  [{label}] BG shared_memory: {len(cache_files)} frames x {first.shape} "
          f"{dtype} = {nbytes/1e9:.2f} GB")
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr[0] = first
    stems = [cache_files[0].stem]
    for i, f in enumerate(cache_files[1:], start=1):
        arr[i] = tifffile.imread(str(f))
        stems.append(f.stem)
    return shm, (shm.name, shape, str(dtype), stems)


def _release_bg_shm(shm):
    if shm is None:
        return
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass


def _bg_recon_one(args):
    """Worker: reconstruct one Pos0 BG frame for a given crop and save as float32 tif."""
    bg_path_str, crop, out_path_str = args
    out_path = Path(out_path_str)
    if out_path.exists():
        return True
    try:
        phase = reconstruct_from_holo(bg_path_str, crop)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(out_path), phase.astype(np.float32))
        return True
    except Exception as e:
        print(f"  [BG RECON ERROR] {Path(bg_path_str).name}: {e}")
        return False


def _precompute_bg_cache(crop, cache_dir, label):
    """Precompute BG phase cache for all Pos0 holos at `crop`."""
    bg_dir = TIMELAPSE_ROOT / "Pos0"
    bg_files = sorted(bg_dir.glob("img_*_ph_000.tif"))
    if not bg_files:
        print(f"  [ERROR] No BG holos in {bg_dir}")
        return False
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for bg in bg_files:
        out_path = cache_dir / (bg.stem + "_phase.tif")
        if out_path.exists():
            continue
        tasks.append((str(bg), crop, str(out_path)))
    if not tasks:
        print(f"  [{label}] BG cache complete ({len(bg_files)} frames) at {cache_dir}")
        return True
    print(f"  [{label}] Precomputing BG cache ({len(tasks)} frames, crop={crop}) -> {cache_dir}")
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS_RECON) as ex:
        futures = {ex.submit(_bg_recon_one, t): t for t in tasks}
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"  bg_recon[{label}]"):
            if f.result():
                done += 1
    print(f"  [{label}] BG cache done: {done}/{len(tasks)}")
    return done == len(tasks)


def _reconstruct_one(args):
    """Reconstruct a single frame (worker function).

    Saves two outputs:
      - output_phase/     : BG-subtracted + region mean subtracted (for ECC)
      - output_phase_raw/ : raw phase, no BG subtraction (for grid_subtract)

    BG phase is loaded from precomputed cache (bg_cache_path) instead of
    re-reconstructed per target — saves 1 unwrap_phase per frame.
    """
    tgt_path_str, bg_cache_path_str, crop, out_path_str, raw_out_path_str, pos_num, pos_split = args

    tgt_path = Path(tgt_path_str)
    out_path = Path(out_path_str)
    raw_out_path = Path(raw_out_path_str)
    if out_path.exists() and raw_out_path.exists():
        return True

    try:
        tgt_phase = reconstruct_from_holo(tgt_path, crop)

        # Save raw phase (for grid_subtract - no BG subtraction)
        if not raw_out_path.exists():
            raw_out_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(raw_out_path), tgt_phase.astype(np.float32))

        # Save BG-subtracted phase (for ECC alignment)
        if not out_path.exists():
            bg_stem = Path(bg_cache_path_str).stem
            if _BG_CACHE:
                bg_phase = _BG_CACHE[bg_stem].astype(np.float64)
            else:
                bg_phase = tifffile.imread(bg_cache_path_str).astype(np.float64)
            phase = tgt_phase - bg_phase
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


def step0_reconstruct(pos_num, pos_dir, bg_shm_initargs=None):
    """Reconstruct all frames for a Pos.

    Saves both BG-subtracted (output_phase/) and raw (output_phase_raw/) phase.

    bg_shm_initargs: tuple (shm_name, shape, dtype_str, stems) passed to the
    worker pool initializer so workers read BG phase from shared_memory instead
    of disk. If None, workers fall back to tifffile.imread (per-frame disk I/O).
    """
    crop = get_crop(pos_num)
    bg_cache_dir = BG_CACHE_BEFORE if pos_num < POS_SPLIT else BG_CACHE_AFTER
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
        bg_cache_path = bg_cache_dir / (raw.stem + "_phase.tif")
        if not bg_cache_path.exists():
            print(f"  [ERROR] BG cache missing: {bg_cache_path}")
            return False
        if out_path.exists() and raw_out_path.exists():
            continue
        tasks.append((str(raw), str(bg_cache_path), crop, str(out_path),
                      str(raw_out_path), pos_num, POS_SPLIT))

    if not tasks:
        return True

    done = 0
    if N_WORKERS_RECON == 1:
        # Single-process: initialize BG cache locally so the lookup matches workers
        if bg_shm_initargs is not None:
            _init_recon_worker(*bg_shm_initargs)
        for t in tqdm(tasks, desc="  recon"):
            if _reconstruct_one(t):
                done += 1
    else:
        pool_kwargs = {"max_workers": N_WORKERS_RECON}
        if bg_shm_initargs is not None:
            pool_kwargs["initializer"] = _init_recon_worker
            pool_kwargs["initargs"] = bg_shm_initargs
        with ProcessPoolExecutor(**pool_kwargs) as ex:
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
# Online consume: reuse crop-subtracted TIFs saved by compute_drift_online.py
# ============================================================
def _check_online_complete(online_pos_dir: Path, n_holos: int, n_channels: int) -> bool:
    """Return True if online Phase B wrote a complete set of TIFs + JSON for this Pos."""
    if not online_pos_dir.is_dir():
        return False
    cs_root = online_pos_dir / "output_phase" / "channels" / "crop_sub_rawraw"
    js_path = online_pos_dir / "output_phase" / "channels" / "pos_shifts_cal_online.json"
    if not cs_root.is_dir() or not js_path.is_file():
        return False
    try:
        js = json.loads(js_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    fr = js.get("frame_results", [])
    n_seen = sum(1 for x in fr if x is not None)
    if n_seen < n_holos:
        return False
    for ch in range(n_channels):
        ch_dir = cs_root / f"ch{ch:02d}"
        if not ch_dir.is_dir():
            return False
        if len(list(ch_dir.glob("*.tif"))) < n_holos:
            return False
    return True


def _synthesize_grid_subtract_log(online_json_path: Path, pos_num: int,
                                   raw_crop) -> dict:
    """Build a grid_subtract_log.json dict from pos_shifts_cal_online.json.

    correct_0pergluc reads `frame_log[*].grid_xi/grid_yi/frame_index` and
    top-level `base_label / grid_z_index / raw_crop / ...`. We materialize
    those fields so Step 4 works bit-for-bit.
    """
    js = json.loads(online_json_path.read_text(encoding="utf-8"))
    fr = js.get("frame_results", [])
    base_label = f"Pos{pos_num}"
    frame_log = []
    for entry in fr:
        if entry is None:
            continue
        xi = int(entry["grid_xi"])
        yi = int(entry["grid_yi"])
        frame_log.append({
            "frame_index": int(entry["frame_index"]),
            "shift_x_avg_px": float(entry.get("shift_x_avg", 0.0)),
            "shift_y_avg_px": float(entry.get("shift_y_avg", 0.0)),
            "grid_xi": xi,
            "grid_yi": yi,
            "grid_pos_label": f"{base_label}_x{xi:+d}_y{yi:+d}",
            "grid_nearest_dist_um": entry.get("grid_nearest_dist_um"),
            "residual_x_px": float(entry.get("residual_x_px", 0.0)),
            "residual_y_px": float(entry.get("residual_y_px", 0.0)),
            "is_outlier_timeseries": False,
        })
    frame_log.sort(key=lambda e: e["frame_index"])
    return {
        "source": "online_crop_sub",
        "base_label": base_label,
        "grid_dir": str(GRID_2PER_DIR),
        "grid_z_index": RAW_GRID_Z_INDEX,
        "tl_z_index": RAW_TL_Z_INDEX,
        "x_step_um": 0.1,
        "y_step_um": 0.1,
        "shift_sign_x": SHIFT_SIGN_X,
        "shift_sign_y": SHIFT_SIGN_Y,
        "apply_subpixel_correction": True,
        "apply_inverse_shift": False,
        "use_raw_phase": True,
        "raw_crop": list(raw_crop),
        "tilt_crop_h_raw": TILT_CROP_H_RAW,
        "frame_log": frame_log,
    }


def _consume_online_pos(online_root: Path, pos_num: int, pos_dir: Path,
                        n_channels: int, move: bool = False) -> bool:
    """Copy/move online crop-sub TIFs into pos_dir and synthesize grid_subtract_log.json.

    Returns True on success. Assumes _check_online_complete already passed.
    """
    pos_label = f"Pos{pos_num}"
    online_pos = online_root / pos_label
    online_cs = online_pos / "output_phase" / "channels" / "crop_sub_rawraw"
    online_js = online_pos / "output_phase" / "channels" / "pos_shifts_cal_online.json"

    channels_dir = pos_dir / "output_phase" / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    out_cs = channels_dir / "crop_sub_rawraw"
    out_cs.mkdir(parents=True, exist_ok=True)

    raw_crop = get_crop(pos_num)

    # Copy/move per-channel TIFs
    total_copied = 0
    for ch in range(n_channels):
        src_dir = online_cs / f"ch{ch:02d}"
        dst_dir = out_cs / f"ch{ch:02d}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.glob("*.tif"):
            dst = dst_dir / src.name
            if dst.exists():
                continue
            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            total_copied += 1

    # Synthesize grid_subtract_log.json
    log = _synthesize_grid_subtract_log(online_js, pos_num, raw_crop)
    log_path = channels_dir / "grid_subtract_log.json"
    tmp = log_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    import os as _os
    _os.replace(str(tmp), str(log_path))

    print(f"  [ONLINE] {'moved' if move else 'copied'} {total_copied} TIFs; "
          f"synthesized grid_subtract_log.json ({len(log['frame_log'])} entries)")
    return True


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
    c0.ECC_CROP_H = ECC_CROP_H
    c0.OUTPUT_CROP_H = None
    c0.POS_NUMBERS_TO_RUN = []
    c0.POS_SPLIT = POS_SPLIT
    c0.FIT_RIGHT = None

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
    ap = argparse.ArgumentParser(description="Full batch pipeline (runs grid 0%% + all timelapse steps from scratch every time)")
    ap.add_argument(
        "--skip-grid-0per",
        action="store_true",
        help="Skip only the initial Grid 0%% reconstruction (when batch_reconstruction_grid has already been run)",
    )
    ap.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not delete output_phase / output_phase_raw; resume from existing artifacts",
    )
    ap.add_argument(
        "--consume-online-crop-sub",
        type=str,
        default=None,
        metavar="CROP_SUB_ROOT",
        help="Reuse crop_sub_rawraw written by compute_drift_online.py. "
             "If complete per Pos, skip step0/2/3 and run only step4. "
             "Incomplete Pos fall back to the standard pipeline.",
    )
    ap.add_argument(
        "--move-online-tifs",
        action="store_true",
        help="Use with --consume-online-crop-sub. Move online TIFs instead of copying "
             "(frees disk space on E:).",
    )
    args = ap.parse_args()

    consume_online_root = Path(args.consume_online_crop_sub) if args.consume_online_crop_sub else None
    if consume_online_root is not None and not consume_online_root.is_dir():
        print(f"[ERROR] --consume-online-crop-sub root not found: {consume_online_root}")
        sys.exit(1)
    if consume_online_root is not None:
        print(f"[ONLINE CONSUME] reusing crop-sub TIFs from: {consume_online_root}")

    if not args.skip_grid_0per:
        if not step_grid_0per_reconstruction(GRID_0PER_DIR):
            print("Aborting: Grid 0% reconstruction failed.")
            sys.exit(1)

    if consume_online_root is not None:
        print("[ONLINE CONSUME] skipping full reset; per-Pos consume path will "
              "reuse online TIFs. Incomplete Pos fall back to full pipeline.")
    elif args.no_reset:
        print("[NO-RESET] keeping existing output_phase / output_phase_raw / progress log")
    else:
        print(
            "[RESET] Ignoring batch_pipeline_progress.json - removing it and "
            f"output_phase / output_phase_raw under Pos{POS_START}-Pos{POS_END} ..."
        )
        _reset_timelapse_for_full_run()

    # Precompute Pos0 BG phase cache once per crop (reused across all target Pos)
    # Skipped in online-consume mode because BG-subtracted output_phase is not
    # needed when consuming pre-saved crop-sub TIFs. If a Pos falls back, we
    # lazily build BG cache inside step0 via _activate_bg.
    if consume_online_root is None:
        print("\n" + "=" * 60)
        print("Precomputing Pos0 BG phase cache (both crops)")
        print("=" * 60)
        need_before = any(p < POS_SPLIT for p in range(POS_START, POS_END + 1))
        need_after  = any(p >= POS_SPLIT for p in range(POS_START, POS_END + 1))
        if need_before:
            if not _precompute_bg_cache(CROP_BEFORE, BG_CACHE_BEFORE, "BEFORE"):
                print("Aborting: BG cache (BEFORE) failed.")
                sys.exit(1)
        if need_after:
            if not _precompute_bg_cache(CROP_AFTER, BG_CACHE_AFTER, "AFTER"):
                print("Aborting: BG cache (AFTER) failed.")
                sys.exit(1)

    prog = {}
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
    print(f"  Progress log (this run only): {PROGRESS_LOG}")
    print("=" * 60)

    # Active BG shared_memory: load on demand per crop group, release on switch.
    # On a 64 GB workstation we cannot afford both groups (~46 GB) simultaneously,
    # so swap when crossing POS_SPLIT.
    bg_shm_obj = None
    bg_shm_args = None
    bg_group_loaded = None

    def _activate_bg(group):
        nonlocal bg_shm_obj, bg_shm_args, bg_group_loaded
        if bg_group_loaded == group:
            return bg_shm_args
        if bg_shm_obj is not None:
            print(f"  [BG shm] releasing {bg_group_loaded}")
            _release_bg_shm(bg_shm_obj)
            bg_shm_obj = None
            bg_shm_args = None
            bg_group_loaded = None
        cache_dir = BG_CACHE_BEFORE if group == "BEFORE" else BG_CACHE_AFTER
        try:
            bg_shm_obj, bg_shm_args = _build_bg_shm(cache_dir, group)
            bg_group_loaded = group
            return bg_shm_args
        except Exception as e:
            print(f"  [BG shm] {group} build failed: {e}  -> fallback to disk reads")
            bg_shm_obj = None
            bg_shm_args = None
            bg_group_loaded = None
            return None

    try:
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

            # ONLINE CONSUME path: if flag set and online TIFs are complete,
            # skip step0 (recon), step2 (pos_shifts), step3 (grid_subtract)
            # and run only step1 (channel_rois) + step4 (correct_0pergluc).
            use_online = False
            if consume_online_root is not None:
                n_holos = len(list(pos_dir.glob("img_*_ph_000.tif")))
                # channel_rois lives in grid_2per; fetch once to know n_channels
                src_rois = (GRID_2PER_DIR / f"Pos{pos_num}_x+0_y+0"
                            / "output_phase" / "channels" / "channel_rois.json")
                n_channels = 0
                if src_rois.exists():
                    try:
                        n_channels = len(json.loads(src_rois.read_text(encoding="utf-8")))
                    except Exception:
                        n_channels = 0
                if n_channels > 0 and _check_online_complete(
                        consume_online_root / pos_label, n_holos, n_channels):
                    use_online = True
                    print(f"  [ONLINE] complete ({n_holos} frames, {n_channels} ch) — "
                          f"skipping step0/2/3")
                else:
                    print(f"  [ONLINE] incomplete or missing — falling back to full pipeline")

            if use_online:
                # Step 1: channel_rois (needed for step4)
                if not step1_channel_rois(pos_num, pos_dir):
                    continue
                mark_done(prog, pos_label, "rois")

                # Consume: copy/move TIFs, synthesize grid_subtract_log.json
                if not _consume_online_pos(consume_online_root, pos_num, pos_dir,
                                            n_channels, move=args.move_online_tifs):
                    print(f"  [FAIL] consume online failed, skipping")
                    continue
                mark_done(prog, pos_label, "gridsub")

                # Step 4: correct_0pergluc
                if step4_correct_0pergluc(pos_num, pos_dir):
                    mark_done(prog, pos_label, "0per")
                continue

            # --- Full offline path ---
            group = "BEFORE" if pos_num < POS_SPLIT else "AFTER"
            current_bg_args = _activate_bg(group)

            # Step 0: Reconstruction
            if step0_reconstruct(pos_num, pos_dir, bg_shm_initargs=current_bg_args):
                mark_done(prog, pos_label, "recon")
            else:
                print(f"  [FAIL] Reconstruction failed, skipping remaining steps")
                continue

            # Step 1: channel_rois.json
            if step1_channel_rois(pos_num, pos_dir):
                mark_done(prog, pos_label, "rois")
            else:
                continue

            # Step 2: compute_pos_shifts
            if step2_compute_pos_shifts(pos_num, pos_dir):
                mark_done(prog, pos_label, "shifts")
            else:
                continue

            # Step 3: grid_subtract
            if step3_grid_subtract(pos_num, pos_dir):
                mark_done(prog, pos_label, "gridsub")
            else:
                continue

            # Step 4: correct_0pergluc
            if step4_correct_0pergluc(pos_num, pos_dir):
                mark_done(prog, pos_label, "0per")
    finally:
        _release_bg_shm(bg_shm_obj)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Batch pipeline complete. Total time: {elapsed/3600:.1f} hours")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
