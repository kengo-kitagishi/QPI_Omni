# %%
"""
batch_reconstruction_grid.py
----------------------------
Batch QPI reconstruction script for grid data (e.g., multipos_test_1)
acquired by generate_grid_pos.py.

Directory structure:
    GRID_DIR/
        Pos0_x+0_y+0/   <- BG (for reconstruction)
            img_000000000_ph_000.tif
            img_000000000_ph_001.tif
            ...
            img_000000000_ph_010.tif
        Pos0_x-1_y+0/   <- BG
            ...
        Pos1_x+0_y+0/   <- Reconstruction target
            img_000000000_ph_000.tif
            ...
        Pos2_x+0_y+0/   <- Reconstruction target
            ...

Correspondence:
    PosX_x{xi}_y{yi}  ->  BG uses the same z from Pos0_x{xi}_y{yi}

Output:
    GRID_DIR/PosX_x{xi}_y{yi}/output_phase/img_000000000_ph_{z:03d}.tif
"""
import argparse
import re
import sys
import os
import numpy as np
from pathlib import Path

# Set CUDA_PATH before any cupy import (cupy reads it at import time)
if not os.environ.get("CUDA_PATH"):
    _nvrtc = Path(sys.prefix) / "Lib/site-packages/nvidia/cuda_nvrtc"
    if _nvrtc.exists():
        os.environ["CUDA_PATH"] = str(_nvrtc)

import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from tqdm import tqdm
import time
import queue as queue_mod
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ============================================================
# Configuration parameters
# ============================================================
# Must match GRID_DIR in pipeline_full.py
GRID_DIR = r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1"

# Base label used as BG (pipeline_full: GRID_BG_BASE_LABEL)
BG_BASE_LABEL = "Pos0"

# Target base labels for reconstruction (None to process all PosX_x*_y* except Pos0)
# Same meaning as pipeline_full: GRID_TARGET_BASE_LABELS
TARGET_BASE_LABELS = None

# Filter target (xi, yi) coordinates (None = process all)
# e.g.: TARGET_COORDS = [(0, 0)]  -> center point only
TARGET_COORDS = None

# Filter z indices (None = process all, list = only those indices)
# e.g.: Z_INDICES = [5]  -> only z=5
Z_INDICES = None

# QPI optical parameters
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

# ---- Crop switching by Pos number (same as pipeline_full.py) ----
# pos_number < POS_SPLIT -> right side (400:2448)  sensor width 2448
# pos_number >= POS_SPLIT -> left side (0:2048)
# Note: BG (Pos0) uses the crop determined by the target's pos_number (not always right)
POS_SPLIT    = 52
CROP_BEFORE  = (0, 2048, 400, 2448)
CROP_AFTER   = (0, 2048,   0, 2048)

# Region for mean-zero adjustment (same as pipeline GRID_MEAN_REGION. None to disable)
MEAN_REGION = None

# Skip if already reconstructed (existing *_phase.tif in output_phase) (pipeline: GRID_SKIP_IF_EXISTS)
SKIP_IF_EXISTS = True

# Whether to also save PNG color maps
SAVE_PNG = False
PNG_DPI  = 150
PNG_VMIN = -2.0
PNG_VMAX =  2.0
# Parallel processing worker count (CPU fallback mode)
N_WORKERS = 24
# GPU+CPU pipeline: consumer thread count for unwrap_phase
N_CPU_CONSUMERS = 24
# ============================================================

# QPI import
try:
    from qpi import QPIParameters, get_field, set_backend, _HAS_CUPY
except ImportError:
    print("ERROR: qpi module not found. Please verify that QPI_Omni/scripts is in PYTHONPATH.")
    sys.exit(1)

# GPU acceleration (get_field uses CuPy FFT, unwrap_phase is CPU-only)
_FORCE_CPU = False
_USE_GPU = False
if _HAS_CUPY and not _FORCE_CPU:
    try:
        import cupy as cp
        cp.array([1.0]) * 2  # smoke test
        set_backend("cupy")
        _USE_GPU = True
        print(f"GPU mode: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    except Exception as e:
        print(f"GPU init failed, using CPU: {e}")
        set_backend("numpy")

# CuPy works in ProcessPoolExecutor: each worker inits its own CUDA context once.
# 4 workers balances GPU utilization vs context overhead (benchmark confirmed).


def scan_grid_folders(grid_dir: Path):
    """
    Enumerate all {label}_x{xi:+d}_y{yi:+d} folders in grid_dir
    and return a dict of {base_label: {(xi, yi): folder_path}}.
    """
    pattern = re.compile(r"^(.+)_x([+-]?\d+)_y([+-]?\d+)$")
    result = {}
    for d in sorted(grid_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            base  = m.group(1)
            xi    = int(m.group(2))
            yi    = int(m.group(3))
            result.setdefault(base, {})[(xi, yi)] = d
    return result


def get_z_files(pos_dir: Path):
    """Return img_000000000_ph_XXX.tif files sorted by z index."""
    files = sorted(pos_dir.glob("img_*_ph_*.tif"))
    return files


def get_z_index(path: Path):
    """img_000000000_ph_010.tif → 10"""
    m = re.search(r"_ph_(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def get_crop(pos_number: int):
    """Return crop region based on Pos number."""
    return CROP_BEFORE if pos_number < POS_SPLIT else CROP_AFTER


def reconstruct_image(img_path: Path, qpi_params, crop):
    """Read a single raw image, crop, phase-reconstruct, and return as ndarray (float64)."""
    img = np.array(Image.open(str(img_path)))
    rs, re_, cs, ce = crop
    img = img[rs:re_, cs:ce]
    field = get_field(img, qpi_params)
    if hasattr(field, "get"):  # CuPy array → numpy
        angle = np.angle(field.get())
    else:
        angle = np.angle(field)
    phase = unwrap_phase(angle)
    return phase


def make_qpi_params(sample_img_path: Path, crop):
    """Get crop size from a single image and create QPIParameters."""
    img = np.array(Image.open(str(sample_img_path)))
    rs, re_, cs, ce = crop
    cropped = img[rs:re_, cs:ce]
    return QPIParameters(
        wavelength=WAVELENGTH,
        NA=NA,
        img_shape=cropped.shape,
        pixelsize=PIXELSIZE,
        offaxis_center=OFFAXIS_CENTER,
    )


def reconstruct_from_holo(holo_path, crop):
    """One-shot: create QPIParameters and reconstruct unwrapped phase in one call.

    Convenience wrapper around make_qpi_params + reconstruct_image.
    """
    qpi = make_qpi_params(holo_path, crop)
    return reconstruct_image(holo_path, qpi, crop)


def _gpu_pipeline(frame_dicts, qpi_params, crop, desc=""):
    """GPU+CPU pipeline: 1 GPU producer (FFT) + N CPU consumers (unwrap+save).

    frame_dicts: list of dicts with keys:
        tgt_path, raw_out_path (or None), out_path (or None),
        bg_raw_path (or None), pos_number, xi, yi, z_idx
    Returns (n_ok, n_err).
    """
    if not frame_dicts:
        return 0, 0

    q = queue_mod.Queue(maxsize=48)
    results_lock = threading.Lock()
    n_ok = [0]
    n_err = [0]

    def _producer():
        if _USE_GPU:
            set_backend("cupy")
        for fd in frame_dicts:
            try:
                img = np.array(Image.open(str(fd["tgt_path"])))
                rs, re_, cs, ce = crop
                img = img[rs:re_, cs:ce]
                field = get_field(img, qpi_params)
                if hasattr(field, "get"):
                    angle_arr = np.angle(field.get())
                else:
                    angle_arr = np.angle(field)
                q.put((fd, angle_arr))
            except Exception:
                with results_lock:
                    n_err[0] += 1
        q.put(None)

    def _consumer():
        while True:
            item = q.get()
            if item is None:
                q.put(None)
                break
            fd, angle_arr = item
            try:
                phase = unwrap_phase(angle_arr)

                raw_out = fd.get("raw_out_path")
                if raw_out and not Path(raw_out).exists():
                    Path(raw_out).parent.mkdir(exist_ok=True)
                    tifffile.imwrite(str(raw_out), phase.astype(np.float32))

                out = fd.get("out_path")
                bg_raw = fd.get("bg_raw_path")
                if out and not Path(out).exists():
                    if bg_raw and Path(bg_raw).exists():
                        phase_bg = tifffile.imread(str(bg_raw)).astype(np.float64)
                        phase_diff = phase - phase_bg
                        h, w = phase_diff.shape
                        pn = fd.get("pos_number", 0)
                        if pn < POS_SPLIT:
                            region = phase_diff[1:h - 1, 1:w // 2]
                        else:
                            region = phase_diff[1:h - 1, w // 2:w - 1]
                        if region.size > 0:
                            phase_diff -= np.mean(region)
                        Path(out).parent.mkdir(exist_ok=True)
                        tifffile.imwrite(str(out), phase_diff.astype(np.float32))
                    else:
                        pass

                with results_lock:
                    n_ok[0] += 1
            except Exception:
                with results_lock:
                    n_err[0] += 1

    n_consumers = N_CPU_CONSUMERS
    print(f"  GPU+CPU pipeline: {len(frame_dicts)} frames, 1 GPU + {n_consumers} CPU  [{desc}]",
          flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1 + n_consumers) as pool:
        pool.submit(_producer)
        futs = [pool.submit(_consumer) for _ in range(n_consumers)]
        for f in futs:
            f.result()
    elapsed = time.perf_counter() - t0
    fps = len(frame_dicts) / elapsed if elapsed > 0 else 0
    print(f"  Done: {n_ok[0]} OK, {n_err[0]} errors  ({elapsed:.1f}s, {fps:.1f} fps)", flush=True)
    return n_ok[0], n_err[0]


def _reconstruct_grid_point(args):
    """ProcessPoolExecutor worker: reconstruct and save all z slices for one grid point.

    Saves two outputs per z slice:
      - output_phase/     : BG-subtracted + region mean subtracted (for ECC)
      - output_phase_raw/ : raw phase, no BG subtraction (for grid_subtract / correct_0pergluc)
    """
    xi, yi, target_dir_str, bg_dir_str, crop, pos_number = args
    # Re-init CuPy in spawned worker (Windows spawn resets qpi.xp to numpy)
    if _HAS_CUPY and not _FORCE_CPU:
        try:
            set_backend("cupy")
        except Exception:
            pass
    target_dir  = Path(target_dir_str)
    bg_dir      = Path(bg_dir_str)
    out_dir     = target_dir / "output_phase"
    raw_out_dir = target_dir / "output_phase_raw"

    z_files_target = {get_z_index(p): p for p in get_z_files(target_dir)}
    z_files_bg     = {get_z_index(p): p for p in get_z_files(bg_dir)}

    if not z_files_target:
        return xi, yi, False, "no z images"

    out_dir.mkdir(exist_ok=True)
    raw_out_dir.mkdir(exist_ok=True)
    sample_path = next(iter(z_files_target.values()))
    try:
        qpi_params = make_qpi_params(sample_path, crop)
    except Exception as e:
        return xi, yi, False, f"QPIParams: {e}"

    folder_ok = True
    for z_idx, tgt_path in sorted(z_files_target.items()):
        if Z_INDICES is not None and z_idx not in Z_INDICES:
            continue
        out_path     = out_dir     / (tgt_path.stem + "_phase.tif")
        raw_out_path = raw_out_dir / (tgt_path.stem + "_phase.tif")
        if SKIP_IF_EXISTS and out_path.exists() and raw_out_path.exists():
            continue
        try:
            phase_target = None

            # Save raw phase (no BG subtraction)
            if not raw_out_path.exists():
                phase_target = reconstruct_image(tgt_path, qpi_params, crop)
                tifffile.imwrite(str(raw_out_path), phase_target.astype(np.float32))

            # BG subtraction for output_phase
            if not out_path.exists():
                # Get target phase (reconstruct or load from saved raw)
                if phase_target is None:
                    phase_target = tifffile.imread(str(raw_out_path)).astype(np.float64)

                # Get BG phase (pre-reconstructed raw first, reconstruct as fallback)
                phase_bg = None
                bg_raw_subdir = "output_phase_raw" if pos_number < POS_SPLIT else "output_phase_raw_crop_after"
                bg_raw_path = bg_dir / bg_raw_subdir / (tgt_path.stem + "_phase.tif")
                if bg_raw_path.exists():
                    phase_bg = tifffile.imread(str(bg_raw_path)).astype(np.float64)
                elif z_idx in z_files_bg:
                    phase_bg = reconstruct_image(z_files_bg[z_idx], qpi_params, crop)

                if phase_bg is None:
                    folder_ok = False
                    continue

                phase_diff = phase_target - phase_bg
                h, w = phase_diff.shape
                if pos_number < POS_SPLIT:
                    region = phase_diff[1:h-1, 1:w//2]
                else:
                    region = phase_diff[1:h-1, w//2:w-1]
                if region.size > 0:
                    phase_diff -= np.mean(region)
                tifffile.imwrite(str(out_path), phase_diff.astype(np.float32))
            if SAVE_PNG:
                import matplotlib.pyplot as plt
                png_path = out_dir / (tgt_path.stem + ".png")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(phase_diff, cmap="RdBu_r", vmin=PNG_VMIN, vmax=PNG_VMAX)
                ax.axis("off")
                ax.set_title(f"{target_dir.name} z={z_idx}")
                plt.tight_layout()
                plt.savefig(str(png_path), dpi=PNG_DPI, bbox_inches="tight")
                plt.close()
        except Exception as e:
            folder_ok = False
    return xi, yi, folder_ok, None


def _reconstruct_bg_raw(args):
    """Save raw phase (no BG subtraction) for BG folders with both crops.

    Downstream scripts (correct_0pergluc) need Pos0 output_phase_raw
    with CROP_BEFORE (for Pos < POS_SPLIT) and CROP_AFTER (for Pos >= POS_SPLIT).
    """
    xi, yi, bg_dir_str = args
    # Re-init CuPy in spawned worker (Windows spawn resets qpi.xp to numpy)
    if _HAS_CUPY and not _FORCE_CPU:
        try:
            set_backend("cupy")
        except Exception:
            pass
    bg_dir = Path(bg_dir_str)

    for crop_name, crop in [("before", CROP_BEFORE), ("after", CROP_AFTER)]:
        raw_out_dir = bg_dir / "output_phase_raw" if crop_name == "before" else bg_dir / "output_phase_raw_crop_after"
        raw_out_dir.mkdir(exist_ok=True)
        z_files = {get_z_index(p): p for p in get_z_files(bg_dir)}
        if not z_files:
            return xi, yi, False, "z files not found"
        sample_path = next(iter(z_files.values()))
        try:
            qpi_params = make_qpi_params(sample_path, crop)
        except Exception as e:
            return xi, yi, False, f"QPIParams ({crop_name}): {e}"
        for z_idx, path in sorted(z_files.items()):
            if Z_INDICES is not None and z_idx not in Z_INDICES:
                continue
            out_path = raw_out_dir / (path.stem + "_phase.tif")
            if out_path.exists():
                continue
            try:
                phase = reconstruct_image(path, qpi_params, crop)
                tifffile.imwrite(str(out_path), phase.astype(np.float32))
            except Exception:
                pass
    return xi, yi, True, None


def _parse_cli():
    p = argparse.ArgumentParser(
        description="Reconstruct raw holograms for each grid point with BG subtraction into output_phase.",
    )
    p.add_argument(
        "--grid-dir",
        type=str,
        default=None,
        help=f"GRID_DIR (default: {GRID_DIR})",
    )
    p.add_argument(
        "--bg-label",
        type=str,
        default=None,
        help=f"BG base label (default: {BG_BASE_LABEL})",
    )
    p.add_argument(
        "--targets",
        nargs="*",
        default=None,
        metavar="LABEL",
        help='Base labels to reconstruct (e.g., Pos6). If omitted, uses TARGET_BASE_LABELS / all Pos (except BG)',
    )
    p.add_argument(
        "--z-indices",
        nargs="+",
        type=int,
        default=None,
        metavar="Z",
        help="Only reconstruct these z indices (e.g., --z-indices 5). Default: all z slices.",
    )
    return p.parse_args()


def main():
    global Z_INDICES
    args = _parse_cli()
    grid_dir = Path(args.grid_dir or GRID_DIR)
    bg_label = args.bg_label or BG_BASE_LABEL
    if args.targets is not None:
        target_labels_override = args.targets if args.targets else None
    else:
        target_labels_override = TARGET_BASE_LABELS
    if args.z_indices is not None:
        Z_INDICES = args.z_indices
        print(f"Z-index filter: {Z_INDICES}")

    if not grid_dir.exists():
        print(f"ERROR: GRID_DIR not found: {grid_dir}")
        sys.exit(1)

    t_start = time.perf_counter()

    # Folder scan
    folders = scan_grid_folders(grid_dir)
    if bg_label not in folders:
        print(f"ERROR: BG folder '{bg_label}_x*_y*' not found: {grid_dir}")
        sys.exit(1)

    bg_map = folders[bg_label]  # {(xi, yi): Path}

    # Determine target base labels
    if target_labels_override is not None:
        target_labels = target_labels_override
    else:
        target_labels = [k for k in sorted(folders.keys()) if k != bg_label]

    print(f"BG base label: {bg_label}  ({len(bg_map)} coordinate points)")
    print(f"Target base labels: {target_labels}  (total {sum(len(folders[l]) for l in target_labels if l in folders)} folders)")

    total_ok = 0
    total_err = 0

    if _USE_GPU:
        # ========== GPU+CPU pipeline mode ==========
        # BG: reconstruct with both crops via pipeline
        print(f"\n{'='*60}")
        print(f"  BG raw phase ({bg_label}) - GPU+CPU pipeline  ({len(bg_map)} points)")
        print(f"{'='*60}")
        for crop_name, crop_val in [("before", CROP_BEFORE), ("after", CROP_AFTER)]:
            bg_frames = []
            for (xi, yi), bg_dir in sorted(bg_map.items()):
                raw_out_dir = bg_dir / ("output_phase_raw" if crop_name == "before"
                                        else "output_phase_raw_crop_after")
                raw_out_dir.mkdir(exist_ok=True)
                z_files = {get_z_index(p): p for p in get_z_files(bg_dir)}
                for z_idx, zf in sorted(z_files.items()):
                    if Z_INDICES is not None and z_idx not in Z_INDICES:
                        continue
                    raw_out = raw_out_dir / (zf.stem + "_phase.tif")
                    if raw_out.exists():
                        continue
                    bg_frames.append({
                        "tgt_path": str(zf), "raw_out_path": str(raw_out),
                        "out_path": None, "bg_raw_path": None,
                        "pos_number": 0, "xi": xi, "yi": yi, "z_idx": z_idx,
                    })
            if bg_frames:
                qpi_bg = make_qpi_params(Path(bg_frames[0]["tgt_path"]), crop_val)
                ok, err = _gpu_pipeline(bg_frames, qpi_bg, crop_val,
                                        f"BG {crop_name}")
                total_ok += ok
                total_err += err
            else:
                print(f"  BG {crop_name}: all exist, skipped")

        # Targets: per Pos, build frame list and run pipeline
        for base_label in target_labels:
            if base_label not in folders:
                print(f"  [WARN] '{base_label}_x*_y*' folders not found. Skipping.")
                continue
            target_map = folders[base_label]
            m = re.match(r"Pos(\d+)$", base_label)
            pos_number = int(m.group(1)) if m else 0
            crop = get_crop(pos_number)
            print(f"\n{'='*60}")
            print(f"  {base_label}  ({len(target_map)} folders)  crop={crop}")
            print(f"{'='*60}")

            target_frames = []
            for (xi, yi) in sorted(target_map.keys()):
                if TARGET_COORDS is not None and (xi, yi) not in TARGET_COORDS:
                    continue
                tgt_dir = target_map[(xi, yi)]
                if (xi, yi) not in bg_map:
                    print(f"  [WARN] BG not found: {bg_label}_x{xi:+d}_y{yi:+d}  -> skipping")
                    total_err += 1
                    continue
                bg_d = bg_map[(xi, yi)]
                out_dir = tgt_dir / "output_phase"
                raw_out_dir = tgt_dir / "output_phase_raw"
                out_dir.mkdir(exist_ok=True)
                raw_out_dir.mkdir(exist_ok=True)
                z_files = {get_z_index(p): p for p in get_z_files(tgt_dir)}
                bg_raw_subdir = ("output_phase_raw" if pos_number < POS_SPLIT
                                 else "output_phase_raw_crop_after")
                for z_idx, tgt_path in sorted(z_files.items()):
                    if Z_INDICES is not None and z_idx not in Z_INDICES:
                        continue
                    out_path = out_dir / (tgt_path.stem + "_phase.tif")
                    raw_out_path = raw_out_dir / (tgt_path.stem + "_phase.tif")
                    if SKIP_IF_EXISTS and out_path.exists() and raw_out_path.exists():
                        continue
                    bg_raw_path = bg_d / bg_raw_subdir / (tgt_path.stem + "_phase.tif")
                    target_frames.append({
                        "tgt_path": str(tgt_path),
                        "raw_out_path": str(raw_out_path),
                        "out_path": str(out_path),
                        "bg_raw_path": str(bg_raw_path) if bg_raw_path.exists() else None,
                        "pos_number": pos_number,
                        "xi": xi, "yi": yi, "z_idx": z_idx,
                    })

            if target_frames:
                qpi = make_qpi_params(Path(target_frames[0]["tgt_path"]), crop)
                ok, err = _gpu_pipeline(target_frames, qpi, crop, base_label)
                total_ok += ok
                total_err += err
            else:
                print(f"  {base_label}: all exist, skipped")

    else:
        # ========== CPU ProcessPoolExecutor fallback ==========
        print(f"\n{'='*60}")
        print(f"  BG raw phase ({bg_label}) - CPU mode  ({len(bg_map)} points)")
        print(f"{'='*60}")
        bg_tasks = [(xi, yi, str(d)) for (xi, yi), d in sorted(bg_map.items())]
        if N_WORKERS == 1:
            bg_results = [_reconstruct_bg_raw(t) for t in tqdm(bg_tasks, desc=f"{bg_label} BG")]
        else:
            bg_results = []
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = {executor.submit(_reconstruct_bg_raw, t): t for t in bg_tasks}
                for fut in tqdm(as_completed(futures), total=len(bg_tasks), desc=f"{bg_label} BG"):
                    bg_results.append(fut.result())
        bg_ok = sum(1 for _, _, ok, _ in bg_results if ok)
        bg_err = sum(1 for _, _, ok, _ in bg_results if not ok)
        print(f"  BG raw: {bg_ok} OK, {bg_err} errors")

        for base_label in target_labels:
            if base_label not in folders:
                print(f"  [WARN] '{base_label}_x*_y*' folders not found. Skipping.")
                continue
            target_map = folders[base_label]
            m = re.match(r"Pos(\d+)$", base_label)
            pos_number = int(m.group(1)) if m else 0
            crop = get_crop(pos_number)
            print(f"\n{'='*60}")
            print(f"  {base_label}  ({len(target_map)} folders)  crop={crop}")
            print(f"{'='*60}")
            tasks = []
            for (xi, yi) in sorted(target_map.keys()):
                if TARGET_COORDS is not None and (xi, yi) not in TARGET_COORDS:
                    continue
                tgt_dir = target_map[(xi, yi)]
                if (xi, yi) not in bg_map:
                    print(f"  [WARN] BG not found: {bg_label}_x{xi:+d}_y{yi:+d}  -> skipping")
                    total_err += 1
                    continue
                bg_d = bg_map[(xi, yi)]
                tasks.append((xi, yi, str(tgt_dir), str(bg_d), crop, pos_number))
            if not tasks:
                continue
            print(f"  Parallel: {len(tasks)} points / {N_WORKERS} workers (CPU)")
            if N_WORKERS == 1:
                results = [_reconstruct_grid_point(args) for args in tqdm(tasks, desc=base_label)]
            else:
                results = []
                with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                    futures = {executor.submit(_reconstruct_grid_point, args): args for args in tasks}
                    for fut in tqdm(as_completed(futures), total=len(tasks), desc=base_label):
                        results.append(fut.result())
            for xi, yi, folder_ok, err_msg in results:
                if folder_ok:
                    total_ok += 1
                else:
                    total_err += 1
                    if err_msg:
                        print(f"  [ERR] ({xi:+d},{yi:+d}): {err_msg}")

    elapsed = time.perf_counter() - t_start
    mode = "GPU pipeline" if _USE_GPU else f"CPU ({N_WORKERS} workers)"
    print(f"\n{'='*60}")
    print(f"Done  ({mode}, {elapsed:.1f}s)")
    print(f"  OK:    {total_ok}")
    print(f"  Error: {total_err}")
    print(f"  (SKIP_IF_EXISTS={SKIP_IF_EXISTS})")


if __name__ == "__main__":
    main()

# %%
