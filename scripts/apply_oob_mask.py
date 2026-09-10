# %%
"""
apply_oob_mask.py
-----------------
Apply grid_subtract's valid_out OOB mask to existing crop_sub output IN PLACE,
without re-running grid_subtract.

Why this exists
---------------
The OOB-zeroing step (valid_out > 0.999 in grid_subtract.process_single_frame)
was added AFTER this dataset's crop_sub was produced online. Re-running
grid_subtract would (a) be slow and (b) overwrite the correct_0pergluc result
with fresh, uncorrected frames. So instead we recompute the exact same
geometric mask and multiply it onto the existing frames.

This must run as the FINAL step, AFTER correct_0pergluc, because
correct_0pergluc does ``img -= delta`` which would re-introduce non-zero values
into an OOB band that was zeroed too early.

The mask is geometry-only: it depends on the per-frame residual subpixel warp
and the per-channel crop position. We reuse grid_subtract.apply_inverse_shift_warp
and extract_rect_roi, and the per-frame residual_x_px / residual_y_px and
grid_xi / grid_yi recorded in pos_shifts_cal_online.json, so the mask is
byte-identical to what grid_subtract would have applied.

Idempotent: re-running multiplies by the same 0/1 mask again (no-op).
"""
import numpy as np
import tifffile
import json
import re
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import binary_erosion
import argparse
import subprocess
import time

sys.path.insert(0, str(Path(__file__).parent))
from grid_subtract import (
    apply_inverse_shift_warp,
    extract_rect_roi,
    load_grid_calibration,
)

# ============================================================
# Configuration
# ============================================================
# crop_sub root (contains PosN/output_phase/channels/<CH_SUBDIR>/chNN/)
CROP_SUB_ROOT = r"E:\260517\2per_0055per_0per_2per_crop_sub"

# Positions to process.
POS_NUMBERS = list(range(1, 105))  # all Pos (offline regen)

# Subfolder under PosN/output_phase/channels/ that contains the chNN/ dirs.
CH_SUBDIR = r"crop_sub_rawraw\z000"

# Per-Pos source for the per-frame grid cell + residual that grid_subtract used.
# grid_subtract_log.json frame_log holds grid_xi/grid_yi/residual_x_px/residual_y_px
# matching the regenerated crop_sub (NOT the stale online JSON).
SHIFTS_FILENAME = "grid_subtract_log.json"

# channel_rois.json source: per-Pos from the grid (same convention as grid_subtract).
GRID_DIR = r"E:\260517\grid_2pergluc_2"

# Grid calibration JSON template ({label} -> PosN).
GRID_CAL_TEMPLATE = r"E:\260517\grid_2pergluc_2\grid_calibration_{label}.json"

# Reconstructed frame size (square) used when building the validity mask.
FULL_DIM = 511

# Wide tilt crop height (axis=1), must match grid_subtract TILT_CROP_H_RAW.
TILT_CROP_H = 270

# valid_out threshold (must match grid_subtract: any boundary-contaminated pixel out).
VALID_THRESH = 0.999

# Erode valid mask by this many px at the OOB boundary (must match grid_subtract
# VALID_ERODE_PX). border_value=1 erodes only pixels adjacent to interior OOB.
VALID_ERODE_PX = 1

# Must match the online run's apply_subpixel_correction flag.
APPLY_SUBPIXEL = True

# Threads for the per-frame loop (cv2 / tifffile release the GIL).
N_PARALLEL_FRAMES = 6

# Pos-level parallelism (in-place I/O-bound): run K Pos concurrently as subprocesses.
K_CONCURRENT = 6
LOG_DIR = Path(__file__).resolve().parent.parent / "drift_session" / "oob_mask_logs"

# Only the 0% glucose frames need re-masking: grid_subtract already OOB-masked
# ALL frames; correct_0pergluc (img-=delta) un-zeroed only the 0% period. Set to
# None,None to process all frames.
FRAME_START = 2019
FRAME_END   = 2885   # exclusive
# ============================================================


def _parse_frame_index(name: str) -> int:
    """img_000000123_ph_000_phase.tif -> 123."""
    m = re.match(r"img_(\d+)_", name)
    if m is None:
        raise ValueError(f"Cannot parse frame index from filename: {name}")
    return int(m.group(1))


def _build_valid_full(residual_x, residual_y):
    """Validity mask of the full reconstructed frame after the residual warp.

    Mirrors grid_subtract.process_single_frame: warp an all-ones frame with the
    same residual; boundary pixels become < 1 where they mix with the warp's
    zero borderValue.
    """
    ones = np.ones((FULL_DIM, FULL_DIM), dtype=np.float32)
    if APPLY_SUBPIXEL and (residual_x != 0.0 or residual_y != 0.0):
        return apply_inverse_shift_warp(ones, residual_x, residual_y).astype(np.float32)
    return ones


def process_pos(pos_num: int):
    label = f"Pos{pos_num}"
    channels_dir = Path(CROP_SUB_ROOT) / label / "output_phase" / "channels"
    ch_parent = channels_dir / CH_SUBDIR
    shifts_path = channels_dir / SHIFTS_FILENAME
    rois_path = (Path(GRID_DIR) / f"{label}_x+0_y+0"
                 / "output_phase" / "channels" / "channel_rois.json")
    cal_path = Path(GRID_CAL_TEMPLATE.format(label=label))

    for p, nm in [(ch_parent, "CH_SUBDIR"), (shifts_path, "shifts"),
                  (rois_path, "channel_rois"), (cal_path, "grid_calibration")]:
        if not p.exists():
            raise FileNotFoundError(f"{label}: {nm} not found: {p}")

    # Idempotent skip: re-running multiplies by the same 0/1 mask, but skip done
    # Pos for speed/resumability.
    marker = ch_parent / ".oob_masked_done"
    if marker.exists():
        print(f"{label}: [SKIP] already OOB-masked")
        return {"label": label, "n_frames": 0, "n_channels": 0,
                "zeroed_pixels": 0, "skipped": True}

    rois = json.loads(rois_path.read_text(encoding="utf-8"))
    n_channels = len(rois)
    grid_cal = load_grid_calibration(str(cal_path))

    shifts = json.loads(shifts_path.read_text(encoding="utf-8"))
    frame_results = (shifts.get("frame_log") or shifts.get("frame_results")
                     or shifts.get("alignment_results"))
    if not frame_results:
        raise RuntimeError(f"{label}: no frame_log/frame_results in {shifts_path}")
    by_index = {}
    for e in frame_results:
        if e is None:
            continue
        by_index[int(e["frame_index"])] = e

    ch_dirs = [ch_parent / f"ch{ch:02d}" for ch in range(n_channels)]
    for d in ch_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"{label}: channel dir missing: {d}")

    ch0_files = sorted(ch_dirs[0].glob("*.tif"))
    if not ch0_files:
        raise RuntimeError(f"{label}: no tif files in {ch_dirs[0]}")
    print(f"{label}: {len(ch0_files)} frames x {n_channels} channels")

    # Per-channel crop centre offsets are constant across frames for a given
    # (grid_xi, grid_yi); resolve cal_dx/cal_dy per frame from the stored cell.
    def _cal_for(entry):
        key = (int(entry["grid_xi"]), int(entry["grid_yi"]))
        if key not in grid_cal:
            raise KeyError(f"{label}: grid cell {key} not in grid calibration")
        return grid_cal[key]

    n_masked_pixels = [0]  # mutable counter shared across threads (GIL-safe int add)

    def _process_frame(name: str):
        idx = _parse_frame_index(name)
        entry = by_index.get(idx)
        if entry is None:
            raise KeyError(f"{label}: frame_index {idx} (file {name}) not in shifts JSON")
        residual_x = float(entry.get("residual_x_px", 0.0))
        residual_y = float(entry.get("residual_y_px", 0.0))
        cal_dx, cal_dy = _cal_for(entry)

        valid_full = _build_valid_full(residual_x, residual_y)

        local_masked = 0
        for ch in range(n_channels):
            roi = rois[ch]
            cx, cy = roi["cx"], roi["cy"]
            crop_w = roi["crop_w"]
            crop_cx = int(round(cx + cal_dx))
            crop_cy = int(round(cy + cal_dy))

            fpath = ch_dirs[ch] / name
            img = tifffile.imread(str(fpath)).astype(np.float32)
            out_crop_h = img.shape[1]

            valid_large = extract_rect_roi(valid_full, crop_cy, crop_cx, crop_w, TILT_CROP_H)
            vstart = (TILT_CROP_H - out_crop_h) // 2
            valid_out = valid_large[:, vstart:vstart + out_crop_h] > VALID_THRESH
            if VALID_ERODE_PX > 0:
                valid_out = binary_erosion(valid_out, iterations=VALID_ERODE_PX,
                                           border_value=1)
            if valid_out.shape != img.shape:
                raise ValueError(
                    f"{label} {name} ch{ch:02d}: mask shape {valid_out.shape} "
                    f"!= img shape {img.shape}")

            local_masked += int(valid_out.size - int(valid_out.sum()))
            tifffile.imwrite(str(fpath), (img * valid_out).astype(np.float32))
        n_masked_pixels[0] += local_masked

    names = [f.name for f in ch0_files]
    if FRAME_START is not None and FRAME_END is not None:
        names = [nm for nm in names
                 if FRAME_START <= _parse_frame_index(nm) < FRAME_END]
        print(f"{label}: re-masking {len(names)} frames in [{FRAME_START},{FRAME_END})")
    with ThreadPoolExecutor(max_workers=N_PARALLEL_FRAMES) as ex:
        futures = [ex.submit(_process_frame, nm) for nm in names]
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc=f"{label} OOB mask"):
            f.result()

    marker.write_text("done")
    print(f"{label}: done. zeroed pixels (all frames x ch): {n_masked_pixels[0]}")
    return {"label": label, "n_frames": len(names), "n_channels": n_channels,
            "zeroed_pixels": n_masked_pixels[0]}


def _needs_run(pos_num):
    marker = (Path(CROP_SUB_ROOT) / f"Pos{pos_num}" / "output_phase" / "channels"
              / CH_SUBDIR / ".oob_masked_done")
    ch_parent = (Path(CROP_SUB_ROOT) / f"Pos{pos_num}" / "output_phase"
                 / "channels" / CH_SUBDIR)
    return ch_parent.is_dir() and not marker.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=None)
    args = ap.parse_args()

    if args.pos is not None:
        try:
            r = process_pos(args.pos)
            print(f"[single] Pos{args.pos} -> {r.get('zeroed_pixels')} zeroed", flush=True)
            sys.exit(0)
        except Exception:
            import traceback; traceback.print_exc()
            sys.exit(1)

    # Orchestrator: K Pos concurrent subprocesses, marker-skip, 3-pass retry
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def run_batch(todo):
        running = {}; idx = 0; ok = fail = 0
        while idx < len(todo) or running:
            while idx < len(todo) and len(running) < K_CONCURRENT:
                n = todo[idx]; idx += 1
                lf = open(LOG_DIR / f"Pos{n}.log", "w")
                p = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--pos", str(n)],
                    stdout=lf, stderr=subprocess.STDOUT)
                running[p] = (n, time.time(), lf)
            time.sleep(2)
            for p in list(running):
                rc = p.poll()
                if rc is None:
                    continue
                n, st, lf = running.pop(p); lf.close()
                ok, fail = (ok + 1, fail) if rc == 0 else (ok, fail + 1)
                print(f"  Pos{n} rc={rc} [ok{ok} fail{fail} run{len(running)}]", flush=True)

    for pass_i in range(1, 4):
        todo = [n for n in POS_NUMBERS if _needs_run(n)]
        if not todo:
            print("All Pos OOB-masked.", flush=True); break
        print(f"\n=== PASS {pass_i}: {len(todo)} Pos (K={K_CONCURRENT}) ===\n{todo}", flush=True)
        run_batch(todo)

    remaining = [n for n in POS_NUMBERS if _needs_run(n)]
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min. "
          f"incomplete={remaining if remaining else 'NONE'}", flush=True)


if __name__ == "__main__":
    main()

# %%
