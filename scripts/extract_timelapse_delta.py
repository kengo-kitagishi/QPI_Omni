# %%
"""
extract_timelapse_delta.py
--------------------------
Extract per-z delta TIFs from the timelapse frame closest to grid(0,0),
for ALL sample positions automatically.

After a z-stack timelapse, this script:
1. Auto-detects PosN directories under TIMELAPSE_ROOT (Pos0=BG skipped).
2. For each Pos, reads pos_shifts JSON to find the frame with minimum
   shift magnitude (closest to grid_2per(0,0) position).
3. For each (timelapse_z, grid_z) pair, loads raw phase for both
   timelapse and grid_2per, warps the timelapse frame to align with
   grid(0,0), and subtracts.
4. Saves delta_z{grid_z:03d}.tif (511x511, float32) per z per Pos.

These delta TIFs replace the need for a separate grid_0per scan.
correct_0pergluc.py consumes them via its DELTA_TIFS_DIR parameter.

Handles timelapse data in two layouts:
  - z-stack:   PosN/z{z:03d}/output_phase_raw/  (from _prerecon_save_all_z)
  - single-z:  PosN/output_phase_raw/           (from batch_pipeline_all_pos)

If output_phase_raw is not available, falls back to on-the-fly
reconstruction from raw holograms (only 1 frame per z -- fast).

Usage:
  1. Configure TIMELAPSE_ROOT, GRID_2PER_DIR, Z_PAIRS, etc. below.
  2. Run:  python scripts/extract_timelapse_delta.py
  3. Output: PosN/output_phase/channels/delta_timelapse/ per Pos.
     Set DELTA_TIFS_DIR in correct_0pergluc.py to that path.
"""
import numpy as np
import re
import tifffile
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grid_subtract import apply_inverse_shift_warp
from batch_reconstruction_grid import (
    reconstruct_from_holo,
    CROP_BEFORE,
    CROP_AFTER,
    POS_SPLIT,
)
from ecc_utils import tilt_fit_crop, ecc_align, remove_outliers_mad, ECC_MIN_CORR
from ecc_utils import to_ecc_input as to_uint8  # float ECC input (no 8-bit quantisation)

# ============================================================
# Configuration
# ============================================================

TIMELAPSE_ROOT = r"E:\260617\2per_corr_zstack_1"

GRID_2PER_DIR = r"E:\260617\0per_grid_0p05um_1"

# When raw phase lives under a different root than shifts/output,
# set SHIFTS_OUTPUT_ROOT to the directory containing PosN with
# output_phase/channels/ (shifts JSON, channel_rois, output).
# None -> same as TIMELAPSE_ROOT.
SHIFTS_OUTPUT_ROOT = r"E:\260617\2per_corr_zstack_1_crop_sub"

# Output root for delta TIFs (PosN/output_phase/channels/<OUTPUT_SUBDIR>/).
# None -> same tree as SHIFTS_OUTPUT_ROOT (or TIMELAPSE_ROOT).
# Set to redirect delta output to a different tree, e.g. the main timelapse
# crop_sub so correct_0pergluc reads it directly.
# For now -> None puts delta in the current crop_sub tree (SHIFTS_OUTPUT_ROOT);
# copy to the production timelapse folder later.
OUTPUT_ROOT = None

# Pos range [inclusive]. Pos0 is always BG, skipped.
# POS_END = None -> auto-detect last Pos.
POS_START = 1
POS_END = 104

# Per-Pos shifts JSON filename (under PosN/output_phase/channels/)
SHIFTS_FILENAME = "pos_shifts_cal_online.json"

# Z_PAIRS: list of (timelapse_z_index, grid_z_index).
# Physical z (um) must match between timelapse and grid z-stacks.
#
# 260416+ grid: 11 slices, -2.0 to +2.0 um, 0.4 um/step, z=5 = 0.0 um
#
# 260508: 3-slice timelapse, tl_z=0 = focus = grid_z=5
# 260517: 11-slice 0per_gluc z-stack; main timelapse plane = +1.2 um = grid_z=8
# 260617: tl and grid are both 11-slice, -2.0..+2.0 um -> tl z = grid z (1:1, all z)
Z_PAIRS = [(z, z) for z in range(11)]

# Minimum frame index to consider (skip early unstable frames).
MIN_FRAME_INDEX = 5

# Frame index range to search [inclusive, exclusive). None = all frames.
FRAME_RANGE = None     # all frames; tune once drift behavior is known post-run

# Output subfolder name (under PosN/output_phase/channels/)
OUTPUT_SUBDIR = "delta_timelapse"

# Recompute ECC shift instead of using pos_shifts_cal_online.json values.
# Runs single-pass ECC (grid(0,0) reference) per channel, filters by
# ECC_MIN_CORR, and averages the remaining shifts.
RECOMPUTE_ECC = True
# ECC_MIN_CORR imported from ecc_utils (single source = 0.99).
ECC_VMIN = -5.0
ECC_VMAX = 2.0
TILT_CROP_H = 270
ECC_CROP_H = 80

# ============================================================


def _recompute_ecc_shift(tl_output_phase, grid_output_phase, rois, fit_right):
    """Single-pass ECC between output_phase images, filtered by ECC_MIN_CORR.

    Returns (avg_sx, avg_sy, n_used, n_total) or None if all channels fail.
    """
    tx_list, ty_list, corr_list = [], [], []
    for roi in rois:
        try:
            ref_crop = tilt_fit_crop(
                grid_output_phase, roi["cy"], roi["cx"], roi["crop_w"],
                ECC_CROP_H, TILT_CROP_H, fit_right=fit_right)
            cur_crop = tilt_fit_crop(
                tl_output_phase, roi["cy"], roi["cx"], roi["crop_w"],
                ECC_CROP_H, TILT_CROP_H, fit_right=fit_right)
        except Exception:
            continue
        if ref_crop is None or cur_crop is None:
            continue
        ref_u8 = to_uint8(ref_crop, ECC_VMIN, ECC_VMAX)
        cur_u8 = to_uint8(cur_crop, ECC_VMIN, ECC_VMAX)
        result = ecc_align(ref_u8, cur_u8)
        if result is None:
            continue
        tx, ty, corr = result
        if corr >= ECC_MIN_CORR:
            tx_list.append(tx)
            ty_list.append(ty)
            corr_list.append(corr)
    if not tx_list:
        return None
    n_raw = len(tx_list)
    tx_arr = np.array(tx_list)
    ty_arr = np.array(ty_list)
    if n_raw >= 3:
        is_out = remove_outliers_mad(tx_list, 5.0) | remove_outliers_mad(ty_list, 5.0)
        used = [i for i, o in enumerate(is_out) if not o]
        if not used:
            used = list(range(n_raw))
    else:
        used = list(range(n_raw))
    return (float(np.mean(tx_arr[used])), float(np.mean(ty_arr[used])),
            len(used), len(rois))


def _load_output_phase(directory, z_index, frame_index=None):
    """Load output_phase image (float64). Tries z-subdir then flat layout."""
    base = Path(directory)
    candidates = []
    if frame_index is not None:
        pat = f"img_{frame_index:09d}_ph_{z_index:03d}_phase.tif"
        candidates.append(base / f"z{z_index:03d}" / "output_phase" / pat)
        candidates.append(base / "output_phase" / pat)
    pat_glob = f"img_*_ph_{z_index:03d}_phase.tif"
    for d in [base / f"z{z_index:03d}" / "output_phase",
              base / "output_phase"]:
        if d.is_dir():
            if frame_index is not None:
                exact = d / f"img_{frame_index:09d}_ph_{z_index:03d}_phase.tif"
                if exact.exists():
                    return tifffile.imread(str(exact)).astype(np.float64), exact
            files = sorted(d.glob(pat_glob))
            if frame_index is not None and frame_index < len(files):
                return tifffile.imread(str(files[frame_index])).astype(np.float64), files[frame_index]
            if files:
                return tifffile.imread(str(files[0])).astype(np.float64), files[0]
    return None, None


def _determine_crop(base_label):
    m = re.match(r"Pos(\d+)", base_label)
    if m and int(m.group(1)) >= POS_SPLIT:
        return CROP_AFTER
    return CROP_BEFORE


def _load_shifts(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = data.get("frame_results") or data.get("alignment_results")
    if not results:
        raise ValueError(f"No frame_results in {path}")
    return [fr for fr in results if fr is not None]


def _get_shift(fr):
    sx = fr.get("shift_x_avg", fr.get("shift_x_avg_px", 0.0))
    sy = fr.get("shift_y_avg", fr.get("shift_y_avg_px", 0.0))
    return float(sx), float(sy)


def _count_available_frames(tl_dir, z_index):
    """Count available raw phase or hologram frames at z_index."""
    tl = Path(tl_dir)
    pat = f"img_*_ph_{z_index:03d}_phase.tif"
    holo_pat = f"img_*_ph_{z_index:03d}.tif"
    for d in [
        tl / "output_phase_raw",
        tl / f"z{z_index:03d}" / "output_phase_raw",
    ]:
        if d.is_dir():
            n = len(list(d.glob(pat)))
            if n > 0:
                return n
    for d in [tl / f"z{z_index:03d}", tl]:
        if d.is_dir():
            n = len(list(d.glob(holo_pat)))
            if n > 0:
                return n
    return None


def _load_timelapse_raw(tl_dir, frame_index, z_index, crop):
    """Load timelapse raw phase at (frame_index, z_index).

    Uses exact filename match (img_{frame_index:09d}_ph_{z_index:03d}_phase.tif)
    to avoid mismatch when file numbering has gaps.

    Search order:
      1. PosN/output_phase_raw/ (flat, single-z pipeline)
      2. PosN/z{z:03d}/output_phase_raw/ (z-stack online prerecon)
      3. PosN/z{z:03d}/ raw hologram (on-the-fly reconstruction)
      4. PosN/ raw hologram flat (on-the-fly reconstruction)
    """
    tl = Path(tl_dir)
    exact_phase = f"img_{frame_index:09d}_ph_{z_index:03d}_phase.tif"
    holo_pat = f"img_*_ph_{z_index:03d}.tif"
    exact_holo = f"img_{frame_index:09d}_ph_{z_index:03d}.tif"

    # 1. Flat output_phase_raw
    flat = tl / "output_phase_raw"
    if flat.is_dir():
        exact_path = flat / exact_phase
        if exact_path.exists():
            img = tifffile.imread(str(exact_path)).astype(np.float64)
            return img, exact_path, "prerecon"

    # 2. Z-subdir output_phase_raw (compute_drift_online z-stack)
    z_raw = tl / f"z{z_index:03d}" / "output_phase_raw"
    if z_raw.is_dir():
        exact_path = z_raw / exact_phase
        if exact_path.exists():
            img = tifffile.imread(str(exact_path)).astype(np.float64)
            return img, exact_path, "prerecon-zdir"

    # 3. Z-subdir raw hologram -> on-the-fly
    z_dir = tl / f"z{z_index:03d}"
    if z_dir.is_dir():
        exact_holo_path = z_dir / exact_holo
        if exact_holo_path.exists():
            print(f"  [reconstruct on-the-fly] {exact_holo_path}")
            phase = reconstruct_from_holo(exact_holo_path, crop)
            return phase.astype(np.float64), exact_holo_path, "on-the-fly"

    # 4. Flat raw hologram -> on-the-fly
    exact_holo_path = tl / exact_holo
    if exact_holo_path.exists():
        print(f"  [reconstruct on-the-fly] {exact_holo_path}")
        phase = reconstruct_from_holo(exact_holo_path, crop)
        return phase.astype(np.float64), exact_holo_path, "on-the-fly"

    raise FileNotFoundError(
        f"No raw phase or hologram found for frame_index={frame_index}, "
        f"z={z_index} under {tl_dir}\n"
        f"Checked: output_phase_raw/ (flat), z{z_index:03d}/output_phase_raw/, "
        f"z{z_index:03d}/ (holo), flat (holo)"
    )


def _load_grid_raw(grid_dir, label, z_index, crop):
    """Load grid(0,0) raw phase at z_index. Falls back to on-the-fly."""
    pos_dir = Path(grid_dir) / f"{label}_x+0_y+0"
    prerecon = (pos_dir / "output_phase_raw"
                / f"img_000000000_ph_{z_index:03d}_phase.tif")
    if prerecon.exists():
        return tifffile.imread(str(prerecon)).astype(np.float64), "prerecon"

    holo = pos_dir / f"img_000000000_ph_{z_index:03d}.tif"
    if not holo.exists():
        raise FileNotFoundError(
            f"Grid data not found at z={z_index}:\n"
            f"  prerecon: {prerecon}\n"
            f"  hologram: {holo}\n"
            f"Run: python batch_reconstruction_grid.py "
            f"--grid-dir \"{grid_dir}\" --z-indices {z_index}"
        )
    print(f"  [reconstruct on-the-fly] grid {holo.name}")
    phase = reconstruct_from_holo(holo, crop)
    return phase.astype(np.float64), "on-the-fly"


def _detect_pos_dirs(root):
    """Return sorted list of (pos_num, pos_dir) for sample positions."""
    root = Path(root)
    result = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"Pos(\d+)$", d.name)
        if not m:
            continue
        pos_num = int(m.group(1))
        if pos_num < POS_START:
            continue
        if POS_END is not None and pos_num > POS_END:
            continue
        result.append((pos_num, d))
    return result


def _shifts_output_dir(pos_label):
    """Return the directory for shifts JSON and output for a given Pos."""
    if SHIFTS_OUTPUT_ROOT:
        return Path(SHIFTS_OUTPUT_ROOT) / pos_label
    return None


def _process_single_pos(pos_label, pos_dir, grid_2per_dir):
    """Process one Pos: find closest frame, compute deltas, save TIFs."""
    so_dir = _shifts_output_dir(pos_label)
    shifts_base = so_dir if so_dir else pos_dir
    shifts_path = shifts_base / "output_phase" / "channels" / SHIFTS_FILENAME
    if not shifts_path.exists():
        print(f"  SKIP: {SHIFTS_FILENAME} not found at {shifts_path}")
        return None

    frame_results = _load_shifts(shifts_path)

    ref_z = Z_PAIRS[0][0]
    n_available = _count_available_frames(str(pos_dir), ref_z)
    if n_available is not None:
        print(f"  Available raw frames at z={ref_z}: {n_available}")

    if FRAME_RANGE:
        start, end = FRAME_RANGE
        candidates = [fr for fr in frame_results if start <= fr["frame_index"] < end]
        print(f"  Frame range: [{start}, {end}), {len(candidates)} candidates")
    else:
        candidates = [fr for fr in frame_results if fr["frame_index"] >= MIN_FRAME_INDEX]
        if n_available is not None:
            candidates = [fr for fr in candidates if fr["frame_index"] < n_available]
        print(f"  Candidates (>= {MIN_FRAME_INDEX}, < {n_available}): {len(candidates)}")

    if not candidates:
        print(f"  SKIP: no candidate frames")
        return None

    closest = min(candidates, key=lambda fr: sum(x**2 for x in _get_shift(fr)))
    closest_idx = closest["frame_index"]
    sx, sy = _get_shift(closest)
    mag = (sx**2 + sy**2) ** 0.5
    print(f"  Closest frame: index={closest_idx}, "
          f"shift=({sx:.3f}, {sy:.3f}) px, magnitude={mag:.3f} px")

    ecc_recomputed = False
    if RECOMPUTE_ECC:
        rois_path = shifts_base / "output_phase" / "channels" / "channel_rois.json"
        if not rois_path.exists():
            rois_path = (Path(grid_2per_dir) / f"{pos_label}_x+0_y+0"
                         / "output_phase" / "channels" / "channel_rois.json")
        if rois_path.exists():
            rois = json.loads(rois_path.read_text(encoding="utf-8"))
            grid_z = Z_PAIRS[0][1]
            tl_op, _ = _load_output_phase(str(pos_dir), grid_z, closest_idx)
            grid_00_dir = Path(grid_2per_dir) / f"{pos_label}_x+0_y+0"
            grid_op, _ = _load_output_phase(str(grid_00_dir), grid_z)
            if tl_op is not None and grid_op is not None:
                m = re.match(r"Pos(\d+)", pos_label)
                fit_right = int(m.group(1)) >= POS_SPLIT if m else False
                ecc_result = _recompute_ecc_shift(tl_op, grid_op, rois, fit_right)
                if ecc_result is not None:
                    sx_old, sy_old = sx, sy
                    sx, sy, n_used, n_total = ecc_result
                    mag = (sx**2 + sy**2) ** 0.5
                    print(f"  ECC recomputed (corr>={ECC_MIN_CORR}): "
                          f"shift=({sx:.3f}, {sy:.3f}) px, mag={mag:.3f} px, "
                          f"{n_used}/{n_total} ch  "
                          f"[was ({sx_old:.3f}, {sy_old:.3f})]")
                    ecc_recomputed = True
                else:
                    print(f"  ECC recompute failed (no ch passed corr>={ECC_MIN_CORR}), "
                          f"using JSON shift")
            else:
                print(f"  ECC recompute: output_phase not found, using JSON shift")
        else:
            print(f"  ECC recompute: channel_rois.json not found, using JSON shift")

    if OUTPUT_ROOT:
        out_base = Path(OUTPUT_ROOT) / pos_label
    else:
        out_base = shifts_base if so_dir else pos_dir
    out_dir = out_base / "output_phase" / "channels" / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    crop = _determine_crop(pos_label)
    grid_crop = _determine_crop(pos_label)
    print(f"  Crop: timelapse={crop}, grid={grid_crop}")

    deltas = {}
    delta_info = []
    for tl_z, grid_z in Z_PAIRS:
        print(f"  --- z: timelapse={tl_z}, grid={grid_z} ---")

        tl_img, tl_path, tl_src = _load_timelapse_raw(
            str(pos_dir), closest_idx, tl_z, crop,
        )
        print(f"    Timelapse: {tl_path.name}  shape={tl_img.shape}  src={tl_src}")

        grid_img, grid_src = _load_grid_raw(
            grid_2per_dir, pos_label, grid_z, grid_crop,
        )
        print(f"    Grid 2per(0,0): shape={grid_img.shape}  src={grid_src}")

        if tl_img.shape != grid_img.shape:
            raise ValueError(
                f"Shape mismatch at {pos_label}: "
                f"timelapse={tl_img.shape} grid={grid_img.shape}"
            )

        aligned = apply_inverse_shift_warp(tl_img, sx, sy)
        delta = aligned - grid_img

        out_path = out_dir / f"delta_z{grid_z:03d}.tif"
        tifffile.imwrite(str(out_path), delta.astype(np.float32))
        print(f"    Saved: {out_path.name}  "
              f"mean={delta.mean():.6f}  std={delta.std():.6f} rad")

        deltas[grid_z] = delta
        delta_info.append({
            "tl_z_index": tl_z,
            "grid_z_index": grid_z,
            "delta_file": out_path.name,
            "tl_source": tl_src,
            "grid_source": grid_src,
            "delta_mean_rad": float(delta.mean()),
            "delta_std_rad": float(delta.std()),
        })

    log = {
        "pos_label": pos_label,
        "timelapse_dir": str(pos_dir),
        "shifts_json": str(shifts_path),
        "grid_2per_dir": str(grid_2per_dir),
        "closest_frame_index": int(closest_idx),
        "closest_shift_x": sx,
        "closest_shift_y": sy,
        "closest_shift_magnitude_px": mag,
        "ecc_recomputed": ecc_recomputed,
        "ecc_min_corr": ECC_MIN_CORR if RECOMPUTE_ECC else None,
        "frame_range": list(FRAME_RANGE) if FRAME_RANGE else None,
        "z_pairs": Z_PAIRS,
        "deltas": delta_info,
    }
    log_path = out_dir / "extract_timelapse_delta_log.json"
    log_path.write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Log: {log_path}")

    return {
        "pos_label": pos_label,
        "out_dir": str(out_dir),
        "closest_idx": closest_idx,
        "shift_x": sx,
        "shift_y": sy,
        "shift_mag": mag,
        "n_z": len(Z_PAIRS),
        "deltas": deltas,
        "delta_info": delta_info,
    }


def main():
    root = Path(TIMELAPSE_ROOT)
    if not root.is_dir():
        raise FileNotFoundError(f"TIMELAPSE_ROOT not found: {root}")

    pos_list = _detect_pos_dirs(root)
    if not pos_list:
        raise RuntimeError(
            f"No Pos directories found in {root} "
            f"(range: Pos{POS_START}..Pos{POS_END or '?'})"
        )

    print(f"TIMELAPSE_ROOT:     {root}")
    print(f"SHIFTS_OUTPUT_ROOT: {SHIFTS_OUTPUT_ROOT or '(same as TIMELAPSE_ROOT)'}")
    print(f"GRID_2PER_DIR:      {GRID_2PER_DIR}")
    print(f"Positions:          {[p[1].name for p in pos_list]}")
    print(f"Z_PAIRS:            {len(Z_PAIRS)} pairs")
    print(f"MIN_FRAME_INDEX:    {MIN_FRAME_INDEX}")
    print(f"FRAME_RANGE:        {FRAME_RANGE}")
    print()

    all_results = []
    for pos_num, pos_dir in pos_list:
        pos_label = pos_dir.name
        print(f"{'='*60}")
        print(f"Processing {pos_label}")
        print(f"{'='*60}")

        result = _process_single_pos(pos_label, pos_dir, GRID_2PER_DIR)
        if result is not None:
            all_results.append(result)
        print()

    if not all_results:
        print("No positions processed successfully.")
        return

    # --- Diagnostic figure (one per Pos) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figure_logger import save_figure

    for res in all_results:
        n_z = res["n_z"]
        deltas = res["deltas"]
        pos_label = res["pos_label"]

        fig, axes = plt.subplots(1, max(n_z, 1), figsize=(5 * n_z, 4), squeeze=False)

        vmax_abs = max(np.percentile(np.abs(d), 99) for d in deltas.values())
        for j, (tl_z, grid_z) in enumerate(Z_PAIRS):
            ax = axes[0, j]
            d = deltas[grid_z]
            im = ax.imshow(d, cmap="RdBu_r", vmin=-vmax_abs, vmax=vmax_abs)
            ax.set_title(f"z_grid={grid_z} (z_tl={tl_z})")
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")
            fig.colorbar(im, ax=ax, label="Phase (rad)")

        fig.suptitle(
            f"{pos_label} timelapse delta -- frame {res['closest_idx']}, "
            f"shift=({res['shift_x']:.2f}, {res['shift_y']:.2f}) px, "
            f"|shift|={res['shift_mag']:.2f} px",
            fontsize=11,
        )
        fig.tight_layout()

        save_figure(
            fig,
            params={
                "pos_label": pos_label,
                "closest_frame_index": res["closest_idx"],
                "shift_x": res["shift_x"],
                "shift_y": res["shift_y"],
                "shift_magnitude_px": res["shift_mag"],
                "n_z_pairs": n_z,
            },
            description=(
                f"{pos_label} timelapse-derived per-z delta: frame {res['closest_idx']} "
                f"vs grid_2per(0,0), {n_z} z-slices"
            ),
            data={
                "delta_means": np.array([di["delta_mean_rad"] for di in res["delta_info"]]),
                "delta_stds": np.array([di["delta_std_rad"] for di in res["delta_info"]]),
                "z_pairs": np.array(Z_PAIRS),
            },
        )
        plt.close(fig)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Summary: {len(all_results)}/{len(pos_list)} positions processed")
    print(f"{'='*60}")
    for res in all_results:
        print(f"  {res['pos_label']}: frame {res['closest_idx']}, "
              f"|shift|={res['shift_mag']:.3f} px -> {res['out_dir']}")
    print(f"\nDelta TIF output: PosN/output_phase/channels/{OUTPUT_SUBDIR}/")


if __name__ == "__main__":
    main()

# %%
