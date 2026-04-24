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

# ============================================================
# Configuration
# ============================================================

TIMELAPSE_ROOT = r"D:\AquisitionData\Kitagishi\260416\ph_zstack_3"

GRID_2PER_DIR = r"C:\260416\2per_gridgluc_1"

# Pos range [inclusive]. Pos0 is always BG, skipped.
# POS_END = None -> auto-detect last Pos.
POS_START = 1
POS_END = None

# Per-Pos shifts JSON filename (under PosN/output_phase/channels/)
SHIFTS_FILENAME = "pos_shifts_cal_online.json"

# Z_PAIRS: list of (timelapse_z_index, grid_z_index).
# Physical z (um) must match between timelapse and grid z-stacks.
#
# 260416+ grid: 11 slices, -2.0 to +2.0 um, 0.4 um/step, z=5 = 0.0 um
#
# 11-slice timelapse (z_start=-2.0, step=0.4, n_z=11) -> 1:1 with grid:
#   tl_z=0 (-2.0 um) -> grid_z=0
#   tl_z=5 ( 0.0 um) -> grid_z=5  (focus, ECC reference)
#   tl_z=10(+2.0 um) -> grid_z=10
Z_PAIRS = [(i, i) for i in range(11)]

# Frame index range to search [inclusive, exclusive). None = all frames.
FRAME_RANGE = None  # e.g., (575, 1151)

# Output subfolder name (under PosN/output_phase/channels/)
OUTPUT_SUBDIR = "delta_timelapse"

# ============================================================


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


def _load_timelapse_raw(tl_dir, frame_index, z_index, crop):
    """Load timelapse raw phase at (frame_index, z_index).

    Search order:
      1. PosN/output_phase_raw/ (flat, single-z pipeline)
      2. PosN/z{z:03d}/output_phase_raw/ (z-stack online prerecon)
      3. PosN/z{z:03d}/ raw hologram (on-the-fly reconstruction)
      4. PosN/ raw hologram flat (on-the-fly reconstruction)
    """
    tl = Path(tl_dir)
    pat = f"img_*_ph_{z_index:03d}_phase.tif"
    holo_pat = f"img_*_ph_{z_index:03d}.tif"

    # 1. Flat output_phase_raw
    flat = tl / "output_phase_raw"
    if flat.is_dir():
        frames = sorted(flat.glob(pat))
        if frame_index < len(frames):
            img = tifffile.imread(str(frames[frame_index])).astype(np.float64)
            return img, frames[frame_index], "prerecon"

    # 2. Z-subdir output_phase_raw (compute_drift_online z-stack)
    z_raw = tl / f"z{z_index:03d}" / "output_phase_raw"
    if z_raw.is_dir():
        frames = sorted(z_raw.glob(pat))
        if frame_index < len(frames):
            img = tifffile.imread(str(frames[frame_index])).astype(np.float64)
            return img, frames[frame_index], "prerecon-zdir"

    # 3. Z-subdir raw hologram -> on-the-fly
    z_dir = tl / f"z{z_index:03d}"
    if z_dir.is_dir():
        holos = sorted(z_dir.glob(holo_pat))
        if frame_index < len(holos):
            h = holos[frame_index]
            print(f"  [reconstruct on-the-fly] {h}")
            phase = reconstruct_from_holo(h, crop)
            return phase.astype(np.float64), h, "on-the-fly"

    # 4. Flat raw hologram -> on-the-fly
    holos = sorted(tl.glob(holo_pat))
    if frame_index < len(holos):
        h = holos[frame_index]
        print(f"  [reconstruct on-the-fly] {h}")
        phase = reconstruct_from_holo(h, crop)
        return phase.astype(np.float64), h, "on-the-fly"

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


def _process_single_pos(pos_label, pos_dir, grid_2per_dir):
    """Process one Pos: find closest frame, compute deltas, save TIFs."""
    shifts_path = pos_dir / "output_phase" / "channels" / SHIFTS_FILENAME
    if not shifts_path.exists():
        print(f"  SKIP: {SHIFTS_FILENAME} not found at {shifts_path}")
        return None

    frame_results = _load_shifts(shifts_path)

    if FRAME_RANGE:
        start, end = FRAME_RANGE
        candidates = [fr for fr in frame_results if start <= fr["frame_index"] < end]
        print(f"  Frame range: [{start}, {end}), {len(candidates)} candidates")
    else:
        candidates = frame_results
        print(f"  All frames: {len(candidates)} candidates")

    if not candidates:
        print(f"  SKIP: no candidate frames")
        return None

    closest = min(candidates, key=lambda fr: sum(x**2 for x in _get_shift(fr)))
    closest_idx = closest["frame_index"]
    sx, sy = _get_shift(closest)
    mag = (sx**2 + sy**2) ** 0.5
    print(f"  Closest frame: index={closest_idx}, "
          f"shift=({sx:.3f}, {sy:.3f}) px, magnitude={mag:.3f} px")

    out_dir = pos_dir / "output_phase" / "channels" / OUTPUT_SUBDIR
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

    print(f"TIMELAPSE_ROOT: {root}")
    print(f"GRID_2PER_DIR:  {GRID_2PER_DIR}")
    print(f"Positions:      {[p[1].name for p in pos_list]}")
    print(f"Z_PAIRS:        {len(Z_PAIRS)} pairs")
    print(f"FRAME_RANGE:    {FRAME_RANGE}")
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
