"""
complete_crop_sub.py
--------------------
Fill in Phase B (crop-subtracted) frames that were skipped during the
running timelapse due to time budget limits in compute_drift_online.py.

Safe to run while timelapse is active: reads only from output_phase_raw/
(Phase A output), writes atomically to crop_sub_rawraw/, and does not
import from or modify compute_drift_online.py.

Usage:
    python complete_crop_sub.py [--dry-run] [--no-cleanup]
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import grid_subtract as gs


def load_config():
    session_dir = Path(__file__).resolve().parent.parent / "drift_session"
    cfg_path = session_dir / "drift_config.json"
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def get_positions(cfg):
    """Return list of (pos_num, pos_label) excluding BG (Pos0)."""
    save_dir = Path(cfg["save_dir"])
    positions = []
    for d in sorted(save_dir.iterdir()):
        m = re.match(r"Pos(\d+)$", d.name)
        if m:
            num = int(m.group(1))
            if num == cfg.get("bg_pos_index", 0):
                continue
            positions.append((num, d.name))
    return positions


def extract_frame_nums(directory, pattern_suffix):
    """Extract sorted set of frame numbers from files matching img_{NNNNNNNNN}_{suffix}."""
    nums = set()
    p = Path(directory)
    if not p.exists():
        return nums
    pat = re.compile(r"img_(\d{9})_" + pattern_suffix)
    for f in p.iterdir():
        m = pat.match(f.name)
        if m:
            nums.add(int(m.group(1)))
    return nums


def interpolate_shift(frame_results, t):
    """Get shift for frame t by nearest-neighbor interpolation from pos_shifts_cal_online."""
    n = len(frame_results)
    if t < n and frame_results[t] is not None:
        r = frame_results[t]
        return r["shift_x_avg"], r["shift_y_avg"]

    best_dist = float("inf")
    best_entry = None
    for delta in range(1, max(t + 1, n - t)):
        for candidate in [t - delta, t + delta]:
            if 0 <= candidate < n and frame_results[candidate] is not None:
                if abs(candidate - t) < best_dist:
                    best_dist = abs(candidate - t)
                    best_entry = frame_results[candidate]
        if best_entry is not None:
            break
    if best_entry is None:
        return 0.0, 0.0
    return best_entry["shift_x_avg"], best_entry["shift_y_avg"]


def append_frame_entry(json_path, frame_entry, cfg, pos_label):
    """Atomic append of one frame entry to pos_shifts_cal_online.json."""
    pos_num = int(re.match(r"Pos(\d+)", pos_label).group(1))
    pos_split = int(cfg.get("pos_split", 0))
    raw_crop = cfg["crop_before"] if pos_num < pos_split else cfg["crop_after"]

    default = {
        "schema_version": 2,
        "source": "online_crop_sub",
        "base_label": pos_label,
        "grid_dir": cfg.get("grid_dir", ""),
        "grid_z_index": int(cfg.get("raw_grid_z_index", 18)),
        "tl_z_index": int(cfg.get("raw_tl_z_index", 0)),
        "x_step_um": float(cfg.get("crop_sub_x_step_um", 0.1)),
        "y_step_um": float(cfg.get("crop_sub_y_step_um", 0.1)),
        "shift_sign_x": int(cfg.get("shift_sign_x", -1)),
        "shift_sign_y": int(cfg.get("shift_sign_y", -1)),
        "apply_subpixel_correction": True,
        "apply_inverse_shift": False,
        "use_raw_phase": True,
        "raw_crop": list(raw_crop),
        "tilt_crop_h_raw": int(cfg.get("tilt_crop_h_raw", 270)),
        "frame_results": [],
    }

    data = default
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                loaded = json.load(f)
            for k, v in default.items():
                loaded.setdefault(k, v)
            loaded["schema_version"] = default["schema_version"]
            data = loaded
        except Exception:
            pass

    fr = data.setdefault("frame_results", [])
    t_idx = int(frame_entry["frame_index"])
    while len(fr) <= t_idx:
        fr.append(None)
    fr[t_idx] = frame_entry
    data["n_frames_seen"] = sum(1 for x in fr if x is not None)

    tmp = json_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp), str(json_path))


def process_position(pos_num, pos_label, cfg, dry_run=False, cleanup=True):
    """Process all missing Phase B frames for one position."""
    save_dir = Path(cfg["save_dir"])
    crop_sub_root = Path(cfg["crop_sub_root"])
    grid_dir = Path(cfg["grid_dir"])
    pos_split = int(cfg.get("pos_split", 0))
    tilt_h = cfg.get("tilt_crop_h_raw", 270)
    z_idx = cfg.get("raw_tl_z_index", 0)
    grid_z = z_idx + cfg["raw_grid_z_index"] - cfg.get("raw_tl_z_index", 0)
    fit_right = pos_num >= pos_split

    phase_raw_dir = save_dir / pos_label / f"z{z_idx:03d}" / "output_phase_raw"
    crop_sub_ch00 = (crop_sub_root / pos_label / "output_phase" /
                     "channels" / "crop_sub_rawraw" / f"z{z_idx:03d}" / "ch00")
    out_base = (crop_sub_root / pos_label / "output_phase" /
                "channels" / "crop_sub_rawraw")
    json_path = (crop_sub_root / pos_label / "output_phase" /
                 "channels" / "pos_shifts_cal_online.json")

    phase_a_frames = extract_frame_nums(phase_raw_dir, r"ph_\d{3}_phase\.tif")
    phase_b_frames = extract_frame_nums(crop_sub_ch00, r"ph_\d{3}\.tif")
    missing = sorted(phase_a_frames - phase_b_frames)

    if not missing:
        return pos_label, 0, []

    rois_path = (grid_dir / f"{pos_label}_x+0_y+0" / "output_phase" /
                 "channels" / "channel_rois.json")
    if not rois_path.exists():
        return pos_label, 0, [f"channel_rois.json not found: {rois_path}"]

    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)

    cal_path = grid_dir / f"grid_calibration_{pos_label}.json"
    grid_cal = gs.load_grid_calibration(str(cal_path)) if cal_path.exists() else {}

    pos_map = gs.scan_grid_positions(grid_dir, pos_label)
    if not pos_map:
        return pos_label, 0, [f"no grid positions found for {pos_label}"]

    pixel_scale_um = (cfg.get("sensor_pixel_size", 3.45e-6)
                      / cfg.get("magnification", 40)
                      * cfg.get("original_dim", 2048)
                      / cfg.get("reconstructed_dim", 511) * 1e6)

    frame_results = []
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            frame_results = data.get("frame_results", [])
        except Exception:
            pass

    if dry_run:
        return pos_label, len(missing), []

    grid_img_cache = {}
    errors = []
    n_saved = 0

    for t in missing:
        try:
            sx, sy = interpolate_shift(frame_results, t)

            xi, yi, dist_um, dx_um, dy_um, cal_dx, cal_dy, residual_x, residual_y = \
                gs.select_grid(sx, sy, pos_map, grid_cal, pixel_scale_um,
                               x_step=cfg.get("crop_sub_x_step_um", 0.1),
                               y_step=cfg.get("crop_sub_y_step_um", 0.1),
                               shift_sign_x=-1, shift_sign_y=-1)

            prerecon_tl = phase_raw_dir / f"img_{t:09d}_ph_{z_idx:03d}_phase.tif"
            if not prerecon_tl.exists():
                errors.append(f"frame {t}: phase_raw not found")
                continue
            tl_img = tifffile.imread(str(prerecon_tl)).astype(np.float64)

            grid_key = (xi, yi)
            if grid_key not in grid_img_cache:
                grid_pos_dir = pos_map.get(grid_key)
                grid_img = None
                if grid_pos_dir is not None:
                    prerecon_grid = (grid_pos_dir / "output_phase_raw" /
                                    f"img_000000000_ph_{grid_z:03d}_phase.tif")
                    if prerecon_grid.exists():
                        grid_img = tifffile.imread(str(prerecon_grid)).astype(np.float64)
                grid_img_cache[grid_key] = grid_img
            grid_img = grid_img_cache[grid_key]

            per_channel_out, _ = gs.process_single_frame(
                tl_img, sx, sy, rois,
                cal_dx, cal_dy, residual_x, residual_y,
                grid_img,
                output_crop_h_override=tilt_h,
                tilt_crop_h_raw=tilt_h,
                use_raw_phase=True,
                apply_subpixel_correction=True,
                fit_right=fit_right,
                apply_inverse_shift=False,
            )

            raw_name = f"img_{t:09d}_ph_{z_idx:03d}.tif"
            for ch in range(len(rois)):
                ch_dir = out_base / f"z{z_idx:03d}" / f"ch{ch:02d}"
                ch_dir.mkdir(parents=True, exist_ok=True)
                final = ch_dir / raw_name
                tmp = ch_dir / (raw_name + ".tmp")
                tifffile.imwrite(str(tmp), per_channel_out[ch].astype(np.float32))
                os.replace(str(tmp), str(final))

            frame_entry = {
                "frame_index": int(t),
                "shift_x_avg": float(sx),
                "shift_y_avg": float(sy),
                "grid_xi": int(xi),
                "grid_yi": int(yi),
                "residual_x_px": float(residual_x),
                "residual_y_px": float(residual_y),
                "grid_nearest_dist_um": float(dist_um) if dist_um is not None else None,
                "source": "complete_crop_sub",
            }
            append_frame_entry(json_path, frame_entry, cfg, pos_label)
            n_saved += 1

        except Exception as e:
            errors.append(f"frame {t}: {e}")

    if cleanup and n_saved > 0:
        raw_dir = save_dir / pos_label / f"z{z_idx:03d}"
        for raw_file in raw_dir.glob(f"img_*_ph_{z_idx:03d}.tif"):
            m = re.match(r"img_(\d{9})_ph_", raw_file.name)
            if m:
                frame_num = int(m.group(1))
                ch00_out = crop_sub_ch00 / raw_file.name
                if ch00_out.exists():
                    raw_file.unlink()

    return pos_label, n_saved, errors


def main():
    parser = argparse.ArgumentParser(description="Complete missing Phase B crop-sub frames")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and report missing frames without processing")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Skip raw hologram cleanup after processing")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel position workers")
    args = parser.parse_args()

    cfg = load_config()
    positions = get_positions(cfg)
    print(f"Positions: {len(positions)}  save_dir: {cfg['save_dir']}")
    print(f"crop_sub_root: {cfg['crop_sub_root']}")
    print(f"grid_dir: {cfg['grid_dir']}")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        total_missing = 0
        for pos_num, pos_label in positions:
            _, n_missing, errs = process_position(pos_num, pos_label, cfg, dry_run=True)
            if n_missing > 0:
                print(f"  {pos_label}: {n_missing} missing")
                total_missing += n_missing
            for e in errs:
                print(f"    ERROR: {e}")
        print(f"\nTotal missing frames: {total_missing}")
        return

    t0 = time.time()
    total_saved = 0
    total_errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for pos_num, pos_label in positions:
            fut = pool.submit(process_position, pos_num, pos_label, cfg,
                              dry_run=False, cleanup=not args.no_cleanup)
            futures[fut] = pos_label

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Positions"):
            pos_label, n_saved, errors = fut.result()
            total_saved += n_saved
            total_errors += len(errors)
            if n_saved > 0 or errors:
                tqdm.write(f"  {pos_label}: saved={n_saved}  errors={len(errors)}")
            for e in errors[:3]:
                tqdm.write(f"    {e}")

    elapsed = time.time() - t0
    print(f"\nDone: {total_saved} frames saved  {total_errors} errors  {elapsed:.1f}s")


if __name__ == "__main__":
    main()
