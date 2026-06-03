"""Compute per-Pos delta TIFs from 0per_gluc timelapse vs grid_2per raw phase.

For each Pos:
  1. Scan all frames in 0per_gluc/PosN/z{Z}/output_phase_raw/
  2. ECC-align each frame against grid_2per/PosN_x+0_y+0/output_phase_raw/
  3. Pick the frame with highest ECC correlation
  4. delta = aligned(best_0per_frame) - grid_2per
  5. Save delta_z{Z:03d}.tif + log JSON

Output matches extract_timelapse_delta.py format so correct_0pergluc.py
can consume it via per-Pos DELTA_TIFS_DIR.

Output: CROP_SUB_ROOT/PosN/output_phase/channels/delta_timelapse/
"""
import numpy as np
import tifffile
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grid_subtract import apply_inverse_shift_warp
from ecc_utils import to_uint8, ecc_align
from compute_pos_shifts import compute_backsub_offset

GRID_0PER_DIR = Path(r"E:\260504\0per_gluc")
GRID_2PER_DIR = Path(r"E:\260504\grid_2pergluc_1")
CROP_SUB_ROOT = Path(r"D:\AquisitionData\Kitagishi\260508\online_crop_sub_zstack")

GRID_Z_INDEX = 5
POS_START = 1
POS_END = 104

VMIN = -5.0
VMAX = 2.0


def _load_raw(path):
    if not path.exists():
        raise FileNotFoundError(f"Raw phase not found: {path}")
    return tifffile.imread(str(path)).astype(np.float64)


def main():
    results = []

    for n in range(POS_START, POS_END + 1):
        label = f"Pos{n}"
        z = GRID_Z_INDEX

        opr_dir = GRID_0PER_DIR / label / f"z{z:03d}" / "output_phase_raw"
        raw_2per_path = (
            GRID_2PER_DIR / f"{label}_x+0_y+0" / "output_phase_raw"
            / f"img_000000000_ph_{z:03d}_phase.tif"
        )

        if not opr_dir.is_dir():
            print(f"  SKIP {label}: 0per raw dir not found")
            continue
        if not raw_2per_path.exists():
            print(f"  SKIP {label}: 2per raw not found")
            continue

        frames_0per = sorted(opr_dir.glob(f"img_*_ph_{z:03d}_phase.tif"))
        if not frames_0per:
            print(f"  SKIP {label}: no 0per frames found")
            continue

        img_2per = _load_raw(raw_2per_path)
        bs_2 = compute_backsub_offset(img_2per)
        ref_u8 = to_uint8(img_2per - bs_2, VMIN, VMAX)

        best_corr = -1.0
        best_tx, best_ty = 0.0, 0.0
        best_frame_idx = 0
        best_frame_path = frames_0per[0]

        for fi, fpath in enumerate(frames_0per):
            img_0 = _load_raw(fpath)
            bs_0 = compute_backsub_offset(img_0)
            tgt_u8 = to_uint8(img_0 - bs_0, VMIN, VMAX)
            ecc_result = ecc_align(ref_u8, tgt_u8, max_iter=20000)
            if ecc_result is None:
                continue
            tx, ty, corr = ecc_result
            if corr > best_corr:
                best_corr = corr
                best_tx, best_ty = tx, ty
                best_frame_idx = fi
                best_frame_path = fpath

        if best_corr < 0:
            print(f"  SKIP {label}: all ECC failed")
            continue

        img_best = _load_raw(best_frame_path)
        aligned = apply_inverse_shift_warp(img_best, best_tx, best_ty)
        delta = aligned - img_2per

        out_dir = CROP_SUB_ROOT / label / "output_phase" / "channels" / "delta_timelapse"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"delta_z{z:03d}.tif"
        tifffile.imwrite(str(out_path), delta.astype(np.float32))

        log = {
            "pos_label": label,
            "grid_0per_dir": str(GRID_0PER_DIR),
            "grid_2per_dir": str(GRID_2PER_DIR),
            "closest_frame_index": best_frame_idx,
            "closest_shift_x": float(best_tx),
            "closest_shift_y": float(best_ty),
            "closest_shift_magnitude_px": float((best_tx**2 + best_ty**2)**0.5),
            "ecc_correlation": float(best_corr),
            "grid_z_index": z,
            "n_frames_scanned": len(frames_0per),
            "deltas": [{
                "tl_z_index": z,
                "grid_z_index": z,
                "delta_file": out_path.name,
                "delta_mean_rad": float(delta.mean()),
                "delta_std_rad": float(delta.std()),
            }],
        }
        log_path = out_dir / "extract_timelapse_delta_log.json"
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  {label}: frame={best_frame_idx}/{len(frames_0per)} "
              f"tx={best_tx:.3f} ty={best_ty:.3f} corr={best_corr:.4f} "
              f"delta_mean={delta.mean():.6f} rad")
        results.append(label)

    print(f"\nDone: {len(results)} Pos processed")


if __name__ == "__main__":
    main()
