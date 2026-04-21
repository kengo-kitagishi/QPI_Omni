"""
Reconstruct + verify + delete holograms, one Pos at a time.
Ensures disk space is freed after each Pos completes.

Usage:
    python run_recon_cycle.py
    python run_recon_cycle.py --grid-dir C:\260416\lowper_gridgluc_1 --start 1 --end 60
"""
import argparse
import subprocess
import sys
import os
import glob
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "batch_reconstruction_grid.py"
EXPECTED_Z = 11


def verify_pos(grid_dir, pos_num):
    """Check all grid points have EXPECTED_Z files in output_phase and output_phase_raw."""
    pattern = f"Pos{pos_num}_x*_y*"
    dirs = sorted(grid_dir.glob(pattern))
    if not dirs:
        return False, f"No dirs matching {pattern}"
    ok = 0
    for d in dirs:
        n_phase = len(list((d / "output_phase").glob("img_*_phase.tif"))) if (d / "output_phase").exists() else 0
        n_raw = len(list((d / "output_phase_raw").glob("img_*_phase.tif"))) if (d / "output_phase_raw").exists() else 0
        if n_phase >= EXPECTED_Z and n_raw >= EXPECTED_Z:
            ok += 1
    return ok == len(dirs), f"{ok}/{len(dirs)}"


def delete_holograms(grid_dir, pos_num, bg_label="Pos0"):
    """Delete raw holograms for a Pos label. Never deletes BG (Pos0)."""
    if f"Pos{pos_num}" == bg_label:
        print(f"  [SKIP] {bg_label} is BG - holograms preserved")
        return 0
    pattern = str(grid_dir / f"Pos{pos_num}_x*" / "img_*.tif")
    files = glob.glob(pattern)
    for f in files:
        os.remove(f)
    return len(files)


def main():
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", type=str, default=r"C:\260416\0per_gridgluc_1")
    parser.add_argument("--start", type=int, default=16)
    parser.add_argument("--end", type=int, default=60)
    args = parser.parse_args()

    grid_dir = Path(args.grid_dir)
    print(f"Grid dir: {grid_dir}")
    print(f"Range: Pos{args.start} - Pos{args.end}")
    print()

    for pos in range(args.start, args.end + 1):
        print(f"\n{'='*60}")
        print(f"  Pos{pos}: reconstruct")
        print(f"{'='*60}")
        t0 = time.perf_counter()

        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--grid-dir", str(grid_dir), "--targets", f"Pos{pos}"],
            capture_output=False,
        )
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            print(f"  [ERROR] Pos{pos} reconstruction failed (exit {result.returncode})")
            continue

        # Verify
        ok, detail = verify_pos(grid_dir, pos)
        if not ok:
            print(f"  [ERROR] Pos{pos} verification failed: {detail}")
            print(f"  Skipping deletion!")
            continue

        # Delete holograms
        n_deleted = delete_holograms(grid_dir, pos)
        print(f"  Pos{pos}: OK ({detail}) | {elapsed:.0f}s | deleted {n_deleted} holograms")

    print(f"\n{'='*60}")
    print("All done.")


if __name__ == "__main__":
    main()
