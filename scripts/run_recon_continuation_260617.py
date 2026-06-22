"""
One-shot scheduled runner for the 260617 _2 -> _1 continuation.

1. Run scheduled_recon_and_calibrate.py (raw: C:\\260617\\2per_grid_0p05um_2,
   output: E:\\260617\\2per_grid_0p05um_1, reuses existing Pos0 BG in _1).
2. Verify each target Pos (Pos76..Pos104) is fully reconstructed + calibrated
   in the output dir: output grid-folder count >= raw grid-folder count AND
   grid_calibration_PosN.json exists.
3. Only for verified Pos, delete the corresponding raw holograms in _2 to free C:.
   Pos0 (BG) raw is deleted last, after all targets are verified (its raw is
   unused because the BG is reused from _1).

Deletion is gated per-Pos: a Pos whose reconstruction/calibration did not fully
complete keeps its raw untouched.
"""
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

RAW_DIR = Path(r"C:\260617\2per_grid_0p05um_2")
OUT_DIR = Path(r"E:\260617\2per_grid_0p05um_1")
BG_LABEL = "Pos0"
TARGETS = [f"Pos{n}" for n in range(76, 105)]  # Pos76..Pos104


def grid_folders(base: Path, label: str):
    """Per-grid-point folders for a base label, e.g. Pos76_x+0_y+0."""
    pat = re.compile(rf"^{label}_x[+-]\d+_y[+-]\d+$")
    if not base.exists():
        return []
    return [d for d in base.iterdir() if d.is_dir() and pat.match(d.name)]


def free_gb(path: Path):
    return shutil.disk_usage(path).free / (1024 ** 3)


def main():
    t0 = time.perf_counter()
    print(f"[runner] start  raw={RAW_DIR}  out={OUT_DIR}", flush=True)
    print(f"[runner] C: free before: {free_gb(Path('C:/')):.1f} GB", flush=True)

    # --- Step 1: reconstruction + detect + calibration ---
    rc = subprocess.run(
        [PYTHON, str(SCRIPT_DIR / "scheduled_recon_and_calibrate.py")],
        cwd=str(SCRIPT_DIR),
    ).returncode
    print(f"[runner] pipeline exit code: {rc}", flush=True)

    # --- Step 2/3: verify per-Pos, then delete verified raw ---
    deleted, kept = [], []
    for label in TARGETS:
        raw_n = len(grid_folders(RAW_DIR, label))
        out_n = len(grid_folders(OUT_DIR, label))
        calib = (OUT_DIR / f"grid_calibration_{label}.json").exists()
        if raw_n == 0:
            kept.append((label, f"no raw (raw={raw_n})"))
            continue
        ok = out_n >= raw_n and calib
        print(f"[verify] {label}: raw={raw_n} out={out_n} calib={calib} -> "
              f"{'OK' if ok else 'INCOMPLETE'}", flush=True)
        if not ok:
            kept.append((label, f"incomplete (out={out_n}/{raw_n}, calib={calib})"))
            continue
        for d in grid_folders(RAW_DIR, label):
            shutil.rmtree(d, ignore_errors=True)
        deleted.append(label)

    # Delete BG raw last, only if every target was verified+deleted.
    if not kept:
        bg_n = len(grid_folders(RAW_DIR, BG_LABEL))
        for d in grid_folders(RAW_DIR, BG_LABEL):
            shutil.rmtree(d, ignore_errors=True)
        print(f"[runner] deleted BG {BG_LABEL} raw ({bg_n} folders) - all targets verified",
              flush=True)
    else:
        print(f"[runner] BG {BG_LABEL} raw KEPT - some targets incomplete", flush=True)

    print(f"\n[runner] deleted raw for {len(deleted)} Pos: {deleted}", flush=True)
    if kept:
        print(f"[runner] KEPT raw for {len(kept)} Pos (not fully done):", flush=True)
        for label, why in kept:
            print(f"    {label}: {why}", flush=True)
    print(f"[runner] C: free after: {free_gb(Path('C:/')):.1f} GB", flush=True)
    print(f"[runner] done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
