"""
Partial runner: reconstruct + calibrate a SPECIFIC list of Pos from the 0per _2
batch into E:\\260617\\0per_grid_0p05um_1, then delete their C: raw to free space.

Use when C: is full mid-acquisition: free just enough space to continue capturing.
Reuses the existing Pos0 BG already reconstructed in _1 (no BG re-run).
Does NOT touch any Pos outside TARGETS (e.g. a partial last Pos being re-captured).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

RAW = Path(r"C:\260617\0per_grid_0p05um_2")
OUT = Path(r"E:\260617\0per_grid_0p05um_1")
Z_INDEX = 5
POS_SPLIT = 53
TARGETS = [f"Pos{n}" for n in range(31, 36)]  # Pos31..Pos35


def grid_folders(base, label):
    pat = re.compile(rf"^{label}_x[+-]\d+_y[+-]\d+$")
    return [d for d in base.iterdir() if d.is_dir() and pat.match(d.name)] if base.exists() else []


def free_gb(path):
    return shutil.disk_usage(path).free / (1024 ** 3)


def main():
    print(f"[partial] targets: {TARGETS}", flush=True)
    print(f"[partial] C: free before: {free_gb(Path('C:/')):.1f} GB", flush=True)

    # Step 1: reconstruction (BG reused from OUT, output to OUT)
    rc = subprocess.run(
        [PYTHON, str(SCRIPT_DIR / "batch_reconstruction_grid.py"),
         "--grid-dir", str(RAW), "--output-dir", str(OUT), "--targets", *TARGETS],
        cwd=str(SCRIPT_DIR),
    ).returncode
    print(f"[partial] recon rc: {rc}", flush=True)
    if rc != 0:
        sys.exit(rc)

    # Step 2: channel detect (skip if already present)
    for label in TARGETS:
        op = OUT / f"{label}_x+0_y+0" / "output_phase"
        rois = op / "channels" / "channel_rois.json"
        if rois.exists():
            continue
        subprocess.run(
            [PYTHON, str(SCRIPT_DIR / "channel_crop.py"),
             "--dir", str(op), "--detect", "--pattern", f"*_ph_{Z_INDEX:03d}_phase.tif"],
            cwd=str(SCRIPT_DIR),
        )

    # Step 3: calibration (in-process via cgp, per Pos)
    import cv2
    cv2.setNumThreads(1)
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import calibrate_grid_positions as cgp
    for label in TARGETS:
        outp = OUT / f"grid_calibration_{label}.json"
        if outp.exists():
            continue
        rois = OUT / f"{label}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json"
        cgp.GRID_DIR = str(OUT)
        cgp.BASE_LABEL = label
        cgp.GRID_Z_INDEX = Z_INDEX
        cgp.CHANNEL_ROIS_JSON = str(rois)
        cgp.OUTPUT_JSON = str(outp)
        cgp.POS_SPLIT = POS_SPLIT
        cgp.main()
        print(f"[partial] calibrated {label}", flush=True)

    # Step 4: verify per-Pos, then delete verified raw
    deleted, kept = [], []
    for label in TARGETS:
        raw_n = len(grid_folders(RAW, label))
        out_n = len(grid_folders(OUT, label))
        calib = (OUT / f"grid_calibration_{label}.json").exists()
        ok = raw_n > 0 and out_n >= raw_n and calib
        print(f"[verify] {label}: raw={raw_n} out={out_n} calib={calib} -> "
              f"{'OK' if ok else 'INCOMPLETE'}", flush=True)
        if ok:
            for d in grid_folders(RAW, label):
                shutil.rmtree(d, ignore_errors=True)
            deleted.append(label)
        else:
            kept.append(label)

    print(f"\n[partial] deleted raw: {deleted}", flush=True)
    if kept:
        print(f"[partial] KEPT (not fully done): {kept}", flush=True)
    print(f"[partial] C: free after: {free_gb(Path('C:/')):.1f} GB", flush=True)


if __name__ == "__main__":
    main()
