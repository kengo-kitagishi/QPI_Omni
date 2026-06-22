"""
Comprehensive runner: reconstruct + calibrate ALL remaining 0per Pos into
E:\\260617\\0per_grid_0p05um_1 (continuation of Pos1-30).

Sources:
  - C:\\260617\\0per_grid_0p05um_2  : Pos31..Pos103 (BG Pos0 reused from _1)
  - C:\\260617\\0per_grid_0p05um_3  : Pos104

Strategy (C: is nearly full -> reconstruction is I/O-throttled):
  reconstruct ONE Pos, then immediately delete its raw to free C:. Freeing space
  per-Pos lets the disk recover so later Pos run at full speed. Calibration +
  channel detect run at the end from the E: output (raw no longer needed).

A loud PC alarm sounds at the very end so the next step isn't missed.
"""
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

RAW2 = Path(r"C:\260617\0per_grid_0p05um_2")
RAW3 = Path(r"C:\260617\0per_grid_0p05um_3")
OUT = Path(r"E:\260617\0per_grid_0p05um_1")
Z_INDEX = 5
POS_SPLIT = 53
BG_LABEL = "Pos0"


def grid_folders(base, label):
    pat = re.compile(rf"^{label}_x[+-]\d+_y[+-]\d+$")
    return [d for d in base.iterdir() if d.is_dir() and pat.match(d.name)] if base.exists() else []


def complete_count(base, label, n_z=11):
    """Number of grid folders that are TRULY reconstructed (all n_z phase tifs).
    Folder existence alone is not enough: output_phase dirs are mkdir'd before
    frames are written, so an interrupted Pos leaves empty/partial folders."""
    cnt = 0
    for d in grid_folders(base, label):
        if len(list((d / "output_phase").glob("*_phase.tif"))) >= n_z:
            cnt += 1
    return cnt


def base_targets(base):
    pat = re.compile(r"^(Pos\d+)_x[+-]\d+_y[+-]\d+$")
    labels = {pat.match(d.name).group(1) for d in base.iterdir()
              if d.is_dir() and pat.match(d.name)} if base.exists() else set()
    labels.discard(BG_LABEL)
    return sorted(labels, key=lambda s: int(s[3:]))


def free_gb(path):
    return shutil.disk_usage(path).free / (1024 ** 3)


def recon_one(raw_dir, label):
    return subprocess.run(
        [PYTHON, str(SCRIPT_DIR / "batch_reconstruction_grid.py"),
         "--grid-dir", str(raw_dir), "--output-dir", str(OUT), "--targets", label],
        cwd=str(SCRIPT_DIR),
    ).returncode


def wake_alert(msg):
    try:
        import winsound
        end = time.time() + 40
        while time.time() < end:
            winsound.Beep(880, 350)
            winsound.Beep(587, 350)
    except Exception as e:
        print(f"[alert] beep failed: {e}", flush=True)
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command",
             "Add-Type -AssemblyName System.Windows.Forms; "
             f"[System.Windows.Forms.MessageBox]::Show('{msg}','QPI recon done')"])
    except Exception as e:
        print(f"[alert] popup failed: {e}", flush=True)


def main():
    t0 = time.perf_counter()
    jobs = [(RAW2, l) for l in base_targets(RAW2)]
    if len(grid_folders(RAW3, "Pos104")) >= 121:
        jobs.append((RAW3, "Pos104"))
    print(f"[all] jobs ({len(jobs)}): {[l for _, l in jobs]}", flush=True)
    print(f"[all] C: free before: {free_gb(Path('C:/')):.1f} GB", flush=True)

    # --- recon loop: one Pos at a time, delete raw right after ---
    for raw_dir, label in jobs:
        raw_n = len(grid_folders(raw_dir, label))
        done_n = complete_count(OUT, label)
        if not (raw_n > 0 and done_n >= raw_n):
            rc = recon_one(raw_dir, label)
            done_n = complete_count(OUT, label)
            print(f"[recon] {label}: rc={rc} done(11z)={done_n}/{raw_n} "
                  f"C_free={free_gb(Path('C:/')):.0f}GB", flush=True)
        else:
            print(f"[recon] {label}: already complete ({done_n} full-z), skip", flush=True)
        # free raw ONLY if every captured grid point is fully reconstructed (11 z)
        if raw_n > 0 and done_n >= raw_n:
            for d in grid_folders(raw_dir, label):
                shutil.rmtree(d, ignore_errors=True)

    # --- channel detect (from E: output) ---
    all_labels = [l for _, l in jobs]
    for label in all_labels:
        op = OUT / f"{label}_x+0_y+0" / "output_phase"
        rois = op / "channels" / "channel_rois.json"
        if rois.exists():
            continue
        subprocess.run(
            [PYTHON, str(SCRIPT_DIR / "channel_crop.py"),
             "--dir", str(op), "--detect", "--pattern", f"*_ph_{Z_INDEX:03d}_phase.tif"],
            cwd=str(SCRIPT_DIR),
        )

    # --- calibration (in-process via cgp) ---
    import cv2
    cv2.setNumThreads(1)
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import calibrate_grid_positions as cgp
    cal_ok, cal_err = [], []
    for label in all_labels:
        outp = OUT / f"grid_calibration_{label}.json"
        if outp.exists():
            cal_ok.append(label)
            continue
        rois = OUT / f"{label}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json"
        cgp.GRID_DIR = str(OUT)
        cgp.BASE_LABEL = label
        cgp.GRID_Z_INDEX = Z_INDEX
        cgp.CHANNEL_ROIS_JSON = str(rois)
        cgp.OUTPUT_JSON = str(outp)
        cgp.POS_SPLIT = POS_SPLIT
        try:
            cgp.main()
            cal_ok.append(label)
            print(f"[calib] {label} OK", flush=True)
        except Exception as e:
            cal_err.append((label, str(e)))
            print(f"[calib] {label} ERROR: {e}", flush=True)

    elapsed = (time.perf_counter() - t0) / 60
    print(f"\n[all] calibrated {len(cal_ok)}/{len(all_labels)}, errors={len(cal_err)}", flush=True)
    for label, e in cal_err:
        print(f"    ERROR {label}: {e}", flush=True)
    print(f"[all] C: free after: {free_gb(Path('C:/')):.1f} GB", flush=True)
    print(f"[all] done in {elapsed:.0f} min", flush=True)
    wake_alert(f"0per recon done: {len(cal_ok)}/{len(all_labels)} calibrated, "
               f"{len(cal_err)} errors. C free {free_gb(Path('C:/')):.0f}GB.")


if __name__ == "__main__":
    main()
