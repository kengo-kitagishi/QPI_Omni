"""Finish both 261004 grids while the scope is unattended, then free E:.

The user authorised deleting the grid RAW HOLOGRAMS once each grid is properly
reconstructed: the holograms are not needed any more if the reconstruction is
sound, delete them outright, and do the same for the 0% grid (2026-10-05).
Raw here means only
E:\261004\<grid>\PosN_x+i_y+j\img_000000000_ph_000.tif (17.48 MB each);
output_phase / output_phase_raw are never touched.

Order matters:
  1. wait until the 0% grid acquisition stops writing to E:
     Deleting 26620 files from E: while Micro-Manager is saving to it is the
     documented way to lose frames, so nothing touches E: until then.
  2. verify the 2% recon on D:, then delete the 2% raw from E:  (frees ~233 GB)
  3. reconstruct + detect + calibrate the 0% grid
  4. verify it, then delete the 0% raw from E:                  (frees ~233 GB)

Every delete is gated on its own verification: a point missing output_phase, a
point short of z slices, or a missing grid_calibration_*.json aborts that
delete and leaves the raw alone. Deletion is permanent -- there is no copy on
F: -- so after this the reconstruction on D: is the only copy of each grid
reference.

The timelapse is stopped while this runs. Resuming it mid-run would put the
acquisition and this script on E: at the same time.
"""
import importlib.util
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "watch_grid_impl", SCRIPT_DIR / "watch_grid_then_recon_260922_sz.py"
)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)

SESSION_DIR = Path(r"E:\261004")
GRID_2PER_RAW = Path(r"E:\261004\2%_grid_hologram_1")
GRID_2PER_OUT = Path(r"D:\AquisitionData\Kitagishi\261004\2%_grid_hologram_1")
GRID_0PER_RAW = Path(r"E:\261004\0%_grid_hologram_1")
GRID_0PER_OUT = Path(r"D:\AquisitionData\Kitagishi\261004\0%_grid_hologram_1")

IDLE_MINUTES = 10      # quiet time on the 0% grid before anything touches E:
POLL_SECONDS = 120

# Geometry shared by both grids of this session (see run_recon_calib_261004.py).
impl.GRID_HALF = 5
impl.N_Z = 1
impl.Z_INDEX = 0
impl.POS_SPLIT = 56
impl.RECON_Z_INDICES = [0]
impl.LOG_PATH = SESSION_DIR / "finish_261004_grids.log"
impl.ALERT_PATH = SESSION_DIR / "finish_261004_grids.ALERT.txt"

log, alert, free_gb = impl.log, impl.alert, impl.free_gb
POINT_RE = re.compile(r"^Pos(\d+)_x([+-]\d+)_y([+-]\d+)$")
N_POINTS = (2 * impl.GRID_HALF + 1) ** 2


def point_dirs(grid_dir):
    return [d for d in grid_dir.iterdir() if d.is_dir() and POINT_RE.match(d.name)]


def complete_pos(grid_dir):
    """Pos numbers whose 121 point folders all hold an image."""
    by_pos = {}
    for d in point_dirs(grid_dir):
        m = POINT_RE.match(d.name)
        try:
            if not any(f.name.startswith("img_") for f in d.iterdir()):
                continue
        except OSError:
            continue
        by_pos.setdefault(int(m.group(1)), set()).add((m.group(2), m.group(3)))
    return sorted(pos for pos, pts in by_pos.items() if len(pts) == N_POINTS)


def wait_until_idle(grid_dir):
    """Block until no new point folder has appeared for IDLE_MINUTES.

    Micro-Manager is saving into this directory, so each point folder is stat'ed
    once and remembered: re-stat'ing all 13310 of them every poll is what made a
    watcher the prime suspect for the Pos50 image-saving failure on 2026-09-09.
    Only names that are new since the last poll cost any I/O, and listing the
    directory itself is one call.
    """
    log(f"Waiting for {grid_dir} to stop writing (idle >= {IDLE_MINUTES} min)")
    seen = {}          # folder name -> mtime, stat'ed once
    newest = 0.0
    last_n = -1
    while True:
        for d in grid_dir.iterdir():
            if d.name in seen or not POINT_RE.match(d.name):
                continue
            try:
                mt = d.stat().st_mtime
            except OSError:
                continue
            seen[d.name] = mt
            if mt > newest:
                newest = mt
        n = len(seen)
        idle_min = (time.time() - newest) / 60.0 if newest else 0.0
        if n != last_n:
            log(f"  {n} points, newest {datetime.fromtimestamp(newest):%H:%M:%S}, "
                f"idle={idle_min:.0f}min, E:free={free_gb('E:/'):.0f}GB")
            last_n = n
        if idle_min >= IDLE_MINUTES:
            log(f"  idle {idle_min:.0f} min: acquisition finished at {n} points")
            return
        time.sleep(POLL_SECONDS)


def verify(out_dir, pos_list, label):
    log(f"Verifying {label}: {out_dir}")
    impl.N_Z = 1
    ok = impl.verify_output(out_dir, pos_list)
    log(f"  {label}: {'verification passed' if ok else 'VERIFICATION FAILED'}")
    return ok


def delete_raw(grid_dir, label):
    victims = point_dirs(grid_dir)
    before = free_gb("E:/")
    log(f"Deleting {len(victims)} raw point folders of {label} under {grid_dir}")
    t0 = time.time()
    for i, d in enumerate(victims, 1):
        shutil.rmtree(d)
        if i % 2000 == 0:
            log(f"  {i}/{len(victims)}  E:free={free_gb('E:/'):.0f}GB"
                f"  ({i/(time.time()-t0):.0f} folders/s)")
    after = free_gb("E:/")
    log(f"{label} raw deleted in {(time.time()-t0)/60:.1f} min. "
        f"E: free {before:.0f} GB -> {after:.0f} GB (+{after-before:.0f} GB)")


def main():
    log("=" * 70)
    log("finish_261004_grids: 0% grid -> recon -> delete both grid raws from E:")
    log("=" * 70)

    # 1. nothing touches E: while Micro-Manager is still saving to it
    wait_until_idle(GRID_0PER_RAW)

    # 2. the 2% grid was reconstructed and calibrated at 15:22 today
    pos_2per = complete_pos(GRID_2PER_RAW)
    if not pos_2per:
        log("2% grid raw is already gone; nothing to delete there")
    elif verify(GRID_2PER_OUT, pos_2per, "2% grid"):
        delete_raw(GRID_2PER_RAW, "2% grid")
    else:
        alert("2% grid verification FAILED. Raw kept. Inspect before deleting.")

    # 3. reconstruct the 0% grid
    log("=" * 70)
    log("Reconstructing the 0% grid")
    import subprocess
    rc = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "run_recon_calib_261004_0per.py"), "--skip-wait"],
        cwd=str(SCRIPT_DIR),
    ).returncode
    log(f"0% grid pipeline exit {rc}")
    if rc != 0:
        alert(f"0% grid recon FAILED (exit {rc}). Its raw is kept.")
        return 1

    # 4. verify, then free the rest of E:
    pos_0per = complete_pos(GRID_0PER_RAW)
    if verify(GRID_0PER_OUT, pos_0per, "0% grid"):
        delete_raw(GRID_0PER_RAW, "0% grid")
    else:
        alert("0% grid verification FAILED. Raw kept. Inspect before deleting.")
        return 1

    alert(f"261004 grids done. E: now {free_gb('E:/'):.0f} GB free. "
          f"The timelapse can be resumed (re-Run the bsh; T=7 picks up the grid "
          f"anchor in drift_session_261004/grid_t0.txt).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
