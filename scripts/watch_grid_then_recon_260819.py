"""
watch_grid_then_recon_260819.py

Driver for the 260819 grid session.

  raw     D:\AquisitionData\Kitagishi\260819\grid_ye_1   Pos0..Pos35 (4356 points)
  output  E:\260819\grid_ye_1

Single batch. The raw on D: is deleted once the reconstruction verifies, so
ALL 11 z planes are reconstructed (RECON_Z_INDICES = None) -- restricting z
would make the discarded planes unrecoverable.

What this does:
  1. Wait until the acquisition finishes (all points present AND no new files
     for IDLE_MINUTES).
  2. Patch the pipeline config, run scheduled_recon_and_calibrate.py
     (recon -> channel detect -> calibration), verify the output.
  3. Delete the raw on D: once the output verifies, freeing ~660 GB.

Run:
    python watch_grid_then_recon_260819.py
    python watch_grid_then_recon_260819.py --dry-run    # patch + report only
    python watch_grid_then_recon_260819.py --skip-wait  # acquisition already done
"""
import argparse
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
SESSION_DIR = Path(r"C:\260819")

# The acquisition this script waits for before doing anything.
WATCH_DIR = Path(r"D:\AquisitionData\Kitagishi\260819\grid_ye_1")
WATCH_POS = list(range(0, 36))                  # Pos0..Pos35

BATCHES = [
    {
        "name": "b1",
        "grid_dir": WATCH_DIR,
        "output_dir": Path(r"E:\260819\grid_ye_1"),
        "pos_list": WATCH_POS,
        "delete_raw": True,                      # frees D: (~660 GB)
    },
]

GRID_HALF     = 5       # 11 x 11 = 121 points per Pos
N_Z           = 11      # z-slices per point
Z_INDEX       = 6       # focus plane used for calibration / channel detect
POS_SPLIT     = 50      # unchanged from the previous session; Pos0..Pos35 all
                        # fall on crop_before
# z planes to reconstruct. None = all N_Z. The raw here is deleted after
# verification, so every plane must be reconstructed -- a focus-only recon would
# throw the rest away permanently. ~23 MB/point x 4356 points = ~100 GB.
RECON_Z_INDICES = None

IDLE_MINUTES  = 15      # acquisition considered finished after this much quiet
POLL_MINUTES  = 5
MIN_FREE_GB_E = 150     # recon output is ~100 GB (23 MB/point x 4356)

# Beep on completion/failure: this run ends unattended and both the raw
# deletion and the next acquisition step wait on it.
BEEP_ENABLED = True

LOG_PATH   = SESSION_DIR / "watch_grid_then_recon.log"
ALERT_PATH = SESSION_DIR / "watch_grid_then_recon.ALERT.txt"
# ============================================================

# The console here is cp932. Non-ASCII in a log line must never abort the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON     = sys.executable
POINT_RE   = re.compile(r"^Pos(\d+)_x([+-]\d+)_y([+-]\d+)$")
N_POINTS   = 2 * GRID_HALF + 1


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def alert(msg):
    """Record something the user must see. Silent unless BEEP_ENABLED."""
    log(f"*** {msg} ***")
    try:
        with open(ALERT_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
    except OSError:
        pass
    if BEEP_ENABLED:
        try:
            import winsound
            for _ in range(6):
                winsound.Beep(880, 400)
                time.sleep(0.15)
        except Exception:
            pass


def free_gb(path):
    return shutil.disk_usage(str(path)).free / 1e9


def point_names(pos_list):
    for pos in pos_list:
        for xi in range(-GRID_HALF, GRID_HALF + 1):
            for yi in range(-GRID_HALF, GRID_HALF + 1):
                yield f"Pos{pos}_x{xi:+d}_y{yi:+d}"


def scan_acquired(grid_dir):
    """Count point folders that actually hold raw images, plus newest mtime.

    Empty point folders are ignored: an interrupted reconstruction can leave
    some behind, and they would otherwise be counted as acquired.
    """
    n, newest = 0, 0.0
    for d in grid_dir.iterdir():
        if not (d.is_dir() and POINT_RE.match(d.name)):
            continue
        try:
            if not any(f.name.startswith("img_") for f in d.iterdir()):
                continue
        except OSError:
            continue
        n += 1
        m = d.stat().st_mtime
        if m > newest:
            newest = m
    return n, newest


def wait_for_acquisition():
    expected = len(WATCH_POS) * N_POINTS * N_POINTS
    last_label = f"Pos{WATCH_POS[-1]}_x+{GRID_HALF}_y+{GRID_HALF}"
    log(f"Waiting for acquisition in {WATCH_DIR} "
        f"(expect {expected} points = {len(WATCH_POS)} Pos x {N_POINTS}x{N_POINTS})")

    t_start = time.time()
    n_start = None
    last_n = -1
    t_last_change = time.time()

    while True:
        if not WATCH_DIR.is_dir():
            log(f"  {WATCH_DIR} does not exist yet ...")
            time.sleep(POLL_MINUTES * 60)
            continue

        n, newest = scan_acquired(WATCH_DIR)
        idle_min = (time.time() - newest) / 60.0 if newest else 0.0
        if n_start is None:
            n_start = n

        if n != last_n:
            t_last_change = time.time()
            elapsed = time.time() - t_start
            rate = (n - n_start) / elapsed if elapsed > 0 and n > n_start else 0
            eta = (expected - n) / rate / 3600 if rate > 0 else float("nan")
            log(f"  {n}/{expected} points ({100.0*n/expected:.1f}%)  "
                f"idle={idle_min:.0f}min  ETA~{eta:.1f}h  D:free={free_gb('D:'):.0f}GB")
            last_n = n

        if n >= expected and (WATCH_DIR / last_label).is_dir() and idle_min >= IDLE_MINUTES:
            log(f"Acquisition complete: {n} points, idle {idle_min:.0f} min")
            return True

        if n > 0 and idle_min >= IDLE_MINUTES * 4 and \
                (time.time() - t_last_change) > IDLE_MINUTES * 4 * 60:
            alert(f"Acquisition STALLED at {n}/{expected} points "
                  f"(idle {idle_min:.0f} min). Not starting reconstruction.")
            return False

        time.sleep(POLL_MINUTES * 60)


def patch_config(path, assignments):
    """Rewrite `NAME = ...` config lines in a script. Fails loudly if not found."""
    p = SCRIPT_DIR / path
    text = p.read_text(encoding="utf-8")
    for name, value in assignments.items():
        pat = re.compile(rf"^({re.escape(name)}\s*=\s*)(.*?)(\s*(?:#.*)?)$", re.M)
        m = pat.search(text)
        if not m:
            raise RuntimeError(f"{path}: config line '{name} = ...' not found")
        if m.group(2).strip() == value:
            continue
        text = pat.sub(lambda mm: f"{mm.group(1)}{value}{mm.group(3)}", text, count=1)
        log(f"  {path}: {name} = {m.group(2).strip()}  ->  {value}")
    p.write_text(text, encoding="utf-8")


def patch_all(grid_dir, output_dir):
    log("Patching pipeline configuration")
    patch_config("batch_reconstruction_grid.py", {
        "GRID_DIR":  f'r"{grid_dir}"',
        "POS_SPLIT": str(POS_SPLIT),
        "Z_INDICES": repr(RECON_Z_INDICES),
    })
    patch_config("parallel_calibrate.py", {
        "GRID_DIR":  f'Path(r"{output_dir}")',
        "Z_INDEX":   str(Z_INDEX),
        "POS_SPLIT": str(POS_SPLIT),
    })
    patch_config("scheduled_recon_and_calibrate.py", {
        "GRID_DIR":   f'Path(r"{grid_dir}")',
        "OUTPUT_DIR": f'Path(r"{output_dir}")',
        "Z_INDEX":    str(Z_INDEX),
        "POS_SPLIT":  str(POS_SPLIT),
        "SKIP_RECON": "False",
    })


def verify_output(output_dir, pos_list):
    """Every point needs N_Z phase tifs; every non-BG Pos needs a calibration json.

    Pos0 is the BG: reconstruction writes output_phase_raw (and the crop_after
    variant) for it and never output_phase, so it is checked against that.
    """
    log(f"Verifying {output_dir}")
    n_z_expected = N_Z if RECON_Z_INDICES is None else len(RECON_Z_INDICES)
    missing, short = [], []
    for name in point_names(pos_list):
        is_bg = name.startswith("Pos0_x")
        ph = output_dir / name / ("output_phase_raw" if is_bg else "output_phase")
        if not ph.is_dir():
            missing.append(name)
        elif len(list(ph.glob("img_*_ph_*_phase.tif"))) < n_z_expected:
            short.append(name)

    missing_cal = [f"Pos{i}" for i in pos_list if i != 0
                   and not (output_dir / f"grid_calibration_Pos{i}.json").exists()]

    log(f"  points missing output_phase : {len(missing)}")
    log(f"  points with < {n_z_expected} z slices    : {len(short)}")
    log(f"  missing grid_calibration_*  : {len(missing_cal)}")
    for lst, tag in ((missing, "missing"), (short, "short"), (missing_cal, "no-cal")):
        for name in lst[:10]:
            log(f"    {tag}: {name}")
        if len(lst) > 10:
            log(f"    ... and {len(lst)-10} more")
    return not (missing or short or missing_cal)


def delete_raw(grid_dir, pos_list):
    keep = set(pos_list)
    victims = []
    for d in grid_dir.iterdir():
        m = POINT_RE.match(d.name) if d.is_dir() else None
        if m and int(m.group(1)) in keep:
            victims.append(d)

    drive = grid_dir.anchor
    before = free_gb(drive)
    log(f"Deleting {len(victims)} raw point folders under {grid_dir}")
    for i, d in enumerate(victims, 1):
        shutil.rmtree(d)
        if i % 1000 == 0:
            log(f"  deleted {i}/{len(victims)}  {drive}free={free_gb(drive):.0f}GB")
    after = free_gb(drive)
    log(f"{drive} free {before:.0f} GB -> {after:.0f} GB (+{after-before:.0f} GB)")
    return after


def run_batch(batch, dry_run):
    name = batch["name"]
    grid_dir, output_dir = batch["grid_dir"], batch["output_dir"]
    n_points = len(batch["pos_list"]) * N_POINTS * N_POINTS

    # pos_list is not always contiguous (batch 2 is Pos0 + Pos72..Pos102).
    runs, pl = [], batch["pos_list"]
    for p in pl:
        if runs and p == runs[-1][1] + 1:
            runs[-1][1] = p
        else:
            runs.append([p, p])
    spec = ",".join(f"Pos{a}" if a == b else f"Pos{a}-{b}" for a, b in runs)

    log("=" * 70)
    log(f"BATCH {name}: {spec}  ({len(pl)} Pos, {n_points} points)")
    log(f"  raw   : {grid_dir}")
    log(f"  output: {output_dir}")
    log("=" * 70)

    if not grid_dir.is_dir():
        alert(f"batch {name}: raw dir missing ({grid_dir}). Skipped.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    patch_all(grid_dir, output_dir)
    if dry_run:
        log(f"--dry-run: not running batch {name}")
        return True

    t0 = time.time()
    rc = subprocess.run(
        [PYTHON, str(SCRIPT_DIR / "scheduled_recon_and_calibrate.py")],
        cwd=str(SCRIPT_DIR),
    ).returncode
    log(f"batch {name}: pipeline exit {rc} after {(time.time()-t0)/3600:.2f} h")
    if rc != 0:
        alert(f"batch {name}: pipeline FAILED (exit {rc}). Raw data kept.")
        return False

    if not verify_output(output_dir, batch["pos_list"]):
        alert(f"batch {name}: verification FAILED. Raw data kept; inspect before deleting.")
        return False

    log(f"batch {name}: verification passed")
    if batch["delete_raw"]:
        after = delete_raw(grid_dir, batch["pos_list"])
        alert(f"batch {name} done; raw deleted, {grid_dir.anchor} now {after:.0f} GB free")
    else:
        alert(f"batch {name} done; raw kept at {grid_dir}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="patch config and report, but never reconstruct or delete")
    ap.add_argument("--skip-wait", action="store_true",
                    help="assume the acquisition is already finished")
    ap.add_argument("--only", choices=[b["name"] for b in BATCHES],
                    help="run a single batch")
    args = ap.parse_args()

    log("=" * 70)
    log(f"watch_grid_then_recon_260819  (POS_SPLIT={POS_SPLIT}, Z_INDEX={Z_INDEX})")
    log("=" * 70)

    if not args.skip_wait and not wait_for_acquisition():
        sys.exit(2)

    if free_gb("E:") < MIN_FREE_GB_E:
        alert(f"E: has only {free_gb('E:'):.0f} GB free (< {MIN_FREE_GB_E}). Aborting.")
        sys.exit(3)

    batches = [b for b in BATCHES if args.only is None or b["name"] == args.only]
    results = {b["name"]: run_batch(b, args.dry_run) for b in batches}

    log("=" * 70)
    for name, ok in results.items():
        log(f"batch {name}: {'OK' if ok else 'FAILED'}")
    log("=" * 70)
    sys.exit(0 if all(results.values()) else 4)


if __name__ == "__main__":
    main()
