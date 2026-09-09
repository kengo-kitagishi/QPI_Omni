"""
watch_grid_then_recon_260908.py

Driver for the 260908 grid session.

  raw     C:\260908\ye_grid_0p05_2
  output  E:\260908\ye_grid_0p05_2

Waits for the acquisition to finish, then runs reconstruction, channel
detection and grid calibration. All 11 z planes are reconstructed.

The raw on C: is deleted once the output verifies, so every z plane is
reconstructed first -- a focus-only recon would discard the rest permanently.

How many Pos this acquisition will reach is not known up front: C: cannot hold
all 99 Pos (166 MB/point x 121 x 99 = ~2.0 TB), so the Pos list is discovered
from disk instead of being declared. Only Pos with all 121 points are
processed; a partially written trailing Pos is left alone (never deleted).

Run:
    python watch_grid_then_recon_260908.py
    python watch_grid_then_recon_260908.py --dry-run    # patch + report only
    python watch_grid_then_recon_260908.py --skip-wait  # acquisition already done
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
SESSION_DIR = Path(r"C:\260908")

# The acquisition this script waits for before doing anything.
WATCH_DIR = Path(r"C:\260908\ye_grid_0p05_3")
OUTPUT_DIR = Path(r"E:\260908\ye_grid_0p05_3")

# Pos list is discovered from disk at run time (see discover_complete_pos).
BATCHES = [
    {
        "name": "b1",
        "grid_dir": WATCH_DIR,
        "output_dir": OUTPUT_DIR,
        "pos_list": None,                        # filled in by main()
        "delete_raw": True,                      # frees C: after verification
    },
]

GRID_HALF     = 5       # 11 x 11 = 121 points per Pos
N_Z           = 11      # z-slices per point
# z plane used for channel detection and calibration. The focus plane for this
# session has not been measured yet, so this is the nominal middle of the stack
# (index 5 = 0.0 um for a -2.0..+2.0 um stack). Every z is reconstructed, so a
# different plane costs only a re-run of steps 2-3 (~100 s for 100 Pos): delete
# the grid_calibration_*.json and channel_rois.json, set Z_INDEX, re-run
# scheduled_recon_and_calibrate.py with SKIP_RECON=True.
Z_INDEX       = 5
POS_SPLIT     = 51      # Pos < 51 -> crop_before, Pos >= 51 -> crop_after
# z planes to reconstruct. None = all N_Z. The raw is deleted after verification,
# so every plane must be reconstructed. ~23 MB/point.
RECON_Z_INDICES = None

IDLE_MINUTES  = 30      # quiet time that ends the wait. Longer than the 260906
                        # value of 15: completion here is idle-only (no expected
                        # point count), so a pause must not trigger recon early.
POLL_MINUTES  = 10      # every poll walks the tree on a HDD the
                        # acquisition is writing to; keep it infrequent
MIN_FREE_GB_OUT = 200   # recon output is ~23 MB/point
MIN_FREE_GB_RAW = 60    # alert when C: gets this low during acquisition

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


# Point folders already confirmed to hold images. C: is a HDD and the
# acquisition is writing to it: opening all ~12000 point folders every poll
# competes with Micro-Manager's saving, so each folder is inspected once.
_CONFIRMED = {}


def scan_points(grid_dir):
    """Point folders that hold raw images, grouped by Pos, plus newest mtime.

    Empty point folders are ignored: an interrupted reconstruction can leave
    some behind, and they would otherwise be counted as acquired.
    """
    by_pos, newest = {}, 0.0
    for d in grid_dir.iterdir():
        m = POINT_RE.match(d.name) if d.is_dir() else None
        if not m:
            continue
        mt = _CONFIRMED.get(d.name)
        if mt is None:
            try:
                if not any(f.name.startswith("img_") for f in d.iterdir()):
                    continue
                mt = d.stat().st_mtime
            except OSError:
                continue
            _CONFIRMED[d.name] = mt
        by_pos.setdefault(int(m.group(1)), set()).add((m.group(2), m.group(3)))
        if mt > newest:
            newest = mt
    return by_pos, newest


def split_complete(by_pos):
    """(complete Pos, partial Pos) -- complete means all 121 points present."""
    full = N_POINTS * N_POINTS
    complete = sorted(pos for pos, pts in by_pos.items() if len(pts) == full)
    partial = sorted(pos for pos, pts in by_pos.items() if len(pts) != full)
    return complete, partial


def wait_for_acquisition():
    """Wait until the acquisition stops writing.

    The Pos count is not declared for this session (C: cannot hold all 99 Pos),
    so completion is idle-based: no new point for IDLE_MINUTES with at least
    one complete Pos on disk.
    """
    log(f"Waiting for acquisition in {WATCH_DIR} (idle >= {IDLE_MINUTES} min "
        f"ends the wait; Pos count is discovered from disk)")

    low_space_warned = False
    last_n = -1

    while True:
        if not WATCH_DIR.is_dir():
            log(f"  {WATCH_DIR} does not exist yet ...")
            time.sleep(POLL_MINUTES * 60)
            continue

        by_pos, newest = scan_points(WATCH_DIR)
        n = sum(len(pts) for pts in by_pos.values())
        complete, partial = split_complete(by_pos)
        idle_min = (time.time() - newest) / 60.0 if newest else 0.0
        c_free = free_gb(WATCH_DIR.anchor)

        if n != last_n:
            log(f"  {n} points  ({len(complete)} complete Pos"
                + (f", partial {partial}" if partial else "")
                + f")  idle={idle_min:.0f}min  C:free={c_free:.0f}GB")
            last_n = n

        if c_free < MIN_FREE_GB_RAW and not low_space_warned:
            alert(f"{WATCH_DIR.anchor} down to {c_free:.0f} GB free -- the "
                  f"acquisition will stop mid-Pos soon. Stop it at a Pos boundary.")
            low_space_warned = True

        if complete and idle_min >= IDLE_MINUTES:
            log(f"Acquisition idle {idle_min:.0f} min: {n} points, "
                f"{len(complete)} complete Pos")
            return True

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
        to_delete = batch["pos_list"]
        if batch.get("keep_bg_raw"):
            to_delete = [pos for pos in to_delete if pos != 0]
            log("  keeping Pos0 raw (BG for the remaining chunks)")
        after = delete_raw(grid_dir, to_delete)
        alert(f"batch {name} done; raw deleted, {grid_dir.anchor} now {after:.0f} GB free")
    else:
        alert(f"batch {name} done; raw kept at {grid_dir}")
    return True


def bg_already_reconstructed():
    """True when Pos0's BG phase is in the output tree.

    Reconstruction subtracts the BG from output_phase_raw, not from the raw
    hologram, so a batch whose Pos0 raw was already deleted still reconstructs
    as long as that BG survives.
    """
    n_z = N_Z if RECON_Z_INDICES is None else len(RECON_Z_INDICES)
    for xi in range(-GRID_HALF, GRID_HALF + 1):
        for yi in range(-GRID_HALF, GRID_HALF + 1):
            d = OUTPUT_DIR / f"Pos0_x{xi:+d}_y{yi:+d}" / "output_phase_raw"
            if not d.is_dir() or len(list(d.glob("img_*_ph_*_phase.tif"))) < n_z:
                return False
    return True


def parse_pos_spec(spec):
    """'0-17', '0-17,40', '5' -> sorted list of Pos numbers."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="patch config and report, but never reconstruct or delete")
    ap.add_argument("--skip-wait", action="store_true",
                    help="assume the acquisition is already finished")
    ap.add_argument("--only", choices=[b["name"] for b in BATCHES],
                    help="run a single batch")
    ap.add_argument("--pos", metavar="SPEC",
                    help="process only these Pos (e.g. 0-17). Implies --skip-wait: "
                         "use it to reconstruct and free C: while the acquisition "
                         "is still running. Pos0 is always added -- it is the BG.")
    ap.add_argument("--delete-bg", action="store_true",
                    help="also delete Pos0 raw. Only for the final chunk: every "
                         "later chunk needs Pos0 to reconstruct the BG.")
    args = ap.parse_args()

    log("=" * 70)
    log(f"watch_grid_then_recon_260908  (POS_SPLIT={POS_SPLIT}, Z_INDEX={Z_INDEX})")
    log("=" * 70)

    if not (args.skip_wait or args.pos) and not wait_for_acquisition():
        sys.exit(2)

    by_pos, _ = scan_points(WATCH_DIR)
    complete, partial = split_complete(by_pos)
    if not complete:
        alert(f"no complete Pos in {WATCH_DIR}. Nothing to reconstruct.")
        sys.exit(5)
    if 0 not in complete and not bg_already_reconstructed():
        alert("Pos0 (the BG) has neither raw nor a reconstruction in the output. "
              "Reconstruction needs one of them. Aborting.")
        sys.exit(6)
    if 0 not in complete:
        log("Pos0 raw is gone, but its reconstructed BG is in the output tree; "
            "reconstruction reads the BG from there.")
    if args.pos:
        wanted = set(parse_pos_spec(args.pos)) | {0}
        missing = sorted(pos for pos in wanted if pos not in complete)
        complete = [pos for pos in complete if pos in wanted]
        if missing:
            log(f"Requested but not complete on disk, skipped: {missing}")
        if len(complete) < 2:
            alert(f"--pos {args.pos} selects nothing beyond the BG. Nothing to do.")
            sys.exit(7)

    for b in BATCHES:
        b["pos_list"] = complete
        # Pos0 is the BG: reconstructing any later chunk needs its raw, so a
        # partial run keeps it. The acquisition is still writing to this disk,
        # which is the whole reason for running a chunk early.
        b["keep_bg_raw"] = bool(args.pos) and not args.delete_bg
    log(f"Pos to process ({len(complete)}): Pos{complete[0]}..Pos{complete[-1]}")
    if partial:
        log(f"Partial Pos left untouched (not reconstructed, not deleted): {partial}")

    out_drive = BATCHES[0]["output_dir"].anchor
    if free_gb(out_drive) < MIN_FREE_GB_OUT:
        alert(f"{out_drive} has only {free_gb(out_drive):.0f} GB free "
              f"(< {MIN_FREE_GB_OUT}). Aborting.")
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
