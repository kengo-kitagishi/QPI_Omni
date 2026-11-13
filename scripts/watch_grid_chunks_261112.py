"""
watch_grid_chunks_261112.py

One chunk of the 261112 run-2 grid (19 x 19 points per Pos at 0.03 um, rows all in the
same direction, one z), acquired by grid_chunks_acquire_261112.bsh into E:\\261112\\grid_2per_cN.
Called by that BeanShell after each chunk (blocking):

    python watch_grid_chunks_261112.py --chunk-dir E:\\261112\\grid_2per_c1 --pos 1-25
    python watch_grid_chunks_261112.py --chunk-dir E:\\261112\\grid_2per_c5 --pos 101-124 --final

  1. every point of Pos0 + the chunk's Pos must be on disk (else exit 2, nothing touched)
  2. scheduled_recon_and_calibrate.py: recon (raw E: -> D:\\...\\recon_0p03um_19x19), channel
     detect, calibration (calibrate_grid_positions.ESTIMATOR = gaussian2d). Outputs of earlier
     chunks are skipped, Pos0's BG is reconstructed once (chunk 1) and reused.
  3. verify: every point has its phase tif, every Pos its grid_calibration json
  4. delete the chunk's raw folder (E: NVMe) so the next chunk fits
  5. --final: prepare_drift_session.py (drift_session_261112_3), then beep

Exit code 0 only when everything above succeeded; the BeanShell stops on anything else.
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
SESSION_DIR = Path(r"E:\261112")
OUTPUT_DIR = Path(r"D:\AquisitionData\Kitagishi\261112\recon_0p03um_25x25")
GRID_HALF = 12         # 25 x 25 = 625 points per Pos (2D stride order)
N_Z = 1
Z_INDEX = 0
POS_SPLIT = 60         # Pos < 60 crop_before, Pos >= 60 crop_after (261112 timelapse.pos rows)
MIN_FREE_GB_OUT = 60   # D: recon output ~0.9 GB/Pos
BEEP_ENABLED = True
LOG_PATH = SESSION_DIR / "grid_chunks_0p03um_25x25.log"
ALERT_PATH = SESSION_DIR / "grid_chunks_0p03um_25x25.ALERT.txt"
# ============================================================

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
POINT_RE = re.compile(r"^Pos(\d+)_x([+-]\d+)_y([+-]\d+)$")
N_POINTS = 2 * GRID_HALF + 1


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def alert(msg):
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


def parse_pos_spec(spec):
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


def point_names(pos_list):
    for pos in pos_list:
        for xi in range(-GRID_HALF, GRID_HALF + 1):
            for yi in range(-GRID_HALF, GRID_HALF + 1):
                yield f"Pos{pos}_x{xi:+d}_y{yi:+d}"


def missing_raw(chunk_dir, pos_list):
    miss = []
    for name in point_names(pos_list):
        d = chunk_dir / name
        try:
            if not d.is_dir() or not any(f.name.startswith("img_") for f in d.iterdir()):
                miss.append(name)
        except OSError:
            miss.append(name)
    return miss


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
        "GRID_DIR": f'r"{grid_dir}"', "POS_SPLIT": str(POS_SPLIT), "Z_INDICES": "None"})
    patch_config("parallel_calibrate.py", {
        "GRID_DIR": f'Path(r"{output_dir}")', "Z_INDEX": str(Z_INDEX), "POS_SPLIT": str(POS_SPLIT)})
    patch_config("scheduled_recon_and_calibrate.py", {
        "GRID_DIR": f'Path(r"{grid_dir}")', "OUTPUT_DIR": f'Path(r"{output_dir}")',
        "Z_INDEX": str(Z_INDEX), "POS_SPLIT": str(POS_SPLIT), "SKIP_RECON": "False"})


def verify_output(output_dir, pos_list):
    log(f"Verifying {output_dir} for Pos {pos_list[0]}..{pos_list[-1]}")
    missing, short = [], []
    for name in point_names(pos_list):
        is_bg = name.startswith("Pos0_x")
        ph = output_dir / name / ("output_phase_raw" if is_bg else "output_phase")
        if not ph.is_dir():
            missing.append(name)
        elif len(list(ph.glob("img_*_ph_*_phase.tif"))) < N_Z:
            short.append(name)
    missing_cal = [f"Pos{i}" for i in pos_list if i != 0
                   and not (output_dir / f"grid_calibration_Pos{i}.json").exists()]
    log(f"  points missing output_phase : {len(missing)}")
    log(f"  points with < {N_Z} z slices    : {len(short)}")
    log(f"  missing grid_calibration_*  : {len(missing_cal)}")
    for lst, tag in ((missing, "missing"), (short, "short"), (missing_cal, "no-cal")):
        for name in lst[:10]:
            log(f"    {tag}: {name}")
        if len(lst) > 10:
            log(f"    ... and {len(lst)-10} more")
    return not (missing or short or missing_cal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True)
    ap.add_argument("--pos", required=True, help="Pos of this chunk, e.g. 1-25 (Pos0 is always added)")
    ap.add_argument("--final", action="store_true", help="after this chunk run prepare_drift_session.py")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--replace", action="store_true",
                    help="re-shoot: delete the existing recon output, channel_rois and grid_calibration "
                         "of this chunk's Pos (never Pos0) before the pipeline, so SKIP_IF_EXISTS does not keep them")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    chunk_dir = Path(args.chunk_dir)
    pos_list = sorted(set(parse_pos_spec(args.pos)) | {0})

    log("=" * 70)
    log(f"watch_grid_chunks_261112  chunk {chunk_dir}  Pos {args.pos}  "
        f"({N_POINTS}x{N_POINTS} points/Pos, POS_SPLIT={POS_SPLIT}, Z_INDEX={Z_INDEX}){'  FINAL' if args.final else ''}")
    log("=" * 70)

    if not chunk_dir.is_dir():
        alert(f"chunk dir missing: {chunk_dir}")
        sys.exit(2)
    miss = missing_raw(chunk_dir, pos_list)
    if miss:
        alert(f"{len(miss)} of {len(pos_list) * N_POINTS * N_POINTS} points have no image in {chunk_dir} "
              f"(e.g. {miss[:5]}). Nothing done.")
        sys.exit(2)
    log(f"raw complete: {len(pos_list)} Pos x {N_POINTS * N_POINTS} points  (E: free {free_gb(chunk_dir.anchor):.0f} GB)")

    if free_gb(OUTPUT_DIR.anchor if OUTPUT_DIR.exists() else OUTPUT_DIR.anchor) < MIN_FREE_GB_OUT:
        alert(f"{OUTPUT_DIR.anchor} has < {MIN_FREE_GB_OUT} GB free. Aborting.")
        sys.exit(3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.replace and not args.dry_run:
        n_dirs = 0
        for pos in pos_list:
            if pos == 0:
                continue
            for name in point_names([pos]):
                d = OUTPUT_DIR / name
                if d.is_dir():
                    shutil.rmtree(d)
                    n_dirs += 1
            cal = OUTPUT_DIR / f"grid_calibration_Pos{pos}.json"
            if cal.exists():
                cal.unlink()
        log(f"--replace: removed {n_dirs} old point outputs and the calibration json of Pos "
            f"{[p for p in pos_list if p != 0][0]}..{pos_list[-1]} (Pos0 kept)")
    patch_all(chunk_dir, OUTPUT_DIR)
    if args.dry_run:
        log("--dry-run: stopping before reconstruction")
        sys.exit(0)

    t0 = time.time()
    rc = subprocess.run([PYTHON, str(SCRIPT_DIR / "scheduled_recon_and_calibrate.py")],
                        cwd=str(SCRIPT_DIR)).returncode
    log(f"pipeline exit {rc} after {(time.time() - t0) / 60:.1f} min")
    if rc != 0:
        alert(f"pipeline FAILED (exit {rc}) on {chunk_dir}. Raw kept.")
        sys.exit(4)
    if not verify_output(OUTPUT_DIR, pos_list):
        alert(f"verification FAILED on {chunk_dir}. Raw kept; inspect before deleting.")
        sys.exit(5)
    log("verification passed")

    if args.keep_raw:
        log(f"--keep-raw: {chunk_dir} left in place")
    else:
        before = free_gb(chunk_dir.anchor)
        t1 = time.time()
        shutil.rmtree(chunk_dir)
        log(f"deleted {chunk_dir} in {time.time() - t1:.0f}s: {chunk_dir.anchor} free {before:.0f} -> {free_gb(chunk_dir.anchor):.0f} GB")

    if args.final:
        log("Running prepare_drift_session.py (drift_session_261112_3)")
        rc = subprocess.run([PYTHON, str(SCRIPT_DIR / "prepare_drift_session.py")],
                            cwd=str(SCRIPT_DIR)).returncode
        if rc != 0:
            alert(f"prepare_drift_session.py FAILED (exit {rc}); run it by hand")
            sys.exit(6)
        alert("grid done, calibrated, raw deleted, drift session ready: open "
              "drift_session_261112_3/realtime_drift_mda_zstack_261112_2.bsh in the MM Script Panel and press Run")
    else:
        log(f"chunk {chunk_dir.name} done")
    sys.exit(0)


if __name__ == "__main__":
    main()
