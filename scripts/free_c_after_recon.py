"""
free_c_after_recon.py

Delete grid raw on C: for the Pos that are already reconstructed, so the next
acquisition can start without waiting for the whole recon to finish.

Raw is only needed by reconstruction. Channel detection and grid calibration
read the reconstructed output, so a Pos can lose its raw as soon as its output
verifies -- calibration can still run afterwards.

Waits until --min-pos Pos have complete output (121 points x N_Z phase tifs),
then deletes exactly those Pos from the raw tree. A Pos still being written is
never touched.

Run:
    python free_c_after_recon.py --min-pos 22
    python free_c_after_recon.py --min-pos 22 --dry-run
"""
import argparse
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

RAW_DIR = Path(r"C:\260908\ye_grid_0p05_2")
OUT_DIR = Path(r"D:\AquisitionData\Kitagishi\260908\ye_grid_0p05_2")
N_Z = 11
N_POINTS = 121
POLL_SECONDS = 120
LOG_PATH = Path(r"C:\260908\free_c_after_recon.log")
BEEP_ENABLED = True

POINT_RE = re.compile(r"^Pos(\d+)_x([+-]\d+)_y([+-]\d+)$")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def free_gb(p):
    return shutil.disk_usage(str(p)).free / 1e9


def complete_pos():
    """Pos whose every point holds N_Z reconstructed phase tifs."""
    counts = {}
    for name in os.listdir(OUT_DIR):
        m = POINT_RE.match(name)
        if not m:
            continue
        d = OUT_DIR / name
        if not d.is_dir():
            continue
        n = 0
        for sub in os.listdir(d):
            p = d / sub
            if p.is_dir() and sub.startswith("output_phase"):
                n += len([f for f in os.listdir(p) if f.endswith("_phase.tif")])
        st = counts.setdefault(int(m.group(1)), [0, 0])
        st[0] += 1
        if n >= N_Z:
            st[1] += 1
    return sorted(pos for pos, (total, ok) in counts.items()
                  if total == N_POINTS and ok == N_POINTS)


def delete_raw(pos_list, dry_run):
    victims = []
    for d in RAW_DIR.iterdir():
        m = POINT_RE.match(d.name) if d.is_dir() else None
        if m and int(m.group(1)) in set(pos_list):
            victims.append(d)
    before = free_gb(RAW_DIR.anchor)
    log(f"{'Would delete' if dry_run else 'Deleting'} {len(victims)} raw point "
        f"folders for {len(pos_list)} Pos under {RAW_DIR}")
    if dry_run:
        return before
    for i, d in enumerate(victims, 1):
        shutil.rmtree(d)
        if i % 500 == 0:
            log(f"  {i}/{len(victims)}  C:free={free_gb(RAW_DIR.anchor):.0f}GB")
    after = free_gb(RAW_DIR.anchor)
    log(f"C: free {before:.0f} GB -> {after:.0f} GB (+{after-before:.0f} GB)")
    return after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pos", type=int, required=True,
                    help="wait until this many Pos are fully reconstructed")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log("=" * 70)
    log(f"free_c_after_recon: waiting for {args.min_pos} complete Pos")
    log(f"  raw   : {RAW_DIR}")
    log(f"  output: {OUT_DIR}")
    log("=" * 70)

    last = -1
    while True:
        done = complete_pos()
        if len(done) != last:
            log(f"  {len(done)}/{args.min_pos} Pos reconstructed  "
                f"(Pos{','.join(str(p) for p in done[:8])}"
                f"{'...' if len(done) > 8 else ''})  C:free={free_gb('C:'):.0f}GB")
            last = len(done)
        if len(done) >= args.min_pos:
            break
        time.sleep(POLL_SECONDS)

    target = done[:args.min_pos]
    log(f"Reached {len(done)} Pos. Deleting raw for: "
        f"{','.join('Pos'+str(p) for p in target)}")
    after = delete_raw(target, args.dry_run)
    log(f"*** done. C: {after:.0f} GB free -- stop the reconstruction and start "
        f"the next acquisition ***")
    if BEEP_ENABLED and not args.dry_run:
        try:
            import winsound
            for _ in range(6):
                winsound.Beep(880, 400)
                time.sleep(0.15)
        except Exception:
            pass


if __name__ == "__main__":
    main()
