"""negate_phase_tree.py -- flip the sign of reconstructed phase in place.

Selecting the conjugate off-axis order reconstructs the conjugate field, so the
unwrapped phase comes out exactly negated. Verified on 260906 optics against a
vis_2 hologram: reconstructing with (1580,383) and with (468,1665) gave phases
agreeing to 4.6e-14 rad after a sign flip. Background subtraction is linear and
MEAN_REGION is disabled, so `output_phase = phase - phase_bg` negates with it.

That makes an off-axis-order mistake recoverable without re-acquiring: negate
every phase TIF, then redo channel detection and grid calibration (those read
the phase and look for dark bands, so they must be recomputed, not negated).

Applying this twice would restore the wrong sign, so each point directory gets a
marker file naming the run that negated it; a directory carrying the marker is
skipped. Files are replaced atomically (temp + os.replace), so an interrupted
run leaves no half-written TIF and can simply be re-run.

Usage
-----
    python scripts/negate_phase_tree.py --root "<grid output dir>" --dry-run
    python scripts/negate_phase_tree.py --root "<grid output dir>"
    python scripts/negate_phase_tree.py --root "<grid output dir>" --verify
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

MARKER = ".phase_negated"
SUBDIRS = ("output_phase", "output_phase_raw", "output_phase_raw_crop_after")
POINT_RE = re.compile(r"^Pos(\d+)_x([+-]\d+)_y([+-]\d+)$")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="grid reconstruction output directory")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="report the sign of channel features instead of modifying anything")
    return p.parse_args()


def point_dirs(root: Path):
    return sorted((d for d in root.iterdir() if d.is_dir() and POINT_RE.match(d.name)),
                  key=lambda d: (int(POINT_RE.match(d.name).group(1)), d.name))


def negate_dir(pdir: Path, dry: bool) -> tuple[int, int]:
    """Negate every phase TIF under one point directory. Returns (files, bytes)."""
    n = nb = 0
    for sub in SUBDIRS:
        d = pdir / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.tif")):
            nb += f.stat().st_size
            n += 1
            if dry:
                continue
            a = tifffile.imread(str(f))
            tmp = f.with_name(f.name + ".tmp")
            tifffile.imwrite(str(tmp), (-a).astype(a.dtype))
            os.replace(tmp, f)          # atomic on the same volume
    if not dry:
        (pdir / MARKER).write_text(
            f"phase negated (conjugate off-axis order correction)\n", encoding="utf-8")
    return n, nb


def verify(root: Path):
    """Report whether channel features read as dips or peaks, per the row profile."""
    print(f"{'point':>22} {'mean':>8} {'p1':>8} {'p99':>8}  features")
    shown = 0
    for pdir in point_dirs(root):
        if not pdir.name.endswith("_x+0_y+0"):
            continue
        d = pdir / "output_phase"
        fs = sorted(d.glob("*.tif")) if d.is_dir() else []
        if not fs:
            continue
        a = tifffile.imread(str(fs[len(fs) // 2])).astype(np.float64)
        prof = a.mean(axis=1)
        dev = prof - np.median(prof)
        kind = "NEGATIVE dips" if abs(dev.min()) > abs(dev.max()) else "POSITIVE peaks"
        print(f"{pdir.name:>22} {a.mean():+8.3f} {np.percentile(a,1):+8.3f} "
              f"{np.percentile(a,99):+8.3f}  {kind}")
        shown += 1
        if shown >= 8:
            break


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    if args.verify:
        verify(root)
        return

    dirs = point_dirs(root)
    todo = [d for d in dirs if not (d / MARKER).exists()]
    done = len(dirs) - len(todo)
    print(f"root      : {root}")
    print(f"point dirs: {len(dirs)}   already negated: {done}   to process: {len(todo)}")
    if not todo:
        print("nothing to do (every point directory carries the marker)")
        return

    if args.dry_run:
        n, nb = 0, 0
        for d in todo[:50]:
            a, b = negate_dir(d, dry=True)
            n += a; nb += b
        est_files = int(n / len(todo[:50]) * len(todo))
        est_gb = nb / len(todo[:50]) * len(todo) / 1e9
        print(f"--dry-run: ~{est_files} TIFs, ~{est_gb:.0f} GB would be rewritten")
        print(f"           (sampled {len(todo[:50])} directories)")
        return

    t0 = time.time()
    files = 0
    nbytes = 0
    for i, d in enumerate(todo, 1):
        n, nb = negate_dir(d, dry=False)
        files += n
        nbytes += nb
        if i % 200 == 0 or i == len(todo):
            el = time.time() - t0
            rate = i / el
            eta = (len(todo) - i) / rate / 60 if rate else float("nan")
            print(f"  {i}/{len(todo)} dirs  {files} TIFs  {nbytes/1e9:.1f} GB  "
                  f"{el/60:.1f} min elapsed  ETA {eta:.1f} min", flush=True)
    print(f"\ndone: {files} TIFs, {nbytes/1e9:.1f} GB, {(time.time()-t0)/60:.1f} min")
    print("Next: delete grid_calibration_*.json and channels/channel_rois.json, "
          "then re-run scheduled_recon_and_calibrate.py with SKIP_RECON=True.")


if __name__ == "__main__":
    main()
