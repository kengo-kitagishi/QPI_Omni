"""
cleanup_verified_raw.py

Reclaim disk on D:/E: by removing data that cannot be lost:

  1. raw holograms whose reconstruction sits in the same point folder
     (the "delete raw after verification" step that C: does every session,
     never run for E:\260714\ye_grid_1)
  2. aborted acquisitions that were never reconstructed and never will be
  3. incomplete reconstruction output that cannot be used
  4. scratch images

Every raw deletion in (1) is verified per point folder: the point keeps its raw
unless output_phase*/ holds at least as many *_phase.tif as there are raw files.

Run:
    python cleanup_verified_raw.py --dry-run
    python cleanup_verified_raw.py
"""
import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

RAW_RE = re.compile(r"^img_\d+_ph_\d+\.tif$")

# (1) point trees where raw and reconstruction live side by side
VERIFIED_RAW_TREES = [
    r"E:\260714\ye_grid_1",
]

# (2)(3) whole trees that hold nothing reproducible or nothing usable
DEAD_TREES = [
    # Pos0/36/37 only (360 of 2783 points); the b2 pipeline ran --dry-run only
    r"D:\AquisitionData\Kitagishi\260906\grid_YE_0p05_hologram_Pos36_57_1",
    # reconstruction output, 660 of 3751 points -- unusable as a grid
    r"E:\260810\grid_b2_pos72_102",
    # empty leftovers
    r"D:\AquisitionData\Kitagishi\260906\grid_YE_0p05_2",
    r"E:\260906\grid_YE_0p05_1",
]

# (4) scratch images; the directory itself stays so the monitor still starts
SCRATCH_DIRS = [
    r"D:\AquisitionData\Kitagishi\basler_image_seq\vis_1",
]

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def free_gb(p):
    return shutil.disk_usage(str(p)).free / 1e9


def clean_verified_raw(root, dry_run):
    """Delete raw only where the point's reconstruction is present."""
    root = Path(root)
    log(f"[1] {root}")
    n_files = n_dirs = n_kept = 0
    freed = kept = 0
    for name in sorted(os.listdir(root)):
        d = root / name
        if not d.is_dir():
            continue
        raws = [f for f in os.listdir(d) if RAW_RE.match(f)]
        if not raws:
            continue
        n_phase = 0
        for sub in os.listdir(d):
            p = d / sub
            if p.is_dir() and sub.startswith("output_phase"):
                n_phase += len([f for f in os.listdir(p) if f.endswith("_phase.tif")])
        size = sum((d / f).stat().st_size for f in raws)
        if n_phase < len(raws):
            n_kept += 1
            kept += size
            continue
        if not dry_run:
            for f in raws:
                (d / f).unlink()
        n_files += len(raws)
        n_dirs += 1
        freed += size
        if n_dirs % 200 == 0 and not dry_run:
            log(f"    {n_dirs} point folders, {freed/1e9:.1f} GB, "
                f"{root.anchor}free={free_gb(root.anchor):.0f}GB")
    log(f"    {'would delete' if dry_run else 'deleted'} {n_files} raw files in "
        f"{n_dirs} point folders ({freed/1e9:.1f} GB)")
    log(f"    kept {n_kept} point folders without reconstruction ({kept/1e9:.1f} GB)")
    return freed


def tree_size(p):
    total = 0
    for dp, _dn, fn in os.walk(p):
        for f in fn:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def drop_tree(path, dry_run):
    p = Path(path)
    if not p.is_dir():
        log(f"[2] {p}: not found, skipped")
        return 0
    size = tree_size(p)
    log(f"[2] {p}: {'would delete' if dry_run else 'deleting'} tree ({size/1e9:.1f} GB)")
    if not dry_run:
        shutil.rmtree(p)
    return size


def clear_contents(path, dry_run):
    p = Path(path)
    if not p.is_dir():
        log(f"[3] {p}: not found, skipped")
        return 0
    size = tree_size(p)
    log(f"[3] {p}: {'would clear' if dry_run else 'clearing'} contents, keeping the "
        f"directories ({size/1e9:.1f} GB)")
    if not dry_run:
        for dp, _dn, fn in os.walk(p):
            for f in fn:
                os.remove(os.path.join(dp, f))
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    before = {d: free_gb(d) for d in ("D:", "E:")}
    freed = 0
    for t in VERIFIED_RAW_TREES:
        freed += clean_verified_raw(t, args.dry_run)
    for t in DEAD_TREES:
        freed += drop_tree(t, args.dry_run)
    for t in SCRATCH_DIRS:
        freed += clear_contents(t, args.dry_run)

    log("")
    log(f"total {'reclaimable' if args.dry_run else 'freed'}: {freed/1e9:.1f} GB")
    for d in ("D:", "E:"):
        log(f"  {d} free {before[d]:.0f} GB -> {free_gb(d):.0f} GB")


if __name__ == "__main__":
    main()
