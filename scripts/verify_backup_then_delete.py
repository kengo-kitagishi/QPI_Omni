"""
verify_backup_then_delete.py

Delete a data tree only after proving an external copy is complete.

Verification is robocopy in list-only mode (/L): it walks both trees and
reports what it *would* copy. Zero files to copy and zero failures means every
source file exists at the destination with the same size and timestamp, so the
source can go. A single missing or newer file aborts the deletion.

The robocopy log alone is not proof -- it records what one run did, not what
the tree looks like now.

Run:
    python verify_backup_then_delete.py --src "D:\AquisitionData\Kitagishi\260810" \
                                        --dst "F:\260810"
    python verify_backup_then_delete.py --src ... --dst ... --verify-only
"""
import argparse
import re
import shutil
import subprocess
import tempfile
import sys
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def free_gb(p):
    return shutil.disk_usage(str(p)).free / 1e9


def robocopy_diff(src, dst):
    """(files_to_copy, bytes_to_copy, failed) that /L reports for src -> dst."""
    cmd = ["robocopy", str(src), str(dst), "/L", "/S", "/E", "/BYTES",
           "/NJH", "/NFL", "/NDL", "/NC", "/NP", "/R:0", "/W:0"]
    out = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    # The summary table is localised (this machine prints it in Japanese), so
    # rows are identified by shape, not by label: the first three lines holding
    # five or more integers are Dirs, Files and Bytes, in that order.
    rows = []
    for line in out.splitlines():
        if ":" not in line:
            continue
        nums = [int(x) for x in line.split(":", 1)[1].split() if x.isdigit()]
        if len(nums) >= 5:
            rows.append(nums)
    if len(rows) < 3:
        raise RuntimeError(f"could not parse robocopy output:\n{out[-2000:]}")
    stats = dict(zip(("Dirs", "Files", "Bytes"), rows[:3]))
    return stats


def fast_rmtree(path):
    """Delete a tree of millions of small files.

    shutil.rmtree is one unlink per syscall on a single thread: on the 2.4 M
    files of a session tree it managed 7 of 311 Pos directories in 13 minutes.
    robocopy mirroring an empty directory over the target does the same work
    across several threads, then the empty skeleton goes in one call.
    """
    with tempfile.TemporaryDirectory() as empty:
        subprocess.run(["robocopy", empty, str(path), "/MIR", "/MT:8",
                        "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/R:0", "/W:0"],
                       capture_output=True)
    shutil.rmtree(path, ignore_errors=True)
    if Path(path).exists():
        raise RuntimeError(f"{path} still exists after deletion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    for p in (src, dst):
        if not p.is_dir():
            log(f"missing: {p}")
            sys.exit(2)

    log(f"verifying {src}  ->  {dst}")
    log("  (robocopy /L walks both trees; this takes a while on USB)")
    stats = robocopy_diff(src, dst)
    for k in ("Dirs", "Files", "Bytes"):
        if k in stats:
            t, c, s, mm, f = stats[k][:5]
            unit = " GB" if k == "Bytes" else ""
            fmt = (lambda v: f"{v/1e9:.1f}") if k == "Bytes" else (lambda v: str(v))
            log(f"  {k:<6} total={fmt(t)}{unit} to-copy={fmt(c)}{unit} "
                f"same={fmt(s)}{unit} mismatch={fmt(mm)} FAILED={fmt(f)}")

    to_copy = stats["Files"][1]
    mismatch = stats["Files"][3]
    failed = stats["Files"][4]
    if to_copy or mismatch or failed:
        log("")
        log(f"NOT a complete copy: {to_copy} files would be copied, "
            f"{mismatch} mismatched, {failed} failed. Source kept.")
        sys.exit(3)

    log("")
    log(f"verified: every file under {src} exists at {dst}, same size and time")
    if args.verify_only:
        return

    before = free_gb(src.anchor)
    log(f"deleting {src}")
    fast_rmtree(src)
    after = free_gb(src.anchor)
    log(f"{src.anchor} free {before:.0f} GB -> {after:.0f} GB (+{after-before:.0f} GB)")


if __name__ == "__main__":
    main()
