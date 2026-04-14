"""
Delete raw timelapse files img_<index>_ph_000.tif where index > threshold
under each Pos* folder of a timelapse root.

Usage:
  python scripts/delete_ph000_frames_over_index.py --root "D:\\...\\ph_260405" --dry-run
  python scripts/delete_ph000_frames_over_index.py --root "D:\\...\\ph_260405" --execute
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PATTERN = re.compile(r"^img_(\d+)_ph_000\.tif$")


def collect_targets(root: Path, threshold: int) -> list[Path]:
    out: list[Path] = []
    if not root.is_dir():
        return out
    for pos in sorted(root.glob("Pos*")):
        if not pos.is_dir():
            continue
        for f in pos.glob("img_*_ph_000.tif"):
            m = PATTERN.match(f.name)
            if m and int(m.group(1)) > threshold:
                out.append(f)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Delete img_*_ph_000.tif with index > threshold.")
    p.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\AquisitionData\Kitagishi\260405\ph_260405"),
        help="Timelapse root containing Pos* folders",
    )
    p.add_argument("--threshold", type=int, default=2400, help="Delete if frame index > this")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="List only, do not delete")
    g.add_argument("--execute", action="store_true", help="Delete matching files")
    args = p.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"ERROR: root does not exist or is not a directory: {root}", file=sys.stderr)
        return 1

    targets = collect_targets(root, args.threshold)
    nums = []
    for t in targets:
        m = PATTERN.match(t.name)
        if m:
            nums.append(int(m.group(1)))

    print(f"root: {root}")
    print(f"threshold: delete if index > {args.threshold}")
    print(f"files matching: {len(targets)}")
    if nums:
        print(f"index min: {min(nums)}  max: {max(nums)}")
    for t in targets[:12]:
        print(f"  {t}")
    if len(targets) > 12:
        print(f"  ... and {len(targets) - 12} more")

    if args.execute:
        deleted = 0
        for t in targets:
            try:
                t.unlink()
                deleted += 1
            except OSError as e:
                print(f"ERROR unlink {t}: {e}", file=sys.stderr)
        print(f"deleted: {deleted} / {len(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
