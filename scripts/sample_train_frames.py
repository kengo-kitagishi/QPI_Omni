"""Randomly sample phase .tif frames from multiple source dirs into a training pool.

Frame distribution over 300 images:
  0-570      : 70 % (210)
  575-1439   : 10 % (30)
  1440+      : 20 % (60)
"""
from __future__ import annotations

import random
import re
import shutil
from pathlib import Path

SOURCES = [
    r"F:\260405\ph_260405\Pos16\output_phase\channels\crop_sub_rawraw\ch04",
    r"F:\260405\ph_260405\Pos9\output_phase\channels\crop_sub_rawraw\ch09",
    r"F:\260405\ph_260405\Pos9\output_phase\channels\crop_sub_rawraw\ch00",
    r"F:\260405\ph_260405\Pos9\output_phase\channels\crop_sub_rawraw\ch01",
    r"F:\260405\ph_260405\Pos9\output_phase\channels\crop_sub_rawraw\ch02",
    r"F:\260405\ph_260405\Pos9\output_phase\channels\crop_sub_rawraw\ch04",
    r"F:\260405\ph_260405\Pos9\output_phase\channels\crop_sub_rawraw\ch08",
]
DEST = Path(r"C:\Users\QPI\Desktop\train")

BUCKETS = [
    ("early",  range(0, 571),      210),   # 0-570
    ("mid",    range(575, 1440),    30),   # 575-1439
    ("late",   range(1440, 10_000), 60),   # 1440+
]

FNAME_RE = re.compile(r"img_(\d+)_ph_\d+_phase\.tif$", re.IGNORECASE)
SEED = 42


def tag_from_dir(d: Path) -> str:
    # .../Pos9/.../crop_sub_rawraw/ch04 -> Pos9_ch04
    parts = d.parts
    pos = next(p for p in parts if p.lower().startswith("pos"))
    ch = d.name  # e.g. "ch04"
    return f"{pos}_{ch}"


def index_sources() -> list[tuple[Path, int, str, str]]:
    """Return list of (src_path, frame_idx, tag, filename)."""
    pool = []
    for s in SOURCES:
        d = Path(s)
        tag = tag_from_dir(d)
        for f in d.iterdir():
            if not f.is_file():
                continue
            m = FNAME_RE.match(f.name)
            if not m:
                continue
            pool.append((f, int(m.group(1)), tag, f.name))
    return pool


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    all_items = index_sources()
    print(f"Indexed {len(all_items)} candidate files from {len(SOURCES)} sources")

    selected: list[tuple[Path, int, str, str]] = []
    for name, frames, n in BUCKETS:
        bucket = [x for x in all_items if x[1] in frames]
        if len(bucket) < n:
            raise RuntimeError(f"bucket '{name}' has {len(bucket)} < {n} requested")
        picks = rng.sample(bucket, n)
        selected.extend(picks)
        print(f"  {name:5s} frames={frames.start}-{frames.stop-1:>5}  "
              f"pool={len(bucket):5d}  picked={n}")

    # Copy with a unique prefix so same-named files from different dirs don't collide.
    for src, frame, tag, fname in selected:
        dst = DEST / f"{tag}_{fname}"
        shutil.copy2(src, dst)

    print(f"\nCopied {len(selected)} files -> {DEST}")


if __name__ == "__main__":
    main()
