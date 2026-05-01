"""
Tail train_loss.log and copy the model file to <path>_e<NNNN> on every save event.

Usage:
  python checkpoint_watcher.py --log <path> --model <path>
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

EPOCH_RE = re.compile(r"Train epoch:\s*(\d+)")
SAVE_RE = re.compile(r"saving network parameters to (.+)$")


def follow(path: Path, poll_sec: float = 1.0):
    pos = path.stat().st_size if path.exists() else 0
    while True:
        if not path.exists():
            time.sleep(poll_sec)
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                f.seek(pos)
                for line in f:
                    yield line.rstrip("\n")
                pos = f.tell()
        except OSError:
            pass
        time.sleep(poll_sec)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path,
                   help="Model file to backup (the one being overwritten)")
    args = p.parse_args()

    print(f"watching log: {args.log}", flush=True)
    print(f"backing up:   {args.model}", flush=True)

    last_epoch = None
    for line in follow(args.log):
        m = EPOCH_RE.search(line)
        if m:
            last_epoch = int(m.group(1))
            continue
        m = SAVE_RE.search(line)
        if m and last_epoch is not None:
            src = args.model
            dst = src.with_name(src.name + f"_e{last_epoch:04d}")
            if dst.exists():
                continue
            try:
                time.sleep(0.5)
                shutil.copy2(src, dst)
                print(f"backup: e{last_epoch:04d} -> {dst.name}", flush=True)
            except Exception as e:
                print(f"!! backup failed at e{last_epoch}: {e!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
