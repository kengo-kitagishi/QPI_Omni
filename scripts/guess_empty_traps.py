"""Guess the empty (cell-free) traps of a timelapse frame for the whitelist review.

A trap counts as empty when the 99th percentile of its crop_sub phase is below THRESH rad. At T=0 of 261108 run 3
the traps split into two groups: empty around 0.3-0.4 rad and with cells around 1.0-2.0 rad, sparse in between;
0.6 rad keeps traps where a cell only touches the edge out of the guess.
The last trap of the Pos at or after --pos-split sits at the edge of the field, where the quadrant subtraction goes
wrong, so it is never guessed (left to the user). The output is only a starting point: the user reviews every tile.

    python guess_empty_traps.py --raw-root D:/.../online_crop_sub_YE_5min_31d_3 --frame 1 --pos-split 60 --out guess.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from channel_contact_sheet import pick_frame  # noqa: E402

THRESH = 0.6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--channel-rel", default="output_phase/channels/crop_sub_rawraw/z000")
    ap.add_argument("--phase-glob", default="img_*_ph_000.tif")
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--pos-split", type=int, required=True)
    ap.add_argument("--thresh", type=float, default=THRESH)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = Path(a.raw_root)
    picks, n_all, n_skip_edge = [], 0, 0
    for pos_dir in sorted(root.glob("Pos*"), key=lambda p: int(p.name[3:])):
        pos = int(pos_dir.name[3:])
        if pos == 0:
            continue
        chs = sorted((d for d in (pos_dir / a.channel_rel).glob("ch*") if d.is_dir()), key=lambda p: int(p.name[2:]))
        for i, ch in enumerate(chs):
            f = pick_frame(sorted(ch.glob(a.phase_glob)), a.frame)
            if f is None:
                continue
            n_all += 1
            if pos >= a.pos_split and i == len(chs) - 1:
                n_skip_edge += 1
                continue
            if float(np.percentile(tifffile.imread(str(f)), 99)) < a.thresh:
                picks.append(f"Pos{pos} {ch.name}")
    Path(a.out).write_text("\n".join(picks) + "\n", encoding="utf-8")
    print(f"{len(picks)} of {n_all} traps guessed empty (p99 < {a.thresh} rad); "
          f"{n_skip_edge} edge traps of Pos>={a.pos_split} left to the user -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
