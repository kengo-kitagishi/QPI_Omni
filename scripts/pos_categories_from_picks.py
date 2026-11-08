"""Classify each Pos of a timelapse from the user's empty-trap picks (whitelist review).

    all_cells : no trap picked as empty (every trap holds cells)
    all_empty : every judgeable trap picked as empty (no cells at all)
    mixed     : some of each
Only `mixed` Pos are analysis targets (user, 2026-11-08: Pos where all traps or no trap hold cells are not analysed).
The last trap of Pos >= --pos-split (field edge) could not be judged and is not counted.

    python pos_categories_from_picks.py --picks drift_session_261108_3/whitelist_review_t1.selected.txt \
        --raw-root D:/.../online_crop_sub_YE_5min_31d_3 --pos-split 60 --out drift_session_261108_3/pos_categories.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--channel-rel", default="output_phase/channels/crop_sub_rawraw/z000")
    ap.add_argument("--pos-split", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    picked = defaultdict(set)
    for line in Path(a.picks).read_text(encoding="utf-8").splitlines():
        if len(line.split()) == 2:
            pos, ch = line.split()
            picked[int(pos[3:])].add(ch)
    rows = []
    for pos_dir in sorted(Path(a.raw_root).glob("Pos*"), key=lambda p: int(p.name[3:])):
        pos = int(pos_dir.name[3:])
        if pos == 0:
            continue
        chs = sorted((d.name for d in (pos_dir / a.channel_rel).glob("ch*") if d.is_dir()), key=lambda s: int(s[2:]))
        judgeable = chs[:-1] if pos >= a.pos_split else chs
        n_empty = len(picked[pos] & set(judgeable))
        cat = "all_cells" if n_empty == 0 else "all_empty" if n_empty == len(judgeable) else "mixed"
        rows.append({"pos": f"Pos{pos}", "n_traps": len(chs), "n_judgeable": len(judgeable), "n_empty": n_empty,
                     "n_with_cells": len(judgeable) - n_empty, "category": cat, "analyse": cat == "mixed"})
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for cat in ("all_cells", "all_empty", "mixed"):
        ps = [r["pos"] for r in rows if r["category"] == cat]
        print(f"{cat:9s} {len(ps):3d}: {' '.join(ps)}")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
