"""Seg candidates for a timelapse: traps with cells in the Pos that are analysed (category 'mixed').

A trap is a candidate when its Pos is 'mixed' in pos_categories.csv and the trap was NOT picked as empty in the
whitelist review. The last trap of Pos >= --pos-split (field edge, quadrant subtraction unreliable) is left out.
Only a starting point for the seg selection page; the user reviews every tile.

    python seg_candidates_from_categories.py --categories drift_session_261108_3/pos_categories.csv \
        --picks drift_session_261108_3/whitelist_review_t1.selected.txt \
        --raw-root D:/.../online_crop_sub_YE_5min_31d_3 --pos-split 60 --out seg_candidates.txt
"""
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--channel-rel", default="output_phase/channels/crop_sub_rawraw/z000")
    ap.add_argument("--pos-split", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.categories, encoding="utf-8") as fh:
        mixed = [r["pos"] for r in csv.DictReader(fh) if r["category"] == "mixed"]
    empty = {" ".join(l.split()) for l in Path(a.picks).read_text(encoding="utf-8").splitlines() if len(l.split()) == 2}
    out = []
    for pos in mixed:
        n = int(pos[3:])
        chs = sorted((d.name for d in (Path(a.raw_root) / pos / a.channel_rel).glob("ch*") if d.is_dir()),
                     key=lambda s: int(s[2:]))
        if n >= a.pos_split:
            chs = chs[:-1]
        out += [f"{pos} {c}" for c in chs if f"{pos} {c}" not in empty]
    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(out)} candidate traps in {len(mixed)} mixed Pos -> {a.out}")


if __name__ == "__main__":
    main()
