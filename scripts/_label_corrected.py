"""Collect the latest corrected phase1-dead sheets (a given inbox date) and add a
big Pos/ch header band to each for one-by-one session review.

Picks, per Pos:ch, the most recent --channel-all sheet under
inbox/<DATE>/fig_panelA_cellcycle, then writes labeled full-size copies to
%TEMP%/corrected_labeled/NN_PosX_chYY.png in Pos order.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DATE = os.environ.get("CORR_DATE", "2026-06-18")
BASE = Path(r"G:/共有ドライブ/wakamotolab_meeting/kitagishi/figure-hub/inbox") / DATE / "fig_panelA_cellcycle"
OUTDIR = Path(os.environ["TEMP"]) / "corrected_labeled"
OUTDIR.mkdir(exist_ok=True)
BAND_H = 70

DESC_RE = re.compile(r"All \d+ .*cell cycles of (Pos\w+) (ch\d+)")


def collect():
    latest = {}  # (pos,ch) -> (run_id, png_path)
    for jf in BASE.rglob("*_f001.json"):
        try:
            meta = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("params", {}).get("style") != "all-cycles-to-2018 per channel (sampled)":
            continue
        pos = meta["params"].get("pos")
        ch = meta["params"].get("ch")
        rid = meta.get("run_id", "")
        png = str(jf).replace(".json", ".png")
        key = (pos, ch)
        if key not in latest or rid > latest[key][0]:
            latest[key] = (rid, png)
    return latest


def main():
    latest = collect()
    order = sorted(latest.keys(), key=lambda k: (int(k[0].replace("Pos", "")), k[1]))
    try:
        font = ImageFont.truetype("arialbd.ttf", 44)
    except Exception:
        font = ImageFont.load_default()
    written = []
    for i, (pos, ch) in enumerate(order, 1):
        rid, png = latest[(pos, ch)]
        try:
            im = Image.open(png).convert("RGB")
        except Exception:
            print("ERR", pos, ch)
            continue
        w, h = im.size
        out_w = max(w, 520)
        canvas = Image.new("RGB", (out_w, h + BAND_H), (0, 0, 0))
        canvas.paste(im, ((out_w - w) // 2, BAND_H))
        ImageDraw.Draw(canvas).text((12, 12), f"{i:02d}.  {pos} {ch}",
                                    fill=(255, 255, 255), font=font)
        op = OUTDIR / f"{i:02d}_{pos}_{ch}.png"
        canvas.save(op)
        written.append(f"{pos} {ch}\t{op}")
        print(f"{i:02d} {pos} {ch}  {Path(png).parent.name}")
    (OUTDIR / "_order.txt").write_text("\n".join(written), encoding="utf-8")
    print("labeled", len(written), "-> ", OUTDIR)


if __name__ == "__main__":
    main()
