"""Add a big Pos/ch header band to each phase1-dead sheet (full size, no scaling).

Originals are tall/narrow with only a tiny title; for one-by-one review in the
session we prepend a tall black band with large white 'PosX chYY  (N cycles,
last row = death)' text so each displayed image is self-identifying at full res.
Outputs to %TEMP%/phase1dead_labeled/NN_PosX_chYY.png in list order.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LIST = Path(os.environ["TEMP"]) / "phase1dead_sheets.txt"
OUTDIR = Path(os.environ["TEMP"]) / "phase1dead_labeled"
OUTDIR.mkdir(exist_ok=True)

BAND_H = 70
BG = (0, 0, 0)
FG = (255, 255, 255)


def main():
    raw = LIST.read_text(encoding="utf-8-sig")
    entries = [ln.split("\t") for ln in raw.splitlines() if ln.strip()]
    try:
        font = ImageFont.truetype("arialbd.ttf", 44)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 44)
        except Exception:
            font = ImageFont.load_default()

    written = []
    for i, (lab, path) in enumerate(entries, 1):
        lab = lab.strip()
        try:
            im = Image.open(path.strip()).convert("RGB")
        except Exception as e:
            print("ERR", lab)
            continue
        w, h = im.size
        # widen a touch so a short sheet still fits the label text
        out_w = max(w, 460)
        canvas = Image.new("RGB", (out_w, h + BAND_H), BG)
        canvas.paste(im, ((out_w - w) // 2, BAND_H))
        d = ImageDraw.Draw(canvas)
        d.text((12, 12), f"{i:02d}.  {lab}", fill=FG, font=font)
        slug = lab.replace(" ", "_")
        op = OUTDIR / f"{i:02d}_{slug}.png"
        canvas.save(op)
        written.append(str(op))
        print(f"{i:02d} {lab} -> {op.name}  ({out_w}x{h+BAND_H})")
    # write order file
    (OUTDIR / "_order.txt").write_text("\n".join(written), encoding="utf-8")


if __name__ == "__main__":
    main()
