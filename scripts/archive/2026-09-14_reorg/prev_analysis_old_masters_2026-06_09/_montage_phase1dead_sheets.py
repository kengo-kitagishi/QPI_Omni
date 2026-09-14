"""Build a labeled gallery montage of the 26 phase1-dead full cell-cycle sheets.

Each sheet is very tall/narrow (one row per cell cycle, last row = death window).
For an at-a-glance overview we scale each to a common cell box, label it with
Pos/ch, and tile them in a grid on black (matches the inferno background).
This is a *viewing aid* only -- originals in the inbox are untouched.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LIST = Path(os.environ["TEMP"]) / "phase1dead_sheets.txt"
OUT = Path(os.environ["TEMP"]) / "phase1dead_gallery.png"

CELL_H = 1500          # target height per sheet (px)
CELL_W = 360           # max width per sheet (px); taller-than-this scale by width
LABEL_H = 46           # label band height
PAD = 14               # padding around each cell
PER_ROW = 9            # sheets per row
BG = (0, 0, 0)
FG = (255, 255, 255)


def load_entries():
    raw = LIST.read_text(encoding="utf-8-sig")  # strip BOM
    out = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        lab, path = line.split("\t")
        out.append((lab.strip(), path.strip()))
    return out


def fit(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = CELL_H / h
    if w * s > CELL_W:
        s = CELL_W / w
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def main():
    entries = load_entries()
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except Exception:
        font = ImageFont.load_default()

    thumbs = []
    for lab, path in entries:
        try:
            im = fit(Image.open(path).convert("RGB"))
        except Exception as e:
            print("ERR", lab)
            continue
        thumbs.append((lab, im))

    cell_w = CELL_W + 2 * PAD
    cell_h = CELL_H + LABEL_H + 2 * PAD
    n = len(thumbs)
    rows = (n + PER_ROW - 1) // PER_ROW
    W = PER_ROW * cell_w
    H = rows * cell_h
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    for i, (lab, im) in enumerate(thumbs):
        r, c = divmod(i, PER_ROW)
        x0 = c * cell_w
        y0 = r * cell_h
        # label
        draw.text((x0 + PAD, y0 + 8), lab, fill=FG, font=font)
        # image centered in the cell box, top-aligned under label
        iw, ih = im.size
        ix = x0 + PAD + (CELL_W - iw) // 2
        iy = y0 + LABEL_H + PAD
        canvas.paste(im, (ix, iy))

    canvas.save(OUT)
    print("saved", OUT)
    print("size", canvas.size, "thumbs", n)


if __name__ == "__main__":
    main()
