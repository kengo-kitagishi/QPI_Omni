"""One Pos over time: crop_sub tiles of every trap (rows) at each frame (columns), inferno, one HTML file.

    python pos_timeseries_sheet.py --raw-root D:/.../online_crop_sub_YE_5min_31d \
        --channel-rel output_phase/channels/crop_sub_rawraw/z000 --pos 1 --out sheet.html
        [--frames 0:200:5] [--vmin -0.2 --vmax 0.2]

Tiles are drawn exactly as in channel_contact_sheet.py (same colour table, same orientation).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from matplotlib import colormaps

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from channel_contact_sheet import _frame_no, tile_png  # noqa: E402

CSS = """
 body{font:13px system-ui,sans-serif;margin:0;background:#111;color:#ddd}
 header{position:sticky;top:0;left:0;background:#1b1b1b;padding:8px 12px;border-bottom:1px solid #333;z-index:9}
 table{border-collapse:collapse;margin:8px 12px} td,th{padding:2px 3px;text-align:center;vertical-align:top}
 th{font-weight:400;color:#9cf;font-size:11px;position:sticky;top:38px;background:#111}
 td.ch{color:#9cf;position:sticky;left:0;background:#111}
 img{display:block;image-rendering:pixelated}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--channel-rel", required=True)
    ap.add_argument("--phase-glob", default="img_*_ph_000.tif")
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("--frames", default=None, help="start:end:step (img numbers); default every frame present")
    ap.add_argument("--vmin", type=float, default=-0.2)
    ap.add_argument("--vmax", type=float, default=0.2)
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--interval-min", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = Path(a.raw_root) / f"Pos{a.pos}" / a.channel_rel
    chs = sorted((d for d in base.glob("ch*") if d.is_dir()), key=lambda p: int(p.name[2:]))
    if not chs:
        print(f"no channels under {base}")
        return 1
    files = {ch.name: {_frame_no(f): f for f in ch.glob(a.phase_glob)} for ch in chs}
    frames = sorted(set().union(*(set(v) for v in files.values())))
    if a.frames:
        lo, hi, step = (int(x) if x else None for x in (a.frames.split(":") + ["", ""])[:3])
        frames = [f for f in frames if (lo is None or f >= lo) and (hi is None or f <= hi)]
        frames = frames[::step or 1]
    lut = bytes(bytearray(int(round(255 * c)) for i in range(256) for c in colormaps["inferno"](i / 255.0)[:3]))

    head = "".join(f"<th>{f}<br>{f * a.interval_min / 60:.1f} h</th>" for f in frames)
    rows = []
    for ch in chs:
        cells = []
        for f in frames:
            p = files[ch.name].get(f)
            if p is None:
                cells.append("<td></td>")
                continue
            img = tifffile.imread(str(p)).astype(np.float32)
            h, w = img.shape
            src = tile_png(img, a.vmin, a.vmax, lut, rotate=True)
            cells.append(f'<td><img src="data:image/png;base64,{src}" width="{h * a.zoom:.0f}" '
                         f'height="{w * a.zoom:.0f}"></td>')
        rows.append(f'<tr><td class="ch">{ch.name}</td>{"".join(cells)}</tr>')
    title = f"Pos{a.pos} over time - {Path(a.raw_root).name}"
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head>"
            f"<body><header>{title} &middot; inferno {a.vmin}-{a.vmax} rad &middot; {len(frames)} frames "
            f"(img number / time)</header><table><tr><th></th>{head}</tr>{''.join(rows)}</table></body></html>")
    out = Path(a.out)
    out.write_text(html, encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {len(chs)} traps x {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
