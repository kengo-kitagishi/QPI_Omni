# -*- coding: utf-8 -*-
"""channel_contact_sheet.py - one HTML page with the same frame of every trap channel.

For choosing by eye which channels go into the analysis: every PosN/chNN crop at one frame,
inferno-mapped, grouped by Pos, with the number of segmented frames next to each tile. Tiles are
click-selectable and the selection can be copied out as "PosN chNN" lines (kept in localStorage),
so the choice leaves the browser as a list.

    python scripts/channel_contact_sheet.py --raw-root D:/.../online_crop_sub_zstack_2 \
        --channel-rel output_phase/channels/crop_sub_rawraw/z000 --phase-glob "img_*_ph_000.tif" \
        --frame 100 --mask-root E:/260908_seg --out E:/260908_seg/_qc/contact_sheet_f100.html

The tiles are 8-bit PNGs that carry the inferno colour table itself (one byte per pixel), which
keeps a 1153-channel sheet at a few MB instead of tens.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from matplotlib import colormaps
from PIL import Image


def _frame_no(p: Path) -> int:
    m = re.search(r"img_0*(\d+)", p.name)
    return int(m.group(1)) if m else -1


def pick_frame(files, frame: int):
    """The requested img number, or the nearest one present (crop_sub can have gaps)."""
    if not files:
        return None
    return min(files, key=lambda p: abs(_frame_no(p) - frame))


def tile_png(img: np.ndarray, vmin: float, vmax: float, lut: bytes, rotate: bool) -> str:
    a = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    idx = (a * 255).astype(np.uint8)
    if rotate:
        idx = np.rot90(idx)          # traps read top-to-bottom, like the chip
    im = Image.fromarray(idx, mode="P")
    im.putpalette(lut)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


CSS = """
 body{font:13px system-ui,sans-serif;margin:0;background:#111;color:#ddd}
 header{position:sticky;top:0;background:#1b1b1b;padding:8px 12px;border-bottom:1px solid #333;z-index:9}
 button{font:inherit;margin-right:6px} #count{color:#8cf}
 .pos{padding:6px 12px 14px} .pos h2{font-size:14px;margin:10px 0 6px;color:#9cf;font-weight:600}
 .row{display:flex;flex-wrap:wrap;gap:8px}
 .t{border:2px solid transparent;border-radius:3px;padding:2px;cursor:pointer;text-align:center}
 .t.sel{border-color:#4ade80;background:#14301c}
 .t img{display:block;image-rendering:pixelated}
 .lab{font-size:11px;color:#aaa;margin-top:2px} .nm{color:#888}
 textarea{width:100%;height:140px;background:#000;color:#ddd;border:1px solid #333}
 dialog{background:#1b1b1b;color:#ddd;border:1px solid #444;width:60%}
"""

JS = """
const KEY = "contact_sheet_sel_" + location.pathname;
let sel = new Set(JSON.parse(localStorage.getItem(KEY) || "[]"));
function paint(){
  document.querySelectorAll(".t[data-k]").forEach(e => e.classList.toggle("sel", sel.has(e.dataset.k)));
  document.getElementById("count").textContent = sel.size;
}
function tog(e){
  const k = e.dataset.k;
  if (sel.has(k)) { sel.delete(k); } else { sel.add(k); }
  localStorage.setItem(KEY, JSON.stringify([...sel]));
  paint();
}
function num(s, pre){ return parseInt(s.split(pre)[1], 10); }
function showSel(){
  const ta = document.getElementById("ta");
  ta.value = [...sel].sort(function(a, b){
    const pa = num(a, "Pos"), pb = num(b, "Pos");
    return pa - pb || num(a, "ch") - num(b, "ch");
  }).join("\\n");
  document.getElementById("dlg").showModal();
  ta.select();
}
function clearSel(){ sel = new Set(); localStorage.setItem(KEY, "[]"); paint(); }
paint();
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--channel-rel", required=True)
    ap.add_argument("--phase-glob", default="img_*_ph_000.tif")
    ap.add_argument("--frame", type=int, default=100)
    ap.add_argument("--mask-root", default=None, help="show how many frames of each channel have masks")
    ap.add_argument("--pos-start", type=int, default=1)
    ap.add_argument("--pos-end", type=int, default=999)
    ap.add_argument("--vmin", type=float, default=0.15)
    ap.add_argument("--vmax", type=float, default=1.95)
    ap.add_argument("--zoom", type=float, default=2.0, help="CSS scale of each tile")
    ap.add_argument("--no-rotate", action="store_true", help="keep the crop wide instead of standing it up")
    ap.add_argument("--workers", type=int, default=16, help="cold reads parallelise well on this disk")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw_root, rel = Path(a.raw_root), Path(a.channel_rel)
    mask_root = Path(a.mask_root) if a.mask_root else None
    lut = bytes(bytearray(
        int(round(255 * c)) for i in range(256) for c in colormaps["inferno"](i / 255.0)[:3]))

    jobs = []
    for pos in range(a.pos_start, a.pos_end + 1):
        base = raw_root / f"Pos{pos}" / rel
        if not base.is_dir():
            continue
        for ch in sorted((d for d in base.glob("ch*") if d.is_dir()), key=lambda p: int(p.name[2:])):
            jobs.append((pos, ch.name, ch))
    if not jobs:
        print(f"no channels under {raw_root}/Pos*/{rel}")
        return 1
    print(f"{len(jobs)} channels; reading frame {a.frame} with {a.workers} threads", flush=True)

    def one(job):
        pos, chname, chdir = job
        f = pick_frame(sorted(chdir.glob(a.phase_glob)), a.frame)
        if f is None:
            return pos, chname, None, -1, 0
        img = tifffile.imread(str(f)).astype(np.float32)
        n_masks = 0
        if mask_root is not None:
            inf = mask_root / chdir.relative_to(raw_root) / "inference_out"
            n_masks = len(list(inf.glob("*_masks.tif"))) if inf.is_dir() else 0
        return pos, chname, tile_png(img, a.vmin, a.vmax, lut, not a.no_rotate), _frame_no(f), n_masks

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(one, jobs))
    print(f"read+encoded in {time.time() - t0:.0f}s", flush=True)

    by_pos = {}
    for pos, chname, b64, fno, n in results:
        by_pos.setdefault(pos, []).append((chname, b64, fno, n))

    parts = ['<!doctype html><meta charset="utf-8">',
             f"<title>frame {a.frame} - {html.escape(raw_root.name)}</title>",
             f"<style>{CSS}</style>",
             "<header>",
             f"<b>frame {a.frame}</b> &middot; {len(jobs)} channels &middot; "
             f"inferno {a.vmin}-{a.vmax} rad &middot; <span id=\"count\">0</span> selected ",
             '<button onclick="showSel()">copy list</button>',
             '<button onclick="clearSel()">clear</button>',
             '<span class="nm">click a tile to select; the number under it is how many frames of '
             'that channel have masks</span></header>',
             '<dialog id="dlg"><textarea id="ta"></textarea><br>'
             '<button onclick="document.getElementById(\'dlg\').close()">close</button></dialog>']
    for pos in sorted(by_pos):
        parts.append(f'<div class="pos"><h2>Pos{pos}</h2><div class="row">')
        for chname, b64, fno, n in by_pos[pos]:
            key = f"Pos{pos} {chname}"
            if b64 is None:
                parts.append(f'<div class="t"><div class="lab">{html.escape(key)}<br>no frame</div></div>')
                continue
            note = "" if fno == a.frame else f"img_{fno}"
            parts.append(
                f'<div class="t" data-k="{html.escape(key)}" onclick="tog(this)">'
                f'<img src="data:image/png;base64,{b64}" style="zoom:{a.zoom}">'
                f'<div class="lab">{html.escape(chname)} <span class="nm">{n}</span> {note}</div></div>')
        parts.append("</div></div>")
    parts.append(f"<script>{JS}</script>")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
