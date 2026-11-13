"""_qc_warp_interp_html_261112.py

One HTML page: every channel of every Pos at one frame, the live crop_sub (bilinear warp)
next to the same channel re-processed offline with bicubic / Lanczos4 / 5th-order spline /
no warp (see _qc_warp_interp_test_261112.run_pos).  Grouped by Pos; the Pos header carries the
frame shift, the grid point chosen and the residual warp the live run applied.  Tiles are
8-bit palette PNGs carrying the inferno LUT (vmin..vmax), base64-inlined.

    python scripts/_qc_warp_interp_html_261112.py --frame 7 --out C:/QPI_analysis/261112/_qc/warp_interp_f007_all.html
"""
import argparse
import base64
import html
import io
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

VARIANTS = ["linear", "cubic", "lanczos4", "spline5", "none"]
CFG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261112\drift_config_zstack.json")


def _lut(vmin, vmax):
    cm = colormaps["inferno"]
    pal = (np.array([cm(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.uint8).ravel().tolist()
    return pal


def encode_tile(a, vmin, vmax, pal):
    idx = np.clip((a - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(idx, mode="P")
    im.putpalette(pal)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def hf_metric(a, sigma=1.5, block=8, q=30):
    from scipy import ndimage
    r = a - ndimage.gaussian_filter(a, sigma)
    h, w = r.shape
    hb, wb = h // block, w // block
    blocks = r[:hb * block, :wb * block].reshape(hb, block, wb, block).std(axis=(1, 3))
    return float(np.percentile(blocks, q))


def _one_pos(args):
    label, t, vmin, vmax = args
    import _qc_warp_interp_test_261112 as T
    cfg = json.load(open(CFG, encoding="utf-8"))
    try:
        r = T.run_pos(cfg, label, t, VARIANTS)
    except Exception as e:  # a Pos without a frame record or grid point is reported, not hidden
        return label, None, repr(e)
    pal = _lut(vmin, vmax)
    n_ch = len(r["rois"])
    rows = []
    for ch in range(n_ch):
        tiles = {}
        for k in VARIANTS:
            a = r["results"][k][ch]
            if a is None:
                tiles[k] = None
                continue
            a = a.astype(np.float64)
            band = r["bands"][ch]
            tiles[k] = dict(png=encode_tile(a, vmin, vmax, pal), hf=hf_metric(a),
                            wall=float(np.abs(a[band]).mean()) if band.any() else float("nan"),
                            shape=a.shape)
        rows.append((ch, tiles))
    rx, ry = r["residual"]
    info = dict(shift=r["shift"], grid=r["grid"], residual=float(np.hypot(rx, ry)), rx=rx, ry=ry)
    return label, (info, rows), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=7)
    ap.add_argument("--vmin", type=float, default=-0.2)
    ap.add_argument("--vmax", type=float, default=0.2)
    ap.add_argument("--zoom", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pos", default=None, help="comma list; default every Pos with a frame record")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(CFG, encoding="utf-8"))
    crop_root = Path(cfg["crop_sub_root"])
    if args.pos:
        labels = [f"Pos{p}" for p in args.pos.split(",")]
    else:
        labels = sorted([d.name for d in crop_root.glob("Pos*") if d.is_dir() and d.name != "Pos0"],
                        key=lambda s: int(s[3:]))
    t0 = time.perf_counter()
    done = {}
    errors = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one_pos, (lab, args.frame, args.vmin, args.vmax)): lab for lab in labels}
        for i, f in enumerate(as_completed(futs), 1):
            lab, res, err = f.result()
            if err:
                errors[lab] = err
            else:
                done[lab] = res
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} Pos  ({time.perf_counter() - t0:.0f}s)", flush=True)

    z = args.zoom
    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>261112 frame {args.frame}: residual-warp interpolation, all channels</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#111;color:#ddd;margin:12px}}
h1{{font-size:15px;margin:4px 0 10px}} h2{{font-size:13px;margin:18px 0 4px;color:#fff}}
.pos{{border-top:1px solid #333;padding-top:6px}}
.hdr{{color:#aaa;font-size:11px;margin-bottom:4px}}
.row{{display:flex;align-items:center;gap:6px;margin:2px 0}}
.lab{{width:46px;font-size:11px;color:#ccc;text-align:right}}
.cell{{display:flex;flex-direction:column;align-items:center}}
.cell img{{image-rendering:pixelated;display:block}}
.cell .m{{font-size:9px;color:#888;height:11px}}
.vh{{display:flex;gap:6px;margin-left:52px;font-size:11px;color:#9cf}}
.vh div{{text-align:center}}
.warn{{color:#f88;font-size:11px}}
</style></head><body>
<h1>261112 frame {args.frame}: crop_sub with the residual sub-pixel warp done by
<span style="color:#9cf">bilinear (live) / bicubic / Lanczos4 / 5th-order spline / none</span>,
inferno {args.vmin}..{args.vmax} rad, every channel of every Pos</h1>
<div class="hdr">Under each tile: hf = high-frequency residual (tile minus Gaussian blur sigma 1.5 px, 8x8-block std, 30th percentile) / wall = mean |phase| in the wall band of the grid channel mask [rad].
Pos header: frame shift vs grid00 (px), grid point used, residual warp applied by the live run (px).</div>
"""]
    if errors:
        parts.append("<div class='warn'>not rendered: " + ", ".join(f"{k} ({html.escape(v)})" for k, v in sorted(errors.items())) + "</div>")
    for lab in labels:
        if lab not in done:
            continue
        info, rows = done[lab]
        sx, sy = info["shift"]
        parts.append(f"<div class='pos'><h2>{lab}</h2><div class='hdr'>shift ({sx:+.3f}, {sy:+.3f}) px &nbsp; grid ({info['grid'][0]:+d},{info['grid'][1]:+d}) &nbsp; "
                     f"residual warp ({info['rx']:+.3f}, {info['ry']:+.3f}) = <b style='color:#fff'>{info['residual']:.3f} px</b></div>")
        w = None
        for ch, tiles in rows:
            for k in VARIANTS:
                if tiles[k] is not None:
                    w = tiles[k]["shape"]
                    break
            if w:
                break
        tw = int(w[1] * z) if w else 360
        parts.append("<div class='vh'>" + "".join(f"<div style='width:{tw}px'>{k}</div>" for k in VARIANTS) + "</div>")
        for ch, tiles in rows:
            parts.append(f"<div class='row'><div class='lab'>ch{ch:02d}</div>")
            for k in VARIANTS:
                tl = tiles[k]
                if tl is None:
                    parts.append(f"<div class='cell'><div style='width:{tw}px;height:{int(40*z)}px;background:#222'></div><div class='m'>not written</div></div>")
                    continue
                hh, ww = tl["shape"]
                parts.append(f"<div class='cell'><img src='data:image/png;base64,{tl['png']}' width='{int(ww*z)}' height='{int(hh*z)}'>"
                             f"<div class='m'>hf {tl['hf']:.4f} / wall {tl['wall']:.3f}</div></div>")
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</body></html>")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"{out}  ({out.stat().st_size/1e6:.1f} MB)  {len(done)} Pos rendered, {len(errors)} errors, {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
