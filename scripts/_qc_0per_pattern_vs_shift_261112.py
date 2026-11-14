"""_qc_0per_pattern_vs_shift_261112.py

Is the fixed-position noise that appears outside the channel in 0% medium the signature of a
misalignment?  Per trap tile (run 3, 2% grid as reference):

  A2 = mean of the online crop_sub tiles over 2% frames, A0 = the same over 0% frames
  F  = A0 - A2: the fixed-pattern change caused by the medium switch (random noise averaged out)
  Mx, My = what a misalignment looks like: a 2% frame re-processed with the residual warp
       offset by +0.2 px in x (resp. y) minus the same frame processed without offset

In the background outside the channel (mask dilated by 6 px removed), F is regressed on
(Mx, My): the fitted coefficients are the misalignment (in units of 0.2 px) that would best
explain the change, R2 says how much of the change a misalignment can explain at all.
An HTML shows A2 | A0 | F | Mx | My per channel.

    python scripts/_qc_0per_pattern_vs_shift_261112.py --out C:/QPI_analysis/261112/_qc/0per_pattern_vs_shift.html
"""
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from matplotlib import colormaps
from PIL import Image
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs

CFG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261112_3\drift_config_zstack.json")
SAVE = Path(r"D:\AquisitionData\Kitagishi\261112\ph_2per_5min_31d_3")
CROP = Path(r"D:\AquisitionData\Kitagishi\261112\online_crop_sub_2per_5min_31d_3")
GRID = Path(r"D:\AquisitionData\Kitagishi\261112\recon_0p03um_25x25")
F2 = list(range(8, 21))
F0 = list(range(85, 101))
T_TEMPLATE = 14
DELTA = 0.2


def tile_png(a, vmin, vmax, pal):
    idx = np.clip((a - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(idx, mode="P")
    im.putpalette(pal)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="3-124:8")
    ap.add_argument("--html-pos", default="11,35,59,83,107")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng, step = args.pos.split(":")
    a_, b_ = rng.split("-")
    pos_list = list(range(int(a_), int(b_) + 1, int(step)))
    html_pos = {int(p) for p in args.html_pos.split(",")}
    cfg = json.load(open(CFG, encoding="utf-8"))
    gs.CH_MASK_THRESH = float(cfg.get("ch_mask_thresh", gs.CH_MASK_THRESH))
    gs.CH_MASK_DILATE = int(cfg.get("ch_mask_dilate", gs.CH_MASK_DILATE))
    gs.CH_MASK_MIN_BG = int(cfg.get("ch_mask_min_bg", gs.CH_MASK_MIN_BG))
    tilt_h = cfg.get("tilt_crop_h_raw", 270)
    out_h = cfg.get("crop_sub_output_crop_h") or tilt_h
    cm = colormaps["inferno"]
    pal = (np.array([cm(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.uint8).ravel().tolist()
    t0 = time.perf_counter()
    coefs, r2s, f_std = [], [], []
    sections = []
    for pos in pos_list:
        label = f"Pos{pos}"
        rois = json.load(open(GRID / f"{label}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json", encoding="utf-8"))
        cal = gs.load_grid_calibration(str(GRID / f"grid_calibration_{label}.json"))
        pm = gs.scan_grid_positions(GRID, label)
        mask_img = tifffile.imread(str(GRID / f"{label}_x+0_y+0" / "output_phase" / "img_000000000_ph_000_phase.tif")).astype(np.float64)
        frames = {f["frame_index"]: f for f in json.load(open(CROP / label / "output_phase" / "channels" / "pos_shifts_cal_online.json", encoding="utf-8"))["frame_results"] if f}
        fit_right = pos >= int(cfg["pos_split"])
        # misalignment templates from one 2% frame
        fr = frames[T_TEMPLATE]
        sx, sy = fr["shift_x_avg"], fr["shift_y_avg"]
        xi, yi, _, _, _, cdx, cdy, rx, ry = gs.select_grid(sx, sy, pm, cal, pixel_scale_um=cfg["pixel_scale_um"], shift_sign_x=-1, shift_sign_y=-1)
        g = tifffile.imread(str(pm[(xi, yi)] / "output_phase_raw" / "img_000000000_ph_000_phase.tif")).astype(np.float64)
        tl = tifffile.imread(str(SAVE / label / "z000" / "output_phase_raw" / f"img_{T_TEMPLATE:09d}_ph_000_phase.tif")).astype(np.float64)
        run = lambda ddx, ddy: gs.process_single_frame(tl, sx, sy, rois, cdx, cdy, rx + ddx, ry + ddy, g, output_crop_h_override=out_h,
                                                       tilt_crop_h_raw=tilt_h, use_raw_phase=True, apply_subpixel_correction=True,
                                                       fit_right=fit_right, apply_inverse_shift=False,
                                                       bg_method=cfg.get("bg_method", "tilt"), ch_mask_img=mask_img)[0]
        base, px, py = run(0, 0), run(DELTA, 0), run(0, DELTA)
        rows = []
        for ch in range(1, len(rois) - 1):
            def avg(flist):
                acc = [tifffile.imread(str(CROP / label / "output_phase" / "channels" / "crop_sub_rawraw" / "z000" / f"ch{ch:02d}" / f"img_{t:09d}_ph_000.tif")).astype(np.float64)
                       for t in flist if (CROP / label / "output_phase" / "channels" / "crop_sub_rawraw" / "z000" / f"ch{ch:02d}" / f"img_{t:09d}_ph_000.tif").exists()]
                return np.mean(acc, axis=0) if len(acc) >= 5 else None
            A2, A0 = avg(F2), avg(F0)
            if A2 is None or A0 is None or base[ch] is None or px[ch] is None or py[ch] is None:
                continue
            Mx = px[ch].astype(np.float64) - base[ch]
            My = py[ch].astype(np.float64) - base[ch]
            roi = rois[ch]
            m = gs.extract_rect_roi(mask_img, int(round(roi["cy"] + cdy)), int(round(roi["cx"] + cdx)), roi["crop_w"], out_h) < gs.CH_MASK_THRESH
            if m.shape != A2.shape:
                continue
            outside = ~ndimage.binary_dilation(m, iterations=6) & (A2 != 0) & (A0 != 0)
            if outside.sum() < 300:
                continue
            F = A0 - A2
            y = F[outside] - F[outside].mean()
            X = np.column_stack([Mx[outside] - Mx[outside].mean(), My[outside] - My[outside].mean()])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum(y ** 2)
            coefs.append(beta * DELTA)          # misalignment in px that best explains the change
            r2s.append(r2)
            f_std.append((y.std(), (A2[outside] - A2[outside].mean()).std()))
            rows.append((ch, A2, A0, F, Mx, My, beta * DELTA, r2))
        if pos in html_pos and rows:
            tw = int(out_h * 1.5)
            heads = ["2% avg", "0% avg", "0% - 2% (change)", "misalign +0.2 px x", "misalign +0.2 px y"]
            vh = "".join("<div style='width:%dpx'>%s</div>" % (tw, h) for h in heads)
            body = []
            for ch, A2, A0, F, Mx, My, bshift, r2 in rows:
                cells = [tile_png(A2, -0.2, 0.2, pal), tile_png(A0, -0.2, 0.2, pal), tile_png(F, -0.1, 0.1, pal),
                         tile_png(Mx, -0.1, 0.1, pal), tile_png(My, -0.1, 0.1, pal)]
                hh, ww = A2.shape
                imgs = "".join(f"<img src='data:image/png;base64,{c}' width='{int(ww*1.5)}' height='{int(hh*1.5)}'>" for c in cells)
                body.append(f"<div class='row'><div class='lab'>ch{ch:02d}<br><span>fit ({bshift[0]:+.3f},{bshift[1]:+.3f}) px<br>R2 {r2:.2f}</span></div>{imgs}</div>")
            sections.append(f"<div class='pos'><h2>{label}</h2><div class='vh'>{vh}</div>{''.join(body)}</div>")
        print(f"  {label} ({time.perf_counter()-t0:.0f}s)", flush=True)
    c = np.array(coefs); r2 = np.array(r2s); fs = np.array(f_std)
    summary = (f"channels: {len(c)}\n"
               f"misalignment that best explains the 0%-2% change outside the channel: dx median {np.median(c[:,0]):+.3f} px (IQR {np.percentile(c[:,0],75)-np.percentile(c[:,0],25):.3f}), "
               f"dy median {np.median(c[:,1]):+.3f} px (IQR {np.percentile(c[:,1],75)-np.percentile(c[:,1],25):.3f})\n"
               f"variance of the change explained by a misalignment (R2): median {np.median(r2):.3f}, p90 {np.percentile(r2,90):.3f}\n"
               f"std of the change outside the channel: median {np.median(fs[:,0]):.4f} rad (2% time-average background itself {np.median(fs[:,1]):.4f} rad)")
    print("\n" + summary)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>261112 run 3: 0% pattern vs misalignment</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;background:#111;color:#ddd;margin:12px}}h1{{font-size:15px}}h2{{font-size:13px;color:#fff;margin:16px 0 4px}}
pre{{font-size:12px;color:#ccc}}.row{{display:flex;align-items:center;gap:6px;margin:2px 0}}.row img{{image-rendering:pixelated}}
.lab{{width:110px;font-size:11px;color:#ccc;text-align:right}}.lab span{{color:#888;font-size:10px}}.vh{{display:flex;gap:6px;margin-left:116px;font-size:11px;color:#9cf}}.vh div{{text-align:center}}</style></head><body>
<h1>261112 run 3, 2% grid as reference: time-averaged tile in 2% (frames {F2[0]}-{F2[-1]}) and 0% (frames {F0[0]}-{F0[-1]}), their difference, and what a 0.2 px misalignment looks like (inferno; tiles -0.2..0.2, change and misalignment -0.1..0.1 rad)</h1>
<pre>{summary}</pre>{''.join(sections)}</body></html>"""
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"{args.out}  ({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
