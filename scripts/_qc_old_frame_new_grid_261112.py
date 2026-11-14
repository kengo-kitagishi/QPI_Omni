"""_qc_old_frame_new_grid_261112.py

How dirty does the subtraction get when the grid is older/newer than the frame?  Takes
frames of earlier runs (their reconstructed raw phase, output_phase_raw, is kept although
the holograms are gone) and subtracts them with TODAY's 25x25 stride grid, next to the same
frame subtracted with its own grid and next to a fresh run-3 frame.

For each Pos and old frame:
  1. shift of the old frame against the NEW grid00 = channel-average gaussian2d alignment of
     tilt-corrected crops (calibrate_grid_positions.get_crops_u8 / ecc_relative, the same
     estimator and crops the calibration uses)
  2. grid_subtract.select_grid in the new calibration -> grid point + residual
  3. grid_subtract.process_single_frame (spline-5 warp, outside_quad background) -> tiles
  The frame + its OWN grid is reprocessed the same way (shift from its online log) so that
  the only difference between the two columns is the age of the reference.

Per channel (end traps excluded): hf (tile minus Gaussian sigma 1.5 px, 8x8-block std, 30th
pct), wall residue (mean |phase| in the wall band of the grid channel mask), background std.
One HTML with the tiles side by side, grouped by Pos.

    python scripts/_qc_old_frame_new_grid_261112.py --out C:/QPI_analysis/261112/_qc/old_frames_new_grid.html
"""
import argparse
import base64
import html
import io
import json
import re
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
import calibrate_grid_positions as cgp

NEW_GRID = Path(r"D:\AquisitionData\Kitagishi\261112\recon_0p03um_25x25")
NEW_CFG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261112_3\drift_config_zstack.json")
# (tag, save_dir, crop_sub root, own grid dir, frame, approx. acquisition time)
OLD = [
    ("run 1 frame 20 (11/13 15:26, own grid 11x11 of 11/13 11:47)",
     Path(r"D:\AquisitionData\Kitagishi\261112\ph_2per_5min_31d_1"),
     Path(r"D:\AquisitionData\Kitagishi\261112\online_crop_sub_2per_5min_31d_1"),
     Path(r"D:\AquisitionData\Kitagishi\261112\recon_2per_grid_1"), 20),
    ("run 2 frame 3 (11/13 22:33, own grid 19x19 of 11/13 16-22h; grid deleted -> own-grid column skipped)",
     Path(r"D:\AquisitionData\Kitagishi\261112\ph_2per_5min_31d_2"),
     Path(r"D:\AquisitionData\Kitagishi\261112\online_crop_sub_2per_5min_31d_2"),
     None, 3),
]
FRESH = ("run 3 frame 6 (11/14 13:25, new grid of 11/13 23:48-11/14 12:40)",
         Path(r"D:\AquisitionData\Kitagishi\261112\ph_2per_5min_31d_3"),
         Path(r"D:\AquisitionData\Kitagishi\261112\online_crop_sub_2per_5min_31d_3"), 6)
POS_SPLIT = 60


def hf_metric(a, sigma=1.5, block=8, q=30):
    r = a - ndimage.gaussian_filter(a, sigma)
    h, w = r.shape
    hb, wb = h // block, w // block
    return float(np.percentile(r[:hb * block, :wb * block].reshape(hb, block, wb, block).std(axis=(1, 3)), q))


def encode_tile(a, vmin, vmax, pal):
    idx = np.clip((a - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(idx, mode="P")
    im.putpalette(pal)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def grid_parts(grid_dir, label, cfg):
    rois = json.load(open(grid_dir / f"{label}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json", encoding="utf-8"))
    cal = gs.load_grid_calibration(str(grid_dir / f"grid_calibration_{label}.json"))
    pos_map = gs.scan_grid_positions(grid_dir, label)
    g00_raw = tifffile.imread(str(grid_dir / f"{label}_x+0_y+0" / "output_phase_raw" / "img_000000000_ph_000_phase.tif")).astype(np.float64)
    mask_img = tifffile.imread(str(grid_dir / f"{label}_x+0_y+0" / "output_phase" / "img_000000000_ph_000_phase.tif")).astype(np.float64)
    return rois, cal, pos_map, g00_raw, mask_img


def subtract(tl_img, sx, sy, grid_dir, rois, cal, pos_map, mask_img, cfg, fit_right):
    xi, yi, _, _, _, cal_dx, cal_dy, rx, ry = gs.select_grid(
        sx, sy, pos_map, cal, pixel_scale_um=cfg["pixel_scale_um"],
        x_step=cfg.get("crop_sub_x_step_um", 0.05), y_step=cfg.get("crop_sub_y_step_um", 0.05),
        shift_sign_x=-1, shift_sign_y=-1)
    g = tifffile.imread(str(pos_map[(xi, yi)] / "output_phase_raw" / "img_000000000_ph_000_phase.tif")).astype(np.float64)
    tilt_h = cfg.get("tilt_crop_h_raw", 270)
    out_h = cfg.get("crop_sub_output_crop_h") or tilt_h
    per_ch, _ = gs.process_single_frame(tl_img, sx, sy, rois, cal_dx, cal_dy, rx, ry, g,
                                        output_crop_h_override=out_h, tilt_crop_h_raw=tilt_h, use_raw_phase=True,
                                        apply_subpixel_correction=True, fit_right=fit_right, apply_inverse_shift=False,
                                        bg_method=cfg.get("bg_method", "tilt"), ch_mask_img=mask_img)
    return per_ch, (xi, yi, float(np.hypot(rx, ry)), cal_dx, cal_dy, out_h)


def measure_shift(g00_raw, tl_img, rois, fit_right):
    """Channel-average gaussian2d shift of tl_img against grid00 (same crops as the calibration)."""
    n = len(rois)
    res = cgp.ecc_relative(cgp.get_crops_u8(g00_raw, rois, n, fit_right), cgp.get_crops_u8(tl_img, rois, n, fit_right), n)
    if res is None:
        return None
    adx, ady, corr = res
    return -adx, -ady, corr          # calibration stores actual = -tx; select_grid works in tx


def metrics(a, band, outer):
    a = a.astype(np.float64)
    return hf_metric(a), float(np.abs(a[band]).mean()) if band.any() else np.nan, float(a[outer].std()) if outer.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="1-124")
    ap.add_argument("--vmin", type=float, default=-0.2)
    ap.add_argument("--vmax", type=float, default=0.2)
    ap.add_argument("--zoom", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a, b = args.pos.split("-")
    pos_list = list(range(int(a), int(b) + 1))
    cfg = json.load(open(NEW_CFG, encoding="utf-8"))
    gs.CH_MASK_THRESH = float(cfg.get("ch_mask_thresh", gs.CH_MASK_THRESH))
    gs.CH_MASK_DILATE = int(cfg.get("ch_mask_dilate", gs.CH_MASK_DILATE))
    gs.CH_MASK_MIN_BG = int(cfg.get("ch_mask_min_bg", gs.CH_MASK_MIN_BG))
    cm = colormaps["inferno"]
    pal = (np.array([cm(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.uint8).ravel().tolist()

    columns = []
    for tag, save, crop, own, fr in OLD:
        if own is not None:
            columns.append((f"{tag.split(' (')[0]} + OWN grid", save, crop, own, fr))
        columns.append((f"{tag.split(' (')[0]} + NEW grid", save, crop, NEW_GRID, fr))
    columns.append((f"{FRESH[0].split(' (')[0]} + NEW grid", FRESH[1], FRESH[2], NEW_GRID, FRESH[3]))

    stats = {c[0]: [] for c in columns}
    sections = []
    t0 = time.perf_counter()
    for pos in pos_list:
        label = f"Pos{pos}"
        fit_right = pos >= POS_SPLIT
        try:
            new_parts = grid_parts(NEW_GRID, label, cfg)
        except FileNotFoundError:
            continue
        rois_new, cal_new, pm_new, g00_new, mask_new = new_parts
        n_ch = len(rois_new)
        tiles = {}
        info = {}
        for name, save, crop, grid_dir, fr in columns:
            tl_path = save / label / "z000" / "output_phase_raw" / f"img_{fr:09d}_ph_000_phase.tif"
            if not tl_path.exists():
                continue
            tl = tifffile.imread(str(tl_path)).astype(np.float64)
            if grid_dir == NEW_GRID:
                rois, cal, pm, g00, mask = new_parts
            else:
                try:
                    rois, cal, pm, g00, mask = grid_parts(grid_dir, label, cfg)
                except FileNotFoundError:
                    continue
            ms = measure_shift(g00, tl, rois, fit_right)
            if ms is None:
                continue
            sx, sy, corr = ms
            per_ch, inf = subtract(tl, sx, sy, grid_dir, rois, cal, pm, mask, cfg, fit_right)
            tiles[name] = (per_ch, rois, mask, inf)
            info[name] = (sx, sy, corr) + inf[:3]
        if not tiles:
            continue
        # bands from the NEW grid mask (all columns share the channel geometry up to <1 px)
        rows_html = []
        for ch in range(n_ch):
            cells = []
            for name, *_ in columns:
                if name not in tiles:
                    cells.append("<div class='cell'><div class='m'>n/a</div></div>")
                    continue
                per_ch, rois, mask, inf = tiles[name]
                a_ = per_ch[ch] if ch < len(per_ch) else None
                if a_ is None:
                    cells.append("<div class='cell'><div class='m'>not written</div></div>")
                    continue
                roi = rois[ch]
                g = gs.extract_rect_roi(mask, int(round(roi["cy"] + inf[4])), int(round(roi["cx"] + inf[3])), roi["crop_w"], inf[5])
                m = g < gs.CH_MASK_THRESH
                band = ndimage.binary_dilation(m, iterations=3) & ~ndimage.binary_erosion(m, iterations=3)
                outer = ~ndimage.binary_dilation(m, iterations=6)
                hf, wall, bg = metrics(a_, band, outer)
                if 0 < ch < n_ch - 1:
                    stats[name].append((hf, wall, bg))
                hh, ww = a_.shape
                cells.append(f"<div class='cell'><img src='data:image/png;base64,{encode_tile(a_.astype(np.float64), args.vmin, args.vmax, pal)}' "
                             f"width='{int(ww*args.zoom)}' height='{int(hh*args.zoom)}'><div class='m'>hf {hf:.4f} / wall {wall:.3f}</div></div>")
            rows_html.append(f"<div class='row'><div class='lab'>ch{ch:02d}</div>{''.join(cells)}</div>")
        hdr = " &nbsp;|&nbsp; ".join(f"{html.escape(n)}: shift ({v[0]:+.2f},{v[1]:+.2f}) corr {v[2]:.4f} grid ({v[3]:+d},{v[4]:+d}) resid {v[5]:.3f} px"
                                     for n, v in info.items())
        tw = int(240 * args.zoom)
        vh = "".join("<div style='width:%dpx'>%s</div>" % (tw, html.escape(n)) for n, *_ in columns)
        sections.append(f"<div class='pos'><h2>{label}</h2><div class='hdr'>{hdr}</div><div class='vh'>{vh}</div>{''.join(rows_html)}</div>")
        if pos % 10 == 0:
            print(f"  {label} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    print(f"\n{'column':44s} {'n ch':>5s} {'hf p50':>7s} {'p90':>6s} {'wall p50':>9s} {'p90':>6s} {'bg std':>7s}")
    lines = ["<table><tr><th>column</th><th>n ch</th><th>hf p50 / p90 [rad]</th><th>wall |res| p50 / p90 [rad]</th><th>bg std p50 [rad]</th></tr>"]
    for name, *_ in columns:
        v = np.array(stats[name])
        if len(v) == 0:
            continue
        print(f"{name:44s} {len(v):5d} {np.median(v[:,0]):7.4f} {np.percentile(v[:,0],90):6.4f} {np.nanmedian(v[:,1]):9.4f} {np.nanpercentile(v[:,1],90):6.4f} {np.nanmedian(v[:,2]):7.4f}")
        lines.append(f"<tr><td>{html.escape(name)}</td><td>{len(v)}</td><td>{np.median(v[:,0]):.4f} / {np.percentile(v[:,0],90):.4f}</td>"
                     f"<td>{np.nanmedian(v[:,1]):.4f} / {np.nanpercentile(v[:,1],90):.4f}</td><td>{np.nanmedian(v[:,2]):.4f}</td></tr>")
    lines.append("</table>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>261112: old frames subtracted with the new grid</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;background:#111;color:#ddd;margin:12px}}h1{{font-size:15px}}h2{{font-size:13px;color:#fff;margin:18px 0 4px}}
.pos{{border-top:1px solid #333;padding-top:6px}}.hdr{{color:#aaa;font-size:11px;margin-bottom:4px}}.row{{display:flex;align-items:center;gap:6px;margin:2px 0}}
.lab{{width:46px;font-size:11px;color:#ccc;text-align:right}}.cell{{display:flex;flex-direction:column;align-items:center}}.cell img{{image-rendering:pixelated;display:block}}
.cell .m{{font-size:9px;color:#888;height:11px}}.vh{{display:flex;gap:6px;margin-left:52px;font-size:11px;color:#9cf}}.vh div{{text-align:center}}
table{{border-collapse:collapse;font-size:12px;margin:8px 0}}td,th{{border:1px solid #444;padding:2px 8px}}</style></head><body>
<h1>261112: frames of earlier runs subtracted with their own grid and with the new 25x25 stride grid (11/13 23:48 - 11/14 12:40), next to a fresh run-3 frame. inferno {args.vmin}..{args.vmax} rad, spline-5 warp everywhere</h1>
<div class='hdr'>Shift of each frame against the respective grid00 measured here with the calibration estimator (gaussian2d, tilt crops), grid point and residual from select_grid. hf / wall as in the other QC pages.</div>
{''.join(lines)}{''.join(sections)}</body></html>"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"\n{out}  ({out.stat().st_size/1e6:.1f} MB)  {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
