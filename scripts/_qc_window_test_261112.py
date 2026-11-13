"""_qc_window_test_261112.py

Does a spectral window on the off-axis reconstruction help the crop_sub?  Offline, on the
re-shot grid of 261112 run 2 (Pos82-84, raw kept in GRID_RAW_KEEP) and on timelapse frames
whose raw was grabbed before the online cleanup (FRAME_RAW_KEEP, PosNN_fT.tif):

  variants (column order in the HTML)
    linear          no window, cv2 bilinear residual warp      (= run 1 pipeline)
    spline5         no window, 5th-order spline residual warp  (= run 2 pipeline)
    spline5+tukey   Tukey window (alpha TUKEY_ALPHA) on the sideband disk, spline5 warp
    spline5+hann    Hann window on the sideband disk, spline5 warp

The reconstruction mirrors batch_reconstruction_grid.reconstruct_image (crop -> FFT ->
sideband disk at NA -> crop to aperturesize -> IFFT -> angle -> unwrap_phase), with the
window multiplied onto the disk mask; the no-window variant is checked against the
pipeline's own output_phase_raw of the same grid point.  The grid point and the residual
used for a frame are the ones the online run logged for that frame
(pos_shifts_cal_online.json), so only the reconstruction and the warp differ.

    python scripts/_qc_window_test_261112.py --frame 4 --pos 82,83,84 --out C:/QPI_analysis/261112/_qc/window_test_f004.html
    python scripts/_qc_window_test_261112.py --pseudo --pos 82 ...   # grid point (0,+1) as the frame, mechanics check
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
from skimage.restoration import unwrap_phase

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE
from qpi import QPIParameters

GRID_RAW_KEEP = Path(r"D:\AquisitionData\Kitagishi\261112\grid_raw_keep_r1")
FRAME_RAW_KEEP = Path(r"D:\AquisitionData\Kitagishi\261112\frame_raw_keep_r2")
RECON = Path(r"D:\AquisitionData\Kitagishi\261112\recon_0p03um_19x19")
CROP_ROOT = Path(r"D:\AquisitionData\Kitagishi\261112\online_crop_sub_2per_5min_31d_2")
CFG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261112_2\drift_config_zstack.json")
POS_SPLIT = 60
CROP_BEFORE = (0, 2048, 400, 2448)
CROP_AFTER = (0, 2048, 0, 2048)
TUKEY_ALPHA = 0.3
VARIANTS = [("linear", None, None), ("spline5", 5, None),
            (f"spline5+tukey{TUKEY_ALPHA}", 5, "tukey"), ("spline5+hann", 5, "hann")]


def radial_window(shape, center, radius, kind):
    yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    r = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / radius
    disk = (r < 1.0).astype(np.float64)
    if kind is None:
        return disk
    if kind == "hann":
        w = 0.5 * (1 + np.cos(np.pi * np.clip(r, 0, 1)))
    elif kind == "tukey":
        a = TUKEY_ALPHA
        w = np.ones_like(r)
        taper = (r > 1 - a) & (r < 1)
        w[taper] = 0.5 * (1 + np.cos(np.pi * (r[taper] - (1 - a)) / a))
    else:
        raise ValueError(kind)
    return disk * w


def reconstruct(holo_path, crop, window):
    img = tifffile.imread(str(holo_path))
    rs, re_, cs, ce = crop
    img = img[rs:re_, cs:ce].astype(np.float64)
    p = QPIParameters(wavelength=WAVELENGTH, NA=NA, img_shape=img.shape, pixelsize=PIXELSIZE,
                      offaxis_center=OFFAXIS_CENTER)
    f = np.fft.fftshift(np.fft.fft2(img))
    f = f * radial_window(img.shape, p.offaxis_center, p.aperturesize / 2, window)
    w = p.aperturesize
    c = p.offaxis_center
    f = f[c[0] - w // 2: c[0] + w // 2 + 1, c[1] - w // 2: c[1] + w // 2 + 1]
    field = np.fft.ifft2(np.fft.ifftshift(f))
    return unwrap_phase(np.angle(field))


def hf_metric(a, sigma=1.5, block=8, q=30):
    r = a - ndimage.gaussian_filter(a, sigma)
    h, w = r.shape
    hb, wb = h // block, w // block
    blocks = r[:hb * block, :wb * block].reshape(hb, block, wb, block).std(axis=(1, 3))
    return float(np.percentile(blocks, q))


def encode_tile(a, vmin, vmax, pal):
    idx = np.clip((a - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(idx, mode="P")
    im.putpalette(pal)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=4)
    ap.add_argument("--pos", default="82,83,84")
    ap.add_argument("--pseudo", action="store_true", help="use grid point (0,+1) as the frame, residual from the calibration")
    ap.add_argument("--vmin", type=float, default=-0.2)
    ap.add_argument("--vmax", type=float, default=0.2)
    ap.add_argument("--zoom", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(CFG, encoding="utf-8"))
    gs.CH_MASK_THRESH = float(cfg.get("ch_mask_thresh", gs.CH_MASK_THRESH))
    gs.CH_MASK_DILATE = int(cfg.get("ch_mask_dilate", gs.CH_MASK_DILATE))
    gs.CH_MASK_MIN_BG = int(cfg.get("ch_mask_min_bg", gs.CH_MASK_MIN_BG))
    tilt_h = cfg.get("tilt_crop_h_raw", 270)
    out_h = cfg.get("crop_sub_output_crop_h") or tilt_h
    cm = colormaps["inferno"]
    pal = (np.array([cm(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.uint8).ravel().tolist()

    sections = []
    summary = []
    t0 = time.perf_counter()
    for pos in [int(p) for p in args.pos.split(",")]:
        label = f"Pos{pos}"
        crop = CROP_BEFORE if pos < POS_SPLIT else CROP_AFTER
        fit_right = pos >= POS_SPLIT
        rois = json.load(open(RECON / f"{label}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json", encoding="utf-8"))
        cal = gs.load_grid_calibration(str(RECON / f"grid_calibration_{label}.json"))
        pos_map = gs.scan_grid_positions(RECON, label)
        if args.pseudo:
            # frame = grid point (0,+1), reference = (0,0); the calibrated offset of (0,+1)
            # is the whole residual, so the warp is exercised (about one grid step).
            xi, yi = 0, 0
            cal_dx, cal_dy = 0.0, 0.0
            sx, sy = cal[(0, 1)]
            residual_x, residual_y = sx, sy
            frame_raw = GRID_RAW_KEEP / f"{label}_x+0_y+1" / "img_000000000_ph_000.tif"
            frame_tag = "grid point (0,+1) as frame, (0,0) as reference"
        else:
            fr = [f for f in json.load(open(CROP_ROOT / label / "output_phase" / "channels" / "pos_shifts_cal_online.json", encoding="utf-8"))["frame_results"]
                  if f and f.get("frame_index") == args.frame][0]
            sx, sy = fr["shift_x_avg"], fr["shift_y_avg"]
            xi, yi, _, _, _, cal_dx, cal_dy, residual_x, residual_y = gs.select_grid(
                sx, sy, pos_map, cal, pixel_scale_um=cfg["pixel_scale_um"],
                x_step=cfg.get("crop_sub_x_step_um", 0.05), y_step=cfg.get("crop_sub_y_step_um", 0.05),
                shift_sign_x=-1, shift_sign_y=-1)
            assert (xi, yi) == (fr["grid_xi"], fr["grid_yi"]), (label, xi, yi, fr)
            frame_raw = FRAME_RAW_KEEP / f"{label}_f{args.frame}.tif"
            frame_tag = f"frame {args.frame}"
        if not frame_raw.exists():
            print(f"{label}: frame raw missing: {frame_raw}")
            continue
        grid_raw = GRID_RAW_KEEP / f"{label}_x{xi:+d}_y{yi:+d}" / "img_000000000_ph_000.tif"
        ch_mask_img = tifffile.imread(str(RECON / f"{label}_x+0_y+0" / "output_phase" / "img_000000000_ph_000_phase.tif")).astype(np.float64)
        pipeline_ref = tifffile.imread(str(RECON / f"{label}_x{xi:+d}_y{yi:+d}" / "output_phase_raw" / "img_000000000_ph_000_phase.tif")).astype(np.float64)

        recon_cache = {}
        results = {}
        orig_order = gs.WARP_SPLINE_ORDER
        try:
            for name, order, window in VARIANTS:
                if window not in recon_cache:
                    g = reconstruct(grid_raw, crop, window)
                    f = reconstruct(frame_raw, crop, window)
                    recon_cache[window] = (g, f)
                g, f = recon_cache[window]
                gs.WARP_SPLINE_ORDER = order
                per_ch, _ = gs.process_single_frame(
                    f, sx, sy, rois, cal_dx, cal_dy, residual_x, residual_y, g,
                    output_crop_h_override=out_h, tilt_crop_h_raw=tilt_h, use_raw_phase=True,
                    apply_subpixel_correction=True, fit_right=fit_right, apply_inverse_shift=False,
                    bg_method=cfg.get("bg_method", "tilt"), ch_mask_img=ch_mask_img)
                results[name] = per_ch
        finally:
            gs.WARP_SPLINE_ORDER = orig_order
        g0 = recon_cache[None][0]
        dev = float(np.abs((g0 - g0.mean()) - (pipeline_ref - pipeline_ref.mean())).max())
        # wall bands
        bands = []
        for roi in rois:
            cx, cy = int(round(roi["cx"] + cal_dx)), int(round(roi["cy"] + cal_dy))
            m = gs.extract_rect_roi(ch_mask_img, cy, cx, roi["crop_w"], out_h) < gs.CH_MASK_THRESH
            bands.append(ndimage.binary_dilation(m, iterations=3) & ~ndimage.binary_erosion(m, iterations=3))
        rows_html = []
        for ch in range(len(rois)):
            cells = []
            for name, _, _ in VARIANTS:
                a = results[name][ch]
                if a is None:
                    cells.append("<div class='cell'><div class='m'>not written</div></div>")
                    continue
                a = a.astype(np.float64)
                hf = hf_metric(a)
                wall = float(np.abs(a[bands[ch]]).mean()) if bands[ch].any() else float("nan")
                summary.append((label, ch, name, hf, wall))
                hh, ww = a.shape
                cells.append(f"<div class='cell'><img src='data:image/png;base64,{encode_tile(a, args.vmin, args.vmax, pal)}' "
                             f"width='{int(ww*args.zoom)}' height='{int(hh*args.zoom)}'><div class='m'>hf {hf:.4f} / wall {wall:.3f}</div></div>")
            rows_html.append(f"<div class='row'><div class='lab'>ch{ch:02d}</div>{''.join(cells)}</div>")
        tw = int(out_h * args.zoom)
        vh = "".join("<div style='width:%dpx'>%s</div>" % (tw, n) for n, _, _ in VARIANTS)
        sections.append(f"<div class='pos'><h2>{label} &mdash; {html.escape(frame_tag)}</h2><div class='hdr'>shift ({sx:+.3f},{sy:+.3f}) px, grid ({xi:+d},{yi:+d}), "
                        f"residual warp ({residual_x:+.3f},{residual_y:+.3f}) px; no-window recon vs pipeline output_phase_raw: max |diff| {dev:.2e} rad</div>"
                        f"<div class='vh'>{vh}</div>{''.join(rows_html)}</div>")
        print(f"{label}: grid ({xi:+d},{yi:+d}) residual ({residual_x:+.3f},{residual_y:+.3f}) px; recon check max|diff| {dev:.2e}  ({time.perf_counter()-t0:.0f}s)")

    # summary table (medians over channels, excluding end traps)
    if summary:
        import collections
        agg = collections.defaultdict(list)
        n_ch = {}
        for label, ch, name, hf, wall in summary:
            agg[name].append((hf, wall))
        lines = ["<table><tr><th>variant</th><th>hf median [rad]</th><th>wall |res| median [rad]</th></tr>"]
        print(f"\n{'variant':22s} {'hf median':>10s} {'wall median':>12s}")
        for name, _, _ in VARIANTS:
            v = np.array(agg[name])
            print(f"{name:22s} {np.median(v[:,0]):10.4f} {np.nanmedian(v[:,1]):12.4f}")
            lines.append(f"<tr><td>{name}</td><td>{np.median(v[:,0]):.4f}</td><td>{np.nanmedian(v[:,1]):.4f}</td></tr>")
        lines.append("</table>")
        table = "".join(lines)
    else:
        table = ""
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>261112 run 2: spectral window test</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;background:#111;color:#ddd;margin:12px}}h1{{font-size:15px}}h2{{font-size:13px;color:#fff;margin:18px 0 4px}}
.pos{{border-top:1px solid #333;padding-top:6px}}.hdr{{color:#aaa;font-size:11px;margin-bottom:4px}}.row{{display:flex;align-items:center;gap:6px;margin:2px 0}}
.lab{{width:46px;font-size:11px;color:#ccc;text-align:right}}.cell{{display:flex;flex-direction:column;align-items:center}}.cell img{{image-rendering:pixelated;display:block}}
.cell .m{{font-size:9px;color:#888;height:11px}}.vh{{display:flex;gap:6px;margin-left:52px;font-size:11px;color:#9cf}}.vh div{{text-align:center}}
table{{border-collapse:collapse;font-size:12px;margin:8px 0}}td,th{{border:1px solid #444;padding:2px 8px}}</style></head><body>
<h1>261112 run 2: residual warp (bilinear / spline5) and a spectral window (Tukey {TUKEY_ALPHA} / Hann) on the sideband, crop_sub inferno {args.vmin}..{args.vmax} rad</h1>
<div class='hdr'>Grid and frame reconstructed offline from the kept raw holograms with the same window; grid point and residual as logged by the online run. hf = tile minus Gaussian blur (sigma 1.5 px), 8x8-block std, 30th pct; wall = mean |phase| in the wall band of the grid channel mask.</div>
{table}{''.join(sections)}</body></html>"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"\n{out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
