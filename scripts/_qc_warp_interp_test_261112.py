"""_qc_warp_interp_test_261112.py

Does a higher-order interpolation in the residual sub-pixel warp reduce the wall residue
("引き残し") in the online crop_sub?  Offline re-run of grid_subtract.process_single_frame on
one frame of a few Pos, byte-identical inputs to the live run (same grid point, same residual,
same background method), with grid_subtract.apply_inverse_shift_warp swapped for:

  linear    cv2.INTER_LINEAR          (what the live pipeline uses; must reproduce the saved tif)
  cubic     cv2.INTER_CUBIC
  lanczos4  cv2.INTER_LANCZOS4
  spline5   scipy.ndimage.shift order=5 (the highest order scipy offers)
  none      no warp at all (residual ignored) -- shows how much the warp matters

The live pipeline is not touched: compute_drift_online.py is re-launched by the BeanShell
at every timepoint, so editing grid_subtract.py mid-run would change the output half way.

Per channel and variant: hf (tile minus Gaussian blur sigma 1.5 px, 8x8-block std, 30th pct),
wall residual (mean |phase| inside the wall band of the grid channel mask, band = dilate(3) -
erode(3) of the grid's phase < CH_MASK_THRESH region), and the tile std.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import figure_logger  # noqa: F401
from figure_logger import save_figure
import matplotlib.pyplot as plt
import grid_subtract as gs

CFG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261112\drift_config_zstack.json")
OUT_DIR = Path(r"C:\QPI_analysis\261112\_qc")
VARIANTS = ["linear", "cubic", "lanczos4", "spline5", "none"]


def make_warp(kind):
    import cv2
    flag = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos4": cv2.INTER_LANCZOS4}

    def warp(img, shift_x, shift_y):
        if kind == "none":
            return img.astype(np.float64)
        if kind == "spline5":
            # ndimage.shift moves content by +shift; the pipeline applies (-shift_x, -shift_y)
            return ndimage.shift(img.astype(np.float64), (-shift_y, -shift_x), order=5,
                                 mode="constant", cval=0.0, prefilter=True)
        h, w = img.shape
        m = np.array([[1.0, 0.0, -shift_x], [0.0, 1.0, -shift_y]], dtype=np.float32)
        return cv2.warpAffine(img.astype(np.float32), m, (w, h), flags=flag[kind]).astype(np.float64)
    return warp


def hf_metric(a, sigma=1.5, block=8, q=30):
    r = a - ndimage.gaussian_filter(a, sigma)
    h, w = r.shape
    hb, wb = h // block, w // block
    blocks = r[:hb * block, :wb * block].reshape(hb, block, wb, block).std(axis=(1, 3))
    return float(np.percentile(blocks, q))


def run_pos(cfg, pos_label, t, variants):
    crop_sub_root = Path(cfg["crop_sub_root"])
    save_dir = Path(cfg["save_dir"])
    grid_dir = Path(cfg["grid_dir"])
    pos_num = int(re.match(r"Pos(\d+)", pos_label).group(1))
    pos_split = int(cfg["pos_split"])
    fit_right = pos_num >= pos_split
    rois = json.load(open(grid_dir / f"{pos_label}_x+0_y+0" / "output_phase" / "channels"
                          / "channel_rois.json", encoding="utf-8"))
    shifts = json.load(open(crop_sub_root / pos_label / "output_phase" / "channels"
                            / "pos_shifts_cal_online.json", encoding="utf-8"))
    fr = [f for f in shifts["frame_results"] if f["frame_index"] == t][0]
    sx, sy = fr["shift_x_avg"], fr["shift_y_avg"]

    pos_map = gs.scan_grid_positions(grid_dir, pos_label)
    grid_cal = gs.load_grid_calibration(str(grid_dir / f"grid_calibration_{pos_label}.json"))
    xi, yi, dist_um, dx_um, dy_um, cal_dx, cal_dy, residual_x, residual_y = gs.select_grid(
        sx, sy, pos_map, grid_cal, pixel_scale_um=cfg["pixel_scale_um"],
        x_step=cfg.get("crop_sub_x_step_um", 0.05), y_step=cfg.get("crop_sub_y_step_um", 0.05),
        shift_sign_x=-1, shift_sign_y=-1)
    assert (xi, yi) == (fr["grid_xi"], fr["grid_yi"]), (pos_label, xi, yi, fr)
    grid_pos_dir = pos_map[(xi, yi)]
    z = cfg["raw_tl_z_index"]
    grid_z = z + cfg["raw_grid_z_index"] - cfg.get("raw_tl_z_index", 0)
    tl_img = tifffile.imread(str(save_dir / pos_label / f"z{z:03d}" / "output_phase_raw"
                                 / f"img_{t:09d}_ph_{z:03d}_phase.tif")).astype(np.float64)
    grid_img = tifffile.imread(str(grid_pos_dir / "output_phase_raw"
                                   / f"img_000000000_ph_{grid_z:03d}_phase.tif")).astype(np.float64)
    ch_mask_img = tifffile.imread(str(grid_pos_dir / "output_phase"
                                      / f"img_000000000_ph_{grid_z:03d}_phase.tif")).astype(np.float64)
    gs.CH_MASK_THRESH = float(cfg.get("ch_mask_thresh", gs.CH_MASK_THRESH))
    gs.CH_MASK_DILATE = int(cfg.get("ch_mask_dilate", gs.CH_MASK_DILATE))
    gs.CH_MASK_MIN_BG = int(cfg.get("ch_mask_min_bg", gs.CH_MASK_MIN_BG))
    tilt_h = cfg.get("tilt_crop_h_raw", 270)
    out_h = cfg.get("crop_sub_output_crop_h") or tilt_h

    results = {}
    orig_warp = gs.apply_inverse_shift_warp
    try:
        for kind in variants:
            gs.apply_inverse_shift_warp = make_warp(kind)
            per_ch, _ = gs.process_single_frame(
                tl_img, sx, sy, rois, cal_dx, cal_dy, residual_x, residual_y, grid_img,
                output_crop_h_override=out_h, tilt_crop_h_raw=tilt_h, use_raw_phase=True,
                apply_subpixel_correction=True, fit_right=fit_right, apply_inverse_shift=False,
                bg_method=cfg.get("bg_method", "tilt"), ch_mask_img=ch_mask_img)
            results[kind] = per_ch
    finally:
        gs.apply_inverse_shift_warp = orig_warp

    # wall band per channel from the grid channel mask, in the output crop geometry
    bands = []
    for ch, roi in enumerate(rois):
        cx, cy = int(round(roi["cx"] + cal_dx)), int(round(roi["cy"] + cal_dy))
        crop_w = roi["crop_w"]
        g = gs.extract_rect_roi(ch_mask_img, cy, cx, crop_w, out_h)
        m = g < gs.CH_MASK_THRESH
        band = ndimage.binary_dilation(m, iterations=3) & ~ndimage.binary_erosion(m, iterations=3)
        bands.append(band)

    # the saved online tif, to verify that "linear" reproduces the live run
    saved = []
    for ch in range(len(rois)):
        p = (crop_sub_root / pos_label / "output_phase" / "channels" / "crop_sub_rawraw"
             / f"z{z:03d}" / f"ch{ch:02d}" / f"img_{t:09d}_ph_{z:03d}.tif")
        saved.append(tifffile.imread(str(p)).astype(np.float32) if p.exists() else None)
    return dict(rois=rois, residual=(residual_x, residual_y), shift=(sx, sy), grid=(xi, yi),
                results=results, bands=bands, saved=saved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=7)
    ap.add_argument("--pos", default="18,20,55,69,70,98,107,120")
    args = ap.parse_args()
    cfg = json.load(open(CFG, encoding="utf-8"))
    t = args.frame
    pos_list = [f"Pos{p}" for p in args.pos.split(",")]

    rows = []
    examples = []
    for label in pos_list:
        r = run_pos(cfg, label, t, VARIANTS)
        rx, ry = r["residual"]
        # reproduction check
        diffs = [float(np.abs(a - b).max()) for a, b in zip(r["results"]["linear"], r["saved"])
                 if a is not None and b is not None]
        print(f"{label}: shift=({r['shift'][0]:+.3f},{r['shift'][1]:+.3f}) px  grid=({r['grid'][0]:+d},{r['grid'][1]:+d})"
              f"  residual warp=({rx:+.3f},{ry:+.3f}) px  |linear - saved online| max = {max(diffs):.2e}")
        n_ch = len(r["rois"])
        for ch in range(n_ch):
            if ch == 0 or ch == n_ch - 1:
                continue
            base = r["results"]["linear"][ch]
            if base is None:
                continue
            band = r["bands"][ch]
            for kind in VARIANTS:
                a = r["results"][kind][ch]
                if a is None:
                    continue
                a = a.astype(np.float64)
                rows.append((label, ch, kind, hf_metric(a), float(np.abs(a[band]).mean()),
                             float(a[~band].std()), float(np.hypot(rx, ry))))
        # example tiles: the channel with the largest wall residual in the linear output
        wr = [(float(np.abs(r["results"]["linear"][ch].astype(np.float64)[r["bands"][ch]]).mean()), ch)
              for ch in range(1, n_ch - 1) if r["results"]["linear"][ch] is not None]
        if wr:
            ch = max(wr)[1]
            examples.append((label, ch, {k: r["results"][k][ch] for k in VARIANTS}))

    rows = np.array(rows, dtype=object)
    print(f"\nframe {t}, {len(set((a, b) for a, b, *_ in rows))} channels (non-end) in {len(pos_list)} Pos")
    print(f"{'variant':9s} {'hf median':>10s} {'wall |res| median':>18s} {'bg std median':>14s}   relative to linear")
    base = {}
    for kind in VARIANTS:
        sel = rows[rows[:, 2] == kind]
        hf, wall, bg = (np.median(sel[:, 3].astype(float)), np.median(sel[:, 4].astype(float)),
                        np.median(sel[:, 5].astype(float)))
        if kind == "linear":
            base = dict(hf=hf, wall=wall, bg=bg)
        print(f"{kind:9s} {hf:10.4f} {wall:18.4f} {bg:14.4f}   "
              f"hf {hf/base['hf']*100:5.1f}%  wall {wall/base['wall']*100:5.1f}%  bg {bg/base['bg']*100:5.1f}%")
    # paired per-channel change vs linear
    print("\nper-channel change vs linear (median over channels, [p10, p90]):")
    lin = {(a, b): (c, d, e) for a, b, k, c, d, e, _ in rows if k == "linear"}
    for kind in VARIANTS[1:]:
        dh, dw = [], []
        for a, b, k, c, d, e, _ in rows:
            if k == kind and (a, b) in lin:
                dh.append(c / lin[(a, b)][0] * 100 - 100)
                dw.append(d / lin[(a, b)][1] * 100 - 100)
        dh, dw = np.array(dh), np.array(dw)
        print(f"  {kind:9s} hf {np.median(dh):+5.1f}% [{np.percentile(dh,10):+5.1f}, {np.percentile(dh,90):+5.1f}]"
              f"   wall {np.median(dw):+5.1f}% [{np.percentile(dw,10):+5.1f}, {np.percentile(dw,90):+5.1f}]")
    print("\nresidual warp magnitude used by the live run, per Pos: " +
          ", ".join(f"{label} {float(r):.3f} px" for label, r in
                    sorted({(a, g) for a, b, k, c, d, e, g in rows})))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"warp_interp_test_f{t:03d}.csv", "w", encoding="utf-8") as f:
        f.write("pos,ch,variant,hf,wall_abs_mean,bg_std,residual_px\n")
        for a, b, k, c, d, e, g in rows:
            f.write(f"{a},{b},{k},{c:.6f},{d:.6f},{e:.6f},{g:.4f}\n")

    # figure: example tiles per variant (inferno -0.2..0.2), rows = examples
    n_ex = len(examples)
    fig, axes = plt.subplots(n_ex, len(VARIANTS), figsize=(183 / 25.4, n_ex * 22 / 25.4))
    axes = np.atleast_2d(axes)
    for i, (label, ch, tiles) in enumerate(examples):
        for j, kind in enumerate(VARIANTS):
            ax = axes[i, j]
            ax.imshow(tiles[kind], cmap="inferno", vmin=-0.2, vmax=0.2, interpolation="nearest",
                      aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(kind, fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{label} ch{ch:02d}", fontsize=6)
    fig.suptitle(f"261112 frame {t}: residual sub-pixel warp interpolation, crop_sub [rad], inferno -0.2..0.2",
                 fontsize=7)
    fig.tight_layout()
    save_figure(fig, params={"frame": t, "pos": args.pos, "variants": VARIANTS, "dataset": "261112"},
                description=("Offline re-run of grid_subtract.process_single_frame on one timelapse frame with the "
                             "residual sub-pixel warp interpolation swapped (bilinear = live pipeline, bicubic, "
                             "Lanczos4, 5th-order spline, no warp). Rows: the channel with the largest wall residue "
                             "per Pos. Colour: crop_sub phase in rad, inferno -0.2..0.2."),
                data={"pos": rows[:, 0].astype(str), "ch": rows[:, 1].astype(int),
                      "variant": rows[:, 2].astype(str), "hf": rows[:, 3].astype(float),
                      "wall_abs_mean": rows[:, 4].astype(float), "bg_std": rows[:, 5].astype(float),
                      "residual_px": rows[:, 6].astype(float)},
                fmt="png")
    plt.close(fig)


if __name__ == "__main__":
    main()
