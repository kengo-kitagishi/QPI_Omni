"""_qc_align_bias_0per_261112.py

Is the 0%-medium frame aligned to the 2% grid with a small systematic offset?
For frames of run 3 in 2% (control) and in 0%, the online subtraction is redone with the
residual warp shifted by (dx, dy) over a +-0.3 px grid, and the misfit at the trap walls is
measured as the high-pass energy (tile minus Gaussian sigma 2 px) in the wall band.  The
refractive-index step inside the channel moves with the frame, so it does not depend on
(dx, dy); the grid-vs-frame wall mismatch does.  The (dx, dy) minimising the energy is the
alignment error of the online shift.  A consistent non-zero optimum in 0% and ~0 in 2% means
the estimator is biased by the medium.

    python scripts/_qc_align_bias_0per_261112.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs

CFG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261112_3\drift_config_zstack.json")
SAVE = Path(r"D:\AquisitionData\Kitagishi\261112\ph_2per_5min_31d_3")
CROP = Path(r"D:\AquisitionData\Kitagishi\261112\online_crop_sub_2per_5min_31d_3")
GRID = Path(r"D:\AquisitionData\Kitagishi\261112\recon_0p03um_25x25")
POS = list(range(3, 125, 8))
PERIODS = {"2%": [14, 20], "0%": [94, 100]}
REGION = sys.argv[1] if len(sys.argv) > 1 else "band"   # "band" = wall band, "outside" = background outside the channel
D = np.round(np.arange(-0.3, 0.3001, 0.1), 3)


def main():
    cfg = json.load(open(CFG, encoding="utf-8"))
    gs.CH_MASK_THRESH = float(cfg.get("ch_mask_thresh", gs.CH_MASK_THRESH))
    gs.CH_MASK_DILATE = int(cfg.get("ch_mask_dilate", gs.CH_MASK_DILATE))
    gs.CH_MASK_MIN_BG = int(cfg.get("ch_mask_min_bg", gs.CH_MASK_MIN_BG))
    tilt_h = cfg.get("tilt_crop_h_raw", 270)
    out_h = cfg.get("crop_sub_output_crop_h") or tilt_h
    t0 = time.perf_counter()
    res = {k: [] for k in PERIODS}
    for pos in POS:
        label = f"Pos{pos}"
        rois = json.load(open(GRID / f"{label}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json", encoding="utf-8"))
        cal = gs.load_grid_calibration(str(GRID / f"grid_calibration_{label}.json"))
        pm = gs.scan_grid_positions(GRID, label)
        mask_img = tifffile.imread(str(GRID / f"{label}_x+0_y+0" / "output_phase" / "img_000000000_ph_000_phase.tif")).astype(np.float64)
        frames = {f["frame_index"]: f for f in json.load(open(CROP / label / "output_phase" / "channels" / "pos_shifts_cal_online.json", encoding="utf-8"))["frame_results"] if f}
        fit_right = pos >= int(cfg["pos_split"])
        for period, flist in PERIODS.items():
            for t in flist:
                fr = frames.get(t)
                p = SAVE / label / "z000" / "output_phase_raw" / f"img_{t:09d}_ph_000_phase.tif"
                if fr is None or not p.exists():
                    continue
                tl = tifffile.imread(str(p)).astype(np.float64)
                sx, sy = fr["shift_x_avg"], fr["shift_y_avg"]
                xi, yi, _, _, _, cdx, cdy, rx, ry = gs.select_grid(sx, sy, pm, cal, pixel_scale_um=cfg["pixel_scale_um"],
                                                                    shift_sign_x=-1, shift_sign_y=-1)
                g = tifffile.imread(str(pm[(xi, yi)] / "output_phase_raw" / "img_000000000_ph_000_phase.tif")).astype(np.float64)
                bands = []
                for roi in rois:
                    m = gs.extract_rect_roi(mask_img, int(round(roi["cy"] + cdy)), int(round(roi["cx"] + cdx)), roi["crop_w"], out_h) < gs.CH_MASK_THRESH
                    if REGION == "outside":
                        bands.append(~ndimage.binary_dilation(m, iterations=6))
                    else:
                        bands.append(ndimage.binary_dilation(m, iterations=3) & ~ndimage.binary_erosion(m, iterations=3))
                E = np.zeros((len(D), len(D)))
                for i, dy in enumerate(D):
                    for j, dx in enumerate(D):
                        per_ch, _ = gs.process_single_frame(tl, sx, sy, rois, cdx, cdy, rx + dx, ry + dy, g,
                                                            output_crop_h_override=out_h, tilt_crop_h_raw=tilt_h, use_raw_phase=True,
                                                            apply_subpixel_correction=True, fit_right=fit_right, apply_inverse_shift=False,
                                                            bg_method=cfg.get("bg_method", "tilt"), ch_mask_img=mask_img)
                        e = []
                        for ch in range(1, len(rois) - 1):
                            a = per_ch[ch]
                            if a is None or a.shape != bands[ch].shape or not bands[ch].any():
                                continue
                            a = a.astype(np.float64)
                            hp = a - ndimage.gaussian_filter(a, 2.0)
                            e.append(np.mean(hp[bands[ch]] ** 2))
                        E[i, j] = np.mean(e) if e else np.nan
                i, j = np.unravel_index(np.nanargmin(E), E.shape)
                # sub-step refinement with a parabola on each axis
                def para(v, k):
                    if 0 < k < len(v) - 1:
                        den = v[k - 1] - 2 * v[k] + v[k + 1]
                        return D[k] + (0.5 * (v[k - 1] - v[k + 1]) / den * 0.1 if den > 0 else 0.0)
                    return D[k]
                bx, by = para(E[i, :], j), para(E[:, i], i)
                res[period].append((pos, t, bx, by, E[i, j] / E[len(D) // 2, len(D) // 2]))
        print(f"  {label} done ({time.perf_counter()-t0:.0f}s)", flush=True)
    print("\noptimal extra warp (dx, dy) in px that minimises the wall-band misfit; 0 = online alignment already optimal")
    for period, v in res.items():
        v = np.array(v)
        print(f"  {period}: n={len(v):3d}  dx median {np.median(v[:,2]):+.3f} (IQR {np.percentile(v[:,2],75)-np.percentile(v[:,2],25):.3f})  "
              f"dy median {np.median(v[:,3]):+.3f} (IQR {np.percentile(v[:,3],75)-np.percentile(v[:,3],25):.3f})  "
              f"misfit reduction at the optimum: median {(1-np.median(v[:,4]))*100:.0f}%")
    json.dump({k: [list(map(float, r)) for r in v] for k, v in res.items()},
              open(r"C:\QPI_analysis\261112\_qc\align_bias_0per.json", "w"), indent=1)


if __name__ == "__main__":
    main()
