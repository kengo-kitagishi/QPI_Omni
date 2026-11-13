"""_qc_calib_gauss_261112.py

Re-run the grid calibration with the gaussian2d estimator (the one the online drift
correction uses) instead of cv2 ECC, and compare the two calibrations point by point.

The live run reads grid_calibration_PosN.json from the grid dir at every timepoint, so the
new calibration is written to a SEPARATE folder (OUT_DIR) and nothing in the grid dir changes.
Only estimator / threshold / paths are swapped on the calibrate_grid_positions module;
the calibration logic itself is untouched.

    python scripts/_qc_calib_gauss_261112.py --pos 18,20,55,69,70,98,107,120 --threads 4
    python scripts/_qc_calib_gauss_261112.py --compare-only --pos ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

GRID_DIR = Path(r"D:\AquisitionData\Kitagishi\261112\recon_2per_grid_1")
CROP_ROOT = Path(r"D:\AquisitionData\Kitagishi\261112\online_crop_sub_2per_5min_31d_1")
OUT_DIR = Path(r"C:\QPI_analysis\261112\_qc\calib_gauss")
POS_SPLIT = 60
Z_INDEX = 0
NOMINAL_PX_PER_STEP = 0.05 / 0.34567514677103717


def run_one(label, threads):
    import cv2
    cv2.setNumThreads(1)
    import ecc_utils
    import calibrate_grid_positions as cgp
    def _align(ref, cur):
        # ecc_align returns None on a missing (out-of-bounds) crop via cv2.error;
        # gaussian2d_align does not guard against None, so do it here.
        if ref is None or cur is None:
            return None
        return ecc_utils.gaussian2d_align(ref, cur)
    cgp.ecc_align = _align                               # estimator swap (same call signature)
    cgp.ECC_MIN_CORR = ecc_utils.NCC_MIN_PEAK           # its own channel-selection threshold
    cgp.GRID_DIR = str(GRID_DIR)
    cgp.BASE_LABEL = label
    cgp.GRID_Z_INDEX = Z_INDEX
    cgp.CHANNEL_ROIS_JSON = str(GRID_DIR / f"{label}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json")
    cgp.POS_SPLIT = POS_SPLIT
    cgp.N_GRID_THREADS = threads
    cgp.SAVE_FIGURES = False
    cgp.OUTPUT_JSON = str(OUT_DIR / f"grid_calibration_{label}.json")
    t0 = time.perf_counter()
    cgp.main()
    return time.perf_counter() - t0


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    pts = {(e["xi"], e["yi"]): e for e in d["positions"]}
    return pts


def compare(labels, frame):
    import grid_subtract as gs
    print(f"\n{'Pos':6s} {'n':>3s} {'|dECC-dG| med/max x':>20s} {'y':>10s} {'slope x ECC/G':>14s} {'slope y ECC/G':>14s} "
          f"{'x cover ECC':>14s} {'x cover G':>14s} {'resid f7 ECC/G':>16s}")
    agg = []
    for label in labels:
        pe = load(GRID_DIR / f"grid_calibration_{label}.json")
        pg = load(OUT_DIR / f"grid_calibration_{label}.json")
        keys = sorted(set(pe) & set(pg))
        dxe = np.array([pe[k]["actual_dx_px"] for k in keys]); dye = np.array([pe[k]["actual_dy_px"] for k in keys])
        dxg = np.array([pg[k]["actual_dx_px"] for k in keys]); dyg = np.array([pg[k]["actual_dy_px"] for k in keys])
        xi = np.array([k[0] for k in keys]); yi = np.array([k[1] for k in keys])
        ddx, ddy = np.abs(dxe - dxg), np.abs(dye - dyg)
        sxe, sxg = np.polyfit(yi, dxe, 1)[0], np.polyfit(yi, dxg, 1)[0]
        sye, syg = np.polyfit(xi, dye, 1)[0], np.polyfit(xi, dyg, 1)[0]
        # residual the live run would have had at `frame` with each calibration
        shifts = json.load(open(CROP_ROOT / label / "output_phase" / "channels" / "pos_shifts_cal_online.json"))
        fr = [f for f in shifts["frame_results"] if f["frame_index"] == frame][0]
        sx, sy = fr["shift_x_avg"], fr["shift_y_avg"]
        pos_map = gs.scan_grid_positions(GRID_DIR, label)
        res = {}
        for tag, path in (("ECC", GRID_DIR / f"grid_calibration_{label}.json"), ("G", OUT_DIR / f"grid_calibration_{label}.json")):
            cal = gs.load_grid_calibration(str(path))
            r = gs.select_grid(sx, sy, pos_map, cal, pixel_scale_um=0.34567514677103717, shift_sign_x=-1, shift_sign_y=-1)
            res[tag] = (r[0], r[1], float(np.hypot(r[7], r[8])))
        print(f"{label:6s} {len(keys):3d} {np.median(ddx):9.3f}/{ddx.max():.3f} {np.median(ddy):4.3f}/{ddy.max():.3f}"
              f"   {sxe:+.3f}/{sxg:+.3f}     {sye:+.3f}/{syg:+.3f}"
              f"   [{-dxe.max():+.2f},{-dxe.min():+.2f}] [{-dxg.max():+.2f},{-dxg.min():+.2f}]"
              f"   {res['ECC'][2]:.3f}/{res['G'][2]:.3f}  grid ECC({res['ECC'][0]:+d},{res['ECC'][1]:+d}) G({res['G'][0]:+d},{res['G'][1]:+d})")
        agg.append((label, np.median(ddx), np.median(ddy), sxe, sxg, sye, syg, res["ECC"][2], res["G"][2]))
    a = np.array([r[1:] for r in agg], dtype=float)
    print(f"\nnominal {NOMINAL_PX_PER_STEP:.3f} px/step.  Over these Pos: median |dx diff| {np.median(a[:,0]):.3f} px, |dy diff| {np.median(a[:,1]):.3f} px;"
          f" slope x ECC {np.median(a[:,2]):.3f} vs G {np.median(a[:,3]):.3f}; slope y ECC {np.median(a[:,4]):.3f} vs G {np.median(a[:,5]):.3f};"
          f" frame-{frame} residual ECC {np.median(a[:,6]):.3f} vs G {np.median(a[:,7]):.3f} px")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="18,20,55,69,70,98,107,120")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frame", type=int, default=7)
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()
    labels = [f"Pos{p}" for p in args.pos.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.compare_only:
        for label in labels:
            if (OUT_DIR / f"grid_calibration_{label}.json").exists():
                print(f"{label}: exists, skipped")
                continue
            dt = run_one(label, args.threads)
            print(f"{label}: gaussian2d calibration done in {dt:.0f}s", flush=True)
    compare(labels, args.frame)


if __name__ == "__main__":
    main()
