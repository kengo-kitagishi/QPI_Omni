"""Produce crop_sub_rawraw output for 260908 with a different sub-pixel interpolation.

Same output as the online run's crop_sub_rawraw/z000/chNN/img_*.tif, same kernel, same grid
node and residual -- only the interpolation of the residual warp is swapped:

  bilinear  cv2.warpAffine INTER_LINEAR  -- what the online run already wrote, not regenerated
                                            (checked bit-identical: max|diff| 0.000 rad)
  spline5   scipy.ndimage.shift(order=5)
  fourier   Fourier shift theorem

The validity mask always keeps the production cv2 warp: process_single_frame warps an
all-ones float32 array with the same function to mark out-of-bounds pixels and relies on cv2
filling outside with 0, so swapping that too would change the mask as well as the interpolation.

Resumable: a (frame, Pos) already written is skipped, so it can be stopped and restarted.

    python run_interp_arms_260908.py --frames 0-1259 --workers 10
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

CROP_SUB = Path(r"D:\AquisitionData\Kitagishi\260908\online_crop_sub_zstack_2")
TL_ROOT = Path(r"D:\AquisitionData\Kitagishi\260908\ph_zstack_2")
GRID_DIR = Path(r"E:\260908\ye_grid_0p05_2")
GRID_Z = 5
POS_SPLIT = 52
TILT_H, OUT_H = 270, 240


def build_arms():
    """Import inside the worker: each process gets its own single-threaded copy."""
    import cv2
    cv2.setNumThreads(1)
    import grid_subtract as gs
    from scipy.ndimage import shift as ndshift

    prod = gs.apply_inverse_shift_warp

    def only_image(fn):
        def wrapped(img, sx, sy):
            if np.asarray(img).dtype == np.float32:      # the ones-array validity mask
                return prod(img, sx, sy)
            return fn(img, sx, sy)
        return wrapped

    def spline5(img, sx, sy):
        return ndshift(img.astype(np.float64), (-sy, -sx), order=5,
                       mode="nearest", prefilter=True)

    def fourier(img, sx, sy):
        f = np.fft.fft2(img.astype(np.float64))
        ky = np.fft.fftfreq(img.shape[0])[:, None]
        kx = np.fft.fftfreq(img.shape[1])[None, :]
        return np.real(np.fft.ifft2(f * np.exp(-2j * np.pi * (ky * -sy + kx * -sx))))

    return gs, prod, {"spline5": only_image(spline5), "fourier": only_image(fourier)}


def do_pos(args):
    pos, frames, out_root, arms_wanted = args
    gs, prod, arm_fns = build_arms()
    label = f"Pos{pos}"
    ch_dir = CROP_SUB / label / "output_phase" / "channels"
    shifts_json, rois_json = ch_dir / "pos_shifts_cal_online.json", ch_dir / "channel_rois.json"
    if not (shifts_json.exists() and rois_json.exists()):
        return label, 0, "no shifts/rois"
    shifts = json.loads(shifts_json.read_text(encoding="utf-8"))
    frame_results = shifts.get("frame_results") or shifts.get("alignment_results") or []
    by_frame = {f["frame_index"]: f for f in frame_results if f}
    rois = json.loads(rois_json.read_text(encoding="utf-8"))
    cal_path = GRID_DIR / f"grid_calibration_{label}.json"
    grid_cal = gs.load_grid_calibration(str(cal_path)) if cal_path.exists() else {}
    fit_right = pos >= POS_SPLIT
    pixel_scale_um = (gs.SENSOR_PIXEL_SIZE / gs.MAGNIFICATION
                      * gs.ORIGINAL_DIM / gs.RECONSTRUCTED_DIM * 1e6)
    grid_cache = {}
    written = 0

    for frame in frames:
        e = by_frame.get(frame)
        if e is None:
            continue
        name = f"img_{frame:09d}_ph_000.tif"
        todo = [a for a in arms_wanted
                if not (Path(out_root) / a / label / "output_phase" / "channels" /
                        "crop_sub_rawraw" / "z000" / "ch00" / name).exists()]
        if not todo:
            continue
        tl_path = TL_ROOT / label / "z000" / "output_phase_raw" / \
            f"img_{frame:09d}_ph_000_phase.tif"
        if not tl_path.exists():
            continue
        xi, yi = int(e["grid_xi"]), int(e["grid_yi"])
        rx, ry = float(e["residual_x_px"]), float(e["residual_y_px"])
        sx, sy = float(e["shift_x_avg"]), float(e["shift_y_avg"])
        if (xi, yi) in grid_cal:
            cal_dx, cal_dy = grid_cal[(xi, yi)]
        else:
            cal_dx = shifts.get("shift_sign_y", 1) * yi * shifts.get("y_step_um", 0.05) / pixel_scale_um
            cal_dy = shifts.get("shift_sign_x", 1) * xi * shifts.get("x_step_um", 0.05) / pixel_scale_um
        if (xi, yi) not in grid_cache:
            g_path = GRID_DIR / f"{label}_x{xi:+d}_y{yi:+d}" / "output_phase_raw" / \
                f"img_000000000_ph_{GRID_Z:03d}_phase.tif"
            grid_cache[(xi, yi)] = (tifffile.imread(str(g_path)).astype(np.float64)
                                    if g_path.exists() else None)
        grid_img = grid_cache[(xi, yi)]
        if grid_img is None:
            continue
        tl_img = tifffile.imread(str(tl_path)).astype(np.float64)

        for arm in todo:
            gs.apply_inverse_shift_warp = arm_fns[arm]
            crops, _ = gs.process_single_frame(
                tl_img, sx, sy, rois, cal_dx, cal_dy, rx, ry, grid_img,
                output_crop_h_override=OUT_H, tilt_crop_h_raw=TILT_H, use_raw_phase=True,
                apply_subpixel_correction=True, fit_right=fit_right, apply_inverse_shift=False)
            gs.apply_inverse_shift_warp = prod
            for ch, crop in enumerate(crops):
                d = Path(out_root) / arm / label / "output_phase" / "channels" / \
                    "crop_sub_rawraw" / "z000" / f"ch{ch:02d}"
                d.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(str(d / name), crop.astype(np.float32))
                written += 1
    return label, written, "ok"


def parse_frames(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0-1259")
    ap.add_argument("--arms", nargs="+", default=["spline5", "fourier"])
    ap.add_argument("--pos-end", type=int, default=98)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-root", default=r"E:\260908_interp_arms")
    a = ap.parse_args()
    frames = parse_frames(a.frames)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    jobs = [(pos, frames, a.out_root, a.arms) for pos in range(1, a.pos_end + 1)]
    print(f"260908: {len(frames)} frames x {len(jobs)} Pos, arms {a.arms}, {a.workers} workers",
          flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(do_pos, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            label, written, status = fut.result()
            done += 1
            el = time.time() - t0
            print(f"[{done}/{len(jobs)}] {label}: {written} crops ({status})  "
                  f"elapsed {el/60:.1f} min, eta {el/done*(len(jobs)-done)/60:.0f} min",
                  flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {a.out_root}", flush=True)


if __name__ == "__main__":
    main()
