"""test_gaussian2d_align.py -- sign and agreement checks for gaussian2d_align.

`ecc_utils.gaussian2d_align` is a drop-in replacement for `ecc_align`, so it must
return the SAME sign convention: a positive tx must mean the same physical
direction for both. The NCC peak offset runs opposite to the ECC warp, so the
adapter negates it -- and a sign error there would silently invert every stage
correction. This test pins it down on real channel crops with known shifts.

Checks
------
1. gaussian2d_align recovers a known (dy, dx) applied to a real crop.
2. gaussian2d_align and ecc_align agree to well under a pixel on the same pair.
3. The peak finder matches the benchmarked implementation in
   bench_subpix_methods.find_peak_gaussian_2d.

Usage
-----
    python scripts/test_gaussian2d_align.py
    python scripts/test_gaussian2d_align.py --grid-dir "E:\\260819\\grid_ye_1" --pos 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))

from ecc_utils import (ecc_align, gaussian2d_align, peak_gaussian_2d,
                       tilt_fit_crop, to_ecc_input, NCC_MARGIN)

ECC_VMIN, ECC_VMAX = -5.0, 2.0
ECC_CROP_H = 80
TILT_CROP_H = 270
WIDE = 160          # wide crop so a shift leaves no boundary artefact
TRUTHS = [(0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0),
          (0.7, -0.4), (-0.6, 0.9), (0.0, 3.0), (0.0, -3.0)]
TOL_TRUTH = 0.25    # px; content is real so a small bias is expected
TOL_AGREE = 0.30    # px; ECC vs Gaussian-2D on the same pair


def shift_fourier(arr, dy, dx):
    ny, nx = arr.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    ph = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    return np.real(np.fft.ifft2(np.fft.fft2(arr) * ph))


def center_crop_cols(wide, out_w):
    start = (wide.shape[1] - out_w) // 2
    return wide[:, start:start + out_w]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid-dir", default=r"E:\260819\grid_ye_1")
    p.add_argument("--pos", type=int, default=3)
    p.add_argument("--grid-z", type=int, default=3)
    p.add_argument("--n-channels", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    grid = Path(args.grid_dir) / f"Pos{args.pos}_x+0_y+0" / "output_phase"
    rois = json.loads((grid / "channels" / "channel_rois.json").read_text(encoding="utf-8"))
    paths = sorted(grid.glob(f"img_*_ph_{args.grid_z:03d}_phase.tif"))
    if not paths:
        raise FileNotFoundError(f"no grid image at z={args.grid_z} under {grid}")
    img = tifffile.imread(str(paths[0])).astype(np.float64)

    wides = []
    for roi in rois:
        w = tilt_fit_crop(img, roi["cy"], roi["cx"], roi["crop_w"],
                          ecc_crop_h=WIDE, tilt_crop_h=TILT_CROP_H, fit_right=False)
        if w is not None:
            wides.append(w)
        if len(wides) >= args.n_channels:
            break
    if not wides:
        raise RuntimeError("no channel fits the wide test window")
    print(f"Pos{args.pos}: {len(wides)} channels, wide crop {wides[0].shape}")

    # --- check 3: peak finder matches the benchmarked implementation ---
    import bench_subpix_methods as bsm
    rng = np.random.default_rng(0)
    surf = rng.random((21, 21)).astype(np.float32)
    surf[9, 12] += 5.0
    a = peak_gaussian_2d(surf)
    b = bsm.find_peak_gaussian_2d(surf)
    assert np.allclose(a, b, atol=1e-9), f"peak finder diverged: {a} vs {b}"
    print(f"peak finder matches bench implementation: {a}")

    # --- checks 1 and 2 ---
    err_g, err_e, agree = [], [], []
    for wide in wides:
        ref = center_crop_cols(wide, ECC_CROP_H).astype(np.float32)
        for dy, dx in TRUTHS:
            mov = center_crop_cols(shift_fourier(wide, dy, dx), ECC_CROP_H).astype(np.float32)
            g = gaussian2d_align(ref, mov)
            e = ecc_align(to_ecc_input(ref, ECC_VMIN, ECC_VMAX),
                          to_ecc_input(mov, ECC_VMIN, ECC_VMAX))
            assert g is not None, "gaussian2d_align returned None"
            if e is None:
                continue
            err_g.append((g[0] - dx, g[1] - dy))
            err_e.append((e[0] - dx, e[1] - dy))
            agree.append((g[0] - e[0], g[1] - e[1]))

    err_g = np.array(err_g); err_e = np.array(err_e); agree = np.array(agree)
    print(f"\nsamples: {len(err_g)}   truths: {TRUTHS}")
    for name, arr in (("Gaussian-2D err", err_g), ("ECC-float err  ", err_e),
                      ("G2D - ECC      ", agree)):
        print(f"  {name}  mean ({arr[:,0].mean():+.4f}, {arr[:,1].mean():+.4f}) px   "
              f"max|.| ({np.abs(arr[:,0]).max():.4f}, {np.abs(arr[:,1]).max():.4f}) px")

    # A sign flip would show up as an error of about twice the applied shift.
    assert np.abs(err_g).max() < TOL_TRUTH, (
        f"gaussian2d_align does not recover the known shift "
        f"(max err {np.abs(err_g).max():.3f} px) -- check the sign adapter")
    assert np.abs(agree).max() < TOL_AGREE, (
        f"gaussian2d_align disagrees with ecc_align by {np.abs(agree).max():.3f} px")
    print(f"\nOK: sign convention matches ecc_align (margin {NCC_MARGIN} px)")


if __name__ == "__main__":
    main()
