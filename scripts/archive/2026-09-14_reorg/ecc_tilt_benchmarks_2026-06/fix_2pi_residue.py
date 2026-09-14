"""Local 2pi-residue correction for crop_sub delta phase images.

The grid_subtract "2pi_correct" (ecc_utils.apply_2pi_tilt_crop) only applies a
single GLOBAL k*2pi offset to flatten the background; it cannot fix individual
pixels left 2pi out of step by the reconstruction unwrap (dense lipid-droplet
cores). This pass re-wraps each delta to [-pi, pi] and runs a proper 2D spatial
unwrap (skimage.restoration.unwrap_phase, same function as reconstruction), then
re-anchors the global level to the original background (multiples of 2pi).

Validated on 260517 Pos2 ch05: good frames unchanged, bad-frame droplet cores
lifted by +2pi (e.g. frame 1821 pixel -0.8 -> +5.48). Surgical: ~0.1 px/frame.

Output goes to a PARALLEL tree (IN_DIR.parent / crop_sub_rawraw_unwrap); the
original data is never modified in place.
"""
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.restoration import unwrap_phase

TWO_PI = 2.0 * np.pi

# --- config -----------------------------------------------------------------
IN_DIR = Path(
    r"e:/260517/2per_0055per_0per_2per_crop_sub/Pos2/output_phase/channels"
    r"/crop_sub_rawraw/z000/ch05"
)
OUT_DIR = Path(str(IN_DIR).replace("crop_sub_rawraw", "crop_sub_rawraw_unwrap"))


def fix_frame(a):
    """Re-wrap to [-pi, pi], 2D-unwrap, re-anchor background to original level."""
    r = unwrap_phase(np.angle(np.exp(1j * a)))
    r = r - np.round(np.median(r - a) / TWO_PI) * TWO_PI
    return r


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(IN_DIR.glob("*.tif"))
    if not files:
        raise SystemExit(f"no tif found in {IN_DIR}")

    n_changed_frames = 0
    total_changed_px = 0
    for i, f in enumerate(files):
        a = tifffile.imread(f).astype(np.float64)
        r = fix_frame(a)
        n = int((np.abs(r - a) > 1.0).sum())
        if n:
            n_changed_frames += 1
            total_changed_px += n
        tifffile.imwrite(OUT_DIR / f.name, r.astype(np.float32))
        if i % 500 == 0:
            print(f"  {i}/{len(files)}  {f.name}  changed_px={n}")

    print(f"\ndone: {len(files)} frames -> {OUT_DIR}")
    print(f"frames with >=1 fixed pixel: {n_changed_frames}")
    print(f"total fixed pixels: {total_changed_px}")


if __name__ == "__main__":
    main()
