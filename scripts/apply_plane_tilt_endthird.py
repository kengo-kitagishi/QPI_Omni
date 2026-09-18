"""Fit a first-order 2D plane on the END THIRD only, and subtract it.

Why this and not the other two:
  grid_subtract's tilt   1D line (slope + intercept) on the width-MEAN profile of the end
                         third. Cannot touch a tilt across the 40 px width.
  apply_final_2d_flatten 2D quadratic over the WHOLE crop, cell pixels masked out. Fits
                         inside the channel, so the surface it subtracts under a cell mask
                         is set by its coefficients, not by data there.
  this script            2D plane (1, x, y) fitted on the end third ONLY -- the same
                         cell-free region the production tilt already uses -- so nothing is
                         ever fitted inside the channel, and no curvature is invented.
                         It only adds the width-direction tilt the 1D line cannot remove.

Input crops are the production tilt-corrected ones, so this is incremental: it removes
whatever plane is left in the end third.

Only the frames given by --frames are written (a contact sheet needs one frame per channel),
into a sibling tree ``<sub>_plane/chNN/``. Nothing is overwritten.

Import-safe: plane_endthird() is reusable, the CLI runs only under __main__.
"""
import argparse
import os
import re

import numpy as np
import tifffile

FIT_FRAC = 3   # end third


def plane_endthird(img, fit_right):
    """Subtract the plane fitted on the end third. Returns the corrected crop."""
    crop_w, out_h = img.shape
    n = max(1, out_h // FIT_FRAC)
    sl = slice(out_h - n, out_h) if fit_right else slice(0, n)
    y, x = np.mgrid[0:crop_w, 0:out_h]
    xs, ys, zs = x[:, sl].ravel(), y[:, sl].ravel(), img[:, sl].ravel()
    A = np.c_[np.ones_like(xs), xs, ys]
    co, *_ = np.linalg.lstsq(A, zs, rcond=None)
    return img - (co[0] + co[1] * x + co[2] * y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="session root holding PosN/output_phase/channels/...")
    ap.add_argument("--sub", default="crop_sub_rawraw/z000",
                    help="channel dir relative to channels/")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--pos", type=int, nargs="+", default=None,
                    help="default: every Pos found")
    ap.add_argument("--pos-split", type=int, default=52,
                    help="Pos >= this fits the right third")
    ap.add_argument("--suffix", default="_plane")
    a = ap.parse_args()

    pos_dirs = sorted(
        (int(m.group(1)), d) for d in os.listdir(a.root)
        if (m := re.match(r"^Pos(\d+)$", d)) and os.path.isdir(os.path.join(a.root, d))
    )
    n_ch = n_frames = 0
    for pos, d in pos_dirs:
        if a.pos and pos not in a.pos:
            continue
        base = os.path.join(a.root, d, "output_phase", "channels", a.sub)
        if not os.path.isdir(base):
            continue
        fit_right = pos >= a.pos_split
        out_base = base + a.suffix
        for ch in sorted(x for x in os.listdir(base) if re.match(r"^ch\d+$", x)):
            os.makedirs(os.path.join(out_base, ch), exist_ok=True)
            wrote = 0
            for t in a.frames:
                name = f"img_{t:09d}_ph_000.tif"
                src = os.path.join(base, ch, name)
                if not os.path.exists(src):
                    continue
                img = tifffile.imread(src).astype(np.float64)
                out = plane_endthird(img, fit_right)
                tifffile.imwrite(os.path.join(out_base, ch, name), out.astype(np.float32))
                wrote += 1
            n_ch += 1
            n_frames += wrote
    print(f"{n_ch} channels, {n_frames} frames -> <{a.sub}{a.suffix}>  "
          f"(plane on the end third, fit_right by Pos >= {a.pos_split})")


if __name__ == "__main__":
    main()
