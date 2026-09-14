"""
render_pos0sub_focus.py
-----------------------
Reproduce the pipeline's Pos0(BG)-subtracted phase (output_phase = phase_raw -
bg_phase, compute_drift_online.py L755 + mean removal L759-761) at an ARBITRARY
z (e.g. the focus plane), which the online run only saved at z010.

Reconstructed from the retained output_phase_raw TIFs (raw holograms are gone):
  out = raw(PosN, z, t) - raw(Pos0, z, t) - mean(region)
Then inferno-RGB at [vmin, vmax], saved as a per-Pos time stack for ImageJ.

VALID ONLY for before-crop positions (pos index < POS_SPLIT): they share the
same crop / FOV background as Pos0. After-crop Pos (>=POS_SPLIT) would need a
Pos0 reconstruction with the after crop, which was not saved -> skipped.
"""
import argparse
import glob
import os
import re

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_FRE = re.compile(r"img_(\d+)_ph_")


def raw_frames(base, pos, z):
    d = os.path.join(base, pos, f"z{z:03d}", "output_phase_raw")
    out = {}
    for f in glob.glob(os.path.join(d, f"img_*_ph_{z:03d}_phase.tif")):
        m = _FRE.search(os.path.basename(f))
        if m:
            out[int(m.group(1))] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", required=True, help="timelapse save dir (has Pos*/zNNN/output_phase_raw)")
    ap.add_argument("--pos", nargs="+", default=["Pos1", "Pos2"])
    ap.add_argument("--bg", default="Pos0")
    ap.add_argument("--z", type=int, required=True)
    ap.add_argument("--vmin", type=float, default=-5.0)
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--cmap", default="inferno")
    args = ap.parse_args()

    base = args.save_dir
    cmap = plt.get_cmap(args.cmap)
    out_dir = os.path.join(base, "_pos0sub_focus")
    os.makedirs(out_dir, exist_ok=True)

    bg = raw_frames(base, args.bg, args.z)
    if not bg:
        raise FileNotFoundError(f"no {args.bg} z{args.z:03d} output_phase_raw under {base}")

    for pos in args.pos:
        fn = raw_frames(base, pos, args.z)
        ts = sorted(set(fn) & set(bg))
        if not ts:
            print(f"[skip] {pos}: no matching frames with {args.bg}")
            continue
        slices = []
        for t in ts:
            ph = tifffile.imread(fn[t]).astype(np.float64) - tifffile.imread(bg[t]).astype(np.float64)
            h, w = ph.shape
            reg = ph[1:h - 1, 1:w // 2]
            if reg.size:
                ph = ph - reg.mean()
            norm = (np.clip(ph, args.vmin, args.vmax) - args.vmin) / (args.vmax - args.vmin)
            slices.append((cmap(norm)[..., :3] * 255).astype(np.uint8))
        stack = np.stack(slices, axis=0)  # (T, H, W, 3)
        out = os.path.join(out_dir, f"{pos}_z{args.z:03d}_pos0sub_inferno.tif")
        tifffile.imwrite(out, stack, photometric="rgb")
        print(f"[ok] {pos}: {len(ts)} T -> {os.path.basename(out)}  (Pos0-sub, z{args.z}, {args.cmap} [{args.vmin},{args.vmax}])")
    print(f"\nDone. -> {out_dir}")


if __name__ == "__main__":
    main()
