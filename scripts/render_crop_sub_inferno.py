"""
render_crop_sub_inferno.py
--------------------------
Offline preview renderer for the online crop_sub_rawraw delta TIFs.

Renders inferno-colormapped PNG previews (default vmin=0, vmax=1.8 rad) from the
float32 crop-subtracted delta TIFs written by compute_drift_online.py Phase B.
This is a post-hoc viewer: it never touches the realtime timelapse loop, and the
original float32 TIFs are left untouched (re-scale any time by re-running).

TIF layout (written by compute_drift_online.py Phase B):
  {root}/{Pos}/output_phase/channels/crop_sub_rawraw/z{zzz}/ch{cc}/img_{ttttttttt}_ph_{zzz}.tif

PNG output (sibling folder next to the TIFs):
  {root}/{Pos}/output_phase/channels/crop_sub_rawraw/z{zzz}/ch{cc}/preview_inferno/img_{ttttttttt}_ph_{zzz}.png

Usage:
  python render_crop_sub_inferno.py                          # defaults (z=10, all Pos/ch, sample 9 frames)
  python render_crop_sub_inferno.py --pos Pos1 Pos2 --frames all
  python render_crop_sub_inferno.py --frames 0 100 500       # specific frame indices
  python render_crop_sub_inferno.py --sample 12              # 12 evenly-spaced frames
  python render_crop_sub_inferno.py --z 10 --ch 2 3 --vmax 1.8
"""
import argparse
import re
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Defaults for the 260606 collar 0.11 session; override on the CLI.
DEFAULT_ROOT = r"D:\AquisitionData\Kitagishi\260606\0p11_zstack_1_crop_sub"
DEFAULT_VMIN = 0.0
DEFAULT_VMAX = 1.6
DEFAULT_CMAP = "inferno"
DEFAULT_Z    = 10          # focus plane (grid/timelapse index 10 = 0.0 um)

_FRAME_RE = re.compile(r"img_(\d+)_ph_")


def frame_index(tif_path):
    """Extract the timepoint index from img_{t}_ph_{z}.tif."""
    m = _FRAME_RE.search(tif_path.name)
    if not m:
        raise ValueError(f"cannot parse frame index from {tif_path.name}")
    return int(m.group(1))


def select_frames(tifs, frames, sample):
    """Pick which TIFs to render. frames=explicit indices, sample=N evenly spaced."""
    if frames == "all":
        return tifs
    if isinstance(frames, list):
        wanted = set(frames)
        return [t for t in tifs if frame_index(t) in wanted]
    # sample N evenly spaced
    if len(tifs) <= sample:
        return tifs
    idx = np.linspace(0, len(tifs) - 1, sample, dtype=int)
    return [tifs[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="crop_sub root (CROP_SUB_ROOT)")
    ap.add_argument("--pos", nargs="+", default=None,
                    help="Pos labels to render (default: all found)")
    ap.add_argument("--z", type=int, nargs="+", default=[DEFAULT_Z],
                    help="z indices to render (default: focus index 10)")
    ap.add_argument("--ch", type=int, nargs="+", default=None,
                    help="channel indices to render (default: all found)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--frames", nargs="+", default=None,
                   help="explicit frame indices, or the literal 'all'")
    g.add_argument("--sample", type=int, default=9,
                   help="render N evenly-spaced frames (default 9)")
    ap.add_argument("--vmin", type=float, default=DEFAULT_VMIN)
    ap.add_argument("--vmax", type=float, default=DEFAULT_VMAX)
    ap.add_argument("--cmap", default=DEFAULT_CMAP)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"crop_sub root not found: {root}")

    # Normalise the frame selection argument.
    if args.frames is None:
        frames = None  # use --sample
    elif len(args.frames) == 1 and args.frames[0].lower() == "all":
        frames = "all"
    else:
        frames = [int(f) for f in args.frames]

    pos_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.pos:
        keep = set(args.pos)
        pos_dirs = [p for p in pos_dirs if p.name in keep]
    if not pos_dirs:
        raise FileNotFoundError(f"no Pos directories under {root} (pos filter={args.pos})")

    total_png = 0
    for pos_dir in pos_dirs:
        base = pos_dir / "output_phase" / "channels" / "crop_sub_rawraw"
        if not base.exists():
            print(f"[skip] {pos_dir.name}: no crop_sub_rawraw yet")
            continue

        for z_idx in args.z:
            z_dir = base / f"z{z_idx:03d}"
            if not z_dir.exists():
                print(f"[skip] {pos_dir.name} z{z_idx:03d}: not found")
                continue

            ch_dirs = sorted(d for d in z_dir.iterdir()
                             if d.is_dir() and d.name.startswith("ch"))
            if args.ch is not None:
                keep_ch = {f"ch{c:02d}" for c in args.ch}
                ch_dirs = [d for d in ch_dirs if d.name in keep_ch]

            for ch_dir in ch_dirs:
                tifs = sorted(ch_dir.glob("img_*_ph_*.tif"))
                if not tifs:
                    continue
                chosen = select_frames(tifs, frames, args.sample)
                out_dir = ch_dir / "preview_inferno"
                out_dir.mkdir(parents=True, exist_ok=True)

                for tif in chosen:
                    img = tifffile.imread(str(tif)).astype(np.float32)
                    out_png = out_dir / (tif.stem + ".png")
                    plt.imsave(str(out_png), img, cmap=args.cmap,
                               vmin=args.vmin, vmax=args.vmax)
                    total_png += 1

                print(f"[ok] {pos_dir.name} z{z_idx:03d} {ch_dir.name}: "
                      f"{len(chosen)}/{len(tifs)} frames -> {out_dir}")

    print(f"\nDone. {total_png} PNG(s) rendered "
          f"(cmap={args.cmap}, vmin={args.vmin}, vmax={args.vmax}).")


if __name__ == "__main__":
    main()
