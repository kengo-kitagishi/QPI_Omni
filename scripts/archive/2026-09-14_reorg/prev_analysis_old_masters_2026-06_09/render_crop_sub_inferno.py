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


def _zlabel(z_idx, z_start, z_step):
    return f"z{z_idx:02d} ({z_start + z_idx * z_step:+.1f}um)"


def render_montage(pos_dirs, ch_filter, frame, vmin, vmax, cmap, z_start, z_step):
    """Through-focus montage: stack all z of one (Pos, ch, frame) vertically.

    One PNG per (Pos, ch) so the in-focus z is obvious at a glance. Focus does
    not move in time, so a single representative frame (default: last) suffices.
    """
    total = 0
    for pos_dir in pos_dirs:
        base = pos_dir / "output_phase" / "channels" / "crop_sub_rawraw"
        if not base.exists():
            print(f"[skip] {pos_dir.name}: no crop_sub_rawraw")
            continue
        z_dirs = sorted(d for d in base.iterdir()
                        if d.is_dir() and re.fullmatch(r"z\d+", d.name))
        if not z_dirs:
            continue
        z_indices = [int(d.name[1:]) for d in z_dirs]

        ch_names = [d.name for d in sorted(z_dirs[0].iterdir())
                    if d.is_dir() and d.name.startswith("ch")]
        if ch_filter is not None:
            keep = {f"ch{c:02d}" for c in ch_filter}
            ch_names = [c for c in ch_names if c in keep]

        out_dir = base / "_zmontage"
        out_dir.mkdir(parents=True, exist_ok=True)

        for ch_name in ch_names:
            sample_tifs = sorted((z_dirs[0] / ch_name).glob("img_*_ph_*.tif"))
            if not sample_tifs:
                continue
            fidx = frame_index(sample_tifs[-1]) if (frame is None or frame < 0) else frame

            panels = []
            for zd, zi in zip(z_dirs, z_indices):
                fp = zd / ch_name / f"img_{fidx:09d}_ph_{zi:03d}.tif"
                img = tifffile.imread(str(fp)).astype(np.float32) if fp.exists() else None
                panels.append((zi, img))

            n = len(panels)
            fig, axes = plt.subplots(n, 1, figsize=(6, max(0.5 * n, 4)))
            if n == 1:
                axes = [axes]
            im = None
            for ax, (zi, img) in zip(axes, panels):
                if img is not None:
                    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
                ax.set_ylabel(_zlabel(zi, z_start, z_step), rotation=0,
                              ha="right", va="center", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{pos_dir.name} {ch_name} frame{fidx}  inferno [{vmin},{vmax}]",
                         fontsize=9)
            if im is not None:
                fig.colorbar(im, ax=axes, shrink=0.5, label="delta phase (rad)")
            out_png = out_dir / f"{ch_name}_frame{fidx:09d}_zmontage.png"
            fig.savefig(str(out_png), dpi=120, bbox_inches="tight")
            plt.close(fig)
            total += 1
            print(f"[ok] {pos_dir.name} {ch_name}: z-montage ({n} z) frame{fidx}")
    print(f"\nDone. {total} z-montage(s) (cmap={cmap}, vmin={vmin}, vmax={vmax}).")


def build_zstacks(pos_dirs, ch_filter, frame, color=False,
                  cmap=DEFAULT_CMAP, vmin=DEFAULT_VMIN, vmax=DEFAULT_VMAX):
    """Assemble a single-frame, all-z multipage TIF per (Pos, ch) for ImageJ.

    color=False: float32 (nz, H, W) -- full data, adjust contrast in ImageJ.
    color=True : inferno-mapped RGB uint8 (nz, H, W, 3) at fixed [vmin, vmax],
                 so ImageJ shows the colormap directly (ImageJ lacks inferno).
    """
    cmap_obj = plt.get_cmap(cmap)
    total = 0
    for pos_dir in pos_dirs:
        base = pos_dir / "output_phase" / "channels" / "crop_sub_rawraw"
        if not base.exists():
            print(f"[skip] {pos_dir.name}: no crop_sub_rawraw")
            continue
        z_dirs = sorted(d for d in base.iterdir()
                        if d.is_dir() and re.fullmatch(r"z\d+", d.name))
        if not z_dirs:
            continue
        z_indices = [int(d.name[1:]) for d in z_dirs]

        ch_names = [d.name for d in sorted(z_dirs[0].iterdir())
                    if d.is_dir() and d.name.startswith("ch")]
        if ch_filter is not None:
            keep = {f"ch{c:02d}" for c in ch_filter}
            ch_names = [c for c in ch_names if c in keep]

        out_dir = base / ("_zstack_inferno" if color else "_zstacks")
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_inferno" if color else ""

        for ch_name in ch_names:
            sample_tifs = sorted((z_dirs[0] / ch_name).glob("img_*_ph_*.tif"))
            if not sample_tifs:
                continue
            fidx = frame_index(sample_tifs[-1]) if (frame is None or frame < 0) else frame

            planes = []
            for zd, zi in zip(z_dirs, z_indices):
                fp = zd / ch_name / f"img_{fidx:09d}_ph_{zi:03d}.tif"
                if fp.exists():
                    planes.append(tifffile.imread(str(fp)).astype(np.float32))
            if not planes:
                continue
            stack = np.stack(planes, axis=0)
            out_tif = out_dir / f"{ch_name}_frame{fidx:09d}_zstack{suffix}.tif"
            if color:
                norm = (np.clip(stack, vmin, vmax) - vmin) / (vmax - vmin)
                rgb = (cmap_obj(norm)[..., :3] * 255).astype(np.uint8)  # (nz, H, W, 3)
                tifffile.imwrite(str(out_tif), rgb, photometric="rgb")
            else:
                tifffile.imwrite(str(out_tif), stack, imagej=True)
            total += 1
            print(f"[ok] {pos_dir.name} {ch_name}: zstack {stack.shape} frame{fidx}"
                  f"{' inferno-RGB' if color else ''}")
    print(f"\nDone. {total} z-stack TIF(s).")


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
    ap.add_argument("--montage", action="store_true",
                    help="through-focus montage: stack all z per (Pos, ch) into one PNG")
    ap.add_argument("--zstack", action="store_true",
                    help="single-frame all-z multipage TIF per (Pos, ch) for ImageJ")
    ap.add_argument("--color", action="store_true",
                    help="with --zstack: write inferno-mapped RGB stack instead of float32")
    ap.add_argument("--frame", type=int, default=None,
                    help="frame index for --montage (default: last available)")
    ap.add_argument("--z-start", type=float, default=-4.0,
                    help="physical z (um) of z-index 0, for montage labels")
    ap.add_argument("--z-step", type=float, default=0.4,
                    help="z step (um) for montage labels")
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

    if args.zstack:
        build_zstacks(pos_dirs, args.ch, args.frame, args.color,
                      args.cmap, args.vmin, args.vmax)
        return

    if args.montage:
        ch_filter = args.ch
        render_montage(pos_dirs, ch_filter, args.frame,
                       args.vmin, args.vmax, args.cmap, args.z_start, args.z_step)
        return

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
