"""Temporal-prior 2pi relevel for crop_sub delta phase (whole-droplet wraps).

Spatial unwrap (fix_2pi_residue.py) only fixes pixels left 2pi off a clearly
brighter neighbor. It CANNOT fix a whole lipid droplet that sank 2pi as a unit,
because once re-wrapped the cell->droplet step (pi..2pi) looks locally smooth.

This pass adds the missing information from the TIME axis: a lipid droplet is a
persistent high-phase object, so a pixel is a "droplet location" if its temporal
98th percentile is high. At those pixels only, a frame that lands negative is a
2pi-wrapped droplet (phase only ever ADDS, so a deep negative is unphysical) and
is lifted by +2pi. Cell-edge halos (also ~ -1 rad) are left alone because their
temporal high baseline is cell-level, not droplet-level.

Discriminator validated on 260517 Pos2 ch05:
  frame 613 droplet hole -0.83 -> +5.45 (matches bright frames), 614 halo stays,
  1100 quiet frame untouched, 1821 single residue also caught.

Output -> parallel tree crop_sub_rawraw_tunwrap; originals never touched.
"""
import numpy as np
import tifffile
from pathlib import Path

TWO_PI = 2.0 * np.pi

# --- config -----------------------------------------------------------------
IN_DIR = Path(
    r"e:/260517/2per_0055per_0per_2per_crop_sub/Pos2/output_phase/channels"
    r"/crop_sub_rawraw/z000/ch05"
)
OUT_DIR = Path(str(IN_DIR).replace("crop_sub_rawraw", "crop_sub_rawraw_tunwrap"))

DROPLET_PCTL = 98      # temporal percentile used as per-pixel high baseline
DROPLET_HI = 3.0       # baseline above this => droplet location (phase >> cell)
NEG_THRESH = -0.5      # at a droplet pixel, value below this => wrapped, lift +2pi


def main():
    files = sorted(IN_DIR.glob("*.tif"))
    if not files:
        raise SystemExit(f"no tif found in {IN_DIR}")
    print(f"loading {len(files)} frames ...")
    stack = np.stack([tifffile.imread(f).astype(np.float32) for f in files])

    phi = np.percentile(stack, DROPLET_PCTL, axis=0)     # (H, W)
    droplet = phi > DROPLET_HI
    print(f"droplet pixels (p{DROPLET_PCTL}>{DROPLET_HI}): {int(droplet.sum())}")
    ys, xs = np.where(droplet)
    print("  locations (y,x):", list(zip(ys.tolist(), xs.tolist())))

    wrap = (stack < NEG_THRESH) & droplet[None]
    stack[wrap] += TWO_PI
    print(f"lifted {int(wrap.sum())} px across "
          f"{int(wrap.any(axis=(1, 2)).sum())} frames")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        tifffile.imwrite(OUT_DIR / f.name, stack[i])
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
