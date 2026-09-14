"""Combine spatial + temporal 2pi correction over the full frame stack.

Two orderings are written to separate trees so they can be compared in ImageJ:

  crop_sub_rawraw_t_then_s : temporal lift THEN spatial unwrap
      -> spatial unwrap re-lowers the isolated high droplet (step < 2pi), which
         REVERTS the temporal lift. Kept only for visual confirmation.
  crop_sub_rawraw_s_then_t : spatial unwrap THEN temporal lift  (RECOMMENDED)
      -> spatial fixes pixel-level residues next to bright neighbors; temporal
         then lifts whole-droplet wraps last, so both fixes survive.

Spatial  = rewrap to [-pi,pi] -> skimage.unwrap_phase -> re-anchor bg (k*2pi).
Temporal = per-pixel p98>3 droplet mask; at those pixels lift frames < -0.5 by 2pi.
Originals are never modified.
"""
import numpy as np
import tifffile
from pathlib import Path
from skimage.restoration import unwrap_phase

TWO_PI = 2.0 * np.pi
IN_DIR = Path(
    r"e:/260517/2per_0055per_0per_2per_crop_sub/Pos2/output_phase/channels"
    r"/crop_sub_rawraw/z000/ch05"
)
DROPLET_PCTL, DROPLET_HI, NEG_THRESH = 98, 3.0, -0.5


def spatial(a):
    r = unwrap_phase(np.angle(np.exp(1j * a)))
    return r - np.round(np.median(r - a) / TWO_PI) * TWO_PI


def temporal_lift(stack):
    """In-place per-pixel temporal-prior 2pi lift. Returns mask count."""
    phi = np.percentile(stack, DROPLET_PCTL, axis=0)
    droplet = phi > DROPLET_HI
    wrap = (stack < NEG_THRESH) & droplet[None]
    stack[wrap] += TWO_PI
    return int(droplet.sum()), int(wrap.sum())


def write_tree(stack, files, suffix):
    out = Path(str(IN_DIR).replace("crop_sub_rawraw", suffix))
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        tifffile.imwrite(out / f.name, stack[i].astype(np.float32))
    print(f"  -> {out}")
    return out


def main():
    files = sorted(IN_DIR.glob("*.tif"))
    print(f"loading {len(files)} frames ...")
    base = np.stack([tifffile.imread(f).astype(np.float32) for f in files])

    # ordering 1: temporal then spatial (reverts droplets; for comparison)
    s1 = base.copy()
    nd, nl = temporal_lift(s1)
    print(f"temporal: droplet_px={nd} lifted_px={nl}")
    for i in range(len(files)):
        s1[i] = spatial(s1[i].astype(np.float64))
    print("[t_then_s]")
    write_tree(s1, files, "crop_sub_rawraw_t_then_s")

    # ordering 2: spatial then temporal (recommended)
    s2 = base.copy()
    for i in range(len(files)):
        s2[i] = spatial(s2[i].astype(np.float64))
    nd2, nl2 = temporal_lift(s2)
    print(f"[s_then_t] temporal after spatial: droplet_px={nd2} lifted_px={nl2}")
    write_tree(s2, files, "crop_sub_rawraw_s_then_t")

    # verify on the known droplet-wrap frame
    idx = {f.name: i for i, f in enumerate(files)}
    j = idx["img_000000613_ph_000_phase.tif"]
    print("\nframe 613 core (x140,y20),(x141,y21):")
    print("  original : %.2f %.2f" % (base[j, 20, 140], base[j, 21, 141]))
    print("  t_then_s : %.2f %.2f" % (s1[j, 20, 140], s1[j, 21, 141]))
    print("  s_then_t : %.2f %.2f" % (s2[j, 20, 140], s2[j, 21, 141]))


if __name__ == "__main__":
    main()
