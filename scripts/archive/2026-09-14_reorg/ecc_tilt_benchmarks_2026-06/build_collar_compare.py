"""
build_collar_compare.py
-----------------------
Cross-collar comparison hyperstacks for the 260606 correction-collar test.

For each (Pos, ch), assemble ONE inferno-RGB ImageJ hyperstack with two axes:
  - Z slider = correction collar (each at its in-focus z)
  - T slider = timepoint
So in ImageJ you scrub collar (Z) and time (T) to judge which collar is best
across the WHOLE timecourse, not just one frame.

Collars have different frame counts: shorter runs hold their last frame so the
time axis stays aligned by frame index; a collar with no data for that Pos/ch is
filled black so the collar (Z) order stays fixed across all files.

Edit COLLARS to add 0.19 / 0.20 once their focus z is determined.
"""
import os
import re

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"D:\AquisitionData\Kitagishi\260606"
OUT_DIR = os.path.join(BASE, "_collar_compare")

# (collar tag, in-focus z-index)  -- focus z from project_260606_collar_focus_z
COLLARS = [
    ("0p11", 17), ("0p12", 19), ("0p13", 20), ("0p14", 20),
    ("0p15", 16), ("0p16", 20), ("0p17", 19), ("0p18", 18),
    ("0p19", 20), ("0p20", 20),
]
POS_LIST = ["Pos1", "Pos2", "Pos3", "Pos4", "Pos5"]   # all Pos (Pos4 had ECC fallback but image still viewable)
CH_LIST = list(range(12))                              # all channels (ch1/ch8 are cell-free -> near-empty)
VMIN, VMAX = 0.0, 1.9
CMAP = "inferno"
SKIP_FIRST_FRAMES = 2   # drop T0/T1 (drift convergence) so T axis = steady state

_FRE = re.compile(r"img_(\d+)_ph_")
_cmap = plt.get_cmap(CMAP)


def collar_frames(pos, ch, cname, z):
    """Return list of float32 focus-z images (frame order) for one collar."""
    ch_dir = os.path.join(
        BASE, f"{cname}_zstack_1_crop_sub", pos, "output_phase",
        "channels", "crop_sub_rawraw", f"z{z:03d}", f"ch{ch:02d}")
    if not os.path.isdir(ch_dir):
        return []
    files = []
    for f in os.listdir(ch_dir):
        m = _FRE.search(f)
        if m and f.endswith(".tif"):
            files.append((int(m.group(1)), os.path.join(ch_dir, f)))
    files.sort()
    files = files[SKIP_FIRST_FRAMES:]   # drop convergence frames T0/T1
    return [tifffile.imread(fp).astype(np.float32) for _, fp in files]


def to_rgb(img):
    norm = (np.clip(img, VMIN, VMAX) - VMIN) / (VMAX - VMIN)
    return (_cmap(norm)[..., :3] * 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for pos in POS_LIST:
        for ch in CH_LIST:
            per_collar = [(c, z, collar_frames(pos, ch, c, z)) for c, z in COLLARS]
            lengths = [len(fr) for _, _, fr in per_collar]
            max_t = max(lengths) if lengths else 0
            if max_t == 0:
                continue
            # infer image shape from first non-empty collar
            H, W = next((fr[0].shape for _, _, fr in per_collar if fr))
            n_collar = len(per_collar)

            arr = np.zeros((max_t, n_collar, H, W, 3), dtype=np.uint8)
            labels = []
            for ci, (cname, z, frames) in enumerate(per_collar):
                labels.append(f"{cname}_z{z}({len(frames)}T)")
                if not frames:
                    continue  # leave black
                for t in range(max_t):
                    src = frames[t] if t < len(frames) else frames[-1]  # hold last
                    arr[t, ci] = to_rgb(src)

            out = os.path.join(OUT_DIR, f"{pos}_ch{ch:02d}_collarT.tif")
            # imagej hyperstack: axes T, Z(=collar), Y, X, S(=RGB)
            tifffile.imwrite(out, arr, imagej=True, metadata={"axes": "TZYXS"})
            total += 1
            print(f"{pos} ch{ch:02d}: T={max_t} x {n_collar} collars -> {os.path.basename(out)}")
            print(f"    Z(collar) order: {' | '.join(labels)}")
    print(f"\nDone. {total} collar x time hyperstack(s) in {OUT_DIR}")
    print(f"(inferno {VMIN}-{VMAX}; ImageJ: Z slider=collar, T slider=time; short runs hold last frame)")


if __name__ == "__main__":
    main()
