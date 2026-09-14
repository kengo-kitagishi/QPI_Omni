"""
build_collar_compare_pos0sub.py
-------------------------------
Cross-collar comparison hyperstacks using the Pos0(BG)-subtracted phase at each
collar's focus z (reproduces compute_drift_online output_phase = phase_raw -
bg_phase - mean, from the retained output_phase_raw TIFs).

Per Pos: one inferno-RGB ImageJ hyperstack, Z slider = collar (at its focus z),
T slider = time. Lets you compare collars across all T with Pos0 subtraction.

VALID ONLY for before-crop positions (Pos1, Pos2 for POS_SPLIT=3): they share
Pos0's crop/FOV. After-crop Pos (Pos3-5) cannot be reproduced (Pos0 after-crop
BG not saved, raw holograms deleted).
"""
import glob
import os
import re

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"D:\AquisitionData\Kitagishi\260606"
OUT_DIR = os.path.join(BASE, "_collar_compare_pos0sub")

# (collar tag, focus z) from project_260606_collar_focus_z
COLLARS = [
    ("0p11", 17), ("0p12", 19), ("0p13", 20), ("0p14", 20), ("0p15", 16),
    ("0p16", 20), ("0p17", 19), ("0p18", 18), ("0p19", 20), ("0p20", 20),
]
POS_LIST = ["Pos1", "Pos2"]   # before-crop only (valid Pos0 subtraction)
BG_POS = "Pos0"
VMIN, VMAX = -5.0, 1.0
CMAP = "inferno"
SKIP_FIRST_FRAMES = 2

_FRE = re.compile(r"img_(\d+)_ph_")
_cmap = plt.get_cmap(CMAP)


def raw_map(collar, pos, z):
    d = os.path.join(BASE, f"{collar}_zstack_1", pos, f"z{z:03d}", "output_phase_raw")
    out = {}
    for f in glob.glob(os.path.join(d, f"img_*_ph_{z:03d}_phase.tif")):
        m = _FRE.search(os.path.basename(f))
        if m:
            out[int(m.group(1))] = f
    return out


def pos0sub_frames(collar, pos, z):
    """Pos0-subtracted, mean-removed frames (sorted by t), skipping T0/T1."""
    fn = raw_map(collar, pos, z)
    bg = raw_map(collar, BG_POS, z)
    ts = sorted(set(fn) & set(bg))
    ts = [t for t in ts if t >= SKIP_FIRST_FRAMES]
    frames = []
    for t in ts:
        ph = tifffile.imread(fn[t]).astype(np.float64) - tifffile.imread(bg[t]).astype(np.float64)
        h, w = ph.shape
        reg = ph[1:h - 1, 1:w // 2]
        if reg.size:
            ph = ph - reg.mean()
        frames.append(ph.astype(np.float32))
    return frames


def to_rgb(img):
    norm = (np.clip(img, VMIN, VMAX) - VMIN) / (VMAX - VMIN)
    return (_cmap(norm)[..., :3] * 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for pos in POS_LIST:
        per_collar = [(c, z, pos0sub_frames(c, pos, z)) for c, z in COLLARS]
        max_t = max((len(fr) for _, _, fr in per_collar), default=0)
        if max_t == 0:
            print(f"[skip] {pos}: no frames")
            continue
        H, W = next(fr[0].shape for _, _, fr in per_collar if fr)
        n = len(per_collar)
        arr = np.zeros((max_t, n, H, W, 3), dtype=np.uint8)
        labels = []
        for ci, (c, z, frames) in enumerate(per_collar):
            labels.append(f"{c}_z{z}({len(frames)}T)")
            for t in range(max_t):
                if frames:
                    arr[t, ci] = to_rgb(frames[t] if t < len(frames) else frames[-1])
        out = os.path.join(OUT_DIR, f"{pos}_pos0sub_collarT.tif")
        tifffile.imwrite(out, arr, imagej=True, metadata={"axes": "TZYXS"})
        print(f"{pos}: T={max_t} x {n} collars -> {os.path.basename(out)}")
        print(f"    Z(collar) order: {' | '.join(labels)}")
    print(f"\nDone. -> {OUT_DIR}  (Pos0-sub, inferno [{VMIN},{VMAX}]; Z=collar, T=time, T0/T1 dropped)")


if __name__ == "__main__":
    main()
