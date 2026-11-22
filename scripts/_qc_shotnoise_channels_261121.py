"""_qc_shotnoise_channels_261121.py

Phase noise at the channel positions against the shot-noise limit, for pairs of grid points of the
same Pos that sit closer than 0.1 px to each other (261121 11x11 grid, raw holograms still on E:).

Per Pos: all 121 holograms are reconstructed with the pipeline (grid_subtract._reconstruct_raw); the
position of every point relative to the first is measured on the channel block (phase correlation,
1/100 px); pairs are binned by their distance.  Per channel crop (40 x 240 px) of a pair:
  measured  = std over the region of (A - B) minus its gaussian(2 px) blur
  predicted = sqrt(mean over the region of 2 * 2 f / (N V^2)) x the fraction the same high-pass keeps
with N = electrons per raw pixel (local mean x GAIN) and V = local fringe visibility, both from the
raw hologram, low-passed over ~34 px (they cannot follow a drop of V on the 2 px wide walls).
Regions: flat bonded PDMS past the dead end / trap interior / wall band / whole crop.

    python scripts/_qc_shotnoise_channels_261121.py 5 10 15 20 25 30
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, zoom
from skimage.registration import phase_cross_correlation

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs
from channel_crop import extract_rect_roi
from optical_config import NA, PIXELSIZE, WAVELENGTH

RAW = r"E:/261121/grid11_s1"
ROIS = r"D:/AquisitionData/Kitagishi/261121/recon_0p03um_25x25"
OUT = Path(r"D:\AquisitionData\Kitagishi\261121\shotnoise_source")
GAIN = 2.6               # e-/DN: acA2440-75um, 10.7 ke- full well / 4095 DN (photon transfer here: 2.4-2.6)
CROP = (0, 2048, 400, 2448)      # Pos < 61
BINS = [(0, 0.02), (0.02, 0.04), (0.04, 0.07), (0.07, 0.10)]
PAIRS_PER_BIN = 6
yy, xx = np.mgrid[:2048, :2048]
fpp = 1 / (PIXELSIZE * 2048)
AP = 2 * round(NA / WAVELENGTH / fpp) + 1
F_FRAC = np.pi * (AP / 2) ** 2 / 2048 ** 2
_w = np.random.default_rng(0).standard_normal((511, 511))
HP_KEEP = float((_w - gaussian_filter(_w, 2)).std() / _w.std())


def vis_dc(a):
    F = np.fft.fftshift(np.fft.fft2(a))
    A = np.abs(F).copy()
    A[1024 - 80:1024 + 80, 1024 - 80:1024 + 80] = 0
    py, px = np.unravel_index(np.argmax(A), A.shape)

    def lp(cy, cx):
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= 60 ** 2
        G = np.zeros_like(F)
        G[m] = F[m]
        return np.fft.ifft2(np.fft.ifftshift(np.roll(G, (1024 - cy, 1024 - cx), axis=(0, 1))))

    dc = np.abs(lp(1024, 1024))
    return 2 * np.abs(lp(py, px)) / dc, dc


def one_pos(n):
    P = f"Pos{n}"
    dirs = sorted(glob.glob(f"{RAW}/{P}_x*"))
    if len(dirs) < 121:
        print(f"{P}: only {len(dirs)} points on disk, skipped")
        return []
    rois = json.load(open(f"{ROIS}/{P}_x+0_y+0/output_phase/channels/channel_rois.json"))
    files = [glob.glob(d + "/img_*.tif")[0] for d in dirs]
    ph = [np.asarray(gs._reconstruct_raw(Path(f), gs._make_qpi_params_raw(Path(f), CROP), CROP), float) for f in files]
    y0, y1 = max(rois[1]["cy"] - 20, 0), min(rois[-2]["cy"] + 20, 511)
    x0, x1 = max(rois[1]["cx"] - 125, 0), min(rois[1]["cx"] + 125, 511)
    hp8 = lambda a: a - gaussian_filter(a, 8)
    ref = hp8(ph[0][y0:y1, x0:x1])
    pos = np.array([phase_cross_correlation(ref, hp8(p[y0:y1, x0:x1]), upsample_factor=100, normalization=None)[0] for p in ph])
    d = np.hypot(pos[:, None, 0] - pos[None, :, 0], pos[:, None, 1] - pos[None, :, 1])
    np.fill_diagonal(d, 9)
    rng = np.random.default_rng(n)
    out = []
    cache = {}
    for lo, hi in BINS:
        pr = np.argwhere((d >= lo) & (d < hi))
        pr = pr[pr[:, 0] < pr[:, 1]]
        if len(pr) == 0:
            continue
        for i, j in pr[rng.choice(len(pr), min(PAIRS_PER_BIN, len(pr)), replace=False)]:
            if i not in cache:
                raw = tifffile.imread(files[i]).astype(np.float64)[CROP[0]:CROP[1], CROP[2]:CROP[3]]
                V, dc = vis_dc(raw)
                cache[i] = zoom(2 * 2 * F_FRAC / (dc * GAIN * V ** 2), 511 / 2048, order=1)
            pv = cache[i]
            for c in range(2, min(10, len(rois) - 1)):
                r = rois[c]
                a = extract_rect_roi(ph[i], r["cy"], r["cx"], 40, 240)
                b = extract_rect_roi(ph[j], r["cy"], r["cx"], 40, 240)
                p = extract_rect_roi(pv, r["cy"], r["cx"], 40, 240)
                diff = a - b
                diff = diff - gaussian_filter(diff, 2)
                g = a - np.median(a[:, 10:90])               # Pos < 61: bonded PDMS on the left, trap on the right
                trap = g < -1.0
                wall = binary_dilation(trap, iterations=3) & ~binary_erosion(trap, iterations=3)
                inside = binary_erosion(trap, iterations=4)
                flat = np.zeros_like(trap)
                flat[4:-4, 8:90] = True
                flat &= ~binary_dilation(trap, iterations=8)
                whole = np.zeros_like(trap)
                whole[2:-2, 2:-2] = True
                if inside.sum() < 200 or flat.sum() < 500:
                    continue
                row = [n, c, d[i, j]]
                for m in (flat, inside, wall, whole):
                    row += [diff[m].std(), np.sqrt(p[m].mean()) * HP_KEEP]
                out.append(row)
    print(f"{P}: {len(out)} channel crops; pair distances available: {[(lo, hi, int(((d >= lo) & (d < hi)).sum() // 2)) for lo, hi in BINS]}", flush=True)
    return out


def main():
    pos = [int(a) for a in sys.argv[1:]] or [5, 10, 15, 20, 25, 30]
    R = np.array([r for n in pos for r in one_pos(n)])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "channel_pairs_shotnoise.npz", rows=R, gain_e_per_dn=GAIN, filter_fraction=F_FRAC, hp_keep=HP_KEEP,
             columns=np.array(["pos", "ch", "pair_distance_px", "flat_meas", "flat_pred", "inside_meas", "inside_pred",
                               "wall_meas", "wall_pred", "whole_meas", "whole_pred"]))
    print(f"\n{len(R)} channel crops, gain {GAIN} e-/DN, f {F_FRAC:.4f}, high-pass keeps {HP_KEEP:.3f} of white noise")
    print("pair distance [px] | region: measured / shot-noise prediction = ratio   [rad]")
    for lo, hi in BINS:
        m = (R[:, 2] >= lo) & (R[:, 2] < hi)
        if not m.sum():
            continue
        s = f"  {lo:.2f}-{hi:.2f} (n={m.sum():3d})"
        for name, k in (("flat", 3), ("trap interior", 5), ("wall band", 7), ("whole crop", 9)):
            s += f" | {name}: {np.median(R[m, k]):.4f} / {np.median(R[m, k + 1]):.4f} = {np.median(R[m, k] / R[m, k + 1]):.2f}"
        print(s)
    print("saved", OUT / "channel_pairs_shotnoise.npz")


if __name__ == "__main__":
    main()
