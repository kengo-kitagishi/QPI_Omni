"""_qc_shotnoise_source_261121.py

Source data for the "phase noise vs shot-noise limit" figure, taken from the raw holograms of the
261121 11x11 grid while they are still on E: (the chunk watcher deletes them after reconstruction).

Pos0 (empty background) is re-shot 121 times per chunk at commanded offsets of <= 0.15 um; pairs of
consecutive shots give
  * photon transfer: variance of the hologram difference vs mean signal, on pixels at the fringe
    extrema (local gradient in the lowest 3 %), so that fringe-phase jitter does not enter -> e-/DN
  * phase noise: both holograms reconstructed with the pipeline (grid_subtract._reconstruct_raw),
    difference high-passed (minus gaussian 8 px); and the per-pixel shot-noise prediction
        var(diff) = 2 * 2 f / (N V^2)
    N = electrons per raw pixel (local mean x gain), V = local fringe visibility (2|sideband|/DC,
    both low-passed with a 60 px radius), f = filter disk area / spectrum area.

    python scripts/_qc_shotnoise_source_261121.py
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, zoom

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs
from optical_config import NA, PIXELSIZE, WAVELENGTH

RAW_GLOB = r"E:/261121/grid11_s*/Pos0_x*"
OUT = Path(r"D:\AquisitionData\Kitagishi\261121\shotnoise_source")
CROP = (0, 2048, 400, 2448)
N_PAIRS = 30
LP_RADIUS = 60

yy, xx = np.mgrid[:2048, :2048]


def local_vis_and_dc(a):
    F = np.fft.fftshift(np.fft.fft2(a))
    A = np.abs(F).copy()
    A[1024 - 80:1024 + 80, 1024 - 80:1024 + 80] = 0
    py, px = np.unravel_index(np.argmax(A), A.shape)

    def lp(cy, cx):
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= LP_RADIUS ** 2
        G = np.zeros_like(F)
        G[m] = F[m]
        return np.fft.ifft2(np.fft.ifftshift(np.roll(G, (1024 - cy, 1024 - cx), axis=(0, 1))))

    dc = np.abs(lp(1024, 1024))
    return 2 * np.abs(lp(py, px)) / dc, dc, (int(py), int(px))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dirs = sorted(glob.glob(RAW_GLOB), key=os.path.getmtime)
    print(f"{len(dirs)} Pos0 points on disk")
    fpp = 1 / (PIXELSIZE * 2048)
    ap = 2 * round(NA / WAVELENGTH / fpp) + 1
    f = np.pi * (ap / 2) ** 2 / 2048 ** 2
    pt_mean, pt_var = [], []
    diffs, preds, vis, dcs, names = [], [], [], [], []
    for i in range(0, min(len(dirs) - 1, 2 * N_PAIRS), 2):
        fa = glob.glob(dirs[i] + "/img_*.tif")[0]
        fb = glob.glob(dirs[i + 1] + "/img_*.tif")[0]
        a = tifffile.imread(fa).astype(np.float64)
        b = tifffile.imread(fb).astype(np.float64)
        # photon transfer on the full frame
        m = (a + b) / 2
        gy, gx = np.gradient(m)
        g = np.hypot(gx, gy)
        flat = (g < np.percentile(g, 3)) & (m < 3500)
        s = b.sum() / a.sum()
        d = b - s * a
        for lo in range(100, 3000, 100):
            k = flat & (m >= lo) & (m < lo + 100)
            if k.sum() > 2000:
                pt_mean.append(m[k].mean())
                pt_var.append(d[k].var() / (1 + s * s))
        # phase noise
        pa = np.asarray(gs._reconstruct_raw(Path(fa), gs._make_qpi_params_raw(Path(fa), CROP), CROP), float)
        pb = np.asarray(gs._reconstruct_raw(Path(fb), gs._make_qpi_params_raw(Path(fb), CROP), CROP), float)
        diff = pb - pa
        diff = diff - gaussian_filter(diff, 8)
        V, dc, peak = local_vis_and_dc(a[CROP[0]:CROP[1], CROP[2]:CROP[3]])
        z = diff.shape[0] / 2048
        diffs.append(diff.astype(np.float32))
        vis.append(zoom(V, z, order=1).astype(np.float32))
        dcs.append(zoom(dc, z, order=1).astype(np.float32))
        names.append(f"{Path(dirs[i]).name}|{Path(dirs[i + 1]).name}")
        print(f"pair {len(diffs)}: {names[-1]}  phase-diff std (centre) {diff[120:-120, 120:-120].std():.4f} rad", flush=True)
    pt_mean, pt_var = np.array(pt_mean), np.array(pt_var)
    slope, icpt = np.polyfit(pt_mean, pt_var, 1)
    gain = 1 / slope
    print(f"photon transfer: var = {slope:.4f} * mean + {icpt:.1f}  ->  {gain:.2f} e-/DN ({len(pt_mean)} bins)")
    diffs, vis, dcs = np.array(diffs), np.array(vis), np.array(dcs)
    pred_var = 2 * 2 * f / (dcs * gain * vis ** 2)
    c = (slice(None), slice(120, -120), slice(120, -120))
    meas = diffs[c].std(axis=(1, 2))
    pred = np.sqrt(pred_var[c].mean(axis=(1, 2)))
    print(f"phase noise of the difference: measured {np.median(meas):.4f} rad, shot-noise prediction {np.median(pred):.4f} rad, "
          f"ratio {np.median(meas / pred):.3f} (range {np.min(meas / pred):.3f}-{np.max(meas / pred):.3f}), {len(meas)} pairs")
    np.savez_compressed(OUT / "pos0_pairs_shotnoise.npz",
                        phase_diff_hp8=diffs, visibility=vis, dc_dn=dcs, pred_var=pred_var.astype(np.float32),
                        pair_names=np.array(names), pt_mean_dn=pt_mean, pt_var_dn2=pt_var,
                        gain_e_per_dn=gain, filter_fraction=f, aperturesize=ap,
                        na=NA, wavelength_m=WAVELENGTH, pixelsize_m=PIXELSIZE, crop=np.array(CROP),
                        measured_std=meas, predicted_std=pred)
    print("saved", OUT / "pos0_pairs_shotnoise.npz")


if __name__ == "__main__":
    main()
