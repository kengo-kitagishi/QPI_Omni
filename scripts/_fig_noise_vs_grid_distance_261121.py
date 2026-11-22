"""_fig_noise_vs_grid_distance_261121.py

Figure: residue of the reference subtraction against the distance to the reference image.

Data: 261121 empty-chip reference grid (25 x 25 points per Pos at 0.03 um, one z, EMM 2% glucose, no
cells), reconstructed phase (output_phase_raw) of the Pos taken after the chip had settled (Pos62-123,
every 4th).  Two grid points of the same Pos play "frame" (A) and "reference" (B); their distance is
taken from the gaussian2d calibration of the grid (grid_calibration_PosN.json).

  no interpolation : A - B
  spline 5         : A - shift(B, calibrated offset, spline order 5)   (grid_subtract.WARP_SPLINE_ORDER)

Per channel crop (40 x 240 px, 0.3457 um/px), residue = std over a region of the difference minus its
gaussian(2 px) blur.  Regions from A: trap interior = phase < -1 rad relative to the bonded side, eroded
4 px; structure-free = bonded PDMS past the dead end of the trap, at least 8 px away from the trap.
Each residue is divided by the noise floor of its Pos = median structure-free residue of the pairs
closer than 0.04 px (equal to the shot-noise limit within 5 % on Pos0 pairs, see
_qc_shotnoise_source_261121.py).

    python scripts/_fig_noise_vs_grid_distance_261121.py
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from scipy.ndimage import shift as nd_shift

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from figure_logger import save_figure          # applies paper.mplstyle on import
import matplotlib.pyplot as plt
import grid_subtract as gs
from channel_crop import extract_rect_roi

GRID = r"D:\AquisitionData\Kitagishi\261121\recon_0p03um_25x25"
POS = list(range(62, 124, 4))
EDGES = [0, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
PAIRS_PER_BIN = 3
CHANNELS = range(2, 10)
PX_UM = 0.34567514677103717
EXAMPLE_POS, EXAMPLE_CH = 102, 5
EXAMPLE_D = [0.01, 0.05, 0.10, 0.20, 0.40]
C_FLAT, C_RAW, C_SPL = "#7f7f7f", "#D55E00", "#0072B2"       # Okabe-Ito: grey / vermillion / blue
CACHE = Path(os.environ.get("FIG_CACHE", "")) if os.environ.get("FIG_CACHE") else None
# scripts/paper.mplstyle is missing on the microscope PC; same settings, set explicitly
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none"})


def hp(a):
    return a - gaussian_filter(a, 2)


def load(P, key):
    return tifffile.imread(os.path.join(GRID, f"{P}_x{key[0]:+d}_y{key[1]:+d}", "output_phase_raw",
                                        "img_000000000_ph_000_phase.tif")).astype(np.float64)


def crops(A, B, roi, ci, cj):
    """A crop, B crop at the same pixels, and B shifted onto A with spline 5 (bigger window, then cut)."""
    cy, cx = int(round(roi["cy"] + ci[1])), int(round(roi["cx"] + ci[0]))
    a = extract_rect_roi(A, cy, cx, 40, 240)
    b = extract_rect_roi(B, cy, cx, 40, 240)
    big = extract_rect_roi(B, cy, cx, 60, 280)
    bw = nd_shift(big, (ci[1] - cj[1], ci[0] - cj[0]), order=gs.WARP_SPLINE_ORDER, mode="nearest")[10:-10, 20:-20]
    return a, b, bw


def regions(a):
    g = a - np.median(a[:, 150:230])            # Pos >= 61: trap on the left, bonded PDMS on the right
    trap = g < -1.0
    inside = binary_erosion(trap, iterations=4)
    flat = np.zeros_like(trap)
    flat[4:-4, 150:232] = True
    flat &= ~binary_dilation(trap, iterations=8)
    return inside, flat


def one_pos(n):
    P = f"Pos{n}"
    cal = gs.load_grid_calibration(os.path.join(GRID, f"grid_calibration_{P}.json"))
    rois = json.load(open(os.path.join(GRID, f"{P}_x+0_y+0", "output_phase", "channels", "channel_rois.json")))
    keys = list(cal)
    xy = np.array([cal[k] for k in keys])
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, 9)
    rng = np.random.default_rng(n)
    rows, imgs = [], {}
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        pr = np.argwhere((d >= lo) & (d < hi))
        pr = pr[pr[:, 0] < pr[:, 1]]
        if len(pr) == 0:
            continue
        for i, j in pr[rng.choice(len(pr), min(PAIRS_PER_BIN, len(pr)), replace=False)]:
            A, B = load(P, keys[i]), load(P, keys[j])
            for c in CHANNELS:
                if c >= len(rois) - 1:
                    continue
                a, b, bw = crops(A, B, rois[c], cal[keys[i]], cal[keys[j]])
                inside, flat = regions(a)
                if inside.sum() < 200 or flat.sum() < 500:
                    continue
                d0, d5 = hp(a - b), hp(a - bw)
                rows.append((n, c, d[i, j], d0[flat].std(), d0[inside].std(), d5[flat].std(), d5[inside].std()))
    if n == EXAMPLE_POS:
        for target in EXAMPLE_D:
            ddx = np.abs(xy[:, None, 0] - xy[None, :, 0])
            ddy = np.abs(xy[:, None, 1] - xy[None, :, 1])
            cost = np.abs(d - target) / max(target, 0.02) + np.abs(ddx - ddy) / np.maximum(d, 1e-6) + np.tril(np.full_like(d, 99))
            i, j = np.unravel_index(np.argmin(cost), d.shape)
            a, b, bw = crops(load(P, keys[i]), load(P, keys[j]), rois[EXAMPLE_CH], cal[keys[i]], cal[keys[j]])
            show = lambda x: x - gaussian_filter(x, 8)
            imgs[target] = (float(d[i, j]), show(a - b), show(a - bw))
    return rows, imgs


def main():
    if CACHE and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        R, imgs = z["R"], z["imgs"].item()
    else:
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(one_pos, POS))
        R = np.array([r for rows, _ in res for r in rows])
        imgs = next(im for _, im in res if im)
        if CACHE:
            np.savez(CACHE, R=R, imgs=np.array(imgs, dtype=object))
    # noise floor per Pos: structure-free residue of the closest pairs (no interpolation)
    floor = {n: np.median(R[(R[:, 0] == n) & (R[:, 2] < 0.04), 3]) for n in np.unique(R[:, 0])}
    f = np.array([floor[n] for n in R[:, 0]])
    fmed = float(np.median(list(floor.values())))
    ratio = {"flat_none": R[:, 3] / f, "trap_none": R[:, 4] / f, "flat_spline5": R[:, 5] / f, "trap_spline5": R[:, 6] / f}
    centre, stats = [], {k: [] for k in ratio}
    nbin = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (R[:, 2] >= lo) & (R[:, 2] < hi)
        centre.append(np.median(R[m, 2]))
        nbin.append(int(m.sum()))
        for k in ratio:
            stats[k].append(np.percentile(ratio[k][m], [25, 50, 75]))
    centre = np.array(centre)
    stats = {k: np.array(v) for k, v in stats.items()}
    print(f"{len(R)} channel crops, {len(floor)} Pos; noise floor median {fmed:.4f} rad")
    print("distance [px] n | flat none | trap none | flat spline5 | trap spline5   (median ratio to the noise floor)")
    for i, c in enumerate(centre):
        print(f"  {c:.3f} {nbin[i]:4d} | " + " | ".join(f"{stats[k][i,1]:5.2f}" for k in ratio))

    vlim = 0.1
    fig = plt.figure(figsize=(183 / 25.4, 80 / 25.4))
    left = fig.add_gridspec(len(EXAMPLE_D), 1, left=0.085, right=0.50, top=0.90, bottom=0.17, hspace=0.18)
    for r, t in enumerate(EXAMPLE_D):
        dist, _, d5 = imgs[t]
        ax = fig.add_subplot(left[r, 0])
        h = ax.imshow(d5, cmap="RdBu_r", vmin=-vlim, vmax=vlim, interpolation="nearest", aspect="auto", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_ylabel(f"{dist:.2f} px", rotation=0, ha="right", va="center")
        if r == len(EXAMPLE_D) - 1:
            ax.plot([150, 150 + 10 / PX_UM], [35, 35], color="k", lw=1.5, solid_capstyle="butt")
            ax.text(150 + 5 / PX_UM, 31, "10 µm", ha="center", va="bottom", fontsize=6)
    fig.text(0.076, 0.925, "distance to the reference", ha="left", va="center", fontsize=7)
    cax = fig.add_axes([0.17, 0.085, 0.24, 0.022])
    cb = fig.colorbar(h, cax=cax, orientation="horizontal", ticks=[-vlim, 0, vlim])
    cb.set_label("phase difference [rad]", labelpad=1)
    cb.outline.set_linewidth(0.5)
    fig.text(0.012, 0.95, "a", fontweight="bold", fontsize=9)
    fig.text(0.555, 0.95, "b", fontweight="bold", fontsize=9)

    ax = fig.add_axes([0.635, 0.17, 0.285, 0.66])
    ax.axhspan(0.95, 1.05, color="k", alpha=0.12, lw=0)
    ax.axhline(1, color="k", lw=0.6)
    for k, col, ls, lab in (("trap_spline5", C_SPL, "-", "trap"), ("flat_spline5", C_FLAT, "--", "structure-free region")):
        ax.fill_between(centre, stats[k][:, 0], stats[k][:, 2], color=col, alpha=0.18, lw=0)
        ax.plot(centre, stats[k][:, 1], ls, color=col, marker="o", ms=2.2, lw=1, label=lab)
    ax.set_ylim(0.5, 6)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("distance to the reference image [px]")
    ax.set_ylabel("residue / shot-noise floor")
    ax.text(0.49, 0.9, "shot-noise floor", ha="right", va="top", fontsize=6)
    ax.legend(frameon=False, loc="upper left", handlelength=2.2, borderaxespad=0.2)
    sec = ax.secondary_xaxis("top", functions=(lambda x: x * PX_UM * 1e3, lambda x: x / (PX_UM * 1e3)))
    sec.set_xlabel("[nm]", labelpad=2)
    rax = ax.secondary_yaxis("right", functions=(lambda y: y * fmed * 1e3, lambda y: y / (fmed * 1e3)))
    rax.set_ylabel("residue [mrad]")

    caption = (
        "Residue of the reference subtraction grows with the distance between the frame and its reference image. "
        "(a) Difference of two phase images of the same empty trap taken at stage positions the stated distance apart "
        f"(Pos{EXAMPLE_POS}, channel {EXAMPLE_CH}; 40 x 240 px crop; colour scale +-{vlim} rad). The second image is shifted onto the "
        "first by the calibrated offset with a 5th-order spline before subtraction, as in the pipeline. For display only, a "
        "Gaussian blur (sigma 8 px) of each difference is subtracted. Pairs displaced along the image diagonal; "
        "representative single pairs. "
        "(b) Residue against distance. Residue = standard deviation, over the region, of the difference image (after the "
        "spline-5 shift) minus its Gaussian blur (sigma 2 px); divided by the noise floor of the position = median residue "
        "in the structure-free region of the pairs closer than 0.04 px, subtracted without a shift. Trap = pixels more than "
        "1 rad below the bonded side, eroded by 4 px; structure-free = bonded PDMS beyond the closed end of the trap, at "
        "least 8 px from it. Distance = separation of the two images from the gaussian2d grid calibration (0.3457 um/px). "
        "Lines: median over channel crops; bands: 25th-75th percentile. Blue: trap; grey dashed: structure-free region. "
        f"Grey band at 1: +-5%. Right axis: the same in mrad (floor {fmed*1e3:.1f} mrad). "
        f"n = {len(R)} channel crops ({len(floor)} positions x up to {PAIRS_PER_BIN} pairs per distance bin x 8 channels; "
        f"per bin {min(nbin)}-{max(nbin)}). No statistical test. "
        "Sample: PDMS trap device without cells, EMM 2% glucose, 1 mL/h, off-axis QPI at 658 nm, 40x NA 0.95, single z; "
        "reference grid 25 x 25 points at 0.03 um per position, acquired 2026-11-22 00:00-07:06 (positions 62-123). "
        "The noise floor equals the shot-noise limit within 5% on background pairs (see _qc_shotnoise_source_261121.py)."
    )
    data = {"pos": R[:, 0], "ch": R[:, 1], "distance_px": R[:, 2], "flat_none_rad": R[:, 3], "trap_none_rad": R[:, 4],
            "flat_spline5_rad": R[:, 5], "trap_spline5_rad": R[:, 6], "noise_floor_rad": f,
            "bin_centre_px": centre, "bin_n": np.array(nbin)}
    for k in stats:
        data[f"{k}_q25_q50_q75"] = stats[k]
    for t in EXAMPLE_D:
        data[f"example_{t:.2f}_none"] = imgs[t][1]
        data[f"example_{t:.2f}_spline5"] = imgs[t][2]
    out = save_figure(fig, params={"noise_floor_rad": fmed, "grid": GRID, "pos": POS, "edges": EDGES, "pairs_per_bin": PAIRS_PER_BIN, "hp_sigma_px": 2,
                                   "spline_order": gs.WARP_SPLINE_ORDER, "example": [EXAMPLE_POS, EXAMPLE_CH]},
                      description="Residue of the reference subtraction vs distance to the reference image, with and without spline-5 interpolation",
                      data=data, caption=caption, fmt="pdf")
    fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main()
