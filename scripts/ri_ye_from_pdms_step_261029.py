"""YE refractive index from the PDMS-wall / trap phase step, 261029 chip (no MilliQ/EtOH on this chip).

Across a trap the light goes through medium instead of PDMS, so the trap is a dip below the walls:
    S = integral over the trap cross-section of (wall baseline - phase) = k * d * w * (n_PDMS - n_medium)
In the same trap with the same optics, depth d and width w cancel in the ratio of 2% EMM (grid) and YE:
    n_YE = n_PDMS - (n_PDMS - n_2per) * S_YE / S_2per
The integral, not the dip depth, is used because the reconstruction blurs the dip but keeps its area.

Inputs (output_phase_raw, i.e. before the Pos0 BG subtraction that would remove the trap structure):
    2% EMM : E:/261029/recon_2per_grid_1/PosN_x+0_y+0/output_phase_raw   (grid centre point)
    YE     : D:/.../261029/ph_YE_5min_31d/PosN/z000/output_phase_raw    (frames 0-27, YE flowing)
Traps: the 261030 whitelist (empty traps, same chip and Pos), edge traps excluded.
Per trap and column: baseline = line fitted to the wall rows on both sides; S = sum(baseline - phase) over the
trap rows; S_trap = median over the central columns. YE: median over frames.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from figure_logger import save_figure  # noqa: E402  (applies paper.mplstyle)
import matplotlib.pyplot as plt  # noqa: E402

GRID = Path(r"E:\261029\recon_2per_grid_1")
YE = Path(r"D:\AquisitionData\Kitagishi\261029\ph_YE_5min_31d")
WHITELIST = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261030_YE\channel_whitelist.json")
N_2PER = 1.335029            # datasets/ri_calibration_2perEMM_from_260517.json (not measured on this chip)
N_PDMS = 1.412               # Sylgard 184 near 658 nm (literature, not measured)
N_PDMS_RANGE = (1.408, 1.416)
EDGE = {0, 11}
HALF_TRAP = 13               # rows cy-13..cy+13 hold the trap dip
WALL = (16, 20)              # rows cy+-16..20 are wall on each side (traps are ~42 px apart)
END_MARGIN = 40              # skip this many columns at each end of the trap ROI


def trap_integral(img, roi):
    cy, cx, L = roi["cy"], roi["cx"], roi["crop_h"]
    y0, y1 = cy - WALL[1], cy + WALL[1] + 1
    x0, x1 = cx - L // 2 + END_MARGIN, min(cx + L // 2 - END_MARGIN, img.shape[1])
    if y0 < 0 or y1 > img.shape[0] or x1 - x0 < 20:
        return np.nan
    block = img[y0:y1, x0:x1].astype(np.float64)
    rows = np.arange(y0, y1) - cy
    wall = (np.abs(rows) >= WALL[0]) & (np.abs(rows) <= WALL[1])
    trap = np.abs(rows) <= HALF_TRAP
    A = np.c_[rows[wall], np.ones(wall.sum())]
    coef, *_ = np.linalg.lstsq(A, block[wall], rcond=None)          # one line per column
    base = np.c_[rows[trap], np.ones(trap.sum())] @ coef
    return float(np.median((base - block[trap]).sum(axis=0)))


YE_GRID = Path(r"D:\AquisitionData\Kitagishi\261029\recon_YE_grid")   # YE grid, same protocol as the 2% grid


def ye_files(pos):
    if "--ye-grid" in sys.argv:
        return sorted((YE_GRID / f"{pos}_x+0_y+0" / "output_phase_raw").glob("img_*_phase.tif"))
    return sorted((YE / pos / "z000" / "output_phase_raw").glob("img_*_phase.tif"))


def main():
    wl = json.loads(WHITELIST.read_text(encoding="utf-8"))["whitelist"]
    rec = []
    for pos, chs in sorted(wl.items(), key=lambda kv: int(kv[0][3:])):
        gdir = GRID / f"{pos}_x+0_y+0"
        rois = json.loads((gdir / "output_phase" / "channels" / "channel_rois.json").read_text(encoding="utf-8"))
        g_files = sorted((gdir / "output_phase_raw").glob("img_*_phase.tif"))
        y_files = ye_files(pos)
        if not g_files or not y_files:
            continue
        g_img = tifffile.imread(str(g_files[0]))
        y_imgs = [tifffile.imread(str(f)) for f in y_files]
        # the YE grid was taken a day later after the chip moved: use its own trap ROIs (same trap count required)
        rois_y = rois
        if "--ye-grid" in sys.argv:
            rois_y = json.loads((YE_GRID / f"{pos}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json")
                                .read_text(encoding="utf-8"))
            if len(rois_y) != len(rois):
                continue
        for ch in chs:
            if ch in EDGE or ch >= len(rois):
                continue
            s2 = trap_integral(g_img, rois[ch])
            sy = np.nanmedian([trap_integral(im, rois_y[ch]) for im in y_imgs])
            if np.isfinite(s2) and np.isfinite(sy) and s2 > 0:
                rec.append((int(pos[3:]), ch, s2, sy))
    a = np.array(rec)
    pos, ch, s2, sy = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    ratio = sy / s2
    med = float(np.median(ratio))
    q1, q3 = np.percentile(ratio, [25, 75])
    mean, sem = float(np.mean(ratio)), float(np.std(ratio, ddof=1) / np.sqrt(ratio.size))   # no bootstrap (user)

    def n_ye(r, n_p=N_PDMS):
        return n_p - (n_p - N_2PER) * r
    res = {
        "n_traps": int(ratio.size), "n_pos": int(np.unique(pos).size),
        "ratio_median": med, "ratio_iqr": [float(q1), float(q3)], "ratio_mean": mean, "ratio_sem": sem,
        "S_2per_median_rad_px": float(np.median(s2)), "S_YE_median_rad_px": float(np.median(sy)),
        "n_YE": n_ye(med), "n_YE_from_mean": n_ye(mean),
        "n_YE_sem": float((N_PDMS - N_2PER) * sem),
        "n_YE_for_n_PDMS_range": [n_ye(med, N_PDMS_RANGE[0]), n_ye(med, N_PDMS_RANGE[1])],
        "n_YE_minus_n_2per": n_ye(med) - N_2PER, "n_2per": N_2PER, "n_PDMS": N_PDMS,
    }
    for k, v in res.items():
        print(f"{k:24s} {v}")

    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 65 / 25.4), constrained_layout=True)
    axes[0].scatter(s2, sy, s=4, alpha=0.5, color="#0072B2", rasterized=True)
    lim = [0, float(max(s2.max(), sy.max())) * 1.05]
    axes[0].plot(lim, lim, ls="--", lw=0.7, color="0.5")
    axes[0].set_xlabel("trap step integral, 2% EMM grid (rad px)")
    axes[0].set_ylabel("trap step integral, YE (rad px)")
    axes[0].set_title(f"n = {ratio.size} empty traps, {res['n_pos']} Pos")
    axes[1].hist(ratio, bins=60, color="#0072B2")
    axes[1].axvline(med, color="k", lw=0.8)
    axes[1].set_xlabel("S_YE / S_2per")
    axes[1].set_ylabel("traps")
    axes[1].set_title(f"median {med:.4f} -> n_YE = {res['n_YE']:.5f}")
    save_figure(
        fig,
        params={"n_2per": N_2PER, "n_PDMS": N_PDMS, "half_trap": HALF_TRAP, "wall": WALL, "end_margin": END_MARGIN},
        description="YE refractive index from the PDMS-wall/trap phase step ratio (261029 chip), empty traps.",
        data={"pos": pos, "ch": ch, "S_2per": s2, "S_YE": sy, "ratio": ratio},
        extra_meta=res,
        caption=("Left: per-trap integral of the phase dip below the PDMS walls (wall baseline fitted per column, "
                 "summed over the trap rows, median over the central columns) in the 2% EMM grid (x+0_y+0, "
                 "output_phase_raw) vs the YE timelapse (frames 0-27, median). Right: ratio S_YE/S_2per; "
                 "n_YE = n_PDMS - (n_PDMS - n_2per) x median ratio with n_PDMS = 1.412 (literature) and "
                 "n_2per = 1.335029 (260517 calibration). Traps: 261030 whitelist (empty), edge traps excluded; "
                 "Spread: interquartile range of the ratio over traps, and mean +- SEM over traps (n = traps). "
                 "No statistical test."),
    )


if __name__ == "__main__":
    main()
