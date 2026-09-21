"""Does subtracting a few percent of the structure actually kill the fixed artifact?

Three tiles per channel, all inferno +-0.2 rad:
  measured     the fixed part of the residual (mean of 42 frames, plane-corrected crop)
  a x structure the grid reference crop scaled by the single best-fit factor a
  left over    measured - a x structure

If the middle tile looks like the left one and the right one goes flat, the artifact is the
same structure recorded a few percent differently. If the right tile still holds the pattern,
it is not.
"""
import json
import os
import sys

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from figure_logger import save_figure  # noqa: E402
from ecc_utils import extract_rect_roi  # noqa: E402
from apply_plane_tilt_endthird import plane_endthird  # noqa: E402

GRID = r"D:\AquisitionData\Kitagishi\260917\grid_hologram_0p05"
CROP = r"E:\260917\online_crop_sub_zstack_test_1"
TILT_H, OUT_H, SIGMA = 270, 240, 6
FRAMES = range(42)
CHANNELS = [(5, 0), (1, 1), (41, 2), (39, 2), (39, 1), (20, 5)]
VLIM = 0.2


def far_end(a, pos):
    p = a.mean(axis=0)
    return p[:20].mean() if pos >= 52 else p[-20:].mean()


rows = []
for pos, ch in CHANNELS:
    fit_right = pos >= 52
    roi = json.load(open(os.path.join(GRID, f"Pos{pos}_x+0_y+0", "output_phase", "channels",
                                      "channel_rois.json"), encoding="utf-8"))[ch]
    stack = []
    for t in FRAMES:
        f = os.path.join(CROP, f"Pos{pos}", "output_phase", "channels", "crop_sub_rawraw",
                         "z000_plane", f"ch{ch:02d}", f"img_{t:09d}_ph_000.tif")
        if os.path.exists(f):
            stack.append(gaussian_filter(tifffile.imread(f).astype(np.float64), SIGMA))
    measured = np.mean(stack, axis=0)

    g = tifffile.imread(os.path.join(GRID, f"Pos{pos}_x+0_y+0", "output_phase_raw",
                                     "img_000000000_ph_005_phase.tif")).astype(np.float64)
    big = extract_rect_roi(g, roi["cy"], roi["cx"], roi["crop_w"], TILT_H)
    s = (TILT_H - OUT_H) // 2
    structure = gaussian_filter(plane_endthird(big[:, s:s + OUT_H], fit_right), SIGMA)

    a = float(np.dot(structure.ravel(), measured.ravel()) / np.dot(structure.ravel(), structure.ravel()))
    model = a * structure
    left = measured - model
    rows.append(((pos, ch), measured, model, left, a,
                 far_end(measured, pos), far_end(left, pos), structure.std()))

print(f"{'Pos ch':10s} {'a [%]':>7s} {'structure rms':>14s} {'far-end measured':>17s} "
      f"{'far-end left over':>18s} {'rms measured':>13s} {'rms left over':>14s}")
for (pos, ch), meas, model, left, a, f0, f1, srms in rows:
    print(f"Pos{pos:<3d}ch{ch:02d} {a*100:+7.2f} {srms:14.2f} {f0:+17.3f} {f1:+18.3f} "
          f"{meas.std():13.4f} {left.std():14.4f}")

fig, axes = plt.subplots(len(rows), 3, figsize=(5.6, 0.80 * len(rows) + 0.6),
                         gridspec_kw={"wspace": 0.06, "hspace": 0.42})
titles = ["measured (mean of 42)", "a x structure", "left over"]
for i, ((pos, ch), meas, model, left, a, f0, f1, _) in enumerate(rows):
    for j, img in enumerate((meas, model, left)):
        ax = axes[i, j]
        im = ax.imshow(img, cmap="inferno", vmin=-VLIM, vmax=VLIM, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if i == 0:
            ax.set_title(titles[j], fontsize=6, pad=3)
    axes[i, 0].set_ylabel(f"Pos{pos} ch{ch:02d}\na={a*100:+.1f}%", fontsize=5.5, rotation=0,
                          ha="right", va="center", labelpad=16)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
cb.set_label("phase [rad]", fontsize=5.5)
cb.ax.tick_params(labelsize=5)

save_figure(
    fig,
    params={"grid": GRID, "crop": CROP, "sigma": SIGMA, "vlim": VLIM,
            "channels": [f"Pos{p}ch{c:02d}" for p, c in CHANNELS], "n_frames": len(list(FRAMES))},
    caption=(
        "Subtracting a few percent of the structure removes the fixed artifact. 260917 test run "
        "(no cells loaded; 2% glucose; single z at grid index 5; 180 s interval; 42 frames). "
        "Operational definitions: 'measured' is the pixelwise mean over all 42 frames of the "
        "grid-subtracted crop after removing a first-order plane fitted on the aperture-end third, "
        f"then blurred with a {SIGMA} px Gaussian; 'structure' is the grid reference crop of the "
        "same channel (z index 5, the working plane) put through the identical crop, plane removal "
        "and blur, i.e. the real trap phase the subtraction has to cancel; a is the single "
        "least-squares scale factor between the two, quoted per row in percent; 'left over' is "
        "measured minus a x structure. All tiles share an inferno scale of +-0.2 rad and are "
        "40 x 240 px stretched to fill the tile. Rows are ordered by |a|; Pos20 ch05 is a clean "
        "reference. n = 1 experiment, one channel per row, 42 frames behind each mean; no error "
        "bars and no statistical test."
    ),
    description=("Measured fixed residual, the best-fit few-percent copy of the grid structure, and "
                 "what is left after removing it, for five flagged channels and one clean one."),
    data={f"Pos{p}ch{c:02d}_{n}": v for ((p, c), meas, model, left, *_) in rows
          for n, v in (("measured", meas), ("model", model), ("left", left))}
    | {"scale_percent": np.array([r[4] * 100 for r in rows])},
)
