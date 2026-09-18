"""The same channel, frame by frame, in inferno. No statistics.

Rows are channels, columns are timepoints, last column is the mean of all 42 frames so the
fixed part can be compared with the individual frames by eye. 260917 test run, no cells, so
every tile should be flat if the background were perfectly removed.
"""
import os
import sys

import numpy as np
import tifffile
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from figure_logger import save_figure  # noqa: E402

CROP = r"E:\260917\online_crop_sub_zstack_test_1"
SUB = "z000_plane"
FRAMES = [14, 15, 16, 17, 18, 19, 20, 21]   # consecutive: frame-to-frame change is visible directly
ALL_FRAMES = list(range(42))
CHANNELS = [(39, 2), (39, 1), (5, 0), (41, 2), (1, 1), (79, 0), (20, 5), (64, 0)]
VLIM = 0.2


def load(pos, ch, t):
    f = os.path.join(CROP, f"Pos{pos}", "output_phase", "channels", "crop_sub_rawraw",
                     SUB, f"ch{ch:02d}", f"img_{t:09d}_ph_000.tif")
    return tifffile.imread(f).astype(np.float64) if os.path.exists(f) else None


rows = []
for pos, ch in CHANNELS:
    tiles = [load(pos, ch, t) for t in FRAMES]
    stack = [x for x in (load(pos, ch, t) for t in ALL_FRAMES) if x is not None]
    if not stack or any(x is None for x in tiles):
        print(f"Pos{pos} ch{ch:02d}: frames missing, skipped")
        continue
    rows.append(((pos, ch), tiles, np.mean(stack, axis=0), len(stack)))

ncol = len(FRAMES) + 1
fig, axes = plt.subplots(len(rows), ncol, figsize=(7.2, 0.62 * len(rows) + 0.5),
                         gridspec_kw={"wspace": 0.06, "hspace": 0.30})
for i, ((pos, ch), tiles, mean_img, n) in enumerate(rows):
    for j, img in enumerate(tiles + [mean_img]):
        ax = axes[i, j]
        im = ax.imshow(img, cmap="inferno", vmin=-VLIM, vmax=VLIM, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if i == 0:
            ax.set_title(f"T={FRAMES[j]}" if j < len(FRAMES) else f"mean of {n}",
                         fontsize=5.5, pad=2)
    axes[i, 0].set_ylabel(f"Pos{pos}\nch{ch:02d}", fontsize=5.5, rotation=0,
                          ha="right", va="center", labelpad=12)
cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
cb.set_label("phase [rad]", fontsize=5.5)
cb.ax.tick_params(labelsize=5)

save_figure(
    fig,
    params={"crop": CROP, "sub": SUB, "frames": FRAMES, "channels": [f"Pos{p}ch{c:02d}" for p, c in CHANNELS],
            "vlim": VLIM, "n_frames_mean": len(ALL_FRAMES)},
    caption=(
        "The same trap channels over time, inferno -0.2 to +0.2 rad. 260917 test run (no cells "
        "loaded; 2% glucose; single z at grid index 5; 180 s interval). Each tile is the "
        "grid-subtracted crop_sub_rawraw crop with a first-order 2D plane fitted on the "
        "aperture-end third removed (scripts/apply_plane_tilt_endthird.py), 40 x 240 px shown "
        "stretched to fill the tile; the long axis runs left to right with the tilt-fit end at "
        "the left for Pos < 52 and at the right for Pos >= 52. Columns are single timepoints "
        "T=0..41 and the last column is the pixelwise mean of all 42 frames of that channel. "
        "Rows: the first five are channels flagged by eye as keeping a far-end ramp, Pos79 ch00 "
        "is the channel whose flatten fell back to identity, and the last two are clean "
        "references. With no cells in the device, a perfectly removed background would leave "
        "every tile uniform. n = 1 experiment; single frames, no averaging within a tile except "
        "the last column, no error bars and no statistical test."
    ),
    description=("Time montage of the low-frequency residual in inferno: single frames next to the "
                 "42-frame mean, for flagged and clean channels of the 260917 test run."),
    data={f"Pos{p}ch{c:02d}_T{t}": img
          for ((p, c), tiles, _, _) in rows for t, img in zip(FRAMES, tiles)}
    | {f"Pos{p}ch{c:02d}_mean": m for ((p, c), _, m, _) in rows},
)
print(f"{len(rows)} channels x {len(FRAMES)} frames + mean")
