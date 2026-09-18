"""Edge artifact QC for the 260917 test timelapse.

Question: the far (non-tilt-fit) end of some trap crops does not come back to 0
and shows a ramp. Is that ramp static in time or does it fluctuate?

Shows the same channel over time in inferno 0-1.8 rad, its long-axis profile per
frame, and the far-end level vs time for flagged channels and a clean one.
"""
import os
import sys

import numpy as np
import tifffile
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_logger import save_figure  # noqa: E402  (applies paper.mplstyle on import)

ROOT = r"E:\260917\online_crop_sub_zstack_test_1"
FRAMES = [0, 3, 6, 9, 12, 14]
MAIN = (39, 2)                     # channel the ramp is strongest on
TRACES = [(39, 2), (1, 1), (42, 1), (20, 5)]   # last one is a clean reference
POS_SPLIT = 52


def crop_path(pos, ch, fr):
    return os.path.join(ROOT, f"Pos{pos}", "output_phase", "channels",
                        "crop_sub_rawraw", "z000", f"ch{ch:02d}",
                        f"img_{fr:09d}_ph_000.tif")


def load(pos, ch, fr):
    return tifffile.imread(crop_path(pos, ch, fr))


def far_end(pos, im):
    """Mean of the 20 px at the end opposite the tilt-fit side."""
    p = im.mean(axis=0)
    return p[-20:].mean() if pos < POS_SPLIT else p[:20].mean()


n_tr = len(FRAMES)
fig = plt.figure(figsize=(7.2, 6.2))
gs = fig.add_gridspec(3, n_tr, height_ratios=[1.0, 1.5, 1.5], hspace=0.55, wspace=0.12)

# --- row 1: the same channel at several frames, inferno 0-1.8 rad ---
mpos, mch = MAIN
ims = {fr: load(mpos, mch, fr) for fr in FRAMES}
for j, fr in enumerate(FRAMES):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(ims[fr], cmap="inferno", vmin=0, vmax=1.8, aspect="auto")
    ax.set_title(f"T={fr}", pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if j == 0:
        ax.set_ylabel(f"Pos{mpos} ch{mch:02d}")
fig.text(0.5, 0.955, f"Pos{mpos} ch{mch:02d}  inferno 0-1.8 rad "
         f"(tilt fit on the left third; far end = right)", ha="center")

# --- row 2: long-axis profile per frame ---
ax = fig.add_subplot(gs[1, :])
cmap = plt.get_cmap("cividis")
prof = {}
for i, fr in enumerate(FRAMES):
    p = ims[fr].mean(axis=0)
    prof[fr] = p
    ax.plot(np.arange(len(p)), p, color=cmap(i / (len(FRAMES) - 1)), lw=0.9,
            label=f"T={fr}")
ax.axhline(0, color="0.6", lw=0.5)
ax.set_xlabel("position along channel [px]  (0 = tilt-fit end)")
ax.set_ylabel("phase [rad]")
ax.legend(frameon=False, ncol=len(FRAMES), fontsize=5, loc="upper left")

# --- row 3: far-end level vs time ---
ax = fig.add_subplot(gs[2, :])
all_fr = list(range(15))
traces = {}
for pos, ch in TRACES:
    y = [far_end(pos, load(pos, ch, fr)) for fr in all_fr]
    traces[f"Pos{pos} ch{ch:02d}"] = y
    ax.plot(all_fr, y, marker="o", ms=2.5, lw=0.9, label=f"Pos{pos} ch{ch:02d}")
ax.axhline(0, color="0.6", lw=0.5)
ax.set_xlabel("timepoint T  (180 s apart)")
ax.set_ylabel("far-end mean [rad]")
ax.legend(frameon=False, fontsize=5)

save_figure(
    fig,
    params={"root": ROOT, "frames": FRAMES, "main": f"Pos{mpos}ch{mch:02d}",
            "traces": [f"Pos{p}ch{c:02d}" for p, c in TRACES],
            "vmin": 0, "vmax": 1.8, "far_end_px": 20, "pos_split": POS_SPLIT},
    caption=(
        "Far-end level of the online crop_sub_rawraw crops of the 260917 test timelapse "
        "(no cells loaded; 2% glucose; single z at grid focus index 5; 180 s interval). "
        "Operational definition: each crop is the drift-corrected, grid-subtracted, "
        "tilt-corrected 40 x 240 px trap crop written online by compute_drift_online; the "
        "profile is the mean over the 40 px short axis, and the far-end level is the mean of "
        "that profile over the 20 px at the end opposite the tilt-fit third (right end for "
        "Pos < 52, left end for Pos >= 52). Row 1: Pos39 ch02 at T=0,3,6,9,12,14 in inferno "
        "0-1.8 rad. Row 2: its long-axis profile per frame. Row 3: far-end level vs timepoint "
        "for three channels where the ramp was flagged by eye plus one clean channel "
        "(Pos20 ch05). n = 1 channel per trace, 15 timepoints, single experiment; no error "
        "bars and no statistical test -- these are single-frame values, not replicates."
    ),
    description=("260917 test timelapse: is the far-end ramp of crop_sub_rawraw static or "
                 "time-varying? Pos39 ch02 in inferno 0-1.8 rad over T=0..14, its long-axis "
                 "profile per frame, and far-end level vs time for three flagged channels "
                 "plus a clean one (Pos20 ch05)."),
    data={f"img_T{fr}": ims[fr] for fr in FRAMES}
    | {f"prof_T{fr}": prof[fr] for fr in FRAMES}
    | {k.replace(" ", "_") + "_far_end": np.array(v) for k, v in traces.items()},
)
for k, v in traces.items():
    v = np.array(v)
    print(f"{k}: far-end min {v.min():+.3f} max {v.max():+.3f} "
          f"ptp {np.ptp(v):.3f} std {v.std():.3f} rad")
