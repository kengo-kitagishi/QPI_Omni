"""What is different about the channels flagged by eye?

For every channel of the six Pos that carry a flagged channel, measure over T=0..14:
  far_res_std   std of the far-end level of the subtracted crop (the artifact itself)
  slope_std     std of the slope fitted on the tilt-fit third of the timelapse frame
  fit_std       structure inside the tilt-fit third (mean over T of its residual std)
  raw_far       how deep the real structure at the far end is
  slope_gap     fitted local slope minus the whole-frame plane slope
Flagged channels are compared with the other channels of the same Pos, so the Pos-level
conditions (stage, drift, grid) are held fixed and only the channel differs.
"""
import json
import os
import sys

import numpy as np
import tifffile
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from figure_logger import save_figure  # noqa: E402
from ecc_utils import extract_rect_roi  # noqa: E402

CFG = json.load(open(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config_zstack.json",
                     encoding="utf-8"))
CROP_SUB = r"E:\260917\online_crop_sub_zstack_test_1"
FLAGGED = {(1, 0), (1, 1), (5, 0), (5, 1), (18, 11), (39, 1), (39, 2), (39, 3),
           (41, 1), (41, 2), (42, 0), (42, 1), (42, 2)}
POS_LIST = sorted({p for p, _ in FLAGGED})
FRAMES = list(range(15))
TILT_H = CFG.get("tilt_crop_h_raw", 270)
OUT_H = CFG.get("crop_sub_output_crop_h") or TILT_H
GRID_DIR = CFG["grid_dir"]


def plane_slope_x(img):
    """Slope of the whole-frame plane along the channel long axis [rad/px]."""
    h, w = img.shape
    y, x = np.mgrid[:h, :w]
    A = np.c_[x.ravel(), y.ravel(), np.ones(x.size)]
    c, *_ = np.linalg.lstsq(A, img.ravel(), rcond=None)
    return c[0]


rows = []
for pos in POS_LIST:
    rois = json.load(open(os.path.join(GRID_DIR, f"Pos{pos}_x+0_y+0", "output_phase",
                                       "channels", "channel_rois.json"), encoding="utf-8"))
    fit_right = pos >= int(CFG["pos_split"])
    tl = {}
    for t in FRAMES:
        f = os.path.join(CFG["save_dir"], f"Pos{pos}", "z000", "output_phase_raw",
                         f"img_{t:09d}_ph_000_phase.tif")
        tl[t] = tifffile.imread(f).astype(np.float64)
    gslope = {t: plane_slope_x(tl[t]) for t in FRAMES}

    for ch, roi in enumerate(rois):
        sub_dir = os.path.join(CROP_SUB, f"Pos{pos}", "output_phase", "channels",
                               "crop_sub_rawraw", "z000", f"ch{ch:02d}")
        if not os.path.isdir(sub_dir):
            continue
        far_res, slopes, fit_stds, raw_fars, gaps = [], [], [], [], []
        for t in FRAMES:
            sf = os.path.join(sub_dir, f"img_{t:09d}_ph_000.tif")
            if not os.path.exists(sf):
                continue
            p = tifffile.imread(sf).mean(axis=0)
            far_res.append(p[:20].mean() if fit_right else p[-20:].mean())

            big = extract_rect_roi(tl[t], roi["cy"], roi["cx"], roi["crop_w"], TILT_H)
            prof = big.mean(axis=0)
            x = np.arange(TILT_H, dtype=float)
            n = max(1, TILT_H // 3)
            xs, ys = (x[-n:], prof[-n:]) if fit_right else (x[:n], prof[:n])
            a, b = np.polyfit(xs, ys, 1)
            slopes.append(a)
            fit_stds.append(np.std(ys - (a * xs + b)))
            raw_fars.append(prof[:20].mean() if fit_right else prof[-20:].mean())
            gaps.append(a - gslope[t])
        if len(far_res) < 10:
            continue
        rows.append({
            "pos": pos, "ch": ch, "flagged": (pos, ch) in FLAGGED,
            "far_res_std": float(np.std(far_res)),
            "far_res_absmax": float(np.max(np.abs(far_res))),
            "slope_std_mrad": float(np.std(slopes) * 1e3),
            "fit_std": float(np.mean(fit_stds)),
            "raw_far": float(np.mean(raw_fars)),
            "slope_gap_mrad": float(np.mean(gaps) * 1e3),
        })

F = [r for r in rows if r["flagged"]]
C = [r for r in rows if not r["flagged"]]
keys = ["far_res_std", "far_res_absmax", "slope_std_mrad", "fit_std", "raw_far", "slope_gap_mrad"]
print(f"flagged n={len(F)}   other n={len(C)}   (same 6 Pos)")
print(f"{'metric':16s} {'flagged median':>15s} {'other median':>14s}")
for k in keys:
    print(f"{k:16s} {np.median([r[k] for r in F]):15.3f} {np.median([r[k] for r in C]):14.3f}")

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
pairs = [("slope_std_mrad", "fit slope jitter [mrad/px]"),
         ("fit_std", "structure in fit third [rad]"),
         ("raw_far", "raw far-end level [rad]")]
for ax, (k, lab) in zip(axes, pairs):
    ax.scatter([r[k] for r in C], [r["far_res_std"] for r in C], s=9, alpha=0.7,
               color="0.55", label="other ch")
    ax.scatter([r[k] for r in F], [r["far_res_std"] for r in F], s=16, alpha=0.9,
               color="#D55E00", label="flagged by eye")
    ax.set_xlabel(lab)
    ax.set_ylabel("far-end residual std [rad]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
r = lambda k: np.corrcoef([x[k] for x in rows], [x["far_res_std"] for x in rows])[0, 1]
axes[0].set_title(f"r = {r('slope_std_mrad'):+.2f}", pad=3)
axes[1].set_title(f"r = {r('fit_std'):+.2f}", pad=3)
axes[2].set_title(f"r = {r('raw_far'):+.2f}", pad=3)
axes[0].legend(frameon=False, fontsize=5)
fig.tight_layout()

save_figure(
    fig,
    params={"pos_list": POS_LIST, "frames": FRAMES, "tilt_h": TILT_H, "out_h": OUT_H,
            "flagged": sorted(f"Pos{p}ch{c:02d}" for p, c in FLAGGED)},
    caption=(
        "What separates the channels flagged by eye from the clean ones, 260917 test timelapse "
        "(no cells; 2% glucose; single z at grid index 5; 180 s interval; the six Pos that "
        "contain a flagged channel, all 12 channels each). Operational definitions, all measured "
        "over T=0..14: 'far-end residual std' is the std over frames of the mean of the 20 px at "
        "the end of the subtracted crop_sub_rawraw crop opposite the tilt-fit third; 'fit slope "
        "jitter' is the std over frames of the slope of a straight line fitted to the mean "
        "profile of the tilt-fit third of the reconstructed timelapse frame; 'structure in fit "
        "third' is the mean over frames of the residual std of that same fit; 'raw far-end level' "
        "is the mean over frames of the uncorrected profile at the far end, i.e. how deep the "
        "real structure there is. Orange = the 13 channels the user picked out by eye, grey = the "
        "other 59 channels of the same Pos. Each point is one channel (n = 15 frames behind each "
        "point); r is Pearson correlation over all 72 channels. Single experiment, no error bars "
        "and no statistical test."
    ),
    description=("Flagged vs clean channels of the same Pos: does fit-slope jitter, structure in "
                 "the tilt-fit third, or the depth of the real far-end structure separate them?"),
    data={k: np.array([r[k] for r in rows]) for k in keys}
    | {"pos": np.array([r["pos"] for r in rows]), "ch": np.array([r["ch"] for r in rows]),
       "flagged": np.array([r["flagged"] for r in rows])},
)

print("\nper-Pos detail (flagged marked *)")
for pos in POS_LIST:
    print(f"Pos{pos}")
    for rr in [x for x in rows if x["pos"] == pos]:
        mark = "*" if rr["flagged"] else " "
        print(f"  {mark} ch{rr['ch']:02d}  res_std {rr['far_res_std']:.3f}  "
              f"slope_jit {rr['slope_std_mrad']:6.2f}  fit_std {rr['fit_std']:.3f}  "
              f"raw_far {rr['raw_far']:+8.2f}  slope_gap {rr['slope_gap_mrad']:+7.2f}")
