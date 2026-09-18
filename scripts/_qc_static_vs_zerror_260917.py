"""Is the low-frequency residual static, and is it a z error baked in at grid time?

Two questions, two measurements, on the 260917 test run (no cells, so everything in a crop
is background and the truth is zero).

1. Static or fluctuating?
   Per channel, stack the plane-corrected crops of every frame. The time MEAN is the part
   that never moves; the per-pixel std over time is the part that does. Their ratio says
   which dominates.

2. If it is static, is it a z mismatch?
   The grid holds 11 z planes 0.4 um apart, so the phase derivative along z is measurable:
   dphi/dz = (z6 - z4) / (2 * 0.4 um), taken through the same crop and plane correction.
   A crop-time / grid-time focus difference of dz shows up as dz * dphi/dz. Fitting
   static_residual = a * dphi/dz gives a in um -- the effective z offset -- and R^2 says how
   much of the static pattern that explains. A high R^2 with a consistent sign across
   channels is the signature of a real focus offset; a low R^2 rules it out.
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

CFG = json.load(open(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config_zstack.json",
                     encoding="utf-8"))
CROP_SUB = r"E:\260917\online_crop_sub_zstack_test_1"
LOG = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log_zstack.json"
FRAMES = list(range(42))
Z_STEP_UM = 0.4
GRID_Z = 5          # the plane the timelapse sits on
TILT_H = CFG.get("tilt_crop_h_raw", 270)
OUT_H = CFG.get("crop_sub_output_crop_h") or TILT_H
LOWPASS_SIGMA_PX = 6      # blur before splitting static vs dynamic
CHANNELS = [(39, 2), (39, 1), (1, 1), (5, 0), (41, 2), (42, 1), (20, 5), (25, 6), (64, 0), (79, 0)]


def plane_crop(img, roi, fit_right):
    """Same crop + end-third plane the sheets show, from a full reconstructed frame."""
    big = extract_rect_roi(img, roi["cy"], roi["cx"], roi["crop_w"], TILT_H)
    start = (TILT_H - OUT_H) // 2
    return plane_endthird(big[:, start:start + OUT_H], fit_right)


log = [e for e in json.load(open(LOG, encoding="utf-8")) if e.get("per_pos")]
rows = []
store = {}
for pos, ch in CHANNELS:
    fit_right = pos >= int(CFG["pos_split"])
    rois = json.load(open(os.path.join(CFG["grid_dir"], f"Pos{pos}_x+0_y+0", "output_phase",
                                       "channels", "channel_rois.json"), encoding="utf-8"))
    roi = rois[ch]

    # --- 1. the measured residual, frame by frame (from the written crops) ---
    stack = []
    for t in FRAMES:
        f = os.path.join(CROP_SUB, f"Pos{pos}", "output_phase", "channels",
                         "crop_sub_rawraw", "z000_plane", f"ch{ch:02d}",
                         f"img_{t:09d}_ph_000.tif")
        if os.path.exists(f):
            stack.append(tifffile.imread(f).astype(np.float64))
    if len(stack) < 10:
        print(f"Pos{pos} ch{ch:02d}: only {len(stack)} frames, skipped")
        continue
    stack = np.array(stack)
    # the question is about the LOW-FREQUENCY pattern, so blur each frame first: per-pixel
    # noise would otherwise land entirely in the "dynamic" term and hide a static pattern.
    lowpass = np.array([gaussian_filter(f, LOWPASS_SIGMA_PX) for f in stack])
    static = lowpass.mean(axis=0)
    dynamic = lowpass.std(axis=0)
    px_noise = float(np.mean([np.std(f - g) for f, g in zip(stack, lowpass)]))

    # --- 2. the z derivative of the grid reference, same crop and plane ---
    # the grid point actually used at the last frame (they differ slightly frame to frame)
    q = [p for p in log[FRAMES[-1]]["positions"] if p["pos_label"] == f"Pos{pos}"]
    cd = [c for c in q[0]["channel_details"] if c["ch"] == ch] if q else []
    xi = cd[0].get("xi3", cd[0].get("xi")) if cd else 0
    yi = cd[0].get("yi3", cd[0].get("yi")) if cd else 0
    if xi is None or yi is None:
        xi = yi = 0
    gdir = os.path.join(CFG["grid_dir"], f"Pos{pos}_x{xi:+d}_y{yi:+d}", "output_phase_raw")
    zs = {}
    for dz in (-1, 0, 1):
        f = os.path.join(gdir, f"img_000000000_ph_{GRID_Z + dz:03d}_phase.tif")
        if os.path.exists(f):
            zs[dz] = plane_crop(tifffile.imread(f).astype(np.float64), roi, fit_right)
    if len(zs) < 3:
        print(f"Pos{pos} ch{ch:02d}: grid z neighbours missing, skipped")
        continue
    dphi_dz = (zs[1] - zs[-1]) / (2 * Z_STEP_UM)      # rad per um

    # --- fit static = a * dphi_dz ---
    x = dphi_dz.ravel()
    y = static.ravel()
    a = float(np.dot(x, y) / np.dot(x, x))
    r2 = float(1 - np.sum((y - a * x) ** 2) / np.sum((y - y.mean()) ** 2))
    rows.append({
        "pos": pos, "ch": ch, "n_frames": len(stack),
        "static_rms": float(static.std()),
        "dynamic_rms": float(dynamic.mean()),
        "ratio": float(static.std() / dynamic.mean()),
        "dz_um": a, "r2": r2,
        "dphi_dz_rms": float(dphi_dz.std()),
        "px_noise": px_noise,
    })
    store[(pos, ch)] = (static, dynamic, dphi_dz)

print(f"{'Pos ch':10s} {'frames':>6s} {'static rms':>11s} {'dynamic rms':>12s} "
      f"{'static/dyn':>11s} {'px noise':>9s} {'fitted dz [um]':>15s} {'R2':>7s}")
for r in rows:
    print(f"Pos{r['pos']:<3d}ch{r['ch']:02d} {r['n_frames']:6d} {r['static_rms']:11.4f} "
          f"{r['dynamic_rms']:12.4f} {r['ratio']:11.2f} {r['px_noise']:9.4f} "
          f"{r['dz_um']:15.3f} {r['r2']:7.2f}")

# ---------------- figure ----------------
key = (39, 2) if (39, 2) in store else list(store)[0]
static, dynamic, dphi_dz = store[key]
fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3], hspace=0.55, wspace=0.35)

for j, (img, title, vlim) in enumerate([
        (static, f"static (mean of {rows[0]['n_frames']} frames)", 0.2),
        (dynamic, "dynamic (std over frames)", 0.2),
        (dphi_dz, "grid dphi/dz [rad/um]", None)]):
    ax = fig.add_subplot(gs[0, j])
    v = vlim if vlim else float(np.abs(img).max())
    im = ax.imshow(img, cmap="inferno", vmin=-v if vlim else -v, vmax=v, aspect="auto")
    ax.set_title(f"Pos{key[0]} ch{key[1]:02d}\n{title}", pad=3, fontsize=6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.05)

ax = fig.add_subplot(gs[1, 0])
ax.scatter([r["dynamic_rms"] for r in rows], [r["static_rms"] for r in rows], s=14,
           color="#0072B2")
lim = max(max(r["static_rms"] for r in rows), max(r["dynamic_rms"] for r in rows)) * 1.1
ax.plot([0, lim], [0, lim], color="0.6", lw=0.6, ls="--")
ax.set_xlabel("dynamic rms [rad]")
ax.set_ylabel("static rms [rad]")
ax.set_title("above the line = mostly static", fontsize=6, pad=3)

ax = fig.add_subplot(gs[1, 1])
ax.scatter([r["dz_um"] for r in rows], [r["r2"] for r in rows], s=14, color="#D55E00")
ax.axvline(0, color="0.6", lw=0.6)
ax.set_xlabel("fitted z offset [um]")
ax.set_ylabel("R$^2$ of static = dz x dphi/dz")
ax.set_ylim(-0.05, 1.0)

ax = fig.add_subplot(gs[1, 2])
ax.scatter(dphi_dz.ravel(), static.ravel(), s=0.5, alpha=0.2, color="0.4", rasterized=True)
xx = np.linspace(dphi_dz.min(), dphi_dz.max(), 10)
ax.plot(xx, rows[0]["dz_um"] * xx, color="#D55E00", lw=1.0,
        label=f"dz = {rows[0]['dz_um']:.3f} um, R$^2$ = {rows[0]['r2']:.2f}")
ax.set_xlabel("grid dphi/dz [rad/um]")
ax.set_ylabel("static residual [rad]")
ax.legend(frameon=False, fontsize=5)
for ax in fig.get_axes()[3:]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

save_figure(
    fig,
    params={"crop_sub": CROP_SUB, "frames": [FRAMES[0], FRAMES[-1]], "z_step_um": Z_STEP_UM,
            "grid_z": GRID_Z, "channels": [f"Pos{p}ch{c:02d}" for p, c in CHANNELS],
            "tilt_h": TILT_H, "out_h": OUT_H, "lowpass_sigma_px": LOWPASS_SIGMA_PX},
    caption=(
        "Is the low-frequency residual static, and is it a focus offset carried in from the grid? "
        "260917 test run (no cells loaded; 2% glucose; single z at grid index 5; 180 s interval; "
        f"{len(FRAMES)} frames). Operational definitions: each crop is the grid-subtracted "
        "crop_sub_rawraw crop with a first-order 2D plane fitted on the aperture-end third removed "
        "(scripts/apply_plane_tilt_endthird.py); 'static' is the per-pixel mean over all frames of "
        "one channel and 'dynamic' the per-pixel std over the same frames, both in rad; "
        "'grid dphi/dz' is (grid z6 - grid z4) / 0.8 um through the identical crop and plane "
        "correction, i.e. the measured phase derivative along z at the working plane; 'fitted z "
        "offset' is the least-squares a in static = a x dphi/dz, in um, with R^2 against the "
        "static pattern's own variance. n = 10 channels of one experiment, each summarising "
        f"{len(FRAMES)} frames; no error bars and no statistical test. Images share a +-0.2 rad "
        "scale except dphi/dz, which is scaled to its own maximum."
    ),
    description=("Static vs dynamic part of the low-frequency residual, and how much of the static "
                 "part a grid-time z offset explains (260917, no cells, 42 frames)."),
    data={f"{k[0]}_{k[1]}_{n}": v for k, vs in store.items()
          for n, v in zip(("static", "dynamic", "dphi_dz"), vs)}
    | {k: np.array([r[k] for r in rows]) for k in
       ("pos", "ch", "static_rms", "dynamic_rms", "ratio", "dz_um", "r2", "px_noise")},
)
