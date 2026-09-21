"""Where does the far-end ramp come from -- the timelapse frame, the grid reference,
or only their difference?

Uses the production kernel (grid_subtract.process_single_frame) unchanged, three ways:
  tl    : grid_img=zeros           -> tilt-corrected timelapse crop, nothing real subtracted
  grid  : tl_img=grid, grid_img=zeros -> tilt-corrected grid reference crop
  sub   : grid_img=grid            -> what the online run writes (crop_sub_rawraw)
No new alignment or warp code: the residual sub-pixel warp, the crop and the tilt fit
all happen inside process_single_frame exactly as in production.
"""
import json
import os
import re
import sys

import numpy as np
import tifffile
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from figure_logger import save_figure  # noqa: E402  (applies paper.mplstyle)
import grid_subtract as gs  # noqa: E402

CONFIG = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config_zstack.json"
LOG = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log_zstack.json"
POS_LABEL = "Pos39"
CH = 2
FRAMES = [6, 9]          # 6 = flat by eye, 9 = strongest ramp

cfg = json.load(open(CONFIG, encoding="utf-8"))
log = [e for e in json.load(open(LOG, encoding="utf-8")) if e.get("per_pos")]

pos_num = int(re.match(r"Pos(\d+)", POS_LABEL).group(1))
pos_split = int(cfg["pos_split"])
raw_crop = tuple(cfg["crop_before"]) if pos_num < pos_split else tuple(cfg["crop_after"])
fit_right = pos_num >= pos_split
tilt_h = cfg.get("tilt_crop_h_raw", 270)
out_h = cfg.get("crop_sub_output_crop_h") or tilt_h

grid_dir = cfg["grid_dir"]
pos_map = gs.scan_grid_positions(__import__("pathlib").Path(grid_dir), POS_LABEL)
grid_cal = gs.load_grid_calibration(os.path.join(grid_dir, f"grid_calibration_{POS_LABEL}.json"))
rois = json.load(open(os.path.join(grid_dir, f"{POS_LABEL}_x+0_y+0", "output_phase",
                                   "channels", "channel_rois.json"), encoding="utf-8"))

res = {}
for t in FRAMES:
    q = [p for p in log[t]["positions"] if p["pos_label"] == POS_LABEL][0]
    sx, sy = q["tx_avg_px"], q["ty_avg_px"]
    xi, yi, dist, _, _, cal_dx, cal_dy, rx, ry = gs.select_grid(
        sx, sy, pos_map, grid_cal, pixel_scale_um=cfg["pixel_scale_um"],
        x_step=cfg.get("crop_sub_x_step_um", 0.05), y_step=cfg.get("crop_sub_y_step_um", 0.05),
        shift_sign_x=-1, shift_sign_y=-1)

    z_idx = cfg.get("raw_tl_z_index", 0)
    grid_z = z_idx + cfg["raw_grid_z_index"] - cfg.get("raw_tl_z_index", 0)
    tl_path = os.path.join(cfg["save_dir"], POS_LABEL, f"z{z_idx:03d}", "output_phase_raw",
                           f"img_{t:09d}_ph_{z_idx:03d}_phase.tif")
    grid_path = os.path.join(str(pos_map[(xi, yi)]), "output_phase_raw",
                             f"img_000000000_ph_{grid_z:03d}_phase.tif")
    tl_img = tifffile.imread(tl_path).astype(np.float64)
    grid_img = tifffile.imread(grid_path).astype(np.float64)

    def run(img, ref, warp):
        out, _ = gs.process_single_frame(
            img, sx, sy, rois, cal_dx, cal_dy,
            rx if warp else 0.0, ry if warp else 0.0, ref,
            output_crop_h_override=out_h, tilt_crop_h_raw=tilt_h,
            use_raw_phase=True, apply_subpixel_correction=warp,
            fit_right=fit_right, apply_inverse_shift=False)
        return out[CH]

    res[t] = {
        # subtracting a zero image keeps the production 2pi + tilt-fit + crop path
        "tl": run(tl_img, np.zeros_like(tl_img), True),
        "grid": run(grid_img, np.zeros_like(grid_img), False),
        "sub": run(tl_img, grid_img, True),
        "meta": (xi, yi, dist, rx, ry, sx, sy),
    }

far = lambda a: (a.mean(axis=0)[-20:].mean() if pos_num < pos_split
                 else a.mean(axis=0)[:20].mean())

fig, axes = plt.subplots(3, len(FRAMES), figsize=(7.0, 5.4),
                         gridspec_kw={"height_ratios": [1, 1, 1.4], "hspace": 0.5})
for j, t in enumerate(FRAMES):
    xi, yi, dist, rx, ry, sx, sy = res[t]["meta"]
    for i, key in enumerate(["tl", "grid"]):
        ax = axes[i, j]
        ax.imshow(res[t][key], cmap="inferno", vmin=0, vmax=1.8, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"T={t} {key}  far={far(res[t][key]):+.2f} rad", pad=2)
    ax = axes[2, j]
    for key, lw in (("tl", 0.9), ("grid", 0.9), ("sub", 1.2)):
        p = res[t][key].mean(axis=0)
        ax.plot(np.arange(len(p)), p, lw=lw, label=f"{key} (far {far(res[t][key]):+.2f})")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("position along channel [px]")
    if j == 0:
        ax.set_ylabel("phase [rad]")
    ax.legend(frameon=False, fontsize=5)
    ax.set_title(f"T={t}  grid pt ({xi},{yi}) dist {dist:.2f} px, residual "
                 f"({rx:+.2f},{ry:+.2f}) px", pad=3)

save_figure(
    fig,
    params={"pos": POS_LABEL, "ch": CH, "frames": FRAMES, "grid_dir": grid_dir,
            "tilt_crop_h_raw": tilt_h, "out_crop_h": out_h, "fit_right": fit_right,
            "vmin": 0, "vmax": 1.8},
    caption=(
        "Origin of the far-end ramp in the 260917 test timelapse (no cells; 2% glucose; single "
        "z at grid index 5; Pos39 ch02). Operational definition: each panel is produced by the "
        "production kernel grid_subtract.process_single_frame on the reconstructed raw phase -- "
        "'tl' is the timelapse frame with the residual sub-pixel warp, the channel crop and the "
        "linear tilt fit on the aperture-end third applied but only a zero image subtracted; 'grid' is the "
        "chosen grid reference put through the same crop and tilt fit; 'sub' is tl minus grid, "
        "i.e. what the online run writes as crop_sub_rawraw. Profiles are means over the 40 px "
        "short axis; 'far' is the mean of the 20 px at the end opposite the tilt-fit third. "
        "T=6 looked flat by eye, T=9 carried the strongest ramp. n = 1 channel x 2 timepoints of "
        "one experiment; no error bars, no statistical test."
    ),
    description=("Is the far-end ramp already in the timelapse frame, in the grid reference, or "
                 "only in their difference? Pos39 ch02 at T=6 (flat) and T=9 (ramped)."),
    data={f"{key}_T{t}": res[t][key] for t in FRAMES for key in ("tl", "grid", "sub")},
)

for t in FRAMES:
    xi, yi, dist, rx, ry, sx, sy = res[t]["meta"]
    print(f"T={t} grid({xi},{yi}) dist {dist:.2f} px  ECC shift ({sx:+.2f},{sy:+.2f}) "
          f"residual warp ({rx:+.2f},{ry:+.2f}) px")
    for key in ("tl", "grid", "sub"):
        a = res[t][key]
        print(f"   {key:4s} far {far(a):+.3f} rad  min {a.min():+.2f} max {a.max():+.2f}")
