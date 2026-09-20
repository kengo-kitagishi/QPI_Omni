"""Measure the off-axis centre per frame and reconstruct with it. Does the artifact change?

Three arms on the SAME raw holograms, each subtracted with the production kernel:

  fixed     both the frame and the grid reference demodulated at the config centre
            (463, 1672) -- what production does today
  adaptive  the centre is measured from THIS frame's own hologram (sub-pixel FFT peak,
            rounded to the pixel the spectrum crop needs) and used for both the frame and
            the grid reference, so the pair stays consistent
  per_image the frame at its own measured centre, the grid at the grid's own measured centre
            -- "every image demodulated at its own carrier", the most literal reading of
            adaptive, and the one that can mismatch the pair

A sub-pixel centre difference is a pure phase ramp and the tilt fit removes it exactly
(checked: 1e-16 rad left), so only the integer part can change anything -- it moves the
aperture mask in the spectrum, which is not a ramp. That is what this measures.

Needs raw holograms, which ph_zstack_test_3 keeps (cleanup_raw_holograms = false).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import batch_reconstruction_grid as brg  # noqa: E402
import grid_subtract as gs  # noqa: E402
from optical_config import OFFAXIS_CENTER as CONFIG_CENTRE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config_zstack.json")
ap.add_argument("--log", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log_zstack.json")
ap.add_argument("--frame", type=int, default=3)
ap.add_argument("--pos-end", type=int, default=30)
ap.add_argument("--out-root", default=r"D:\AquisitionData\Kitagishi\260917\_adaptive_offaxis")
ap.add_argument("--grid-raw-root", default=r"E:\260917\grid_hologram_0p05_3",
                help="where the grid RAW holograms live (grid_dir in the config is the recon output)")
a = ap.parse_args()

cfg = json.load(open(a.config, encoding="utf-8"))
log = [e for e in json.load(open(a.log, encoding="utf-8")) if e.get("per_pos")]
entry = next((e for e in log if e["timepoint"] == a.frame), None)
if entry is None:
    sys.exit(f"frame {a.frame} is not in the drift log")
tilt_h = cfg.get("tilt_crop_h_raw", 270)
out_h = cfg.get("crop_sub_output_crop_h") or tilt_h
pos_split = int(cfg["pos_split"])
grid_dir = Path(cfg["grid_dir"])
grid_z = int(cfg["raw_grid_z_index"])


def measure_centre(path, crop):
    """Sub-pixel FFT peak of the +1 order, and the pixel the spectrum crop would use."""
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        img = img[..., 0]
    arr = img[crop[0]:crop[1], crop[2]:crop[3]].astype(float)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
    h, w = mag.shape
    Y, X = np.ogrid[:h, :w]
    masked = mag.copy()
    masked[(Y - h // 2) ** 2 + (X - w // 2) ** 2 < 100 ** 2] = 0
    masked[h // 2:, :] = 0                      # keep the +1 order (upper half)
    pr, pc = np.unravel_index(np.argmax(masked), masked.shape)
    win = 5
    reg = mag[pr - win:pr + win + 1, pc - win:pc + win + 1]
    rr, cc = np.mgrid[pr - win:pr + win + 1, pc - win:pc + win + 1]
    sub = ((rr * reg).sum() / reg.sum(), (cc * reg).sum() / reg.sum())
    return sub, (int(round(sub[0])), int(round(sub[1])))


def recon(path, crop, centre):
    brg.OFFAXIS_CENTER = centre                 # parameter override of the production recon
    return brg.reconstruct_from_holo(Path(path), crop).astype(np.float64)


ARMS = ("fixed", "adaptive", "per_image")
stats = {k: [] for k in ARMS}
centres_seen = {k: [] for k in ARMS}
n_pos = 0
for p in entry["positions"]:
    pos = int(re.match(r"Pos(\d+)", p["pos_label"]).group(1))
    if pos > a.pos_end:
        continue
    label, sx, sy = p["pos_label"], p["tx_avg_px"], p["ty_avg_px"]
    fit_right = pos >= pos_split
    raw_crop = tuple(cfg["crop_before"]) if pos < pos_split else tuple(cfg["crop_after"])
    tl_holo = Path(cfg["save_dir"]) / label / "z000" / f"img_{a.frame:09d}_ph_000.tif"
    if not tl_holo.exists():
        continue
    pos_map = gs.scan_grid_positions(grid_dir, label)
    cal_path = grid_dir / f"grid_calibration_{label}.json"
    if not pos_map or not cal_path.exists():
        continue
    xi, yi, _, _, _, cal_dx, cal_dy, rx, ry = gs.select_grid(
        sx, sy, pos_map, gs.load_grid_calibration(str(cal_path)),
        pixel_scale_um=cfg["pixel_scale_um"],
        x_step=cfg.get("crop_sub_x_step_um", 0.05), y_step=cfg.get("crop_sub_y_step_um", 0.05),
        shift_sign_x=-1, shift_sign_y=-1)
    # pos_map points into the recon output; the raw hologram of that grid point is on the
    # acquisition disk under the same folder name
    grid_holo = Path(a.grid_raw_root) / pos_map[(xi, yi)].name / f"img_000000000_ph_{grid_z:03d}.tif"
    if not grid_holo.exists():
        continue
    rois = json.load(open(grid_dir / f"{label}_x+0_y+0" / "output_phase" / "channels" /
                          "channel_rois.json", encoding="utf-8"))

    tl_sub, tl_px = measure_centre(tl_holo, raw_crop)
    g_sub, g_px = measure_centre(grid_holo, raw_crop)
    plan = {"fixed": (CONFIG_CENTRE, CONFIG_CENTRE),
            "adaptive": (tl_px, tl_px),
            "per_image": (tl_px, g_px)}

    for arm, (c_tl, c_grid) in plan.items():
        centres_seen[arm].append((c_tl, c_grid))
        tl_img = recon(tl_holo, raw_crop, c_tl)
        grid_img = recon(grid_holo, raw_crop, c_grid)
        crops, _ = gs.process_single_frame(
            tl_img, sx, sy, rois, cal_dx, cal_dy, rx, ry, grid_img,
            output_crop_h_override=out_h, tilt_crop_h_raw=tilt_h, use_raw_phase=True,
            apply_subpixel_correction=True, fit_right=fit_right, apply_inverse_shift=False)
        for ch, crop in enumerate(crops):
            prof = crop.mean(axis=0)
            stats[arm].append(abs(prof[:20].mean() if fit_right else prof[-20:].mean()))
            d = Path(a.out_root) / arm / label / "output_phase" / "channels" / \
                "crop_sub_rawraw" / "z000" / f"ch{ch:02d}"
            d.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(d / f"img_{a.frame:09d}_ph_000.tif"), crop.astype(np.float32))
    if n_pos < 5:
        print(f"  {label}: frame peak {tl_sub[0]:.2f},{tl_sub[1]:.2f} -> px {tl_px}   "
              f"grid peak {g_sub[0]:.2f},{g_sub[1]:.2f} -> px {g_px}")
    n_pos += 1

print(f"\nframe {a.frame}, {n_pos} Pos, same raw holograms in every arm")
for arm in ARMS:
    v = np.array(stats[arm])
    if not len(v):
        continue
    uniq = {c for c in centres_seen[arm]}
    print(f"  {arm:9s}: n={len(v)}  |far-end| median {np.median(v):.4f}  "
          f"p90 {np.percentile(v, 90):.4f}  max {v.max():.3f}  >0.1 rad: {(v > 0.1).sum()}  "
          f"({len(uniq)} distinct centre pairs)")
print(f"crops under {a.out_root}\\<arm>\\")
