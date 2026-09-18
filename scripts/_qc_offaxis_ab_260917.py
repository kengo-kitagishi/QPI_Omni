"""Same frames, two off-axis centres. Does the choice change the artifact?

A sub-pixel carrier error is a pure ramp and the end-third plane fit removes it exactly, so
only the INTEGER choice can change anything: it moves the aperture mask in the spectrum, which
is not a ramp. The measured peak sits at about (462.8, 1672.0) while the config uses
(463, 1672), so the two candidates are one pixel apart in row.

Both the grid reference and the timelapse frame are reconstructed with the SAME centre inside
each arm, exactly as production does, then subtracted with the production kernel.

Reads the newest surviving raw hologram of each Pos (the online run deletes older ones) and
writes the re-subtracted crops to D:, so the acquisition disk is only read.
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

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config_zstack.json")
ap.add_argument("--log", default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log_zstack.json")
ap.add_argument("--centres", default="463,1672;462,1672",
                help="semicolon separated row,col pairs; first one is the production centre")
ap.add_argument("--pos-end", type=int, default=40)
ap.add_argument("--out-root", default=r"D:\AquisitionData\Kitagishi\260917\_offaxis_ab")
a = ap.parse_args()

cfg = json.load(open(a.config, encoding="utf-8"))
log = [e for e in json.load(open(a.log, encoding="utf-8")) if e.get("per_pos")]
by_frame = {e["timepoint"]: e for e in log}
tilt_h = cfg.get("tilt_crop_h_raw", 270)
out_h = cfg.get("crop_sub_output_crop_h") or tilt_h
pos_split = int(cfg["pos_split"])
grid_dir = Path(cfg["grid_dir"])
grid_z = int(cfg["raw_grid_z_index"])
centres = [tuple(int(v) for v in c.split(",")) for c in a.centres.split(";")]

stats = {c: [] for c in centres}
n_pos = 0
for pos in range(1, a.pos_end + 1):
    label = f"Pos{pos}"
    tl_dir = Path(cfg["save_dir"]) / label / "z000"
    holos = sorted(tl_dir.glob("img_*_ph_000.tif"))
    if not holos:
        continue
    holo = holos[-1]                      # the newest one the cleanup has not taken yet
    frame = int(re.search(r"img_(\d+)_", holo.name).group(1))
    entry = by_frame.get(frame)
    if entry is None:
        continue
    p = next((q for q in entry["positions"] if q["pos_label"] == label), None)
    if p is None:
        continue
    sx, sy = p["tx_avg_px"], p["ty_avg_px"]
    fit_right = pos >= pos_split
    raw_crop = tuple(cfg["crop_before"]) if pos < pos_split else tuple(cfg["crop_after"])

    pos_map = gs.scan_grid_positions(grid_dir, label)
    cal_path = grid_dir / f"grid_calibration_{label}.json"
    if not pos_map or not cal_path.exists():
        continue
    xi, yi, _, _, _, cal_dx, cal_dy, rx, ry = gs.select_grid(
        sx, sy, pos_map, gs.load_grid_calibration(str(cal_path)),
        pixel_scale_um=cfg["pixel_scale_um"],
        x_step=cfg.get("crop_sub_x_step_um", 0.05), y_step=cfg.get("crop_sub_y_step_um", 0.05),
        shift_sign_x=-1, shift_sign_y=-1)
    grid_holo = pos_map[(xi, yi)] / f"img_000000000_ph_{grid_z:03d}.tif"
    if not grid_holo.exists():
        continue
    rois = json.load(open(grid_dir / f"{label}_x+0_y+0" / "output_phase" / "channels" /
                          "channel_rois.json", encoding="utf-8"))

    for centre in centres:
        brg.OFFAXIS_CENTER = centre          # parameter override of the production reconstruction
        tl_img = brg.reconstruct_from_holo(holo, raw_crop).astype(np.float64)
        grid_img = brg.reconstruct_from_holo(grid_holo, raw_crop).astype(np.float64)
        crops, _ = gs.process_single_frame(
            tl_img, sx, sy, rois, cal_dx, cal_dy, rx, ry, grid_img,
            output_crop_h_override=out_h, tilt_crop_h_raw=tilt_h, use_raw_phase=True,
            apply_subpixel_correction=True, fit_right=fit_right, apply_inverse_shift=False)
        tag = f"c{centre[0]}_{centre[1]}"
        for ch, crop in enumerate(crops):
            prof = crop.mean(axis=0)
            stats[centre].append(abs(prof[:20].mean() if fit_right else prof[-20:].mean()))
            d = Path(a.out_root) / tag / label / "output_phase" / "channels" / \
                "crop_sub_rawraw" / "z000" / f"ch{ch:02d}"
            d.mkdir(parents=True, exist_ok=True)
            # one frame per channel is all a contact sheet needs; name them alike across arms
            tifffile.imwrite(str(d / "img_000000000_ph_000.tif"), crop.astype(np.float32))
    n_pos += 1

print(f"{n_pos} Pos, newest surviving raw hologram of each")
for centre in centres:
    v = np.array(stats[centre])
    if not len(v):
        continue
    print(f"  centre {centre}: n={len(v)}  |far-end| median {np.median(v):.4f}  "
          f"p90 {np.percentile(v, 90):.4f}  max {v.max():.3f}  >0.1 rad: {(v > 0.1).sum()}")
print(f"crops under {a.out_root}\\c<row>_<col>\\")
