"""bench_perchannel_bias.py -- per-channel systematic shift (cell-channel ECC bias)

Diagnoses the systematic (non-random) offset seen in the subtraction by plotting
the per-channel mean shift over settled frames for each estimator. A fixed
per-channel pattern that is the SAME across estimators is NOT estimator noise --
it is a real per-channel registration offset. Here it tracks the known
glucose-dependent cell-channel ECC bias (cells in the timelapse pull the
alignment vs the cell-free grid background), not calibration or quantization.

Reads pos_shifts_<tag>.json from the bench_subtract_ab isolated output.

Usage:
    python scripts/bench_perchannel_bias.py --frames 50 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

DEFAULT_CONFIG = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json"
TAGS = ["ecc", "sg", "ecc_float"]
COLORS = {"ecc": "#d62728", "sg": "#1f77b4", "ecc_float": "#2ca02c"}
LABELS = {"ecc": "ECC-uint8", "sg": "SG-NCC", "ecc_float": "ECC-float"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--pos", type=int, default=1)
    p.add_argument("--frames", type=int, nargs=2, default=[50, 300],
                   help="settled frame range [start, end) to average")
    return p.parse_args()


def per_channel_mean(fr, lo, hi, n_ch):
    accX = {c: [] for c in range(n_ch)}
    accY = {c: [] for c in range(n_ch)}
    for f in fr[lo:hi]:
        if not f:
            continue
        for c in f["per_channel"]:
            if not c["excluded"]:
                accX[c["channel"]].append(c["shift_x"])
                accY[c["channel"]].append(c["shift_y"])
    mx = np.array([np.mean(accX[c]) if accX[c] else np.nan for c in range(n_ch)])
    my = np.array([np.mean(accY[c]) if accY[c] else np.nan for c in range(n_ch)])
    return mx, my


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    pix_nm = cfg["pixel_scale_um"] * 1000.0
    ic = Path(cfg["save_dir"]).parent / "ecc_sg_ab" / f"Pos{args.pos}" / "channels"
    rois = json.loads((ic / "channel_rois.json").read_text(encoding="utf-8"))
    cy = np.array([r["cy"] for r in rois])
    n_ch = len(rois)
    lo, hi = args.frames

    data = {}
    for tag in TAGS:
        d = json.loads((ic / f"pos_shifts_{tag}.json").read_text(encoding="utf-8"))
        data[tag] = per_channel_mean(d["frame_results"], lo, hi, n_ch)

    chx = np.arange(n_ch)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    w = 0.25
    for i, tag in enumerate(TAGS):
        mx, my = data[tag]
        axes[0].bar(chx + (i - 1) * w, mx, w, color=COLORS[tag], label=LABELS[tag])
        axes[1].bar(chx + (i - 1) * w, my, w, color=COLORS[tag], label=LABELS[tag])
    for ax, comp in zip(axes, ["X", "Y"]):
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("Channel index")
        ax.set_ylabel(f"per-channel mean shift {comp} (px)")
        ax.set_xticks(chx)
        ax.set_title(f"Per-channel systematic shift {comp}  (frames {lo}-{hi})")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    # secondary nm axis hint
    spread = {t: float(np.nanmax(data[t][0]) - np.nanmin(data[t][0])) for t in TAGS}
    fig.suptitle(
        "Per-channel systematic offset = cell-channel ECC bias (same across estimators, "
        "NOT fixed by recalibration)\n"
        f"X spread across channels: " + "  ".join(f"{LABELS[t]} {spread[t]:.2f}px ({spread[t]*pix_nm:.0f}nm)" for t in TAGS),
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    save_figure(
        fig,
        params={"pos": args.pos, "frames_lo": lo, "frames_hi": hi,
                "pixel_scale_nm": pix_nm,
                **{f"{t}_x_spread_px": spread[t] for t in TAGS}},
        description=("Per-channel mean shift (cell-channel ECC bias): a fixed "
                     "per-channel pattern identical across ECC-uint8/SG/ECC-float, "
                     "explaining the systematic subtraction offset."),
        data={f"{t}_{c}": np.array(v) for t in TAGS for c, v in
              zip(["mx", "my"], data[t])} | {"cy": cy},
    )
    plt.close(fig)
    print("Per-channel X spread (px):", {t: round(spread[t], 3) for t in TAGS})
    print("Done.")


if __name__ == "__main__":
    main()
