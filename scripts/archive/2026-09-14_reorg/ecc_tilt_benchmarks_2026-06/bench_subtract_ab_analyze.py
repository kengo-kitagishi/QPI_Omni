"""bench_subtract_ab_analyze.py -- real-data precision from the A/B shift logs

Reads pos_shifts_<tag>.json produced by bench_subtract_ab.py and quantifies
estimator precision on REAL data without ground truth, using the cross-channel
scatter idea: at a fixed stage position every channel sees the SAME true shift,
so the spread of the per-channel shift estimates within a frame IS the
estimator noise. Smaller cross-channel std = more precise estimator.

Also plots the aggregated per-frame shift over time for each method.

Usage:
    python scripts/bench_subtract_ab_analyze.py
    python scripts/bench_subtract_ab_analyze.py --pos 1 --skip-start 2
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
LABELS = {"ecc": "ECC-uint8 (current)", "sg": "SG-NCC (Qiita)", "ecc_float": "ECC-float"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--pos", type=int, default=1)
    p.add_argument("--skip-start", type=int, default=2,
                   help="drop the first N startup frames (large pre-lock drift)")
    return p.parse_args()


def load_method(iso_channels, tag):
    d = json.loads((iso_channels / f"pos_shifts_{tag}.json").read_text(encoding="utf-8"))
    fr = d["frame_results"]
    avg_x = np.array([f["shift_x_avg"] if f and f["shift_x_avg"] is not None else np.nan for f in fr])
    avg_y = np.array([f["shift_y_avg"] if f and f["shift_y_avg"] is not None else np.nan for f in fr])
    # per-frame cross-channel std over USED (non-excluded) channels
    cc_std_x, cc_std_y = [], []
    for f in fr:
        if not f:
            cc_std_x.append(np.nan); cc_std_y.append(np.nan); continue
        xs = [c["shift_x"] for c in f["per_channel"] if not c["excluded"]]
        ys = [c["shift_y"] for c in f["per_channel"] if not c["excluded"]]
        cc_std_x.append(np.std(xs) if len(xs) >= 2 else np.nan)
        cc_std_y.append(np.std(ys) if len(ys) >= 2 else np.nan)
    return avg_x, avg_y, np.array(cc_std_x), np.array(cc_std_y)


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    pix_nm = cfg["pixel_scale_um"] * 1000.0
    iso_channels = Path(cfg["save_dir"]).parent / "ecc_sg_ab" / f"Pos{args.pos}" / "channels"

    data = {tag: load_method(iso_channels, tag) for tag in TAGS}
    s = args.skip_start

    print(f"Cross-channel scatter (real-data precision), frames {s}..end:")
    summary = {}
    for tag in TAGS:
        _, _, ccx, ccy = data[tag]
        mx = np.nanmean(ccx[s:]); my = np.nanmean(ccy[s:])
        summary[tag] = (mx, my)
        print(f"  {LABELS[tag]:22s}  std_x={mx:.4f}px ({mx*pix_nm:.1f}nm)  "
              f"std_y={my:.4f}px ({my*pix_nm:.1f}nm)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    frames = np.arange(len(data["ecc"][0]))

    # --- [0,0] aggregated shift_x over time (settled view) ---
    ax = axes[0, 0]
    for tag in TAGS:
        ax.plot(frames[s:], data[tag][0][s:], lw=0.8, color=COLORS[tag], label=LABELS[tag])
    ax.set_xlabel("Frame"); ax.set_ylabel("shift_x_avg (px)")
    ax.set_title("Aggregated X shift over time")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    # --- [0,1] aggregated shift_y over time ---
    ax = axes[0, 1]
    for tag in TAGS:
        ax.plot(frames[s:], data[tag][1][s:], lw=0.8, color=COLORS[tag], label=LABELS[tag])
    ax.set_xlabel("Frame"); ax.set_ylabel("shift_y_avg (px)")
    ax.set_title("Aggregated Y shift over time")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    # --- [1,0] cross-channel std over time (X) -- the precision metric ---
    ax = axes[1, 0]
    for tag in TAGS:
        ax.plot(frames[s:], data[tag][2][s:], lw=0.8, color=COLORS[tag],
                label=f"{LABELS[tag]}: mean {summary[tag][0]*pix_nm:.1f}nm")
    ax.set_xlabel("Frame"); ax.set_ylabel("cross-channel std of shift_x (px)")
    ax.set_title("Per-frame cross-channel scatter X  (lower = more precise)")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    # --- [1,1] cross-channel std histogram (X+Y pooled) ---
    ax = axes[1, 1]
    bins = np.linspace(0, max(0.001, np.nanmax([np.nanmax(data[t][2][s:]) for t in TAGS]) * 1.05), 40)
    for tag in TAGS:
        pooled = np.concatenate([data[tag][2][s:], data[tag][3][s:]])
        pooled = pooled[~np.isnan(pooled)]
        ax.hist(pooled, bins=bins, histtype="step", lw=1.8, color=COLORS[tag],
                density=True, label=LABELS[tag])
    ax.set_xlabel("cross-channel std (px)"); ax.set_ylabel("density")
    ax.set_title("Cross-channel scatter distribution (X & Y pooled)")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Subtraction A/B real-data precision (260517 Pos{args.pos}, {len(frames)} frames)\n"
        f"cross-channel scatter = estimator noise (no ground truth needed)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_figure(
        fig,
        params={"pos": args.pos, "n_frames": int(len(frames)), "skip_start": s,
                "pixel_scale_nm": pix_nm,
                **{f"{t}_ccstd_x_px": float(summary[t][0]) for t in TAGS},
                **{f"{t}_ccstd_y_px": float(summary[t][1]) for t in TAGS}},
        description=(
            "Real-data estimator precision from the subtract A/B shift logs: "
            "per-frame cross-channel scatter of the per-channel shift estimates "
            "(ECC-uint8 vs SG-NCC vs ECC-float)."),
        data={f"{t}_{k}": data[t][i] for t in TAGS
              for i, k in enumerate(["avg_x", "avg_y", "ccstd_x", "ccstd_y"])},
    )
    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
