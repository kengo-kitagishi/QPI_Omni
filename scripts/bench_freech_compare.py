"""bench_freech_compare.py -- all-channel vs cell-free-average subtraction

Compares the grid-subtracted crops produced with the all-channel aggregate shift
(crop_sub_<tag>) against the cell-free-average shift (crop_sub_<tag>_freech, high
corr threshold dropping cell-bearing channels). Shows the time-mean subtracted
crop for each and their difference, for the cell channels where the cell-channel
ECC bias was largest. The difference panel = the effect of removing the bias.

Usage:
    python scripts/bench_freech_compare.py --tag ecc_float --channels 0 3 5 9
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
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

DEFAULT_CONFIG = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--pos", type=int, default=1)
    p.add_argument("--tag", default="ecc_float")
    p.add_argument("--channels", type=int, nargs="+", default=[0, 3, 5, 9])
    p.add_argument("--vlim", type=float, default=0.5)
    p.add_argument("--dlim", type=float, default=0.1)
    p.add_argument("--nmax", type=int, default=300)
    return p.parse_args()


def time_mean(folder, ch, nmax):
    fs = sorted((folder / f"ch{ch:02d}").glob("*.tif"))[:nmax]
    stack = np.stack([tifffile.imread(str(f)).astype(np.float32) for f in fs])
    return stack.mean(0)


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(cfg["save_dir"]).parent / "ecc_sg_ab" / f"Pos{args.pos}"
    all_dir = root / f"crop_sub_{args.tag}"
    free_dir = root / f"crop_sub_{args.tag}_freech"

    chans = args.channels
    fig, axes = plt.subplots(len(chans), 3, figsize=(11, 1.7 * len(chans) + 1))
    if len(chans) == 1:
        axes = axes[np.newaxis, :]
    titles = [f"all-ch avg", f"cell-free avg", "difference (free − all)"]

    diff_rms = []
    for r, ch in enumerate(chans):
        a = time_mean(all_dir, ch, args.nmax)
        f = time_mean(free_dir, ch, args.nmax)
        d = f - a
        diff_rms.append(float(np.sqrt(np.mean(d ** 2))))
        for c, (ax, panel, is_d) in enumerate(zip(axes[r], [a, f, d], [False, False, True])):
            lim = args.dlim if is_d else args.vlim
            ax.imshow(panel, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"ch{ch:02d}", fontsize=9)

    print(f"[{args.tag}] time-mean diff RMS (free-all) per channel: " +
          ", ".join(f"ch{chans[i]}={v:.4f}" for i, v in enumerate(diff_rms)))

    fig.suptitle(
        f"All-channel vs cell-free averaging ({args.tag}, 260517 Pos{args.pos}, grid z=8)\n"
        f"time-mean subtracted crop; difference = effect of removing cell-channel ECC bias",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    save_figure(
        fig,
        params={"pos": args.pos, "tag": args.tag, "channels": chans,
                "nmax": args.nmax, "vlim": args.vlim, "dlim": args.dlim,
                "diff_rms_mean": float(np.mean(diff_rms))},
        description=(f"All-channel vs cell-free-average subtraction time-means for {args.tag}, "
                     "with difference panels showing the cell-channel-bias correction."),
        data={f"all_ch{ch}": time_mean(all_dir, ch, args.nmax) for ch in chans} |
             {f"free_ch{ch}": time_mean(free_dir, ch, args.nmax) for ch in chans},
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
