"""bench_subtract_ab_visual.py -- side-by-side subtracted crops + method diffs

Renders the grid-subtracted crop for each estimator (ECC-uint8 / SG-NCC /
ECC-float) for selected channels at a chosen frame, plus method-difference
images. The difference (method A - method B) cancels the common grid-subtracted
cell signal and isolates the effect of the estimator's shift choice, so any
bright residual in a diff panel = where the alignment differs. This is the
visual companion to the by-eye comparison of the crop_sub_* folders.

Usage:
    python scripts/bench_subtract_ab_visual.py --frame 150 --channels 3 5 7
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
TAGS = ["ecc", "sg", "ecc_float"]
LABELS = {"ecc": "ECC-uint8 (current)", "sg": "SG-NCC (Qiita)", "ecc_float": "ECC-float"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--pos", type=int, default=1)
    p.add_argument("--frame", type=int, default=150)
    p.add_argument("--channels", type=int, nargs="+", default=[3, 5, 7])
    p.add_argument("--suffix", default="", help="crop_sub_<tag><suffix> folder suffix")
    p.add_argument("--vlim", type=float, default=0.6, help="subtracted image color limit (rad)")
    p.add_argument("--dlim", type=float, default=0.15, help="difference image color limit (rad)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(cfg["save_dir"]).parent / "ecc_sg_ab" / f"Pos{args.pos}"

    def load(tag, ch):
        fs = sorted((root / f"crop_sub_{tag}{args.suffix}" / f"ch{ch:02d}").glob("*.tif"))
        return tifffile.imread(str(fs[args.frame])).astype(np.float32)

    chans = args.channels
    ncol = 5  # ecc | sg | ecc_float | sg-ecc | ecc_float-ecc
    fig, axes = plt.subplots(len(chans), ncol, figsize=(16, 1.7 * len(chans) + 1))
    if len(chans) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["ECC-uint8", "SG-NCC", "ECC-float",
                  "SG-NCC − ECC", "ECC-float − ECC"]
    diff_rms = {("sg", "ecc"): [], ("ecc_float", "ecc"): []}

    for r, ch in enumerate(chans):
        imgs = {t: load(t, ch) for t in TAGS}
        d_sg = imgs["sg"] - imgs["ecc"]
        d_fl = imgs["ecc_float"] - imgs["ecc"]
        diff_rms[("sg", "ecc")].append(float(np.sqrt(np.mean(d_sg ** 2))))
        diff_rms[("ecc_float", "ecc")].append(float(np.sqrt(np.mean(d_fl ** 2))))
        panels = [imgs["ecc"], imgs["sg"], imgs["ecc_float"], d_sg, d_fl]
        for c, (ax, panel) in enumerate(zip(axes[r], panels)):
            is_diff = c >= 3
            lim = args.dlim if is_diff else args.vlim
            ax.imshow(panel, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"ch{ch:02d}", fontsize=9)

    print("Diff RMS (rad) per channel:")
    for key, vals in diff_rms.items():
        print(f"  {key[0]}-{key[1]}: " + ", ".join(f"ch{chans[i]}={v:.4f}" for i, v in enumerate(vals)))

    fig.suptitle(
        f"Subtraction A/B visual (260517 Pos{args.pos}, frame {args.frame}, grid z=8)\n"
        f"subtracted crops |vlim|={args.vlim}  diffs |vlim|={args.dlim} rad  "
        f"(diff isolates the estimator's shift effect)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    save_figure(
        fig,
        params={"pos": args.pos, "frame": args.frame, "channels": chans,
                "grid_z": 8, "vlim": args.vlim, "dlim": args.dlim,
                **{f"diffrms_{a}_{b}_mean": float(np.mean(v)) for (a, b), v in diff_rms.items()}},
        description=("Visual A/B of grid-subtracted crops for ECC-uint8 / SG-NCC / "
                     "ECC-float with method-difference panels (grid z=8)."),
        data={f"{t}_ch{ch}": load(t, ch) for t in TAGS for ch in chans},
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
