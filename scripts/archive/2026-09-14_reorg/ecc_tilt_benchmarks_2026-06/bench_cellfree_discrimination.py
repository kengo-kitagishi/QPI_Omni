"""bench_cellfree_discrimination.py -- can the alignment score separate
cell-bearing from cell-free channels?

Production drops cell-bearing channels from the drift average by thresholding
the per-channel ECC correlation against the cell-free grid(0,0) reference
(corr < ecc_min_corr -> "has cells" -> excluded). This script asks two things
on REAL 260517 data, using the exact production preprocessing:

  1. Does the threshold 0.99 actually separate cell vs non-cell channels?
  2. Would the NCC peak value (TM_CCOEFF_NORMED max -- the score the 2D-Gaussian
     / SG-NCC pipeline produces) separate them better than ECC correlation?

For each (channel, frame) it aligns the timelapse crop to the grid(0,0)
reference crop (same tilt_fit_crop / uint8 / float as production) and records
three scores:
  ECC-uint8 corr : production path
  ECC-float corr : ECC on float input
  NCC peak       : cv2.matchTemplate(TM_CCOEFF_NORMED).max() (the SG/Gaussian score)

Cell channels are 0,3,5,9,10 (project_cell_ch_ecc_bias). Frames are sampled
evenly across the whole timelapse for cell-content diversity.

Output: per-channel score strip plots (cell vs non-cell, with the 0.99 line and
the best separating threshold) + a separability summary (ROC-AUC, error at 0.99,
best threshold) printed and saved to E:\\260517\\ecc_sg_ab.

Usage
-----
    python scripts/bench_cellfree_discrimination.py
    python scripts/bench_cellfree_discrimination.py --n-frames 120 --grid-z 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure
from ecc_utils import ecc_align, to_uint8, to_ecc_input, tilt_fit_crop
from bench_subtract_ab import sg_ncc_align, DEFAULT_CONFIG

OUT_DIR = r"E:\260517\ecc_sg_ab"
CELL_CHANNELS = {0, 3, 5, 9, 10}
PROD_THRESH = 0.99


# ==========================================================================
# Scores
# ==========================================================================

SCORE_NAMES = ["ECC-uint8", "ECC-float (production)", "NCC peak (2D-Gauss/SG)"]


def channel_scores(ref_f, mov_f, vmin, vmax):
    """Return {score_name: value or nan} for one (ref, mov) crop pair."""
    out = {n: np.nan for n in SCORE_NAMES}
    r_u8 = ecc_align(to_uint8(ref_f, vmin, vmax), to_uint8(mov_f, vmin, vmax))
    if r_u8 is not None:
        out["ECC-uint8"] = float(r_u8[2])
    # ECC-float = production NEW path: clip([vmin,vmax]) -> float32 (to_ecc_input).
    r_f = ecc_align(to_ecc_input(ref_f, vmin, vmax), to_ecc_input(mov_f, vmin, vmax))
    if r_f is not None:
        out["ECC-float (production)"] = float(r_f[2])
    r_ncc = sg_ncc_align(ref_f.astype(np.float32), mov_f.astype(np.float32))
    if r_ncc is not None:
        out["NCC peak (2D-Gauss/SG)"] = float(r_ncc[2])
    return out


# ==========================================================================
# Separability metrics
# ==========================================================================

def roc_auc(noncell, cell):
    """AUC for "non-cell scores higher than cell" (rank-sum / Mann-Whitney)."""
    noncell = np.asarray(noncell)
    cell = np.asarray(cell)
    if len(noncell) == 0 or len(cell) == 0:
        return np.nan
    allv = np.concatenate([noncell, cell])
    ranks = allv.argsort().argsort() + 1.0
    r_pos = ranks[:len(noncell)].sum()
    auc = (r_pos - len(noncell) * (len(noncell) + 1) / 2) / (len(noncell) * len(cell))
    return float(auc)


def best_threshold(noncell, cell):
    """Threshold maximizing balanced accuracy (non-cell kept, cell dropped)."""
    noncell = np.asarray(noncell)
    cell = np.asarray(cell)
    cand = np.unique(np.concatenate([noncell, cell]))
    best_t, best_ba = np.nan, -1.0
    for t in cand:
        keep_nc = (noncell >= t).mean()      # non-cell correctly kept
        drop_c = (cell < t).mean()           # cell correctly dropped
        ba = 0.5 * (keep_nc + drop_c)
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_t, best_ba


def rates_at(thresh, noncell, cell):
    noncell = np.asarray(noncell)
    cell = np.asarray(cell)
    keep_nc = float((noncell >= thresh).mean()) if len(noncell) else np.nan
    drop_c = float((cell < thresh).mean()) if len(cell) else np.nan
    return keep_nc, drop_c


# ==========================================================================
# Main
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--pos", type=int, default=1)
    p.add_argument("--zsub", default="z000")
    p.add_argument("--grid-z", type=int, default=8,
                   help="grid z plane index for the reference (260517 uses 8)")
    p.add_argument("--n-frames", type=int, default=120,
                   help="frames sampled evenly across the whole timelapse")
    p.add_argument("--out-dir", default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    pos = args.pos
    grid_dir = Path(cfg["grid_dir"])
    fit_right = pos >= cfg["pos_split"]
    vmin, vmax = cfg["ecc_vmin"], cfg["ecc_vmax"]

    # Timelapse frames (sample evenly across the whole run for cell diversity).
    tl_pos_dir = Path(cfg["save_dir"]) / f"Pos{pos}" / args.zsub
    phase_dir = tl_pos_dir / "output_phase"
    all_paths = sorted(phase_dir.glob("img_*_ph_000_phase.tif"))
    if not all_paths:
        raise FileNotFoundError(f"no timelapse phase frames in {phase_dir}")
    idx = np.linspace(0, len(all_paths) - 1, min(args.n_frames, len(all_paths))).astype(int)
    frame_paths = [all_paths[i] for i in idx]

    # Grid(0,0) reference image + per-channel ROIs.
    grid_rois = grid_dir / f"Pos{pos}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json"
    rois = json.loads(grid_rois.read_text(encoding="utf-8"))
    grid_ref_path = (grid_dir / f"Pos{pos}_x+0_y+0" / "output_phase"
                     / f"img_000000000_ph_{args.grid_z:03d}_phase.tif")
    grid_img = tifffile.imread(str(grid_ref_path)).astype(np.float64)

    n_ch = len(rois)
    cell_set = {c for c in CELL_CHANNELS if c < n_ch}
    print(f"Pos{pos}  channels={n_ch}  cell={sorted(cell_set)}  "
          f"non-cell={[c for c in range(n_ch) if c not in cell_set]}")
    print(f"Frames sampled: {len(frame_paths)} / {len(all_paths)}   "
          f"grid z index {args.grid_z}   fit_right={fit_right}")

    # Reference crops per channel (cell-free grid).
    ref_crops = []
    for roi in rois:
        g = tilt_fit_crop(grid_img, roi["cy"], roi["cx"], roi["crop_w"],
                          cfg["ecc_crop_h"], cfg["tilt_crop_h"], fit_right=fit_right)
        ref_crops.append(None if g is None else g.astype(np.float32))

    # scores[name] -> array (n_frames, n_ch) of the score (nan if failed/OOB)
    scores = {n: np.full((len(frame_paths), n_ch), np.nan) for n in SCORE_NAMES}

    def proc(ti_path):
        ti, path = ti_path
        img = tifffile.imread(str(path)).astype(np.float64)
        row = {n: np.full(n_ch, np.nan) for n in SCORE_NAMES}
        for ch, roi in enumerate(rois):
            ref = ref_crops[ch]
            if ref is None:
                continue
            mov = tilt_fit_crop(img, roi["cy"], roi["cx"], roi["crop_w"],
                                cfg["ecc_crop_h"], cfg["tilt_crop_h"], fit_right=fit_right)
            if mov is None:
                continue
            sc = channel_scores(ref, mov.astype(np.float32), vmin, vmax)
            for n in SCORE_NAMES:
                row[n][ch] = sc[n]
        return ti, row

    with ThreadPoolExecutor(max_workers=8) as ex:
        for ti, row in ex.map(proc, list(enumerate(frame_paths))):
            for n in SCORE_NAMES:
                scores[n][ti] = row[n]

    # ---- Separability summary ----
    noncell_ch = [c for c in range(n_ch) if c not in cell_set]
    summary = {}
    print("\n=== Separability (cell-free should score HIGH, cell LOW) ===")
    print(f"{'score':24s} {'AUC':>6s} {'@0.99 keepNC':>13s} {'@0.99 dropC':>12s} "
          f"{'bestThr':>8s} {'bestBA':>7s}")
    for n in SCORE_NAMES:
        nc = scores[n][:, noncell_ch].ravel()
        cc = scores[n][:, sorted(cell_set)].ravel()
        nc = nc[~np.isnan(nc)]
        cc = cc[~np.isnan(cc)]
        auc = roc_auc(nc, cc)
        keep_nc, drop_c = rates_at(PROD_THRESH, nc, cc)
        bt, bba = best_threshold(nc, cc)
        summary[n] = dict(auc=auc, keep_nc_099=keep_nc, drop_c_099=drop_c,
                          best_thr=bt, best_ba=bba,
                          nc_median=float(np.median(nc)), cc_median=float(np.median(cc)))
        print(f"{n:24s} {auc:6.3f} {keep_nc*100:12.1f}% {drop_c*100:11.1f}% "
              f"{bt:8.4f} {bba:7.3f}")

    # ---- Figure: per-channel score strips, 3 panels ----
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=False)
    rng = np.random.default_rng(0)
    for ax, n in zip(axes, SCORE_NAMES):
        for ch in range(n_ch):
            vals = scores[n][:, ch]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            is_cell = ch in cell_set
            color = "#d62728" if is_cell else "#1f77b4"
            x = ch + rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(x, vals, s=8, alpha=0.35, color=color, edgecolors="none")
            ax.scatter([ch], [np.median(vals)], s=60, marker="_",
                       color="k", zorder=5, linewidths=1.8)
        s = summary[n]
        ax.axhline(PROD_THRESH, color="green", lw=1.2, ls="--",
                   label=f"prod thr 0.99 (keepNC {s['keep_nc_099']*100:.0f}%, dropC {s['drop_c_099']*100:.0f}%)")
        if np.isfinite(s["best_thr"]):
            ax.axhline(s["best_thr"], color="purple", lw=1.0, ls=":",
                       label=f"best thr {s['best_thr']:.4f} (BA {s['best_ba']:.3f})")
        ax.set_title(f"{n}\nAUC={s['auc']:.3f}")
        ax.set_xlabel("channel index")
        ax.set_ylabel("alignment score vs cell-free grid")
        ax.set_xticks(range(n_ch))
        ax.legend(frameon=False, fontsize=7, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)

    # Legend proxy for cell/non-cell.
    from matplotlib.lines import Line2D
    proxies = [Line2D([0], [0], marker="o", ls="", color="#d62728", label="cell ch (0,3,5,9,10)"),
               Line2D([0], [0], marker="o", ls="", color="#1f77b4", label="non-cell ch")]
    fig.legend(handles=proxies, loc="upper right", frameon=False, fontsize=8)
    fig.suptitle(
        f"Cell vs cell-free channel discrimination by alignment score  "
        f"(260517 Pos{pos}, {len(frame_paths)} frames, grid z={args.grid_z})",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cellfree_discrimination.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure: {out_png}")

    flat = {f"{n}_{k}": float(v) for n, d in summary.items() for k, v in d.items()}
    save_figure(
        fig,
        params={**flat, "pos": pos, "grid_z": args.grid_z,
                "n_frames": len(frame_paths), "prod_thresh": PROD_THRESH,
                "cell_channels": sorted(cell_set)},
        description=(
            "Discrimination of cell-bearing vs cell-free channels by per-channel "
            "alignment score against the cell-free grid(0,0) reference. Compares "
            "ECC-uint8 corr (production), ECC-float corr, and NCC peak value "
            "(TM_CCOEFF_NORMED max, the 2D-Gaussian/SG score). Tests whether 0.99 "
            "separates them and which score separates best (ROC-AUC)."),
        data={f"score_{n}": scores[n] for n in SCORE_NAMES},
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
