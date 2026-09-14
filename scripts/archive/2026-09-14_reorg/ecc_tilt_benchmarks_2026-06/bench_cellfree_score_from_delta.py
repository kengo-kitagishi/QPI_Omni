"""bench_cellfree_score_from_delta.py -- which score separates cell-bearing channels?

Production drops cell-bearing channels from the drift average by thresholding the
ECC-float correlation against the cell-free grid reference (`ecc_min_corr`). If the
displacement estimator moves to the NCC/Gaussian-2D path, the natural score becomes
the NCC peak value. This script asks which of the two separates better, on the
current session's data.

The label is derived independently of both scores: the online crop_sub delta
(timelapse minus grid) carries cell signal directly, so a channel is labelled
cell-bearing when the delta's 95th percentile exceeds --label-thresh. On 260819
that statistic is cleanly bimodal (empty <= 0.14 rad, cells >= 0.43 rad), so the
label does not depend on a delicate cut.

Scores compared, both with production preprocessing (tilt_fit_crop at the
production ECC_CROP_H / TILT_CROP_H, float input, no uint8):
  ECC-float corr : cv2.findTransformECC correlation (what production thresholds)
  NCC peak       : cv2.matchTemplate(TM_CCOEFF_NORMED).max()

Read-only: nothing under the grid or the timelapse tree is written.

Usage
-----
    python scripts/bench_cellfree_score_from_delta.py \
        --grid-dir "E:\\260819\\grid_ye_1" --pos 3 14 18 32 73 85 \
        --tl-root "D:\\AquisitionData\\Kitagishi\\260819\\ph_zstack_1" \
        --delta-root "D:\\AquisitionData\\Kitagishi\\260819\\online_crop_sub_zstack"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from ecc_utils import ECC_MIN_CORR, tilt_fit_crop, to_ecc_input, ecc_align
from figure_logger import save_figure

# A production timelapse is running on this machine; leave it CPU headroom.
cv2.setNumThreads(2)

ECC_VMIN, ECC_VMAX = -5.0, 2.0
ECC_CROP_H = 80
TILT_CROP_H = 270
NCC_MARGIN = 14


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid-dir", required=True)
    p.add_argument("--tl-root", required=True, help="timelapse root holding PosN/z000/output_phase")
    p.add_argument("--delta-root", required=True, help="crop_sub root holding PosN/.../crop_sub_rawraw")
    p.add_argument("--pos", type=int, nargs="+", required=True)
    p.add_argument("--grid-z", type=int, default=3)
    p.add_argument("--tl-z", type=int, default=0)
    p.add_argument("--pos-split", type=int, default=51)
    p.add_argument("--frames", type=int, default=4, help="most recent frames averaged per channel")
    p.add_argument("--label-thresh", type=float, default=0.3,
                   help="delta 95th percentile (rad) above which a channel counts as cell-bearing")
    p.add_argument("--edge-margin", type=int, default=10)
    return p.parse_args()


def roc_auc(score, label):
    """AUC for 'higher score means cell-free' (label True = cell-bearing)."""
    s = np.asarray(score, float)
    y = np.asarray(label, bool)
    pos, neg = s[~y], s[y]          # cell-free should score higher
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def best_threshold(score, label):
    """Threshold maximising balanced accuracy; keep score >= thr as cell-free."""
    s = np.asarray(score, float)
    y = np.asarray(label, bool)
    best, best_thr = -1.0, np.nan
    for thr in np.unique(s):
        keep_free = (s[~y] >= thr).mean()
        drop_cell = (s[y] < thr).mean()
        ba = 0.5 * (keep_free + drop_cell)
        if ba > best:
            best, best_thr = ba, thr
    return best_thr, best


def main():
    args = parse_args()
    grid_dir = Path(args.grid_dir)
    rows = []

    for pos in args.pos:
        rois_path = (grid_dir / f"Pos{pos}_x+0_y+0" / "output_phase" / "channels"
                     / "channel_rois.json")
        rois = json.loads(rois_path.read_text(encoding="utf-8"))
        fit_right = pos >= args.pos_split

        ref_paths = sorted((grid_dir / f"Pos{pos}_x+0_y+0" / "output_phase")
                           .glob(f"img_*_ph_{args.grid_z:03d}_phase.tif"))
        tl_paths = sorted((Path(args.tl_root) / f"Pos{pos}" / f"z{args.tl_z:03d}"
                           / "output_phase").glob("img_*_phase.tif"))
        if not ref_paths or not tl_paths:
            print(f"Pos{pos}: missing reference or timelapse frames, skipped")
            continue
        tl_paths = tl_paths[-args.frames:]

        ref_img = tifffile.imread(str(ref_paths[0])).astype(np.float64)
        h_img = ref_img.shape[0]
        tl_imgs = [tifffile.imread(str(p)).astype(np.float64) for p in tl_paths]

        for ch, roi in enumerate(rois):
            hw = roi["crop_w"] // 2
            if min(roi["cy"] - hw, (h_img - 1) - (roi["cy"] + hw)) < args.edge_margin:
                continue
            ref = tilt_fit_crop(ref_img, roi["cy"], roi["cx"], roi["crop_w"],
                                ecc_crop_h=ECC_CROP_H, tilt_crop_h=TILT_CROP_H,
                                fit_right=fit_right)
            if ref is None:
                continue

            dfiles = sorted((Path(args.delta_root) / f"Pos{pos}" / "output_phase" / "channels"
                             / "crop_sub_rawraw" / f"z{args.tl_z:03d}" / f"ch{ch:02d}").glob("*.tif"))
            if not dfiles:
                continue
            p95 = float(np.mean([np.percentile(tifffile.imread(str(f)), 95)
                                 for f in dfiles[-args.frames:]]))

            eccs, nccs = [], []
            for tl in tl_imgs:
                mov = tilt_fit_crop(tl, roi["cy"], roi["cx"], roi["crop_w"],
                                    ecc_crop_h=ECC_CROP_H, tilt_crop_h=TILT_CROP_H,
                                    fit_right=fit_right)
                if mov is None:
                    continue
                r = ecc_align(to_ecc_input(ref, ECC_VMIN, ECC_VMAX),
                              to_ecc_input(mov, ECC_VMIN, ECC_VMAX))
                if r is not None:
                    eccs.append(r[2])
                m = NCC_MARGIN
                tmpl = mov[m:mov.shape[0] - m, m:mov.shape[1] - m].astype(np.float32)
                if tmpl.size and tmpl.shape[0] > 0 and tmpl.shape[1] > 0:
                    nccs.append(float(cv2.matchTemplate(ref.astype(np.float32), tmpl,
                                                        cv2.TM_CCOEFF_NORMED).max()))
            if not eccs or not nccs:
                continue
            rows.append(dict(pos=pos, ch=ch, delta_p95=p95,
                             cell=p95 > args.label_thresh,
                             ecc=float(np.mean(eccs)), ncc=float(np.mean(nccs))))
        print(f"Pos{pos}: {sum(r['pos'] == pos for r in rows)} channels measured", flush=True)

    if not rows:
        raise RuntimeError("no channel measured")

    cell = np.array([r["cell"] for r in rows])
    ecc = np.array([r["ecc"] for r in rows])
    ncc = np.array([r["ncc"] for r in rows])
    print(f"\nchannels: {len(rows)}   cell-bearing: {cell.sum()}   cell-free: {(~cell).sum()}")

    print(f"\n{'score':12s} {'AUC':>7} {'thr':>8} {'keep free':>10} {'drop cell':>10} {'best thr':>9} {'best BA':>8}")
    summary = {}
    for name, s, prod_thr in (("ECC-float", ecc, ECC_MIN_CORR), ("NCC peak", ncc, ECC_MIN_CORR)):
        auc = roc_auc(s, cell)
        keep = (s[~cell] >= prod_thr).mean()
        drop = (s[cell] < prod_thr).mean()
        bthr, bba = best_threshold(s, cell)
        kb = (s[~cell] >= bthr).mean()
        db = (s[cell] < bthr).mean()
        summary[name] = dict(auc=auc, prod_thr=prod_thr, keep=keep, drop=drop,
                             best_thr=float(bthr), best_ba=bba, keep_best=kb, drop_best=db)
        print(f"{name:12s} {auc:7.4f} {prod_thr:8.4f} {keep*100:9.1f}% {drop*100:9.1f}% "
              f"{bthr:9.4f} {bba:8.4f}")
        print(f"{'':12s} at best thr: keep free {kb*100:.1f}%  drop cell {db*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, s) in zip(axes, (("ECC-float corr", ecc), ("NCC peak", ncc))):
        for lab, mask, color in (("cell-free", ~cell, "#1f77b4"), ("cell-bearing", cell, "#d62728")):
            ax.scatter(np.random.default_rng(0).normal(0 if lab == "cell-free" else 1,
                                                       0.06, mask.sum()),
                       s[mask], s=14, alpha=0.7, color=color, label=lab)
        ax.axhline(ECC_MIN_CORR, color="k", ls="--", lw=0.8, label=f"{ECC_MIN_CORR}")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["cell-free", "cell-bearing"])
        ax.set_ylabel(name)
        ax.set_title(f"{name}  (AUC {roc_auc(s, cell):.4f})")
        ax.legend(fontsize=7)
    fig.suptitle("Cell/cell-free separability, label from crop_sub delta p95", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(
        fig,
        params={"pos": args.pos, "grid_z": args.grid_z, "frames": args.frames,
                "label_thresh": args.label_thresh, "edge_margin": args.edge_margin,
                "ecc_min_corr": ECC_MIN_CORR, "summary": summary},
        description=("Which score separates cell-bearing from cell-free channels: "
                     "production ECC-float correlation vs the NCC peak used by the "
                     "Gaussian-2D path. Labels come from the crop_sub delta, "
                     "independent of both scores."),
        data={"pos": np.array([r["pos"] for r in rows]),
              "ch": np.array([r["ch"] for r in rows]),
              "delta_p95": np.array([r["delta_p95"] for r in rows]),
              "cell": cell, "ecc_corr": ecc, "ncc_peak": ncc},
    )
    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
