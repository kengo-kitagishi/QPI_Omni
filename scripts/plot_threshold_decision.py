"""plot_threshold_decision.py -- publication figure justifying ecc_min_corr.

Loads the per-channel alignment-score data produced by
bench_cellfree_discrimination.py and renders the decision rationale for the
cell / cell-free channel threshold:

  (A) score distributions of cell vs cell-free channels, with the chosen
      threshold and the false-positive / false-negative tails shaded;
  (B) error-rate trade-off vs threshold (false drop of cell-free channels,
      false keep of cell channels, balanced accuracy) with the chosen and
      previous thresholds marked;
  (C) ROC curve with the operating points marked and AUC annotated.

Detection convention (positive = "channel has cells", i.e. should be dropped):
  drop if score < threshold.
    TP : cell channel dropped        (score <  t)
    FN : cell channel kept (leaks!)  (score >= t)
    FP : cell-free channel dropped   (score <  t)  -> lost good data
    TN : cell-free channel kept      (score >= t)

Default score = "ECC-float (production)" -- the estimator production actually
uses. Cell channels: 0,3,5,9,10.

Usage
-----
    python scripts/plot_threshold_decision.py
    python scripts/plot_threshold_decision.py --threshold 0.994 --prev 0.99
    python scripts/plot_threshold_decision.py --score "NCC peak (2D-Gauss/SG)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

OUT_DIR = r"E:\260517\ecc_sg_ab"
INBOX_GLOB_BASE = r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\figure-hub\inbox"
CELL_CHANNELS = [0, 3, 5, 9, 10]

# Colorblind-safe palette
C_FREE = "#0072B2"   # cell-free (blue)
C_CELL = "#D55E00"   # cell (vermillion)
C_CHOSEN = "#009E73" # chosen threshold (green)
C_PREV = "#999999"   # previous threshold (grey)


def find_latest_npz():
    cands = sorted(Path(INBOX_GLOB_BASE).glob(
        "*/bench_cellfree_discrimination/*/*_data.npz"),
        key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError("no bench_cellfree_discrimination *_data.npz found")
    return cands[-1]


def rates(noncell, cell, t):
    """Return dict of detection rates at threshold t (drop if score < t)."""
    fp = float((noncell < t).mean())        # cell-free wrongly dropped
    tn = 1.0 - fp
    tp = float((cell < t).mean())            # cell correctly dropped
    fn = 1.0 - tp                            # cell kept (leak)
    ba = 0.5 * (tp + tn)
    return dict(fp=fp, tn=tn, tp=tp, fn=fn, ba=ba)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=None, help="score npz (default: latest bench run)")
    p.add_argument("--score", default="ECC-float (production)",
                   help="which score key to plot")
    p.add_argument("--threshold", type=float, default=0.994, help="chosen threshold")
    p.add_argument("--prev", type=float, default=0.99, help="previous threshold for reference")
    p.add_argument("--out-dir", default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    npz_path = Path(args.npz) if args.npz else find_latest_npz()
    d = np.load(npz_path, allow_pickle=True)
    key = f"score_{args.score}"
    if key not in d:
        raise KeyError(f"{key} not in {list(d.keys())}")
    arr = d[key]  # (n_frames, n_ch)
    n_ch = arr.shape[1]
    cell_idx = [c for c in CELL_CHANNELS if c < n_ch]
    free_idx = [c for c in range(n_ch) if c not in cell_idx]

    cell = arr[:, cell_idx].ravel()
    free = arr[:, free_idx].ravel()
    cell = cell[np.isfinite(cell)]
    free = free[np.isfinite(free)]

    t, tprev = args.threshold, args.prev
    r = rates(free, cell, t)
    rp = rates(free, cell, tprev)

    # AUC (cell detection: lower score = more cell-like) via rank-sum.
    allv = np.concatenate([free, cell])
    ranks = allv.argsort().argsort() + 1.0
    auc_free_high = (ranks[:len(free)].sum() - len(free) * (len(free) + 1) / 2) / (len(free) * len(cell))
    auc = float(auc_free_high)  # P(free > cell) = cell-detection AUC

    # Filename-safe tag from the score name.
    tag = (args.score.lower().replace(" ", "_").replace("(", "").replace(")", "")
           .replace("/", "_").replace("-", "_").replace("__", "_").strip("_"))

    # Threshold sweep -- the core decision table (drop if score < t).
    lo = max(0.90, float(np.floor(min(cell.min(), free.min()) * 50) / 50))
    ts = np.linspace(lo, 0.9999, 400)
    fp_s = np.array([float((free < tt).mean()) for tt in ts])       # cell-free dropped
    fn_s = np.array([float((cell >= tt).mean()) for tt in ts])      # cell kept (leak)
    ba_s = np.array([0.5 * (float((cell < tt).mean()) + float((free >= tt).mean())) for tt in ts])
    t_best = float(ts[np.argmax(ba_s)])

    print(f"npz: {npz_path.name}")
    print(f"score: {args.score}   n_free={len(free)}  n_cell={len(cell)}  AUC={auc:.4f}")
    print(f"@ chosen t={t}:  FP(false-drop free)={r['fp']*100:.1f}%  "
          f"FN(cell leak)={r['fn']*100:.1f}%  BA={r['ba']:.3f}")
    print(f"@ prev   t={tprev}: FP={rp['fp']*100:.1f}%  FN={rp['fn']*100:.1f}%  BA={rp['ba']:.3f}")
    print(f"best-BA threshold = {t_best:.4f}")

    # ----- publication style -----
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.linewidth": 1.0, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "figure.dpi": 200,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # ===== Panel A: score distributions + FP/FN tails =====
    ax = axes[0]
    bins = np.linspace(lo, 1.0, 60)
    hf, _ = np.histogram(free, bins=bins, density=True)
    hc, _ = np.histogram(cell, bins=bins, density=True)
    ctrs = 0.5 * (bins[:-1] + bins[1:])
    ax.step(ctrs, hf, where="mid", color=C_FREE, lw=1.8, label=f"cell-free ch (n={len(free)})")
    ax.step(ctrs, hc, where="mid", color=C_CELL, lw=1.8, label=f"cell ch (n={len(cell)})")
    # FP tail: cell-free below threshold (lost good data).
    mfp = ctrs < t
    ax.fill_between(ctrs, 0, hf, where=mfp, step="mid", color=C_FREE, alpha=0.30,
                    label=f"FP: cell-free dropped ({r['fp']*100:.1f}%)")
    # FN tail: cell at/above threshold (leak into average).
    mfn = ctrs >= t
    ax.fill_between(ctrs, 0, hc, where=mfn, step="mid", color=C_CELL, alpha=0.30,
                    label=f"FN: cell kept / leak ({r['fn']*100:.1f}%)")
    ax.axvline(t, color=C_CHOSEN, lw=2.0, label=f"chosen t = {t}")
    ax.axvline(tprev, color=C_PREV, lw=1.4, ls="--", label=f"previous t = {tprev}")
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel("alignment correlation vs cell-free grid")
    ax.set_ylabel("probability density")
    ax.set_title(f"(A) score distributions\n{args.score}")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # ===== Panel B: error-rate trade-off vs threshold =====
    ax = axes[1]
    ax.plot(ts, fp_s * 100, color=C_FREE, lw=2.0, label="FP: cell-free dropped")
    ax.plot(ts, fn_s * 100, color=C_CELL, lw=2.0, label="FN: cell kept (leak)")
    ax2 = ax.twinx()
    ax2.plot(ts, ba_s, color="k", lw=1.6, ls="-.", label="balanced accuracy")
    ax2.set_ylabel("balanced accuracy")
    ax2.set_ylim(0.5, 1.0)
    ax.axvline(t, color=C_CHOSEN, lw=2.0)
    ax.axvline(tprev, color=C_PREV, lw=1.4, ls="--")
    ax.annotate(f"chosen {t}\nFP {r['fp']*100:.1f}% / FN {r['fn']*100:.1f}%",
                xy=(t, max(r['fp'], r['fn']) * 100), xytext=(8, 18),
                textcoords="offset points", fontsize=9, color=C_CHOSEN,
                arrowprops=dict(arrowstyle="->", color=C_CHOSEN, lw=1.0))
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel("threshold")
    ax.set_ylabel("error rate (%)")
    ax.set_title(f"(B) error trade-off vs threshold\n(best BA at t = {t_best:.4f})")
    # combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="center left")
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    # ===== Panel C: ROC =====
    ax = axes[2]
    roc_t = np.linspace(0.0, 1.0001, 600)
    tpr = np.array([(cell < tt).mean() for tt in roc_t])   # cell dropped
    fpr = np.array([(free < tt).mean() for tt in roc_t])   # cell-free dropped
    ax.plot(fpr, tpr, color="#444444", lw=2.0)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    for tt, col, lab in [(t, C_CHOSEN, f"t={t}"), (tprev, C_PREV, f"t={tprev}")]:
        fx = float((free < tt).mean())
        ty = float((cell < tt).mean())
        ax.scatter([fx], [ty], s=70, color=col, zorder=5, edgecolors="k", linewidths=0.6)
        ax.annotate(lab, xy=(fx, ty), xytext=(10, -4), textcoords="offset points",
                    fontsize=9, color=col)
    ax.set_xlabel("FP rate  (cell-free wrongly dropped)")
    ax.set_ylabel("TP rate  (cell correctly dropped)")
    ax.set_title(f"(C) ROC\nAUC = {auc:.3f}")
    ax.set_xlim(-0.02, 0.6)
    ax.set_ylim(0.4, 1.02)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Threshold choice for cell / cell-free channel discrimination "
        f"(260517 Pos1, {args.score})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"threshold_decision_{tag}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"threshold_decision_{tag}.pdf", bbox_inches="tight")
    print(f"Saved: {out_png}")

    # ---- Human-readable numeric exports (necessary + sufficient to replot) ----
    import csv
    # 1. Raw pooled scores (the source of truth -- regenerates every panel).
    sweep_csv = out_dir / f"threshold_sweep_{tag}.csv"
    with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "FP_cellfree_dropped", "FN_cell_kept_leak",
                    "balanced_accuracy", "TP_cell_dropped", "TPR=cell_dropped",
                    "FPR=cellfree_dropped"])
        for tt, fpv, fnv, bav in zip(ts, fp_s, fn_s, ba_s):
            tpr_v = float((cell < tt).mean())
            fpr_v = float((free < tt).mean())
            w.writerow([f"{tt:.5f}", f"{fpv:.5f}", f"{fnv:.5f}", f"{bav:.5f}",
                        f"{tpr_v:.5f}", f"{tpr_v:.5f}", f"{fpr_v:.5f}"])
    scores_csv = out_dir / f"scores_pooled_{tag}.csv"
    with open(scores_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "score"])
        for v in free:
            w.writerow(["cell_free", f"{v:.6f}"])
        for v in cell:
            w.writerow(["cell", f"{v:.6f}"])
    # 2. Per-channel score matrix (frame x channel), for full reproducibility.
    perchan_npz = out_dir / f"scores_per_channel_{tag}.npz"
    np.savez_compressed(perchan_npz, scores=arr, cell_idx=np.array(cell_idx),
                        free_idx=np.array(free_idx), score_name=args.score)
    print(f"Saved data: {sweep_csv.name}, {scores_csv.name}, {perchan_npz.name}")

    save_figure(
        fig,
        params={"score": args.score, "score_tag": tag, "threshold": t, "prev": tprev,
                "fp_at_t": r["fp"], "fn_at_t": r["fn"], "ba_at_t": r["ba"],
                "tp_at_t": r["tp"], "tn_at_t": r["tn"],
                "fp_at_prev": rp["fp"], "fn_at_prev": rp["fn"], "ba_at_prev": rp["ba"],
                "auc": auc, "best_ba_threshold": t_best,
                "n_free": len(free), "n_cell": len(cell),
                "cell_channels": cell_idx, "free_channels": free_idx,
                "source_npz": npz_path.name},
        description=(
            "Decision figure for the cell/cell-free channel threshold ecc_min_corr "
            f"using the {args.score} alignment score. (A) score distributions with "
            "FP/FN tails shaded, (B) FP/FN/balanced-accuracy vs threshold, (C) ROC. "
            "Saved data is necessary+sufficient to replot: raw pooled cell/cell-free "
            "scores, the per-channel score matrix, the full threshold sweep "
            "(FP/FN/BA/TPR/FPR), and the ROC curve."),
        data={
            # raw source of truth
            "cell_scores": cell, "free_scores": free,
            "scores_per_channel": arr,
            "cell_idx": np.array(cell_idx), "free_idx": np.array(free_idx),
            # derived (for convenience)
            "sweep_threshold": ts, "sweep_fp": fp_s, "sweep_fn": fn_s, "sweep_ba": ba_s,
            "roc_fpr": fpr, "roc_tpr": tpr, "roc_threshold": roc_t,
        },
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
