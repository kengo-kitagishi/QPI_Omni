"""
Evaluate every saved checkpoint of an omnipose training run on a small
validation sample, then plot per-epoch metrics so you can see where
overfitting starts and which checkpoint generalizes best.

Metrics per checkpoint:
  - n_cells_mean / n_cells_std : per-frame detected cell count
  - area_mean_um2 / area_cv    : per-cell area statistics across the val set
  - frame_to_frame_iou_median  : median IoU between consecutive-frame masks
                                 of overlapping cells (proxy for temporal stability)

Usage:
  python checkpoint_eval.py \\
    --model-dir "C:/Users/QPI/Desktop/train/omni_model_d20/models" \\
    --val-dir   "G:/マイドライブ/ch02" \\
    --n-frames 30 \\
    --pixel-size-um 0.348
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_logger import save_figure  # noqa: E402

EPOCH_RE = re.compile(r"_e(\d{4})$")


def find_checkpoints(model_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in model_dir.iterdir():
        m = EPOCH_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def sample_frames(val_dir: Path, n: int) -> list[Path]:
    files = sorted(val_dir.glob("*.tif"))
    files = [f for f in files if not f.name.endswith("_masks.tif")
             and "inference_out" not in f.parts]
    if not files:
        return []
    if len(files) <= n:
        return files
    idx = np.linspace(0, len(files) - 1, n).astype(int)
    return [files[i] for i in idx]


def per_cell_areas(mask: np.ndarray, pixel_size_um: float) -> np.ndarray:
    if mask is None:
        return np.zeros(0)
    ids, counts = np.unique(mask, return_counts=True)
    keep = ids > 0
    return counts[keep] * (pixel_size_um ** 2)


def frame_pair_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    if m1 is None or m2 is None:
        return np.nan
    if int(m1.max()) == 0 or int(m2.max()) == 0:
        return np.nan
    ids1 = np.unique(m1); ids1 = ids1[ids1 > 0]
    ids2 = np.unique(m2); ids2 = ids2[ids2 > 0]
    ious = []
    for i in ids1:
        a = (m1 == i)
        best = 0.0
        for j in ids2:
            b = (m2 == j)
            inter = int(np.logical_and(a, b).sum())
            if inter == 0:
                continue
            uni = int(np.logical_or(a, b).sum())
            iou = inter / uni
            if iou > best:
                best = iou
        if best > 0:
            ious.append(best)
    return float(np.median(ious)) if ious else np.nan


def evaluate_checkpoint(model_path: Path, frames: list[Path],
                        pixel_size_um: float, eval_kw: dict) -> dict:
    from cellpose_omni.models import CellposeModel
    model = CellposeModel(gpu=True, pretrained_model=str(model_path),
                          omni=True, nchan=1, nclasses=3, dim=2)
    masks = []
    for f in frames:
        img = tifffile.imread(str(f))
        try:
            m, _, _ = model.eval([img], **eval_kw)
            masks.append(np.asarray(m[0], dtype=np.int32))
        except Exception as e:
            print(f"  eval failed on {f.name}: {e!r}", flush=True)
            masks.append(np.zeros_like(img, dtype=np.int32))

    n_cells = np.array([int(m.max()) if m is not None else 0 for m in masks])
    all_areas = np.concatenate([per_cell_areas(m, pixel_size_um) for m in masks]
                               or [np.zeros(0)])
    ious = [frame_pair_iou(masks[i], masks[i + 1]) for i in range(len(masks) - 1)]
    ious = [v for v in ious if not np.isnan(v)]

    return dict(
        n_cells_mean=float(n_cells.mean()) if len(n_cells) else 0.0,
        n_cells_std=float(n_cells.std()) if len(n_cells) else 0.0,
        n_cells_total=int(n_cells.sum()),
        area_mean_um2=float(all_areas.mean()) if len(all_areas) else 0.0,
        area_cv=float(all_areas.std() / all_areas.mean())
                if len(all_areas) and all_areas.mean() > 0 else 0.0,
        f2f_iou_median=float(np.median(ious)) if ious else np.nan,
        n_frames=len(frames),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--val-dir", required=True, type=Path)
    p.add_argument("--n-frames", type=int, default=30)
    p.add_argument("--pixel-size-um", type=float, default=0.348)
    p.add_argument("--diameter", type=float, default=20)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    args = p.parse_args()

    cps = find_checkpoints(args.model_dir)
    if not cps:
        print(f"No checkpoints with _eNNNN suffix in {args.model_dir}")
        return 1
    frames = sample_frames(args.val_dir, args.n_frames)
    if not frames:
        print(f"No frames in {args.val_dir}")
        return 1
    print(f"checkpoints: {[e for e, _ in cps]}", flush=True)
    print(f"validation frames: {len(frames)} from {args.val_dir.name}", flush=True)

    eval_kw = dict(
        channels=None, channel_axis=None,
        diameter=args.diameter, normalize=True, tile=False,
        net_avg=True, omni=True, verbose=False,
        flow_threshold=args.flow_threshold,
        mask_threshold=0, min_size=10,
    )

    rows = []
    for epoch, path in cps:
        print(f"\n--- epoch {epoch} ---", flush=True)
        try:
            stats = evaluate_checkpoint(path, frames, args.pixel_size_um, eval_kw)
        except Exception as e:
            print(f"!! failed: {e!r}", flush=True)
            continue
        stats["epoch"] = epoch
        rows.append(stats)
        print(f"  n_cells={stats['n_cells_mean']:.1f}+-{stats['n_cells_std']:.1f}, "
              f"area={stats['area_mean_um2']:.2f}um^2 (cv={stats['area_cv']:.2f}), "
              f"f2f_iou={stats['f2f_iou_median']:.3f}", flush=True)

    df = pd.DataFrame(rows).sort_values("epoch")
    print("\n=== results ===")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"Checkpoint sweep on {args.val_dir.name} "
                 f"(N={len(frames)} frames, d={args.diameter}, flow={args.flow_threshold})")

    axes[0, 0].errorbar(df["epoch"], df["n_cells_mean"], yerr=df["n_cells_std"],
                        marker="o", capsize=3)
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("N cells / frame")
    axes[0, 0].set_title("Detection count (lower std = stable)")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(df["epoch"], df["area_mean_um2"], "o-", color="tab:orange")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Mean cell area (um^2)")
    axes[0, 1].set_title("Area drift (overfit may inflate or shrink)")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(df["epoch"], df["area_cv"], "o-", color="tab:green")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Area CV")
    axes[1, 0].set_title("Area CV (variability across cells)")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(df["epoch"], df["f2f_iou_median"], "o-", color="tab:red")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Frame-to-frame median IoU")
    axes[1, 1].set_title("Temporal mask stability (higher = better)")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save_figure(
        fig,
        params={
            "val_dir": args.val_dir.name,
            "n_frames": len(frames),
            "diameter": args.diameter,
            "flow_threshold": args.flow_threshold,
            "n_checkpoints": len(df),
        },
        description=f"Checkpoint sweep: {len(df)} epochs evaluated on {len(frames)} frames "
                    f"from {args.val_dir.name}; metrics for picking best epoch / detecting overfit",
    )

    out_csv = args.model_dir / "checkpoint_eval.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nCSV: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
