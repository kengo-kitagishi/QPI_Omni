"""
Temporary test script: Omnipose segmentation → volume trace overlay
for selected Pos/ch directories under crop_sub_rawraw.

Usage:
  cd scripts
  python test_seg_volume_overlay.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

# ── Configuration ────────────────────────────────────────────────────
PH_ROOT = Path(r"F:\260405\ph_260405")
CROP_SUB = "crop_sub_rawraw"

# 処理対象チャネル（Pos/ch の組）
TARGETS: list[tuple[str, str]] = [
    ("Pos9", "ch09"),
    ("Pos9", "ch10"),
    ("Pos16", "ch04"),
]

# Omnipose model
MODEL_PATH = (
    r"C:\Users\QPI\Desktop\train\omni_model\models"
    r"\cellpose_residual_on_style_on_concatenation_off_omni_abstract"
    r"_nclasses_3_nchan_1_dim_2_omni_model_2026_04_13_10_54_41.173761"
)
USE_GPU = True
NCHAN = 1
NCLASSES = 3

EVAL_PARAMS = dict(
    channels=None,
    channel_axis=None,
    diameter=15,
    normalize=True,
    tile=False,
    net_avg=True,
    omni=True,
    verbose=False,
    flow_threshold=0.11,
    mask_threshold=0,
    min_size=10,
)

# volume trace overlay parameters
PIXEL_SIZE_UM = 0.348
WAVELENGTH_NM = 663.0
N_MEDIUM = 1.333
ALPHA_RI = 0.00018
MIN_AREA = 20
EXCLUDE_BORDER = True


# ── Step 1: Segmentation ────────────────────────────────────────────
def run_segmentation(channel_dir: Path) -> None:
    """Run Omnipose inference on all TIFs in channel_dir, save masks to inference_out/."""
    from cellpose_omni import io
    from cellpose_omni.models import CellposeModel

    outdir = channel_dir / "inference_out"
    outdir.mkdir(exist_ok=True)

    # Check if already done (resume-friendly)
    existing_masks = set(p.stem.replace("_masks", "") for p in outdir.glob("*_masks.tif"))

    files = io.get_image_files(str(channel_dir), mask_filter="_masks", look_one_level_down=False)
    if not isinstance(files, (list, tuple)):
        files = list(files)
    proc_files = []
    for f in files:
        if isinstance(f, (list, tuple)):
            proc_files.append(f[0] if f else None)
        else:
            proc_files.append(f)
    files = [f for f in proc_files if f is not None]

    # Skip already-processed frames
    todo = [f for f in files if Path(f).stem not in existing_masks]
    print(f"  {len(files)} total, {len(existing_masks)} done, {len(todo)} to process")
    if not todo:
        print("  → All frames already segmented, skipping.")
        return

    model = CellposeModel(
        gpu=USE_GPU, pretrained_model=str(MODEL_PATH), omni=True, nchan=NCHAN, nclasses=NCLASSES, dim=2
    )

    t0 = time.time()
    processed = 0
    skipped = 0
    errors = 0

    for i, f in enumerate(todo, 1):
        base = Path(f).stem
        try:
            img = tifffile.imread(f)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {base}  X read error: {e}")
            empty = np.zeros((64, 64), dtype=np.uint16)
            tifffile.imwrite(str(outdir / f"{base}_masks.tif"), empty)
            errors += 1
            continue

        try:
            masks, flows, _ = model.eval([img], **EVAL_PARAMS)
        except ValueError as e:
            empty = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(str(outdir / f"{base}_masks.tif"), empty)
            skipped += 1
            continue
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {base}  X error: {e}")
            empty = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(str(outdir / f"{base}_masks.tif"), empty)
            errors += 1
            continue

        if masks is None or np.max(masks) == 0:
            empty = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(str(outdir / f"{base}_masks.tif"), empty)
            skipped += 1
            continue

        out_mask = masks[0].astype(np.uint16)
        tifffile.imwrite(str(outdir / f"{base}_masks.tif"), out_mask)
        processed += 1

        if i % 200 == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(todo) - i) / rate if rate > 0 else 0
            print(
                f"  [{i}/{len(todo)}] processed={processed} skip={skipped} err={errors} "
                f"({rate:.1f} img/s, ETA {eta/60:.0f}min)"
            )

    elapsed = time.time() - t0
    print(f"  Segmentation done: {processed} ok, {skipped} skip, {errors} err  ({elapsed:.0f}s)")


# ── Step 2: Volume trace overlay ─────────────────────────────────────
def run_volume_trace_overlay(
    targets: list[tuple[str, str]],
) -> None:
    """Build summary for each target channel and produce overlay plot."""
    # Import from the existing pipeline
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from central_cell_track_figures import (
        apply_style,
        build_frame_pairs,
        build_summary_table,
    )
    from batch_volume_trace_overlay import make_volume_trace_overlay
    from figure_logger import save_figure

    apply_style("manuscript")

    series: list[tuple[str, "pd.DataFrame"]] = []
    import pandas as pd

    for pos, ch in targets:
        channel_dir = PH_ROOT / pos / "output_phase" / "channels" / CROP_SUB / ch
        label = f"{pos}_{ch}"
        print(f"[overlay] Building summary for {label} ...")

        inf_dir = channel_dir / "inference_out"
        if not inf_dir.is_dir() or not list(inf_dir.glob("*_masks.tif")):
            print(f"  → Skipped (no masks)")
            continue

        pairs = build_frame_pairs(channel_dir)
        if not pairs:
            print(f"  → Skipped (no frame pairs)")
            continue

        summary_df = build_summary_table(
            pairs,
            min_area=MIN_AREA,
            exclude_border=EXCLUDE_BORDER,
            pixel_size_um=PIXEL_SIZE_UM,
            wavelength_nm=WAVELENGTH_NM,
            n_medium=N_MEDIUM,
            alpha_ri=ALPHA_RI,
        )

        if summary_df.empty or summary_df["volume_um3_rod"].isna().all():
            print(f"  → Skipped (no volume data)")
            continue

        series.append((label, summary_df))
        print(f"  → OK ({len(summary_df)} frames, {summary_df['volume_um3_rod'].notna().sum()} with volume)")

    if not series:
        print("[overlay] No series to plot.")
        return

    print(f"\n[overlay] Plotting {len(series)} series ...")
    plot_ns = argparse.Namespace(
        time_interval_min=None,
        volume_ylim=(0.0, 400.0),
        mean_ri_ylim=(1.34, 1.37),
        mass_ylim=(0.0, 40000.0),
        preset="manuscript",
        vline_frames=None,
    )

    fig = make_volume_trace_overlay(series, plot_ns)
    save_figure(
        fig,
        params={
            "ph_root": str(PH_ROOT),
            "crop_sub": CROP_SUB,
            "targets": [f"{p}_{c}" for p, c in targets],
            "n_series": len(series),
        },
        description="Test: seg→volume overlay for 5ch (Pos9)",
    )
    plt.close(fig)
    print("[overlay] Done. Figure saved via figure_logger.")


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("test_seg_volume_overlay")
    print(f"  PH_ROOT : {PH_ROOT}")
    print(f"  TARGETS : {TARGETS}")
    print("=" * 60)

    # Step 1: Segmentation
    for pos, ch in TARGETS:
        channel_dir = PH_ROOT / pos / "output_phase" / "channels" / CROP_SUB / ch
        if not channel_dir.is_dir():
            print(f"\n[seg] {pos}/{ch}: directory not found, skipping")
            continue
        print(f"\n[seg] {pos}/{ch}  ({channel_dir})")
        run_segmentation(channel_dir)

    # Step 2: Volume trace overlay
    print("\n" + "=" * 60)
    print("Volume trace overlay")
    print("=" * 60)
    run_volume_trace_overlay(TARGETS)


if __name__ == "__main__":
    main()
