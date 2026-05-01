"""
Temporary: volume trace overlay with 1px mask erosion and thin lines (lw=0.5).
Also outputs normal (no erosion) version with thin lines.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_erosion
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from central_cell_track_figures import (
    apply_style,
    build_frame_pairs,
    calc_optical_metrics,
    calc_rod_volume_um3,
    load_raw_image,
    select_center_cell_for_frame,
)
from batch_volume_trace_overlay import make_volume_trace_overlay, series_to_npz_dict
from figure_logger import save_figure

PH_ROOT = Path(r"F:\260405\ph_260405")
CROP_SUB = "crop_sub_rawraw"

# All available channels
ALL_TARGETS: list[tuple[str, str]] = [
    ("Pos9", "ch00"),
    ("Pos9", "ch01"),
    ("Pos9", "ch02"),
    ("Pos9", "ch04"),
    ("Pos9", "ch08"),
    ("Pos9", "ch09"),
    ("Pos9", "ch10"),
    ("Pos16", "ch04"),
]


def eroded_select_center_cell(pair, min_area=20, exclude_border=True):
    """Like select_center_cell_for_frame but erodes each mask label by 1px."""
    result = {
        "frame_index": pair.frame_index,
        "frame_name": pair.frame_name,
        "tracked": False,
        "label_id": 0,
        "area_px": 0,
        "centroid_y": np.nan,
        "centroid_x": np.nan,
        "bbox_ymin": 0,
        "bbox_ymax": 0,
        "bbox_xmin": 0,
        "bbox_xmax": 0,
        "major_axis_px": np.nan,
        "minor_axis_px": np.nan,
        "orientation_rad": np.nan,
        "integrated_intensity": np.nan,
    }
    if pair.raw_path is None or pair.mask_path is None:
        return result

    raw = load_raw_image(pair.raw_path)
    mask = tifffile.imread(pair.mask_path)

    if mask.max() == 0:
        return result

    # Erode each label by 1px
    eroded_mask = np.zeros_like(mask)
    for lbl in np.unique(mask):
        if lbl == 0:
            continue
        eroded = binary_erosion(mask == lbl, iterations=1)
        if eroded.any():
            eroded_mask[eroded] = lbl

    if eroded_mask.max() == 0:
        return result

    props = measure.regionprops(eroded_mask, intensity_image=raw.astype(float))
    if not props:
        return result

    h, w = eroded_mask.shape
    cy, cx = h / 2.0, w / 2.0

    valid = []
    for rp in props:
        if rp.area < min_area:
            continue
        if exclude_border:
            ymin, xmin, ymax, xmax = rp.bbox
            if ymin <= 0 or xmin <= 0 or ymax >= h or xmax >= w:
                continue
        valid.append(rp)

    if not valid:
        return result

    best = min(valid, key=lambda rp: (rp.centroid[0] - cy) ** 2 + (rp.centroid[1] - cx) ** 2)
    result["tracked"] = True
    result["label_id"] = best.label
    result["area_px"] = best.area
    result["centroid_y"] = best.centroid[0]
    result["centroid_x"] = best.centroid[1]
    ymin, xmin, ymax, xmax = best.bbox
    result["bbox_ymin"] = ymin
    result["bbox_ymax"] = ymax
    result["bbox_xmin"] = xmin
    result["bbox_xmax"] = xmax
    result["major_axis_px"] = best.major_axis_length
    result["minor_axis_px"] = best.minor_axis_length
    result["orientation_rad"] = best.orientation
    result["integrated_intensity"] = float(np.sum(raw[eroded_mask == best.label]))
    return result


def build_summary(pairs, erode=False):
    rows = []
    for pair in pairs:
        try:
            if erode:
                row = eroded_select_center_cell(pair)
            else:
                row = select_center_cell_for_frame(pair, min_area=20, exclude_border=True)
            rows.append(row)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["frame_label"] = df["frame_name"]
    px = 0.348
    df["total_phase"] = np.where(
        df["tracked"].to_numpy(dtype=bool),
        df["integrated_intensity"].to_numpy(dtype=float),
        np.nan,
    )
    df["volume_um3_rod"] = np.where(
        df["tracked"].to_numpy(dtype=bool),
        [
            calc_rod_volume_um3(ma, mi, px)
            if np.isfinite(ma) and np.isfinite(mi) and mi > 0
            else np.nan
            for ma, mi in zip(
                df["major_axis_px"].to_numpy(dtype=float),
                df["minor_axis_px"].to_numpy(dtype=float),
            )
        ],
        np.nan,
    )
    om = [
        calc_optical_metrics(tp, vol, px, 663.0, 1.333, 0.00018)
        for tp, vol in zip(
            df["total_phase"].to_numpy(dtype=float),
            df["volume_um3_rod"].to_numpy(dtype=float),
        )
    ]
    df["mean_ri"] = [v[0] for v in om]
    df["mean_concentration"] = [v[1] for v in om]
    df["mass_pg"] = [v[2] for v in om]
    return df


def get_available_targets():
    """Return targets that have inference_out with masks."""
    available = []
    for pos, ch in ALL_TARGETS:
        inf = PH_ROOT / pos / "output_phase" / "channels" / CROP_SUB / ch / "inference_out"
        if inf.is_dir() and list(inf.glob("*_masks.tif")):
            available.append((pos, ch))
    return available


def main():
    targets = get_available_targets()
    print(f"Available targets: {[(p, c) for p, c in targets]}")

    apply_style("manuscript")
    plt.rcParams["lines.linewidth"] = 0.25  # extra thin

    plot_ns = argparse.Namespace(
        time_interval_min=None,
        volume_ylim=(0.0, 400.0),
        mean_ri_ylim=(1.34, 1.37),
        mass_ylim=(0.0, 40000.0),
        preset="manuscript",
        vline_frames=[575.0, 875.0, 1439.0],
    )

    # ---- Plot 1: Normal mask, thin lines ----
    print("\n=== Normal mask, thin lines (lw=0.25) ===")
    series_normal = []
    for pos, ch in targets:
        ch_dir = PH_ROOT / pos / "output_phase" / "channels" / CROP_SUB / ch
        label = f"{pos}_{ch}"
        pairs = build_frame_pairs(ch_dir)
        if not pairs:
            continue
        df = build_summary(pairs, erode=False)
        if df.empty or df["volume_um3_rod"].isna().all():
            continue
        n = df["volume_um3_rod"].notna().sum()
        print(f"  {label}: {n} valid")
        series_normal.append((label, df))

    if series_normal:
        fig1 = make_volume_trace_overlay(series_normal, plot_ns)
        save_figure(
            fig1,
            params={
                "targets": [f"{p}_{c}" for p, c in targets],
                "n_series": len(series_normal),
                "lw": 0.25,
            },
            data=series_to_npz_dict(series_normal),
            description=f"Volume overlay {len(series_normal)}ch thin lines (lw=0.25)",
        )
        plt.close(fig1)
        print("  Normal plot saved.")

    # ---- Plot 2: Eroded mask, thin lines ----
    print("\n=== Eroded mask (-1px), thin lines (lw=0.25) ===")
    series_eroded = []
    for pos, ch in targets:
        ch_dir = PH_ROOT / pos / "output_phase" / "channels" / CROP_SUB / ch
        label = f"{pos}_{ch}_eroded"
        pairs = build_frame_pairs(ch_dir)
        if not pairs:
            continue
        print(f"  {pos}_{ch}: building eroded summary ...")
        df = build_summary(pairs, erode=True)
        if df.empty or df["volume_um3_rod"].isna().all():
            print(f"    skip")
            continue
        n = df["volume_um3_rod"].notna().sum()
        print(f"    OK: {n} valid")
        series_eroded.append((label, df))

    if series_eroded:
        fig2 = make_volume_trace_overlay(series_eroded, plot_ns)
        save_figure(
            fig2,
            params={
                "targets": [f"{p}_{c}" for p, c in targets],
                "n_series": len(series_eroded),
                "lw": 0.25,
                "mask_erosion_px": 1,
            },
            data=series_to_npz_dict(series_eroded),
            description=f"Volume overlay {len(series_eroded)}ch eroded mask (-1px) thin lines",
        )
        plt.close(fig2)
        print("  Eroded plot saved.")

    print("\nDone.")


if __name__ == "__main__":
    main()
