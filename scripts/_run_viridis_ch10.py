"""Run intensity_kymograph (viridis) for Pos51/ch10 and save a standalone colorbar.

Re-uses central_cell_track_figures.make_intensity_kymograph and figure_logger.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

from central_cell_track_figures import (
    DIRECT_RUN_CONFIG,
    apply_style,
    build_frame_pairs,
    build_summary_table,
    determine_crop_size,
    make_intensity_kymograph,
)
from figure_logger import save_figure


CH_DIR = Path(r"f:/260514/Pos51/output_phase/channels/crop_sub_raw_raw/ch10")
LABEL = "Pos51_ch10"
KYMO_VMIN = -0.5
KYMO_VMAX = 2.0


def build_args() -> argparse.Namespace:
    ns = argparse.Namespace(**DIRECT_RUN_CONFIG)
    ns.indir = CH_DIR
    ns.pixel_size_um = 0.348
    ns.time_interval_min = None
    ns.preset = "manuscript"
    ns.kymograph_vmin = KYMO_VMIN
    ns.kymograph_vmax = KYMO_VMAX
    ns._media_schedule_parsed = None
    ns._media_ri = None
    ns._n_milliq_resolved = None
    ns._calibration_id_used = None
    return ns


def make_standalone_colorbar() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(1.2, 3.4), constrained_layout=True)
    cb = ColorbarBase(
        ax,
        cmap=plt.get_cmap("viridis"),
        norm=Normalize(vmin=KYMO_VMIN, vmax=KYMO_VMAX),
        orientation="vertical",
    )
    cb.set_label("Mean intensity in axis-aligned rectangle")
    return fig


def main() -> int:
    args = build_args()
    apply_style(args.preset)

    pairs = build_frame_pairs(CH_DIR)
    if not pairs:
        print(f"no frame pairs in {CH_DIR}")
        return 1
    print(f"frame pairs: {len(pairs)}")

    summary_df = build_summary_table(
        pairs,
        min_area=args.min_area,
        exclude_border=args.exclude_border,
        pixel_size_um=args.pixel_size_um,
        wavelength_nm=args.wavelength_nm,
        n_medium=args.n_medium,
        alpha_ri=args.alpha_ri,
    )
    if summary_df.empty:
        print("summary empty")
        return 1

    crop_size = determine_crop_size(summary_df, args.crop_margin)
    fig_kymo = make_intensity_kymograph(summary_df, crop_size, args, variant_label=LABEL)
    if fig_kymo is None:
        print("kymograph could not be built")
        return 1

    save_figure(
        fig_kymo,
        params={
            "label": LABEL,
            "indir": str(CH_DIR),
            "kymograph_vmin": KYMO_VMIN,
            "kymograph_vmax": KYMO_VMAX,
            "n_frames": len(pairs),
        },
        description=f"Intensity kymograph (viridis) for {LABEL}",
    )
    plt.close(fig_kymo)

    fig_cbar = make_standalone_colorbar()
    save_figure(
        fig_cbar,
        params={
            "label": f"{LABEL}_colorbar",
            "cmap": "viridis",
            "vmin": KYMO_VMIN,
            "vmax": KYMO_VMAX,
            "axis_label": "Mean intensity in axis-aligned rectangle",
        },
        description=f"Standalone viridis colorbar for {LABEL} intensity kymograph",
    )
    plt.close(fig_cbar)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
