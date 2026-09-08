"""Per-channel volume / mean RI / mass-density traces, replotted from npz.

Loads the *_data.npz sidecars produced by batch_volume_trace_overlay
(which already contain frame_index, volume_um3_rod, mean_ri, mass_pg per series)
and emits one 3-panel figure per series. No filesystem rescan.

Mass density = (mass_pg / volume_um3_rod) * 1000  (a.u.; matches the upstream
factor-of-1000 convention used in the overlay's mass panel).

Usage:
    python batch_per_channel_traces.py --from-npz <path/to/*_data.npz> [...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from batch_volume_trace_overlay import load_series_from_npz
from central_cell_track_figures import apply_style
from figure_logger import save_figure


def make_per_channel_trace(
    label: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> plt.Figure:
    x = df["frame_index"].to_numpy(dtype=float)
    if args.time_interval_min is not None:
        x = x * args.time_interval_min
        x_label = "Time [min]"
    else:
        x_label = "Frame"

    volume = df["volume_um3_rod"].to_numpy(dtype=float)
    mean_ri = df["mean_ri"].to_numpy(dtype=float)
    mass_pg = df["mass_pg"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mass_density = np.where(volume > 0, (mass_pg / volume) * 1000.0, np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.9), constrained_layout=True, sharex=True)

    axes[0].plot(x, volume, color="#1f77b4", lw=1.2)
    axes[0].set_title(f"A  Rod volume  [{label}]", loc="left")
    axes[0].set_ylabel("Volume [um^3]")
    axes[0].set_ylim(*args.volume_ylim)
    axes[0].grid(True, alpha=0.3, linestyle="--")

    if np.isfinite(mean_ri).any():
        axes[1].plot(x, mean_ri, color="#ff7f0e", lw=1.2)
    axes[1].set_title("B  Mean RI", loc="left")
    axes[1].set_ylabel("Mean RI")
    axes[1].set_ylim(*args.mean_ri_ylim)
    axes[1].grid(True, alpha=0.3, linestyle="--")

    if np.isfinite(mass_density).any():
        axes[2].plot(x, mass_density, color="#2ca02c", lw=1.2)
    axes[2].set_title("C  Mass density  (mass / volume x 1000)", loc="left")
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("Mass density [a.u.]")
    axes[2].grid(True, alpha=0.3, linestyle="--")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return fig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from-npz",
        type=Path,
        nargs="+",
        required=True,
        help="One or more *_data.npz files saved by batch_volume_trace_overlay.",
    )
    p.add_argument("--time-interval-min", type=float, default=None)
    p.add_argument("--preset", choices=["manuscript", "presentation", "qc"], default="manuscript")
    p.add_argument("--volume-ylim", nargs=2, type=float, default=[0.0, 400.0])
    p.add_argument("--mean-ri-ylim", nargs=2, type=float, default=[1.34, 1.37])
    p.add_argument("--no-save", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    apply_style(args.preset)

    n_total = 0
    for npz in args.from_npz:
        npz_path = npz.expanduser().resolve()
        if not npz_path.is_file():
            print(f"skip (not a file): {npz_path}")
            continue
        series, meta = load_series_from_npz(npz_path)
        if args.time_interval_min is None and "time_interval_min" in meta:
            args.time_interval_min = meta["time_interval_min"]
        print(f"[per_channel] {npz_path.name}: {len(series)} series")
        for label, df in series:
            fig = make_per_channel_trace(label, df, args)
            if not args.no_save:
                save_figure(
                    fig,
                    params={
                        "label": label,
                        "source_npz": str(npz_path),
                        "time_interval_min": args.time_interval_min,
                    },
                    description=f"Per-channel volume / mean RI / mass-density (x1000) for {label}",
                )
            plt.close(fig)
            n_total += 1

    print(f"[per_channel] wrote {n_total} figures.")
    return 0 if n_total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
