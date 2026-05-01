"""
Scan ph-root/Pos*/output_phase/channels/crop_sub_rawraw/ch* for inference_out,
build per-channel summary like central_cell_track_figures, then overlay all
volume_trace-style series on one figure (labels: PosN_chXX).

Examples:
  cd scripts
  python batch_volume_trace_overlay.py --ph-root F:/path/to/ph_260405 --mass-ylim 0 40000
  python batch_volume_trace_overlay.py --ph-root ... --pos Pos1
  python batch_volume_trace_overlay.py --ph-root ... --quick-list   # fast filesystem scan only
  python batch_volume_trace_overlay.py --ph-root ... --list-only    # full volume validation, slow
  python batch_volume_trace_overlay.py --ph-root ... --pipette-vlines   # vertical dotted lines at 575,875,1439 frames
  python batch_volume_trace_overlay.py --from-npz path/to/*_data.npz --pipette-vlines   # replot from saved npz (no ph-root)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from central_cell_track_figures import (
    apply_style,
    build_frame_pairs,
    build_summary_table,
    natural_key,
)
from figure_logger import save_figure


def series_to_npz_dict(
    series: list[tuple[str, pd.DataFrame]],
    time_interval_min: float | None = None,
) -> dict[str, np.ndarray]:
    """save_figure(data=...) 用。load_series_from_npz と対になる。"""
    out: dict[str, np.ndarray] = {"n_series": np.array(len(series), dtype=np.int64)}
    for i, (label, df) in enumerate(series):
        out[f"label_{i}"] = np.array([label], dtype=object)
        out[f"frame_index_{i}"] = df["frame_index"].to_numpy(dtype=np.float64)
        out[f"volume_um3_rod_{i}"] = df["volume_um3_rod"].to_numpy(dtype=np.float64)
        out[f"mean_ri_{i}"] = df["mean_ri"].to_numpy(dtype=np.float64)
        out[f"mass_pg_{i}"] = df["mass_pg"].to_numpy(dtype=np.float64)
    if time_interval_min is not None:
        out["time_interval_min"] = np.array(float(time_interval_min), dtype=np.float64)
    return out


def load_series_from_npz(path: Path) -> tuple[list[tuple[str, pd.DataFrame]], dict[str, float]]:
    """
    figure_logger の *_data.npz（本スクリプトが data= で保存したもの）を読み込む。
    戻り値: (series, meta) meta は time_interval_min など。
    """
    z = np.load(path, allow_pickle=True)
    if "n_series" not in z.files:
        raise KeyError(
            f"npz has no 'n_series'. Keys: {sorted(z.files)!r}. "
            "Use a *_data.npz produced by this script with save_figure(data=...)."
        )
    n = int(z["n_series"])
    series: list[tuple[str, pd.DataFrame]] = []
    for i in range(n):
        lab = z[f"label_{i}"]
        label = str(lab[0]) if getattr(lab, "shape", ()) else str(lab)
        df = pd.DataFrame(
            {
                "frame_index": np.asarray(z[f"frame_index_{i}"], dtype=np.float64),
                "volume_um3_rod": np.asarray(z[f"volume_um3_rod_{i}"], dtype=np.float64),
                "mean_ri": np.asarray(z[f"mean_ri_{i}"], dtype=np.float64),
                "mass_pg": np.asarray(z[f"mass_pg_{i}"], dtype=np.float64),
            }
        )
        series.append((label, df))
    meta: dict[str, float] = {}
    if "time_interval_min" in z.files:
        meta["time_interval_min"] = float(z["time_interval_min"])
    return series, meta


def discover_pos_dirs(ph_root: Path) -> list[Path]:
    out = [p for p in ph_root.iterdir() if p.is_dir() and p.name.startswith("Pos")]
    return sorted(out, key=lambda p: natural_key(p.name))


def discover_channel_dirs(pos_root: Path, crop_sub: str) -> list[Path]:
    base = pos_root / "output_phase" / "channels" / crop_sub
    if not base.is_dir():
        return []
    dirs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("ch")]
    return sorted(dirs, key=lambda p: natural_key(p.name))


def channel_has_volume_trace_data(summary_df: pd.DataFrame) -> bool:
    if summary_df.empty:
        return False
    return not summary_df["volume_um3_rod"].isna().all()


def make_volume_trace_overlay(
    series: list[tuple[str, pd.DataFrame]],
    plot_ns: argparse.Namespace,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.9), constrained_layout=True, sharex=True)
    cmap = plt.get_cmap("tab20")
    n = len(series)

    if n == 0:
        raise ValueError("no series")

    first_x_label = "Time [min]" if plot_ns.time_interval_min is not None else "Frame"

    for idx, (label, df) in enumerate(series):
        color = cmap(idx % 20)
        x = df["frame_index"].to_numpy(dtype=float)
        if plot_ns.time_interval_min is not None:
            x = x * plot_ns.time_interval_min

        volume = df["volume_um3_rod"].to_numpy(dtype=float)
        mean_ri = df["mean_ri"].to_numpy(dtype=float)
        mass_pg = df["mass_pg"].to_numpy(dtype=float)

        axes[0].plot(x, volume, color=color, lw=1.0, label=label, alpha=0.9)
        if np.isfinite(mean_ri).any():
            axes[1].plot(x, mean_ri, color=color, lw=1.0, alpha=0.9)
        if np.isfinite(mass_pg).any():
            axes[2].plot(x, mass_pg, color=color, lw=1.0, alpha=0.9)

    axes[0].set_title("A  Rod volume estimate (all Pos / ch)", loc="left")
    axes[0].set_ylabel("Volume [um^3]")
    axes[0].set_ylim(*plot_ns.volume_ylim)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].set_title("B  Mean RI", loc="left")
    axes[1].set_ylabel("Mean RI")
    axes[1].set_ylim(*plot_ns.mean_ri_ylim)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[2].set_title("C  Total mass", loc="left")
    axes[2].set_xlabel(first_x_label)
    axes[2].set_ylabel("Total mass [pg]")
    axes[2].set_ylim(*plot_ns.mass_ylim)
    axes[2].grid(True, alpha=0.3, linestyle="--")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    ncol = 2 if n <= 16 else 3
    fontsize = 6 if n > 20 else 7
    axes[0].legend(loc="upper left", fontsize=fontsize, ncol=ncol, framealpha=0.92)

    vline_frames = getattr(plot_ns, "vline_frames", None)
    if vline_frames:
        for ax in axes:
            for fv in vline_frames:
                x_v = fv * plot_ns.time_interval_min if plot_ns.time_interval_min is not None else fv
                ax.axvline(
                    x_v,
                    color="0.35",
                    ls=":",
                    lw=1.0,
                    alpha=0.9,
                    zorder=1,
                )

    if plot_ns.preset != "manuscript":
        fig.suptitle(f"Volume overlay ({n} series)", y=1.02)
    return fig


def enumerate_candidate_dirs(
    ph_root: Path,
    crop_sub: str,
    pos_filter: str | None,
) -> tuple[list[tuple[str, Path]], list[tuple[str, str]]]:
    """Filesystem-only: inference_out exists and has at least one *_masks.tif."""
    candidates: list[tuple[str, Path]] = []
    skipped: list[tuple[str, str]] = []

    if pos_filter is not None:
        want = ph_root / pos_filter
        if not want.is_dir():
            return [], [(str(pos_filter), "pos_not_found")]
        pos_dirs = [want.resolve()]
    else:
        pos_dirs = discover_pos_dirs(ph_root)

    for pos_dir in pos_dirs:
        pos_name = pos_dir.name
        for channel_dir in discover_channel_dirs(pos_dir, crop_sub):
            label = f"{pos_name}_{channel_dir.name}"
            inf = channel_dir / "inference_out"
            if not inf.is_dir():
                skipped.append((label, "no_inference_out"))
                continue
            if not list(inf.glob("*_masks.tif")):
                skipped.append((label, "no_mask_tifs"))
                continue
            candidates.append((label, channel_dir))

    return candidates, skipped


def collect_series(
    ph_root: Path,
    crop_sub: str,
    pos_filter: str | None,
    args: argparse.Namespace,
) -> tuple[list[tuple[str, pd.DataFrame]], list[tuple[str, str]]]:
    included: list[tuple[str, pd.DataFrame]] = []
    skipped: list[tuple[str, str]] = []

    candidates, pre_skipped = enumerate_candidate_dirs(ph_root, crop_sub, pos_filter)
    skipped.extend(pre_skipped)

    for label, channel_dir in candidates:
        pairs = build_frame_pairs(channel_dir)
        if not pairs:
            skipped.append((label, "no_frame_pairs"))
            continue

        summary_df = build_summary_table(
            pairs,
            min_area=args.min_area,
            exclude_border=args.exclude_border,
            pixel_size_um=args.pixel_size_um,
            wavelength_nm=args.wavelength_nm,
            n_medium=args.n_medium,
            alpha_ri=args.alpha_ri,
        )
        if not channel_has_volume_trace_data(summary_df):
            skipped.append((label, "no_volume_trace_data"))
            continue
        included.append((label, summary_df))

    return included, skipped


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Overlay volume / mean RI / total mass for all Pos/ch with inference_out."
    )
    p.add_argument(
        "--ph-root",
        type=Path,
        default=None,
        help="Parent of Pos1, Pos2, ... (e.g. F:/.../ph_260405). Not needed with --from-npz.",
    )
    p.add_argument(
        "--from-npz",
        type=Path,
        default=None,
        help="Rebuild overlay from *_data.npz (from save_figure data=); skips filesystem scan.",
    )
    p.add_argument(
        "--crop-sub",
        type=str,
        default="crop_sub_rawraw",
        help="Folder under channels/ (default: crop_sub_rawraw).",
    )
    p.add_argument(
        "--pos",
        type=str,
        default=None,
        help="Only this position folder (e.g. Pos1). Default: all Pos* under ph-root.",
    )
    p.add_argument("--min-area", type=int, default=20)
    p.set_defaults(exclude_border=True)
    bg = p.add_mutually_exclusive_group()
    bg.add_argument("--exclude-border", dest="exclude_border", action="store_true")
    bg.add_argument("--allow-border", dest="exclude_border", action="store_false")
    p.add_argument("--pixel-size-um", type=float, default=0.348)
    p.add_argument("--time-interval-min", type=float, default=None)
    p.add_argument("--wavelength-nm", type=float, default=663.0)
    p.add_argument("--n-medium", type=float, default=1.333)
    p.add_argument("--alpha-ri", type=float, default=0.00018)
    p.add_argument(
        "--preset",
        choices=["manuscript", "presentation", "qc"],
        default="manuscript",
    )
    p.add_argument("--volume-ylim", nargs=2, type=float, default=[0.0, 400.0])
    p.add_argument("--mean-ri-ylim", nargs=2, type=float, default=[1.34, 1.37])
    p.add_argument("--mass-ylim", nargs=2, type=float, default=[0.0, 40000.0])
    p.add_argument(
        "--vline-frames",
        nargs="+",
        type=float,
        default=None,
        metavar="F",
        help="Vertical dotted lines at frame F (x = F, or F * --time-interval-min if time axis).",
    )
    p.add_argument(
        "--pipette-vlines",
        action="store_true",
        help="Shorthand for --vline-frames 575 875 1439 (exchange / glucose timing).",
    )
    p.add_argument(
        "--list-only",
        action="store_true",
        help="Run full validation (slow over many Pos) and exit without plotting.",
    )
    p.add_argument(
        "--quick-list",
        action="store_true",
        help="Fast: only check inference_out + mask tifs (no volume validation).",
    )
    p.add_argument("--no-save", action="store_true", help="Skip figure_logger.")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.from_npz is None and args.ph_root is None:
        print("[batch_volume_trace_overlay] error: need --ph-root or --from-npz")
        return 1
    if args.from_npz is not None and args.ph_root is not None:
        print("[batch_volume_trace_overlay] error: use only one of --ph-root or --from-npz")
        return 1

    apply_style(args.preset)

    if args.from_npz is not None:
        npz_path = args.from_npz.expanduser().resolve()
        if not npz_path.is_file():
            print(f"[batch_volume_trace_overlay] not a file: {npz_path}")
            return 1
        try:
            included, npz_meta = load_series_from_npz(npz_path)
        except (KeyError, ValueError, OSError) as e:
            print(f"[batch_volume_trace_overlay] failed to load npz: {e}")
            return 1
        if args.time_interval_min is None and "time_interval_min" in npz_meta:
            args.time_interval_min = npz_meta["time_interval_min"]
            print(f"[batch_volume_trace_overlay] time_interval_min from npz: {args.time_interval_min}")
        skipped: list[tuple[str, str]] = []
        print(f"[batch_volume_trace_overlay] from-npz: {npz_path}")
        print(f"[batch_volume_trace_overlay] included ({len(included)}): {[x[0] for x in included]}")
    else:
        ph_root = args.ph_root.expanduser().resolve()
        if not ph_root.is_dir():
            print(f"[batch_volume_trace_overlay] not a directory: {ph_root}")
            return 1

        if args.quick_list:
            candidates, skipped = enumerate_candidate_dirs(ph_root, args.crop_sub, args.pos)
            print(f"[batch_volume_trace_overlay] ph_root: {ph_root}")
            print(
                f"[batch_volume_trace_overlay] quick-list candidates ({len(candidates)}): "
                f"{[x[0] for x in candidates]}"
            )
            print(f"[batch_volume_trace_overlay] quick-list skipped ({len(skipped)}): {skipped}")
            return 0 if candidates or skipped else 1

        included, skipped = collect_series(
            ph_root,
            args.crop_sub,
            args.pos,
            args,
        )

        print(f"[batch_volume_trace_overlay] ph_root: {ph_root}")
        print(f"[batch_volume_trace_overlay] included ({len(included)}): {[x[0] for x in included]}")
        print(f"[batch_volume_trace_overlay] skipped ({len(skipped)}): {skipped}")

    if args.list_only:
        return 0 if (included or skipped) else 1

    if not included:
        print("[batch_volume_trace_overlay] nothing to plot.")
        return 1

    if args.pipette_vlines and args.vline_frames is not None:
        print("[batch_volume_trace_overlay] error: use only one of --pipette-vlines or --vline-frames")
        return 1
    vline_frames: list[float] | None
    if args.pipette_vlines:
        vline_frames = [575.0, 875.0, 1439.0]
    elif args.vline_frames is not None:
        vline_frames = list(args.vline_frames)
    else:
        vline_frames = None

    plot_ns = argparse.Namespace(
        time_interval_min=args.time_interval_min,
        volume_ylim=tuple(args.volume_ylim),
        mean_ri_ylim=tuple(args.mean_ri_ylim),
        mass_ylim=tuple(args.mass_ylim),
        preset=args.preset,
        vline_frames=vline_frames,
    )

    fig = make_volume_trace_overlay(included, plot_ns)
    if not args.no_save:
        params = {
            "crop_sub": args.crop_sub,
            "pos_filter": args.pos,
            "n_series": len(included),
            "series": [x[0] for x in included],
            "skipped": skipped,
            "mass_ylim": list(plot_ns.mass_ylim),
            "vline_frames": vline_frames,
            "time_interval_min": args.time_interval_min,
        }
        if args.from_npz is not None:
            params["source_npz"] = str(args.from_npz.expanduser().resolve())
        else:
            params["ph_root"] = str(args.ph_root.expanduser().resolve())
        save_figure(
            fig,
            params=params,
            data=series_to_npz_dict(included, time_interval_min=args.time_interval_min),
            description=(
                "Pos-crossing overlay: rod volume, mean RI, total mass for channels with inference_out"
            ),
        )
    plt.close(fig)
    print("[batch_volume_trace_overlay] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
