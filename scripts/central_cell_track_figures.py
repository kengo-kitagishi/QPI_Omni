from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Rectangle
from skimage import measure
from skimage.transform import rotate

from figure_logger import save_figure
from ri_calibration import (
    load_calibration,
    n_medium_at_frame,
    parse_media_schedule,
)

ALL_FIGURES = [
    "overlay_strip",
    "contour_montage",
    "shape_trace",
    "volume_trace",
    "ghost_contour",
    "intensity_kymograph",
    "mask_kymograph",
    "qc_overview",
]

DIRECT_RUN_CONFIG = {
    "indir": None,
    "pos_root": None,
    "channel": None,
    "outdir": None,
    "figure": ALL_FIGURES,
    "formats": ["png", "pdf", "svg"],
    "min_area": 20,
    "exclude_border": True,
    "crop_margin": 12,
    "align_major_axis": True,
    "flip_horizontal": False,
    "panel_count": 6,
    "pixel_size_um": 0.348,
    "time_interval_min": None,
    "scalebar_um": 2.0,
    "wavelength_nm": 658.0,
    "n_medium": 1.333,
    "alpha_ri": 0.00018,
    "ri_calibration": None,
    "calibration_id": None,
    "media_schedule": None,
    "n_milliq": None,
    "preset": "manuscript",
    "no_save": False,
    "attach_source_tifs": False,
    "write_track_tifs": False,
    "kymograph_vmin": -0.5,
    "kymograph_vmax": 2.0,
    "volume_ylim": [0.0, 400.0],
    "mean_ri_ylim": [1.33, 1.42],
    "mass_ylim": [0.0, 500.0],
    "media_switch_frames": [575.0, 875.0, 1439.0],
    "kymograph_sample_hours": 48.0,
    "kymograph_window_frames": 10,
}

PRESET_STYLE = {
    "manuscript": {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
    },
    "presentation": {
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "lines.linewidth": 2.0,
    },
    "qc": {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
    },
}


@dataclass
class FramePair:
    frame_index: int
    frame_name: str
    raw_path: Path | None
    mask_path: Path | None


def natural_key(text: str) -> list[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def strip_mask_suffix(path: Path) -> str:
    if path.stem.endswith("_masks"):
        return path.stem[: -len("_masks")]
    return path.stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select the center-nearest cell mask in each frame from a "
            "run_omnipose_chm_batch.py channel directory and generate figures."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--indir",
        type=Path,
        help="Channel directory such as Pos3/output_phase/channels/crop_sub_rawraw/ch00",
    )
    group.add_argument(
        "--pos-root",
        type=Path,
        help="Position root directory such as .../Pos3. Use with --channel.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        help="Channel number for --pos-root mode, e.g. 0 for ch00.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Local directory for CSV/JSON/TIF outputs. Default: results/central_cell_track/<timestamp>_<pos>_<ch>.",
    )
    parser.add_argument(
        "--figure",
        nargs="+",
        choices=ALL_FIGURES,
        default=ALL_FIGURES,
        help="Figures to generate.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        help="Formats passed to figure_logger.save_figure().",
    )
    parser.add_argument("--min-area", type=int, default=20, help="Minimum mask area in pixels.")
    parser.set_defaults(exclude_border=True)
    border_group = parser.add_mutually_exclusive_group()
    border_group.add_argument(
        "--exclude-border",
        dest="exclude_border",
        action="store_true",
        help="Exclude masks touching the image border. Default: on.",
    )
    border_group.add_argument(
        "--allow-border",
        dest="exclude_border",
        action="store_false",
        help="Allow border-touching masks.",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=12,
        help="Extra pixels around the selected mask when building fixed-size crops.",
    )
    parser.set_defaults(align_major_axis=True)
    align_group = parser.add_mutually_exclusive_group()
    align_group.add_argument(
        "--align-major-axis",
        dest="align_major_axis",
        action="store_true",
        help="Rotate crops so the major axis is horizontal. Default: on.",
    )
    align_group.add_argument(
        "--no-align-major-axis",
        dest="align_major_axis",
        action="store_false",
        help="Disable rotation alignment.",
    )
    parser.add_argument(
        "--flip-horizontal",
        action="store_true",
        help="Flip aligned crops horizontally after major-axis alignment.",
    )
    parser.add_argument(
        "--panel-count",
        type=int,
        default=6,
        help="Number of sampled frames to show in montage/strip figures.",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=0.348,
        help="Pixel size in micrometers for scale bars and metric conversion. Default: 0.348.",
    )
    parser.add_argument(
        "--time-interval-min",
        type=float,
        default=None,
        help="Time interval per frame in minutes.",
    )
    parser.add_argument(
        "--scalebar-um",
        type=float,
        default=2.0,
        help="Scale bar length in micrometers.",
    )
    parser.add_argument(
        "--wavelength-nm",
        type=float,
        default=663.0,
        help="Illumination wavelength in nanometers for mean-RI estimation.",
    )
    parser.add_argument(
        "--n-medium",
        type=float,
        default=1.333,
        help="Medium refractive index for mean-RI estimation.",
    )
    parser.add_argument(
        "--alpha-ri",
        type=float,
        default=0.00018,
        help="Specific refractive increment used for concentration and mass estimation.",
    )
    parser.add_argument(
        "--ri-calibration",
        type=Path,
        default=None,
        help=(
            "Path to ri_calibration_results.json (append-history schema). "
            "When set, mean-RI is computed with frame-dependent n_medium "
            "looked up from --media-schedule, and protein density uses MilliQ as baseline."
        ),
    )
    parser.add_argument(
        "--calibration-id",
        type=str,
        default=None,
        help="Specific calibration entry id; default uses the JSON's `active`.",
    )
    parser.add_argument(
        "--media-schedule",
        type=str,
        default=None,
        help=(
            "Frame->medium step schedule, e.g. '0:wo_2,575:wo_0,1439:wo_2'. "
            "Required when --ri-calibration is set."
        ),
    )
    parser.add_argument(
        "--n-milliq",
        type=float,
        default=None,
        help=(
            "MilliQ refractive index used as baseline for protein density. "
            "Defaults to the calibration's reference.n_miliq when --ri-calibration is set."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=["manuscript", "presentation", "qc"],
        default="manuscript",
        help="Figure style preset.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Build figures but skip figure_logger saves.",
    )
    parser.add_argument(
        "--attach-source-tifs",
        action="store_true",
        help="Copy source raw/mask TIFFs into figure_logger inbox. Default: off.",
    )
    parser.add_argument(
        "--write-track-tifs",
        action="store_true",
        help="Write track_masks.tif and aligned crop TIFF stacks into the local output directory. Default: off.",
    )
    parser.add_argument(
        "--kymograph-vmin",
        type=float,
        default=-0.5,
        help="Lower intensity limit for rectangular kymograph display.",
    )
    parser.add_argument(
        "--kymograph-vmax",
        type=float,
        default=2.0,
        help="Upper intensity limit for rectangular kymograph display.",
    )
    parser.add_argument(
        "--volume-ylim",
        nargs=2,
        type=float,
        default=[0.0, 400.0],
        metavar=("YMIN", "YMAX"),
        help="Y-axis limits for volume trace. Default: 0 400.",
    )
    parser.add_argument(
        "--mean-ri-ylim",
        nargs=2,
        type=float,
        default=[1.34, 1.37],
        metavar=("YMIN", "YMAX"),
        help="Y-axis limits for mean-RI trace. Default: 1.34 1.37.",
    )
    parser.add_argument(
        "--mass-ylim",
        nargs=2,
        type=float,
        default=[0.0, 500000.0],
        metavar=("YMIN", "YMAX"),
        help="Y-axis limits for total-mass trace in pg. Default: 0 500000.",
    )
    parser.add_argument(
        "--media-switch-frames",
        nargs="+",
        type=float,
        default=[575.0, 875.0, 1439.0],
        help="Vertical dotted guide lines for media switches, specified in frame index units.",
    )
    parser.add_argument(
        "--kymograph-sample-hours",
        type=float,
        default=48.0,
        help="Sampling interval in hours for sparse kymograph output. Default: 48.",
    )
    parser.add_argument(
        "--kymograph-window-frames",
        type=int,
        default=10,
        help="Number of consecutive frames for the short-window kymograph. Default: 10.",
    )
    return parser


def build_args_from_direct_run_config() -> argparse.Namespace:
    cfg = dict(DIRECT_RUN_CONFIG)
    indir = cfg.get("indir")
    pos_root = cfg.get("pos_root")
    channel = cfg.get("channel")

    if indir is None and pos_root is None:
        raise SystemExit(
            "No CLI arguments were provided. Set DIRECT_RUN_CONFIG['indir'] or DIRECT_RUN_CONFIG['pos_root'] before running directly."
        )
    if indir is None and channel is None:
        raise SystemExit(
            "DIRECT_RUN_CONFIG['channel'] is required when using DIRECT_RUN_CONFIG['pos_root']."
        )

    path_keys = {"indir", "pos_root", "outdir"}
    for key in path_keys:
        if cfg.get(key) is not None:
            cfg[key] = Path(cfg[key])

    if cfg["figure"] is None:
        cfg["figure"] = list(ALL_FIGURES)
    else:
        cfg["figure"] = list(cfg["figure"])

    if cfg["formats"] is None:
        cfg["formats"] = ["png", "pdf", "svg"]
    else:
        cfg["formats"] = list(cfg["formats"])

    return argparse.Namespace(**cfg)


def parse_runtime_args() -> argparse.Namespace:
    if len(sys.argv) > 1:
        return build_parser().parse_args()
    return build_args_from_direct_run_config()


def resolve_channel_dir(args: argparse.Namespace) -> Path:
    if args.indir is not None:
        return args.indir.expanduser().resolve()
    if args.channel is None:
        raise SystemExit("--channel is required with --pos-root")
    return (
        args.pos_root.expanduser().resolve()
        / "output_phase"
        / "channels"
        / "crop_sub_rawraw"
        / f"ch{args.channel:02d}"
    )


def detect_pos_name(channel_dir: Path) -> str:
    for part in channel_dir.parts[::-1]:
        if part.startswith("Pos"):
            return part
    return "unknown_pos"


def default_outdir(channel_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pos_name = detect_pos_name(channel_dir)
    ch_name = channel_dir.name
    return Path("results") / "central_cell_track" / f"{stamp}_{pos_name}_{ch_name}"


def is_binary_mask(mask: np.ndarray) -> bool:
    if mask.dtype == bool:
        return True
    values = np.unique(mask)
    return values.size <= 2 and set(values.tolist()).issubset({0, 1})


def load_label_image(mask_path: Path) -> np.ndarray:
    arr = tifffile.imread(mask_path)
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask image, got shape={arr.shape} for {mask_path}")
    if is_binary_mask(arr):
        arr = measure.label(arr > 0, connectivity=1)
    return arr.astype(np.int32, copy=False)


def load_raw_image(raw_path: Path) -> np.ndarray:
    arr = tifffile.imread(raw_path)
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D raw image, got shape={arr.shape} for {raw_path}")
    return arr


def build_frame_pairs(channel_dir: Path) -> list[FramePair]:
    if not channel_dir.is_dir():
        raise FileNotFoundError(f"Channel directory not found: {channel_dir}")

    inference_dir = channel_dir / "inference_out"
    raw_paths = [
        p
        for p in sorted(channel_dir.glob("*.tif"), key=lambda x: natural_key(x.name))
        if not p.stem.endswith("_masks") and not p.stem.endswith("_binary")
    ]
    mask_paths = sorted(inference_dir.glob("*_masks.tif"), key=lambda x: natural_key(x.name))

    raw_map = {p.stem: p for p in raw_paths}
    mask_map = {strip_mask_suffix(p): p for p in mask_paths}
    frame_names = sorted(set(raw_map) | set(mask_map), key=natural_key)

    pairs: list[FramePair] = []
    for idx, name in enumerate(frame_names):
        pairs.append(
            FramePair(
                frame_index=idx,
                frame_name=name,
                raw_path=raw_map.get(name),
                mask_path=mask_map.get(name),
            )
        )
    return pairs


def compute_alignment_rotation_deg(mask: np.ndarray) -> float:
    labeled = measure.label(mask > 0, connectivity=1)
    props = measure.regionprops(labeled)
    if not props:
        return 0.0
    prop = max(props, key=lambda p: p.area)
    orientation_deg = float(np.degrees(prop.orientation))
    rotate_deg = 90.0 - orientation_deg
    while rotate_deg > 90.0:
        rotate_deg -= 180.0
    while rotate_deg <= -90.0:
        rotate_deg += 180.0
    return rotate_deg


def extract_centered_crop(
    image: np.ndarray,
    center_y: float,
    center_x: float,
    size: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    cy = int(round(center_y))
    cx = int(round(center_x))
    half = size // 2
    y0 = cy - half
    x0 = cx - half
    y1 = y0 + size
    x1 = x0 + size

    out = np.full((size, size), fill_value, dtype=image.dtype)
    src_y0 = max(y0, 0)
    src_x0 = max(x0, 0)
    src_y1 = min(y1, image.shape[0])
    src_x1 = min(x1, image.shape[1])

    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def rotate_crop_pair(
    raw_crop: np.ndarray | None,
    mask_crop: np.ndarray,
    align_major_axis: bool,
    flip_horizontal: bool,
) -> tuple[np.ndarray | None, np.ndarray]:
    if not align_major_axis:
        out_raw = raw_crop
        out_mask = mask_crop
        if flip_horizontal:
            if out_raw is not None:
                out_raw = np.fliplr(out_raw)
            out_mask = np.fliplr(out_mask)
        return out_raw, out_mask
    angle_deg = compute_alignment_rotation_deg(mask_crop > 0)
    if abs(angle_deg) < 1e-6:
        out_raw = raw_crop
        out_mask = mask_crop
        if flip_horizontal:
            if out_raw is not None:
                out_raw = np.fliplr(out_raw)
            out_mask = np.fliplr(out_mask)
        return out_raw, out_mask

    out_raw = None
    if raw_crop is not None:
        out_raw = rotate(
            raw_crop,
            angle_deg,
            resize=False,
            order=1,
            preserve_range=True,
            mode="constant",
            cval=float(np.median(raw_crop)),
        ).astype(np.float32)
    out_mask = rotate(
        mask_crop.astype(np.float32),
        angle_deg,
        resize=False,
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    )
    out_mask = (out_mask > 0.5).astype(np.uint8)
    if flip_horizontal:
        if out_raw is not None:
            out_raw = np.fliplr(out_raw)
        out_mask = np.fliplr(out_mask)
    return out_raw, out_mask


def candidate_rows_for_frame(
    mask_label: np.ndarray,
    raw_image: np.ndarray | None,
    min_area: int,
) -> pd.DataFrame:
    h, w = mask_label.shape
    props = measure.regionprops(mask_label, intensity_image=raw_image)
    rows: list[dict[str, object]] = []

    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        touches_border = minr <= 0 or minc <= 0 or maxr >= h or maxc >= w
        distance_to_center = float(
            math.hypot(prop.centroid[0] - (h / 2.0), prop.centroid[1] - (w / 2.0))
        )
        row = {
            "label": int(prop.label),
            "area_px": int(prop.area),
            "centroid_y": float(prop.centroid[0]),
            "centroid_x": float(prop.centroid[1]),
            "bbox_ymin": int(minr),
            "bbox_xmin": int(minc),
            "bbox_ymax": int(maxr),
            "bbox_xmax": int(maxc),
            "major_axis_px": float(getattr(prop, "major_axis_length", np.nan)),
            "minor_axis_px": float(getattr(prop, "minor_axis_length", np.nan)),
            "orientation_rad": float(getattr(prop, "orientation", np.nan)),
            "eccentricity": float(getattr(prop, "eccentricity", np.nan)),
            "solidity": float(getattr(prop, "solidity", np.nan)),
            "touches_border": bool(touches_border),
            "distance_to_center_px": distance_to_center,
            "passes_min_area": bool(prop.area >= min_area),
        }
        if raw_image is not None:
            intens = raw_image[mask_label == prop.label]
            if intens.size:
                row["mean_intensity"] = float(np.mean(intens))
                row["median_intensity"] = float(np.median(intens))
                row["integrated_intensity"] = float(np.sum(intens))
            else:
                row["mean_intensity"] = np.nan
                row["median_intensity"] = np.nan
                row["integrated_intensity"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def calc_rod_volume_um3(major_px: float, minor_px: float, pixel_size_um: float) -> float:
    length_um = float(major_px) * pixel_size_um
    width_um = float(minor_px) * pixel_size_um
    r_um = width_um / 2.0
    h_um = length_um - 2.0 * r_um
    if h_um < 0:
        return float((4.0 / 3.0) * np.pi * (r_um ** 3))
    return float((4.0 / 3.0) * np.pi * (r_um ** 3) + np.pi * (r_um ** 2) * h_um)


def calc_optical_metrics(
    total_phase: float,
    volume_um3: float,
    pixel_size_um: float,
    wavelength_nm: float,
    n_medium: float,
    alpha_ri: float,
    n_protein_basis: float | None = None,
) -> tuple[float, float, float]:
    """mean_RI uses n_medium; protein concentration / mass uses n_protein_basis
    (defaults to n_medium when not given, preserving legacy behavior)."""
    if not np.isfinite(total_phase) or not np.isfinite(volume_um3) or volume_um3 <= 0 or pixel_size_um <= 0:
        return np.nan, np.nan, np.nan

    wavelength_um = wavelength_nm * 1e-3
    pixel_area_um2 = pixel_size_um ** 2
    mean_ri = n_medium + (float(total_phase) * wavelength_um * pixel_area_um2) / (2.0 * np.pi * float(volume_um3))

    if not np.isfinite(mean_ri):
        return np.nan, np.nan, np.nan
    basis = float(n_protein_basis) if n_protein_basis is not None else float(n_medium)
    mean_conc = (mean_ri - basis) / alpha_ri if alpha_ri > 0 else np.nan
    mass_pg = mean_conc * volume_um3 * 1e-3 if np.isfinite(mean_conc) else np.nan  # [mg/mL] × [µm³] × 1e-3 → [pg]
    return float(mean_ri), float(mean_conc), float(mass_pg)


def select_center_cell_for_frame(
    pair: FramePair,
    min_area: int,
    exclude_border: bool,
) -> dict[str, object]:
    base_row: dict[str, object] = {
        "frame_index": pair.frame_index,
        "frame_name": pair.frame_name,
        "raw_path": str(pair.raw_path.resolve()) if pair.raw_path else "",
        "mask_path": str(pair.mask_path.resolve()) if pair.mask_path else "",
        "label": 0,
        "tracked": False,
        "selection_flag": "missing_mask" if pair.mask_path is None else "missing_cell",
        "centroid_y": np.nan,
        "centroid_x": np.nan,
        "area_px": np.nan,
        "major_axis_px": np.nan,
        "minor_axis_px": np.nan,
        "orientation_rad": np.nan,
        "eccentricity": np.nan,
        "solidity": np.nan,
        "bbox_ymin": np.nan,
        "bbox_xmin": np.nan,
        "bbox_ymax": np.nan,
        "bbox_xmax": np.nan,
        "distance_to_center_px": np.nan,
        "touches_border": False,
        "mean_intensity": np.nan,
        "median_intensity": np.nan,
        "integrated_intensity": np.nan,
        "image_height_px": np.nan,
        "image_width_px": np.nan,
    }
    if pair.mask_path is None:
        return base_row

    raw_image = None
    if pair.raw_path is not None and pair.raw_path.exists():
        raw_image = load_raw_image(pair.raw_path)
    mask_label = load_label_image(pair.mask_path)
    base_row["image_height_px"] = int(mask_label.shape[0])
    base_row["image_width_px"] = int(mask_label.shape[1])

    df = candidate_rows_for_frame(mask_label, raw_image, min_area)
    if df.empty:
        return base_row

    candidates = df[df["passes_min_area"]].copy()
    if candidates.empty:
        all_sorted = df.sort_values("distance_to_center_px")
        row = dict(base_row)
        for key, value in all_sorted.iloc[0].to_dict().items():
            row[key] = value
        row["label"] = int(row["label"])
        row["tracked"] = True
        row["selection_flag"] = "low_confidence"
        row["raw_path"] = base_row["raw_path"]
        row["mask_path"] = base_row["mask_path"]
        row["frame_index"] = base_row["frame_index"]
        row["frame_name"] = base_row["frame_name"]
        return row

    if exclude_border:
        non_border = candidates[~candidates["touches_border"]].copy()
        if not non_border.empty:
            candidates = non_border
            selection_flag = "ok"
        else:
            selection_flag = "border_fallback"
    else:
        selection_flag = "ok"

    chosen = dict(base_row)
    for key, value in candidates.sort_values("distance_to_center_px").iloc[0].to_dict().items():
        chosen[key] = value
    chosen["label"] = int(chosen["label"])
    chosen["tracked"] = True
    chosen["selection_flag"] = selection_flag
    chosen["raw_path"] = base_row["raw_path"]
    chosen["mask_path"] = base_row["mask_path"]
    chosen["frame_index"] = base_row["frame_index"]
    chosen["frame_name"] = base_row["frame_name"]
    return chosen


def build_summary_table(
    frame_pairs: list[FramePair],
    min_area: int,
    exclude_border: bool,
    pixel_size_um: float | None,
    wavelength_nm: float,
    n_medium: float,
    alpha_ri: float,
    media_schedule: list[tuple[int, str]] | None = None,
    media_ri: dict[str, float] | None = None,
    n_milliq: float | None = None,
) -> pd.DataFrame:
    """Build per-frame summary table.

    Frame-dependent medium RI:
      If `media_schedule` and `media_ri` are given, n_medium for each frame is
      looked up from the schedule. Otherwise the constant `n_medium` is used.

    Protein basis:
      If `n_milliq` is given, `mean_concentration` and `mass_pg` use MilliQ as
      the baseline (independent of which medium the cell sits in). Otherwise
      the per-frame n_medium is used (legacy behavior).
    """
    rows = [select_center_cell_for_frame(pair, min_area=min_area, exclude_border=exclude_border) for pair in frame_pairs]
    df = pd.DataFrame(rows)
    df["frame_label"] = df["frame_name"]
    df["total_phase"] = np.where(
        df["tracked"].to_numpy(dtype=bool),
        df["integrated_intensity"].to_numpy(dtype=float),
        np.nan,
    )
    if pixel_size_um is not None and pixel_size_um > 0:
        df["volume_um3_rod"] = np.where(
            df["tracked"].to_numpy(dtype=bool),
            [
                calc_rod_volume_um3(major, minor, pixel_size_um)
                if np.isfinite(major) and np.isfinite(minor) and minor > 0
                else np.nan
                for major, minor in zip(
                    df["major_axis_px"].to_numpy(dtype=float),
                    df["minor_axis_px"].to_numpy(dtype=float),
                )
            ],
            np.nan,
        )
    else:
        df["volume_um3_rod"] = np.nan

    use_schedule = bool(media_schedule) and bool(media_ri)
    if use_schedule:
        n_medium_per_row = [
            n_medium_at_frame(int(f), media_schedule, media_ri)
            for f in df["frame_index"].to_numpy()
        ]
    else:
        n_medium_per_row = [float(n_medium)] * len(df)
    df["n_medium_used"] = n_medium_per_row
    df["n_milliq_used"] = float(n_milliq) if n_milliq is not None else np.nan

    if pixel_size_um is not None and pixel_size_um > 0:
        optical_metrics = [
            calc_optical_metrics(
                total_phase, volume_um3, pixel_size_um, wavelength_nm,
                nm_row, alpha_ri, n_protein_basis=n_milliq,
            )
            for total_phase, volume_um3, nm_row in zip(
                df["total_phase"].to_numpy(dtype=float),
                df["volume_um3_rod"].to_numpy(dtype=float),
                n_medium_per_row,
            )
        ]
        df["mean_ri"] = [vals[0] for vals in optical_metrics]
        df["mean_concentration"] = [vals[1] for vals in optical_metrics]
        df["mass_pg"] = [vals[2] for vals in optical_metrics]
    else:
        df["mean_ri"] = np.nan
        df["mean_concentration"] = np.nan
        df["mass_pg"] = np.nan
    return df


def determine_crop_size(summary_df: pd.DataFrame, margin: int) -> int:
    selected = summary_df[summary_df["tracked"]].copy()
    if selected.empty:
        return 64
    heights = selected["bbox_ymax"] - selected["bbox_ymin"]
    widths = selected["bbox_xmax"] - selected["bbox_xmin"]
    max_extent = int(np.nanmax(np.maximum(heights.to_numpy(dtype=float), widths.to_numpy(dtype=float))))
    side = max_extent + 2 * int(margin)
    side = max(side, 32)
    if side % 2 == 1:
        side += 1
    return side


def build_selected_mask(label_image: np.ndarray, label_value: int) -> np.ndarray:
    if label_value <= 0:
        return np.zeros_like(label_image, dtype=np.uint8)
    return (label_image == label_value).astype(np.uint8)


def build_crops_for_row(
    row: pd.Series,
    crop_size: int,
    align_major_axis: bool,
    flip_horizontal: bool,
) -> tuple[np.ndarray | None, np.ndarray]:
    mask_path = Path(row["mask_path"]) if row["mask_path"] else None
    if not row["tracked"] or mask_path is None or not mask_path.exists():
        empty_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        return None, empty_mask

    label_img = load_label_image(mask_path)
    selected_mask = build_selected_mask(label_img, int(row["label"]))
    center_y = float(row["centroid_y"])
    center_x = float(row["centroid_x"])
    mask_crop = extract_centered_crop(selected_mask, center_y, center_x, crop_size, fill_value=0)

    raw_crop = None
    if row["raw_path"]:
        raw_path = Path(row["raw_path"])
        if raw_path.exists():
            raw_img = load_raw_image(raw_path)
            raw_crop = extract_centered_crop(
                raw_img,
                center_y,
                center_x,
                crop_size,
                fill_value=float(np.median(raw_img)),
            ).astype(np.float32)

    return rotate_crop_pair(
        raw_crop,
        mask_crop,
        align_major_axis=align_major_axis,
        flip_horizontal=flip_horizontal,
    )


def select_panel_rows(summary_df: pd.DataFrame, panel_count: int) -> pd.DataFrame:
    valid = summary_df[summary_df["tracked"]].copy()
    if valid.empty:
        return valid
    n = len(valid)
    if n <= panel_count:
        return valid
    sample_idx = np.unique(np.linspace(0, n - 1, panel_count).round().astype(int))
    return valid.iloc[sample_idx].copy()


def frame_time_label(row: pd.Series, time_interval_min: float | None) -> str:
    if time_interval_min is None:
        return f"f{int(row['frame_index']):03d}"
    t_min = float(row["frame_index"]) * time_interval_min
    if abs(t_min - round(t_min)) < 1e-6:
        return f"{int(round(t_min))} min"
    return f"{t_min:.1f} min"


def add_scale_bar(
    ax: plt.Axes,
    image_shape: tuple[int, int],
    pixel_size_um: float | None,
    scalebar_um: float,
    color: str,
    linewidth: float,
    outline_color: str | None = None,
) -> None:
    if pixel_size_um is None or pixel_size_um <= 0:
        return
    bar_px = scalebar_um / pixel_size_um
    h, w = image_shape
    x1 = w * 0.88
    x0 = x1 - bar_px
    y = h * 0.90
    if outline_color:
        ax.plot([x0, x1], [y, y], color=outline_color, lw=linewidth + 1.6, solid_capstyle="butt")
    ax.plot([x0, x1], [y, y], color=color, lw=linewidth, solid_capstyle="butt")
    ax.text(
        (x0 + x1) / 2.0,
        y - max(3, h * 0.04),
        f"{scalebar_um:g} um",
        ha="center",
        va="bottom",
        color=color,
        bbox=None if outline_color is None else dict(boxstyle="round,pad=0.15", fc=(0, 0, 0, 0.35), ec="none"),
    )


def contours_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    return measure.find_contours(mask.astype(float), level=0.5)


def rotate_clockwise_90(image: np.ndarray) -> np.ndarray:
    return np.rot90(image, k=-1)


def tight_crop_single(mask: np.ndarray, margin: int) -> tuple[slice, slice]:
    if not np.any(mask):
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    rows = np.flatnonzero(np.any(mask > 0, axis=1))
    cols = np.flatnonzero(np.any(mask > 0, axis=0))
    r0 = max(0, int(rows[0]) - margin)
    r1 = min(mask.shape[0], int(rows[-1]) + margin + 1)
    c0 = max(0, int(cols[0]) - margin)
    c1 = min(mask.shape[1], int(cols[-1]) + margin + 1)
    return slice(r0, r1), slice(c0, c1)


def centered_rectangle_slices(
    image_shape: tuple[int, int],
    height: int,
    width: int,
) -> tuple[slice, slice]:
    img_h, img_w = image_shape
    height = max(1, min(int(height), img_h))
    width = max(1, min(int(width), img_w))
    center_y = img_h // 2
    center_x = img_w // 2
    y0 = max(0, center_y - (height // 2))
    x0 = max(0, center_x - (width // 2))
    y1 = min(img_h, y0 + height)
    x1 = min(img_w, x0 + width)
    y0 = max(0, y1 - height)
    x0 = max(0, x1 - width)
    return slice(y0, y1), slice(x0, x1)


def determine_kymograph_box(summary_df: pd.DataFrame, crop_size: int, margin: int) -> tuple[int, int]:
    valid = summary_df[summary_df["tracked"]].copy()
    if valid.empty:
        side = max(8, crop_size // 2)
        return side, side

    major = valid["major_axis_px"].to_numpy(dtype=float)
    minor = valid["minor_axis_px"].to_numpy(dtype=float)
    major = major[np.isfinite(major)]
    minor = minor[np.isfinite(minor)]
    rect_width = int(np.ceil(np.max(major))) if major.size else crop_size
    rect_height = int(np.ceil(np.max(minor))) if minor.size else crop_size
    rect_width = min(crop_size, max(8, rect_width + 2 * margin))
    rect_height = min(crop_size, max(8, rect_height + 2 * margin))
    return rect_height, rect_width


def extract_centered_rectangle(image: np.ndarray, rect_height: int, rect_width: int) -> np.ndarray:
    row_slice, col_slice = centered_rectangle_slices(image.shape, rect_height, rect_width)
    return image[row_slice, col_slice]


def panel_letters(n: int) -> list[str]:
    letters = []
    for idx in range(n):
        letters.append(chr(ord("A") + idx))
    return letters


def apply_style(preset: str) -> dict[str, object]:
    style = PRESET_STYLE[preset]
    plt.rcParams.update(style)
    return style


def make_overlay_strip(
    summary_df: pd.DataFrame,
    crop_size: int,
    args: argparse.Namespace,
) -> plt.Figure | None:
    if summary_df.empty or summary_df["raw_path"].eq("").all():
        return None

    raw_crops_rot: list[np.ndarray] = []
    mask_crops_rot: list[np.ndarray] = []
    rows_used: list[pd.Series] = []
    valid_raw_pixels: list[np.ndarray] = []

    for _, row in summary_df.iterrows():
        raw_crop, mask_crop = build_crops_for_row(
            row,
            crop_size,
            args.align_major_axis,
            args.flip_horizontal,
        )
        if raw_crop is not None:
            raw_rot = rotate_clockwise_90(raw_crop)
            if np.any(mask_crop):
                valid_raw_pixels.append(raw_rot[np.isfinite(raw_rot)])
        else:
            raw_rot = np.full((crop_size, crop_size), np.nan, dtype=np.float32)
        mask_rot = rotate_clockwise_90(mask_crop)
        raw_crops_rot.append(raw_rot)
        mask_crops_rot.append(mask_rot)
        rows_used.append(row)

    if not rows_used or not valid_raw_pixels:
        return None

    fill_value = float(np.nanmedian(np.concatenate(valid_raw_pixels)))
    raw_crops_rot = [
        np.where(np.isfinite(raw_crop), raw_crop, fill_value).astype(np.float32)
        for raw_crop in raw_crops_rot
    ]

    tight_margin = max(1, min(3, args.crop_margin // 4))
    raw_crops_tight: list[np.ndarray] = []
    mask_crops_tight: list[np.ndarray] = []
    heights: list[int] = []
    widths: list[int] = []
    for raw_crop, mask_crop in zip(raw_crops_rot, mask_crops_rot):
        row_slice, col_slice = tight_crop_single(mask_crop, margin=tight_margin)
        raw_tight = raw_crop[row_slice, col_slice]
        mask_tight = mask_crop[row_slice, col_slice]
        raw_crops_tight.append(raw_tight)
        mask_crops_tight.append(mask_tight)
        heights.append(raw_tight.shape[0])
        widths.append(raw_tight.shape[1])

    target_height = max(heights)
    raw_crops_rot = []
    mask_crops_rot = []
    for raw_crop, mask_crop in zip(raw_crops_tight, mask_crops_tight):
        pad_total = target_height - raw_crop.shape[0]
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        raw_crops_rot.append(
            np.pad(raw_crop, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=fill_value)
        )
        mask_crops_rot.append(
            np.pad(mask_crop, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=0)
        )

    stack = np.concatenate(raw_crops_rot, axis=1)
    vmin = float(np.nanpercentile(stack, 2))
    vmax = float(np.nanpercentile(stack, 98))
    strip = stack
    strip_height = raw_crops_rot[0].shape[0]
    fig_width = min(14.0, max(6.0, strip.shape[1] / 260.0))
    fig_height = min(4.6, max(2.6, strip_height / 26.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    contour_color = "#ffcc33" if args.preset == "presentation" else "#d55e00"

    ax.imshow(strip, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    x_offset = 0
    for mask_crop in mask_crops_rot:
        for contour in contours_from_mask(mask_crop):
            ax.plot(contour[:, 1] + x_offset, contour[:, 0], color=contour_color, lw=0.7)
        x_offset += mask_crop.shape[1]

    x_edges = np.cumsum([0, *[raw_crop.shape[1] for raw_crop in raw_crops_rot]])
    xtick_idx = np.linspace(0, len(rows_used) - 1, min(6, len(rows_used))).round().astype(int)
    xtick_pos = [(x_edges[i] + x_edges[i + 1]) / 2.0 for i in xtick_idx]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([frame_time_label(rows_used[i], args.time_interval_min) for i in xtick_idx])
    ax.set_xlabel("Time")

    yticks = np.linspace(0, strip_height - 1, 5)
    ax.set_yticks(yticks)
    if args.pixel_size_um is not None and args.pixel_size_um > 0:
        ax.set_yticklabels([f"{(y - (strip_height / 2.0)) * args.pixel_size_um:.1f}" for y in yticks])
        ax.set_ylabel("Position along major axis [um]")
    else:
        ax.set_ylabel("Position along major axis [px]")

    ax.text(
        0.01,
        0.98,
        frame_time_label(rows_used[0], args.time_interval_min),
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.18", fc=(1, 1, 1, 0.78), ec="none"),
    )
    ax.text(
        0.99,
        0.98,
        frame_time_label(rows_used[-1], args.time_interval_min),
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.18", fc=(1, 1, 1, 0.78), ec="none"),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if args.preset != "manuscript":
        fig.suptitle("Center-nearest representative cell: overlay kymograph", y=1.02)
    return fig


def make_contour_montage(
    panel_rows: pd.DataFrame,
    crop_size: int,
    args: argparse.Namespace,
) -> plt.Figure | None:
    if panel_rows.empty:
        return None
    n = len(panel_rows)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.25), constrained_layout=True)
    if n == 1:
        axes = [axes]
    contour_color = "#111111"
    for ax, (_, row) in zip(axes, panel_rows.iterrows()):
        _, mask_crop = build_crops_for_row(
            row,
            crop_size,
            args.align_major_axis,
            args.flip_horizontal,
        )
        if np.any(mask_crop):
            ax.imshow(mask_crop, cmap="Greys", vmin=0, vmax=1, alpha=0.18)
            for contour in contours_from_mask(mask_crop):
                ax.plot(contour[:, 1], contour[:, 0], color=contour_color, lw=1.0)
        else:
            ax.add_patch(Rectangle((0, 0), crop_size, crop_size, facecolor="white", edgecolor="none"))
            ax.text(
                0.5,
                0.5,
                "missing",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#888888",
            )
        ax.text(
            0.05,
            0.96,
            frame_time_label(row, args.time_interval_min),
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=contour_color,
        )
        ax.set_facecolor("white")
        ax.set_axis_off()

    add_scale_bar(
        axes[-1],
        (crop_size, crop_size),
        args.pixel_size_um,
        args.scalebar_um,
        color=contour_color,
        linewidth=1.2,
    )
    if args.preset != "manuscript":
        fig.suptitle("Center-nearest representative cell: contour montage", y=1.02)
    return fig


def make_shape_trace(summary_df: pd.DataFrame, args: argparse.Namespace) -> plt.Figure:
    df = summary_df.copy()
    x = df["frame_index"].to_numpy(dtype=float)
    if args.time_interval_min is not None:
        x = x * args.time_interval_min
        x_label = "Time [min]"
    else:
        x_label = "Frame"

    area = df["area_px"].to_numpy(dtype=float)
    major = df["major_axis_px"].to_numpy(dtype=float)
    minor = df["minor_axis_px"].to_numpy(dtype=float)
    aspect = major / np.where(minor > 0, minor, np.nan)

    if args.pixel_size_um is not None and args.pixel_size_um > 0:
        px = args.pixel_size_um
        area = area * (px ** 2)
        major = major * px
        minor = minor * px
        area_label = "Area [um^2]"
        axis_label = "Axis length [um]"
    else:
        area_label = "Area [px]"
        axis_label = "Axis length [px]"

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), constrained_layout=True)
    axes = axes.ravel()
    letters = panel_letters(4)
    series = [
        ("Area", area, area_label),
        ("Major axis", major, axis_label),
        ("Minor axis", minor, axis_label),
        ("Aspect ratio", aspect, "Major / minor"),
    ]
    color = "#2a6f97"
    missing_mask = ~df["tracked"].to_numpy(dtype=bool)
    missing_half_width = (args.time_interval_min / 2.0) if args.time_interval_min else 0.45

    for ax, letter, (title, y, ylabel) in zip(axes, letters, series):
        ax.plot(x, y, color=color, lw=1.2)
        if missing_mask.any():
            for xv, is_missing in zip(x, missing_mask):
                if is_missing:
                    ax.axvspan(xv - missing_half_width, xv + missing_half_width, color="#eeeeee", zorder=0)
        ax.set_title(f"{letter}  {title}", loc="left")
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if args.preset != "manuscript":
        fig.suptitle("Center-nearest representative cell: shape trace", y=1.02)
    return fig


def make_volume_trace(summary_df: pd.DataFrame, args: argparse.Namespace) -> plt.Figure | None:
    if summary_df["volume_um3_rod"].isna().all():
        return None

    df = summary_df.copy()
    x = df["frame_index"].to_numpy(dtype=float)
    if args.time_interval_min is not None:
        x = x * args.time_interval_min
        x_label = "Time [min]"
    else:
        x_label = "Frame"

    volume = df["volume_um3_rod"].to_numpy(dtype=float)
    mean_ri = df["mean_ri"].to_numpy(dtype=float)
    mass_pg = df["mass_pg"].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.9), constrained_layout=True, sharex=True)
    missing_mask = ~df["tracked"].to_numpy(dtype=bool)
    missing_half_width = (args.time_interval_min / 2.0) if args.time_interval_min else 0.45
    media_switches = [float(frame) for frame in args.media_switch_frames]

    axes[0].plot(x, volume, color="#1f77b4", lw=1.5)
    axes[0].set_title("A  Rod volume estimate", loc="left")
    axes[0].set_ylabel("Volume [um^3]")
    axes[0].set_ylim(*args.volume_ylim)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    if np.isfinite(mean_ri).any():
        axes[1].plot(x, mean_ri, color="#ff7f0e", lw=1.5)
    else:
        axes[1].text(0.5, 0.5, "Mean RI unavailable", transform=axes[1].transAxes, ha="center", va="center")
    axes[1].set_title("B  Mean RI", loc="left")
    axes[1].set_ylabel("Mean RI")
    axes[1].set_ylim(*args.mean_ri_ylim)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    if np.isfinite(mass_pg).any():
        axes[2].plot(x, mass_pg, color="#2ca02c", lw=1.5)
    else:
        axes[2].text(0.5, 0.5, "Mass unavailable", transform=axes[2].transAxes, ha="center", va="center")
    axes[2].set_title("C  Total mass", loc="left")
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("Total mass [pg]")
    axes[2].set_ylim(*args.mass_ylim)
    axes[2].grid(True, alpha=0.3, linestyle="--")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    if missing_mask.any():
        for ax in axes:
            for xv, is_missing in zip(x, missing_mask):
                if is_missing:
                    ax.axvspan(xv - missing_half_width, xv + missing_half_width, color="#eeeeee", zorder=0)

    for ax in axes:
        for switch_time in media_switches:
            ax.axvline(switch_time, color="#666666", lw=0.9, ls=":", alpha=0.9, zorder=1)

    if args.preset != "manuscript":
        fig.suptitle("Center-nearest representative cell: volume estimate", y=1.02)
    return fig


def build_aligned_stack(
    summary_df: pd.DataFrame,
    crop_size: int,
    args: argparse.Namespace,
) -> tuple[list[pd.Series], list[np.ndarray | None], list[np.ndarray]]:
    rows_used: list[pd.Series] = []
    raw_crops: list[np.ndarray | None] = []
    mask_crops: list[np.ndarray] = []
    for _, row in summary_df.iterrows():
        if not row["tracked"]:
            continue
        raw_crop, mask_crop = build_crops_for_row(
            row,
            crop_size,
            args.align_major_axis,
            args.flip_horizontal,
        )
        rows_used.append(row)
        raw_crops.append(raw_crop)
        mask_crops.append(mask_crop)
    return rows_used, raw_crops, mask_crops


def make_ghost_contour(
    summary_df: pd.DataFrame,
    crop_size: int,
    args: argparse.Namespace,
) -> plt.Figure | None:
    rows_used, _, mask_crops = build_aligned_stack(summary_df, crop_size, args)
    if not mask_crops:
        return None
    fig, ax = plt.subplots(figsize=(3.3, 3.3), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    n = len(mask_crops)
    for idx, (row, mask_crop) in enumerate(zip(rows_used, mask_crops)):
        color = cmap(idx / max(1, n - 1))
        for contour in contours_from_mask(mask_crop):
            ax.plot(contour[:, 1], contour[:, 0], color=color, lw=0.9, alpha=0.9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_axis_off()
    add_scale_bar(
        ax,
        (crop_size, crop_size),
        args.pixel_size_um,
        args.scalebar_um,
        color="#111111",
        linewidth=1.2,
    )
    if rows_used:
        ax.text(0.03, 0.97, frame_time_label(rows_used[0], args.time_interval_min), transform=ax.transAxes, ha="left", va="top")
        ax.text(0.97, 0.97, frame_time_label(rows_used[-1], args.time_interval_min), transform=ax.transAxes, ha="right", va="top")
    if args.preset != "manuscript":
        fig.suptitle("Center-nearest representative cell: ghost contour", y=1.02)
    return fig


def longitudinal_mask_profile(mask_crop: np.ndarray) -> np.ndarray:
    return mask_crop.astype(float).sum(axis=0)


def select_kymograph_rows_every_hours(
    summary_df: pd.DataFrame,
    time_interval_min: float | None,
    sample_hours: float,
) -> pd.DataFrame:
    tracked = summary_df[summary_df["tracked"]].copy()
    if tracked.empty or time_interval_min is None or time_interval_min <= 0 or sample_hours <= 0:
        return tracked.iloc[0:0].copy()

    step = max(1, int(round((sample_hours * 60.0) / time_interval_min)))
    sampled = tracked.iloc[::step].copy()
    if not sampled.empty and int(sampled.iloc[-1]["frame_index"]) != int(tracked.iloc[-1]["frame_index"]):
        sampled = pd.concat([sampled, tracked.iloc[[-1]].copy()], ignore_index=False)
        sampled = sampled[~sampled.index.duplicated(keep="first")]
    return sampled


def select_kymograph_rows_consecutive(summary_df: pd.DataFrame, window_frames: int) -> pd.DataFrame:
    tracked = summary_df[summary_df["tracked"]].copy()
    if tracked.empty or window_frames <= 0:
        return tracked.iloc[0:0].copy()

    frame_indices = tracked["frame_index"].to_numpy(dtype=int)
    splits = np.where(np.diff(frame_indices) != 1)[0] + 1
    segments = np.split(np.arange(len(tracked)), splits)
    if not segments:
        return tracked.iloc[0:0].copy()

    best_segment = max(segments, key=len)
    if len(best_segment) <= window_frames:
        return tracked.iloc[best_segment].copy()

    start = (len(best_segment) - window_frames) // 2
    chosen = best_segment[start : start + window_frames]
    return tracked.iloc[chosen].copy()


def build_kymograph_variants(summary_df: pd.DataFrame, args: argparse.Namespace) -> list[tuple[str, str, pd.DataFrame]]:
    variants: list[tuple[str, str, pd.DataFrame]] = []

    sampled_48h = select_kymograph_rows_every_hours(
        summary_df,
        time_interval_min=args.time_interval_min,
        sample_hours=args.kymograph_sample_hours,
    )
    if not sampled_48h.empty:
        label = f"every {args.kymograph_sample_hours:g} h"
        variants.append(("48h", label, sampled_48h))

    window_df = select_kymograph_rows_consecutive(summary_df, args.kymograph_window_frames)
    if not window_df.empty:
        label = f"contiguous {len(window_df)} frames"
        variants.append(("10frame", label, window_df))

    return variants


def rectangular_longitudinal_profile(raw_crop: np.ndarray, rect_height: int, rect_width: int) -> np.ndarray:
    rect = extract_centered_rectangle(raw_crop, rect_height, rect_width)
    return rect.mean(axis=0).astype(float)


def rectangular_mask_occupancy_profile(mask_crop: np.ndarray, rect_height: int, rect_width: int) -> np.ndarray:
    rect = extract_centered_rectangle(mask_crop.astype(float), rect_height, rect_width)
    return rect.mean(axis=0).astype(float)


def longitudinal_intensity_profile(raw_crop: np.ndarray, mask_crop: np.ndarray) -> np.ndarray:
    weighted = np.where(mask_crop > 0, raw_crop, 0.0)
    counts = mask_crop.sum(axis=0).astype(float)
    sums = weighted.sum(axis=0).astype(float)
    prof = np.full(raw_crop.shape[1], np.nan, dtype=float)
    valid = counts > 0
    prof[valid] = sums[valid] / counts[valid]
    return prof


def make_intensity_kymograph(
    summary_df: pd.DataFrame,
    crop_size: int,
    args: argparse.Namespace,
    variant_label: str | None = None,
) -> plt.Figure | None:
    rows_used, raw_crops, mask_crops = build_aligned_stack(summary_df, crop_size, args)
    rect_margin = max(1, min(4, args.crop_margin // 4))
    rect_height, rect_width = determine_kymograph_box(summary_df, crop_size, rect_margin)
    profiles = []
    labels = []
    for row, raw_crop, _mask_crop in zip(rows_used, raw_crops, mask_crops):
        if raw_crop is None:
            continue
        profiles.append(rectangular_longitudinal_profile(raw_crop, rect_height, rect_width))
        labels.append(row)
    if not profiles:
        return None
    arr = np.stack(profiles, axis=0)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    im = ax.imshow(
        arr.T,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        interpolation="nearest",
        vmin=args.kymograph_vmin,
        vmax=args.kymograph_vmax,
    )
    xtick_idx = np.linspace(0, len(labels) - 1, min(5, len(labels))).round().astype(int)
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([frame_time_label(labels[i], args.time_interval_min) for i in xtick_idx])
    ax.set_xlabel("Time")

    yticks = np.linspace(0, rect_width - 1, 5)
    ax.set_yticks(yticks)
    if args.pixel_size_um:
        ax.set_yticklabels([f"{(y - rect_width / 2) * args.pixel_size_um:.1f}" for y in yticks])
        ax.set_ylabel("Position along major axis [um]")
    else:
        ax.set_ylabel("Position along major axis [px]")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean intensity in axis-aligned rectangle")
    if args.preset != "manuscript":
        title = "Center-nearest representative cell: intensity kymograph"
        if variant_label:
            title = f"{title} ({variant_label})"
        fig.suptitle(title, y=1.02)
    return fig


def make_mask_kymograph(
    summary_df: pd.DataFrame,
    crop_size: int,
    args: argparse.Namespace,
    variant_label: str | None = None,
) -> plt.Figure | None:
    rows_used, _, mask_crops = build_aligned_stack(summary_df, crop_size, args)
    if not mask_crops:
        return None
    rect_margin = max(1, min(4, args.crop_margin // 4))
    rect_height, rect_width = determine_kymograph_box(summary_df, crop_size, rect_margin)
    arr = np.stack(
        [rectangular_mask_occupancy_profile(mask_crop, rect_height, rect_width) for mask_crop in mask_crops],
        axis=0,
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    im = ax.imshow(
        arr.T,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    xtick_idx = np.linspace(0, len(rows_used) - 1, min(5, len(rows_used))).round().astype(int)
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([frame_time_label(rows_used[i], args.time_interval_min) for i in xtick_idx])
    ax.set_xlabel("Time")

    yticks = np.linspace(0, rect_width - 1, 5)
    ax.set_yticks(yticks)
    if args.pixel_size_um:
        ax.set_yticklabels([f"{(y - rect_width / 2) * args.pixel_size_um:.1f}" for y in yticks])
        ax.set_ylabel("Position along major axis [um]")
    else:
        ax.set_ylabel("Position along major axis [px]")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mask occupancy in rectangle")
    if args.preset != "manuscript":
        title = "Center-nearest representative cell: mask kymograph"
        if variant_label:
            title = f"{title} ({variant_label})"
        fig.suptitle(title, y=1.02)
    return fig


def make_qc_overview(summary_df: pd.DataFrame, args: argparse.Namespace) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel()

    valid = summary_df[summary_df["tracked"]].copy()
    if not valid.empty:
        h = int(valid["image_height_px"].dropna().iloc[0])
        w = int(valid["image_width_px"].dropna().iloc[0])
        ax0.plot(valid["centroid_x"], valid["centroid_y"], color="#2a6f97", lw=1.2)
        cx = w / 2.0
        cy = h / 2.0
        cross = max(4.0, min(h, w) * 0.03)
        ax0.plot([cx - cross, cx + cross], [cy, cy], color="red", lw=1.0)
        ax0.plot([cx, cx], [cy - cross, cy + cross], color="red", lw=1.0)
        ax0.set_xlim(0, w)
        ax0.set_ylim(h, 0)
        ax0.set_title("A  Selected centroids", loc="left")
        ax0.set_aspect("equal")
    else:
        ax0.text(0.5, 0.5, "No tracked frames", transform=ax0.transAxes, ha="center", va="center")
        ax0.set_title("A  Selected centroids", loc="left")

    x = summary_df["frame_index"].to_numpy(dtype=float)
    if args.time_interval_min is not None:
        x = x * args.time_interval_min
        xlabel = "Time [min]"
    else:
        xlabel = "Frame"
    ax1.plot(x, summary_df["distance_to_center_px"], color="#2a6f97", lw=1.2)
    ax1.set_title("B  Distance to center", loc="left")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Distance [px]")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    counts = summary_df["selection_flag"].fillna("unknown").value_counts()
    ax2.bar(counts.index.astype(str), counts.values, color="#7a7a7a")
    ax2.set_title("C  Selection flags", loc="left")
    ax2.tick_params(axis="x", rotation=30)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3.plot(x, summary_df["area_px"], color="#6a4c93", lw=1.2)
    ax3.set_title("D  Area QC", loc="left")
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel("Area [px]")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.suptitle("Center-nearest representative cell: QC overview", y=1.02)
    return fig


def save_fig_with_formats(
    fig: plt.Figure,
    kind: str,
    args: argparse.Namespace,
    channel_dir: Path,
    outdir: Path,
    source_tifs: Iterable[str],
) -> dict[str, str]:
    if args.no_save:
        return {}
    saved: dict[str, str] = {}
    params = {
        "indir": str(channel_dir),
        "figure_kind": kind,
        "preset": args.preset,
        "min_area": args.min_area,
        "exclude_border": bool(args.exclude_border),
        "crop_margin": args.crop_margin,
        "align_major_axis": bool(args.align_major_axis),
        "flip_horizontal": bool(args.flip_horizontal),
        "panel_count": args.panel_count,
        "pixel_size_um": args.pixel_size_um,
        "time_interval_min": args.time_interval_min,
        "wavelength_nm": args.wavelength_nm,
        "n_medium": args.n_medium,
        "alpha_ri": args.alpha_ri,
        "ri_calibration": str(args.ri_calibration) if args.ri_calibration else None,
        "calibration_id": getattr(args, "calibration_id_used", None) or args.calibration_id,
        "media_schedule": args.media_schedule,
        "n_milliq": getattr(args, "n_milliq_used", None) or args.n_milliq,
        "kymograph_vmin": args.kymograph_vmin,
        "kymograph_vmax": args.kymograph_vmax,
        "volume_ylim": args.volume_ylim,
        "mean_ri_ylim": args.mean_ri_ylim,
        "mass_ylim": args.mass_ylim,
        "media_switch_frames": args.media_switch_frames,
        "kymograph_sample_hours": args.kymograph_sample_hours,
        "kymograph_window_frames": args.kymograph_window_frames,
    }
    extra_meta = {"local_output_dir": str(outdir.resolve())}
    source_list = list(dict.fromkeys(source_tifs)) if args.attach_source_tifs else None
    for fmt in args.formats:
        path = save_figure(
            fig,
            params=params,
            description=kind,
            fmt=fmt,
            publish=False,
            extra_meta=extra_meta,
            source_tifs=source_list,
        )
        saved[fmt] = str(path)
    return saved


def write_intermediate_outputs(
    summary_df: pd.DataFrame,
    crop_size: int,
    args: argparse.Namespace,
    outdir: Path,
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    summary_csv = outdir / "track_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    outputs = {
        "track_summary_csv": str(summary_csv.resolve()),
    }

    if not args.write_track_tifs:
        return outputs

    mask_stack = []
    crop_stack = []
    raw_crop_stack = []
    has_raw_crop = False

    for _, row in summary_df.iterrows():
        if row["mask_path"]:
            label_img = load_label_image(Path(row["mask_path"]))
            if row["tracked"]:
                selected_mask = build_selected_mask(label_img, int(row["label"]))
            else:
                selected_mask = np.zeros_like(label_img, dtype=np.uint8)
        else:
            h = int(row["image_height_px"]) if pd.notna(row["image_height_px"]) else crop_size
            w = int(row["image_width_px"]) if pd.notna(row["image_width_px"]) else crop_size
            selected_mask = np.zeros((h, w), dtype=np.uint8)
        mask_stack.append(selected_mask.astype(np.uint8))

        raw_crop, mask_crop = build_crops_for_row(
            row,
            crop_size,
            args.align_major_axis,
            args.flip_horizontal,
        )
        crop_stack.append(mask_crop.astype(np.uint8))
        if raw_crop is not None:
            raw_crop_stack.append(raw_crop.astype(np.float32))
            has_raw_crop = True
        else:
            raw_crop_stack.append(np.zeros((crop_size, crop_size), dtype=np.float32))

    track_masks_path = outdir / "track_masks.tif"
    tifffile.imwrite(track_masks_path, np.stack(mask_stack, axis=0))
    outputs["track_masks_tif"] = str(track_masks_path.resolve())

    track_crops_path = outdir / "track_crops.tif"
    tifffile.imwrite(track_crops_path, np.stack(crop_stack, axis=0))
    outputs["track_crops_tif"] = str(track_crops_path.resolve())

    if has_raw_crop:
        track_raw_crops_path = outdir / "track_raw_crops.tif"
        tifffile.imwrite(track_raw_crops_path, np.stack(raw_crop_stack, axis=0).astype(np.float32))
        outputs["track_raw_crops_tif"] = str(track_raw_crops_path.resolve())

    return outputs


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, list):
            result[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            result[key] = value
    return result


def write_metadata(
    outdir: Path,
    channel_dir: Path,
    summary_df: pd.DataFrame,
    args: argparse.Namespace,
    intermediate_outputs: dict[str, str],
    figure_outputs: dict[str, dict[str, str]],
) -> Path:
    flags = summary_df["selection_flag"].value_counts(dropna=False).to_dict()
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "channel_dir": str(channel_dir.resolve()),
        "inference_out_dir": str((channel_dir / "inference_out").resolve()),
        "position_name": detect_pos_name(channel_dir),
        "channel_name": channel_dir.name,
        "selection_rule": "center-nearest mask per frame after min-area and optional border exclusion",
        "args": serializable_args(args),
        "n_frames_total": int(len(summary_df)),
        "n_selected_frames": int(summary_df["tracked"].sum()),
        "selection_flag_counts": {str(k): int(v) for k, v in flags.items()},
        "raw_files": [p for p in summary_df["raw_path"].tolist() if p],
        "mask_files": [p for p in summary_df["mask_path"].tolist() if p],
        "intermediate_outputs": intermediate_outputs,
        "figure_outputs": figure_outputs,
    }
    metadata_path = outdir / "track_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def main() -> int:
    args = parse_runtime_args()
    channel_dir = resolve_channel_dir(args)
    outdir = (args.outdir or default_outdir(channel_dir)).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    apply_style(args.preset)

    frame_pairs = build_frame_pairs(channel_dir)
    if not frame_pairs:
        raise SystemExit(f"No frames found under {channel_dir}")

    print(f"[central_cell_track] channel_dir: {channel_dir}")
    print(f"[central_cell_track] paired frames: {len(frame_pairs)}")

    media_schedule = None
    media_ri = None
    n_milliq = args.n_milliq
    calibration_id_used = None
    if args.ri_calibration is not None:
        cal = load_calibration(args.ri_calibration, args.calibration_id)
        media_ri = cal.media
        calibration_id_used = cal.calibration_id
        if n_milliq is None:
            n_milliq = cal.n_miliq
        if not args.media_schedule:
            raise SystemExit(
                "--media-schedule is required when --ri-calibration is set "
                "(e.g. '0:wo_2,575:wo_0,1439:wo_2')."
            )
        media_schedule = parse_media_schedule(args.media_schedule)
        print(
            f"[central_cell_track] calibration: {cal.calibration_id} "
            f"({cal.calibrated_at})"
        )
        print(f"[central_cell_track] media RI: {cal.media}")
        print(f"[central_cell_track] schedule: {media_schedule}")
        print(f"[central_cell_track] n_milliq for protein density: {n_milliq}")
    elif args.media_schedule:
        raise SystemExit(
            "--media-schedule was given without --ri-calibration; "
            "schedule lookup needs the calibration JSON for media RI values."
        )

    args.calibration_id_used = calibration_id_used
    args.n_milliq_used = n_milliq

    summary_df = build_summary_table(
        frame_pairs,
        min_area=args.min_area,
        exclude_border=args.exclude_border,
        pixel_size_um=args.pixel_size_um,
        wavelength_nm=args.wavelength_nm,
        n_medium=args.n_medium,
        alpha_ri=args.alpha_ri,
        media_schedule=media_schedule,
        media_ri=media_ri,
        n_milliq=n_milliq,
    )
    crop_size = determine_crop_size(summary_df, margin=args.crop_margin)
    panel_rows = select_panel_rows(summary_df, panel_count=args.panel_count)

    intermediate_outputs = write_intermediate_outputs(summary_df, crop_size, args, outdir)

    figure_outputs: dict[str, dict[str, str]] = {}
    selected_source_tifs = []
    if not panel_rows.empty:
        for _, row in panel_rows.iterrows():
            if row["raw_path"]:
                selected_source_tifs.append(row["raw_path"])
            if row["mask_path"]:
                selected_source_tifs.append(row["mask_path"])

    for kind in args.figure:
        fig = None
        if kind == "overlay_strip":
            fig = make_overlay_strip(summary_df, crop_size, args)
        elif kind == "contour_montage":
            fig = make_contour_montage(panel_rows, crop_size, args)
        elif kind == "shape_trace":
            fig = make_shape_trace(summary_df, args)
        elif kind == "volume_trace":
            fig = make_volume_trace(summary_df, args)
        elif kind == "ghost_contour":
            fig = make_ghost_contour(summary_df, crop_size, args)
        elif kind == "intensity_kymograph":
            variants = build_kymograph_variants(summary_df, args)
            if not variants:
                print(f"[central_cell_track] skip {kind}: no valid data")
                continue
            for variant_suffix, variant_label, variant_df in variants:
                fig = make_intensity_kymograph(variant_df, crop_size, args, variant_label=variant_label)
                if fig is None:
                    print(f"[central_cell_track] skip {kind}_{variant_suffix}: no valid data")
                    continue
                save_kind = f"{kind}_{variant_suffix}"
                figure_outputs[save_kind] = save_fig_with_formats(
                    fig,
                    kind=save_kind,
                    args=args,
                    channel_dir=channel_dir,
                    outdir=outdir,
                    source_tifs=selected_source_tifs,
                )
                plt.close(fig)
            continue
        elif kind == "mask_kymograph":
            variants = build_kymograph_variants(summary_df, args)
            if not variants:
                print(f"[central_cell_track] skip {kind}: no valid data")
                continue
            for variant_suffix, variant_label, variant_df in variants:
                fig = make_mask_kymograph(variant_df, crop_size, args, variant_label=variant_label)
                if fig is None:
                    print(f"[central_cell_track] skip {kind}_{variant_suffix}: no valid data")
                    continue
                save_kind = f"{kind}_{variant_suffix}"
                figure_outputs[save_kind] = save_fig_with_formats(
                    fig,
                    kind=save_kind,
                    args=args,
                    channel_dir=channel_dir,
                    outdir=outdir,
                    source_tifs=selected_source_tifs,
                )
                plt.close(fig)
            continue
        elif kind == "qc_overview":
            fig = make_qc_overview(summary_df, args)

        if fig is None:
            print(f"[central_cell_track] skip {kind}: no valid data")
            continue

        figure_outputs[kind] = save_fig_with_formats(
            fig,
            kind=kind,
            args=args,
            channel_dir=channel_dir,
            outdir=outdir,
            source_tifs=selected_source_tifs,
        )
        plt.close(fig)

    metadata_path = write_metadata(
        outdir=outdir,
        channel_dir=channel_dir,
        summary_df=summary_df,
        args=args,
        intermediate_outputs=intermediate_outputs,
        figure_outputs=figure_outputs,
    )

    print(f"[central_cell_track] summary_csv: {intermediate_outputs['track_summary_csv']}")
    print(f"[central_cell_track] metadata_json: {metadata_path}")
    print(f"[central_cell_track] local_output_dir: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
