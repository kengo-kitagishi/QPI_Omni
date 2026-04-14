from __future__ import annotations

import argparse
import json
import math
import re
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
        choices=["overlay_strip", "contour_montage", "shape_trace"],
        default=["overlay_strip", "contour_montage", "shape_trace"],
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
        "--panel-count",
        type=int,
        default=6,
        help="Number of sampled frames to show in montage/strip figures.",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="Pixel size in micrometers for scale bars and metric conversion.",
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
    return parser


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


def compute_major_axis_angle_deg(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if ys.size < 3:
        return 0.0
    coords = np.column_stack([xs - xs.mean(), ys - ys.mean()])
    try:
        _, _, vh = np.linalg.svd(coords, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    vx, vy = vh[0]
    return float(np.degrees(np.arctan2(vy, vx)))


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
) -> tuple[np.ndarray | None, np.ndarray]:
    if not align_major_axis:
        return raw_crop, mask_crop
    angle_deg = compute_major_axis_angle_deg(mask_crop > 0)
    if abs(angle_deg) < 1e-6:
        return raw_crop, mask_crop

    out_raw = None
    if raw_crop is not None:
        out_raw = rotate(
            raw_crop,
            -angle_deg,
            resize=False,
            order=1,
            preserve_range=True,
            mode="constant",
            cval=float(np.median(raw_crop)),
        ).astype(np.float32)
    out_mask = rotate(
        mask_crop.astype(np.float32),
        -angle_deg,
        resize=False,
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    )
    return out_raw, (out_mask > 0.5).astype(np.uint8)


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
) -> pd.DataFrame:
    rows = [select_center_cell_for_frame(pair, min_area=min_area, exclude_border=exclude_border) for pair in frame_pairs]
    df = pd.DataFrame(rows)
    df["frame_label"] = df["frame_name"]
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

    return rotate_crop_pair(raw_crop, mask_crop, align_major_axis=align_major_axis)


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
    panel_rows: pd.DataFrame,
    crop_size: int,
    args: argparse.Namespace,
) -> plt.Figure | None:
    if panel_rows.empty or panel_rows["raw_path"].eq("").all():
        return None
    crops: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    rows_used: list[pd.Series] = []
    for _, row in panel_rows.iterrows():
        raw_crop, mask_crop = build_crops_for_row(row, crop_size, args.align_major_axis)
        if raw_crop is None:
            continue
        crops.append(raw_crop)
        masks.append(mask_crop)
        rows_used.append(row)
    if not crops:
        return None

    stack = np.stack(crops)
    vmin = float(np.nanpercentile(stack, 2))
    vmax = float(np.nanpercentile(stack, 98))
    n = len(crops)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    contour_color = "#ffcc33" if args.preset == "presentation" else "#d55e00"

    for ax, raw_crop, mask_crop, row in zip(axes, crops, masks, rows_used):
        ax.imshow(raw_crop, cmap="gray", vmin=vmin, vmax=vmax)
        for contour in contours_from_mask(mask_crop):
            ax.plot(contour[:, 1], contour[:, 0], color=contour_color, lw=1.0)
        ax.text(
            0.04,
            0.96,
            frame_time_label(row, args.time_interval_min),
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.18", fc=(1, 1, 1, 0.78), ec="none"),
        )
        ax.set_axis_off()

    add_scale_bar(
        axes[-1],
        crops[-1].shape,
        args.pixel_size_um,
        args.scalebar_um,
        color="white",
        linewidth=2.0 if args.preset == "presentation" else 1.2,
        outline_color="#111111",
    )
    if args.preset != "manuscript":
        fig.suptitle("Center-nearest representative cell: overlay strip", y=1.02)
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
        _, mask_crop = build_crops_for_row(row, crop_size, args.align_major_axis)
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
        ax.plot(x, y, color=color, lw=1.2, marker="o", markersize=2.5)
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
        "panel_count": args.panel_count,
        "pixel_size_um": args.pixel_size_um,
        "time_interval_min": args.time_interval_min,
    }
    extra_meta = {"local_output_dir": str(outdir.resolve())}
    source_list = list(dict.fromkeys(source_tifs))
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

    mask_stack = []
    crop_stack = []
    raw_crop_stack = []
    has_raw_crop = False

    for _, row in summary_df.iterrows():
        if row["mask_path"]:
            label_img = load_label_image(Path(row["mask_path"]))
            selected_mask = build_selected_mask(label_img, int(row["label"])) if row["tracked"] else np.zeros_like(label_img, dtype=np.uint8)
        else:
            h = int(row["image_height_px"]) if pd.notna(row["image_height_px"]) else crop_size
            w = int(row["image_width_px"]) if pd.notna(row["image_width_px"]) else crop_size
            selected_mask = np.zeros((h, w), dtype=np.uint8)
        mask_stack.append(selected_mask.astype(np.uint8))

        raw_crop, mask_crop = build_crops_for_row(row, crop_size, args.align_major_axis)
        crop_stack.append(mask_crop.astype(np.uint8))
        if raw_crop is not None:
            raw_crop_stack.append(raw_crop.astype(np.float32))
            has_raw_crop = True
        else:
            raw_crop_stack.append(np.zeros((crop_size, crop_size), dtype=np.float32))

    track_masks_path = outdir / "track_masks.tif"
    tifffile.imwrite(track_masks_path, np.stack(mask_stack, axis=0))

    track_crops_path = outdir / "track_crops.tif"
    tifffile.imwrite(track_crops_path, np.stack(crop_stack, axis=0))

    outputs = {
        "track_summary_csv": str(summary_csv.resolve()),
        "track_masks_tif": str(track_masks_path.resolve()),
        "track_crops_tif": str(track_crops_path.resolve()),
    }

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
    args = build_parser().parse_args()
    channel_dir = resolve_channel_dir(args)
    outdir = (args.outdir or default_outdir(channel_dir)).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    apply_style(args.preset)

    frame_pairs = build_frame_pairs(channel_dir)
    if not frame_pairs:
        raise SystemExit(f"No frames found under {channel_dir}")

    print(f"[central_cell_track] channel_dir: {channel_dir}")
    print(f"[central_cell_track] paired frames: {len(frame_pairs)}")

    summary_df = build_summary_table(
        frame_pairs,
        min_area=args.min_area,
        exclude_border=args.exclude_border,
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
            fig = make_overlay_strip(panel_rows, crop_size, args)
        elif kind == "contour_montage":
            fig = make_contour_montage(panel_rows, crop_size, args)
        elif kind == "shape_trace":
            fig = make_shape_trace(summary_df, args)

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
