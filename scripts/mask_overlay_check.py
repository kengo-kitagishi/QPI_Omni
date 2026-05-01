"""
mask_overlay_check.py — Overlay masks on sub_rawraw images with axis visualization

sub_raw 画像にセグメンテーションマスクの輪郭・major axis・minor axis を
オーバーレイ描画して、マスク品質を目視確認するためのスクリプト。

Usage:
  python mask_overlay_check.py                            # default targets
  python mask_overlay_check.py --pos Pos9 --ch ch09       # single channel
  python mask_overlay_check.py --frames 0 100 500         # specific frames
  python mask_overlay_check.py --every 100                 # every 100th frame
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
from skimage import measure

# ── Add scripts/ to path for local imports ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mask_morphology import extract_cell_morphology, smooth_mask
from scipy import ndimage as ndi

# ── Configuration ──────────────────────────────────────────────────
PH_ROOT = Path(r"F:\260405\ph_260405")
CROP_SUB = "crop_sub_rawraw"
PIXEL_SIZE_UM = 0.348
WAVELENGTH_NM = 663.0
N_MEDIUM = 1.333
ALPHA_RI = 0.00018

# Default targets
DEFAULT_TARGETS: list[tuple[str, str]] = [
    ("Pos9", "ch09"),
    ("Pos9", "ch10"),
    ("Pos16", "ch04"),
]

# Smooth params: Gaussian only (no closing/opening/hole-fill)
SMOOTH_KWARGS = dict(
    closing_radius=0,
    opening_radius=0,
    fill_holes=False,
    gaussian_sigma=1.0,
    rethreshold=0.5,
)


# ── Core functions ─────────────────────────────────────────────────

def draw_axis_line(
    ax: plt.Axes,
    cy: float, cx: float,
    angle_rad: float,
    length_px: float,
    color: str = "cyan",
    linewidth: float = 1.5,
    label: str | None = None,
) -> None:
    """Draw a line centered at (cx, cy) with given orientation and length."""
    dx = length_px / 2 * np.cos(angle_rad)
    dy = length_px / 2 * np.sin(angle_rad)
    ax.plot(
        [cx - dx, cx + dx], [cy - dy, cy + dy],
        color=color, linewidth=linewidth, label=label,
    )


def overlay_single_frame(
    img: np.ndarray,
    mask: np.ndarray,
    frame_label: str = "",
    min_area_px: int = 20,
    pixel_size_um: float = PIXEL_SIZE_UM,
    margin: int = 5,
) -> plt.Figure:
    """Create overlay figure: image + mask contours + major/minor axes.

    Returns a matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: raw image + mask contour ──
    ax_left = axes[0]
    ax_left.imshow(img, cmap="gray", interpolation="none")
    ax_left.set_title(f"sub_rawraw + mask contour\n{frame_label}", fontsize=10)

    # Draw contours for each label
    labels = np.unique(mask)
    labels = labels[labels != 0]

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(labels), 1)))

    for i, lbl in enumerate(labels):
        cell_binary = (mask == lbl).astype(np.uint8)
        if cell_binary.sum() < min_area_px:
            continue
        contours = measure.find_contours(cell_binary, 0.5)
        for c in contours:
            ax_left.plot(c[:, 1], c[:, 0], color=colors[i % len(colors)],
                         linewidth=1.2, alpha=0.9)

    ax_left.axis("off")

    # ── Right: image + smoothed contour + axes ──
    ax_right = axes[1]
    ax_right.imshow(img, cmap="gray", interpolation="none")
    ax_right.set_title(f"smoothed contour + axes\n{frame_label}", fontsize=10)

    legend_added = {"major": False, "minor": False}

    for i, lbl in enumerate(labels):
        cell_binary = (mask == lbl)
        if cell_binary.sum() < min_area_px:
            continue

        # Crop for efficiency
        ys, xs = np.where(cell_binary)
        r0 = max(ys.min() - margin, 0)
        r1 = min(ys.max() + margin + 1, mask.shape[0])
        c0 = max(xs.min() - margin, 0)
        c1 = min(xs.max() + margin + 1, mask.shape[1])
        crop = cell_binary[r0:r1, c0:c1]

        # 1px erosion → Gaussian smoothing
        eroded_crop = ndi.binary_erosion(crop, iterations=1)
        smoothed_crop = smooth_mask(eroded_crop, **SMOOTH_KWARGS)

        # Draw smoothed contour
        smoothed_full = np.zeros_like(mask, dtype=bool)
        smoothed_full[r0:r1, c0:c1] = smoothed_crop
        contours = measure.find_contours(smoothed_full.astype(np.uint8), 0.5)
        for c in contours:
            ax_right.plot(c[:, 1], c[:, 0], color=colors[i % len(colors)],
                          linewidth=1.2, alpha=0.9)

        # Extract morphology from smoothed crop
        morph = extract_cell_morphology(
            smoothed_crop, label=int(lbl), pixel_size_um=pixel_size_um,
        )

        if morph.area_px == 0:
            continue

        # Centroid in global coords
        cy_global = morph.centroid_yx[0] + r0
        cx_global = morph.centroid_yx[1] + c0

        # skimage orientation: angle between row-axis (y-down) and major axis, CCW
        # Convert to angle from x-axis in image coords (y-down)
        theta = morph.orientation_rad
        major_angle = np.pi / 2 - theta
        minor_angle = major_angle + np.pi / 2

        # Draw major axis (long axis)
        draw_axis_line(
            ax_right, cy_global, cx_global,
            angle_rad=major_angle,
            length_px=morph.long_axis_px,
            color="#00ffff",
            linewidth=1.5,
            label="major axis" if not legend_added["major"] else None,
        )
        legend_added["major"] = True

        # Draw minor axis (short axis)
        draw_axis_line(
            ax_right, cy_global, cx_global,
            angle_rad=minor_angle,
            length_px=morph.short_axis_px,
            color="#ff6600",
            linewidth=1.5,
            label="minor axis" if not legend_added["minor"] else None,
        )
        legend_added["minor"] = True

        # Label text
        ax_right.text(
            cx_global, cy_global - morph.short_axis_px / 2 - 3,
            f"L={morph.long_axis_um:.1f}\nW={morph.short_axis_um:.1f}",
            color="white", fontsize=6, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5),
        )

    ax_right.axis("off")
    if legend_added["major"]:
        ax_right.legend(loc="upper right", fontsize=8, framealpha=0.7)

    fig.tight_layout()
    return fig


def process_channel(
    pos: str,
    ch: str,
    frame_indices: list[int] | None = None,
    every: int | None = None,
    max_frames: int = 10,
    out_dir: Path | None = None,
) -> None:
    """Process a single channel: load images + masks, create overlay figures."""
    channel_dir = PH_ROOT / pos / "output_phase" / "channels" / CROP_SUB / ch
    inf_dir = channel_dir / "inference_out"

    if not channel_dir.is_dir():
        print(f"  [skip] {pos}/{ch}: channel dir not found")
        return
    if not inf_dir.is_dir():
        print(f"  [skip] {pos}/{ch}: inference_out not found")
        return

    # List available mask files
    mask_files = sorted(inf_dir.glob("*_masks.tif"))
    if not mask_files:
        print(f"  [skip] {pos}/{ch}: no mask files")
        return

    # Determine which frames to process
    if frame_indices is not None:
        selected = [mask_files[i] for i in frame_indices if i < len(mask_files)]
    elif every is not None:
        selected = mask_files[::every]
    else:
        # Sample evenly
        n = len(mask_files)
        step = max(1, n // max_frames)
        selected = mask_files[::step][:max_frames]

    # Output directory
    if out_dir is None:
        out_dir = channel_dir / "mask_overlay_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {pos}/{ch}: {len(mask_files)} total, processing {len(selected)} frames")
    print(f"  output → {out_dir}")

    for mf in selected:
        # Derive image path from mask filename
        img_name = mf.stem.replace("_masks", "") + ".tif"
        img_path = channel_dir / img_name

        if not img_path.exists():
            print(f"    [skip] {img_name}: image not found")
            continue

        img = tifffile.imread(str(img_path))
        mask = tifffile.imread(str(mf))

        frame_label = f"{pos}/{ch}/{mf.stem}"

        fig = overlay_single_frame(img, mask, frame_label=frame_label)

        out_path = out_dir / f"{mf.stem}_overlay.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved: {out_path.name}")


# ── Time-series extraction ─────────────────────────────────────────

def calc_rod_volume_um3(major_px: float, minor_px: float, pixel_size_um: float) -> float:
    """Rod volume: cylinder + two hemispherical caps."""
    length_um = float(major_px) * pixel_size_um
    width_um = float(minor_px) * pixel_size_um
    r_um = width_um / 2.0
    h_um = length_um - 2.0 * r_um
    if h_um < 0:
        return float((4.0 / 3.0) * np.pi * (r_um ** 3))
    return float((4.0 / 3.0) * np.pi * (r_um ** 3) + np.pi * (r_um ** 2) * h_um)


def calc_mean_ri(
    total_phase: float,
    volume_um3: float,
    pixel_size_um: float = PIXEL_SIZE_UM,
    wavelength_nm: float = WAVELENGTH_NM,
    n_medium: float = N_MEDIUM,
) -> float:
    """Mean RI from integrated phase and rod volume."""
    if not np.isfinite(total_phase) or not np.isfinite(volume_um3) or volume_um3 <= 0:
        return np.nan
    wavelength_um = wavelength_nm * 1e-3
    pixel_area_um2 = pixel_size_um ** 2
    return n_medium + (total_phase * wavelength_um * pixel_area_um2) / (2.0 * np.pi * volume_um3)


def extract_full_timeseries(
    pos: str,
    ch: str,
    min_area_px: int = 20,
    margin: int = 5,
    pixel_size_um: float = PIXEL_SIZE_UM,
) -> pd.DataFrame:
    """Extract major/minor axis, volume, and mean RI for all frames.

    For each frame, reports the *largest* cell (by area).
    Returns a DataFrame with columns:
    frame, long_axis_um, short_axis_um, area_um2, volume_um3,
    total_phase, mean_ri, mass_pg, n_cells.
    """
    channel_dir = PH_ROOT / pos / "output_phase" / "channels" / CROP_SUB / ch
    inf_dir = channel_dir / "inference_out"
    mask_files = sorted(inf_dir.glob("*_masks.tif"))

    nan_row = dict(frame=0, long_axis_um=np.nan, short_axis_um=np.nan,
                   area_um2=np.nan, volume_um3=np.nan,
                   total_phase=np.nan, mean_ri=np.nan, mass_pg=np.nan,
                   n_cells=0)

    rows = []
    for fi, mf in enumerate(mask_files):
        mask = tifffile.imread(str(mf))
        labels = np.unique(mask)
        labels = labels[labels != 0]

        if len(labels) == 0:
            row = dict(nan_row)
            row["frame"] = fi
            rows.append(row)
            continue

        # Load phase image for RI calculation
        img_name = mf.stem.replace("_masks", "") + ".tif"
        img_path = channel_dir / img_name
        phase_img = None
        if img_path.exists():
            phase_img = tifffile.imread(str(img_path)).astype(np.float64)

        best_morph = None
        best_area = 0
        best_total_phase = np.nan
        for lbl in labels:
            cell_binary = (mask == lbl)
            area = cell_binary.sum()
            if area < min_area_px:
                continue

            ys, xs = np.where(cell_binary)
            r0 = max(ys.min() - margin, 0)
            r1 = min(ys.max() + margin + 1, mask.shape[0])
            c0 = max(xs.min() - margin, 0)
            c1 = min(xs.max() + margin + 1, mask.shape[1])
            crop = cell_binary[r0:r1, c0:c1]

            eroded = ndi.binary_erosion(crop, iterations=1)
            smoothed = smooth_mask(eroded, **SMOOTH_KWARGS)
            morph = extract_cell_morphology(
                smoothed, label=int(lbl), pixel_size_um=pixel_size_um,
            )

            if morph.area_px > best_area:
                best_area = morph.area_px
                best_morph = morph
                # Integrated phase from smoothed mask in global coords
                if phase_img is not None:
                    smoothed_global = np.zeros_like(mask, dtype=bool)
                    smoothed_global[r0:r1, c0:c1] = smoothed
                    best_total_phase = float(np.sum(phase_img[smoothed_global]))

        if best_morph is None or best_morph.area_px == 0:
            row = dict(nan_row)
            row["frame"] = fi
            row["n_cells"] = len(labels)
            rows.append(row)
        else:
            vol = calc_rod_volume_um3(
                best_morph.long_axis_px, best_morph.short_axis_px, pixel_size_um,
            )
            ri = calc_mean_ri(best_total_phase, vol, pixel_size_um)
            conc = (ri - N_MEDIUM) / ALPHA_RI if np.isfinite(ri) else np.nan
            mass = conc * vol if np.isfinite(conc) else np.nan
            rows.append(dict(
                frame=fi,
                long_axis_um=best_morph.long_axis_um,
                short_axis_um=best_morph.short_axis_um,
                area_um2=best_morph.area_um2,
                volume_um3=vol,
                total_phase=best_total_phase,
                mean_ri=ri,
                mass_pg=mass,
                n_cells=len(labels),
            ))

        if (fi + 1) % 500 == 0:
            print(f"    [{fi + 1}/{len(mask_files)}]")

    return pd.DataFrame(rows)


def plot_full_timeseries(
    series: list[tuple[str, pd.DataFrame]],
) -> plt.Figure:
    """Plot volume, mean RI, total mass, major/minor axis (5 panels)."""
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    for label, df in series:
        valid = df.dropna(subset=["volume_um3"])
        axes[0].plot(valid["frame"], valid["volume_um3"],
                     linewidth=0.6, alpha=0.8, label=label)
        axes[1].plot(valid["frame"], valid["mean_ri"],
                     linewidth=0.6, alpha=0.8, label=label)
        axes[2].plot(valid["frame"], valid["mass_pg"],
                     linewidth=0.6, alpha=0.8, label=label)
        axes[3].plot(valid["frame"], valid["long_axis_um"],
                     linewidth=0.6, alpha=0.8, label=label)
        axes[4].plot(valid["frame"], valid["short_axis_um"],
                     linewidth=0.6, alpha=0.8, label=label)

    axes[0].set_ylabel("Volume [um^3]")
    axes[0].set_title("A  Rod volume estimate")
    axes[0].set_ylim(0.0, 400.0)
    axes[0].legend(fontsize=7, loc="upper right", ncol=2)

    axes[1].set_ylabel("Mean RI")
    axes[1].set_title("B  Mean RI")
    axes[1].set_ylim(1.355, 1.400)
    axes[1].legend(fontsize=7, loc="upper right", ncol=2)

    axes[2].set_ylabel("Total mass [pg]")
    axes[2].set_title("C  Total mass")
    axes[2].set_ylim(0.0, 40000.0)
    axes[2].legend(fontsize=7, loc="upper right", ncol=2)

    axes[3].set_ylabel("Major axis (µm)")
    axes[3].set_title("D  Major axis (long axis)")
    axes[3].legend(fontsize=7, loc="upper right", ncol=2)

    axes[4].set_ylabel("Minor axis (µm)")
    axes[4].set_xlabel("Frame")
    axes[4].set_title("E  Minor axis (short axis)")
    axes[4].legend(fontsize=7, loc="upper right", ncol=2)

    fig.tight_layout()
    return fig


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay masks on sub_rawraw with axis visualization",
    )
    parser.add_argument("--pos", type=str, help="Position (e.g. Pos9)")
    parser.add_argument("--ch", type=str, help="Channel (e.g. ch09)")
    parser.add_argument(
        "--frames", type=int, nargs="+",
        help="Specific frame indices to process",
    )
    parser.add_argument(
        "--every", type=int,
        help="Process every Nth frame",
    )
    parser.add_argument(
        "--max-frames", type=int, default=10,
        help="Max frames to sample when no --frames/--every (default: 10)",
    )
    parser.add_argument(
        "--timeseries-only", action="store_true",
        help="Skip overlay figures, only extract axis time-series",
    )
    args = parser.parse_args()

    if args.pos and args.ch:
        targets = [(args.pos, args.ch)]
    else:
        targets = DEFAULT_TARGETS

    print("=" * 60)
    print("mask_overlay_check")
    print(f"  PH_ROOT: {PH_ROOT}")
    print(f"  Smoothing: erode 1px → Gaussian (sigma=1.0)")
    print(f"  Targets: {targets}")
    print("=" * 60)

    # Step 1: Overlay figures (sampled frames)
    if not args.timeseries_only:
        for pos, ch in targets:
            print(f"\n[overlay] {pos}/{ch}")
            process_channel(
                pos, ch,
                frame_indices=args.frames,
                every=args.every,
                max_frames=args.max_frames,
            )

    # Step 2: Full time-series (volume, mean RI, axes)
    print("\n" + "=" * 60)
    print("Full time-series (volume, mean RI, axes)")
    print("=" * 60)

    ts_series: list[tuple[str, pd.DataFrame]] = []
    for pos, ch in targets:
        label = f"{pos}_{ch}"
        print(f"\n[timeseries] {label}")
        df = extract_full_timeseries(pos, ch)
        valid = df["volume_um3"].notna().sum()
        print(f"  → {len(df)} frames, {valid} with data")
        ts_series.append((label, df))

        # Save CSV
        channel_dir = PH_ROOT / pos / "output_phase" / "channels" / CROP_SUB / ch
        csv_path = channel_dir / "mask_overlay_check" / "full_timeseries.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(csv_path), index=False)
        print(f"  → CSV: {csv_path}")

    if ts_series:
        fig = plot_full_timeseries(ts_series)

        # Build npz data dict from all series
        data_dict = {}
        for label, df in ts_series:
            for col in df.columns:
                data_dict[f"{label}_{col}"] = df[col].values

        from figure_logger import save_figure
        save_figure(
            fig,
            params={
                "ph_root": str(PH_ROOT),
                "crop_sub": CROP_SUB,
                "targets": [f"{p}_{c}" for p, c in targets],
                "n_series": len(ts_series),
                "smooth": "erode_1px + gaussian_sigma1.0",
                "pixel_size_um": PIXEL_SIZE_UM,
                "wavelength_nm": WAVELENGTH_NM,
                "n_medium": N_MEDIUM,
                "alpha_ri": ALPHA_RI,
            },
            description="Volume / mean RI / axis time-series (erode 1px + Gaussian smooth)",
            data=data_dict,
        )
        plt.close(fig)
        print("\nTime-series plot saved via figure_logger.")

    print("\nDone.")


if __name__ == "__main__":
    main()
