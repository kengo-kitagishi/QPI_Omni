#!/usr/bin/env python3
"""
channel_crop.py - Rectangular crop of microfluidic channels from timelapse images

Usage:
  # Step 1: Channel detection & preview -> Save ROI settings to JSON
  python channel_crop.py --dir "e:/Acuisition/kitagishi/260301/movetest_8/Pos4" --detect

  # Manually edit JSON to remove unwanted channels or adjust cy/cx

  # Step 2: Apply to all frames and output per-channel TIFF stacks
  python channel_crop.py --dir "e:/Acuisition/kitagishi/260301/movetest_8/Pos4" --apply

  # Run all at once (detect -> apply)
  python channel_crop.py --dir "e:/Acuisition/kitagishi/260301/movetest_8/Pos4"

ROI JSON format (channels/channel_rois.json):
  [
    {"cy": 112, "cx": 1224, "crop_w": 30, "crop_h": 120},
    ...
  ]
  - cy     : Channel center Y coordinate [px]
  - cx     : Channel center X coordinate [px]  (horizontal center of crop)
  - crop_w : Channel width direction (Y) size [px]
  - crop_h : Channel longitudinal direction (X) size [px]

Output: <dir>/channels/channel_00.tif, channel_01.tif, ...
        Each file is a uint16 TIFF stack with shape (T, crop_w, crop_h)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from figure_logger import save_figure
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

DEFAULT_CROP_W  = 40
DEFAULT_CROP_H  = 270
DEFAULT_PATTERN = "img_*_ph_000.tif"
ROI_FILENAME    = "channel_rois.json"


# ────────────────────────────────────────────────────────────────
#  I/O
# ────────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    return tifffile.imread(str(path))


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo + 1e-9), 0, 1)


# ────────────────────────────────────────────────────────────────
#  Channel detection
# ────────────────────────────────────────────────────────────────

def detect_channels(img: np.ndarray, min_distance: int = 35,
                    prominence_sigma: float = 0.3):
    """Detect channel center Y coordinates from minima of the row-mean profile."""
    profile = np.mean(img.astype(np.float32), axis=1)  # shape: (H,)
    inv = profile.max() - profile
    peaks, _ = find_peaks(inv, distance=min_distance,
                           prominence=inv.std() * prominence_sigma)
    return peaks, profile


def detect_channel_edge_x(img: np.ndarray, cy: int, crop_w: int,
                           side: str = "left",
                           dark_threshold: float = 0.4) -> int:
    """Detect channel edge position in X direction (for raw intensity images, legacy implementation)."""
    y1 = max(0, cy - crop_w // 2)
    y2 = min(img.shape[0], cy + crop_w // 2)
    strip = img[y1:y2, :].astype(np.float32)
    col_profile = np.mean(strip, axis=0)

    lo, hi = col_profile.min(), col_profile.max()
    norm = (col_profile - lo) / (hi - lo + 1e-9)
    is_dark = norm < dark_threshold

    dark_indices = np.where(is_dark)[0]
    if len(dark_indices) == 0:
        return img.shape[1] // 2

    if side == "left":
        return int(dark_indices[0])
    else:
        return int(dark_indices[-1])


def detect_channel_edge_x_gradient(img: np.ndarray, cy: int, crop_w: int,
                                    x_min: int = 100, x_max: int = 400,
                                    smooth_sigma: float = 2.0) -> int:
    """Return cx as the steepest descent point (minimum gradient) of the column-mean profile.

    Searches within [x_min, x_max]. Compatible with phase images (float32).
    Detects the point where the channel wall transitions sharply from near 0 to negative.
    """
    h, w = img.shape
    x_min = max(0, x_min)
    x_max = min(w, x_max)
    y1 = max(0, cy - crop_w // 2)
    y2 = min(h, cy + crop_w // 2)
    strip = img[y1:y2, x_min:x_max].astype(np.float64)
    col_profile = np.mean(strip, axis=0)
    if smooth_sigma > 0:
        col_profile = gaussian_filter1d(col_profile, sigma=smooth_sigma)
    grad = np.gradient(col_profile)
    cx_rel = int(np.argmax(np.abs(grad)))
    return x_min + cx_rel


def _filter_cx_mad(rois: list, mad_thresh: float = 10.0) -> list:
    """Exclude channels with cx MAD outliers."""
    if len(rois) < 3:
        return rois
    cxs = np.array([r["cx"] for r in rois], dtype=np.float64)
    med = float(np.median(cxs))
    mad = float(np.median(np.abs(cxs - med)))
    if mad == 0:
        return rois
    valid = []
    for r, cx in zip(rois, cxs):
        if abs(cx - med) <= mad_thresh * mad:
            valid.append(r)
        else:
            print(f"  skip cy={r['cy']}: cx={int(cx)} MAD outlier "
                  f"(median={med:.0f}, MAD={mad:.1f})")
    return valid


# ────────────────────────────────────────────────────────────────
#  Rectangular crop (no interpolation)
# ────────────────────────────────────────────────────────────────

def extract_rect_roi(img: np.ndarray, cy: int, cx: int,
                     crop_w: int, crop_h: int) -> np.ndarray:
    """Extract a crop_w x crop_h rectangle centered at (cx, cy)."""
    h, w = img.shape
    y1 = cy - crop_w // 2
    y2 = y1 + crop_w
    x1 = cx - crop_h // 2
    x2 = x1 + crop_h

    # Boundary clip (zero-pad if out of bounds)
    pad_y0 = max(0, -y1);  y1 = max(0, y1)
    pad_y1 = max(0, y2 - h); y2 = min(h, y2)
    pad_x0 = max(0, -x1);  x1 = max(0, x1)
    pad_x1 = max(0, x2 - w); x2 = min(w, x2)

    crop = img[y1:y2, x1:x2]
    if any([pad_y0, pad_y1, pad_x0, pad_x1]):
        crop = np.pad(crop, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant")
    return crop


# ────────────────────────────────────────────────────────────────
#  Step 1: Detection & Preview
# ────────────────────────────────────────────────────────────────

def run_detect(img_path: Path, crop_w: int, crop_h: int, out_dir: Path,
               min_dist: int, prominence_sigma: float, side: str = "left",
               dark_threshold: float = 0.4,
               x_start: int = None, x_end: int = None,
               cx_min: int = 100, cx_max: int = 400,
               cx_mad_thresh: float = 10.0):
    img = load_image(img_path)
    h, w = img.shape

    centers, profile = detect_channels(img, min_distance=min_dist,
                                        prominence_sigma=prominence_sigma)

    rois_raw = []
    for cy in centers:
        if x_start is not None and x_end is not None:
            cx = (x_start + x_end) // 2
            ch = x_end - x_start
        else:
            cx = detect_channel_edge_x_gradient(img, int(cy), crop_w,
                                                 x_min=cx_min, x_max=cx_max)
            ch = crop_h
        rois_raw.append({"cy": int(cy), "cx": cx, "crop_w": crop_w, "crop_h": ch})

    # ── cx MAD outlier exclusion ──
    rois_raw = _filter_cx_mad(rois_raw, mad_thresh=cx_mad_thresh)

    # ── Filter: cx range check only ──
    rois = []
    for roi in rois_raw:
        cy_, cx_, cw_, ch_ = roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]
        # cx range check
        if not (cx_min <= cx_ <= cx_max):
            print(f"  skip cy={cy_}: cx={cx_} outside [{cx_min}, {cx_max}]")
            continue
        rois.append(roi)

    print(f"Detected channels: {len(rois_raw)} -> after filtering: {len(rois)}")
    for i, r in enumerate(rois):
        print(f"  ch{i:02d}: y={r['cy']}, x={r['cx']}")

    # ─── Preview ───
    fig, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(14, 7))
    disp = normalize_for_display(img)

    ax_img.imshow(disp, cmap="gray", aspect="equal")
    ax_img.set_title(f"Detected channels ({len(rois)})", fontsize=12)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rois), 1)))
    for i, roi in enumerate(rois):
        actual_ch = roi["crop_h"]
        rect = mpatches.Rectangle(
            (roi["cx"] - actual_ch // 2, roi["cy"] - crop_w // 2),
            actual_ch, crop_w,
            linewidth=1.5, edgecolor=colors[i % 10], facecolor="none",
        )
        ax_img.add_patch(rect)
        ax_img.text(roi["cx"] - actual_ch // 2 + 2,
                    roi["cy"] - crop_w // 2 - 3,
                    f"ch{i:02d}", color=colors[i % 10], fontsize=6)
    ax_img.set_xlim(0, w)
    ax_img.set_ylim(h, 0)

    ax_prof.plot(profile, np.arange(len(profile)), color="steelblue", lw=0.8)
    ax_prof.scatter(profile[centers], centers, color="red", s=20, zorder=5,
                    label="Detected")
    ax_prof.invert_xaxis()
    ax_prof.set_title("Row-mean profile (Y direction)", fontsize=12)
    ax_prof.set_xlabel("Mean intensity")
    ax_prof.set_ylabel("Y coordinate [px]")
    ax_prof.set_ylim(h, 0)
    ax_prof.legend(fontsize=8)

    plt.suptitle(str(img_path), fontsize=8)
    logged_path = save_figure(
        fig,
        params={
            "source_image": str(img_path),
            "n_channels": len(rois),
            "crop_w": int(crop_w),
            "crop_h_default": int(crop_h),
            "min_dist": int(min_dist),
            "prominence_sigma": float(prominence_sigma),
            "side": side,
            "dark_threshold": float(dark_threshold),
            "x_start": x_start,
            "x_end": x_end,
        },
        description=f"channel_detection_preview {img_path.parent.name}",
        publish=False,
        dpi=150,
        fmt="png",
    )
    preview_path = out_dir / "channel_detection_preview.png"
    shutil.copy2(logged_path, preview_path)
    print(f"Preview saved: {preview_path}")
    print(f"figure_logger saved: {logged_path}")
    plt.close(fig)

    roi_path = out_dir / ROI_FILENAME
    with open(roi_path, "w", encoding="utf-8") as f:
        json.dump(rois, f, indent=2, ensure_ascii=False)
    print(f"ROI settings saved: {roi_path}")
    print()
    print("Remove unwanted channels from the JSON or adjust cy/cx, then run --apply.")

    return rois


# ────────────────────────────────────────────────────────────────
#  Step 2: Apply to all frames
# ────────────────────────────────────────────────────────────────

def run_apply(img_dir: Path, pattern: str, rois: list, out_dir: Path):
    files = sorted(img_dir.glob(pattern))
    if not files:
        print(f"No images found: {img_dir / pattern}")
        sys.exit(1)

    n_frames   = len(files)
    n_channels = len(rois)
    print(f"Frames: {n_frames}, Channels: {n_channels}")

    stacks = [[] for _ in range(n_channels)]

    for i, f in enumerate(files):
        if i % 50 == 0:
            print(f"  Processing: {i+1}/{n_frames}  ({f.name})")
        img = load_image(f)
        for ch_idx, roi in enumerate(rois):
            crop = extract_rect_roi(img, roi["cy"], roi["cx"],
                                    roi["crop_w"], roi["crop_h"])
            stacks[ch_idx].append(crop)

    print("Saving...")
    for ch_idx, stack in enumerate(stacks):
        arr = np.array(stack, dtype=np.float32)  # (T, crop_w, crop_h)
        out_path = out_dir / f"channel_{ch_idx:02d}.tif"
        tifffile.imwrite(str(out_path), arr)
        print(f"  ch{ch_idx:02d}: {out_path}  shape={arr.shape}")

    print("Done")


# ────────────────────────────────────────────────────────────────
#  main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rectangular crop of microfluidic channels from timelapse images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir",     required=True, help="Image directory")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Filename pattern")
    parser.add_argument("--detect",  action="store_true", help="Detection only (no --apply)")
    parser.add_argument("--apply",   action="store_true", help="Apply only (requires JSON)")
    parser.add_argument("--crop-w",  type=int, default=DEFAULT_CROP_W,
                        help=f"Channel width direction crop size [px] (default={DEFAULT_CROP_W})")
    parser.add_argument("--crop-h",  type=int, default=DEFAULT_CROP_H,
                        help=f"Channel longitudinal direction crop size [px] (default={DEFAULT_CROP_H})")
    parser.add_argument("--min-dist",    type=int,   default=35,
                        help="Minimum channel detection spacing [px] (default=35)")
    parser.add_argument("--prominence",  type=float, default=0.3,
                        help="Prominence threshold (multiple of std, default=0.3)")
    parser.add_argument("--side",        choices=["left", "right"], default="left",
                        help="Which edge of the channel to use as cx (default=left)")
    parser.add_argument("--dark-threshold", type=float, default=0.4,
                        help="Normalized intensity threshold for dark regions 0-1 (default=0.4)")
    parser.add_argument("--cx-min", type=int, default=100,
                        help="Lower bound of cx search range [px] (default=100)")
    parser.add_argument("--cx-max", type=int, default=400,
                        help="Upper bound of cx search range [px] (default=400)")
    parser.add_argument("--cx-mad-thresh", type=float, default=10.0,
                        help="cx MAD outlier threshold (default=10.0)")
    args = parser.parse_args()

    img_dir = Path(args.dir)
    out_dir = img_dir / "channels"
    out_dir.mkdir(exist_ok=True)

    files = sorted(img_dir.glob(args.pattern))
    if not files:
        print(f"No images found: {img_dir / args.pattern}")
        sys.exit(1)

    do_detect = not args.apply
    do_apply  = not args.detect
    rois = None

    if do_detect:
        rois = run_detect(files[0], args.crop_w, args.crop_h, out_dir,
                          min_dist=args.min_dist,
                          prominence_sigma=args.prominence,
                          cx_min=args.cx_min,
                          cx_max=args.cx_max,
                          cx_mad_thresh=args.cx_mad_thresh)

    if do_apply:
        if rois is None:
            roi_path = out_dir / ROI_FILENAME
            if not roi_path.exists():
                print(f"ROI file not found: {roi_path}")
                print("Run --detect first.")
                sys.exit(1)
            with open(roi_path, encoding="utf-8") as f:
                rois = json.load(f)
        run_apply(img_dir, args.pattern, rois, out_dir)


if __name__ == "__main__":
    main()
