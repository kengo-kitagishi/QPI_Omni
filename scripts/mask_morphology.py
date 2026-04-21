"""
mask_morphology.py — Mask smoothing and axis extraction for QPI cell analysis

Smoothing of segmentation masks and major/minor axis and width profile
extraction using the SuperSegger-style rotate-and-project algorithm.

Usage:
    from mask_morphology import extract_all_cells, morphology_to_dataframe

    morphs = extract_all_cells(labeled_mask, pixel_size_um=0.348)
    df = morphology_to_dataframe(morphs, frame_index=0)

Algorithm reference:
    SuperSegger (Stylianidou et al. 2016) — rotate mask to align long axis,
    project along columns to obtain width profile, extract L1/L2mean/neck.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure, morphology, transform


# =========================================================================
# Data structure
# =========================================================================

@dataclass
class CellMorphology:
    """Shape measurements for a single cell mask."""

    label: int
    centroid_yx: tuple[float, float]
    area_px: int
    area_um2: float
    long_axis_px: float
    long_axis_um: float
    short_axis_px: float          # mean width (endcap 10% trimmed)
    short_axis_um: float
    orientation_rad: float        # same convention as skimage regionprops
    width_profile: np.ndarray = field(repr=False)   # 1-D, pixels
    neck_width_px: float = 0.0
    neck_position: float = 0.5    # 0..1 along long axis
    eccentricity: float = 0.0
    solidity: float = 1.0


# =========================================================================
# Mask smoothing
# =========================================================================

def smooth_mask(
    binary_mask: np.ndarray,
    closing_radius: int = 3,
    opening_radius: int = 1,
    fill_holes: bool = True,
    gaussian_sigma: float = 1.0,
    rethreshold: float = 0.5,
) -> np.ndarray:
    """Smooth a single-cell binary mask.

    Pipeline:
        morphological closing  → fills small gaps / concavities
        morphological opening  → removes thin protrusions
        hole filling           → fills internal holes
        Gaussian blur + threshold → smooth contour edges

    Parameters
    ----------
    binary_mask : 2-D bool or uint8 array
    closing_radius : int
        Disk radius for closing (~1 µm at 0.348 µm/px with default 3).
    opening_radius : int
        Disk radius for opening. 1 removes single-pixel protrusions.
    fill_holes : bool
        Fill internal holes via binary_fill_holes.
    gaussian_sigma : float
        Gaussian sigma for edge smoothing. 0 to skip.
    rethreshold : float
        Threshold after Gaussian blur (0.5 preserves area).

    Returns
    -------
    np.ndarray (bool)
    """
    mask = binary_mask.astype(bool)

    if closing_radius > 0:
        selem_c = morphology.disk(closing_radius)
        mask = ndimage.binary_closing(mask, structure=selem_c)

    if opening_radius > 0:
        selem_o = morphology.disk(opening_radius)
        mask = ndimage.binary_opening(mask, structure=selem_o)

    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)

    if gaussian_sigma > 0:
        blurred = ndimage.gaussian_filter(mask.astype(np.float64),
                                          sigma=gaussian_sigma)
        mask = blurred > rethreshold

    return mask.astype(bool)


# =========================================================================
# Axis extraction (rotate-and-project)
# =========================================================================

def extract_cell_morphology(
    binary_mask: np.ndarray,
    label: int = 1,
    pixel_size_um: float = 0.348,
    endcap_trim_frac: float = 0.10,
) -> CellMorphology:
    """Extract long/short axis and width profile from a single-cell mask.

    Algorithm (SuperSegger style):
        1. regionprops → orientation, centroid, eccentricity, solidity
        2. Rotate mask so long axis is horizontal
        3. Column-wise sum → width profile
        4. Long axis = span of nonzero columns
        5. Short axis = mean of width profile (endcap trimmed)
        6. Neck = minimum of width profile

    Parameters
    ----------
    binary_mask : 2-D bool array (single cell)
    label : int
        Label ID for bookkeeping.
    pixel_size_um : float
    endcap_trim_frac : float
        Fraction of long axis to trim from each end when computing
        mean short axis (default 0.10 = 10%).

    Returns
    -------
    CellMorphology
    """
    mask = binary_mask.astype(bool)
    labeled = measure.label(mask)
    props_list = measure.regionprops(labeled)
    if not props_list:
        # empty mask fallback
        return CellMorphology(
            label=label, centroid_yx=(0.0, 0.0),
            area_px=0, area_um2=0.0,
            long_axis_px=0.0, long_axis_um=0.0,
            short_axis_px=0.0, short_axis_um=0.0,
            orientation_rad=0.0,
            width_profile=np.array([]),
        )

    props = props_list[0]
    orientation = props.orientation  # radians, angle of major axis vs row axis

    # --- Rotate mask so long axis is horizontal ---
    # skimage orientation: angle between row axis (y) and major axis, CCW.
    # To align major axis with column axis (x, horizontal):
    #   rotate CCW by (90° - orientation_deg)
    rotation_deg = 90.0 - np.degrees(orientation)
    rotated = transform.rotate(
        mask.astype(np.uint8), rotation_deg,
        resize=True, order=0, preserve_range=True,
    ).astype(bool)

    # --- Width profile: sum along rows for each column ---
    col_sums = rotated.sum(axis=0).astype(float)
    nonzero_cols = np.where(col_sums > 0)[0]

    if len(nonzero_cols) < 2:
        return CellMorphology(
            label=label,
            centroid_yx=(float(props.centroid[0]), float(props.centroid[1])),
            area_px=int(props.area),
            area_um2=props.area * pixel_size_um ** 2,
            long_axis_px=1.0, long_axis_um=pixel_size_um,
            short_axis_px=1.0, short_axis_um=pixel_size_um,
            orientation_rad=orientation,
            width_profile=np.array([1.0]),
            eccentricity=props.eccentricity,
            solidity=props.solidity,
        )

    # Long axis
    col_start = nonzero_cols[0]
    col_end = nonzero_cols[-1]
    long_axis_px = float(col_end - col_start + 1)

    # Width profile (only within nonzero span)
    width_profile = col_sums[col_start:col_end + 1]

    # Short axis: mean width with endcap trimming
    n_wp = len(width_profile)
    trim = int(endcap_trim_frac * n_wp)
    if trim > 0 and n_wp > 2 * trim + 1:
        core = width_profile[trim:-trim]
    else:
        core = width_profile
    short_axis_px = float(np.mean(core))

    # Neck detection
    neck_width_px = float(np.min(width_profile))
    neck_idx = int(np.argmin(width_profile))
    neck_position = neck_idx / max(n_wp - 1, 1)

    return CellMorphology(
        label=label,
        centroid_yx=(float(props.centroid[0]), float(props.centroid[1])),
        area_px=int(props.area),
        area_um2=props.area * pixel_size_um ** 2,
        long_axis_px=long_axis_px,
        long_axis_um=long_axis_px * pixel_size_um,
        short_axis_px=short_axis_px,
        short_axis_um=short_axis_px * pixel_size_um,
        orientation_rad=orientation,
        width_profile=width_profile,
        neck_width_px=neck_width_px,
        neck_position=neck_position,
        eccentricity=props.eccentricity,
        solidity=props.solidity,
    )


# =========================================================================
# Multi-cell operations
# =========================================================================

def smooth_labeled_mask(
    labeled_mask: np.ndarray,
    margin: int = 5,
    **smooth_kwargs,
) -> np.ndarray:
    """Apply smooth_mask to each label independently.

    Each cell is cropped with a bounding-box margin, smoothed, then
    written back. This prevents adjacent cells from merging.

    Parameters
    ----------
    labeled_mask : 2-D uint16 array (label image)
    margin : int
        Pixels of padding around each cell's bounding box.
    **smooth_kwargs
        Forwarded to smooth_mask().

    Returns
    -------
    np.ndarray (uint16)
    """
    output = np.zeros_like(labeled_mask)
    labels = np.unique(labeled_mask)
    labels = labels[labels != 0]

    for lbl in labels:
        cell_mask = labeled_mask == lbl
        ys, xs = np.where(cell_mask)
        r0 = max(ys.min() - margin, 0)
        r1 = min(ys.max() + margin + 1, labeled_mask.shape[0])
        c0 = max(xs.min() - margin, 0)
        c1 = min(xs.max() + margin + 1, labeled_mask.shape[1])

        crop = cell_mask[r0:r1, c0:c1]
        smoothed = smooth_mask(crop, **smooth_kwargs)
        output[r0:r1, c0:c1][smoothed] = lbl

    return output


def extract_all_cells(
    labeled_mask: np.ndarray,
    pixel_size_um: float = 0.348,
    smooth: bool = True,
    margin: int = 5,
    min_area_px: int = 20,
    **smooth_kwargs,
) -> list[CellMorphology]:
    """Extract morphology for every cell in a labeled mask.

    Parameters
    ----------
    labeled_mask : 2-D uint16 array
    pixel_size_um : float
    smooth : bool
        Whether to smooth each cell mask before extraction.
    margin : int
        BBox padding for cropping.
    min_area_px : int
        Skip regions smaller than this (debris).
    **smooth_kwargs
        Forwarded to smooth_mask().

    Returns
    -------
    list[CellMorphology]
    """
    results = []
    labels = np.unique(labeled_mask)
    labels = labels[labels != 0]

    for lbl in labels:
        cell_mask = labeled_mask == lbl
        if cell_mask.sum() < min_area_px:
            continue

        # Crop for efficiency
        ys, xs = np.where(cell_mask)
        r0 = max(ys.min() - margin, 0)
        r1 = min(ys.max() + margin + 1, labeled_mask.shape[0])
        c0 = max(xs.min() - margin, 0)
        c1 = min(xs.max() + margin + 1, labeled_mask.shape[1])
        crop = cell_mask[r0:r1, c0:c1]

        if smooth:
            crop = smooth_mask(crop, **smooth_kwargs)

        morph = extract_cell_morphology(
            crop, label=int(lbl), pixel_size_um=pixel_size_um,
        )
        # Adjust centroid to global coordinates
        morph.centroid_yx = (
            morph.centroid_yx[0] + r0,
            morph.centroid_yx[1] + c0,
        )
        results.append(morph)

    return results


# =========================================================================
# DataFrame conversion
# =========================================================================

def morphology_to_dataframe(
    morphologies: list[CellMorphology],
    frame_index: int | None = None,
) -> pd.DataFrame:
    """Convert morphology list to a pandas DataFrame.

    Column names are chosen for compatibility with existing pipeline scripts.
    """
    rows = []
    for m in morphologies:
        row = {
            "label": m.label,
            "centroid_y": m.centroid_yx[0],
            "centroid_x": m.centroid_yx[1],
            "area_px": m.area_px,
            "area_um2": m.area_um2,
            "long_axis_px": m.long_axis_px,
            "long_axis_um": m.long_axis_um,
            "short_axis_px": m.short_axis_px,
            "short_axis_um": m.short_axis_um,
            "orientation_rad": m.orientation_rad,
            "neck_width_px": m.neck_width_px,
            "neck_position": m.neck_position,
            "eccentricity": m.eccentricity,
            "solidity": m.solidity,
        }
        if frame_index is not None:
            row["frame"] = frame_index
        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================================
# Synthetic test utilities
# =========================================================================

def make_synthetic_rod(
    length_px: int = 40,
    width_px: int = 10,
    angle_deg: float = 30.0,
    canvas_size: int = 80,
) -> np.ndarray:
    """Create a synthetic rod-shaped mask for testing.

    The rod is a rectangle with semicircular endcaps, centered and rotated.

    Returns
    -------
    np.ndarray (bool), shape (canvas_size, canvas_size)
    """
    from skimage.draw import disk as draw_disk

    mask = np.zeros((canvas_size, canvas_size), dtype=bool)
    cy, cx = canvas_size // 2, canvas_size // 2
    r = width_px / 2

    # Rectangle body (horizontal, will rotate later)
    body_half = (length_px - width_px) / 2  # half-length of cylindrical part
    y0 = int(cy - r)
    y1 = int(cy + r)
    x0 = int(cx - body_half)
    x1 = int(cx + body_half)
    mask[y0:y1 + 1, x0:x1 + 1] = True

    # Semicircular endcaps
    rr, cc = draw_disk((cy, x0), r + 0.5, shape=mask.shape)
    mask[rr, cc] = True
    rr, cc = draw_disk((cy, x1), r + 0.5, shape=mask.shape)
    mask[rr, cc] = True

    # Rotate
    if angle_deg != 0:
        mask = transform.rotate(
            mask.astype(np.uint8), angle_deg,
            resize=False, order=0, preserve_range=True,
        ).astype(bool)

    return mask


# =========================================================================
# Self-test
# =========================================================================

def _self_test():
    """Quick self-test with synthetic rod masks."""
    print("=== mask_morphology self-test ===\n")

    for length, width, angle in [(40, 10, 0), (30, 8, 45), (50, 12, -20)]:
        mask = make_synthetic_rod(length, width, angle, canvas_size=100)
        morph = extract_cell_morphology(mask, pixel_size_um=1.0)

        err_long = abs(morph.long_axis_px - length)
        err_short = abs(morph.short_axis_px - width)

        status_l = "OK" if err_long <= 3 else "WARN"
        status_s = "OK" if err_short <= 3 else "WARN"

        print(f"  Rod({length}x{width}, {angle}°):")
        print(f"    long_axis={morph.long_axis_px:.1f}  (expected {length}, "
              f"err={err_long:.1f}) [{status_l}]")
        print(f"    short_axis={morph.short_axis_px:.1f} (expected {width}, "
              f"err={err_short:.1f}) [{status_s}]")
        print(f"    neck_width={morph.neck_width_px:.1f}, "
              f"neck_pos={morph.neck_position:.2f}")
        print()

    # Noisy mask test
    print("  Noisy rod (40x10, 0°, holes + salt):")
    mask = make_synthetic_rod(40, 10, 0, canvas_size=100)
    rng = np.random.default_rng(42)
    # Add holes
    holes = rng.random(mask.shape) < 0.05
    noisy = mask & ~holes
    # Add salt
    salt = rng.random(mask.shape) < 0.01
    noisy = noisy | salt

    smoothed = smooth_mask(noisy)
    morph_noisy = extract_cell_morphology(noisy, pixel_size_um=1.0)
    morph_clean = extract_cell_morphology(smoothed, pixel_size_um=1.0)

    print(f"    Before smooth: long={morph_noisy.long_axis_px:.1f}, "
          f"short={morph_noisy.short_axis_px:.1f}")
    print(f"    After smooth:  long={morph_clean.long_axis_px:.1f}, "
          f"short={morph_clean.short_axis_px:.1f}")
    print(f"    Expected:      long=40, short=10")
    print()

    print("=== Self-test complete ===")


if __name__ == "__main__":
    _self_test()
