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
    # solid-of-revolution volume integrated from the width profile (px^3).
    # 0.0 for methods that don't compute it. Morphometrics-style: treats each
    # cross-section as a circle of radius w_perp(x)/2 and integrates along the
    # medial axis, so it captures taper/curvature the 2-axis rod formula misses.
    volume_profile_px3: float = 0.0


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
    endcap_trim_frac: float | str = 0.10,
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
    endcap_trim_frac : float or str
        How much of each end of the long axis to drop before averaging the
        width profile into a short axis.

        * ``float`` (default 0.10): fixed fraction of the long axis trimmed
          from each end (legacy behaviour).
        * ``"adaptive"``: estimate the semicircular endcap length from a
          rough short-axis measurement (cap radius ≈ short_axis / 2) and trim
          exactly that many pixels from each end. Short cells have caps that
          eat well past a fixed 10 % window; trimming the true cap length
          removes the aspect-ratio-dependent bias that otherwise inflates the
          measured width as the cell elongates through its cycle.

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
    if isinstance(endcap_trim_frac, str) and endcap_trim_frac == "adaptive":
        # Rough short axis from the central 80 %, then assume each
        # semicircular cap spans ~ (short_axis / 2) pixels and trim that.
        trim0 = int(0.10 * n_wp)
        if n_wp > 2 * trim0 + 1:
            rough_short = float(np.mean(width_profile[trim0:n_wp - trim0]))
        else:
            rough_short = float(np.mean(width_profile))
        cap_len_px = rough_short / 2.0
        trim = int(np.clip(cap_len_px, 1, max(n_wp // 3, 1)))
    else:
        trim = int(float(endcap_trim_frac) * n_wp)
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
# Medial-axis (Morphometrics-style) width measurement
# =========================================================================

def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge replication (length-preserving)."""
    window = max(int(window), 1)
    if window <= 1 or y.size <= 2:
        return y.astype(float)
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(y.astype(float), half, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def rotate_mask_horizontal(binary_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate a single-cell mask so its long axis is horizontal.

    Shared by the width methods and the QC tools. Returns (rotated_bool,
    orientation_rad). Empty masks return (the mask, 0.0).
    """
    mask = binary_mask.astype(bool)
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    if not props:
        return mask, 0.0
    p = max(props, key=lambda r: r.area)
    rot = transform.rotate(
        mask.astype(np.uint8), 90.0 - np.degrees(p.orientation),
        resize=True, order=0, preserve_range=True,
    ).astype(bool)
    return rot, float(p.orientation)


def column_run_counts(rotated: np.ndarray, c0: int, c1: int) -> np.ndarray:
    """Per-column count of vertical connected runs over span [c0, c1].

    A clean rod has exactly one run per body column. Columns with >1 run mean
    the slice crosses several mask segments (bent cell, division neck, a
    neighbour/debris blob in the crop) — there the rotate-and-project width is
    over-counted, so this is the load-bearing diagnostic for mesh quality.
    """
    sub = rotated[:, c0:c1 + 1].astype(np.int8)
    rises = (sub[1:, :] == 1) & (sub[:-1, :] == 0)
    counts = rises.sum(axis=0)
    counts += (sub[0, :] == 1).astype(int)  # a run starting at the top edge
    return counts.astype(int)


def cross_section_quality(binary_mask: np.ndarray) -> tuple[float, int]:
    """(fraction of body columns with >1 cross-section, max cross-sections).

    Body columns = the central span after trimming ~one cap length from each
    end, matching the window the width methods average over.
    """
    rot, _ = rotate_mask_horizontal(binary_mask)
    col_sums = rot.sum(axis=0).astype(float)
    nz = np.where(col_sums > 0)[0]
    if len(nz) < 2:
        return 0.0, 0
    c0, c1 = nz[0], nz[-1]
    n_col = c1 - c0 + 1
    counts = column_run_counts(rot, c0, c1)
    h_vert = col_sums[c0:c1 + 1]
    trim0 = int(0.10 * n_col)
    rough = float(np.mean(h_vert[trim0:n_col - trim0])) if n_col > 2 * trim0 + 1 else float(np.mean(h_vert))
    cap = int(np.clip(rough / 2.0, 1, max(n_col // 3, 1)))
    if n_col > 2 * cap + 1:
        body = counts[cap:n_col - cap]
    else:
        body = counts
    multi_frac = float(np.mean(body > 1)) if body.size else 0.0
    return multi_frac, int(counts.max()) if counts.size else 0


def measure_single_cell_medial(
    binary_mask: np.ndarray,
    label: int = 0,
    pixel_size_um: float = 1.0,
    smoothing_window_frac: float = 0.15,
    plateau_top_frac: float = 0.50,
) -> CellMorphology:
    """Width measurement via a smoothed medial-axis perpendicular profile.

    This is mathematically equivalent to Morphometrics' medial-axis
    perpendicular width when the cell is roughly straight (true for S. pombe
    mothers held in a chamber). Unlike the second-moment ellipse fit, it has
    no aspect-ratio-dependent bias.

    Algorithm
    ---------
    1. Rotate the mask so its long axis is horizontal (same convention as
       ``extract_cell_morphology``).
    2. For every column in the nonzero span measure the vertical chord length
       ``h_vert`` (number of mask pixels) and the column centroid ``y_c``.
    3. Smooth ``y_c`` over a window of ``smoothing_window_frac`` of the span
       to obtain the medial axis; its slope gives the local tangent angle
       ``θ(x) = arctan(dy/dx)``.
    4. Perpendicular width ``w_perp = h_vert · cos(θ)`` — a column slice
       through a band of true width ``w`` tilted by ``θ`` has length
       ``w / cos(θ)``, so the true width is recovered by *multiplying* by
       ``cos(θ)`` (the bias is second order for straight cells, θ≈0).
    5. Short axis = mean ``w_perp`` over the central plateau (columns whose
       width ≥ ``plateau_top_frac`` × max width), which excludes the tapering
       semicircular caps.
    6. Long axis = arc length of the smoothed medial axis across the span,
       ``∫ √(1 + (dy/dx)²) dx`` (≈ horizontal span for straight cells).

    Parameters
    ----------
    binary_mask : 2-D bool array (single cell)
    label : int
    pixel_size_um : float
    smoothing_window_frac : float
        Medial-axis smoothing window as a fraction of the long-axis span.
    plateau_top_frac : float
        Width threshold (fraction of max width) defining the central body.

    Returns
    -------
    CellMorphology
    """
    mask = binary_mask.astype(bool)
    labeled = measure.label(mask)
    props_list = measure.regionprops(labeled)
    if not props_list:
        return CellMorphology(
            label=label, centroid_yx=(0.0, 0.0),
            area_px=0, area_um2=0.0,
            long_axis_px=0.0, long_axis_um=0.0,
            short_axis_px=0.0, short_axis_um=0.0,
            orientation_rad=0.0, width_profile=np.array([]),
        )

    props = props_list[0]
    orientation = props.orientation
    rotation_deg = 90.0 - np.degrees(orientation)
    rotated = transform.rotate(
        mask.astype(np.uint8), rotation_deg,
        resize=True, order=0, preserve_range=True,
    ).astype(bool)

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
            orientation_rad=orientation, width_profile=np.array([1.0]),
            eccentricity=props.eccentricity, solidity=props.solidity,
        )

    c0, c1 = nonzero_cols[0], nonzero_cols[-1]
    h_vert = col_sums[c0:c1 + 1]                  # vertical chord per column
    n_col = len(h_vert)

    # Column centroid (y center) for the medial axis
    rows_idx = np.arange(rotated.shape[0], dtype=float)[:, None]
    col_slice = rotated[:, c0:c1 + 1].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        y_c = (rows_idx * col_slice).sum(axis=0) / np.maximum(col_slice.sum(axis=0), 1e-9)

    window = max(int(smoothing_window_frac * n_col), 3)
    y_smooth = _moving_average(y_c, window)
    dydx = np.gradient(y_smooth)
    theta = np.arctan(dydx)
    cos_t = np.cos(theta)

    w_perp = h_vert * cos_t                       # true perpendicular width

    # Body selection: trim the semicircular caps by their length rather than
    # by a width threshold. A fixed-fraction-of-max plateau lets cap shoulders
    # (width between the threshold and the full body) leak into the mean, and
    # the leaked fraction shrinks as the body grows — which reintroduces an
    # aspect-ratio-dependent drift. Cap length ≈ body radius ≈ rough_short / 2,
    # so trimming that many columns from each end isolates the flat body.
    trim0 = int(0.10 * n_col)
    if n_col > 2 * trim0 + 1:
        rough_short = float(np.mean(w_perp[trim0:n_col - trim0]))
    else:
        rough_short = float(np.mean(w_perp))
    cap_len = int(np.clip(rough_short / 2.0, 1, max(n_col // 3, 1)))
    if n_col > 2 * cap_len + 1:
        body = w_perp[cap_len:n_col - cap_len]
    else:
        body = w_perp
    # plateau_top_frac guards against any residual taper inside the body window.
    w_max = float(np.max(body)) if body.size else 0.0
    if w_max > 0:
        keep = body >= plateau_top_frac * w_max
        short_axis_px = float(np.mean(body[keep])) if keep.any() else float(np.mean(body))
    else:
        short_axis_px = float(np.mean(body))

    # Long axis: arc length of the smoothed medial axis across the span.
    arc_step = np.sqrt(1.0 + dydx ** 2)
    long_axis_px = float(np.sum(arc_step))

    # Solid-of-revolution volume: each column is a disk of radius w_perp/2,
    # integrated along the medial-axis arc length (captures caps + any taper).
    volume_profile_px3 = float(np.sum(np.pi * (w_perp / 2.0) ** 2 * arc_step))

    neck_width_px = float(np.min(w_perp))
    neck_idx = int(np.argmin(w_perp))
    neck_position = neck_idx / max(n_col - 1, 1)

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
        width_profile=w_perp,
        neck_width_px=neck_width_px,
        neck_position=neck_position,
        eccentricity=props.eccentricity,
        solidity=props.solidity,
        volume_profile_px3=volume_profile_px3,
    )


def measure_all_modes(binary_mask: np.ndarray) -> dict | None:
    """One regionprops + one rotation → all per-mode pixel measurements.

    Computationally equivalent to calling skimage regionprops +
    extract_cell_morphology(endcap_trim_frac='adaptive') +
    measure_single_cell_medial() + cross_section_quality() separately, but
    rotates the mask once instead of three times — the batch driver's hot path.
    Returns None for an empty/degenerate mask. All lengths in pixels.
    """
    mask = binary_mask.astype(bool)
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    if not props:
        return None
    p = max(props, key=lambda r: r.area)
    rot = transform.rotate(
        mask.astype(np.uint8), 90.0 - np.degrees(p.orientation),
        resize=True, order=0, preserve_range=True,
    ).astype(bool)
    col_sums = rot.sum(axis=0).astype(float)
    nz = np.where(col_sums > 0)[0]
    if len(nz) < 2:
        return None
    c0, c1 = int(nz[0]), int(nz[-1])
    n_col = c1 - c0 + 1
    h_vert = col_sums[c0:c1 + 1]

    # rough short (central 80%) shared by adaptive trim and cross-section window
    trim0 = int(0.10 * n_col)
    if n_col > 2 * trim0 + 1:
        rough = float(np.mean(h_vert[trim0:n_col - trim0]))
    else:
        rough = float(np.mean(h_vert))
    cap = int(np.clip(rough / 2.0, 1, max(n_col // 3, 1)))

    # --- supersegger_adaptive ---
    core = h_vert[cap:n_col - cap] if n_col > 2 * cap + 1 else h_vert
    adaptive_short = float(np.mean(core))
    adaptive_long = float(n_col)

    # --- medial axis (perpendicular width) ---
    rows_idx = np.arange(rot.shape[0], dtype=float)[:, None]
    col_slice = rot[:, c0:c1 + 1].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        y_c = (rows_idx * col_slice).sum(axis=0) / np.maximum(col_slice.sum(axis=0), 1e-9)
    y_smooth = _moving_average(y_c, max(int(0.15 * n_col), 3))
    dydx = np.gradient(y_smooth)
    arc_step = np.sqrt(1.0 + dydx ** 2)
    w_perp = h_vert * np.cos(np.arctan(dydx))
    if n_col > 2 * trim0 + 1:
        rough_m = float(np.mean(w_perp[trim0:n_col - trim0]))
    else:
        rough_m = float(np.mean(w_perp))
    cap_m = int(np.clip(rough_m / 2.0, 1, max(n_col // 3, 1)))
    body = w_perp[cap_m:n_col - cap_m] if n_col > 2 * cap_m + 1 else w_perp
    w_max = float(np.max(body)) if body.size else 0.0
    if w_max > 0:
        keep = body >= 0.50 * w_max
        medial_short = float(np.mean(body[keep])) if keep.any() else float(np.mean(body))
    else:
        medial_short = float(np.mean(body))
    medial_long = float(np.sum(arc_step))
    medial_profile_px3 = float(np.sum(np.pi * (w_perp / 2.0) ** 2 * arc_step))

    # --- cross-section quality ---
    counts = column_run_counts(rot, c0, c1)
    body_counts = counts[cap:n_col - cap] if n_col > 2 * cap + 1 else counts
    multi_frac = float(np.mean(body_counts > 1)) if body_counts.size else 0.0
    max_xsec = int(counts.max()) if counts.size else 0

    return {
        "sk_minor_px": float(p.minor_axis_length),
        "sk_major_px": float(p.major_axis_length),
        "area_px": int(p.area),
        "adaptive_short_px": adaptive_short,
        "adaptive_long_px": adaptive_long,
        "medial_short_px": medial_short,
        "medial_long_px": medial_long,
        "medial_profile_px3": medial_profile_px3,
        "multi_xsec_frac": multi_frac,
        "max_xsec": max_xsec,
    }


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


def synthesize_rod_mask(L_px: float, r_px: float, canvas_pad: int = 10) -> np.ndarray:
    """Ideal rod = cylinder body + 2 semicircular caps (true L, true 2r known).

    Returns a horizontal rod so the ground-truth long axis is ``L_px`` and the
    true width is ``2 * r_px``. Used to quantify each method's bias vs the
    second-moment ellipse fit across aspect ratios.
    """
    H = int(2 * r_px) + 2 * canvas_pad
    W = int(L_px) + 2 * canvas_pad
    yy, xx = np.indices((H, W))
    cy = H / 2.0
    cyl_x0, cyl_x1 = canvas_pad + r_px, canvas_pad + L_px - r_px
    cylinder = (xx >= cyl_x0) & (xx <= cyl_x1) & (np.abs(yy - cy) <= r_px)
    left_cap = ((xx - cyl_x0) ** 2 + (yy - cy) ** 2 <= r_px ** 2) & (xx <= cyl_x0)
    right_cap = ((xx - cyl_x1) ** 2 + (yy - cy) ** 2 <= r_px ** 2) & (xx >= cyl_x1)
    return cylinder | left_cap | right_cap


def _validate_correction():
    """Aspect-ratio sweep: compare width measured by each method vs truth.

    For an ideal rod of known width 2r, report the ratio measured/true for:
      - skimage minor axis (the biased baseline)
      - fixed_trim short axis (legacy 10 %)
      - adaptive_trim short axis (Task A)
      - medial_axis short axis (Task B, Morphometrics-equivalent)
    Pass criterion: adaptive_trim and medial_axis within +/-1 % of truth.
    """
    print("\n=== width-method validation (true 2r recovery vs aspect ratio) ===")
    print(f"{'AR':>4} {'true_2r':>8} {'skimage':>9} {'fixed':>9} "
          f"{'adaptive':>9} {'medial':>9}")
    # Use a large radius so single-pixel rasterization rounding (±1 px) is a
    # small fraction of the width; define truth from the actual rasterized body
    # chord (central column of the horizontal rod) rather than the nominal 2r,
    # which differs by up to a pixel after rasterization.
    r_px = 25.0
    # AR=1 is a degenerate disk (no cylindrical body); the biological mother
    # cycles between AR≈2.0 and 3.4, so the pass criterion is evaluated over
    # the rod range AR≥2 where a "width" is well defined.
    worst_adaptive = 0.0
    worst_medial = 0.0
    for AR in [1.0, 2.0, 3.0, 5.0, 10.0]:
        L_px = 2.0 * r_px * AR
        mask = synthesize_rod_mask(L_px, r_px)
        # rasterized body width = vertical chord at the central column
        col_sums_h = mask.sum(axis=0)
        true_2r = float(col_sums_h[len(col_sums_h) // 2])

        labeled = measure.label(mask)
        props = measure.regionprops(labeled)[0]
        sk_minor = float(props.minor_axis_length)

        m_fixed = extract_cell_morphology(mask, pixel_size_um=1.0,
                                          endcap_trim_frac=0.10)
        m_adapt = extract_cell_morphology(mask, pixel_size_um=1.0,
                                          endcap_trim_frac="adaptive")
        m_med = measure_single_cell_medial(mask, pixel_size_um=1.0)

        r_sk = sk_minor / true_2r
        r_fx = m_fixed.short_axis_px / true_2r
        r_ad = m_adapt.short_axis_px / true_2r
        r_md = m_med.short_axis_px / true_2r
        if AR >= 2.0:
            worst_adaptive = max(worst_adaptive, abs(r_ad - 1.0))
            worst_medial = max(worst_medial, abs(r_md - 1.0))
        print(f"{AR:>4.0f} {true_2r:>8.2f} {r_sk:>9.4f} {r_fx:>9.4f} "
              f"{r_ad:>9.4f} {r_md:>9.4f}")
    ok_a = "PASS" if worst_adaptive <= 0.01 else "FAIL"
    ok_m = "PASS" if worst_medial <= 0.01 else "FAIL"
    print(f"\n[rod range AR>=2] worst |adaptive-1| = {100*worst_adaptive:.2f} % "
          f"[{ok_a}]  (<=1 % to pass)")
    print(f"[rod range AR>=2] worst |medial-1|   = {100*worst_medial:.2f} % "
          f"[{ok_m}]  (<=1 % to pass)")


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
    _validate_correction()
