"""mask_volume_schematic.py — illustrate the mask-direct medial-axis volume.

This draws the *actual* computation used by the corrected-volume pipeline
(``mask_morphology.measure_single_cell_medial`` / ``measure_all_modes``): the
cell is treated as a solid of revolution, where every column along the medial
axis contributes a disk of radius ``w_perp / 2`` and the volume is

    V = Σ_columns  π · (w_perp / 2)²  · arc_step           (pixels³)

The schematic reproduces that geometry faithfully and overlays it on the cell
in its natural orientation, like the bottom-right panel of the QPI methods
figure:

    * cyan   — the RAW mask contour (exactly what the measurement sees; the
               measurement path does NOT smooth the mask, so neither do we)
    * red    — the medial / long axis (the smoothed centerline the algorithm
               integrates along)
    * gray   — the short-axis cross-sections (each ``w_perp`` chord, perpendicular
               to the local tangent), i.e. the disks being stacked

Faithfulness note
-----------------
The volume integration uses EVERY column (1-px spacing ≈ 0.346 µm); there is no
sub-sampling in the real computation. Drawing every 1-px slice would fill the
cell solid, so the gray slices are sub-sampled for legibility (``--slice-step``,
display only). The reported volume is always the full per-column integral.

How the geometry is recovered
-----------------------------
The measurement runs in a rotated frame (long axis made horizontal). We replicate
skimage's ``transform.rotate(resize=True)`` with an explicit ``SimilarityTransform``
so we keep the coordinate map, compute the per-column geometry there exactly as
``measure_single_cell_medial`` does, then map the medial axis and slice endpoints
back into the original (un-rotated) crop frame with that transform. The replicate
is asserted array-equal to ``transform.rotate`` in the self-test, so the mapping
is version-independent.

Usage
-----
    # synthetic demo (no data needed) — writes results/260517/schematic_demo.png
    python scripts/mask_volume_schematic.py --synthetic

    # real mother cell
    python scripts/mask_volume_schematic.py --pos Pos27 --ch ch06 --frame 600
    python scripts/mask_volume_schematic.py --pos Pos27 --ch ch06   # auto-pick frame
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from skimage import measure, transform
from skimage.transform import SimilarityTransform, warp

sys.path.insert(0, str(Path(__file__).parent))
from mask_morphology import _moving_average  # noqa: E402

# pipeline constants (match mask_morphology.measure_all_modes exactly)
SMOOTH_WINDOW_FRAC = 0.15
# EFD adopted method uses a stronger centerline smoothing so the perpendicular
# cross-sections align to the cell axis (user choice 2026-06; section tilt
# max 16.8 deg -> 10.5 deg vs 0.15, volume change <0.5 %). NOT shared with the
# cos-theta reproduction above, which must stay at 0.15 to match the pipeline.
EFD_SMOOTH_WINDOW_FRAC = 0.30
PIXEL_SIZE_UM_DEFAULT = 0.34567514677103717


# =========================================================================
# Geometry container
# =========================================================================

@dataclass
class MedialSchematic:
    """Geometry of the solid-of-revolution volume, in original crop coords.

    All coordinates are (x=col, y=row) in the pixel frame of the input mask.
    """

    contour_xy: list[np.ndarray]              # raw mask contour(s), each (N, 2) xy
    medial_xy: np.ndarray                     # medial axis polyline (M, 2) xy
    slice_p0_xy: np.ndarray                   # short-axis chord start points (M, 2)
    slice_p1_xy: np.ndarray                   # short-axis chord end points   (M, 2)
    w_perp_px: np.ndarray = field(repr=False)  # perpendicular width per column (M,)
    h_vert_px: np.ndarray = field(repr=False)  # raw vertical chord per column (M,)
    theta_rad: np.ndarray = field(repr=False)  # local medial-axis tangent angle (M,)
    arc_step_px: np.ndarray = field(repr=False)  # arc length per column (M,)
    cap_len: int = 0                          # #columns each end that form the
    #                                           hemispherical cap (perpendicular
    #                                           width is degenerate there)
    volume_px3: float = 0.0
    pixel_size_um: float = PIXEL_SIZE_UM_DEFAULT

    @property
    def volume_um3(self) -> float:
        return self.volume_px3 * self.pixel_size_um ** 3

    @property
    def long_axis_px(self) -> float:
        return float(np.sum(self.arc_step_px))


# =========================================================================
# Rotation that keeps the coordinate map
# =========================================================================

def rotate_with_map(mask: np.ndarray, angle_deg: float):
    """Replicate ``skimage.transform.rotate(mask, angle_deg, resize=True, order=0)``
    and also return the transform that maps rotated coords back to original.

    Returns
    -------
    rotated : 2-D bool array
        Identical to ``transform.rotate(mask.astype(uint8), angle_deg,
        resize=True, order=0, preserve_range=True).astype(bool)``.
    to_original : callable
        ``to_original(xy)`` maps points (x=col, y=row) in the rotated frame to
        the original frame. This is exactly the ``inverse_map`` skimage hands to
        ``warp``, so ``rotated`` and the geometry stay consistent by construction.
    """
    rows, cols = mask.shape[:2]
    center = np.array((cols, rows)) / 2.0 - 0.5
    tform = (SimilarityTransform(translation=-center)
             + SimilarityTransform(rotation=np.deg2rad(angle_deg))
             + SimilarityTransform(translation=center))

    # resize: grow the canvas so nothing is clipped (same as skimage)
    corners = np.array([[0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0]])
    warped_corners = tform.inverse(corners)
    minc, minr = warped_corners.min(axis=0)
    maxc, maxr = warped_corners.max(axis=0)
    out_rows = int(np.round(maxr - minr + 1))
    out_cols = int(np.round(maxc - minc + 1))
    tform = SimilarityTransform(translation=(minc, minr)) + tform

    rotated = warp(mask.astype(float), tform, output_shape=(out_rows, out_cols),
                   order=0, preserve_range=True) > 0.5
    return rotated, tform


# =========================================================================
# Core: reproduce measure_single_cell_medial geometry, mapped to original frame
# =========================================================================

def medial_axis_geometry(
    binary_mask: np.ndarray,
    pixel_size_um: float = PIXEL_SIZE_UM_DEFAULT,
    smoothing_window_frac: float = SMOOTH_WINDOW_FRAC,
) -> MedialSchematic | None:
    """Recover the solid-of-revolution geometry the pipeline integrates.

    Mirrors ``mask_morphology.measure_single_cell_medial`` column-for-column
    (rotate to horizontal → per-column vertical chord ``h_vert`` and centroid
    ``y_c`` → smoothed medial axis → ``w_perp = h_vert·cosθ`` → disk integral),
    then maps the medial axis and every short-axis chord back to the original
    crop frame so they can be drawn over the cell in its natural orientation.
    """
    mask = binary_mask.astype(bool)
    labeled = measure.label(mask)
    props_list = measure.regionprops(labeled)
    if not props_list:
        return None
    props = max(props_list, key=lambda r: r.area)

    # 1) rotate so the long axis is horizontal (keep the coordinate map)
    rotation_deg = 90.0 - np.degrees(props.orientation)
    rotated, to_original = rotate_with_map(mask, rotation_deg)

    col_sums = rotated.sum(axis=0).astype(float)
    nonzero_cols = np.where(col_sums > 0)[0]
    if len(nonzero_cols) < 2:
        return None
    c0, c1 = int(nonzero_cols[0]), int(nonzero_cols[-1])
    cols = np.arange(c0, c1 + 1)
    n_col = cols.size
    h_vert = col_sums[c0:c1 + 1]                       # vertical chord per column

    # 2) per-column centroid -> smoothed medial axis (axis smoothing only;
    #    the mask itself is never smoothed)
    rows_idx = np.arange(rotated.shape[0], dtype=float)[:, None]
    col_slice = rotated[:, c0:c1 + 1].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        y_c = (rows_idx * col_slice).sum(axis=0) / np.maximum(col_slice.sum(axis=0), 1e-9)
    window = max(int(smoothing_window_frac * n_col), 3)
    y_smooth = _moving_average(y_c, window)

    # 3) local tangent -> perpendicular width and arc length (pipeline formulae)
    dydx = np.gradient(y_smooth)
    theta = np.arctan(dydx)
    w_perp = h_vert * np.cos(theta)
    arc_step = np.sqrt(1.0 + dydx ** 2)
    volume_px3 = float(np.sum(np.pi * (w_perp / 2.0) ** 2 * arc_step))

    # 4) build the geometry in the rotated frame
    medial_rot = np.column_stack([cols.astype(float), y_smooth])         # (M, 2) xy
    # short-axis chord: centered on the medial axis, perpendicular to the
    # local tangent. tangent = (cosθ, sinθ); perpendicular = (-sinθ, cosθ).
    perp = np.column_stack([-np.sin(theta), np.cos(theta)])              # (M, 2)
    half = (w_perp / 2.0)[:, None]
    p0_rot = medial_rot - half * perp
    p1_rot = medial_rot + half * perp

    # 5) map every point back to the original crop frame
    medial_xy = to_original(medial_rot)
    p0_xy = to_original(p0_rot)
    p1_xy = to_original(p1_rot)

    # 5b) cap length = #columns each end occupied by the hemispherical cap.
    #     The perpendicular-width model is only valid along the cylindrical body;
    #     at the rounded tip a "perpendicular disk" of diameter w_perp can exceed
    #     the contour. cap ~= body radius ~= median(h_vert)/2 (same heuristic the
    #     measurement uses to trim caps before averaging the width).
    cap_len = int(np.clip(np.median(h_vert) / 2.0, 1, max(n_col // 3, 1)))

    # 6) raw mask contour in the original frame (find_contours -> (row, col))
    contours = measure.find_contours(mask.astype(float), 0.5)
    contour_xy = [np.column_stack([c[:, 1], c[:, 0]]) for c in contours]  # -> xy

    return MedialSchematic(
        contour_xy=contour_xy,
        medial_xy=medial_xy,
        slice_p0_xy=p0_xy,
        slice_p1_xy=p1_xy,
        w_perp_px=w_perp,
        h_vert_px=h_vert,
        theta_rad=theta,
        arc_step_px=arc_step,
        cap_len=cap_len,
        volume_px3=volume_px3,
        pixel_size_um=pixel_size_um,
    )


# =========================================================================
# Adopted method: EFD-smoothed contour + single midpoint update
# =========================================================================

def _close_contour(c: np.ndarray) -> np.ndarray:
    return c if np.allclose(c[0], c[-1]) else np.vstack([c, c[:1]])


def efd_smooth_contour(contour_xy: np.ndarray, k: int = 6, n_points: int = 512) -> np.ndarray:
    """Fourier-descriptor (EFD-equivalent) low-pass of a closed contour.

    Resample to uniform arc length, FFT the complex boundary, keep the DC term
    plus the lowest ``k`` harmonics each side, inverse FFT. A low-order Fourier
    truncation cannot self-intersect or add local wiggles, removes the pixel
    staircase while preserving size (area within ~0.1 %), and is insensitive to
    ``k``. Use ``k>=10`` for a dividing cell with a real neck so the concavity
    is not bridged.
    """
    c = _close_contour(np.asarray(contour_xy, dtype=float))
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    tu = np.linspace(0.0, t[-1], n_points, endpoint=False)
    Z = np.fft.fft(np.interp(tu, t, c[:, 0]) + 1j * np.interp(tu, t, c[:, 1]))
    keep = np.zeros(n_points, dtype=bool)
    keep[0] = True
    keep[1:k + 1] = True
    keep[-k:] = True
    Z[~keep] = 0
    zs = np.fft.ifft(Z)
    return _close_contour(np.column_stack([zs.real, zs.imag]))


def _chord_to_contour(contour: np.ndarray, P: np.ndarray, d: np.ndarray):
    """Two contour-intersection points of the infinite line P + t*d straddling
    P (nearest positive and nearest negative t). Returns (p_neg, p_pos) or None.
    contour is a closed (N, 2) xy polyline."""
    a = contour[:-1]
    b = contour[1:]
    e = b - a
    r = a - P
    det = -d[0] * e[:, 1] + e[:, 0] * d[1]
    ok = np.abs(det) > 1e-9
    t = np.full(len(e), np.nan)
    s = np.full(len(e), np.nan)
    t[ok] = (-r[ok, 0] * e[ok, 1] + e[ok, 0] * r[ok, 1]) / det[ok]
    s[ok] = (d[0] * r[ok, 1] - d[1] * r[ok, 0]) / det[ok]
    val = ok & (s >= 0) & (s <= 1)
    tv = t[val]
    pos = tv[tv > 1e-6]
    neg = tv[tv < -1e-6]
    if pos.size == 0 or neg.size == 0:
        return None
    return P + neg.max() * d, P + pos.min() * d


def efd_section_geometry(
    binary_mask: np.ndarray,
    pixel_size_um: float = PIXEL_SIZE_UM_DEFAULT,
    efd_k: int = 6,
    smoothing_window_frac: float = EFD_SMOOTH_WINDOW_FRAC,
) -> MedialSchematic | None:
    """ADOPTED volume method: EFD-smoothed contour + ONE midpoint update.

    1. Seed the centerline and the perpendicular section directions from the
       stable cos-theta medial geometry (single pass, no iteration — iterating
       the perpendicular-from-tangent step diverges without Odermatt's
       crossing-removal/re-spacing guards).
    2. Smooth the raw mask contour with ``efd_smooth_contour`` (de-stairs the
       square-pixel boundary without biasing size).
    3. For each section, intersect the perpendicular line with the smoothed
       contour: the two intersection points give the short axis (their distance)
       and the long-axis update (their midpoint).
    4. Volume = sum of disks pi*(w/2)^2 * ds along the updated centerline.

    The cross-sections are bounded by the contour, so they never overshoot;
    ``cap_len`` is 0 (no cap trimming needed). Volume is measured on the smoothed
    contour (only ~+0.5 % vs the raw contour; the dominant volume driver is the
    width definition, not the smoothing).
    """
    base = medial_axis_geometry(binary_mask, pixel_size_um=pixel_size_um,
                                smoothing_window_frac=smoothing_window_frac)
    if base is None:
        return None
    contour = efd_smooth_contour(max(base.contour_xy, key=len), k=efd_k)

    # Section directions are perpendicular to the centerline's OWN local tangent
    # (not the regionprops second-moment axis), so the drawn cross-sections are
    # actually perpendicular to the centerline. Using the centerline polyline's
    # tangent (one shot, no centerline feedback) keeps this stable — unlike
    # iterating the midpoint update, which diverges.
    centers = base.medial_xy
    tan = np.gradient(centers, axis=0)
    tan = tan / np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    perp = np.column_stack([-tan[:, 1], tan[:, 0]])      # rotate tangent +90 deg

    p0 = centers.copy()
    p1 = centers.copy()
    mids = centers.copy()
    w = np.zeros(len(centers))
    for i, (P, d) in enumerate(zip(centers, perp)):
        c = _chord_to_contour(contour, P, d)
        if c is None:
            continue
        a, b = c
        p0[i], p1[i] = a, b
        mids[i] = (a + b) / 2.0
        w[i] = float(np.linalg.norm(b - a))

    tan = np.gradient(mids, axis=0)
    theta = np.arctan2(tan[:, 1], tan[:, 0])
    arc_step = np.linalg.norm(tan, axis=1)
    volume_px3 = float(np.sum(np.pi * (w / 2.0) ** 2 * arc_step))

    return MedialSchematic(
        contour_xy=[contour],
        medial_xy=mids,
        slice_p0_xy=p0,
        slice_p1_xy=p1,
        w_perp_px=w,
        h_vert_px=w,
        theta_rad=theta,
        arc_step_px=arc_step,
        cap_len=0,
        volume_px3=volume_px3,
        pixel_size_um=pixel_size_um,
    )


# =========================================================================
# Plot
# =========================================================================

def plot_schematic(
    geo: MedialSchematic,
    slice_step: int = 3,
    title: str | None = None,
    show_caps: bool = False,
    method_note: str | None = None,
    contour_color: str = "deepskyblue",
    axis_color: str = "red",
    slice_color: str = "0.55",
):
    """Render cyan contour + red long axis + gray short-axis slices.

    Slices are the TRUE perpendicular width ``w_perp`` (the disks the volume
    integrates) — never clipped, so their length is the real measured value.

    slice_step is DISPLAY ONLY: every ``slice_step``-th column is drawn so the
    chords read as hatching. The reported volume always uses every column.
    show_caps : also draw the hemispherical-cap columns. Off by default because
        a "perpendicular disk" is geometrically degenerate at the rounded tip
        (its diameter can exceed the contour); the caps are shown as outline
        only. The volume integral is unaffected either way.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    p0, p1 = geo.slice_p0_xy, geo.slice_p1_xy
    n = len(geo.medial_xy)
    cap = 0 if show_caps else int(geo.cap_len)
    body = np.arange(cap, n - cap)

    # gray short-axis cross-sections (the disks being stacked) — drawn first so
    # the contour and axis sit on top
    sel = body[::max(int(slice_step), 1)]
    for i in sel:
        ax.plot([p0[i, 0], p1[i, 0]], [p0[i, 1], p1[i, 1]],
                color=slice_color, lw=0.8, zorder=1)

    # cyan raw mask contour
    for c in geo.contour_xy:
        ax.plot(c[:, 0], c[:, 1], color=contour_color, lw=2.2, zorder=3)

    # red medial / long axis
    ax.plot(geo.medial_xy[:, 0], geo.medial_xy[:, 1],
            color=axis_color, lw=2.0, zorder=4)

    ax.set_aspect("equal")
    ax.invert_yaxis()           # image convention: row 0 at top
    ax.margins(0.08)            # keep the cell off the frame edges
    ax.axis("off")

    n_col = len(geo.medial_xy)
    # body width is a property of the cell, independent of whether caps are
    # drawn: always average over the cap-trimmed cylindrical body.
    cb = int(geo.cap_len)
    body_w = geo.w_perp_px[cb:n_col - cb] if (cb > 0 and n_col > 2 * cb) else geo.w_perp_px
    step_note = ("every section" if slice_step <= 1 else f"every {slice_step}")
    if cb == 0:
        cap_note = "all sections drawn"
    elif show_caps:
        cap_note = "caps shown"
    else:
        cap_note = f"caps ({geo.cap_len}px/end) as outline only"
    note = method_note or "solid-of-revolution medial-axis integral"
    caption = (f"V = {geo.volume_um3:.2f} µm³    "
               f"L = {geo.long_axis_px * geo.pixel_size_um:.2f} µm    "
               f"⟨w⟩_body = {np.mean(body_w) * geo.pixel_size_um:.2f} µm\n"
               f"{note}   ·   {n_col} sections   ·   {step_note}, {cap_note}")
    # caption below the axes so it never overlaps the cell
    fig.text(0.5, 0.02, caption, ha="center", va="bottom",
             fontsize=8, family="monospace")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


# =========================================================================
# Data loading (real mother cell)
# =========================================================================

def load_mother_crop(pos: str, ch: str, frame: int | None, margin: int = 8,
                     use_cache: bool = True):
    """Load the raw binary crop of the tracked mother (rank=1) at a frame.

    Returns (binary_crop, frame, pixel_size_um). If ``frame`` is None, picks the
    rank=1 frame whose long axis is near the 75th percentile (a well-elongated
    growth-phase cell). Crops are cached under results/260517/_crop_cache so
    repeat renders skip the (slow) inbox index scan; only an explicit frame is
    cached (auto-pick needs the lineage CSV anyway).
    """
    import pandas as pd

    cache_dir = Path(__file__).resolve().parent.parent / "results" / "260517" / "_crop_cache"
    if use_cache and frame is not None:
        cp = cache_dir / f"{pos}_{ch}_f{frame}.npz"
        if cp.exists():
            d = np.load(cp)
            return d["binary"].astype(bool), int(d["frame"]), float(d["px"])

    from qpi_paths import (find_lineage_csv, find_run_params,
                           mask_dir_for_channel)
    from recompute_axes_from_masks import (_label_at_centroid,
                                           _load_label_image, build_mask_index)

    csv_path = find_lineage_csv(pos, ch)
    if csv_path is None:
        raise FileNotFoundError(f"no lineage CSV for {pos}_{ch}")
    inf_dir = mask_dir_for_channel(csv_path)
    if inf_dir is None:
        raise FileNotFoundError(f"no mask dir for {pos}_{ch}")

    params_path = find_run_params(csv_path)
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path else {}
    px = float(params.get("pixel_size_um", PIXEL_SIZE_UM_DEFAULT))

    df = pd.read_csv(csv_path)
    m = df[df["rank"] == 1].sort_values("frame").copy()
    if m.empty:
        raise ValueError(f"no rank=1 rows for {pos}_{ch}")

    mask_index = build_mask_index(inf_dir)
    if frame is None:
        # Prefer a well-elongated mother (clearest illustration of stacked
        # disks): target the 75th percentile of long axis, which lands in the
        # growth phase rather than the short starvation cells.
        if "long_axis_um" in m.columns:
            target = m["long_axis_um"].quantile(0.75)
            m = m.assign(_d=(m["long_axis_um"] - target).abs()).sort_values("_d")
        for cand in m["frame"].astype(int):
            if cand in mask_index:
                frame = int(cand)
                break
        if frame is None:
            raise ValueError("no rank=1 frame has a matching mask file")
    if frame not in mask_index:
        raise FileNotFoundError(f"no mask file for frame {frame}")

    row = m[m["frame"] == frame].iloc[0]
    mask = _load_label_image(mask_index[frame])
    lbl = _label_at_centroid(mask, row["centroid_y_px"], row["centroid_x_px"])
    if lbl == 0:
        raise ValueError(f"mother label not found at frame {frame}")
    binary_full = mask == lbl
    ys, xs = np.where(binary_full)
    r0 = max(int(ys.min()) - margin, 0)
    r1 = min(int(ys.max()) + margin + 1, binary_full.shape[0])
    c0 = max(int(xs.min()) - margin, 0)
    c1 = min(int(xs.max()) + margin + 1, binary_full.shape[1])
    crop = binary_full[r0:r1, c0:c1]
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_dir / f"{pos}_{ch}_f{frame}.npz",
                 binary=crop, frame=frame, px=px)
    return crop, frame, px


# =========================================================================
# Synthetic self-test (no data required)
# =========================================================================

def _make_bent_rod(length=120, width=26, bend=18, canvas=200) -> np.ndarray:
    """A gently bent capsule, then rotated, to exercise the full geometry path."""
    yy, xx = np.indices((canvas, canvas)).astype(float)
    cx = canvas / 2.0
    xs = np.linspace(-length / 2, length / 2, 400)
    ys = bend * np.sin(np.pi * (xs + length / 2) / length)   # single arch
    mask = np.zeros((canvas, canvas), dtype=bool)
    r = width / 2.0
    for sx, sy in zip(xs, ys):
        mask |= (xx - (cx + sx)) ** 2 + (yy - (cx + sy)) ** 2 <= r ** 2
    return transform.rotate(mask.astype(np.uint8), 35.0, resize=True,
                            order=0, preserve_range=True).astype(bool)


def _self_test():
    print("=== mask_volume_schematic self-test ===")
    rng = np.random.default_rng(0)
    for ang in [0.0, 23.0, 61.0, -40.0]:
        rod = _make_bent_rod()
        rod = transform.rotate(rod.astype(np.uint8), ang, resize=True,
                               order=0, preserve_range=True).astype(bool)
        # verify the rotation replicate is byte-identical to skimage's rotate
        deg = 17.3
        mine, _ = rotate_with_map(rod, deg)
        ref = transform.rotate(rod.astype(np.uint8), deg, resize=True, order=0,
                               preserve_range=True).astype(bool)
        ok = mine.shape == ref.shape and np.array_equal(mine, ref)
        geo = medial_axis_geometry(rod, pixel_size_um=1.0)
        assert geo is not None
        # round-trip sanity: medial axis stays inside the mask bbox
        ys, xs = np.where(rod)
        inside = (geo.medial_xy[:, 0].min() >= xs.min() - 2 and
                  geo.medial_xy[:, 0].max() <= xs.max() + 2 and
                  geo.medial_xy[:, 1].min() >= ys.min() - 2 and
                  geo.medial_xy[:, 1].max() <= ys.max() + 2)
        print(f"  rod@{ang:+.0f}deg: rotate-replicate={'OK' if ok else 'FAIL'}  "
              f"medial-in-bbox={'OK' if inside else 'FAIL'}  "
              f"V={geo.volume_px3:9.0f}px^3  L={geo.long_axis_px:6.1f}px  "
              f"n_slices={len(geo.medial_xy)}")
        assert ok and inside
    print("=== self-test passed ===")


# =========================================================================
# CLI
# =========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos", default="Pos27")
    ap.add_argument("--ch", default="ch06")
    ap.add_argument("--frame", type=int, default=None,
                    help="rank=1 frame to draw (default: a typical one)")
    ap.add_argument("--method", choices=["efd", "projection"], default="efd",
                    help="efd (adopted): EFD-smoothed contour + one midpoint "
                         "update, width/centerline from contour intersection. "
                         "projection: legacy w_perp = h_vert*cos(theta).")
    ap.add_argument("--efd-k", type=int, default=6,
                    help="EFD harmonics for contour smoothing (use >=10 for a "
                         "dividing cell with a neck)")
    ap.add_argument("--slice-step", type=int, default=1,
                    help="draw every Nth short-axis chord (display only; "
                         "the volume always integrates every section)")
    ap.add_argument("--show-caps", action="store_true",
                    help="also draw the hemispherical-cap columns (their "
                         "perpendicular disk can slightly exceed the contour); "
                         "by default caps are shown as outline only")
    ap.add_argument("--synthetic", action="store_true",
                    help="draw a synthetic bent rod (no data needed)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default=None, help="explicit output PNG path "
                    "(otherwise uses figure_logger save_figure)")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    if args.synthetic:
        binary = _make_bent_rod()
        px = 1.0
        frame = -1
        title = "synthetic bent rod — medial-axis volume"
    else:
        binary, frame, px = load_mother_crop(args.pos, args.ch, args.frame)
        title = f"{args.pos}_{args.ch} mother  frame {frame}"

    if args.method == "efd":
        geo = efd_section_geometry(binary, pixel_size_um=px, efd_k=args.efd_k)
        method_note = (f"EFD K={args.efd_k} contour + 1 midpoint update "
                       f"(width & centerline from contour intersection)")
    else:
        geo = medial_axis_geometry(binary, pixel_size_um=px)
        method_note = "w_perp = h_vert·cos θ  (solid of revolution)"
    if geo is None:
        print("could not build geometry (empty/degenerate mask)", file=sys.stderr)
        sys.exit(1)
    fig = plot_schematic(geo, slice_step=args.slice_step, title=title,
                         show_caps=args.show_caps, method_note=method_note)

    # full plot data for later restyling from npz (every panel array)
    data = {
        "medial_xy": geo.medial_xy,
        "slice_p0_xy": geo.slice_p0_xy,
        "slice_p1_xy": geo.slice_p1_xy,
        "w_perp_px": geo.w_perp_px,
        "h_vert_px": geo.h_vert_px,
        "theta_rad": geo.theta_rad,
        "arc_step_px": geo.arc_step_px,
        "cap_len": np.array([geo.cap_len]),
        "volume_px3": np.array([geo.volume_px3]),
        "pixel_size_um": np.array([geo.pixel_size_um]),
    }
    for i, c in enumerate(geo.contour_xy):
        data[f"contour_xy_{i}"] = c

    if args.out:
        fig.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"wrote {args.out}")
    else:
        from figure_logger import save_figure
        out = save_figure(
            fig,
            params={
                "pos": args.pos, "ch": args.ch, "frame": frame,
                "method": args.method, "efd_k": args.efd_k,
                "slice_step_display": args.slice_step,
                "pixel_size_um": px,
                "volume_um3": geo.volume_um3,
                "long_axis_um": geo.long_axis_px * px,
                "smoothing_window_frac": SMOOTH_WINDOW_FRAC,
            },
            description=(
                "Mask-direct cell-volume schematic (adopted method, --method efd): "
                "the raw mask contour is EFD-smoothed (K=6) to de-stair the square "
                "pixels without biasing size; a single midpoint update takes the "
                "short axis and centerline from the perpendicular line's contour "
                "intersections; volume is the solid-of-revolution disk integral. "
                "Cyan = EFD contour, red = updated centerline, gray = cross-sections. "
                "Iterating further diverges, so exactly one update is used."),
            data=data,
        )
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
