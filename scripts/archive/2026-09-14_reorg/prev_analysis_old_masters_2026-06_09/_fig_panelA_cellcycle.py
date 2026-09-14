"""Panel-A style figures for the 260517 Pos1/z000 lineage.

Two eLife-Odermatt-2021 style panels, both with cells aligned to a common
orientation (long axis vertical) via a PCA rotation of the segmentation mask:

  fig1A-style  (geometry overlay):
      left  = cell with overlay   contour = cyan, long axis (medial) = red,
              short-axis cross-sections = gray
      right = the same cell, raw phase image in the chosen colormap
      (the per-z-position strip from the original fig1A is intentionally omitted)

  fig2A-style  (cell-cycle strip):
      one mother cell sampled across a single cell cycle, each frame rotated to
      a common vertical orientation and tiled left->right in the chosen colormap

Colormap defaults to ``inferno`` (matches the eLife density LUT); pass
``--cmap viridis`` for the viridis variant.

Geometry is computed the same way as the mask-direct medial-axis volume tool
(recompute_axes_from_masks / mask_morphology): per-row mask centroid gives the
medial axis, its local slope gives the tangent, and the short axis is the
row width projected perpendicular to that tangent.

Usage:
    python scripts/_fig_panelA_cellcycle.py
    python scripts/_fig_panelA_cellcycle.py --ch ch00 --cmap viridis
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage import measure

from figure_logger import save_figure

# ---- styling -------------------------------------------------------------
CYAN = "#56B4E9"   # contour
RED = "#D55E00"    # long axis (medial)
GRAY = "#BBBBBB"   # short-axis cross sections
PX_DEFAULT = 0.34567514677103717  # um/px (260517 run param)

DATASET_ROOT = Path(r"f:/260517/2per_0055per_0per_2per_crop_sub")
Z = "z000"


def data_root(pos: str) -> Path:
    return DATASET_ROOT / pos / "output_phase" / "channels" / "crop_sub_rawraw" / Z
_IMG_RE = re.compile(r"img_0*(\d+)")


# ---- IO ------------------------------------------------------------------
def phase_path(ch_dir: Path, frame: int) -> Path:
    return ch_dir / f"img_{frame:09d}_ph_000_phase.tif"


def mask_path(ch_dir: Path, frame: int) -> Path:
    return ch_dir / "inference_out" / f"img_{frame:09d}_ph_000_phase_masks.tif"


def load_pixel_size(ch_dir: Path) -> float:
    p = ch_dir / "inference_out" / "lineage_out" / "lineage_run_params.json"
    if p.exists():
        try:
            return float(json.loads(p.read_text()).get("pixel_size_um", PX_DEFAULT))
        except Exception:
            pass
    return PX_DEFAULT


def label_at(mask: np.ndarray, cy: float, cx: float) -> int:
    """Label under the mother centroid, with a small neighbourhood fallback."""
    h, w = mask.shape
    yi, xi = int(round(cy)), int(round(cx))
    if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
        return int(mask[yi, xi])
    y0, y1 = max(yi - 2, 0), min(yi + 3, h)
    x0, x1 = max(xi - 2, 0), min(xi + 3, w)
    nz = mask[y0:y1, x0:x1]
    nz = nz[nz != 0]
    if nz.size:
        v, c = np.unique(nz, return_counts=True)
        return int(v[np.argmax(c)])
    return 0


def crop_mother(ch_dir: Path, frame: int, cy: float, cx: float, margin: int = 6):
    """Return (phase_crop, binary_crop) for the mother in this frame, or None."""
    pp, mp = phase_path(ch_dir, frame), mask_path(ch_dir, frame)
    if not pp.exists() or not mp.exists():
        return None
    phase = np.squeeze(tifffile.imread(pp)).astype(np.float32)
    mask = np.squeeze(tifffile.imread(mp))
    lbl = label_at(mask, cy, cx)
    if lbl == 0:
        return None
    binary = mask == lbl
    ys, xs = np.where(binary)
    if ys.size == 0:
        return None
    r0, r1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, binary.shape[0])
    c0, c1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, binary.shape[1])
    return phase[r0:r1, c0:c1], binary[r0:r1, c0:c1]


# ---- orientation alignment ----------------------------------------------
def pca_angle_deg(binary: np.ndarray) -> float:
    """Angle (deg) to rotate so the cell's principal axis becomes vertical."""
    ys, xs = np.where(binary)
    ys = ys - ys.mean()
    xs = xs - xs.mean()
    cov = np.cov(np.vstack([xs, ys]))
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, np.argmax(evals)]   # principal direction
    ang = np.degrees(np.arctan2(vy, vx))  # angle of long axis vs +x
    return ang - 90.0                     # rotate this much to make it vertical


def align_vertical(phase: np.ndarray, binary: np.ndarray, fill: float,
                   margin: int = 4):
    """Rotate phase+mask so the long axis is vertical; re-crop to the cell.

    `margin` is the black border (px) kept around the cell; smaller values pack
    neighbouring tiles tighter in a strip."""
    ang = pca_angle_deg(binary)
    ph = ndimage.rotate(phase, ang, reshape=True, order=1, mode="constant", cval=fill)
    bw = ndimage.rotate(binary.astype(np.float32), ang, reshape=True, order=1,
                        mode="constant", cval=0.0) > 0.5
    ys, xs = np.where(bw)
    if ys.size == 0:
        return ph, bw
    m = margin
    r0, r1 = max(ys.min() - m, 0), min(ys.max() + m + 1, bw.shape[0])
    c0, c1 = max(xs.min() - m, 0), min(xs.max() + m + 1, bw.shape[1])
    return ph[r0:r1, c0:c1], bw[r0:r1, c0:c1]


# ---- geometry (vertical cell: long axis ~ y) -----------------------------
def _smooth(y: np.ndarray, win: int = 5) -> np.ndarray:
    if len(y) < win or win < 2:
        return y.astype(float)
    k = np.ones(win) / win
    pad = win // 2
    return np.convolve(np.pad(y, pad, mode="edge"), k, mode="valid")[: len(y)]


def cell_geometry(binary: np.ndarray, n_cross: int = 9):
    """Medial axis, short-axis cross sections, contour for a vertical cell.

    Returns dict with:
      medial : (N,2) array of (x, y) along the long axis
      cross  : list of ((x0,y0),(x1,y1)) short-axis segments (perp to tangent)
      contour: (M,2) array of (x, y) outline
    """
    ys = np.where(binary.any(axis=1))[0]
    if ys.size < 3:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    rows = np.arange(y0, y1 + 1)
    xc = np.full(rows.shape, np.nan)
    half = np.zeros(rows.shape)
    for i, y in enumerate(rows):
        cols = np.where(binary[y])[0]
        if cols.size:
            xc[i] = cols.mean()
            half[i] = (cols.max() - cols.min() + 1) / 2.0
    ok = ~np.isnan(xc)
    rows, xc, half = rows[ok], xc[ok], half[ok]
    if rows.size < 3:
        return None
    xc_s = _smooth(xc)
    medial = np.column_stack([xc_s, rows.astype(float)])

    # local tangent (dx/dy) -> perpendicular short-axis direction
    dx = np.gradient(xc_s, rows.astype(float))
    theta = np.arctan2(dx, 1.0)            # tangent angle vs vertical
    cos_t = np.cos(theta)

    # cross sections over the central body (drop the two end caps)
    cap = max(int(0.12 * rows.size), 1)
    body = np.arange(cap, rows.size - cap)
    if body.size == 0:
        body = np.arange(rows.size)
    sel = body[np.linspace(0, body.size - 1, min(n_cross, body.size)).astype(int)]
    cross = []
    for i in sel:
        w_perp = (2 * half[i]) * cos_t[i]       # true perpendicular width
        # perpendicular unit vector to tangent (dx,1): (1,-dx)/|.|
        px_, py_ = 1.0, -dx[i]
        nrm = np.hypot(px_, py_)
        ux, uy = px_ / nrm, py_ / nrm
        cxx, cyy = xc_s[i], rows[i]
        cross.append((
            (cxx - ux * w_perp / 2, cyy - uy * w_perp / 2),
            (cxx + ux * w_perp / 2, cyy + uy * w_perp / 2),
        ))

    cont = measure.find_contours(binary.astype(float), 0.5)
    contour = max(cont, key=len)[:, ::-1] if cont else np.empty((0, 2))  # (x,y)
    return {"medial": medial, "cross": cross, "contour": contour}


# ---- cell-cycle detection ------------------------------------------------
def _division_indices(L: np.ndarray, ratio: float = 0.7,
                      hold_ratio: float = 0.75, hold_k: int = 4) -> list[int]:
    """Indices i where a true division happens between i and i+1.

    A division roughly halves the length AND the new (daughter) cell stays
    short for the next few frames. Single-frame segmentation glitches dip for
    one frame and recover, so the post-drop median rejects them."""
    div = []
    n = len(L)
    for i in range(n - 1):
        if L[i + 1] < ratio * L[i]:
            k1 = min(i + 1 + hold_k, n)
            if np.median(L[i + 1:k1]) < hold_ratio * L[i]:
                div.append(i)
    return div


def _drop_glitch_frames(seg: pd.DataFrame) -> pd.DataFrame:
    """Drop single-frame length dips (seg-failure frames) inside a cycle."""
    L = seg["long_axis_um"].to_numpy()
    keep = np.ones(len(L), bool)
    for i in range(1, len(L) - 1):
        if L[i] < 0.7 * min(L[i - 1], L[i + 1]):
            keep[i] = False
    return seg.iloc[keep]


def rank_cycles(m: pd.DataFrame, min_len: int = 16, max_len: int = 70,
                growth_max_frame: int = 2000) -> list[pd.DataFrame]:
    """All single cell cycles in the growth phase, ranked best-first.

    Cycles are bounded by glitch-robust division events; each is cleaned of
    residual single-frame dips. Ranked by elongation x monotonicity so the
    cleanest, most textbook 'short -> long -> divide' cycles come first."""
    m = m.sort_values("frame").reset_index(drop=True)
    L = m["long_axis_um"].to_numpy()
    div = _division_indices(L)
    bounds = [0, *[d + 1 for d in div], len(m)]
    cands = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = _drop_glitch_frames(m.iloc[a:b])
        if not (min_len <= len(seg) <= max_len):
            continue
        if seg["frame"].min() <= 30 or seg["frame"].max() > growth_max_frame:
            continue
        Ls = seg["long_axis_um"].to_numpy()
        grow = Ls[-1] - Ls[0]
        mono = float((np.diff(Ls) > 0).mean())
        if grow <= 0:
            continue
        cands.append((grow * mono, grow, mono, seg))
    cands.sort(key=lambda t: t[0], reverse=True)
    return [c[3] for c in cands]


def pick_one_cycle(m: pd.DataFrame, rank: int = 1):
    """The rank-th best single cell cycle (1-based)."""
    cands = rank_cycles(m)
    if not cands:
        return m.sort_values("frame").iloc[:16]
    return cands[min(rank, len(cands)) - 1]


def load_bad_frames(ch_dir: Path, pos: str) -> set[int]:
    """Frame numbers flagged bad by the drift QC (bad_frames_used.json)."""
    p = ch_dir / "inference_out" / "lineage_out" / "bad_frames_used.json"
    if not p.exists():
        return set()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        bt = (d.get(pos) or {}).get("bad_timepoints") or {}
        return {int(k) for k in bt.keys()}
    except Exception:
        return set()


def enumerate_all_cycles(m: pd.DataFrame, bad_frames: set[int],
                         max_frame: int = 2018, min_len: int = 10,
                         max_len: int = 70, min_kept: int = 8):
    """Every division-terminated cycle up to ``max_frame``, with bad frames
    dropped from the tiles (not the whole cycle).

    Looser than ``enumerate_clean_cycles``: keeps cycles even if they had a
    missing frame, as long as the ORIGINAL segment is a plausible single cycle
    (length in [min_len, max_len] — this rejects merged segments from missed
    divisions and 1-2 frame spurious-division fragments). Within a kept cycle,
    rows flagged ``is_outlier`` or listed in ``bad_frames`` are removed.

    Returns (kept, skipped) where kept is a list of cleaned cycle DataFrames and
    skipped is a dict of exclusion counts for honest reporting."""
    m = m.sort_values("frame").reset_index(drop=True)
    L = m["long_axis_um"].to_numpy()
    div = _division_indices(L)
    divset = set(div)
    bounds = [0, *[d + 1 for d in div], len(m)]
    kept = []
    skipped = {"too_long_merged": 0, "too_short_fragment": 0,
               "few_after_clean": 0, "not_div_terminated": 0}
    for a, b in zip(bounds[:-1], bounds[1:]):
        raw = m.iloc[a:b]
        f0, f1 = int(raw["frame"].min()), int(raw["frame"].max())
        if f1 > max_frame:
            continue
        if (b - 1) not in divset:
            skipped["not_div_terminated"] += 1
            continue
        if len(raw) > max_len:
            skipped["too_long_merged"] += 1
            continue
        if len(raw) < min_len:
            skipped["too_short_fragment"] += 1
            continue
        bad = raw["is_outlier"].to_numpy(dtype=bool) if "is_outlier" in raw else \
            np.zeros(len(raw), bool)
        bad |= raw["frame"].isin(bad_frames).to_numpy()
        seg = raw[~bad]
        if len(seg) < min_kept:
            skipped["few_after_clean"] += 1
            continue
        kept.append(seg)
    return kept, skipped


def enumerate_clean_cycles(m: pd.DataFrame, min_len: int = 18, max_len: int = 75,
                           sm_lo: float = 3.0, sm_hi: float = 4.8):
    """All intact cell cycles of one mother track, in time order.

    'Intact' = the segment is bounded by a real division at its end, has NO
    missing frames (consecutive frame numbers), loses nothing to glitch removal,
    and passes loose sanity (mean short axis in [sm_lo, sm_hi], grows > 3 um).
    Returns list of DataFrames (each one cycle)."""
    m = m.sort_values("frame").reset_index(drop=True)
    L = m["long_axis_um"].to_numpy()
    div = _division_indices(L)
    divset = set(div)
    bounds = [0, *[d + 1 for d in div], len(m)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if (b - 1) not in divset:          # cycle must END at a real division
            continue
        raw = m.iloc[a:b]
        f0, f1 = int(raw["frame"].min()), int(raw["frame"].max())
        if not (min_len <= len(raw) <= max_len):
            continue
        if (f1 - f0 + 1) != len(raw):       # no missing frames
            continue
        seg = _drop_glitch_frames(raw)
        if len(seg) != len(raw):            # no glitch-dropped frames
            continue
        Ls = seg["long_axis_um"].to_numpy()
        sm = float(seg["short_axis_um"].mean())
        if not (sm_lo <= sm <= sm_hi) or (Ls[-1] - Ls[0]) < 3:
            continue
        out.append(seg)
    return out


# ---- figure 1: geometry overlay -----------------------------------------
def make_overlay_fig(ch_dir: Path, row, px: float, cmap: str):
    crop = crop_mother(ch_dir, int(row["frame"]), row["centroid_y_px"],
                       row["centroid_x_px"])
    if crop is None:
        raise RuntimeError("could not crop mother for overlay frame")
    phase, binary = crop
    fill = float(np.percentile(phase, 1))
    phase_v, binary_v = align_vertical(phase, binary, fill)
    geo = cell_geometry(binary_v)
    if geo is None:
        raise RuntimeError("geometry failed")

    vmin, vmax = np.percentile(phase_v[binary_v], [1, 99])
    H, W = phase_v.shape
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(3.4, 3.4 * H / max(W, 1) + 0.2), constrained_layout=True
    )

    # left: overlay on white
    axL.set_facecolor("white")
    if geo["contour"].size:
        axL.plot(geo["contour"][:, 0], geo["contour"][:, 1], color=CYAN, lw=1.6)
    for (p0, p1) in geo["cross"]:
        axL.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GRAY, lw=1.0, zorder=2)
    axL.plot(geo["medial"][:, 0], geo["medial"][:, 1], color=RED, lw=1.6, zorder=3)
    axL.set_xlim(0, W)
    axL.set_ylim(H, 0)
    axL.set_aspect("equal")
    axL.axis("off")
    axL.set_title("geometry", fontsize=8)

    # right: raw phase, chosen colormap
    axR.imshow(phase_v, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axR.set_aspect("equal")
    axR.axis("off")
    axR.set_title("phase", fontsize=8)

    # scale bar on the phase panel (5 um)
    sb = 5.0 / px
    axR.plot([W * 0.12, W * 0.12 + sb], [H * 0.92, H * 0.92], color="white", lw=2.5)
    axR.text(W * 0.12 + sb / 2, H * 0.88, "5 µm", color="white", fontsize=7,
             ha="center", va="bottom")

    return fig, {
        "contour_xy": geo["contour"], "medial_xy": geo["medial"],
        "cross_segments": np.array([[*p0, *p1] for p0, p1 in geo["cross"]]),
        "phase_crop": phase_v, "vmin": vmin, "vmax": vmax,
    }


# ---- figure 2: cell-cycle strip -----------------------------------------
def build_strip_canvas(ch_dir: Path, seg: pd.DataFrame, n_tiles: int | None = 13,
                       gap: int = 1, crop_margin: int = 4):
    """Build a single horizontal strip canvas for one cell cycle.

    Returns (canvas, vmin, vmax, frames, times, Ws) with each tile rotated to a
    common vertical orientation and centred on a shared baseline; background is
    filled with the per-cell 1st-percentile so it reads black in inferno.

    If ``n_tiles`` is None, every frame in ``seg`` is shown (no sub-sampling).
    Otherwise the first (just after division) and last (just before division)
    frames are always kept, with the rest sampled evenly in between."""
    if n_tiles is None or n_tiles >= len(seg):
        rows = seg
    else:
        idx = np.unique(np.r_[0, np.linspace(0, len(seg) - 1, n_tiles).round(),
                              len(seg) - 1]).astype(int)
        rows = seg.iloc[idx]

    tiles, masks, times, frames, fill_probe = [], [], [], [], []
    for _, r in rows.iterrows():
        crop = crop_mother(ch_dir, int(r["frame"]), r["centroid_y_px"],
                           r["centroid_x_px"])
        if crop is None:
            continue
        phase, binary = crop
        fill_probe.append(np.percentile(phase, 1))
    fill = float(np.median(fill_probe)) if fill_probe else 0.0

    for _, r in rows.iterrows():
        crop = crop_mother(ch_dir, int(r["frame"]), r["centroid_y_px"],
                           r["centroid_x_px"])
        if crop is None:
            continue
        phase, binary = crop
        ph_v, bw_v = align_vertical(phase, binary, fill, margin=crop_margin)
        tiles.append(ph_v)
        masks.append(bw_v)
        times.append(float(r["time_h"]))
        frames.append(int(r["frame"]))
    if not tiles:
        raise RuntimeError("no tiles for strip")

    Hs = max(t.shape[0] for t in tiles)
    # Pack tiles at their own widths (not a fixed max-width slot) so narrow
    # cells don't carry side padding; only `gap` px separate neighbours.
    total_w = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    canvas = np.full((Hs, total_w), fill, np.float32)
    interior = []
    cx = 0
    for j, t in enumerate(tiles):
        h, w = t.shape
        r0 = (Hs - h) // 2
        canvas[r0:r0 + h, cx:cx + w] = t
        interior.append(t[masks[j]])
        cx += w + gap
    allpix = np.concatenate(interior)
    vmin, vmax = np.percentile(allpix, [1, 99.5])
    Ws = max(t.shape[1] for t in tiles)
    return canvas, float(vmin), float(vmax), np.array(frames), np.array(times), Ws


def make_contact_sheet(specs: list[dict], px: float, cmap: str,
                       n_tiles: int | None = 13, gap: int = 1,
                       vmin: float | None = None, vmax: float | None = None):
    """One figure stacking many cell-cycle strips, one per row.

    specs: list of {pos, ch, seg, label}. If vmin/vmax are given, all rows share
    that fixed color scale (absolute phase is then comparable across cells and a
    single shared colorbar is drawn); otherwise each row is scaled to its own
    percentiles. Strips are left-aligned on a shared width."""
    canvases = []
    for s in specs:
        ch_dir = data_root(s["pos"]) / s["ch"]
        try:
            cv, rvmin, rvmax, frames, times, Ws = build_strip_canvas(
                ch_dir, s["seg"], s.get("n_tiles", n_tiles), gap=gap)
        except Exception as e:
            print(f"  [skip] {s['label']}: {e}")
            continue
        canvases.append({"cv": cv, "vmin": rvmin, "vmax": rvmax,
                         "frames": frames, "times": times,
                         "label": s["label"], "px": px})
    if not canvases:
        raise RuntimeError("no rows for contact sheet")

    fixed = vmin is not None and vmax is not None
    maxW = max(c["cv"].shape[1] for c in canvases)
    Hrow = max(c["cv"].shape[0] for c in canvases)
    n = len(canvases)
    row_gap = 4

    # Compose ALL rows into ONE array so every pixel has the SAME size ->
    # uniform magnification across rows (separate per-row axes would each
    # fit-to-box and zoom differently). Per-row scaled mode is normalised to
    # [0,1] before compositing; fixed mode keeps raw phase values.
    lo, hi = (vmin, vmax) if fixed else (0.0, 1.0)
    total_h = n * Hrow + (n - 1) * row_gap
    composite = np.full((total_h, maxW), lo, np.float32)
    row_centers = []
    for i, c in enumerate(canvases):
        cv = c["cv"]
        if not fixed:                      # normalise this row to [0,1]
            rng = max(c["vmax"] - c["vmin"], 1e-6)
            cv = np.clip((cv - c["vmin"]) / rng, 0, 1)
        h, w = cv.shape
        y0 = i * (Hrow + row_gap) + (Hrow - h) // 2
        composite[y0:y0 + h, 0:w] = cv     # left-aligned
        row_centers.append(i * (Hrow + row_gap) + Hrow / 2)

    # figure sized to the composite aspect (square pixels at display)
    label_w = max(int(0.18 * maxW), 70)
    fig_w = 11.0
    fig_h = fig_w * total_h / (maxW + label_w)
    fig, ax = plt.subplots(figsize=(fig_w, min(fig_h, 200)))
    im = ax.imshow(composite, cmap=cmap, vmin=lo, vmax=hi,
                   interpolation="nearest", extent=[0, maxW, total_h, 0])
    ax.set_xlim(-label_w, maxW)
    ax.set_ylim(total_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for c, yc in zip(canvases, row_centers):
        ax.text(-4, yc, c["label"], ha="right", va="center", fontsize=6.5)

    # scale bar on the last row
    sb = 5.0 / canvases[-1]["px"]
    yb = (n - 1) * (Hrow + row_gap) + Hrow - 3
    ax.plot([3, 3 + sb], [yb, yb], color="white", lw=2.5)
    ax.text(3 + sb / 2, yb - 3, "5 µm", color="white", fontsize=6,
            ha="center", va="bottom")

    scale_txt = (f"phase fixed {vmin:g}-{vmax:g} rad" if fixed
                 else "per-row scaled")
    ax.set_title("Single cell cycles (each row: one mother, birth -> "
                 f"division; inferno, {scale_txt})", fontsize=9)
    if fixed:
        cb = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
        cb.set_label("phase (rad)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
    return fig, canvases


def make_strip_fig(ch_dir: Path, seg: pd.DataFrame, px: float, cmap: str,
                   n_tiles: int = 13):
    idx = np.linspace(0, len(seg) - 1, min(n_tiles, len(seg))).astype(int)
    rows = seg.iloc[idx]

    tiles, masks = [], []
    fill_probe = []
    for _, r in rows.iterrows():
        crop = crop_mother(ch_dir, int(r["frame"]), r["centroid_y_px"],
                           r["centroid_x_px"])
        if crop is None:
            continue
        phase, binary = crop
        fill_probe.append(np.percentile(phase, 1))
    fill = float(np.median(fill_probe)) if fill_probe else 0.0

    times, frames = [], []
    for _, r in rows.iterrows():
        crop = crop_mother(ch_dir, int(r["frame"]), r["centroid_y_px"],
                           r["centroid_x_px"])
        if crop is None:
            continue
        phase, binary = crop
        ph_v, bw_v = align_vertical(phase, binary, fill)
        tiles.append(ph_v)
        masks.append(bw_v)
        times.append(float(r["time_h"]))
        frames.append(int(r["frame"]))
    if not tiles:
        raise RuntimeError("no tiles for strip")

    Hs = max(t.shape[0] for t in tiles)
    Ws = max(t.shape[1] for t in tiles)
    gap = 3
    canvas = np.full((Hs, len(tiles) * (Ws + gap) - gap), fill, np.float32)
    interior = []
    for j, t in enumerate(tiles):
        h, w = t.shape
        r0 = (Hs - h) // 2
        c0 = j * (Ws + gap) + (Ws - w) // 2
        canvas[r0:r0 + h, c0:c0 + w] = t
        interior.append(t[masks[j]])
    allpix = np.concatenate(interior)
    vmin, vmax = np.percentile(allpix, [1, 99.5])

    # size the figure to the canvas aspect so there is no vertical whitespace
    cH, cW = canvas.shape
    panel_w = 6.5
    panel_h = panel_w * cH / cW
    fig, ax = plt.subplots(figsize=(panel_w + 0.9, panel_h + 0.7),
                           constrained_layout=True)
    im = ax.imshow(canvas, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")
    ax.set_aspect("equal")

    # scale bar (5 um) bottom-left
    sb = 5.0 / px
    ax.plot([4, 4 + sb], [Hs - 4, Hs - 4], color="white", lw=2.5)
    ax.text(4 + sb / 2, Hs - 7, "5 µm", color="white", fontsize=7,
            ha="center", va="bottom")

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("phase (rad)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(
        f"mother cell cycle  (frame {frames[0]}->{frames[-1]}, "
        f"{times[0]:.1f}->{times[-1]:.1f} h)", fontsize=8)
    return fig, {"canvas": canvas, "vmin": vmin, "vmax": vmax,
                 "times_h": np.array(times), "frames": np.array(frames)}


# ---- main ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="Pos1")
    ap.add_argument("--ch", default="ch00")
    ap.add_argument("--cmap", default="inferno",
                    help="inferno (eLife density LUT) or viridis")
    ap.add_argument("--ranks", default="1-4",
                    help="which ranked cell cycles to render as strips, "
                         "e.g. '1-4' or '1,3,5'")
    ap.add_argument("--frames", default=None,
                    help="render one explicit cycle 'f0-f1' instead of ranking")
    ap.add_argument("--strip-only", action="store_true",
                    help="skip the fig1A geometry overlay")
    ap.add_argument("--contact", default=None,
                    help="render ONE montage of many cycles; comma-separated "
                         "'Pos:ch:f0-f1' specs, e.g. "
                         "'Pos6:ch03:522-570,Pos6:ch05:668-711'")
    ap.add_argument("--vmin", type=float, default=None,
                    help="fixed lower phase for a shared color scale")
    ap.add_argument("--vmax", type=float, default=None,
                    help="fixed upper phase for a shared color scale")
    ap.add_argument("--gap", type=int, default=1,
                    help="pixels between tiles in a strip")
    ap.add_argument("--channel-all", default=None,
                    help="render ONE sheet of ALL intact cycles of a channel, "
                         "e.g. 'Pos18:ch02'")
    ap.add_argument("--rows", default=None,
                    help="for --channel-all: keep only these 1-based cycle rows, "
                         "e.g. '1-9'. The death row (unless --no-death) is still "
                         "appended last and uses the TRUE last cycle.")
    ap.add_argument("--no-death", action="store_true",
                    help="for --channel-all: do not append the (death) row")
    ap.add_argument("--last-row", default=None,
                    help="for --channel-all: make this explicit frame range the "
                         "final row (e.g. '892-910'); cycles after it and the "
                         "auto (death) row are dropped. Overrides --rows/--no-death.")
    ap.add_argument("--tail-from", default=None,
                    help="for --channel-all: after the cell stops dividing, show "
                         "ONE long final row from this frame, sampled at the same "
                         "per-tile frame interval as a cycle (so a long arrest "
                         "reads as a long row). Drops later fragment cycles + auto death.")
    ap.add_argument("--tail-to", default=None,
                    help="end frame for --tail-from (default: last detected cycle's "
                         "end frame)")
    ap.add_argument("--max-cycle-len", type=int, default=70,
                    help="for --channel-all: max frames for a div-bounded segment "
                         "to count as one cycle (above this = assumed missed-"
                         "division merge). Raise to keep a genuinely long "
                         "(filamentous) cycle.")
    ap.add_argument("--single", default=None,
                    help="render ONE high-res cell-cycle strip (no colorbar) "
                         "plus a separate thin horizontal colorbar file, "
                         "e.g. 'Pos18:ch02:633-670'")
    ap.add_argument("--crop-margin", type=int, default=2,
                    help="black border px kept around each cell (smaller=tighter)")
    ap.add_argument("--n-tiles", type=int, default=13)
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    if args.single:
        import matplotlib as mpl
        pos, ch, fr = args.single.split(":")
        f0, f1 = (int(x) for x in fr.split("-"))
        ch_dir = data_root(pos) / ch
        px = load_pixel_size(ch_dir)
        dd = pd.read_csv(ch_dir / "inference_out" / "lineage_out" /
                         "lineage_data3D.csv")
        mm = dd[dd["rank"] == 1].sort_values("frame").reset_index(drop=True)
        seg = _drop_glitch_frames(
            mm[(mm["frame"] >= f0) & (mm["frame"] <= f1)].reset_index(drop=True))
        cv, rvmin, rvmax, frames, times, Ws = build_strip_canvas(
            ch_dir, seg, n_tiles=args.n_tiles, gap=args.gap,
            crop_margin=args.crop_margin)
        vmin = args.vmin if args.vmin is not None else rvmin
        vmax = args.vmax if args.vmax is not None else rvmax

        # --- strip (no colorbar, scale bar bottom-left), sized to canvas ---
        cH, cW = cv.shape
        panel_w = 7.0
        figS, axS = plt.subplots(figsize=(panel_w, panel_w * cH / cW))
        axS.imshow(cv, cmap=args.cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
        axS.set_axis_off()
        sb = 5.0 / px
        axS.plot([3, 3 + sb], [cH - 3, cH - 3], color="white", lw=3)
        axS.text(3 + sb / 2, cH - 5, "5 µm", color="white", fontsize=8,
                 ha="center", va="bottom")
        figS.subplots_adjust(left=0, right=1, top=1, bottom=0)
        save_figure(
            figS,
            params={"dataset": "260517", "pos": pos, "z": Z, "ch": ch,
                    "frame_min": f0, "frame_max": f1, "cmap": args.cmap,
                    "vmin": vmin, "vmax": vmax, "gap_px": args.gap,
                    "crop_margin_px": args.crop_margin, "pixel_size_um": px,
                    "n_tiles": int(len(frames)), "dpi": args.dpi,
                    "style": "eLife fig2A single strip (high-res)"},
            description=(
                f"High-res single cell-cycle strip {pos} {ch} {f0}-{f1} "
                f"({times[0]:.1f}-{times[-1]:.1f} h), tiles rotated vertical, "
                f"tight packing (gap {args.gap}, margin {args.crop_margin}), "
                f"5 um scale bar, fixed {vmin}-{vmax} rad, {args.cmap}; "
                f"colorbar saved separately."),
            data={"frames": frames, "times_h": times,
                  "vmin": float(vmin), "vmax": float(vmax)},
            dpi=args.dpi, fmt="png",
        )
        plt.close(figS)

        # --- separate thin horizontal colorbar (eLife-style) ---
        figC, axC = plt.subplots(figsize=(2.2, 0.34))
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb = mpl.colorbar.ColorbarBase(
            axC, cmap=plt.get_cmap(args.cmap), norm=norm,
            orientation="horizontal")
        cb.set_label("phase (rad)", fontsize=7, labelpad=2)
        cb.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cb.ax.tick_params(labelsize=6, length=2, width=0.4)
        cb.outline.set_linewidth(0.4)
        save_figure(
            figC,
            params={"dataset": "260517", "cmap": args.cmap,
                    "vmin": vmin, "vmax": vmax, "orientation": "horizontal",
                    "style": "eLife-style thin colorbar (separate file)"},
            description=(f"Separate thin horizontal colorbar for the {pos} {ch} "
                         f"{f0}-{f1} strip, {args.cmap} {vmin}-{vmax} rad."),
            dpi=args.dpi, fmt="png",
        )
        plt.close(figC)
        print(f"done single {pos} {ch} {f0}-{f1}")
        return

    if args.channel_all:
        pos, ch = args.channel_all.split(":")
        ch_dir = data_root(pos) / ch
        px = load_pixel_size(ch_dir)
        dd = pd.read_csv(ch_dir / "inference_out" / "lineage_out" /
                         "lineage_data3D.csv")
        mm = dd[dd["rank"] == 1].sort_values("frame").reset_index(drop=True)
        bad = load_bad_frames(ch_dir, pos)
        # keep all real cycles; short ones just sample fewer tiles (n_tiles is a
        # cap, not a requirement), so sampling is roughly-but-not-exactly even
        cycles, skipped = enumerate_all_cycles(mm, bad, min_kept=6,
                                               max_len=args.max_cycle_len)
        if not cycles:
            print(f"no cycles for {pos} {ch} (skipped={skipped})")
            return
        med_len = int(np.median([len(c) for c in cycles]))
        cyc_last = int(cycles[-1]["frame"].max())
        TAIL_FRAMES = 36   # ~3 h of 5-min frames for the death/arrest final row

        # which detected cycle rows to display (1-based), default all
        disp = cycles
        if args.rows:
            a, b = (int(x) for x in args.rows.split("-"))
            disp = cycles[max(a - 1, 0):b]

        # an explicit final row overrides the auto (death) window
        final_spec = None
        if args.tail_from is not None:
            # final row = the death/arrest window right after the cell stops
            # dividing: a short span (default ~3 h) sampled into n_tiles like
            # every other row, so the last row is NOT wider than the cycle rows.
            tf0 = int(args.tail_from)
            tf1 = int(args.tail_to) if args.tail_to is not None else tf0 + TAIL_FRAMES
            disp = [c for c in cycles if int(c["frame"].max()) < tf0]
            tail = mm[(mm["frame"] >= tf0) & (mm["frame"] <= tf1)]
            if "is_outlier" in tail:
                tail = tail[~tail["is_outlier"].to_numpy(dtype=bool)]
            tail = tail[~tail["frame"].isin(bad)]
            final_spec = {"pos": pos, "ch": ch, "seg": tail,
                          "label": f"{tf0}-{tf1}\n(death)"}
        elif args.last_row:
            # make an explicit frame range the final row (e.g. a chosen last
            # reliable cycle); cycles after it are dropped, no auto death.
            lf0, lf1 = (int(x) for x in args.last_row.split("-"))
            disp = [c for c in cycles if int(c["frame"].max()) < lf0]
            lr = _drop_glitch_frames(
                mm[(mm["frame"] >= lf0) & (mm["frame"] <= lf1)].reset_index(drop=True))
            if "is_outlier" in lr:
                lr = lr[~lr["is_outlier"].to_numpy(dtype=bool)]
            lr = lr[~lr["frame"].isin(bad)]
            final_spec = {"pos": pos, "ch": ch, "seg": lr, "label": f"{lf0}-{lf1}"}

        specs = []
        for seg in disp:
            f0, f1 = int(seg["frame"].min()), int(seg["frame"].max())
            # label is the frame range (data are 5-min frames, so frame numbers,
            # not hours, are the meaningful per-row id)
            specs.append({"pos": pos, "ch": ch, "seg": seg, "label": f"{f0}-{f1}"})

        if final_spec is not None:
            specs.append(final_spec)
        elif not args.no_death:
            # append the "next cycle" after the last completed one -- the cell
            # dies here instead of dividing, so this row is the death window.
            win = int(min(max(med_len, 30), 70))
            tail = mm[(mm["frame"] > cyc_last) & (mm["frame"] <= cyc_last + win)]
            if "is_outlier" in tail:
                tail = tail[~tail["is_outlier"].to_numpy(dtype=bool)]
            tail = tail[~tail["frame"].isin(bad)]
            if len(tail) >= 4:
                tf0, tf1 = int(tail["frame"].min()), int(tail["frame"].max())
                specs.append({"pos": pos, "ch": ch, "seg": tail,
                              "label": f"{tf0}-{tf1}\n(death)"})
        # sub-sample each cycle to --n-tiles (endpoints birth+division kept)
        fig, rows = make_contact_sheet(specs, px, args.cmap, gap=args.gap,
                                       vmin=args.vmin, vmax=args.vmax,
                                       n_tiles=args.n_tiles)
        if args.tail_from is not None:
            mode_txt = f"tail from {args.tail_from}"
        elif args.last_row:
            mode_txt = f"last row {args.last_row}"
        elif args.rows:
            mode_txt = f"rows {args.rows}" + ("" if args.no_death else " + death")
        else:
            mode_txt = "all cycles" + ("" if args.no_death else " + death")
        fig.axes[0].set_title(
            f"{pos} {ch} - {len(rows)} rows ({mode_txt}; frame-labeled, "
            f"inferno {args.vmin}-{args.vmax} rad)", fontsize=9)
        save_figure(
            fig,
            params={"dataset": "260517", "pos": pos, "z": Z, "ch": ch,
                    "cmap": args.cmap, "pixel_size_um": px, "n_rows": len(rows),
                    "gap_px": args.gap, "vmin": args.vmin, "vmax": args.vmax,
                    "max_frame": 2018, "n_tiles": args.n_tiles,
                    "rows": args.rows, "death_row": (not args.no_death),
                    "skipped": json.dumps(skipped),
                    "style": "all-cycles-to-2018 per channel (sampled)"},
            description=(
                f"All {len(rows)} division-terminated cell cycles of {pos} {ch} "
                f"up to frame 2018, {args.n_tiles} frames/cycle sampled "
                f"(endpoints kept; outlier/bad frames removed from tiles); "
                f"fixed {args.vmin}-{args.vmax} rad, "
                f"{args.cmap}. Skipped segments: {skipped} (too_long_merged = "
                f"missed-division artifacts, too_short_fragment = spurious "
                f"divisions)."),
            data={f"row{i}_frames": r["frames"] for i, r in enumerate(rows)} |
                 {f"row{i}_times_h": r["times"] for i, r in enumerate(rows)},
        )
        plt.close(fig)
        print(f"done {pos} {ch}: {len(rows)} cycles  skipped={skipped}")
        return

    if args.contact:
        specs = []
        for tok in args.contact.split(","):
            pos, ch, fr = tok.strip().split(":")
            f0, f1 = (int(x) for x in fr.split("-"))
            cd = data_root(pos) / ch
            dd = pd.read_csv(cd / "inference_out" / "lineage_out" /
                             "lineage_data3D.csv")
            mm = dd[dd["rank"] == 1].sort_values("frame").reset_index(drop=True)
            seg = _drop_glitch_frames(
                mm[(mm["frame"] >= f0) & (mm["frame"] <= f1)].reset_index(drop=True))
            # 260517 media schedule (frames): 2% until 2019, 0.0055% to 2307,
            # 0% to 2885, then 2% refeed.
            ph = ("2%" if f0 < 2019 else "0.0055%" if f0 < 2307
                  else "0%" if f0 < 2885 else "refeed2%")
            specs.append({"pos": pos, "ch": ch, "seg": seg,
                          "label": f"{pos} {ch}  [{ph}]\n{f0}-{f1}  "
                                   f"{seg['time_h'].iloc[0]:.0f}h"})
        px = load_pixel_size(data_root(specs[0]["pos"]) / specs[0]["ch"])
        fig, rows = make_contact_sheet(specs, px, args.cmap, gap=args.gap,
                                       vmin=args.vmin, vmax=args.vmax)
        save_figure(
            fig,
            params={"dataset": "260517", "z": Z, "cmap": args.cmap,
                    "pixel_size_um": px, "n_rows": len(rows),
                    "specs": args.contact, "gap_px": args.gap,
                    "vmin": args.vmin, "vmax": args.vmax,
                    "style": "eLife fig2A cell-cycle contact sheet"},
            description=(
                f"Contact sheet of {len(rows)} single cell cycles "
                f"({args.contact}); each row is one mother across one cycle, "
                f"tiles rotated to a common vertical orientation, per-row "
                f"scaled, {args.cmap}."),
            data={f"row{i}_frames": r["frames"] for i, r in enumerate(rows)} |
                 {f"row{i}_times_h": r["times"] for i, r in enumerate(rows)},
        )
        plt.close(fig)
        print("done")
        return

    ch_dir = data_root(args.pos) / args.ch
    px = load_pixel_size(ch_dir)
    df = pd.read_csv(ch_dir / "inference_out" / "lineage_out" / "lineage_data3D.csv")
    m = df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)

    # parse --ranks into a 1-based list
    ranks: list[int] = []
    for tok in args.ranks.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            ranks.extend(range(int(a), int(b) + 1))
        elif tok:
            ranks.append(int(tok))

    # ---- overlay: a clean single-rod mid-cycle frame ----
    if not args.strip_only:
        cand = m[(m["long_axis_um"].between(8.5, 12)) &
                 (m["short_axis_um"].between(2.8, 4.5))]
        if "multi_xsec_frac" in cand and cand["multi_xsec_frac"].notna().any():
            cand = cand.sort_values("multi_xsec_frac")
        ov_row = cand.iloc[0] if len(cand) else m.iloc[len(m) // 2]
        fig1, data1 = make_overlay_fig(ch_dir, ov_row, px, args.cmap)
        save_figure(
            fig1,
            params={"dataset": "260517", "pos": "Pos1", "z": "z000", "ch": args.ch,
                    "frame": int(ov_row["frame"]), "cmap": args.cmap,
                    "pixel_size_um": px, "style": "eLife fig1A geometry overlay"},
            description=(
                f"Geometry overlay (fig1A-style) for {args.ch} frame "
                f"{int(ov_row['frame'])}: cyan contour, red medial long axis, gray "
                f"short-axis cross-sections, beside the raw phase in {args.cmap}. "
                f"Cell rotated to a vertical orientation via mask PCA."),
            data={k: np.asarray(v) for k, v in data1.items()
                  if isinstance(v, np.ndarray)} | {
                "vmin": float(data1["vmin"]), "vmax": float(data1["vmax"])},
        )
        plt.close(fig1)

    # ---- cell-cycle strips ----
    if args.frames:
        f0, f1 = (int(x) for x in args.frames.split("-"))
        seg = _drop_glitch_frames(
            m[(m["frame"] >= f0) & (m["frame"] <= f1)].reset_index(drop=True))
        cycle_specs = [(0, seg)]
        print(f"rendering explicit cycle {f0}-{f1} ({len(seg)} frames)")
    else:
        cands = rank_cycles(m)
        print(f"{len(cands)} single cell cycles available; rendering ranks {ranks}")
        cycle_specs = [(rk, cands[rk - 1]) for rk in ranks
                       if 1 <= rk <= len(cands)]

    for rk, seg in cycle_specs:
        Ls = seg["long_axis_um"].to_numpy()
        mono = float((np.diff(Ls) > 0).mean())
        fig2, data2 = make_strip_fig(ch_dir, seg, px, args.cmap)
        save_figure(
            fig2,
            params={"dataset": "260517", "pos": args.pos, "z": Z, "ch": args.ch,
                    "cmap": args.cmap, "pixel_size_um": px, "cycle_rank": rk,
                    "frame_min": int(seg["frame"].min()),
                    "frame_max": int(seg["frame"].max()),
                    "grow_um": round(float(Ls[-1] - Ls[0]), 2),
                    "monotonicity": round(mono, 2),
                    "n_tiles": int(len(data2["frames"])),
                    "style": "eLife fig2A cell-cycle strip"},
            description=(
                f"Cell-cycle strip (fig2A-style, rank {rk}) for {args.ch}: one "
                f"mother cell across a single cycle (frames "
                f"{int(seg['frame'].min())}-{int(seg['frame'].max())}, "
                f"{seg['time_h'].iloc[0]:.1f}-{seg['time_h'].iloc[-1]:.1f} h, "
                f"elongation {Ls[-1]-Ls[0]:.1f} um), each frame rotated to a "
                f"common vertical orientation and tiled in {args.cmap}."),
            data={"times_h": data2["times_h"], "frames": data2["frames"],
                  "vmin": float(data2["vmin"]), "vmax": float(data2["vmax"])},
        )
        plt.close(fig2)
    print("done")


if __name__ == "__main__":
    main()
