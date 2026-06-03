"""
central_cell_lineage_tracker.py — Mother Machine 1D lineage tracker.

Builds a lineage tree rooted at the central (mother) cell by ranking all masks
in each frame by their distance from the image x-center, then propagating IDs
frame-by-frame with a two-pointer rank iteration:

  - continuation      : area ratio > DIV_AREA_RATIO_MIN       -> same ID
  - division          : curr[r] + curr[r+1] ~= prev[r_prev]   -> inner=parent, outer=new daughter
  - outlier           : neither holds                         -> continuation + is_outlier flag

Only mother's descendants are kept in the tree; cells present at frame 0 at
rank >= 2 (unknown lineage) are tracked for bookkeeping but excluded from the
tree figure.

Input:
    channel_dir/                         raw phase tifs (*.tif)
      inference_out/*_masks.tif          segmentation masks

Output (under <channel_dir>/inference_out/lineage_out/):
    lineage_table.csv                    per-frame per-cell metrics
    lineage_cells.json                   per-cell birth/death/parent summary
    + PDF/PNG figures via figure_logger (tree + volume/RI timeseries)

Example:
    python3 central_cell_lineage_tracker.py \\
        --indir /Volumes/2604/260405/ph_260405/Pos9/output_phase/channels/crop_sub_rawraw/ch00 \\
        --pixel-size-um 0.348 --time-interval-min 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
from skimage import measure

from ri_calibration import (
    load_calibration,
    n_medium_at_frame,
    parse_media_schedule,
)

# Division / tracking thresholds
DIV_AREA_RATIO_MIN = 0.68     # continuation if curr/prev > this
DIV_SUM_TOL = 0.30            # |(a+b) - prev| / prev < this -> division confirmed

# 3-frame outlier rule (checked on volume and area independently; OR):
#   outlier if any of:
#     curr/prev < OUT_PREV_LOW
#     curr/prev > OUT_PREV_HIGH
#     curr/next < OUT_NEXT_LOW   (curr is less than next/1.8; i.e. next is >=1.8x curr)
OUT_PREV_LOW = 0.30
OUT_PREV_HIGH = 1.50
OUT_NEXT_LOW = 1.0 / 1.8


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class FrameData:
    frame: int
    rank: int
    area_px: int
    centroid_x: float
    centroid_y: float
    major_axis_px: float
    minor_axis_px: float
    total_phase: float
    touches_border: bool = False
    volume_um3_rod: float = np.nan
    mean_ri: float = np.nan
    mass_pg: float = np.nan
    n_medium_used: float = np.nan
    is_outlier: bool = False


@dataclass
class CellInfo:
    cell_id: int
    parent_id: Optional[int]
    birth_frame: Optional[int]
    death_frame: Optional[int] = None
    in_tree: bool = False   # True if a descendant of mother
    frames: list[FrameData] = field(default_factory=list)


# =============================================================================
# I/O helpers (adapted from central_cell_track_figures.py)
# =============================================================================
_TP_RE = re.compile(r"img_0*(\d+)_")


def natural_key(text: str) -> list[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def parse_timepoint(stem: str) -> Optional[int]:
    m = _TP_RE.match(stem)
    return int(m.group(1)) if m else None


def strip_mask_suffix(stem: str) -> str:
    return stem[: -len("_masks")] if stem.endswith("_masks") else stem


def _extract_pos_label(channel_dir: Path) -> Optional[str]:
    for part in channel_dir.parts:
        if part.startswith("Pos") and part[3:].isdigit():
            return part
    return None


def load_bad_timepoints(bad_frames_path: Path, pos_label: str) -> tuple[set[int], dict[int, list[str]]]:
    """Return (bad_timepoints, bad_reasons) for one Pos from bad_frames.json.

    bad_reasons maps timepoint -> list of reason strings (e.g. ["zure(7.3px,...)", "modori(...)"]).
    """
    with open(bad_frames_path) as f:
        data = json.load(f)
    entry = data.get(pos_label, {})
    bt_dict = entry.get("bad_timepoints", {})
    bad_tp = {int(k) for k in bt_dict}
    bad_reasons = {
        int(k): (list(v) if isinstance(v, list) else [str(v)])
        for k, v in bt_dict.items()
    }
    return bad_tp, bad_reasons


def is_binary_mask(mask: np.ndarray) -> bool:
    if mask.dtype == bool:
        return True
    vals = np.unique(mask)
    return vals.size <= 2 and set(vals.tolist()).issubset({0, 1})


def load_label_image(path: Path) -> np.ndarray:
    arr = np.squeeze(np.asarray(tifffile.imread(path)))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {arr.shape} at {path}")
    if is_binary_mask(arr):
        arr = measure.label(arr > 0, connectivity=1)
    return arr.astype(np.int32, copy=False)


def load_phase_image(path: Path) -> Optional[np.ndarray]:
    """Load phase tif. Returns None if the file is corrupt/empty (not a valid TIFF)."""
    try:
        arr = np.squeeze(np.asarray(tifffile.imread(path)))
    except Exception as e:
        print(f"[warn] failed to read phase tif {path.name}: {e}", file=sys.stderr)
        return None
    if arr.ndim != 2:
        print(f"[warn] unexpected phase shape {arr.shape} at {path.name}", file=sys.stderr)
        return None
    return arr.astype(np.float32, copy=False)


def collect_frame_pairs(
    channel_dir: Path,
    bad_timepoints: Optional[set[int]] = None,
) -> list[tuple[int, Path, Optional[Path], bool]]:
    """Return [(timepoint, mask_path, raw_path, is_bad), ...] in timepoint order.

    Bad frames are kept in the list but marked with is_bad=True so the caller
    can route them to a separate "raw measurements only, no tracking" path.
    """
    inference_dir = channel_dir / "inference_out"
    mask_paths = sorted(inference_dir.glob("*_masks.tif"), key=lambda p: natural_key(p.name))
    raw_paths = [
        p for p in sorted(channel_dir.glob("*.tif"), key=lambda p: natural_key(p.name))
        if not p.stem.endswith(("_masks", "_binary"))
    ]
    raw_map = {p.stem: p for p in raw_paths}
    pairs: list[tuple[int, Path, Optional[Path], bool]] = []
    n_bad = 0
    for mpath in mask_paths:
        stem = strip_mask_suffix(mpath.stem)
        tp = parse_timepoint(stem)
        if tp is None:
            raise ValueError(
                f"Cannot parse timepoint from mask filename: {mpath.name}. "
                f"Expected pattern: img_NNNNNNNNN_*_masks.tif"
            )
        is_bad = bool(bad_timepoints and tp in bad_timepoints)
        if is_bad:
            n_bad += 1
        pairs.append((tp, mpath, raw_map.get(stem), is_bad))
    if n_bad:
        print(f"[info] {n_bad} bad-frame timepoints flagged (excluded from tracking, "
              f"raw measurements saved to lineage_bad_frames.csv)", file=sys.stderr)
    return pairs


# =============================================================================
# Physics: rod volume, RI, mass  (eqs. taken from central_cell_track_figures.py)
# =============================================================================
def calc_rod_volume_um3(major_px: float, minor_px: float, pixel_size_um: float) -> float:
    L = float(major_px) * pixel_size_um
    w = float(minor_px) * pixel_size_um
    r = w / 2.0
    h = L - 2.0 * r
    if h < 0:
        return float((4.0 / 3.0) * np.pi * r**3)
    return float((4.0 / 3.0) * np.pi * r**3 + np.pi * r**2 * h)


def calc_optical_metrics(
    total_phase: float, volume_um3: float, pixel_size_um: float,
    wavelength_nm: float, n_medium: float, alpha_ri: float,
    n_protein_basis: float | None = None,
) -> tuple[float, float, float]:
    if not (np.isfinite(total_phase) and np.isfinite(volume_um3) and volume_um3 > 0 and pixel_size_um > 0):
        return np.nan, np.nan, np.nan
    wl_um = wavelength_nm * 1e-3
    px_area = pixel_size_um**2
    mean_ri = n_medium + (total_phase * wl_um * px_area) / (2.0 * np.pi * volume_um3)
    if not np.isfinite(mean_ri):
        return np.nan, np.nan, np.nan
    basis = float(n_protein_basis) if n_protein_basis is not None else float(n_medium)
    conc = (mean_ri - basis) / alpha_ri if alpha_ri > 0 else np.nan
    mass = conc * volume_um3 * 1e-3 if np.isfinite(conc) else np.nan
    return float(mean_ri), float(conc), float(mass)


# =============================================================================
# Per-frame cell extraction
# =============================================================================
def extract_cells_from_frame(
    mask_label: np.ndarray,
    phase: Optional[np.ndarray],
    min_area: int,
) -> pd.DataFrame:
    h, w = mask_label.shape
    props = measure.regionprops(mask_label, intensity_image=phase)
    rows = []
    x_center = w / 2.0
    for p in props:
        if p.area < min_area:
            continue
        cy, cx = float(p.centroid[0]), float(p.centroid[1])
        total_phase = float(np.sum(phase[mask_label == p.label])) if phase is not None else np.nan
        minr, minc, maxr, maxc = p.bbox
        touches = (minr <= 0) or (minc <= 0) or (maxr >= h) or (maxc >= w)
        rows.append({
            "label": int(p.label),
            "area_px": int(p.area),
            "centroid_y": cy,
            "centroid_x": cx,
            "major_axis_px": float(getattr(p, "major_axis_length", np.nan)),
            "minor_axis_px": float(getattr(p, "minor_axis_length", np.nan)),
            "total_phase": total_phase,
            "dist_x": abs(cx - x_center),
            "touches_border": bool(touches),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # drop cells touching the image border: treat as exited
    df = df[~df["touches_border"]].reset_index(drop=True)
    if df.empty:
        return df
    # rank 1, 2, 3, ... by x distance from center
    df = df.sort_values("dist_x").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# =============================================================================
# Core: rank-pointer iteration for ID propagation
# =============================================================================
class LineageState:
    def __init__(self) -> None:
        self._next_id = 0
        self.cells: dict[int, CellInfo] = {}

    def new_id(self, parent: Optional[int], birth: Optional[int], in_tree: bool) -> int:
        cid = self._next_id
        self._next_id += 1
        self.cells[cid] = CellInfo(cell_id=cid, parent_id=parent,
                                    birth_frame=birth, in_tree=in_tree)
        return cid

    def append(self, cid: int, fd: FrameData) -> None:
        self.cells[cid].frames.append(fd)


def _make_frame_data(row: pd.Series, frame: int, rank: int, is_outlier: bool = False) -> FrameData:
    return FrameData(
        frame=frame, rank=rank,
        area_px=int(row["area_px"]),
        centroid_x=float(row["centroid_x"]),
        centroid_y=float(row["centroid_y"]),
        major_axis_px=float(row["major_axis_px"]),
        minor_axis_px=float(row["minor_axis_px"]),
        total_phase=float(row["total_phase"]),
        touches_border=bool(row.get("touches_border", False)),
        is_outlier=is_outlier,
    )


def update_rank_to_id(
    state: LineageState,
    prev_ids: list[int],
    prev_areas: list[int],
    curr_df: pd.DataFrame,
    t: int,
) -> list[int]:
    """Two-pointer rank iteration. Returns rank_to_id for frame t."""
    N_prev = len(prev_ids)
    N_curr = len(curr_df)
    result: list[Optional[int]] = [None] * N_curr

    r_prev = 0
    r_curr = 0
    while r_curr < N_curr and r_prev < N_prev:
        prev_A = prev_areas[r_prev]
        curr_A = int(curr_df.iloc[r_curr]["area_px"])
        ratio = curr_A / prev_A if prev_A > 0 else 0.0

        if ratio > DIV_AREA_RATIO_MIN:
            # continuation: same ID at this rank
            cid = prev_ids[r_prev]
            result[r_curr] = cid
            state.append(cid, _make_frame_data(curr_df.iloc[r_curr], t, r_curr + 1))
            r_curr += 1
            r_prev += 1
        elif r_curr + 1 < N_curr and _is_division(prev_A, curr_A, int(curr_df.iloc[r_curr + 1]["area_px"])):
            parent = prev_ids[r_prev]
            # inner product keeps parent ID
            result[r_curr] = parent
            state.append(parent, _make_frame_data(curr_df.iloc[r_curr], t, r_curr + 1))
            # outer product is new daughter; in_tree iff parent is in tree
            in_tree = state.cells[parent].in_tree
            new_id = state.new_id(parent=parent, birth=t, in_tree=in_tree)
            result[r_curr + 1] = new_id
            state.append(new_id, _make_frame_data(curr_df.iloc[r_curr + 1], t, r_curr + 2))
            r_curr += 2
            r_prev += 1
        else:
            # unexpected: area dropped but not a clean division. treat as continuation + outlier.
            cid = prev_ids[r_prev]
            result[r_curr] = cid
            state.append(cid, _make_frame_data(curr_df.iloc[r_curr], t, r_curr + 1, is_outlier=True))
            r_curr += 1
            r_prev += 1

    # remaining current cells (r_curr < N_curr, r_prev exhausted): new cells entered from outside
    # in Mother Machine's open end this can happen if a cell re-appears or segmentation split.
    # treat as new unknown-lineage cells (not in tree).
    while r_curr < N_curr:
        new_id = state.new_id(parent=None, birth=t, in_tree=False)
        result[r_curr] = new_id
        state.append(new_id, _make_frame_data(curr_df.iloc[r_curr], t, r_curr + 1))
        r_curr += 1

    # remaining previous cells: exited the frame
    for r in range(r_prev, N_prev):
        cid = prev_ids[r]
        if state.cells[cid].death_frame is None:
            state.cells[cid].death_frame = t - 1

    return [c for c in result if c is not None]  # type: ignore[return-value]


def _is_division(prev_A: int, curr_A: int, next_A: int) -> bool:
    """True if curr+next ~ prev (within DIV_SUM_TOL)."""
    if prev_A <= 0:
        return False
    return abs((curr_A + next_A) - prev_A) / prev_A < DIV_SUM_TOL


# =============================================================================
# Outlier detection: 3-frame rule (prev/next ratio on volume AND area)
# =============================================================================
def _three_frame_outlier(curr: float, prev: float, nxt: float) -> bool:
    """OR-condition outlier rule:
    - curr/prev < 0.3  OR  curr/prev > 1.5  OR  curr/next < 0.5 (curr < next/2)
    """
    if not (np.isfinite(curr) and np.isfinite(prev) and np.isfinite(nxt)):
        return False
    if prev <= 0 or nxt <= 0:
        return False
    r_prev = curr / prev
    r_next = curr / nxt
    return (r_prev < OUT_PREV_LOW) or (r_prev > OUT_PREV_HIGH) or (r_next < OUT_NEXT_LOW)


def apply_outlier_detection(state: LineageState) -> None:
    """Flag frame as outlier if both (curr vs prev) and (curr vs next) ratios are out-of-range,
    for either volume or area. Division ±1 frames (birth boundary) are excluded.
    """
    for cid, cell in state.cells.items():
        if len(cell.frames) < 3:
            continue
        cell.frames.sort(key=lambda f: f.frame)
        volumes = np.array([f.volume_um3_rod for f in cell.frames], dtype=float)
        areas = np.array([float(f.area_px) for f in cell.frames], dtype=float)
        n = len(cell.frames)
        birth = cell.birth_frame
        for i in range(1, n - 1):
            f = cell.frames[i]
            # skip divisiosn boundary
            if birth is not None and abs(f.frame - birth) <= 1:
                continue
            if _three_frame_outlier(volumes[i], volumes[i - 1], volumes[i + 1]) or \
               _three_frame_outlier(areas[i], areas[i - 1], areas[i + 1]):
                f.is_outlier = True


# =============================================================================
# Metric computation (after tracking)
# =============================================================================
def compute_metrics_for_all(
    state: LineageState,
    pixel_size_um: float, wavelength_nm: float, n_medium: float, alpha_ri: float,
    media_schedule: list[tuple[int, str]] | None = None,
    media_ri: dict[str, float] | None = None,
    n_milliq: float | None = None,
) -> None:
    """Compute volume/mean_RI/mass for every frame.

    Frame-dependent medium RI: when `media_schedule` + `media_ri` are provided,
    n_medium for each frame is looked up from the schedule. Otherwise the scalar
    `n_medium` is used for every frame (legacy behavior).

    Protein basis: when `n_milliq` is given, mass uses MilliQ as the baseline so
    a step in medium RI doesn't show up as a step in dry mass. Otherwise the
    per-frame n_medium is the basis (legacy behavior).
    """
    use_schedule = bool(media_schedule) and bool(media_ri)
    for cell in state.cells.values():
        for f in cell.frames:
            f.volume_um3_rod = calc_rod_volume_um3(f.major_axis_px, f.minor_axis_px, pixel_size_um)
            nm_f = (
                n_medium_at_frame(int(f.frame), media_schedule, media_ri)
                if use_schedule else float(n_medium)
            )
            f.n_medium_used = nm_f
            ri, _conc, mass = calc_optical_metrics(
                f.total_phase, f.volume_um3_rod, pixel_size_um, wavelength_nm,
                nm_f, alpha_ri, n_protein_basis=n_milliq,
            )
            f.mean_ri = ri
            f.mass_pg = mass


# =============================================================================
# Output: CSV / JSON
# =============================================================================
def build_bad_frames_table(
    bad_records: list[dict],
    pixel_size_um: float,
    time_interval_min: Optional[float],
    media_schedule: Optional[list[tuple[int, str]]] = None,
    media_ri: Optional[dict[str, float]] = None,
    bad_reasons: Optional[dict[int, list[str]]] = None,
) -> pd.DataFrame:
    """Build the lineage_bad_frames.csv table from raw region records.

    Each row = one detected mask region in one bad frame. NOT tracked, so no
    cell_id. mean_ri/mass_pg are deliberately NaN because total_phase is read
    on a drift-uncorrected phase image and is not trustworthy. Morphology
    (area, major/minor, volume_um3_rod) is stored at face value since shape
    is locally observed and largely independent of the centroid drift error.
    """
    use_schedule = bool(media_schedule) and bool(media_ri)
    bad_reasons = bad_reasons or {}
    rows = []
    for r in bad_records:
        L_um = r["major_axis_px"] * pixel_size_um
        w_um = r["minor_axis_px"] * pixel_size_um
        volume_um3 = calc_rod_volume_um3(
            r["major_axis_px"], r["minor_axis_px"], pixel_size_um
        )
        t_h = (
            r["frame"] * (time_interval_min / 60.0)
            if time_interval_min else np.nan
        )
        n_med = (
            n_medium_at_frame(int(r["frame"]), media_schedule, media_ri)
            if use_schedule else np.nan
        )
        reason = "; ".join(bad_reasons.get(int(r["frame"]), []))
        rows.append({
            "frame": int(r["frame"]),
            "time_h": t_h,
            "rank_in_frame": int(r["rank_in_frame"]),
            "label": int(r["label"]),
            "area_px": int(r["area_px"]),
            "area_um2": int(r["area_px"]) * (pixel_size_um ** 2),
            "long_axis_um": L_um,
            "short_axis_um": w_um,
            "centroid_x_px": float(r["centroid_x_px"]),
            "centroid_y_px": float(r["centroid_y_px"]),
            "total_phase": float(r["total_phase"]),
            "volume_um3_rod": volume_um3,
            "mean_ri": np.nan,   # intentionally NaN: drift-uncorrected phase
            "mass_pg": np.nan,   # intentionally NaN: depends on mean_ri
            "n_medium_used": n_med,
            "touches_border": bool(r["touches_border"]),
            "bad_reason": reason,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "frame", "time_h", "rank_in_frame", "label",
            "area_px", "area_um2", "long_axis_um", "short_axis_um",
            "centroid_x_px", "centroid_y_px", "total_phase",
            "volume_um3_rod", "mean_ri", "mass_pg",
            "n_medium_used", "touches_border", "bad_reason",
        ])
    return (
        pd.DataFrame(rows)
          .sort_values(["frame", "rank_in_frame"])
          .reset_index(drop=True)
    )


def build_long_table(
    state: LineageState,
    time_interval_min: Optional[float],
    pixel_size_um: float,
    n_milliq: float | None = None,
) -> pd.DataFrame:
    """Per-frame long table (data3D style). Outlier frames mask out volume/RI/mass to NaN."""
    rows = []
    n_milliq_val = float(n_milliq) if n_milliq is not None else np.nan
    for cid, cell in state.cells.items():
        for f in cell.frames:
            t_h = f.frame * (time_interval_min / 60.0) if time_interval_min else np.nan
            hide = f.is_outlier or f.touches_border
            v = np.nan if hide else f.volume_um3_rod
            ri = np.nan if hide else f.mean_ri
            mass = np.nan if hide else f.mass_pg
            rows.append({
                "cell_id": cid,
                "parent_id": cell.parent_id if cell.parent_id is not None else -1,
                "in_tree": cell.in_tree,
                "birth_frame": cell.birth_frame if cell.birth_frame is not None else -1,
                "death_frame": cell.death_frame if cell.death_frame is not None else -1,
                "frame": f.frame,
                "time_h": t_h,
                "rank": f.rank,
                "area_px": f.area_px,
                "area_um2": f.area_px * (pixel_size_um ** 2),
                "long_axis_um": f.major_axis_px * pixel_size_um,
                "short_axis_um": f.minor_axis_px * pixel_size_um,
                "centroid_x_px": f.centroid_x,
                "centroid_y_px": f.centroid_y,
                "total_phase": f.total_phase,
                "volume_um3_rod": v,
                "mean_ri": ri,
                "mass_pg": mass,
                "n_medium_used": f.n_medium_used,
                "n_milliq_used": n_milliq_val,
                "is_outlier": f.is_outlier,
                "touches_border": f.touches_border,
            })
    if not rows:
        # No cells ever detected — return an empty frame with the expected schema
        # so downstream code (per_channel_figures, batch_figures) can read it.
        return pd.DataFrame(columns=[
            "cell_id", "parent_id", "in_tree", "birth_frame", "death_frame",
            "frame", "time_h", "rank",
            "area_px", "area_um2", "long_axis_um", "short_axis_um",
            "centroid_x_px", "centroid_y_px", "total_phase",
            "volume_um3_rod", "mean_ri", "mass_pg",
            "n_medium_used", "n_milliq_used",
            "is_outlier", "touches_border",
        ])
    return pd.DataFrame(rows).sort_values(["cell_id", "frame"]).reset_index(drop=True)


def _children_of(state: LineageState, parent_id: int) -> list[int]:
    return sorted(
        (cid for cid, c in state.cells.items() if c.parent_id == parent_id),
        key=lambda cid: state.cells[cid].birth_frame or 0,
    )


def _generation(state: LineageState, cell: CellInfo) -> int:
    g = 0
    cur = cell
    while cur.parent_id is not None:
        parent = state.cells.get(cur.parent_id)
        if parent is None:
            break
        g += 1
        cur = parent
    return g


def _dist_to_edge(f: FrameData, image_width_px: int, image_height_px: int) -> float:
    return float(min(f.centroid_x, image_width_px - f.centroid_x,
                     f.centroid_y, image_height_px - f.centroid_y))


def build_clist_table(
    state: LineageState,
    image_width_px: int,
    image_height_px: int,
    pixel_size_um: float,
    time_interval_min: Optional[float],
) -> pd.DataFrame:
    """Per-cell summary table (SuperSegger clist style).

    Captures birth/death values, mother/daughter IDs, generation, growth metrics,
    adapted for QPI (adds RI, mass, volume).
    """
    rows = []
    for cid, cell in state.cells.items():
        if not cell.frames:
            continue
        f_birth = cell.frames[0]
        f_death = cell.frames[-1]
        valid = [x for x in cell.frames if not x.is_outlier]
        daughters = _children_of(state, cid)
        d1 = daughters[0] if len(daughters) >= 1 else -1
        d2 = daughters[1] if len(daughters) >= 2 else -1

        # growth: dL per-frame, length ratio death/birth
        L_series = np.array([x.major_axis_px for x in cell.frames]) * pixel_size_um
        if len(L_series) >= 2:
            dL = np.diff(L_series)
            dL_max = float(np.nanmax(dL))
            dL_min = float(np.nanmin(dL))
        else:
            dL_max = dL_min = np.nan
        L_ratio = float(L_series[-1] / L_series[0]) if L_series[0] > 0 else np.nan

        # volume/RI/mass at birth and death
        def _mass_of(fr: FrameData) -> float:
            return np.nan if fr.is_outlier else fr.mass_pg
        def _ri_of(fr: FrameData) -> float:
            return np.nan if fr.is_outlier else fr.mean_ri
        def _vol_of(fr: FrameData) -> float:
            return np.nan if fr.is_outlier else fr.volume_um3_rod

        rows.append({
            "cell_id": cid,
            "mother_id": cell.parent_id if cell.parent_id is not None else -1,
            "daughter1_id": d1,
            "daughter2_id": d2,
            "generation": _generation(state, cell),
            "in_tree": cell.in_tree,
            "n_frames": len(cell.frames),
            "n_outliers": sum(1 for x in cell.frames if x.is_outlier),

            "birth_frame": cell.birth_frame if cell.birth_frame is not None else f_birth.frame,
            "death_frame": cell.death_frame if cell.death_frame is not None else f_death.frame,
            "age_frames": f_death.frame - f_birth.frame,
            "birth_time_h": (f_birth.frame * time_interval_min / 60.0) if time_interval_min else np.nan,
            "death_time_h": (f_death.frame * time_interval_min / 60.0) if time_interval_min else np.nan,
            "age_h": ((f_death.frame - f_birth.frame) * time_interval_min / 60.0) if time_interval_min else np.nan,

            "long_axis_birth_um": f_birth.major_axis_px * pixel_size_um,
            "long_axis_death_um": f_death.major_axis_px * pixel_size_um,
            "short_axis_birth_um": f_birth.minor_axis_px * pixel_size_um,
            "short_axis_death_um": f_death.minor_axis_px * pixel_size_um,
            "area_birth_um2": f_birth.area_px * (pixel_size_um ** 2),
            "area_death_um2": f_death.area_px * (pixel_size_um ** 2),
            "volume_birth_um3": _vol_of(f_birth),
            "volume_death_um3": _vol_of(f_death),

            "mean_ri_birth": _ri_of(f_birth),
            "mean_ri_death": _ri_of(f_death),
            "mass_birth_pg": _mass_of(f_birth),
            "mass_death_pg": _mass_of(f_death),
            "n_medium_birth": float(f_birth.n_medium_used),
            "n_medium_death": float(f_death.n_medium_used),

            "x_pos_birth_px": f_birth.centroid_x,
            "y_pos_birth_px": f_birth.centroid_y,
            "x_pos_death_px": f_death.centroid_x,
            "y_pos_death_px": f_death.centroid_y,
            "dist_to_edge_birth_px": _dist_to_edge(f_birth, image_width_px, image_height_px),

            "dL_max_um_per_frame": dL_max,
            "dL_min_um_per_frame": dL_min,
            "L_death_over_birth": L_ratio,

            "rank_birth": f_birth.rank,
            "rank_death": f_death.rank,

            "mean_volume_um3": float(np.nanmean([_vol_of(x) for x in valid])) if valid else np.nan,
            "mean_ri_over_life": float(np.nanmean([_ri_of(x) for x in valid])) if valid else np.nan,
            "mean_mass_pg": float(np.nanmean([_mass_of(x) for x in valid])) if valid else np.nan,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "cell_id", "mother_id", "daughter1_id", "daughter2_id",
            "generation", "in_tree", "n_frames", "n_outliers",
            "birth_frame", "death_frame", "age_frames",
            "birth_time_h", "death_time_h", "age_h",
            "long_axis_birth_um", "long_axis_death_um",
            "short_axis_birth_um", "short_axis_death_um",
            "area_birth_um2", "area_death_um2",
            "volume_birth_um3", "volume_death_um3",
            "mean_ri_birth", "mean_ri_death",
            "mass_birth_pg", "mass_death_pg",
            "n_medium_birth", "n_medium_death",
            "x_pos_birth_px", "y_pos_birth_px",
            "x_pos_death_px", "y_pos_death_px",
            "dist_to_edge_birth_px",
            "dL_max_um_per_frame", "dL_min_um_per_frame", "L_death_over_birth",
            "rank_birth", "rank_death",
            "mean_volume_um3", "mean_ri_over_life", "mean_mass_pg",
        ])
    return pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)


def cells_summary_json(state: LineageState) -> list[dict]:
    out = []
    for cid, cell in state.cells.items():
        out.append({
            "cell_id": cid,
            "parent_id": cell.parent_id,
            "daughter_ids": _children_of(state, cid),
            "generation": _generation(state, cell),
            "in_tree": cell.in_tree,
            "birth_frame": cell.birth_frame,
            "death_frame": cell.death_frame,
            "n_frames": len(cell.frames),
        })
    return out


# =============================================================================
# Main pipeline
# =============================================================================
def run(
    channel_dir: Path,
    pixel_size_um: float,
    time_interval_min: Optional[float],
    wavelength_nm: float,
    n_medium: float,
    alpha_ri: float,
    min_area: int,
    max_frames: Optional[int],
    ri_calibration: Optional[Path] = None,
    calibration_id: Optional[str] = None,
    media_schedule_str: Optional[str] = None,
    n_milliq: Optional[float] = None,
    bad_frames: Optional[Path] = None,
) -> None:
    bad_tp: Optional[set[int]] = None
    bad_reasons: dict[int, list[str]] = {}
    pos_label: Optional[str] = None
    if bad_frames is not None:
        pos_label = _extract_pos_label(channel_dir)
        if pos_label is None:
            raise RuntimeError(
                f"Cannot determine Pos label from path: {channel_dir}. "
                f"Expected a 'PosN' component in the directory path."
            )
        bad_tp, bad_reasons = load_bad_timepoints(bad_frames, pos_label)
        print(f"[info] {pos_label}: {len(bad_tp)} bad timepoints flagged", file=sys.stderr)

    pairs = collect_frame_pairs(channel_dir, bad_timepoints=bad_tp)
    if max_frames is not None:
        pairs = pairs[:max_frames]
    if not pairs:
        raise RuntimeError(f"No mask files found in {channel_dir}/inference_out")
    print(f"[info] {len(pairs)} frames to process", file=sys.stderr)

    # --- RI calibration / media schedule (optional) ---
    media_schedule: Optional[list[tuple[int, str]]] = None
    media_ri: Optional[dict[str, float]] = None
    calibration_id_used: Optional[str] = None
    n_milliq_used: Optional[float] = n_milliq
    if ri_calibration is not None:
        cal = load_calibration(ri_calibration, calibration_id)
        media_ri = dict(cal.media)
        calibration_id_used = cal.calibration_id
        if n_milliq_used is None:
            n_milliq_used = cal.n_miliq
        if not media_schedule_str:
            raise RuntimeError(
                "--media-schedule is required when --ri-calibration is set "
                "(e.g. '0:wo_2,575:wo_0,1439:wo_2')."
            )
        media_schedule = parse_media_schedule(media_schedule_str)
        print(
            f"[info] calibration: {cal.calibration_id} ({cal.calibrated_at})",
            file=sys.stderr,
        )
        print(f"[info] media RI: {cal.media}", file=sys.stderr)
        print(f"[info] schedule: {media_schedule}", file=sys.stderr)
        print(f"[info] n_milliq for protein basis: {n_milliq_used}", file=sys.stderr)
    elif media_schedule_str:
        raise RuntimeError(
            "--media-schedule was given without --ri-calibration; "
            "schedule lookup needs the calibration JSON for media RI values."
        )

    state = LineageState()
    rank_to_id_prev: list[int] = []
    prev_areas: list[int] = []

    expected_shape: Optional[tuple[int, int]] = None
    skipped_frames: list[int] = []
    bad_records: list[dict] = []
    first_valid_done = False  # init happens on the first non-bad, non-degenerate frame

    for t, mask_path, raw_path, is_bad in pairs:
        mask = load_label_image(mask_path)
        if expected_shape is None and mask.size > 0 and mask.max() > 0:
            expected_shape = mask.shape
        # skip frames with unexpected shape (segmentation placeholder)
        if expected_shape is not None and mask.shape != expected_shape:
            skipped_frames.append(t)
            continue

        phase = load_phase_image(raw_path) if raw_path and raw_path.exists() else None
        df = extract_cells_from_frame(mask, phase, min_area)

        # ---- bad frame branch: collect raw measurements, do NOT track ----
        if is_bad:
            for _, row in df.iterrows():
                bad_records.append({
                    "frame": int(t),
                    "rank_in_frame": int(row["rank"]),
                    "label": int(row["label"]),
                    "area_px": int(row["area_px"]),
                    "centroid_x_px": float(row["centroid_x"]),
                    "centroid_y_px": float(row["centroid_y"]),
                    "major_axis_px": float(row["major_axis_px"]),
                    "minor_axis_px": float(row["minor_axis_px"]),
                    "total_phase": float(row["total_phase"]),
                    "touches_border": bool(row.get("touches_border", False)),
                })
            continue

        # ---- empty mask after init: skip; before init: still let init happen
        # (matches original t==0 behavior which allowed degenerate empty init)
        if first_valid_done and (mask.size == 0 or mask.max() == 0):
            skipped_frames.append(t)
            continue

        if not first_valid_done:
            # initialization on the FIRST non-bad frame (was t==0 originally):
            #   rank 1 = mother (in_tree), others = unknown-lineage
            rank_to_id_prev = []
            prev_areas = []
            for idx, row in df.iterrows():
                in_tree = (idx == 0)
                birth = None  # already present at first valid frame
                new_id = state.new_id(parent=None, birth=birth, in_tree=in_tree)
                rank_to_id_prev.append(new_id)
                prev_areas.append(int(row["area_px"]))
                state.append(new_id, _make_frame_data(row, t, int(row["rank"])))
            first_valid_done = True
            continue

        rank_to_id_prev = update_rank_to_id(state, rank_to_id_prev, prev_areas, df, t)
        prev_areas = [int(df.iloc[i]["area_px"]) for i in range(len(df))][: len(rank_to_id_prev)]

        if (t + 1) % 200 == 0:
            print(f"[info] processed {t+1}/{len(pairs)} frames, {len(state.cells)} lineages so far", file=sys.stderr)

    if skipped_frames:
        print(f"[info] skipped {len(skipped_frames)} bad/empty mask frames (e.g. {skipped_frames[:5]})",
              file=sys.stderr)

    # finalize: cells still alive get death=last frame
    last_frame = pairs[-1][0]
    for cell in state.cells.values():
        if cell.death_frame is None and cell.frames:
            cell.death_frame = cell.frames[-1].frame

    compute_metrics_for_all(
        state, pixel_size_um, wavelength_nm, n_medium, alpha_ri,
        media_schedule=media_schedule, media_ri=media_ri, n_milliq=n_milliq_used,
    )
    apply_outlier_detection(state)

    # === outputs ===
    out_dir = channel_dir / "inference_out" / "lineage_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # image size (from the last mask loaded)
    last_mask = load_label_image(pairs[-1][1])
    img_h, img_w = last_mask.shape

    df_long = build_long_table(state, time_interval_min, pixel_size_um, n_milliq=n_milliq_used)
    long_path = out_dir / "lineage_data3D.csv"
    df_long.to_csv(long_path, index=False)
    print(f"[ok] wrote {long_path}  ({len(df_long)} rows)", file=sys.stderr)

    df_clist = build_clist_table(state, img_w, img_h, pixel_size_um, time_interval_min)
    clist_path = out_dir / "clist.csv"
    df_clist.to_csv(clist_path, index=False)
    print(f"[ok] wrote {clist_path}  ({len(df_clist)} cells)", file=sys.stderr)

    json_path = out_dir / "lineage_cells.json"
    with open(json_path, "w") as f:
        json.dump(cells_summary_json(state), f, indent=2)
    print(f"[ok] wrote {json_path}", file=sys.stderr)

    # === bad frames: raw mask measurements (NOT tracked, no cell_id) ===
    df_bad = build_bad_frames_table(
        bad_records,
        pixel_size_um=pixel_size_um,
        time_interval_min=time_interval_min,
        media_schedule=media_schedule,
        media_ri=media_ri,
        bad_reasons=bad_reasons,
    )
    bad_csv_path = out_dir / "lineage_bad_frames.csv"
    df_bad.to_csv(bad_csv_path, index=False)
    print(f"[ok] wrote {bad_csv_path}  ({len(df_bad)} bad-region records "
          f"across {len(bad_tp) if bad_tp else 0} bad timepoints)", file=sys.stderr)

    # Copy the relevant Pos entry from bad_frames.json next to the CSVs
    # for reproducibility / QC.
    if bad_frames is not None and pos_label is not None:
        try:
            with open(bad_frames) as f:
                bf_full = json.load(f)
            pos_entry = bf_full.get(pos_label, {})
            bad_used_path = out_dir / "bad_frames_used.json"
            with open(bad_used_path, "w") as f:
                json.dump(
                    {
                        "source": str(bad_frames),
                        "pos_label": pos_label,
                        pos_label: pos_entry,
                    },
                    f, indent=2, ensure_ascii=False,
                )
            print(f"[ok] wrote {bad_used_path}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] failed to copy bad_frames entry: {e}", file=sys.stderr)

    n_tree = sum(1 for c in state.cells.values() if c.in_tree)
    n_div = sum(1 for c in state.cells.values() if c.in_tree and c.parent_id is not None)
    n_outlier = int(df_long["is_outlier"].sum())
    print(f"[info] in-tree cells={n_tree}, divisions detected={n_div}, outlier frames={n_outlier}",
          file=sys.stderr)

    params = {
        "channel_dir": str(channel_dir),
        "n_frames": len(pairs),
        "pixel_size_um": pixel_size_um,
        "time_interval_min": time_interval_min,
        "wavelength_nm": wavelength_nm,
        "n_medium": n_medium,
        "alpha_ri": alpha_ri,
        "min_area": min_area,
        "div_area_ratio_min": DIV_AREA_RATIO_MIN,
        "div_sum_tol": DIV_SUM_TOL,
        "out_prev_low": OUT_PREV_LOW,
        "out_prev_high": OUT_PREV_HIGH,
        "out_next_low": OUT_NEXT_LOW,
        "n_tree_cells": n_tree,
        "n_divisions": n_div,
        "n_outlier_frames": n_outlier,
        "calibration_id": calibration_id_used,
        "media_schedule": media_schedule_str,
        "media_ri": media_ri,
        "n_milliq": n_milliq_used,
        "bad_frames_used": (
            {
                "path": str(bad_frames),
                "pos_label": pos_label,
                "n_bad_timepoints": len(bad_tp) if bad_tp else 0,
                "n_bad_region_records": len(df_bad),
            } if bad_frames is not None else None
        ),
    }

    # dump calibration/schedule next to the CSVs so plotting scripts (which
    # read only CSV + this JSON) have everything they need.
    run_meta_path = out_dir / "lineage_run_params.json"
    with open(run_meta_path, "w") as f:
        json.dump(params, f, indent=2, default=str)
    print(f"[ok] wrote {run_meta_path}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", type=Path, required=True,
                   help="Channel directory containing phase TIFs and inference_out/*_masks.tif")
    p.add_argument("--pixel-size-um", type=float, default=0.348)
    p.add_argument("--time-interval-min", type=float, default=5.0,
                   help="Minutes per frame (set to 0 or negative to disable time axis)")
    p.add_argument("--wavelength-nm", type=float, default=658.0)
    p.add_argument("--n-medium", type=float, default=1.333)
    p.add_argument("--alpha-ri", type=float, default=0.00018)
    p.add_argument("--min-area", type=int, default=20)
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit to first N frames (for quick testing)")
    p.add_argument("--ri-calibration", type=Path, default=None,
                   help="Path to RI calibration append-history JSON "
                        "(written by calibrate_ri.py). If set, --media-schedule "
                        "is required and n_medium becomes frame-dependent.")
    p.add_argument("--calibration-id", type=str, default=None,
                   help="Pick a non-active calibration entry by id (defaults to "
                        "the JSON's 'active' field).")
    p.add_argument("--media-schedule", type=str, default=None,
                   help="Frame-keyed medium switches, e.g. '0:wo_2,575:wo_0,1439:wo_2'. "
                        "Must start at frame 0. Requires --ri-calibration.")
    p.add_argument("--n-milliq", type=float, default=None,
                   help="Override the protein-density baseline (defaults to "
                        "calibration's reference.n_miliq). Falls back to per-frame "
                        "n_medium when no calibration is given (legacy behavior).")
    p.add_argument("--bad-frames", type=Path, default=None,
                   help="Path to bad_frames.json (from extract_bad_frames.py). "
                        "Bad timepoints for this Pos are excluded from tracking.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    tim = args.time_interval_min if args.time_interval_min and args.time_interval_min > 0 else None
    run(
        channel_dir=args.indir,
        pixel_size_um=args.pixel_size_um,
        time_interval_min=tim,
        wavelength_nm=args.wavelength_nm,
        n_medium=args.n_medium,
        alpha_ri=args.alpha_ri,
        min_area=args.min_area,
        max_frames=args.max_frames,
        ri_calibration=args.ri_calibration,
        calibration_id=args.calibration_id,
        media_schedule_str=args.media_schedule,
        n_milliq=args.n_milliq,
        bad_frames=args.bad_frames,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
