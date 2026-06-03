"""
central_cell_lineage_overlay.py
-------------------------------
Render per-frame overlay PNGs showing the lineage assignment produced by
central_cell_lineage_tracker.py.

For each frame, we replay the same rank-pointer tracking logic and keep the
(label_id -> cell_id) mapping. Then we draw a semi-transparent colored overlay
on top of the phase image so the user can visually verify that:

- the mother (cell_id = 0) stays at the center,
- divisions produce an inner-ID (mother-side, parent_id inherited) and an
  outer-ID (new daughter),
- border-touching cells are dropped (shown in dark gray),
- non-in-tree cells (unknown origin) are muted (light gray).

Usage:
    python central_cell_lineage_overlay.py \
        --indir /Volumes/2604/260405/ph_260405/Pos9/output_phase/channels/crop_sub_rawraw/ch01 \
        --upscale 4 --alpha 0.55
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# reuse tracker internals
from central_cell_lineage_tracker import (
    LineageState,
    _make_frame_data,
    collect_frame_pairs,
    extract_cells_from_frame,
    load_label_image,
    load_phase_image,
    update_rank_to_id,
)

# =============================================================================
# Color assignment
# =============================================================================
# Okabe-Ito palette (color-blind safe). mother is fixed to vermillion.
OKABE_ITO = [
    (213,  94,   0),   # vermillion (reserved for mother, cell_id=0)
    (230, 159,   0),   # orange
    ( 86, 180, 233),   # sky blue
    (  0, 158, 115),   # bluish green
    (240, 228,  66),   # yellow
    (  0, 114, 178),   # blue
    (204, 121, 167),   # reddish purple
    (  0,   0,   0),   # black (not used for fill; reserved)
]
MOTHER_COLOR = OKABE_ITO[0]
DAUGHTER_CYCLE = OKABE_ITO[1:7]  # 6 rotating colors
NON_IN_TREE_COLOR = (170, 170, 170)   # light gray
BORDER_COLOR = (70, 70, 70)            # dark gray for border-touching cells


def color_for_cell(cell_id: int, in_tree: bool) -> tuple[int, int, int]:
    if cell_id == 0:
        return MOTHER_COLOR
    if not in_tree:
        return NON_IN_TREE_COLOR
    # deterministic color cycling for daughter lineages (by cell_id)
    return DAUGHTER_CYCLE[cell_id % len(DAUGHTER_CYCLE)]


# =============================================================================
# Phase normalization
# =============================================================================
def phase_to_gray(phase: Optional[np.ndarray], shape: tuple[int, int],
                  vlow: float, vhigh: float) -> np.ndarray:
    """Normalize a phase image to uint8 grayscale. Returns a H x W uint8 array."""
    if phase is None or phase.shape != shape:
        return np.full(shape, 60, dtype=np.uint8)  # dark placeholder
    arr = np.clip(phase, vlow, vhigh)
    if vhigh > vlow:
        arr = (arr - vlow) / (vhigh - vlow) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def estimate_phase_range(pairs, n_samples: int = 30) -> tuple[float, float]:
    """Sample a few phase frames to determine a stable display range."""
    lo_list, hi_list = [], []
    step = max(1, len(pairs) // n_samples)
    for i in range(0, len(pairs), step):
        _, _, phase_path = pairs[i]
        if phase_path is None or not phase_path.exists():
            continue
        ph = load_phase_image(phase_path)
        if ph is None:
            continue
        lo_list.append(np.percentile(ph, 1))
        hi_list.append(np.percentile(ph, 99))
        if len(lo_list) >= n_samples:
            break
    if not lo_list:
        return 0.0, 1.0
    return float(np.median(lo_list)), float(np.median(hi_list))


# =============================================================================
# Overlay rendering
# =============================================================================
def render_overlay(
    phase: Optional[np.ndarray],
    mask: np.ndarray,
    label_to_cell: dict[int, int],
    cell_in_tree: dict[int, bool],
    vlow: float, vhigh: float,
    alpha: float,
    upscale: int,
    annotation: Optional[str] = None,
    mother_only: bool = False,
) -> Image.Image:
    """Build an RGB PIL image: phase grayscale + semi-transparent colored masks."""
    h, w = mask.shape
    gray = phase_to_gray(phase, (h, w), vlow, vhigh)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    # paint per label
    used_labels = set(label_to_cell.keys())
    for lab in np.unique(mask):
        if lab == 0:
            continue
        region = mask == lab
        if int(lab) in used_labels:
            cid = label_to_cell[int(lab)]
            if mother_only and cid != 0:
                continue
            color = np.array(color_for_cell(cid, cell_in_tree.get(cid, False)),
                             dtype=np.float32)
        else:
            if mother_only:
                continue
            # label present but dropped by extract (border-touching or too small)
            color = np.array(BORDER_COLOR, dtype=np.float32)
        rgb[region] = rgb[region] * (1.0 - alpha) + color * alpha

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")

    if upscale and upscale > 1:
        img = img.resize((w * upscale, h * upscale), resample=Image.NEAREST)

    if annotation:
        # lightweight frame number in top-left (PIL default font)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((4, 2), annotation, fill=(255, 255, 255))
    return img


# =============================================================================
# Main driver: replay tracking + emit overlays
# =============================================================================
def run(channel_dir: Path, out_dir: Path, upscale: int, alpha: float,
        min_area: int, max_frames: Optional[int],
        mother_only: bool = False) -> None:
    pairs = collect_frame_pairs(channel_dir)
    if max_frames is not None:
        pairs = pairs[:max_frames]
    if not pairs:
        raise RuntimeError(f"no mask files in {channel_dir}/inference_out")
    print(f"[info] {len(pairs)} frames", file=sys.stderr)

    vlow, vhigh = estimate_phase_range(pairs)
    print(f"[info] phase display range: [{vlow:.4f}, {vhigh:.4f}]", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)

    state = LineageState()
    rank_ids_prev: list[int] = []
    prev_areas: list[int] = []
    expected_shape: Optional[tuple[int, int]] = None
    skipped = 0

    n_digits = max(4, len(str(len(pairs) - 1)))

    for t, mask_path, phase_path in pairs:
        mask = load_label_image(mask_path)
        if expected_shape is None and mask.size > 0 and mask.max() > 0:
            expected_shape = mask.shape

        # bad-shape or empty placeholder: emit a blank frame but keep state
        bad = False
        if expected_shape is not None and mask.shape != expected_shape:
            bad = True
        if t > 0 and (mask.size == 0 or mask.max() == 0):
            bad = True

        phase = load_phase_image(phase_path) if phase_path and phase_path.exists() else None

        if bad:
            # write a blank-ish frame for continuity
            h, w = expected_shape if expected_shape else (40, 270)
            blank = np.zeros((h, w), dtype=np.uint16)
            img = render_overlay(phase, blank, {}, {}, vlow, vhigh, alpha, upscale,
                                  annotation=f"t={t} (skipped)",
                                  mother_only=mother_only)
            img.save(out_dir / f"frame_{t:0{n_digits}d}.png")
            skipped += 1
            continue

        df = extract_cells_from_frame(mask, phase, min_area)

        # build label -> cell_id for this frame
        if t == 0 or not rank_ids_prev:
            # initialization (matches tracker.run)
            rank_ids_prev = []
            prev_areas = []
            for idx, row in df.iterrows():
                in_tree = (idx == 0)
                new_id = state.new_id(parent=None, birth=None, in_tree=in_tree)
                rank_ids_prev.append(new_id)
                prev_areas.append(int(row["area_px"]))
                state.append(new_id, _make_frame_data(row, t, int(row["rank"])))
            curr_ids = list(rank_ids_prev)
        else:
            curr_ids = update_rank_to_id(state, rank_ids_prev, prev_areas, df, t)
            rank_ids_prev = curr_ids
            prev_areas = [int(df.iloc[i]["area_px"]) for i in range(len(df))][: len(curr_ids)]

        # curr_df rows (order = rank) -> curr_ids (same order). df already filtered to non-border.
        labels = [int(df.iloc[i]["label"]) for i in range(len(curr_ids))]
        label_to_cell = dict(zip(labels, curr_ids))
        cell_in_tree = {cid: state.cells[cid].in_tree for cid in curr_ids}

        img = render_overlay(phase, mask, label_to_cell, cell_in_tree,
                              vlow, vhigh, alpha, upscale,
                              annotation=f"t={t}",
                              mother_only=mother_only)
        img.save(out_dir / f"frame_{t:0{n_digits}d}.png")

        if (t + 1) % 200 == 0:
            print(f"[info] rendered {t+1}/{len(pairs)}", file=sys.stderr)

    print(f"[ok] wrote {len(pairs)} frames -> {out_dir}", file=sys.stderr)
    if skipped:
        print(f"[info] {skipped} frames marked as skipped (bad mask)", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indir", required=True, type=Path,
                   help="channel dir containing inference_out/*_masks.tif and phase TIFs")
    p.add_argument("--outdir", type=Path, default=None,
                   help="output dir (default: <indir>/inference_out/lineage_overlay)")
    p.add_argument("--upscale", type=int, default=4,
                   help="integer upscale factor for easier viewing (default 4)")
    p.add_argument("--alpha", type=float, default=0.55,
                   help="overlay opacity, 0..1 (default 0.55)")
    p.add_argument("--min-area", type=int, default=20)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--mother-only", action="store_true",
                   help="render overlay only for the mother cell (cell_id=0); "
                        "all other labels are left as bare phase")
    return p


def main() -> int:
    args = build_parser().parse_args()
    default_sub = "lineage_overlay_mother" if args.mother_only else "lineage_overlay"
    out_dir = args.outdir if args.outdir else args.indir / "inference_out" / default_sub
    run(args.indir, out_dir, args.upscale, args.alpha, args.min_area, args.max_frames,
        mother_only=args.mother_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
