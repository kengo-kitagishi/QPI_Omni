"""recompute_axes_from_masks.py — mask-direct axis/volume/RI recomputation.

The lineage pipeline records ``short_axis_um`` / ``long_axis_um`` from a
second-moment ellipse fit (skimage regionprops), which over-estimates the
width of a rod in an aspect-ratio-dependent way and so fabricates a within-
cycle width increase. This driver re-measures the tracked mother (rank=1)
directly from the segmentation masks using bias-free width methods:

  - ``skimage_legacy``      : regionprops major/minor (reproduces the CSV; xcheck)
  - ``supersegger_adaptive``: rotate-and-project, adaptive endcap trim
  - ``medial_axis``         : Morphometrics-equivalent perpendicular width

For every mode it recomputes rod volume from the measured axes and — because a
changed volume changes the optics — re-derives ``mean_ri`` and ``mass_pg`` from
the per-frame integrated phase (``total_phase``, taken from the lineage CSV) and
the per-frame medium index (``n_medium_used``). Optical formulae are imported
from ``central_cell_lineage_tracker`` so they stay bit-identical to the pipeline.

Output (long format, one row per frame x mode):
    frame, cell_id, mode,
    short_axis_um, long_axis_um, volume_um3_rod,
    short_axis_px, long_axis_px, area_px,
    total_phase, mean_ri, mass_pg, time_h

Usage:
    python scripts/recompute_axes_from_masks.py --pos Pos27 --ch ch06 \
        --out results/260517/recomputed_axes/Pos27_ch06.csv
    python scripts/recompute_axes_from_masks.py --pos Pos27 --ch ch06 \
        --frame-max 2200        # phase1 + last cycle only (faster pilot)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from qpi_paths import (  # noqa: E402
    find_lineage_csv, find_run_params, mask_dir_for_channel, results_dir,
)
from mask_morphology import (  # noqa: E402
    extract_cell_morphology, measure_single_cell_medial, cross_section_quality,
    measure_all_modes,
)
from mask_volume_schematic import efd_section_geometry  # noqa: E402
from central_cell_lineage_tracker import (  # noqa: E402
    calc_rod_volume_um3, calc_optical_metrics,
)
from skimage import measure  # noqa: E402

DEFAULT_MODES = ["skimage_legacy", "supersegger_adaptive", "medial_axis"]
_IMG_RE = re.compile(r"img_0*(\d+)")


def build_mask_index(inference_dir: Path) -> dict[int, Path]:
    """Map absolute img number -> *_masks.tif path."""
    index: dict[int, Path] = {}
    for p in inference_dir.glob("*_masks.tif"):
        m = _IMG_RE.search(p.name)
        if m:
            index[int(m.group(1))] = p
    return index


def _load_label_image(path: Path) -> np.ndarray:
    arr = np.squeeze(np.asarray(tifffile.imread(path)))
    if arr.ndim != 2:
        raise ValueError(f"expected 2D mask, got {arr.shape} at {path}")
    # Relabel a binary mask so each blob is distinct (matches the tracker).
    uniq = np.unique(arr)
    if uniq.size <= 2 and set(uniq.tolist()) <= {0, 1}:
        arr = measure.label(arr > 0, connectivity=1)
    return arr.astype(np.int32, copy=False)


def _label_at_centroid(mask: np.ndarray, cy: float, cx: float) -> int:
    """Label containing the mother centroid; fall back to the dominant label
    in a small neighbourhood (centroids of bent/concave cells can land on
    background)."""
    h, w = mask.shape
    yi, xi = int(round(cy)), int(round(cx))
    if 0 <= yi < h and 0 <= xi < w and mask[yi, xi] != 0:
        return int(mask[yi, xi])
    y0, y1 = max(yi - 2, 0), min(yi + 3, h)
    x0, x1 = max(xi - 2, 0), min(xi + 3, w)
    nbhd = mask[y0:y1, x0:x1]
    nz = nbhd[nbhd != 0]
    if nz.size:
        vals, counts = np.unique(nz, return_counts=True)
        return int(vals[np.argmax(counts)])
    return 0


def _measure_modes(binary: np.ndarray, modes: list[str], pixel_size_um: float):
    """Return {mode: (short_px, long_px, area_px, volume_profile_px3)}.

    volume_profile_px3 is the solid-of-revolution volume from the width profile
    (only the medial method computes it; 0.0 otherwise)."""
    out: dict[str, tuple[float, float, int, float]] = {}
    area_px = int(binary.sum())
    if "skimage_legacy" in modes:
        lab = measure.label(binary)
        props = measure.regionprops(lab)
        if props:
            p = max(props, key=lambda r: r.area)
            out["skimage_legacy"] = (
                float(p.minor_axis_length), float(p.major_axis_length),
                int(p.area), 0.0,
            )
    if "supersegger_adaptive" in modes:
        m = extract_cell_morphology(binary, pixel_size_um=pixel_size_um,
                                    endcap_trim_frac="adaptive")
        out["supersegger_adaptive"] = (m.short_axis_px, m.long_axis_px, area_px, 0.0)
    if "medial_axis" in modes:
        m = measure_single_cell_medial(binary, pixel_size_um=pixel_size_um)
        out["medial_axis"] = (m.short_axis_px, m.long_axis_px, area_px,
                              m.volume_profile_px3)
    return out


def recompute_channel(pos: str, ch: str, modes: list[str],
                      frame_max: int | None = None,
                      progress_every: int = 250,
                      read_workers: int = 1) -> pd.DataFrame | None:
    csv_path = find_lineage_csv(pos, ch)
    if csv_path is None:
        print(f"[skip] no lineage CSV for {pos}_{ch}", file=sys.stderr)
        return None
    inf_dir = mask_dir_for_channel(csv_path)
    if inf_dir is None:
        print(f"[skip] no mask dir for {pos}_{ch}", file=sys.stderr)
        return None
    params_path = find_run_params(csv_path)
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path else {}
    px = float(params.get("pixel_size_um", 0.34567514677103717))
    wl = float(params.get("wavelength_nm", 658.0))
    alpha = float(params.get("alpha_ri", 0.00018))
    # Dry-mass baseline = MilliQ water RI (the pipeline passes n_milliq as the
    # protein basis so a medium-RI step doesn't show up as a mass step). Prefer
    # the per-frame n_milliq_used column; fall back to the run param.
    n_prot = params.get("n_milliq")

    df = pd.read_csv(csv_path)
    m = df[df["rank"] == 1].sort_values("frame").copy()
    if frame_max is not None:
        m = m[m["frame"] <= frame_max]
    if m.empty:
        print(f"[skip] no rank=1 rows for {pos}_{ch}", file=sys.stderr)
        return None

    mask_index = build_mask_index(inf_dir)
    rows = []
    n_total = len(m)
    n_missing = 0

    def _measure_row(r, mask):
        """Measure one frame's mother and append its per-mode rows."""
        nonlocal n_missing
        frame = int(r["frame"])
        lbl = _label_at_centroid(mask, r["centroid_y_px"], r["centroid_x_px"])
        if lbl == 0:
            n_missing += 1
            return
        binary_full = mask == lbl
        # Crop to the cell bbox (+margin); the mask is the whole chamber and
        # rotating the full frame is wasteful. The crop is measurement-identical.
        ys, xs = np.where(binary_full)
        if ys.size == 0:
            n_missing += 1
            return
        margin = 6
        r0 = max(int(ys.min()) - margin, 0)
        r1 = min(int(ys.max()) + margin + 1, binary_full.shape[0])
        c0 = max(int(xs.min()) - margin, 0)
        c1 = min(int(xs.max()) + margin + 1, binary_full.shape[1])
        binary = binary_full[r0:r1, c0:c1]
        n_medium = float(r.get("n_medium_used", params.get("n_medium", 1.333)))
        row_milliq = r.get("n_milliq_used", np.nan)
        basis = float(row_milliq) if np.isfinite(row_milliq) else n_prot
        total_phase = float(r["total_phase"])
        a = measure_all_modes(binary)  # one rotation, all modes
        if a is None:
            n_missing += 1
            return
        multi_frac, max_xsec = a["multi_xsec_frac"], a["max_xsec"]
        # EFD-contour-intersection volume (adopted method): EFD K=6 smoothed
        # contour + one midpoint update, solid-of-revolution integral. One extra
        # geometry pass on the same crop; identical for every mode, so the same
        # efd columns are attached to each mode's row.
        geo_efd = efd_section_geometry(binary, pixel_size_um=px)
        if geo_efd is not None and geo_efd.volume_um3 > 0:
            vol_efd = geo_efd.volume_um3
            ri_efd, _ce, mass_efd = calc_optical_metrics(
                total_phase, vol_efd, px, wl, n_medium, alpha, basis,
            )
            _w = geo_efd.w_perp_px
            short_efd = float(np.median(_w[_w > 0])) * px if np.any(_w > 0) else np.nan
            long_efd = geo_efd.long_axis_px * px
        else:
            vol_efd = ri_efd = mass_efd = short_efd = long_efd = np.nan
        per_mode = {
            "skimage_legacy": (a["sk_minor_px"], a["sk_major_px"], a["area_px"], 0.0),
            "supersegger_adaptive": (a["adaptive_short_px"], a["adaptive_long_px"],
                                     a["area_px"], 0.0),
            "medial_axis": (a["medial_short_px"], a["medial_long_px"],
                            a["area_px"], a["medial_profile_px3"]),
        }
        for mode in modes:
            if mode not in per_mode:
                continue
            short_px, long_px, area_px, vol_prof_px3 = per_mode[mode]
            vol = calc_rod_volume_um3(long_px, short_px, px)
            mean_ri, _conc, mass = calc_optical_metrics(
                total_phase, vol, px, wl, n_medium, alpha, basis,
            )
            vol_profile = vol_prof_px3 * px ** 3 if vol_prof_px3 > 0 else np.nan
            if np.isfinite(vol_profile) and vol_profile > 0:
                ri_p, _c, mass_p = calc_optical_metrics(
                    total_phase, vol_profile, px, wl, n_medium, alpha, basis,
                )
            else:
                ri_p, mass_p = np.nan, np.nan
            rows.append({
                "frame": frame, "cell_id": int(r["cell_id"]), "mode": mode,
                "short_axis_um": short_px * px, "long_axis_um": long_px * px,
                "volume_um3_rod": vol, "volume_profile_um3": vol_profile,
                "short_axis_px": short_px, "long_axis_px": long_px,
                "area_px": area_px, "total_phase": total_phase,
                "mean_ri": mean_ri, "mass_pg": mass,
                "mean_ri_profile": ri_p, "mass_pg_profile": mass_p,
                "volume_efd_um3": vol_efd, "mean_ri_efd": ri_efd,
                "mass_pg_efd": mass_efd, "short_axis_efd_um": short_efd,
                "long_axis_efd_um": long_efd,
                "time_h": float(r["time_h"]),
                "multi_xsec_frac": multi_frac, "max_xsec": max_xsec,
            })

    # Robocopy-style I/O: read this one channel's masks with a thread pool
    # (parallel reads within a single directory keep the USB HDD's head local,
    # so it benefits from queue depth; concurrent reads across DIFFERENT channel
    # dirs thrash, which is why the process pool stays at 1 channel at a time).
    # tifffile releases the GIL during decode, so the main thread measures while
    # the pool prefetches — I/O and CPU overlap. Chunked to bound memory.
    work = [(r, mask_index[int(r["frame"])]) for _, r in m.iterrows()
            if int(r["frame"]) in mask_index]
    n_missing += n_total - len(work)
    done = 0
    CHUNK = 384
    with ThreadPoolExecutor(max_workers=read_workers) as ex:
        for s in range(0, len(work), CHUNK):
            chunk = work[s:s + CHUNK]
            futs = {ex.submit(_load_label_image, mp): r for r, mp in chunk}
            for fut in as_completed(futs):
                r = futs[fut]
                try:
                    mask = fut.result()
                except Exception:
                    n_missing += 1
                    continue
                _measure_row(r, mask)
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"  {pos}_{ch}: {done}/{n_total} frames", file=sys.stderr)

    if n_missing:
        print(f"[info] {pos}_{ch}: {n_missing}/{n_total} frames had no usable mask",
              file=sys.stderr)
    if not rows:
        return None
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True)
    ap.add_argument("--ch", required=True)
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--frame-max", type=int, default=None,
                    help="only process frames <= this (e.g. 2200 for phase1 pilot)")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default results/260517/recomputed_axes/<pos>_<ch>.csv)")
    args = ap.parse_args()

    out_df = recompute_channel(args.pos, args.ch, args.modes, args.frame_max)
    if out_df is None:
        print("no output produced", file=sys.stderr)
        sys.exit(1)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = results_dir() / "recomputed_axes" / f"{args.pos}_{args.ch}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"wrote {len(out_df)} rows ({out_df['frame'].nunique()} frames x "
          f"{out_df['mode'].nunique()} modes) -> {out_path}")


if __name__ == "__main__":
    main()
