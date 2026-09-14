"""build_phase1_dataset_260517.py - the publication dataset: first 7 days in 2% glucose.

Derives, from the frozen master tracking data, a self-contained package that
covers img_0002..img_2017 (2016 frames = 168.0 h at 5 min/frame, time 0 = img_2).
img_2018 is excluded on purpose: the medium change scheduled at img_2019 already
perturbs the phase at img_2018 (see figure fig_switch_frame_check_260517,
2026-09-10), so img_2017 is the last unperturbed 2% frame.

No re-tracking: the tracker links frames causally, so per-frame rows are taken
verbatim from the master. Only the per-cell summaries are recomputed inside the
window, with explicit censoring flags (cells present at the window start, cells
still alive at the window end).

Package layout (<out_dir>/):
    README.md                      what / how / how to load (English)
    SCHEMA.md                      column definitions of every table
    MANIFEST.json                  provenance (master tag, window, counts, boundary check)
    SHA256SUMS.txt
    parameters.json                optical / tracking parameters used
    cells_frames.csv.gz            every tracked cell x frame in the window (from the master long table)
    cells.csv.gz                   one row per cell, recomputed inside the window (+ censoring flags)
    divisions.csv.gz               one row per division event inside the window
    channels.csv                   one row per channel: mother coverage, QC and classification flags
    excluded_frames.csv            drift bad frames per position inside the window (excluded from tracking)
    bad_frame_measurements.csv.gz  raw measurements on those excluded frames (no tracking, no RI)

Usage:
    python scripts/build_phase1_dataset_260517.py                     # from the LATEST master
    python scripts/build_phase1_dataset_260517.py --from-working-tree --out <dir>   # test on D:\\260517_seg
    python scripts/build_phase1_dataset_260517.py --end-frame 2018    # alternative window
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
import _retrack_260517_newmodel as chain  # noqa: E402

FIRST_FRAME = int(chain.FRAME_MIN)        # 2 : time 0 of the master
END_FRAME_DEFAULT = 2017                  # last unperturbed 2% frame
DT_MIN = float(chain.DT_MIN)
EXPECTED_MEDIUM = "wo_2"
VALUE_COLS = ["volume_um3_rod", "volume_um3_efd", "long_axis_um", "short_axis_um",
              "mean_ri", "mass_pg", "density_pg_um3", "mean_ri_efd", "mass_pg_efd", "density_pg_um3_efd"]
MEAN_COLS = ["volume_um3_rod", "volume_um3_efd", "mean_ri", "mass_pg", "density_pg_um3",
             "mean_ri_efd", "mass_pg_efd", "density_pg_um3_efd"]
OTHER_SESSION_SCRATCH = Path(r"C:\TEMP\claude\C--Users-QPI\945bff36-8800-4b3a-911b-252a0b4214fa\scratchpad")
EDGE_CHANNELS = ("ch00", "ch11")   # decided 2026-09-14: edge traps excluded from the analysis cohort


def dataset_name(end_frame: int = END_FRAME_DEFAULT) -> str:
    return f"phase1_img{FIRST_FRAME:04d}-{end_frame:04d}"


def _t_h(frame) -> float:
    return (np.asarray(frame, dtype=float) - FIRST_FRAME) * DT_MIN / 60.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_window(long_csv: Path, end_frame: int, log):
    """Chunk-read the master long table; keep window rows + mother rows around the boundary."""
    win_parts, bnd_parts = [], []
    n_total = 0
    t0 = time.time()
    for chunk in pd.read_csv(long_csv, chunksize=500_000):
        n_total += len(chunk)
        f = chunk["frame"].to_numpy()
        win_parts.append(chunk[(f >= FIRST_FRAME) & (f <= end_frame)])
        b = chunk[(chunk["cell_id"] == 0) & (f >= end_frame - 2) & (f <= end_frame + 4)]
        if len(b):
            bnd_parts.append(b[["pos", "ch", "frame", "total_phase", "mean_ri", "is_outlier", "touches_border"]])
    win = pd.concat(win_parts, ignore_index=True)
    bnd = pd.concat(bnd_parts, ignore_index=True) if bnd_parts else pd.DataFrame()
    log(f"read {n_total} master rows -> {len(win)} rows in window img_{FIRST_FRAME}..img_{end_frame} ({time.time() - t0:.0f}s)")
    return win, bnd


def _cells_table(win: pd.DataFrame, end_frame: int) -> pd.DataFrame:
    hidden = win["is_outlier"].astype(bool) | win["touches_border"].astype(bool)
    w = win.assign(_hidden=hidden).sort_values(["pos", "ch", "cell_id", "frame"])
    key = ["pos", "ch", "cell_id"]
    base = w.groupby(key, sort=False).agg(
        channel_id=("channel_id", "first"), parent_id=("parent_id", "first"), in_tree=("in_tree", "first"),
        birth_frame=("birth_frame", "first"), first_frame=("frame", "min"), last_frame=("frame", "max"),
        n_frames=("frame", "size"), n_hidden=("_hidden", "sum"), n_outlier=("is_outlier", "sum"),
        rank_first=("rank", "first"), rank_last=("rank", "last"),
    ).reset_index()
    valid = w[~w["_hidden"]]
    firsts = valid.groupby(key, sort=False)[VALUE_COLS].first().add_suffix("_first_valid")
    lasts = valid.groupby(key, sort=False)[VALUE_COLS].last().add_suffix("_last_valid")
    means = valid.groupby(key, sort=False)[MEAN_COLS].mean().add_prefix("mean_")
    n_valid = valid.groupby(key, sort=False).size().rename("n_valid_frames")
    cells = base.merge(firsts, on=key, how="left").merge(lasts, on=key, how="left") \
                .merge(means, on=key, how="left").merge(n_valid, on=key, how="left")
    cells["n_valid_frames"] = cells["n_valid_frames"].fillna(0).astype(int)
    cells["present_at_window_start"] = cells["first_frame"] == FIRST_FRAME
    cells["birth_observed"] = cells["birth_frame"] >= FIRST_FRAME
    cells["alive_at_window_end"] = cells["last_frame"] == end_frame      # right-censored
    cells["observed_span_h"] = _t_h(cells["last_frame"]) - _t_h(cells["first_frame"])
    cells["first_time_h"] = _t_h(cells["first_frame"])
    cells["last_time_h"] = _t_h(cells["last_frame"])
    # daughters born inside the window
    born = cells[(cells["parent_id"] >= 0) & (cells["birth_frame"] >= FIRST_FRAME)]
    nd = born.groupby(["pos", "ch", "parent_id"]).size().rename("n_daughters_in_window").reset_index() \
             .rename(columns={"parent_id": "cell_id"})
    cells = cells.merge(nd, on=key, how="left")
    cells["n_daughters_in_window"] = cells["n_daughters_in_window"].fillna(0).astype(int)
    # generation = number of ancestors reachable inside the window (mother = 0)
    gen = np.zeros(len(cells), dtype=int)
    for (pos, ch), g in cells.groupby(["pos", "ch"], sort=False):
        parent = dict(zip(g["cell_id"].to_numpy(), g["parent_id"].to_numpy()))
        for idx, cid in zip(g.index, g["cell_id"].to_numpy()):
            depth, cur, seen = 0, cid, set()
            while cur in parent and parent[cur] >= 0 and parent[cur] in parent and cur not in seen:
                seen.add(cur); cur = parent[cur]; depth += 1
            gen[cells.index.get_loc(idx)] = depth
    cells["generation_in_window"] = gen
    order = ["pos", "ch", "channel_id", "cell_id", "parent_id", "in_tree", "generation_in_window",
             "birth_frame", "birth_observed", "present_at_window_start", "alive_at_window_end",
             "first_frame", "last_frame", "first_time_h", "last_time_h", "observed_span_h",
             "n_frames", "n_valid_frames", "n_hidden", "n_outlier", "n_daughters_in_window",
             "rank_first", "rank_last"]
    rest = [c for c in cells.columns if c not in order]
    return cells[order + rest]


def _divisions_table(win: pd.DataFrame, cells: pd.DataFrame, qc: pd.DataFrame | None = None) -> pd.DataFrame:
    d = cells[(cells["parent_id"] >= 0) & (cells["birth_frame"] >= FIRST_FRAME)][
        ["pos", "ch", "channel_id", "parent_id", "cell_id", "birth_frame", "in_tree"]].rename(
        columns={"cell_id": "daughter_id", "birth_frame": "frame"})
    d["time_h"] = _t_h(d["frame"])
    hidden = win["is_outlier"].astype(bool) | win["touches_border"].astype(bool)
    vol = win.loc[~hidden, ["pos", "ch", "cell_id", "frame", "volume_um3_rod"]]
    vol = vol.set_index(["pos", "ch", "cell_id", "frame"])["volume_um3_rod"]

    def _lookup(pos, ch, cid, frame):
        try:
            return float(vol.loc[(pos, ch, cid, frame)])
        except KeyError:
            return np.nan
    d["parent_volume_before_um3"] = [_lookup(p, c, pid, f - 1) for p, c, pid, f in
                                     zip(d["pos"], d["ch"], d["parent_id"], d["frame"])]
    d["parent_volume_after_um3"] = [_lookup(p, c, pid, f) for p, c, pid, f in
                                    zip(d["pos"], d["ch"], d["parent_id"], d["frame"])]
    d["daughter_birth_volume_um3"] = [_lookup(p, c, did, f) for p, c, did, f in
                                      zip(d["pos"], d["ch"], d["daughter_id"], d["frame"])]
    d["is_mother_division"] = d["parent_id"] == 0
    # division QC (division_qc_260517.py): mass / volume validation of every candidate
    qcols = ["validated", "method", "outlier_near", "n_pre", "n_post", "mass_ratio", "volume_ratio", "reason"]
    if qc is not None and len(qc):
        q = qc[["pos", "ch", "parent_id", "daughter_id"] + qcols].drop_duplicates(["pos", "ch", "parent_id", "daughter_id"])
        d = d.merge(q, on=["pos", "ch", "parent_id", "daughter_id"], how="left")
        d["validated"] = d["validated"].astype("boolean").fillna(False).astype(bool)
        d["method"] = d["method"].fillna("not_evaluated")
    else:
        d["validated"] = False
        d["method"] = "not_evaluated"
        for c in qcols[2:]:
            d[c] = np.nan
    return d.sort_values(["pos", "ch", "frame", "daughter_id"]).reset_index(drop=True)


def _excluded_frames(bad_json: Path | None, end_frame: int) -> pd.DataFrame:
    rows = []
    if bad_json is not None and bad_json.exists():
        j = json.loads(bad_json.read_text(encoding="utf-8"))
        for pos, entry in j.items():
            if not (isinstance(entry, dict) and "bad_timepoints" in entry):
                continue
            for f, reasons in entry["bad_timepoints"].items():
                f = int(f)
                if FIRST_FRAME <= f <= end_frame:
                    rows.append({"pos": pos, "frame": f, "time_h": _t_h(f),
                                 "reasons": ";".join(reasons) if isinstance(reasons, list) else str(reasons)})
    df = pd.DataFrame(rows, columns=["pos", "frame", "time_h", "reasons"])
    return df.sort_values(["pos", "frame"]).reset_index(drop=True) if len(df) else df


def _channels_table(win, cells, divisions, excluded, channel_index: Path | None, yaml_path: Path | None,
                    qc_dir: Path | None, end_frame: int) -> pd.DataFrame:
    n_window_frames = end_frame - FIRST_FRAME + 1
    key = ["pos", "ch"]
    ch = win.groupby(key).agg(channel_id=("channel_id", "first"), n_cells=("cell_id", "nunique"),
                              n_frames_with_cells=("frame", "nunique"), n_rows=("frame", "size")).reset_index()
    mcols = ["first_frame", "last_frame", "n_frames", "n_hidden", "alive_at_window_end", "n_daughters_in_window"]
    if "n_daughters_validated_in_window" in cells.columns:
        mcols.append("n_daughters_validated_in_window")
    mother = cells[cells["cell_id"] == 0][key + mcols]
    mother = mother.rename(columns={"first_frame": "mother_first_frame", "last_frame": "mother_last_frame",
                                    "n_frames": "mother_frames", "n_hidden": "mother_hidden_frames",
                                    "alive_at_window_end": "mother_alive_at_window_end",
                                    "n_daughters_in_window": "mother_divisions",
                                    "n_daughters_validated_in_window": "mother_divisions_validated"})
    ch = ch.merge(mother, on=key, how="left")
    ch["mother_present"] = ch["mother_frames"].notna()
    nbad = excluded.groupby("pos").size().rename("n_excluded_frames_pos") if len(excluded) else pd.Series(dtype=int, name="n_excluded_frames_pos")
    ch = ch.merge(nbad, left_on="pos", right_index=True, how="left")
    ch["n_excluded_frames_pos"] = ch["n_excluded_frames_pos"].fillna(0).astype(int)
    ch["mother_coverage"] = ch["mother_frames"] / (n_window_frames - ch["n_excluded_frames_pos"])
    if channel_index is not None and channel_index.exists():
        ci = pd.read_csv(channel_index)[["pos", "ch", "n_masks"]].rename(columns={"n_masks": "n_masks_full_run"})
        ch = ci.merge(ch, on=key, how="left")            # keep channels with masks but no window cells
        ch["channel_id"] = ch["pos"] + "_" + ch["ch"]
        for c in ("n_cells", "n_frames_with_cells", "n_rows"):
            ch[c] = ch[c].fillna(0).astype(int)
        ch["mother_present"] = ch["mother_present"].astype("boolean").fillna(False).astype(bool)
    # curated classification (docs/channel_classification_260517.yaml)
    ch["classification_status"] = None
    ch["classification_phase1_outcome"] = None
    if yaml_path is not None and yaml_path.exists():
        import yaml
        y = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        for pos, chans in (y.get("positions") or {}).items():
            for cname, fields in (chans or {}).items():
                if not fields:
                    continue
                sel = (ch["pos"] == pos) & (ch["ch"] == cname)
                ch.loc[sel, "classification_status"] = fields.get("status")
                ch.loc[sel, "classification_phase1_outcome"] = (fields.get("phase1") or {}).get("outcome")
    # masks-only QC pass (crop out-of-bounds channels)
    ch["qc_oob_excluded"] = False
    for d in (qc_dir, OTHER_SESSION_SCRATCH):
        p = (d / "short_oob_exclude.json") if d is not None else None
        if p is not None and p.exists():
            ex = set(json.loads(p.read_text(encoding="utf-8")).get("excluded", []))
            ch["qc_oob_excluded"] = [f"{p_} {c_}" in ex for p_, c_ in zip(ch["pos"], ch["ch"])]
            break
    # edge traps of the crop (ch00 / ch11) carry vignetting / edge artefacts: excluded from the analysis cohort
    ch["edge_channel"] = ch["ch"].isin(EDGE_CHANNELS)
    ch["analysis_recommended"] = ((ch["classification_status"] == "cells") & (~ch["qc_oob_excluded"].astype(bool))
                                  & (~ch["edge_channel"]) & ch["mother_present"].astype(bool))
    ch["pos_num"] = ch["pos"].str[3:].astype(int)
    return ch.sort_values(["pos_num", "ch"]).drop(columns="pos_num").reset_index(drop=True)


def _boundary_check(bnd: pd.DataFrame, end_frame: int) -> dict:
    if bnd.empty:
        return {}
    ok = bnd[~(bnd["is_outlier"].astype(bool) | bnd["touches_border"].astype(bool))]
    g = ok.groupby("frame").agg(median_total_phase=("total_phase", "median"),
                                median_mean_ri=("mean_ri", "median"), n_mothers=("frame", "size"))
    ref = g["median_total_phase"].get(end_frame, np.nan)
    out = {}
    for f, r in g.iterrows():
        out[str(int(f))] = {"median_total_phase": round(float(r.median_total_phase), 3),
                            "total_phase_vs_end_frame_%": round((float(r.median_total_phase) / ref - 1) * 100, 2) if ref else None,
                            "median_mean_ri": round(float(r.median_mean_ri), 5), "n_mothers": int(r.n_mothers)}
    return out


def _schema_md(end_frame: int) -> str:
    return f"""# Schema - 260517 phase-1 dataset (img_{FIRST_FRAME:04d}..img_{end_frame:04d})

Frame numbers are absolute acquisition indices (img_NNN). `time_h = (frame - {FIRST_FRAME}) * {DT_MIN:g} / 60`
(time 0 = img_{FIRST_FRAME}). Cells are identified by (pos, ch, cell_id); `channel_id = pos + "_" + ch`.
`cell_id 0` is the central (mother) cell of the trap; daughters carry `parent_id`.

## cells_frames.csv.gz (one row = one cell in one frame)
| column | meaning |
|---|---|
| pos, ch, channel_id | position and trap channel |
| cell_id, parent_id, in_tree | cell identity; parent_id -1 = no parent observed; in_tree = descendant of the mother |
| birth_frame, death_frame | first / last frame of the cell in the FULL tracking (death_frame may exceed the window; use cells.csv flags for censoring) |
| frame, time_h | absolute frame and time |
| rank | 1 = closest to the trap centre (mother), 2, 3, ... outward |
| mask_label | label value of this cell in the frame's segmentation mask (masks not included in this package) |
| area_px, area_um2 | mask area |
| long_axis_um, short_axis_um | from the smoothed cell contour ("yellow" contour: elliptic-Fourier K=6 smoothing of the mask boundary, shrunk 0.5 px inward). long = arc length of the centerline after one midpoint update; short = mean chord width over the central body (chords perpendicular to the centerline, caps excluded, widths >= 50% of max) |
| centroid_x_px, centroid_y_px | centroid in the channel crop |
| total_phase | integrated phase over the mask (rad * px) |
| volume_um3_rod | capsule (rod) volume from the yellow-contour axes: (4/3) pi r^3 + pi r^2 (L - 2r), r = short/2 |
| volume_um3_efd | adopted volume: solid of revolution of the yellow-contour chords, sum pi (w/2)^2 ds along the updated centerline |
| mean_ri | mean refractive index = n_medium_used + total_phase * lambda * A_px / (2 pi V_rod) |
| mass_pg | dry mass = (mean_ri - n_milliq_used) / alpha_ri * V_rod * 1e-3 |
| density_pg_um3 | dry-mass density = mass_pg / volume_um3_rod |
| mean_ri_efd, mass_pg_efd, density_pg_um3_efd | the same three quantities computed with volume_um3_efd |
| n_medium_used, medium_name, n_milliq_used | medium RI, medium label (all `wo_2` here), protein baseline RI |
| is_outlier | frame failed the tracker's continuation/division area rules (3-frame rule); physics columns are NaN |
| touches_border | mask touches the crop border; physics columns are NaN |

## cells.csv.gz (one row = one cell, computed inside the window only)
| column | meaning |
|---|---|
| generation_in_window | number of ancestors reachable inside the window (mother = 0) |
| birth_observed | birth_frame >= {FIRST_FRAME} (False: cell already present at window start) |
| present_at_window_start | first observed frame == img_{FIRST_FRAME} (left-censored history) |
| alive_at_window_end | last observed frame == img_{end_frame} (right-censored: fate unknown inside the window) |
| first_frame, last_frame, first_time_h, last_time_h, observed_span_h | observed extent inside the window |
| n_frames, n_valid_frames, n_hidden, n_outlier | rows, rows with valid physics, hidden rows (outlier or border), outlier rows |
| n_daughters_in_window | daughters born inside the window |
| rank_first, rank_last | trap rank at first / last frame |
| <value>_first_valid, <value>_last_valid | volume / axes / RI / mass / density at the first and last valid frame |
| mean_<value> | mean over valid frames |

## divisions.csv.gz (one row = one candidate division event inside the window)
parent_id, daughter_id, frame, time_h, parent_volume_before_um3 (frame - 1), parent_volume_after_um3,
daughter_birth_volume_um3 (rod volumes), is_mother_division (parent is cell 0), in_tree.

The tracker calls a division from a single frame's areas, so transient segmentation splits
produce spurious daughters. Every candidate is therefore re-examined (division_qc_260517.py)
with the parent's mass_pg_efd / volume_um3_efd before and after the event:
`validated` (bool) is the flag to use; `method` = direct (no tracker outlier within +-1 frame),
rescued (outlier nearby but post/pre mass in 0.25-0.78, post/pre volume in 0.25-0.85 and the
two ratios within 0.25 of each other, medians of up to 3 valid points within +-8 frames),
rejected (ratios outside those limits), insufficient (< 2 valid points on a side),
duplicate (rescued candidate within 1 h of an already validated event of the same parent).
`mass_ratio`, `volume_ratio`, `n_pre`, `n_post`, `outlier_near`, `reason` document the decision.
Outlier frames are excluded from mass / growth fits but a cell cycle is not discarded for them.
`cells.csv.gz: n_daughters_validated_in_window` and `channels.csv: mother_divisions_validated`
count validated events only.

## channels.csv (one row = one trap channel)
n_masks_full_run (frames with a segmentation mask in the full run), n_cells, n_frames_with_cells, n_rows,
mother_present, mother_first_frame, mother_last_frame, mother_frames, mother_hidden_frames,
mother_alive_at_window_end, mother_divisions, n_excluded_frames_pos, mother_coverage
(= mother_frames / (window frames - excluded frames of the position)), classification_status and
classification_phase1_outcome (curated by eye: cells / empty / dead_at_start / unused; alive / dead),
qc_oob_excluded (crop out-of-bounds detected in a masks-only QC pass; recommended to exclude),
edge_channel (ch00 / ch11, the outermost traps of the crop: vignetting and edge artefacts; excluded from
the analysis cohort since 2026-09-14), analysis_recommended (= status cells AND mother present AND NOT
qc_oob_excluded AND NOT edge_channel: the cohort every analysis should start from).

## excluded_frames.csv
Drift bad frames (stage drift / registration failure) per position; excluded before tracking.
`bad_frame_measurements.csv.gz` holds the raw mask measurements on those frames (no tracking, no RI).
"""


def _readme_md(end_frame: int, manifest: dict) -> str:
    c = manifest["counts"]
    return f"""# 260517 phase-1 dataset: fission yeast growth in 2% glucose, first 7 days

Single-cell time series from a quantitative phase imaging (QPI) mother-machine
experiment (Schizosaccharomyces pombe, 104 positions x up to 12 trap channels,
5 min/frame). This package contains the first 7 days in 2% glucose only:
absolute frames img_{FIRST_FRAME:04d}..img_{end_frame:04d} ({end_frame - FIRST_FRAME + 1} frames, {_t_h(end_frame):.1f} h),
time 0 = img_{FIRST_FRAME}. The medium was switched to 0.0055% glucose at img_2019; img_2018
already shows the optical signature of the change and is therefore excluded.

Derived from master tracking dataset `{manifest['derived_from']}` (segmentation: Omnipose
`{manifest['parameters'].get('model', '')}`; tracking: central_cell_lineage_tracker.py with
mask-direct medial-axis morphology). Per-frame rows are verbatim; per-cell summaries are
recomputed inside the window with explicit censoring flags. Built {manifest['built']}.

| table | rows |
|---|---|
| cells_frames.csv.gz | {c['cells_frames']} |
| cells.csv.gz | {c['cells']} |
| divisions.csv.gz | {c['divisions']} |
| channels.csv | {c['channels']} |
| excluded_frames.csv | {c['excluded_frames']} |
| bad_frame_measurements.csv.gz | {c['bad_frame_measurements']} |

## Load

```python
import pandas as pd
frames = pd.read_csv("cells_frames.csv.gz")
cells = pd.read_csv("cells.csv.gz")
chans = pd.read_csv("channels.csv")
good = chans[chans.analysis_recommended]     # cells, mother present, not OOB, not an edge trap (ch00/ch11)
mother = frames[(frames.cell_id == 0) & frames.channel_id.isin(good.channel_id)]
valid = mother[~(mother.is_outlier | mother.touches_border)]
```

Column definitions: `SCHEMA.md`. Parameters: `parameters.json`. Integrity: `sha256sum -c SHA256SUMS.txt`.
Segmentation masks and phase images are not part of this package (row -> mask via `frame` + `mask_label`).
"""


def build(consolidated_dir: Path, inputs_dir: Path | None, out_dir: Path, derived_from: str,
          qc_dir: Path | None = None, end_frame: int = END_FRAME_DEFAULT,
          yaml_path: Path | None = None, log=print) -> Path:
    t_all = time.time()
    long_csv = consolidated_dir / "all_cells_lineage_data3D.csv.gz"
    if not long_csv.exists():
        raise FileNotFoundError(long_csv)
    cons_manifest = json.loads((consolidated_dir / "manifest.json").read_text(encoding="utf-8")) \
        if (consolidated_dir / "manifest.json").exists() else {}
    yaml_path = yaml_path or (REPO / "docs" / "channel_classification_260517.yaml")
    bad_json = None
    for cand in ([inputs_dir / "bad_frames.json"] if inputs_dir else []) + [chain.BAD]:
        if cand is not None and Path(cand).exists():
            bad_json = Path(cand); break

    out_dir.mkdir(parents=True, exist_ok=True)
    win, bnd = _read_window(long_csv, end_frame, log)
    win.insert(2, "channel_id", win["pos"].astype(str) + "_" + win["ch"].astype(str))
    n_bad_medium = int((win["medium_name"] != EXPECTED_MEDIUM).sum())
    if n_bad_medium:
        log(f"WARN: {n_bad_medium} window rows have medium_name != {EXPECTED_MEDIUM}")

    cells = _cells_table(win, end_frame)
    qc_src = consolidated_dir / "all_cells_divisions_qc.csv.gz"
    qc = pd.read_csv(qc_src) if qc_src.exists() else None
    if qc is None:
        log("WARN: no all_cells_divisions_qc.csv.gz in the consolidated dir; divisions carry validated=False")
    divisions = _divisions_table(win, cells, qc)
    nv = divisions[divisions["validated"]].groupby(["pos", "ch", "parent_id"]).size() \
        .rename("n_daughters_validated_in_window").reset_index().rename(columns={"parent_id": "cell_id"})
    cells = cells.merge(nv, on=["pos", "ch", "cell_id"], how="left")
    cells["n_daughters_validated_in_window"] = cells["n_daughters_validated_in_window"].fillna(0).astype(int)
    excluded = _excluded_frames(bad_json, end_frame)
    channels = _channels_table(win, cells, divisions, excluded, consolidated_dir / "channel_index.csv",
                               yaml_path, qc_dir, end_frame)
    bad_meas_src = consolidated_dir / "all_cells_lineage_bad_frames.csv.gz"
    bad_meas = pd.read_csv(bad_meas_src) if bad_meas_src.exists() else pd.DataFrame()
    if len(bad_meas):
        bad_meas = bad_meas[(bad_meas["frame"] >= FIRST_FRAME) & (bad_meas["frame"] <= end_frame)]
        bad_meas.insert(2, "channel_id", bad_meas["pos"].astype(str) + "_" + bad_meas["ch"].astype(str))

    win.to_csv(out_dir / "cells_frames.csv.gz", index=False)
    cells.to_csv(out_dir / "cells.csv.gz", index=False)
    divisions.to_csv(out_dir / "divisions.csv.gz", index=False)
    channels.to_csv(out_dir / "channels.csv", index=False)
    excluded.to_csv(out_dir / "excluded_frames.csv", index=False)
    bad_meas.to_csv(out_dir / "bad_frame_measurements.csv.gz", index=False)

    n_med = sorted(win["n_medium_used"].dropna().unique().tolist())
    n_mq = sorted(win["n_milliq_used"].dropna().unique().tolist())
    params = {
        "frame_first": FIRST_FRAME, "frame_last": end_frame, "n_frames": end_frame - FIRST_FRAME + 1,
        "time_zero_frame": FIRST_FRAME, "time_interval_min": DT_MIN, "duration_h": float(_t_h(end_frame)),
        "medium": "2% glucose (wo_2)", "n_medium_used": n_med, "n_milliq_used": n_mq,
        "pixel_size_um": cons_manifest.get("pixel_size_um", float(chain.PIXEL_UM)),
        "wavelength_nm": cons_manifest.get("wavelength_nm", float(chain.WAVELENGTH_NM)),
        "alpha_ri_ml_per_g": cons_manifest.get("alpha_ri", float(chain.ALPHA_RI)),
        "model": Path(cons_manifest.get("model", chain.MODEL)).name,
        "media_schedule_full_experiment": cons_manifest.get("media_schedule", chain.MEDIA_SCHEDULE),
        "first_switch_scheduled_frame": 2019,
        "first_perturbed_frame_observed": 2018,
        "tracker": "central_cell_lineage_tracker.py (mask-direct medial axis; 3-frame outlier rule; drift bad frames excluded)",
    }
    (out_dir / "parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    manifest = {
        "dataset": dataset_name(end_frame), "derived_from": derived_from, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window": {"frame_first": FIRST_FRAME, "frame_last": end_frame, "n_frames": end_frame - FIRST_FRAME + 1},
        "counts": {"cells_frames": int(len(win)), "cells": int(len(cells)), "divisions": int(len(divisions)),
                   "divisions_validated": int(divisions["validated"].sum()),
                   "mother_divisions": int(divisions["is_mother_division"].sum()),
                   "mother_divisions_validated": int((divisions["is_mother_division"] & divisions["validated"]).sum()),
                   "channels": int(len(channels)), "channels_with_mother": int(channels["mother_present"].sum()),
                   "excluded_frames": int(len(excluded)), "bad_frame_measurements": int(len(bad_meas)),
                   "rows_with_unexpected_medium": n_bad_medium},
        "boundary_check_mother_median": _boundary_check(bnd, end_frame),
        "parameters": params,
        "source_consolidated_manifest": cons_manifest,
    }
    (out_dir / "SCHEMA.md").write_text(_schema_md(end_frame), encoding="utf-8")
    (out_dir / "README.md").write_text(_readme_md(end_frame, manifest), encoding="utf-8")
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"{_sha256(p)} *{p.name}" for p in sorted(out_dir.iterdir()) if p.is_file() and p.name != "SHA256SUMS.txt"]
    with (out_dir / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"phase-1 dataset built: {len(win)} rows / {len(cells)} cells / {len(divisions)} divisions / "
        f"{len(channels)} channels -> {out_dir} ({time.time() - t_all:.0f}s)")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--end-frame", type=int, default=END_FRAME_DEFAULT)
    ap.add_argument("--from-working-tree", action="store_true",
                    help="use D:\\260517_seg\\_lineage_consolidated instead of the master")
    ap.add_argument("--master-tag", default=None, help="master tag (default LATEST)")
    ap.add_argument("--out", default=None, help="output dir (default <master>/derived/<dataset>)")
    args = ap.parse_args()

    if args.from_working_tree:
        cons, inputs, qc, src = chain.CONSOLIDATED, None, None, "working-tree (D:\\260517_seg, unpublished)"
        if args.out is None:
            raise SystemExit("--out is required with --from-working-tree")
        out = Path(args.out)
    else:
        import qpi_paths as qp
        if args.master_tag:
            import os
            os.environ["QPI_LINEAGE_MASTER"] = args.master_tag
        md = qp.master_dir()
        if md is None:
            raise SystemExit("no master published; use --from-working-tree for a test build")
        cons, inputs, qc, src = md / "consolidated", md / "inputs", md / "qc", md.name
        out = Path(args.out) if args.out else md / "derived" / dataset_name(args.end_frame)
    build(cons, inputs, out, derived_from=src, qc_dir=qc, end_frame=args.end_frame)


if __name__ == "__main__":
    main()
