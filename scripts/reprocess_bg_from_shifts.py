"""Replay saved grid alignment with outside-channel BG correction, then rebuild CSVs.

Use a separate dataset YAML/output tree. No raw reconstruction or alignment is rerun.
Missing inputs and insufficient BG are errors; an uncorrected crop is never segmented.
Per-Pos manifests bind resumable outputs to the source metadata and correction code.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import tifffile
import yaml

from seg_omnipose import load_channel_filter, _ensure_cuda_dlls_on_path

_ensure_cuda_dlls_on_path()
import grid_subtract as gs

REPO = Path(__file__).resolve().parent.parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def frame_entries(shifts, frame_min, frame_max, allow_missing=False):
    entries = shifts.get("frame_results") or shifts.get("alignment_results") or []
    result = {}
    for entry in entries:
        if entry is None:
            continue
        frame = int(entry["frame_index"])
        if frame_min <= frame <= frame_max:
            if frame in result:
                raise ValueError(f"Duplicate frame_index {frame}")
            result[frame] = entry
    expected = set(range(frame_min, frame_max + 1))
    if result.keys() != expected and not allow_missing:
        raise ValueError(f"Missing alignment frames: {sorted(expected - result.keys())[:20]}")
    return [result[f] for f in sorted(result)]


def safe_output(output, inputs):
    output = Path(output).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"Output must be separate from input: {output} / {source}")
    return output


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def run_pos(cfg, pos, selected=None, plan=False, max_frames=None):
    bg, paths = cfg["background_replay"], cfg["paths"]
    source = resolve_path(bg["source_crop_root"]) / f"Pos{pos}" / "output_phase/channels"
    tl_root, grid_root = resolve_path(bg["tl_root"]), resolve_path(bg["grid_dir"])
    root = safe_output(resolve_path(paths["raw_root"]),
                       [resolve_path(bg["source_crop_root"]), tl_root, grid_root])
    shifts_path, rois_path = source / "pos_shifts_cal_online.json", source / "channel_rois.json"
    shifts, rois = read_json(shifts_path), read_json(rois_path)
    ids = [i for i in range(len(rois)) if selected is None or (pos, f"ch{i:02d}") in selected]
    if not ids:
        return None
    if selected is not None:
        unknown = {ch for p, ch in selected if p == pos} - {f"ch{i:02d}" for i in ids}
        if unknown:
            raise ValueError(f"Pos{pos}: selected channels absent from ROIs: {unknown}")
    entries = frame_entries(shifts, int(bg["frame_min"]), int(bg["frame_max"]), allow_missing=True)
    missing_alignment = sorted(set(range(int(bg["frame_min"]), int(bg["frame_max"]) + 1))
                               - {int(e["frame_index"]) for e in entries})
    # Existing online crops were never produced for failed alignment frames. Preserve these
    # physical frame gaps, but refuse to drop a frame that previously had a valid crop.
    for frame in missing_alignment:
        for i in ids:
            original = source / "crop_sub_rawraw/z000" / f"ch{i:02d}" / f"img_{frame:09d}_ph_000.tif"
            if original.is_file():
                raise ValueError(f"Crop exists but alignment is missing: {original}")
    if missing_alignment:
        print(f"  Pos{pos}: preserving {len(missing_alignment)} original missing-alignment frames", flush=True)
    if not entries:
        raise ValueError(f"Pos{pos}: no aligned frames")
    if max_frames:
        entries = entries[:max_frames]
    if not shifts.get("use_raw_phase", False) or shifts.get("apply_inverse_shift", False):
        raise ValueError(f"Pos{pos}: unsupported source alignment mode")
    grid_z = int(bg["grid_z"])
    if int(shifts["grid_z_index"]) != grid_z or int(shifts["tl_z_index"]) != 0:
        raise ValueError(f"Pos{pos}: source z indices do not match replay")
    if Path(shifts["grid_dir"]).resolve() != grid_root.resolve():
        raise ValueError(f"Pos{pos}: grid directory differs from the recorded alignment")
    cal_path = grid_root / f"grid_calibration_Pos{pos}.json"
    cal = gs.load_grid_calibration(str(cal_path))
    tl_dir = tl_root / f"Pos{pos}" / "z000/output_phase_raw"
    available = {p.name: p for p in tl_dir.iterdir() if p.suffix == ".tif"}
    needed = [f"img_{e['frame_index']:09d}_ph_000_phase.tif" for e in entries]
    missing = [name for name in needed if name not in available]
    if missing:
        raise FileNotFoundError(f"Pos{pos}: {len(missing)} missing raw phase frames: {missing[:5]}")
    grid_files = {}
    for key in {(int(e["grid_xi"]), int(e["grid_yi"])) for e in entries}:
        if key not in cal:
            raise ValueError(f"Pos{pos}: no calibration for grid node {key}")
        node = grid_root / f"Pos{pos}_x{key[0]:+d}_y{key[1]:+d}"
        grid_files[key] = tuple(node / part / f"img_000000000_ph_{grid_z:03d}_phase.tif"
                                for part in ("output_phase_raw", "output_phase"))
        for path in grid_files[key]:
            if not path.is_file():
                raise FileNotFoundError(path)
    out = root / f"Pos{pos}" / paths["channel_rel"]
    metadata_files = [shifts_path, rois_path, cal_path, Path(__file__),
                      REPO / "scripts/grid_subtract.py", REPO / "scripts/ecc_utils.py"]
    signature = {
        # Exclusion is a selection policy, not a pixel-processing parameter. Keep it out of
        # per-Pos fingerprints so a resumed run can reuse Pos1-5 outputs after dropping a bad ch.
        "background": {k: v for k, v in bg.items() if k != "exclude_channels"},
        "channels": ids, "frames": [e["frame_index"] for e in entries],
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in metadata_files},
        "phase_files": {str(available[n]): [available[n].stat().st_size, available[n].stat().st_mtime_ns]
                        for n in needed},
        "grid_files": {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                       for pair in grid_files.values() for p in pair},
        "out_root": str(root), "channel_rel": paths["channel_rel"],
    }
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    print(f"Pos{pos}: {len(entries)} frames x {len(ids)} channels -> {out}", flush=True)
    if plan:
        return {"pos": pos, "frames": len(entries), "channels": len(ids)}
    manifest = out / "bg_replay.json"
    if manifest.exists():
        prior = read_json(manifest)
        if prior["fingerprint"] != fingerprint:
            # The replay driver may gain resume-only safeguards between runs. If this Pos is
            # already complete and every expected selected channel has all frame files, retain
            # the existing pixel outputs and refresh the provenance fingerprint.
            complete_files = all(
                len(list((out / f"ch{i:02d}").glob("img_*_ph_000.tif"))) >= len(entries)
                for i in ids
            )
            if not prior.get("complete") or not complete_files:
                raise ValueError(f"Pos{pos}: inputs changed; use a new output tree")
            print(f"  Pos{pos}: complete output found; refreshing replay fingerprint", flush=True)
            prior["fingerprint"] = fingerprint
            prior["signature"] = signature
            write_json(manifest, prior)
            return prior
    elif out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing to reuse an unmarked output tree: {out}")
    report = {"fingerprint": fingerprint, "signature": signature, "complete": False,
              "original_missing_alignment_frames": missing_alignment,
              "n_frames": len(entries), "n_channels": len(ids), "errors": []}
    write_json(manifest, report)
    for i in ids:
        (out / f"ch{i:02d}").mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=121)
    def grid_pair(xi, yi):
        images = tuple(tifffile.imread(p).astype(np.float64) for p in grid_files[xi, yi])
        if any(not np.isfinite(image).all() for image in images):
            raise ValueError(f"Nonfinite grid phase: Pos{pos} node {(xi, yi)}")
        return images

    def process(entry):
        frame = int(entry["frame_index"])
        outputs = [out / f"ch{i:02d}" / f"img_{frame:09d}_ph_000.tif" for i in ids]
        if all(p.is_file() for p in outputs):
            return []
        tl = tifffile.imread(available[f"img_{frame:09d}_ph_000_phase.tif"]).astype(np.float64)
        if not np.isfinite(tl).all():
            raise ValueError(f"Pos{pos} frame {frame}: nonfinite raw phase")
        xi, yi = int(entry["grid_xi"]), int(entry["grid_yi"])
        grid, mask = grid_pair(xi, yi)
        if not (tl.shape == grid.shape == mask.shape):
            raise ValueError(f"Pos{pos} frame {frame}: mismatched phase shapes")
        crops, _ = gs.process_single_frame(
            tl, entry["shift_x_avg"], entry["shift_y_avg"], [rois[i] for i in ids],
            *cal[xi, yi], entry["residual_x_px"], entry["residual_y_px"], grid,
            output_crop_h_override=int(bg["out_h"]), tilt_crop_h_raw=int(bg["tilt_h"]),
            apply_subpixel_correction=bool(shifts["apply_subpixel_correction"]),
            fit_right=pos >= int(bg["pos_split"]), bg_method="outside_quad", ch_mask_img=mask)
        errors = []
        for i, crop, path in zip(ids, crops, outputs):
            if crop is None:
                errors.append({"pos": pos, "ch": f"ch{i:02d}", "frame": frame,
                               "reason": "insufficient_background"})
                continue
            if crop.shape != (rois[i]["crop_w"], int(bg["out_h"])) or not np.isfinite(crop).all():
                raise ValueError(f"Invalid output: {path}")
            temp = path.with_suffix(".tif.tmp")
            tifffile.imwrite(temp, crop)
            temp.replace(path)
        return errors

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=int(bg.get("workers", 2))) as pool:
        for count, errors in enumerate(pool.map(process, entries), 1):
            report["errors"].extend(errors)
            if count % 100 == 0:
                print(f"  Pos{pos}: {count}/{len(entries)}, {time.monotonic()-t0:.1f}s", flush=True)
    report["seconds"] = time.monotonic() - t0
    report["complete"] = not report["errors"]
    write_json(manifest, report)
    if report["errors"]:
        raise ValueError(f"Pos{pos}: {len(report['errors'])} crops lack enough BG; see {manifest}")
    print(f"  Pos{pos}: BG complete ({report['seconds']:.1f}s)", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("yaml", type=Path)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--max-frames", type=int, help="BG smoke test; use a separate output YAML")
    ap.add_argument("--through-csv", action="store_true")
    a = ap.parse_args()
    if a.max_frames is not None and (a.max_frames < 1 or a.through_csv):
        ap.error("--max-frames must be positive and cannot be combined with --through-csv")
    cfg = yaml.safe_load(a.yaml.read_text(encoding="utf-8"))
    bg = cfg["background_replay"]
    selected = load_channel_filter(resolve_path(bg["channels_file"])) if bg.get("channels_file") else None
    excluded = {(int(p), str(ch)) for p, ch in (bg.get("exclude_channels") or [])}
    if selected is not None:
        selected -= excluded
    if excluded:
        print(f"excluded channels: {sorted(excluded)}", flush=True)
    start, end = a.start or cfg["positions"]["start"], a.end or cfg["positions"]["end"]
    positions = [p for p in range(start, end + 1) if selected is None or any(n == p for n, _ in selected)]
    if not positions:
        ap.error("no selected positions")
    if (gs.CH_MASK_THRESH, gs.CH_MASK_DILATE, gs.CH_MASK_MIN_BG, gs.VALID_ERODE_PX) != (-1.0, 2, 500, 1):
        raise ValueError("Production BG parameters changed; review replay configuration")
    import cv2
    cv2.setNumThreads(1)
    for pos in positions:
        run_pos(cfg, pos, selected, a.plan, a.max_frames)
        if a.through_csv and not a.plan:
            cmd = [sys.executable, "-u", str(REPO / "scripts/run_dataset_pipeline.py"), str(a.yaml.resolve()),
                   "--stages", "seg,track", "--start", str(pos), "--end", str(pos)]
            if bg.get("channels_file"):
                cmd += ["--channels-file", str(resolve_path(bg["channels_file"]))]
            subprocess.run(cmd, check=True)
    if a.through_csv and not a.plan:
        cmd = [sys.executable, "-u", str(REPO / "scripts/run_dataset_pipeline.py"), str(a.yaml.resolve()),
               "--stages", "qc,consolidate"]
        if bg.get("channels_file"):
            cmd += ["--channels-file", str(resolve_path(bg["channels_file"]))]
        subprocess.run(cmd, check=True)
        consolidated = resolve_path(cfg["paths"]["consolidated_dir"])
        import pandas as pd
        index = pd.read_csv(consolidated / "channel_index.csv")
        actual = {(int(str(row.pos).removeprefix("Pos")), row.ch) for row in index.itertuples()}
        expected = {(p, ch) for p, ch in selected if start <= p <= end} if selected is not None else actual
        if actual != expected:
            raise ValueError(f"CSV channel mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
        subprocess.run([
            sys.executable, "-u", str(REPO / "scripts/lineage_html_gallery_260517.py"),
            "--source", "csv", "--csv", str(consolidated / "all_cells_lineage_data3D.csv.gz"),
            "--pos-min", str(start), "--pos-max", str(end), "--max-lineages", str(len(expected)),
            "--min-mother-frames", "1", "--include-edge-channels", "--end-frame", str(bg["frame_max"]),
            "--dataset-label", cfg["dataset"], "--ylim-ri", "auto", "--ylim-mass", "auto", "--ylim-vol", "auto",
            "--out", str(consolidated.parent / "_qc/lineage_html")], check=True)


if __name__ == "__main__":
    main()
