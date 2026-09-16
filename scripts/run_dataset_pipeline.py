# -*- coding: utf-8 -*-
"""run_dataset_pipeline.py - one entry point from phase crops to a frozen master dataset.

Everything dataset-specific lives in datasets/<id>.yaml (paths, medium switch frames, RI
calibration, drift bad frames, Omnipose checkpoint, tracker constants). This driver only
sequences the generic tools and records provenance:

    seg          scripts/seg_omnipose.py per Pos (GPU only). Channels with inference_out/_DONE are skipped.
    track        scripts/central_cell_lineage_tracker.py per channel that has masks and no production lineage.
                 production := lineage_run_params.json matches the yaml (media_schedule, frame_min) and
                 lineage_data3D.csv carries tracking.production_marker (volume_um3_efd).
    qc           division_qc_260517.run_lineage_dir on every production lineage -> divisions_qc.csv.
    consolidate  every production lineage -> <consolidated_dir>/all_cells_*.csv.gz, channel_index.csv, manifest.json.
    publish      freeze consolidated + per_channel + inputs + code -> <master_root>/<tag>/ (read-only,
                 SHA256SUMS.txt, MANIFEST.json, README.md, SCHEMA.md), update LATEST.txt, mirror without
                 per_channel, then run the yaml's `derived` hook (e.g. the phase-1 publication package).

Usage:
    python scripts/run_dataset_pipeline.py datasets/260517.yaml --plan
    python scripts/run_dataset_pipeline.py datasets/260517.yaml                        # seg,track,qc,consolidate
    python scripts/run_dataset_pipeline.py datasets/260517.yaml --stages seg,track --start 53 --end 60
    python scripts/run_dataset_pipeline.py datasets/260517.yaml --stages consolidate,publish --tag v20260915_yellow

seg and track are pipelined per Pos (Pos N is tracked while Pos N+1 segments). Every stage is
resumable: re-run the same command after a crash or a reboot. Nothing is written under raw_root.
Run it with the `omnipose` env python (environment/); subprocesses use the same interpreter.
Log: <mask_root>/_pipeline/<id>_pipeline.log.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import queue
import shutil
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
PY = sys.executable
STAGES = ("seg", "track", "qc", "consolidate", "publish")
PER_CHANNEL_FILES = ["lineage_data3D.csv", "clist.csv", "lineage_cells.json", "lineage_bad_frames.csv",
                     "bad_frames_used.json", "lineage_run_params.json", "divisions_qc.csv"]
CODE_FILES = ["run_dataset_pipeline.py", "seg_omnipose.py", "central_cell_lineage_tracker.py",
              "mask_volume_schematic.py", "mask_morphology.py", "ri_calibration.py",
              "division_qc_260517.py", "qpi_paths.py"]
_lock = threading.Lock()
_LOG: Path | None = None


def _path(s) -> Path:
    p = Path(str(s))
    return p if p.is_absolute() else (REPO / p).resolve()


class Dataset:
    """datasets/<id>.yaml, resolved."""

    def __init__(self, yaml_path: Path):
        self.yaml_path = Path(yaml_path).resolve()
        cfg = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8"))
        self.cfg = cfg
        self.id = str(cfg["dataset"])
        self.description = str(cfg.get("description") or "")
        p = cfg["paths"]
        self.raw_root = _path(p["raw_root"])
        self.mask_root = _path(p["mask_root"])
        self.rel = Path(str(p["channel_rel"]))
        self.phase_glob = str(p.get("phase_glob") or "img_*_ph_000_phase.tif")
        self.consolidated = (_path(p["consolidated_dir"]) if p.get("consolidated_dir")
                             else self.mask_root / "_lineage_consolidated")
        self.master_root = _path(p["master_root"])
        self.mirror_root = _path(p["mirror_root"]) if p.get("mirror_root") else None
        self.overrides = [(int(o["pos_min"]), int(o["pos_max"]), _path(o["root"]))
                          for o in (p.get("raw_root_overrides") or [])]
        pos = cfg.get("positions") or {}
        self.pos_start = int(pos.get("start", 1))
        self.pos_end = int(pos.get("end", 999))
        s = cfg["segmentation"]
        self.model = _path(s["model"])
        self.seg_eval = dict(s.get("eval") or {})
        self.gate = dict(s.get("gate") or {})
        self.seg_workers = int(s.get("workers", 6))
        t = cfg["tracking"]
        self.tracking = t
        self.media_schedule = str(t["media_schedule"])
        self.frame_min = int(t["frame_min"])
        self.cal = _path(t["ri_calibration"]) if t.get("ri_calibration") else None
        self.bad = _path(t["bad_frames"]) if t.get("bad_frames") else None
        self.marker = str(t.get("production_marker") or "volume_um3_efd")
        self.track_workers = int(t.get("workers", 1))
        self.edge_channels = list(cfg.get("edge_channels") or [])
        self.inputs_extra = [_path(x) for x in (cfg.get("inputs_extra") or [])]
        self.qc_extra = [_path(x) for x in (cfg.get("qc_extra") or [])]
        self.derived = cfg.get("derived") or None
        self.channel_filter = None   # {(pos, 'chNN')} from --channels-file
        self.channels_file = None
        self.log_path = self.mask_root / "_pipeline" / f"{self.id}_pipeline.log"

    def raw_root_for(self, n: int) -> Path:
        for lo, hi, root in self.overrides:
            if lo <= n <= hi:
                return root
        return self.raw_root

    def raw_pos(self, n: int) -> Path:
        return self.raw_root_for(n) / f"Pos{n}" / self.rel

    def mask_pos(self, n: int) -> Path:
        return self.mask_root / f"Pos{n}" / self.rel


# ------------------------------------------------------------------ helpers
def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_stamp()}] {msg}"
    with _lock:
        print(line, flush=True)
        if _LOG is not None:
            with _LOG.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def _run(cmd: list[str], env: dict | None = None) -> int:
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    if env:
        e.update(env)
    return subprocess.run(cmd, env=e).returncode


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _set_readonly_tree(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD)


def _clear_readonly_tree(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD | stat.S_IWRITE)


def _channels(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    return sorted((p for p in base.iterdir() if p.is_dir() and p.name.startswith("ch")),
                  key=lambda p: int(p.name[2:]))


def count_masks(inference_dir: Path) -> int:
    n = 0
    try:
        with os.scandir(inference_dir) as it:
            for e in it:
                if e.name.endswith("_masks.tif"):
                    n += 1
    except FileNotFoundError:
        pass
    return n


def is_production(ds: Dataset, lo: Path) -> bool:
    params = lo / "lineage_run_params.json"
    csv = lo / "lineage_data3D.csv"
    if not (params.exists() and csv.exists()):
        return False
    try:
        j = json.loads(params.read_text(encoding="utf-8"))
        if j.get("media_schedule") != ds.media_schedule or j.get("frame_min") != ds.frame_min:
            return False
        return ds.marker in pd.read_csv(csv, nrows=0).columns
    except Exception:  # noqa: BLE001
        return False


def production_channels(ds: Dataset, start: int | None = None, end: int | None = None) -> list[tuple[str, str, Path]]:
    out = []
    for pos_dir in sorted(ds.mask_root.glob("Pos*"), key=lambda p: int(p.name[3:])):
        n = int(pos_dir.name[3:])
        if (start is not None and n < start) or (end is not None and n > end):
            continue
        for ch in _channels(pos_dir / ds.rel):
            if ds.channel_filter is not None and (n, ch.name) not in ds.channel_filter:
                continue
            lo = ch / "inference_out" / "lineage_out"
            if is_production(ds, lo):
                out.append((pos_dir.name, ch.name, lo))
    return out


# ------------------------------------------------------------------ stage: seg
def seg_done(ds: Dataset, n: int) -> bool:
    raw_chs = _channels(ds.raw_pos(n))
    return bool(raw_chs) and all((ds.mask_pos(n) / c.name / "inference_out" / "_DONE").exists() for c in raw_chs)


def stage_seg(ds: Dataset, n: int, workers: int, max_files: int | None = None) -> bool:
    if not ds.raw_pos(n).is_dir():
        _log(f"seg Pos{n}: no raw dir {ds.raw_pos(n)} - skipped")
        return False
    if seg_done(ds, n) and not max_files:
        _log(f"seg Pos{n}: already done")
        return True
    t0 = time.time()
    cmd = [PY, "-u", str(SCRIPTS / "seg_omnipose.py"),
           "--raw-root", str(ds.raw_root_for(n)), "--mask-root", str(ds.mask_root),
           "--channel-rel", ds.rel.as_posix(), "--model", str(ds.model),
           "--pos-start", str(n), "--pos-end", str(n), "--workers", str(workers),
           "--phase-glob", ds.phase_glob]
    if ds.channels_file:
        cmd += ["--channels-file", str(ds.channels_file)]
    if ds.seg_eval:
        cmd += ["--eval-json", json.dumps(ds.seg_eval)]
    if "phase_hi" in ds.gate:
        cmd += ["--gate-hi", str(ds.gate["phase_hi"])]
    if "min_px" in ds.gate:
        cmd += ["--gate-min-px", str(ds.gate["min_px"])]
    if max_files:
        cmd += ["--max-files", str(max_files)]
    rc = _run(cmd)
    ok = rc == 0 and (bool(max_files) or seg_done(ds, n))
    _log(f"seg Pos{n}: rc={rc} ok={ok} ({time.time() - t0:.0f}s)")
    return ok


# ------------------------------------------------------------------ stage: track
def worklist(ds: Dataset, n: int, force: bool) -> tuple[list[tuple[Path, Path]], int, int]:
    """[(mask_channel_dir, raw_channel_dir)], n_skipped_done, n_skipped_empty."""
    targets: list[tuple[Path, Path]] = []
    n_done = n_empty = 0
    for ch in _channels(ds.mask_pos(n)):
        if ds.channel_filter is not None and (n, ch.name) not in ds.channel_filter:
            continue
        inf = ch / "inference_out"
        if count_masks(inf) == 0:
            n_empty += 1
            continue
        if not force and is_production(ds, inf / "lineage_out"):
            n_done += 1
            continue
        raw = ds.raw_pos(n) / ch.name
        if not raw.is_dir():
            _log(f"WARN no raw phase dir for Pos{n}/{ch.name}: {raw} - skipped")
            continue
        targets.append((ch, raw))
    return targets, n_done, n_empty


def tracker_cmd(ds: Dataset, mask_ch: Path, raw_ch: Path) -> list[str]:
    t = ds.tracking
    cmd = [PY, "-u", str(SCRIPTS / "central_cell_lineage_tracker.py"),
           "--indir", str(mask_ch), "--raw-dir", str(raw_ch),
           "--pixel-size-um", str(t.get("pixel_size_um", 0.34567514677103717)),
           "--time-interval-min", str(t.get("time_interval_min", 5.0)),
           "--wavelength-nm", str(t.get("wavelength_nm", 658.0)),
           "--alpha-ri", str(t.get("alpha_ri", 0.00018)),
           "--media-schedule", ds.media_schedule,
           "--frame-min", str(ds.frame_min)]
    if ds.cal:
        cmd += ["--ri-calibration", str(ds.cal)]
    if t.get("calibration_id"):
        cmd += ["--calibration-id", str(t["calibration_id"])]
    if ds.bad:
        cmd += ["--bad-frames", str(ds.bad)]
    for key, flag in (("n_milliq", "--n-milliq"), ("min_area", "--min-area"), ("frame_max", "--frame-max")):
        if t.get(key) is not None:
            cmd += [flag, str(t[key])]
    return cmd


def stage_track(ds: Dataset, n: int, force: bool, workers: int) -> bool:
    targets, n_done, n_empty = worklist(ds, n, force)
    _log(f"track Pos{n}: {len(targets)} channels to track ({n_done} done, {n_empty} empty)")
    t0 = time.time()
    n_err = 0

    def one(pair):
        mask_ch, raw_ch = pair
        t1 = time.time()
        rc = _run(tracker_cmd(ds, mask_ch, raw_ch))
        _log(("ok" if rc == 0 else "!!") + f" track Pos{n}/{mask_ch.name} rc={rc} ({time.time() - t1:.0f}s)")
        return rc

    if workers > 1 and len(targets) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            n_err = sum(1 for rc in ex.map(one, targets) if rc != 0)
    else:
        n_err = sum(1 for pair in targets if one(pair) != 0)
    _log(f"track Pos{n}: done, errors={n_err} ({time.time() - t0:.0f}s)")
    return n_err == 0


def run_seg_track(ds: Dataset, poss: list[int], stages: set[str], force_track: bool,
                  seg_workers: int, track_workers: int, max_files: int | None) -> list[str]:
    """seg -> track pipelined per Pos (two threads, one queue)."""
    failures: list[str] = []
    if "seg" in stages and "track" in stages:
        q: queue.Queue = queue.Queue()

        def seg_worker():
            for n in poss:
                if stage_seg(ds, n, seg_workers, max_files):
                    q.put(n)
                else:
                    failures.append(f"seg Pos{n}")
            q.put(None)

        def track_worker():
            while True:
                n = q.get()
                if n is None:
                    return
                try:
                    if not stage_track(ds, n, force_track, track_workers):
                        failures.append(f"track Pos{n}")
                except Exception as e:  # noqa: BLE001
                    failures.append(f"track Pos{n}: {e!r}")
                    _log(f"!! track Pos{n} raised {e!r}")

        threads = [threading.Thread(target=seg_worker, name="seg", daemon=True),
                   threading.Thread(target=track_worker, name="track", daemon=True)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    elif "seg" in stages:
        for n in poss:
            if not stage_seg(ds, n, seg_workers, max_files):
                failures.append(f"seg Pos{n}")
    elif "track" in stages:
        for n in poss:
            if not ds.mask_pos(n).is_dir():
                continue
            if not stage_track(ds, n, force_track, track_workers):
                failures.append(f"track Pos{n}")
    return failures


# ------------------------------------------------------------------ stage: qc
def stage_qc(ds: Dataset, start: int | None, end: int | None, force: bool = False) -> None:
    import division_qc_260517 as dqc
    chans = production_channels(ds, start, end)
    t0 = time.time()
    n_ok = n_err = 0
    for pos, ch, lo in chans:
        try:
            dqc.run_lineage_dir(lo, force=force)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_err += 1
            _log(f"WARN division QC failed for {pos}/{ch}: {e!r}")
    _log(f"qc: {n_ok} channels ok, {n_err} failed ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ stage: consolidate
def consolidate(ds: Dataset) -> Path:
    """Concatenate every production lineage (all cells, all channels) into ds.consolidated."""
    import division_qc_260517 as dqc
    out = ds.consolidated
    out.mkdir(parents=True, exist_ok=True)
    long_path = out / "all_cells_lineage_data3D.csv.gz"
    clist_path = out / "all_cells_clist.csv.gz"
    bad_path = out / "all_cells_lineage_bad_frames.csv.gz"
    qc_path = out / "all_cells_divisions_qc.csv.gz"
    index_rows = []
    t0 = time.time()
    n_ch = 0
    with gzip.open(long_path, "wt", encoding="utf-8", newline="") as f_long, \
         gzip.open(clist_path, "wt", encoding="utf-8", newline="") as f_clist, \
         gzip.open(bad_path, "wt", encoding="utf-8", newline="") as f_bad, \
         gzip.open(qc_path, "wt", encoding="utf-8", newline="") as f_qc:
        first_long = first_clist = first_bad = first_qc = True
        for pos, chname, lo in production_channels(ds):
            df = pd.read_csv(lo / "lineage_data3D.csv")
            df.insert(0, "ch", chname)
            df.insert(0, "pos", pos)
            df.to_csv(f_long, header=first_long, index=False)
            first_long = False
            n_ch += 1
            n_cells = None
            cl_csv = lo / "clist.csv"
            if cl_csv.exists():
                cl = pd.read_csv(cl_csv)
                cl.insert(0, "ch", chname)
                cl.insert(0, "pos", pos)
                cl.to_csv(f_clist, header=first_clist, index=False)
                first_clist = False
                n_cells = int(len(cl))
            bad_csv = lo / "lineage_bad_frames.csv"
            if bad_csv.exists():
                bd = pd.read_csv(bad_csv)
                if len(bd):
                    bd.insert(0, "ch", chname)
                    bd.insert(0, "pos", pos)
                    bd.to_csv(f_bad, header=first_bad, index=False)
                    first_bad = False
            n_div_valid = None
            try:
                qpath = dqc.run_lineage_dir(lo)
                if qpath is not None:
                    q = pd.read_csv(qpath)
                    if len(q):
                        n_div_valid = int((q["is_mother_division"] & q["validated"]).sum())
                        q.insert(0, "ch", chname)
                        q.insert(0, "pos", pos)
                        q.to_csv(f_qc, header=first_qc, index=False)
                        first_qc = False
                    else:
                        n_div_valid = 0
            except Exception as e:  # noqa: BLE001
                _log(f"WARN division QC failed for {pos}/{chname}: {e!r}")
            params = json.loads((lo / "lineage_run_params.json").read_text(encoding="utf-8"))
            mother = df[df["rank"] == 1] if "rank" in df.columns else df.iloc[0:0]
            index_rows.append({
                "pos": pos, "ch": chname,
                "edge_channel": chname in ds.edge_channels,
                "n_masks": count_masks(lo.parent),
                "n_rows": int(len(df)),
                "n_cells": n_cells if n_cells is not None else int(df["cell_id"].nunique()),
                "n_in_tree_cells": int(df.loc[df["in_tree"] == True, "cell_id"].nunique()),  # noqa: E712
                "n_divisions": params.get("n_divisions"),
                "n_mother_divisions_validated": n_div_valid,
                "n_outlier_rows": int(df["is_outlier"].sum()),
                "mother_frames": int(len(mother)),
                "mother_frames_with_ri": int(mother["mean_ri"].notna().sum()) if len(mother) else 0,
                "first_frame": int(df["frame"].min()) if len(df) else None,
                "last_frame": int(df["frame"].max()) if len(df) else None,
                "lineage_csv": str(lo / "lineage_data3D.csv"),
            })
            if n_ch % 50 == 0:
                _log(f"  consolidated {n_ch} channels so far ({time.time() - t0:.0f}s)")
    idx = pd.DataFrame(index_rows)
    for c in ("n_divisions", "n_mother_divisions_validated", "first_frame", "last_frame"):
        if c in idx.columns:
            idx[c] = idx[c].astype("Int64")
    idx.to_csv(out / "channel_index.csv", index=False)
    manifest = {
        "dataset": ds.id,
        "description": ds.description,
        "created": _stamp(),
        "n_channels": n_ch,
        "n_rows_total": int(idx["n_rows"].sum()) if len(idx) else 0,
        "model": str(ds.model),
        "mask_root": str(ds.mask_root),
        "raw_root": str(ds.raw_root),
        "raw_root_overrides": [{"pos_min": lo, "pos_max": hi, "root": str(r)} for lo, hi, r in ds.overrides],
        "channel_rel": ds.rel.as_posix(),
        "ri_calibration": str(ds.cal) if ds.cal else None,
        "media_schedule": ds.media_schedule,
        "bad_frames": str(ds.bad) if ds.bad else None,
        "frame_min": ds.frame_min,
        "pixel_size_um": float(ds.tracking.get("pixel_size_um", 0.34567514677103717)),
        "time_interval_min": float(ds.tracking.get("time_interval_min", 5.0)),
        "wavelength_nm": float(ds.tracking.get("wavelength_nm", 658.0)),
        "alpha_ri": float(ds.tracking.get("alpha_ri", 0.00018)),
        "production_marker": ds.marker,
        "edge_channels": ds.edge_channels,
        "tracker": str(SCRIPTS / "central_cell_lineage_tracker.py"),
        "driver": str(SCRIPTS / "run_dataset_pipeline.py"),
        "dataset_yaml": str(ds.yaml_path),
        "files": {"lineage_data3D": long_path.name, "clist": clist_path.name, "bad_frames": bad_path.name,
                  "divisions_qc": qc_path.name, "channel_index": "channel_index.csv"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    local = REPO / "results" / ds.id
    local.mkdir(parents=True, exist_ok=True)
    idx.to_csv(local / "channel_index.csv", index=False)
    (local / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"consolidated {n_ch} channels, {manifest['n_rows_total']} rows -> {out} ({time.time() - t0:.0f}s)")
    return out


# ------------------------------------------------------------------ stage: publish
def publish(ds: Dataset, tag: str, force: bool, mirror: bool, readonly: bool) -> Path:
    cons_manifest = ds.consolidated / "manifest.json"
    if not cons_manifest.exists():
        raise SystemExit(f"consolidated manifest not found: {cons_manifest} (run --stages consolidate first)")
    cons = json.loads(cons_manifest.read_text(encoding="utf-8"))
    dest = ds.master_root / tag
    if dest.exists():
        if not force:
            raise SystemExit(f"master {dest} already exists; use --force-publish to replace it")
        _log(f"--force-publish: removing existing {dest}")
        _clear_readonly_tree(dest)
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    _log(f"publishing master {tag} -> {dest}")
    sizes: dict[str, int] = {}

    def add(kind: str, src: Path, dst: Path) -> None:
        sizes[kind] = sizes.get(kind, 0) + _copy(src, dst)

    for p in sorted(ds.consolidated.iterdir()):
        if p.is_file():
            add("consolidated", p, dest / "consolidated" / p.name)
    chans = production_channels(ds)
    if len(chans) != cons.get("n_channels"):
        _log(f"WARN: {len(chans)} production channels now, consolidated manifest says {cons.get('n_channels')} "
             f"- re-run --stages consolidate before publishing")
    for pos, ch, lo in chans:
        for name in PER_CHANNEL_FILES:
            f = lo / name
            if f.exists():
                add("per_channel", f, dest / "per_channel" / pos / ch / name)
    _log(f"copied {len(chans)} per-channel lineage dirs")

    inputs = dest / "inputs"
    for src in [ds.cal, ds.bad, ds.yaml_path, *ds.inputs_extra]:
        if src is not None and src.exists():
            add("inputs", src, inputs / src.name)
        elif src is not None:
            _log(f"WARN input not found, not archived: {src}")
    if ds.model.exists():
        add("inputs", ds.model, inputs / "model" / ds.model.name)
        reg = REPO / "models" / "MODELS.json"
        if reg.exists():
            entry = next((e for e in json.loads(reg.read_text(encoding="utf-8"))["models"]
                          if e["file"] == ds.model.name), None)
            if entry:
                (inputs / "model" / "MODEL.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
    else:
        _log(f"WARN model checkpoint not found, not archived: {ds.model}")
    for name in CODE_FILES:
        src = SCRIPTS / name
        if src.exists():
            add("code", src, dest / "code" / name)
    (dest / "code").mkdir(exist_ok=True)
    (dest / "code" / "git_head.txt").write_text(_git_head() + "\n", encoding="utf-8")
    for src in ds.qc_extra:
        if src.exists():
            add("qc", src, dest / "qc" / src.name)
    schema = REPO / "docs" / "LINEAGE_DATAFRAME_SCHEMA.md"
    if schema.exists():
        _copy(schema, dest / "SCHEMA.md")

    manifest = {
        "tag": tag,
        "dataset": ds.id,
        "description": ds.description,
        "published": _stamp(),
        "git_head": _git_head(),
        "dataset_yaml": ds.yaml_path.name,
        "source_mask_root": str(ds.mask_root),
        "source_raw_root": str(ds.raw_root),
        "raw_root_overrides": cons.get("raw_root_overrides", []),
        "model_checkpoint": ds.model.name,
        "consolidated_manifest": cons,
        "n_channels": len(chans),
        "channels": [f"{p}_{c}" for p, c, _ in chans],
        "edge_channels": ds.edge_channels,
        "bytes": sizes,
        "layout": {
            "consolidated": "all-cell long table (pos, ch, cell_id, frame, ...) + clist + bad_frames + divisions_qc + channel_index",
            "per_channel/PosN/chNN": "verbatim tracker outputs per channel",
            "inputs": "RI calibration, bad_frames, dataset yaml, extra provenance, Omnipose checkpoint (+ MODEL.json)",
            "code": "scripts as used (see git_head.txt)",
            "qc": "channel-level QC files listed in the yaml",
            "derived": "packages built from this master by the yaml's derived hook",
        },
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = f"""# {ds.id} master tracking dataset - {tag}

{ds.description}

Frozen output of the all-cell lineage tracking (mother and daughters) of every trap channel.
Segmentation: Omnipose `{ds.model.name}` (see inputs/model/MODEL.json).
Tracking: `code/central_cell_lineage_tracker.py` driven by `code/run_dataset_pipeline.py` with
`inputs/{ds.yaml_path.name}` (media schedule `{ds.media_schedule}`, frame_min {ds.frame_min},
drift bad frames excluded, yellow-contour geometry: volume_um3_rod / volume_um3_efd).
Published {manifest['published']}.

This directory is READ-ONLY. Do not edit in place; publish a new tag instead.
Downstream analyses read from here (`qpi_paths` picks LATEST automatically).

## Load

```python
import pandas as pd
df = pd.read_csv(r"{dest / 'consolidated' / 'all_cells_lineage_data3D.csv.gz'}")
mother = df[df["rank"] == 1]                        # central (mother) cell per channel
good = df[~(df.is_outlier | df.touches_border)]
```

Column definitions: `SCHEMA.md`. Per-channel verbatim outputs: `per_channel/PosN/chNN/`.
Edge channels {ds.edge_channels} are included here but excluded from analysis cohorts
(`consolidated/channel_index.csv: edge_channel`).
Mask pixels for any row: open
`{ds.mask_root}\\PosN\\...\\chNN\\inference_out\\img_<frame>_ph_000_phase_masks.tif` and select `mask_label`.

## Integrity

`SHA256SUMS.txt` lists every file. Verify with `sha256sum -c SHA256SUMS.txt` (Git Bash).
"""
    (dest / "README.md").write_text(readme, encoding="utf-8")

    if ds.derived:
        try:
            mod = importlib.import_module(str(ds.derived["module"]))
            fn = getattr(mod, str(ds.derived.get("function") or "build"))
            kwargs = dict(ds.derived.get("kwargs") or {})
            name = str(ds.derived.get("name") or "derived")
            fn(dest / "consolidated", inputs, dest / "derived" / name, derived_from=tag,
               qc_dir=dest / "qc", log=_log, **kwargs)
        except Exception as e:  # noqa: BLE001
            _log(f"WARN: derived package failed (master itself is complete): {e!r}")

    lines = []
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            lines.append(f"{_sha256(p)} *{p.relative_to(dest).as_posix()}")
    with (dest / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as fh:  # LF for sha256sum -c
        fh.write("\n".join(lines) + "\n")
    _log(f"hashed {len(lines)} files")
    if readonly:
        _set_readonly_tree(dest)
    (ds.master_root / "LATEST.txt").write_text(tag + "\n", encoding="utf-8")
    _log(f"LATEST -> {tag}")

    if mirror and ds.mirror_root is not None:
        if ds.mirror_root.parent.exists():
            mdst = ds.mirror_root / tag
            n = 0
            for p in dest.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(dest)
                if rel.parts[0] == "per_channel":
                    continue
                _copy(p, mdst / rel)
                n += 1
            ds.mirror_root.mkdir(parents=True, exist_ok=True)
            (ds.mirror_root / "LATEST.txt").write_text(tag + "\n", encoding="utf-8")
            _log(f"mirrored {n} files -> {mdst}")
        else:
            _log(f"WARN: mirror root not reachable, skipped: {ds.mirror_root}")
    _log(f"DONE master {tag}: {len(chans)} channels, {sum(sizes.values()) / 1e6:.0f} MB at {dest}")
    return dest


# ------------------------------------------------------------------ plan
def plan(ds: Dataset, poss: list[int], stages: set[str], tag: str | None) -> None:
    print(f"dataset {ds.id}: {ds.description}")
    print(f"  python      {PY}")
    print(f"  raw_root    {ds.raw_root}  ({'ok' if ds.raw_root.exists() else 'MISSING'})")
    for lo, hi, root in ds.overrides:
        print(f"    Pos{lo}-{hi} -> {root}  ({'ok' if root.exists() else 'MISSING'})")
    print(f"  mask_root   {ds.mask_root}  ({'ok' if ds.mask_root.exists() else 'will be created'})")
    print(f"  channel_rel {ds.rel.as_posix()}")
    print(f"  model       {ds.model}  ({'ok' if ds.model.exists() else 'MISSING'})")
    print(f"  calibration {ds.cal}  ({'ok' if ds.cal and ds.cal.exists() else 'MISSING' if ds.cal else 'none'})")
    print(f"  bad_frames  {ds.bad}  ({'ok' if ds.bad and ds.bad.exists() else 'MISSING' if ds.bad else 'none'})")
    print(f"  media       {ds.media_schedule}; frame_min {ds.frame_min}; marker {ds.marker}")
    print(f"  stages      {', '.join(s for s in STAGES if s in stages)}; Pos {poss[0]}..{poss[-1]}")
    tot = dict(raw_ch=0, seg_needed=0, track=0, done=0, empty=0)
    for n in poss:
        raw_chs = _channels(ds.raw_pos(n))
        mask_chs = _channels(ds.mask_pos(n))
        if not raw_chs and not mask_chs:
            continue
        sd = seg_done(ds, n)
        targets, n_done, n_empty = worklist(ds, n, False) if mask_chs else ([], 0, 0)
        tot["raw_ch"] += len(raw_chs)
        tot["seg_needed"] += 0 if sd else len(raw_chs)
        tot["track"] += len(targets)
        tot["done"] += n_done
        tot["empty"] += n_empty
        print(f"  Pos{n:<4d} raw ch {len(raw_chs):2d} ({ds.raw_root_for(n).drive or ds.raw_root_for(n)})  "
              f"seg {'done' if sd else 'NEEDED'}  track: {len(targets)} to do, {n_done} done, {n_empty} empty")
    print(f"  totals: raw channels {tot['raw_ch']}, seg needed {tot['seg_needed']}, track to do {tot['track']}, "
          f"production {tot['done']}, empty {tot['empty']}")
    n_prod = len(production_channels(ds))
    print(f"  production channels (all Pos): {n_prod}")
    print(f"  consolidated -> {ds.consolidated}  ({'exists' if (ds.consolidated / 'manifest.json').exists() else 'not yet'})")
    if "publish" in stages:
        dest = ds.master_root / (tag or "v<YYYYMMDD>")
        print(f"  publish -> {dest}  ({'EXISTS' if dest.exists() else 'new'}); mirror {ds.mirror_root}")
        if ds.derived:
            print(f"  derived hook: {ds.derived.get('module')}.{ds.derived.get('function', 'build')} -> derived/{ds.derived.get('name')}")


# ------------------------------------------------------------------ main
def main() -> int:
    global _LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml", help="datasets/<id>.yaml")
    ap.add_argument("--stages", default="seg,track,qc,consolidate",
                    help="comma list of " + ",".join(STAGES) + " (default seg,track,qc,consolidate)")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--tag", default=None, help="master tag for publish (default v<YYYYMMDD>)")
    ap.add_argument("--plan", action="store_true", help="print what would run and exit")
    ap.add_argument("--force-track", action="store_true", help="re-track channels that already have a production lineage")
    ap.add_argument("--force-qc", action="store_true", help="recompute divisions_qc.csv even when up to date")
    ap.add_argument("--max-files", type=int, default=None, help="seg smoke test: first N frames per channel, no _DONE")
    ap.add_argument("--channels-file", default=None,
                    help="restrict every stage to the channels listed in this file "
                         "(channel_contact_sheet.py --serve writes it when you press "
                         "'analyse these channels')")
    ap.add_argument("--seg-workers", type=int, default=None)
    ap.add_argument("--track-workers", type=int, default=None)
    ap.add_argument("--consolidated-dir", default=None, help="override paths.consolidated_dir")
    ap.add_argument("--master-root", default=None, help="override paths.master_root")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--no-readonly", action="store_true")
    ap.add_argument("--force-publish", action="store_true", help="replace an existing master tag")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass

    ds = Dataset(Path(a.yaml))
    if a.channels_file:
        from seg_omnipose import load_channel_filter
        ds.channels_file = Path(a.channels_file)
        ds.channel_filter = load_channel_filter(a.channels_file)
        print(f"channel filter: {a.channels_file} "
              f"({len(ds.channel_filter)} channels)")
    if a.consolidated_dir:
        ds.consolidated = _path(a.consolidated_dir)
    if a.master_root:
        ds.master_root = _path(a.master_root)
    stages = {s.strip() for s in a.stages.split(",") if s.strip()}
    bad = stages - set(STAGES)
    if bad:
        raise SystemExit(f"unknown stage(s): {sorted(bad)}; choose from {STAGES}")
    start = a.start if a.start is not None else ds.pos_start
    end = a.end if a.end is not None else ds.pos_end
    poss = list(range(start, end + 1))
    seg_workers = a.seg_workers or ds.seg_workers
    track_workers = a.track_workers or ds.track_workers
    tag = a.tag or time.strftime("v%Y%m%d")

    if a.plan:
        plan(ds, poss, stages, a.tag)
        return 0

    ds.log_path.parent.mkdir(parents=True, exist_ok=True)
    _LOG = ds.log_path
    _log(f"=== {ds.id} pipeline START stages={sorted(stages)} Pos{start}..{end} yaml={ds.yaml_path} ===")
    if "seg" in stages and not ds.model.exists():
        raise SystemExit(f"model checkpoint not found: {ds.model}")
    failures = run_seg_track(ds, poss, stages, a.force_track, seg_workers, track_workers, a.max_files)
    if "qc" in stages:
        stage_qc(ds, start, end, force=a.force_qc)
    if "consolidate" in stages:
        consolidate(ds)
    if "publish" in stages:
        publish(ds, tag, force=a.force_publish, mirror=not a.no_mirror, readonly=not a.no_readonly)
    if failures:
        _log("FAILURES: " + "; ".join(failures))
    _log(f"=== {ds.id} pipeline END ({'with failures' if failures else 'ok'}) ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
