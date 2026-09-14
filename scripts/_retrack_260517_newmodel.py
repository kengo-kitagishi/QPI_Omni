"""_retrack_260517_newmodel.py - production lineage tracking of the 260517
dataset on the NEW Omnipose masks, for every channel of every position.

Masks: D:\\260517_seg\\PosN\\output_phase\\channels\\crop_sub_rawraw\\z000\\chNN\\inference_out
       (written 2026-09-07..09 by the omni_model_d20_2026_09_07_12_41_31 checkpoint;
       only frames with >=1 detected cell have a *_masks.tif)
Phase: H:\\260517\\2per_0055per_0per_2per_crop_sub\\PosN\\...\\z000\\chNN\\img_*_ph_000_phase.tif
       (BUFFALO USB HDD; read-only here)

The tracker is called with ``--indir <D: channel> --raw-dir <H: channel>``, so
lineage_out is written next to the masks on D: and nothing is written to H:.

Production parameters (same as the June 2026 run; see _chain_seg_260517.py and
_retrack_corrected_260517.py):
  --ri-calibration   H:\\260517\\grid_2pergluc_2\\ri_calibration_results.json
                     active entry 260517_2per_0055per_0per_2per_manual_20260608T000220
                     (wo_2=1.33503, wo_0=wo_0p0055=1.33274, n_milliq=1.3312)
  --media-schedule   0:wo_2,2019:wo_0p0055,2307:wo_0,2885:wo_2   (absolute img_NNN)
  --bad-frames       H:\\260517\\drift_session_20260521T2142\\bad_frames.json
  --frame-min 2      (img_0/1 dropped; img_2 is time 0)
  --pixel-size-um 0.34567514677103717  --time-interval-min 5  --wavelength-nm 658  --alpha-ri 0.00018

Outlier handling is the tracker's own three-stage scheme:
  1. drift bad frames (bad_frames.json) are excluded BEFORE linking; their raw
     measurements go to lineage_bad_frames.csv;
  2. masks touching the image border are dropped from linking (cell exited);
  3. frames failing the continuation / division area rules are kept as rows with
     is_outlier=True and volume / RI / mass / density set to NaN, so cell IDs stay
     continuous across the glitch.

Resumable: a channel is skipped when its lineage_run_params.json already records
this media schedule and frame_min, and lineage_data3D.csv carries the
``density_pg_um3`` column. The masks-only QC runs of 2026-09-09 (media_schedule
null, total_phase NaN) do not satisfy this and are redone.

Single-stream by default: H: is a single-spindle USB HDD and parallel reads
thrash badly (2026-06 notes). No figures and no Notion posts are produced.

After the loop (or with --consolidate-only) every production lineage CSV is
concatenated into D:\\260517_seg\\_lineage_consolidated\\ together with a
per-channel index and a manifest recording model, parameters and timestamps.

Usage:
  python scripts/_retrack_260517_newmodel.py                 # all Pos, resume
  python scripts/_retrack_260517_newmodel.py --start 1 --end 10
  python scripts/_retrack_260517_newmodel.py --consolidate-only
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
PY = r"C:\Users\QPI\anaconda3\envs\omnipose\python.exe"
TRACKER = SCRIPTS / "central_cell_lineage_tracker.py"
LOG = SCRIPTS / "_retrack_260517_newmodel.log"

MASK_ROOT = Path(r"D:\260517_seg")
RAW_ROOT = Path(r"H:\260517\2per_0055per_0per_2per_crop_sub")
REL = Path("output_phase") / "channels" / "crop_sub_rawraw" / "z000"
CONSOLIDATED = MASK_ROOT / "_lineage_consolidated"

MODEL = (r"C:\Users\QPI\Desktop\train\omni_model_d20\models"
         r"\cellpose_residual_on_style_on_concatenation_off_omni_abstract"
         r"_nclasses_3_nchan_1_dim_2_omni_model_d20_2026_09_07_12_41_31.782047")
CAL = Path(r"H:\260517\grid_2pergluc_2\ri_calibration_results.json")
BAD = Path(r"H:\260517\drift_session_20260521T2142\bad_frames.json")
MEDIA_SCHEDULE = "0:wo_2,2019:wo_0p0055,2307:wo_0,2885:wo_2"
PIXEL_UM = "0.34567514677103717"
DT_MIN = "5.0"
WAVELENGTH_NM = "658.0"
ALPHA_RI = "0.00018"
FRAME_MIN = 2

PRODUCTION_MARKER = "volume_um3_efd"   # column only the 2026-09-14+ tracker writes (yellow-contour geometry)

# 2026-09-14 tilt fix: Pos >= 53 are mirrored traps whose crops were tilt-fitted on the
# wrong (cell) side by grid_subtract. refit_tilt_right_260517.py re-flattens them into
# D:\260517_tiltfix; those positions read their phase from there instead of H:.
RAW_ROOT_TILTFIX = Path(r"D:\260517_tiltfix")
TILTFIX_POS_MIN = 53


def raw_root_for(pos_name: str) -> Path:
    try:
        n = int(pos_name[3:])
    except ValueError:
        return RAW_ROOT
    return RAW_ROOT_TILTFIX if n >= TILTFIX_POS_MIN else RAW_ROOT


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    line = f"[{_stamp()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_masks(inference_dir: Path) -> int:
    n = 0
    try:
        with os.scandir(inference_dir) as it:
            for e in it:
                if e.name.endswith("_masks.tif"):
                    n += 1
    except FileNotFoundError:
        return 0
    return n


def is_production(lineage_out: Path) -> bool:
    """True if this channel already has a production (phase-aware) lineage."""
    params = lineage_out / "lineage_run_params.json"
    csv = lineage_out / "lineage_data3D.csv"
    if not (params.exists() and csv.exists()):
        return False
    try:
        j = json.loads(params.read_text(encoding="utf-8"))
        if j.get("media_schedule") != MEDIA_SCHEDULE or j.get("frame_min") != FRAME_MIN:
            return False
        head = pd.read_csv(csv, nrows=0)
        return PRODUCTION_MARKER in head.columns
    except Exception:
        return False


def channel_worklist(start: int, end: int, force: bool) -> tuple[list[tuple[str, Path, Path]], int, int]:
    """[(pos_name, mask_channel_dir, raw_channel_dir)], n_skipped_done, n_skipped_empty."""
    targets: list[tuple[str, Path, Path]] = []
    n_done = n_empty = 0
    for n in range(start, end + 1):
        pos = f"Pos{n}"
        z = MASK_ROOT / pos / REL
        if not z.is_dir():
            continue
        for ch in sorted(p for p in z.iterdir() if p.is_dir() and p.name.startswith("ch")):
            inf = ch / "inference_out"
            if count_masks(inf) == 0:
                n_empty += 1
                continue
            if (not force) and is_production(inf / "lineage_out"):
                n_done += 1
                continue
            raw = raw_root_for(pos) / pos / REL / ch.name
            if not raw.is_dir():
                _log(f"WARN no raw phase dir for {pos}/{ch.name}: {raw} - skipped")
                continue
            targets.append((pos, ch, raw))
    return targets, n_done, n_empty


def run_tracker(mask_ch: Path, raw_ch: Path) -> int:
    cmd = [
        PY, "-u", str(TRACKER),
        "--indir", str(mask_ch),
        "--raw-dir", str(raw_ch),
        "--pixel-size-um", PIXEL_UM,
        "--time-interval-min", DT_MIN,
        "--wavelength-nm", WAVELENGTH_NM,
        "--alpha-ri", ALPHA_RI,
        "--ri-calibration", str(CAL),
        "--media-schedule", MEDIA_SCHEDULE,
        "--bad-frames", str(BAD),
        "--frame-min", str(FRAME_MIN),
    ]
    return subprocess.run(cmd).returncode


def consolidate() -> None:
    """Concatenate every production lineage CSV (all cells, all channels)."""
    CONSOLIDATED.mkdir(parents=True, exist_ok=True)
    long_path = CONSOLIDATED / "all_cells_lineage_data3D.csv.gz"
    clist_path = CONSOLIDATED / "all_cells_clist.csv.gz"
    bad_path = CONSOLIDATED / "all_cells_lineage_bad_frames.csv.gz"
    qc_path = CONSOLIDATED / "all_cells_divisions_qc.csv.gz"
    import division_qc_260517 as dqc
    index_rows = []
    t0 = time.time()
    n_ch = 0
    with gzip.open(long_path, "wt", encoding="utf-8", newline="") as f_long, \
         gzip.open(clist_path, "wt", encoding="utf-8", newline="") as f_clist, \
         gzip.open(bad_path, "wt", encoding="utf-8", newline="") as f_bad, \
         gzip.open(qc_path, "wt", encoding="utf-8", newline="") as f_qc:
        first_long = first_clist = first_bad = first_qc = True
        for pos_dir in sorted(MASK_ROOT.glob("Pos*"), key=lambda p: int(p.name[3:])):
            z = pos_dir / REL
            if not z.is_dir():
                continue
            for ch in sorted(p for p in z.iterdir() if p.is_dir() and p.name.startswith("ch")):
                lo = ch / "inference_out" / "lineage_out"
                if not is_production(lo):
                    continue
                df = pd.read_csv(lo / "lineage_data3D.csv")
                df.insert(0, "ch", ch.name)
                df.insert(0, "pos", pos_dir.name)
                df.to_csv(f_long, header=first_long, index=False)
                first_long = False
                n_ch += 1
                cl_csv = lo / "clist.csv"
                n_cells = n_div = None
                if cl_csv.exists():
                    cl = pd.read_csv(cl_csv)
                    cl.insert(0, "ch", ch.name)
                    cl.insert(0, "pos", pos_dir.name)
                    cl.to_csv(f_clist, header=first_clist, index=False)
                    first_clist = False
                    n_cells = int(len(cl))
                bad_csv = lo / "lineage_bad_frames.csv"
                if bad_csv.exists():
                    bd = pd.read_csv(bad_csv)
                    if len(bd):
                        bd.insert(0, "ch", ch.name)
                        bd.insert(0, "pos", pos_dir.name)
                        bd.to_csv(f_bad, header=first_bad, index=False)
                        first_bad = False
                # division QC (mass / volume validation of every daughter birth)
                n_div_valid = None
                try:
                    qpath = dqc.run_lineage_dir(lo)
                    if qpath is not None:
                        q = pd.read_csv(qpath)
                        n_div_valid = int((q["is_mother_division"] & q["validated"]).sum()) if len(q) else 0
                        if len(q):
                            q.insert(0, "ch", ch.name)
                            q.insert(0, "pos", pos_dir.name)
                            q.to_csv(f_qc, header=first_qc, index=False)
                            first_qc = False
                except Exception as e:
                    _log(f"WARN division QC failed for {pos_dir.name}/{ch.name}: {e!r}")
                params = json.loads((lo / "lineage_run_params.json").read_text(encoding="utf-8"))
                n_div = params.get("n_divisions")
                mother = df[df["rank"] == 1] if "rank" in df.columns else df.iloc[0:0]
                index_rows.append({
                    "pos": pos_dir.name, "ch": ch.name,
                    "n_masks": count_masks(ch / "inference_out"),
                    "n_rows": int(len(df)),
                    "n_cells": n_cells if n_cells is not None else int(df["cell_id"].nunique()),
                    "n_in_tree_cells": int(df.loc[df["in_tree"] == True, "cell_id"].nunique()),
                    "n_divisions": n_div,
                    "n_mother_divisions_validated": n_div_valid,
                    "n_outlier_rows": int(df["is_outlier"].sum()),
                    "mother_frames": int(len(mother)),
                    "mother_frames_with_ri": int(mother["mean_ri"].notna().sum()),
                    "first_frame": int(df["frame"].min()) if len(df) else None,
                    "last_frame": int(df["frame"].max()) if len(df) else None,
                    "lineage_csv": str(lo / "lineage_data3D.csv"),
                })
                if n_ch % 50 == 0:
                    _log(f"  consolidated {n_ch} channels so far ({time.time() - t0:.0f}s)")
    idx = pd.DataFrame(index_rows)
    for c in ("n_divisions", "n_mother_divisions_validated", "first_frame", "last_frame"):
        if c in idx.columns:
            idx[c] = idx[c].astype("Int64")  # keep frame numbers integral (nullable)
    idx.to_csv(CONSOLIDATED / "channel_index.csv", index=False)
    manifest = {
        "created": _stamp(),
        "n_channels": n_ch,
        "n_rows_total": int(idx["n_rows"].sum()) if len(idx) else 0,
        "model": MODEL,
        "mask_root": str(MASK_ROOT),
        "raw_root": str(RAW_ROOT),
        "raw_root_tiltfix": str(RAW_ROOT_TILTFIX),
        "tiltfix_pos_min": TILTFIX_POS_MIN,
        "tiltfix_note": ("Pos >= tiltfix_pos_min: phase crops re-flattened with the background fitted on the "
                         "right third (refit_tilt_right_260517.py); grid_subtract had used the cell side."),
        "ri_calibration": str(CAL),
        "media_schedule": MEDIA_SCHEDULE,
        "bad_frames": str(BAD),
        "frame_min": FRAME_MIN,
        "pixel_size_um": float(PIXEL_UM),
        "time_interval_min": float(DT_MIN),
        "wavelength_nm": float(WAVELENGTH_NM),
        "alpha_ri": float(ALPHA_RI),
        "tracker": str(TRACKER),
        "files": {
            "lineage_data3D": long_path.name,
            "clist": clist_path.name,
            "bad_frames": bad_path.name,
            "divisions_qc": qc_path.name,
            "channel_index": "channel_index.csv",
        },
    }
    (CONSOLIDATED / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # small copies in the repo results dir for quick inspection (results/ is gitignored)
    local = SCRIPTS.parent / "results" / "260517_newmodel"
    local.mkdir(parents=True, exist_ok=True)
    idx.to_csv(local / "channel_index.csv", index=False)
    (local / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(f"consolidated {n_ch} channels, {manifest['n_rows_total']} rows -> {CONSOLIDATED} "
         f"({time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1, help="lowest Pos number")
    ap.add_argument("--end", type=int, default=104, help="highest Pos number")
    ap.add_argument("--force", action="store_true",
                    help="re-track channels that already have a production lineage")
    ap.add_argument("--consolidate-only", action="store_true",
                    help="skip tracking; only rebuild the consolidated tables")
    ap.add_argument("--no-consolidate", action="store_true",
                    help="track only; do not rebuild the consolidated tables at the end")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent tracker processes (default 1: H: is a single-spindle "
                         "USB HDD; try 2 only if throughput is verified to improve)")
    args = ap.parse_args()

    if args.consolidate_only:
        _log("=== consolidate-only ===")
        consolidate()
        return

    LOG.write_text(f"=== 260517 new-model re-track START {_stamp()} ===\n", encoding="utf-8")
    _log(f"Pos {args.start}..{args.end}  force={args.force}")
    for must in (CAL, BAD, TRACKER):
        if not must.exists():
            _log(f"FATAL: required file not found: {must}")
            sys.exit(1)
    if not Path(MODEL).exists():
        _log(f"WARN: model checkpoint not found at {MODEL} (provenance only; tracking does not need it)")

    targets, n_done, n_empty = channel_worklist(args.start, args.end, args.force)
    _log(f"queued {len(targets)} channels ({n_done} already production, {n_empty} without masks)")
    if not targets:
        _log("nothing to track")
    n_ok = n_err = 0
    t_all = time.time()

    def _one(item: tuple[int, tuple[str, Path, Path]]) -> tuple[str, int, float]:
        i, (pos, mask_ch, raw_ch) = item
        tag = f"{pos}/{mask_ch.name}"
        _log(f"--- [{i}/{len(targets)}] {tag} ---")
        t0 = time.time()
        rc = run_tracker(mask_ch, raw_ch)
        return tag, rc, time.time() - t0

    def _report(tag: str, rc: int, dt: float) -> None:
        nonlocal n_ok, n_err
        if rc != 0:
            n_err += 1
            _log(f"!! {tag} tracker rc={rc} ({dt:.0f}s)")
        else:
            n_ok += 1
            _log(f"ok {tag} ({dt:.0f}s)")

    if args.workers <= 1:
        for item in enumerate(targets, 1):
            _report(*_one(item))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one, item) for item in enumerate(targets, 1)]
            for fut in as_completed(futs):
                _report(*fut.result())
    _log(f"=== tracking DONE: ok={n_ok} err={n_err} ({(time.time() - t_all) / 3600:.2f} h) ===")

    if not args.no_consolidate:
        consolidate()
    _log("=== 260517 new-model re-track END ===")


if __name__ == "__main__":
    main()
