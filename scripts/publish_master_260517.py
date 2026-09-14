"""publish_master_260517.py - freeze the completed 260517 all-cell tracking data
as a versioned MASTER dataset that every downstream analysis starts from.

Source (working tree, may be overwritten by future re-runs):
    D:\\260517_seg\\PosN\\...\\z000\\chNN\\inference_out\\lineage_out\\   (production runs only)
    D:\\260517_seg\\_lineage_consolidated\\                              (from _retrack_260517_newmodel.py)

Destination (frozen, read-only, versioned):
    D:\\QPI_master\\260517\\<tag>\\
        README.md              what this is, how it was made, how to load it
        MANIFEST.json          provenance: model, parameters, git HEAD, counts, sizes
        SHA256SUMS.txt         checksum of every file below
        SCHEMA.md              column definitions (copy of docs/LINEAGE_DATAFRAME_SCHEMA.md)
        consolidated/          all_cells_lineage_data3D.csv.gz, all_cells_clist.csv.gz,
                               all_cells_lineage_bad_frames.csv.gz, channel_index.csv, manifest.json
        per_channel/PosN/chNN/ lineage_data3D.csv, clist.csv, lineage_cells.json,
                               lineage_bad_frames.csv, bad_frames_used.json, lineage_run_params.json
        inputs/                ri_calibration_results.json, bad_frames.json,
                               channel_classification_260517.yaml, model/<checkpoint>,
                               run_seg_260517_fast2.py (segmentation driver, if still present)
        code/                  tracker + chain + morphology + calibration modules as used, git_head.txt
        qc/                    channel-level QC json of the 2026-09-09 masks-only pass (if present)
    D:\\QPI_master\\260517\\LATEST.txt   -> <tag>

Mirror (compact, no per_channel): G:\\共有ドライブ\\wakamotolab_meeting\\kitagishi\\data_master\\260517\\<tag>\\

qpi_paths.resolve_lineage_csv() / find_lineage_csv() pick the LATEST master
automatically (override with QPI_LINEAGE_MASTER=<tag>, disable with
QPI_LINEAGE_SOURCE=inbox), so downstream scripts read this dataset by default.

Usage:
    python scripts/publish_master_260517.py                       # tag v<YYYYMMDD>_newmodel
    python scripts/publish_master_260517.py --tag v20260911_newmodel
    python scripts/publish_master_260517.py --dest C:\\tmp\\m --no-mirror --no-readonly   # test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
import _retrack_260517_newmodel as chain  # noqa: E402  (constants + is_production)

MASTER_ROOT = Path(r"D:\QPI_master\260517")
MIRROR_ROOT = Path(r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\data_master\260517")
OTHER_SESSION_SCRATCH = Path(r"C:\TEMP\claude\C--Users-QPI\945bff36-8800-4b3a-911b-252a0b4214fa\scratchpad")

PER_CHANNEL_FILES = [
    "lineage_data3D.csv", "clist.csv", "lineage_cells.json",
    "lineage_bad_frames.csv", "bad_frames_used.json", "lineage_run_params.json",
    "divisions_qc.csv",
]
CODE_FILES = [
    "central_cell_lineage_tracker.py", "mask_morphology.py", "ri_calibration.py",
    "_retrack_260517_newmodel.py", "publish_master_260517.py", "qpi_paths.py",
]
QC_FILES = ["short_oob_exclude.json", "tree_qc.json", "trackability.json"]
LOG = SCRIPTS / "publish_master_260517.log"


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


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _copy(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clear_readonly_tree(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            p.chmod(stat.S_IWRITE | stat.S_IREAD)


def _set_readonly_tree(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            p.chmod(stat.S_IREAD)


def production_channels() -> list[tuple[str, str, Path]]:
    out = []
    for pos_dir in sorted(chain.MASK_ROOT.glob("Pos*"), key=lambda p: int(p.name[3:])):
        z = pos_dir / chain.REL
        if not z.is_dir():
            continue
        for ch in sorted(p for p in z.iterdir() if p.is_dir() and p.name.startswith("ch")):
            lo = ch / "inference_out" / "lineage_out"
            if chain.is_production(lo):
                out.append((pos_dir.name, ch.name, lo))
    return out


def publish(tag: str, dest_root: Path, mirror: bool, readonly: bool, force: bool) -> Path:
    src_cons = chain.CONSOLIDATED
    cons_manifest = src_cons / "manifest.json"
    if not cons_manifest.exists():
        raise SystemExit(f"consolidated manifest not found: {cons_manifest} "
                         f"(run _retrack_260517_newmodel.py [--consolidate-only] first)")
    cons = json.loads(cons_manifest.read_text(encoding="utf-8"))

    dest = dest_root / tag
    if dest.exists():
        if not force:
            raise SystemExit(f"master {dest} already exists; use --force to replace it")
        _log(f"--force: removing existing {dest}")
        _clear_readonly_tree(dest)
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    _log(f"publishing master {tag} -> {dest}")

    sizes: dict[str, int] = {}

    # consolidated tables
    for p in sorted(src_cons.iterdir()):
        if p.is_file():
            sizes["consolidated"] = sizes.get("consolidated", 0) + _copy(p, dest / "consolidated" / p.name)

    # per-channel lineage outputs (production only)
    chans = production_channels()
    if len(chans) != cons.get("n_channels"):
        _log(f"WARN: {len(chans)} production channels found now, but consolidated manifest says "
             f"{cons.get('n_channels')} - consider re-running --consolidate-only before publishing")
    for pos, ch, lo in chans:
        for name in PER_CHANNEL_FILES:
            f = lo / name
            if f.exists():
                sizes["per_channel"] = sizes.get("per_channel", 0) + _copy(f, dest / "per_channel" / pos / ch / name)
    _log(f"copied {len(chans)} per-channel lineage dirs")

    # inputs / provenance
    inputs = dest / "inputs"
    for src in (chain.CAL, chain.BAD, REPO / "docs" / "channel_classification_260517.yaml"):
        if src.exists():
            sizes["inputs"] = sizes.get("inputs", 0) + _copy(src, inputs / src.name)
    model = Path(chain.MODEL)
    if model.exists():
        # short archive name: the full checkpoint name (~125 chars) risks Windows MAX_PATH
        short = ("omni_model_d20_" + model.name.split("omni_model_d20_")[-1]
                 if "omni_model_d20_" in model.name else model.name)
        sizes["inputs"] = sizes.get("inputs", 0) + _copy(model, inputs / "model" / short)
    else:
        _log(f"WARN: model checkpoint not found, not archived: {model}")
    seg_driver = OTHER_SESSION_SCRATCH / "run_seg_260517_fast2.py"
    if seg_driver.exists():
        sizes["inputs"] = sizes.get("inputs", 0) + _copy(seg_driver, inputs / seg_driver.name)
    else:
        _log("WARN: segmentation driver run_seg_260517_fast2.py not found (scratchpad gone?)")

    # code snapshot
    for name in CODE_FILES:
        src = SCRIPTS / name
        if src.exists():
            sizes["code"] = sizes.get("code", 0) + _copy(src, dest / "code" / name)
    (dest / "code" / "git_head.txt").write_text(_git_head() + "\n", encoding="utf-8")

    # QC artefacts of the masks-only pass (channel-level exclusions etc.)
    for name in QC_FILES:
        src = OTHER_SESSION_SCRATCH / name
        if src.exists():
            sizes["qc"] = sizes.get("qc", 0) + _copy(src, dest / "qc" / name)

    # schema
    schema = REPO / "docs" / "LINEAGE_DATAFRAME_SCHEMA.md"
    if schema.exists():
        _copy(schema, dest / "SCHEMA.md")

    manifest = {
        "tag": tag,
        "dataset": "260517 2per_0055per_0per_2per (glucose 2% -> 0.0055% -> 0% -> 2%), 104 Pos",
        "published": _stamp(),
        "git_head": _git_head(),
        "source_mask_root": str(chain.MASK_ROOT),
        "source_raw_root": str(chain.RAW_ROOT),
        "model_checkpoint": str(model),
        "consolidated_manifest": cons,
        "n_channels": len(chans),
        "channels": [f"{p}_{c}" for p, c, _ in chans],
        "bytes": sizes,
        "layout": {
            "consolidated": "all-cell long table (pos, ch, cell_id, frame, ...) + clist + bad_frames + channel_index",
            "per_channel/PosN/chNN": "verbatim tracker outputs per channel",
            "inputs": "RI calibration, bad_frames, channel classification, Omnipose checkpoint, seg driver",
            "code": "scripts as used (see git_head.txt)",
            "qc": "channel-level QC from the 2026-09-09 masks-only pass",
        },
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = f"""# 260517 master tracking dataset - {tag}

Frozen output of the all-cell lineage tracking of the 260517 glucose media-switch
experiment (2% -> 0.0055% -> 0% -> 2%, 104 positions, 5 min/frame, single z).
Segmentation: Omnipose `{model.name}` (trained 2026-09-07).
Tracking: `code/central_cell_lineage_tracker.py` via `code/_retrack_260517_newmodel.py`
(RI calibration, media schedule `{chain.MEDIA_SCHEDULE}`, drift bad frames excluded,
frame_min {chain.FRAME_MIN}, pixel {chain.PIXEL_UM} um). Published {manifest['published']}.

This directory is READ-ONLY. Do not edit in place; publish a new tag instead.
All downstream analyses should read from here (qpi_paths picks LATEST automatically).

## Load

```python
import pandas as pd
df = pd.read_csv(r"{dest / 'consolidated' / 'all_cells_lineage_data3D.csv.gz'}")
mother = df[df["rank"] == 1]                 # central (mother) cell per channel
good = df[~(df.is_outlier | df.touches_border)]
```

Column definitions: `SCHEMA.md`. Per-channel verbatim outputs: `per_channel/PosN/chNN/`.
Mask pixels for any row: `inputs`-independent - open
`{chain.MASK_ROOT}\\PosN\\...\\chNN\\inference_out\\img_<frame>_ph_000_phase_masks.tif` and select `mask_label`.

## Integrity

`SHA256SUMS.txt` lists every file. Verify with `sha256sum -c SHA256SUMS.txt` (Git Bash).
"""
    (dest / "README.md").write_text(readme, encoding="utf-8")

    # derived publication dataset: first 7 days in 2% glucose (img_0002..img_2017).
    # Never let this step block the master itself.
    try:
        import build_phase1_dataset_260517 as ph1
        ph1.build(dest / "consolidated", inputs, dest / "derived" / ph1.dataset_name(),
                  derived_from=tag, qc_dir=dest / "qc", log=_log)
    except Exception as e:
        _log(f"WARN: phase-1 derived dataset failed: {e!r}")

    # checksums (everything except the sums file itself)
    lines = []
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            lines.append(f"{_sha256(p)} *{p.relative_to(dest).as_posix()}")
    # LF only (write_text would emit CRLF on Windows and `sha256sum -c` then fails)
    with (dest / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    _log(f"hashed {len(lines)} files")

    if readonly:
        _set_readonly_tree(dest)
    (dest_root / "LATEST.txt").write_text(tag + "\n", encoding="utf-8")
    _log(f"LATEST -> {tag}")

    if mirror:
        if MIRROR_ROOT.parent.parent.exists():
            mdst = MIRROR_ROOT / tag
            mdst.mkdir(parents=True, exist_ok=True)
            n = 0
            for p in dest.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(dest)
                if rel.parts[0] == "per_channel":
                    continue  # compact mirror: consolidated tables carry every row
                _copy(p, mdst / rel)
                n += 1
            (MIRROR_ROOT / "LATEST.txt").write_text(tag + "\n", encoding="utf-8")
            _log(f"mirrored {n} files -> {mdst}")
        else:
            _log(f"WARN: mirror root not reachable, skipped: {MIRROR_ROOT}")

    total_mb = sum(sizes.values()) / 1e6
    _log(f"DONE master {tag}: {len(chans)} channels, {total_mb:.0f} MB at {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None, help="version tag (default v<YYYYMMDD>_newmodel)")
    ap.add_argument("--dest", default=str(MASTER_ROOT), help="master root (default D:\\QPI_master\\260517)")
    ap.add_argument("--no-mirror", action="store_true", help="skip the G: mirror")
    ap.add_argument("--no-readonly", action="store_true", help="do not set the read-only attribute")
    ap.add_argument("--force", action="store_true", help="replace an existing tag")
    args = ap.parse_args()
    tag = args.tag or time.strftime("v%Y%m%d_newmodel")
    publish(tag, Path(args.dest), mirror=not args.no_mirror,
            readonly=not args.no_readonly, force=args.force)


if __name__ == "__main__":
    main()
