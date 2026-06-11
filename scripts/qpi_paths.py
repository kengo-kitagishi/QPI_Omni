"""qpi_paths.py — cross-platform path resolution for 260517 analysis scripts.

The analysis was originally developed on macOS, where lineage CSVs, the
figure-hub inbox, the channel-classification YAML and the results/ directory
all lived under hardcoded ``/Users/kitak/...`` paths. On the Windows analysis
PC the same data is reached differently:

  - figure-hub inbox: Google Drive for Desktop shared drive (G:/H:/F:/I:)
  - raw masks: external drive (F:), recorded per channel in
    ``lineage_run_params.json["channel_dir"]``
  - YAML / results: inside the repository itself

This module centralises that resolution so the downstream scripts can be made
portable without scattering platform branches everywhere. The figure-hub inbox
root is resolved by reusing ``figure_logger._resolve_inbox_root`` (which already
handles the NFC/NFD Unicode quirk of "共有ドライブ" on Windows NTFS).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import _resolve_inbox_root  # noqa: E402


# ---------------------------------------------------------------------------
# Repo-relative resources
# ---------------------------------------------------------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def yaml_path() -> Path:
    """Channel-classification YAML (checked into the repo)."""
    return repo_root() / "docs" / "channel_classification_260517.yaml"


def results_dir() -> Path:
    """Local results dir for 260517 (created on demand)."""
    d = repo_root() / "results" / "260517"
    d.mkdir(parents=True, exist_ok=True)
    return d


def inbox_root() -> Path:
    """figure-hub inbox root on this machine (G:/... on Windows)."""
    return _resolve_inbox_root()


# ---------------------------------------------------------------------------
# Lineage CSV discovery across the inbox (both naming conventions)
# ---------------------------------------------------------------------------
def _iter_date_subdirs(subdir_name: str):
    """Yield every <inbox>/<YYYY-MM-DD>/<subdir_name> directory that exists."""
    root = inbox_root()
    if not root.exists():
        return
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        cand = date_dir / subdir_name
        if cand.exists():
            yield cand


@lru_cache(maxsize=1)
def build_lineage_index() -> dict:
    """Map (pos, ch) -> latest lineage_data3D.csv path across the whole inbox.

    Scans both inbox layouts:
      * per_channel_figures: one run dir per channel; the channel label lives
        in the f001.json metadata (params.channel_label).
      * batch_figures: flat run dirs with files named
        ``<Pos>_<ch>__lineage_data3D.csv``.

    "Latest" is decided by file mtime so a newer rerun supersedes an older one.
    """
    import json

    index: dict[tuple[str, str], tuple[float, Path]] = {}

    def _consider(key: tuple[str, str], csv_path: Path):
        try:
            m = csv_path.stat().st_mtime
        except OSError:
            return
        prev = index.get(key)
        if prev is None or m > prev[0]:
            index[key] = (m, csv_path)

    # per_channel_figures
    for pcf in _iter_date_subdirs("per_channel_figures"):
        for run_dir in pcf.iterdir():
            if not run_dir.is_dir():
                continue
            csv_path = run_dir / "lineage_data3D.csv"
            if not csv_path.exists():
                continue
            label = None
            for jp in run_dir.glob("*f001.json"):
                try:
                    meta = json.loads(jp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                label = meta.get("params", {}).get("channel_label")
                break
            if not label or "_" not in label:
                continue
            pos, ch = label.split("_", 1)
            _consider((pos, ch), csv_path)

    # batch_figures
    for bf in _iter_date_subdirs("batch_figures"):
        for run_dir in bf.iterdir():
            if not run_dir.is_dir():
                continue
            for csv_path in run_dir.glob("*__lineage_data3D.csv"):
                stem = csv_path.name[: -len("__lineage_data3D.csv")]
                if "_" not in stem:
                    continue
                pos, ch = stem.split("_", 1)
                _consider((pos, ch), csv_path)

    return {k: v[1] for k, v in index.items()}


def find_lineage_csv(pos: str, ch: str) -> Path | None:
    """Latest lineage_data3D.csv for (pos, ch), or None if absent."""
    return build_lineage_index().get((pos, ch))


# ---------------------------------------------------------------------------
# Corrected (mask-direct) lineage resolution
# ---------------------------------------------------------------------------
# Downstream scripts pick up corrected mother axes/volume/RI/mass transparently
# when QPI_USE_CORRECTED=1. The volume variant (rod | profile) is chosen with
# QPI_VOLUME_VARIANT so the same script renders either set without code changes.
def use_corrected() -> bool:
    return os.environ.get("QPI_USE_CORRECTED", "").strip() in {"1", "true", "yes", "on"}


def volume_variant() -> str:
    v = os.environ.get("QPI_VOLUME_VARIANT", "rod").strip().lower()
    return v if v in {"rod", "profile"} else "rod"


def corrected_run_dir(pos: str, ch: str, variant: str | None = None) -> Path | None:
    """results/260517/corrected_lineage_<variant>/<pos>_<ch> if it exists."""
    variant = variant or volume_variant()
    d = results_dir() / f"corrected_lineage_{variant}" / f"{pos}_{ch}"
    return d if (d / "lineage_data3D.csv").exists() else None


def find_corrected_lineage_csv(pos: str, ch: str,
                               variant: str | None = None) -> Path | None:
    d = corrected_run_dir(pos, ch, variant)
    return (d / "lineage_data3D.csv") if d is not None else None


def resolve_lineage_csv(pos: str, ch: str) -> Path | None:
    """Corrected CSV when QPI_USE_CORRECTED=1 and it exists, else the inbox CSV.

    This is the single entry point downstream scripts should call so a run can
    be flipped between raw and corrected data with one env var.
    """
    if use_corrected():
        c = find_corrected_lineage_csv(pos, ch)
        if c is not None:
            return c
    return find_lineage_csv(pos, ch)


def find_run_params(csv_path: Path) -> Path | None:
    """Locate the lineage_run_params.json that goes with a lineage CSV.

    per_channel_figures: sibling ``lineage_run_params.json``.
    batch_figures: sibling ``<Pos>_<ch>__lineage_run_params.json``.
    """
    sib = csv_path.parent / "lineage_run_params.json"
    if sib.exists():
        return sib
    name = csv_path.name
    if name.endswith("__lineage_data3D.csv"):
        prefix = name[: -len("lineage_data3D.csv")]
        cand = csv_path.parent / f"{prefix}lineage_run_params.json"
        if cand.exists():
            return cand
    return None


def find_clist_csv(csv_path: Path) -> Path | None:
    """Locate the clist.csv sibling of a lineage CSV (both layouts)."""
    sib = csv_path.parent / "clist.csv"
    if sib.exists():
        return sib
    name = csv_path.name
    if name.endswith("__lineage_data3D.csv"):
        prefix = name[: -len("lineage_data3D.csv")]
        cand = csv_path.parent / f"{prefix}clist.csv"
        if cand.exists():
            return cand
    return None


def mask_dir_for_channel(csv_path: Path) -> Path | None:
    """Resolve the inference_out directory holding per-frame mask TIFFs.

    Reads ``channel_dir`` from the run params (an absolute path on the machine
    that produced the lineage, e.g. ``F:\\260517\\...\\ch06``) and appends
    ``inference_out``. Returns None if the params or directory are missing.
    """
    import json

    params_path = find_run_params(csv_path)
    if params_path is None:
        return None
    try:
        params = json.loads(params_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    ch_dir = params.get("channel_dir")
    if not ch_dir:
        return None
    inf = Path(ch_dir) / "inference_out"
    return inf if inf.exists() else None


if __name__ == "__main__":
    idx = build_lineage_index()
    print(f"inbox root: {inbox_root()}")
    print(f"lineage channels indexed: {len(idx)}")
    for key in sorted(idx)[:5]:
        print(f"  {key[0]}_{key[1]} -> {idx[key]}")
    # probe the pilot channel
    p = find_lineage_csv("Pos27", "ch06")
    print(f"\nPos27_ch06 lineage: {p}")
    if p is not None:
        print(f"  params: {find_run_params(p)}")
        print(f"  clist:  {find_clist_csv(p)}")
        print(f"  masks:  {mask_dir_for_channel(p)}")
