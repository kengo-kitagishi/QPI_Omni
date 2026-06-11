"""gold_standard.py — portable gold-standard mother selection for 260517.

Self-contained so it works on the Windows analysis PC without the macOS
hardcoded paths in overlay_gold_standard_and_phase1_dead / quality_check_*.

Gold-standard definition (matches the established criterion): a phase1
survivor whose every inter-division interval in phase1 lies in
[GOLD_MIN_H, GOLD_MAX_H], minus the visually flagged MANUAL_EXCLUDE channels.

Divisions are detected with ``rank1_division_frames`` (the issue-recommended
function — it unions rank=1 cell_id switches with clist daughter births and so
does not miss divisions after the mother's cell_id=0 ends, unlike the older
``mother_division_frames``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from qpi_paths import find_lineage_csv, find_clist_csv, yaml_path  # noqa: E402

TIME_INTERVAL_MIN = 5.0
GOLD_MIN_H = 2.5
GOLD_MAX_H = 5.0
PHASE1_FRAMES = 2019  # divisions counted only below this frame

# Visually flagged bad channels (copied from
# overlay_gold_standard_and_phase1_dead.MANUAL_EXCLUDE, 2026-06-10).
MANUAL_EXCLUDE = {
    ("Pos37", "ch11"), ("Pos17", "ch11"), ("Pos22", "ch11"), ("Pos24", "ch00"),
    ("Pos36", "ch11"), ("Pos6", "ch06"), ("Pos10", "ch03"), ("Pos10", "ch00"),
    ("Pos13", "ch11"), ("Pos15", "ch01"), ("Pos18", "ch01"), ("Pos20", "ch11"),
    ("Pos24", "ch03"), ("Pos26", "ch05"), ("Pos26", "ch11"), ("Pos31", "ch11"),
    ("Pos14", "ch07"), ("Pos36", "ch05"), ("Pos38", "ch11"), ("Pos39", "ch11"),
    ("Pos40", "ch11"), ("Pos41", "ch11"), ("Pos43", "ch00"), ("Pos44", "ch02"),
    ("Pos44", "ch09"), ("Pos46", "ch00"), ("Pos39", "ch00"), ("Pos40", "ch03"),
    ("Pos38", "ch00"), ("Pos42", "ch00"), ("Pos23", "ch01"), ("Pos32", "ch02"),
    ("Pos30", "ch07"), ("Pos7", "ch06"), ("Pos32", "ch03"),
}


def rank1_division_frames(df: pd.DataFrame,
                          clist: pd.DataFrame | None = None) -> list[int]:
    """Mother-position divisions, unioning rank=1 cell_id switches with clist
    daughter births of any cell that ever held rank=1."""
    divs: set[int] = set()
    m = df[df["rank"] == 1].sort_values("frame")
    if m.empty:
        return []
    cells = m["cell_id"].to_numpy()
    frames = m["frame"].to_numpy()
    rank1_ids = set(int(c) for c in np.unique(cells))
    for i in range(1, len(cells)):
        if cells[i] != cells[i - 1]:
            divs.add(int(frames[i]))
    if clist is not None and "mother_id" in clist.columns:
        born = clist[clist["mother_id"].isin(list(rank1_ids))]
        for bf in born["birth_frame"].astype(int):
            divs.add(int(bf))
    return sorted(divs)


def death_frame_proxy(pos: str, ch: str) -> int | None:
    """Proxy phase1 death frame = last frame the rank=1 mother is tracked.

    The YAML carries only outcome+notes (no death_frame), and the original
    death_frame.txt annotations are macOS-local. For a phase1-dead mother the
    tracked rank=1 trajectory ends at death, so its last frame is a reasonable
    proxy. Returns None if no lineage / no mother rows.
    """
    csv = find_lineage_csv(pos, ch)
    if csv is None:
        return None
    try:
        df = pd.read_csv(csv, usecols=["rank", "frame"])
    except Exception:
        return None
    m = df[df["rank"] == 1]
    if m.empty:
        return None
    return int(m["frame"].max())


def phase1_dead_sorted_by_death_proxy() -> list[tuple[str, str, int]]:
    """(pos, ch, death_frame_proxy) for phase1-dead channels, sorted ascending,
    MANUAL_EXCLUDE removed. Death frame is a lineage-derived proxy."""
    out = []
    for pos, ch in list_phase1_dead():
        if (pos, ch) in MANUAL_EXCLUDE:
            continue
        d = death_frame_proxy(pos, ch)
        if d is None:
            continue
        out.append((pos, ch, d))
    return sorted(out, key=lambda t: t[2])


def intervals_h(div_frames) -> np.ndarray:
    d = np.asarray(sorted(div_frames), dtype=float)
    if d.size < 2:
        return np.array([])
    return np.diff(d) * TIME_INTERVAL_MIN / 60.0


def list_phase1_survivors() -> list[tuple[str, str]]:
    """(pos, ch) where YAML status=cells and phase1.outcome=alive."""
    data = yaml.safe_load(yaml_path().read_text(encoding="utf-8"))
    out = []
    for pos_name, channels in (data.get("positions") or {}).items():
        if not channels:
            continue
        for ch_name, fields in channels.items():
            if not fields or fields.get("status") != "cells":
                continue
            p1 = fields.get("phase1") or {}
            if p1.get("outcome") != "alive":
                continue
            out.append((pos_name, ch_name))
    return out


def list_all_cells() -> list[tuple[str, str]]:
    """Every (pos, ch) with YAML status=cells (any phase1/phase2 outcome).

    Superset cohort covering gold / revived / never-revived / phase1-dead, used
    to recompute everything any downstream figure might need."""
    data = yaml.safe_load(yaml_path().read_text(encoding="utf-8"))
    out = []
    for pos_name, channels in (data.get("positions") or {}).items():
        if not channels:
            continue
        for ch_name, fields in channels.items():
            if fields and fields.get("status") == "cells":
                out.append((pos_name, ch_name))
    return out


def figure_cohort() -> list[tuple[str, str]]:
    """Union of every channel any downstream figure needs:
    gold-standard ∪ revived ∪ never-revived ∪ phase1-dead.

    phase2-dead mothers are a subset of never-revived (alive in phase1, did not
    revive), so they are already covered. This is the exact recompute set —
    tighter than all status=cells (which includes channels no figure plots)."""
    from overlay_mean_sd_band_full_timecourse import (
        list_revived_mothers, list_never_revived_mothers,
    )
    union = set(select_gold_standard())
    union |= set(list_revived_mothers())
    union |= set(list_never_revived_mothers())
    union |= set(list_phase1_dead())
    return sorted(union, key=lambda t: (t[0], t[1]))


def list_phase1_dead() -> list[tuple[str, str]]:
    """(pos, ch) where YAML phase1.outcome indicates the mother died in phase1."""
    data = yaml.safe_load(yaml_path().read_text(encoding="utf-8"))
    out = []
    for pos_name, channels in (data.get("positions") or {}).items():
        if not channels:
            continue
        for ch_name, fields in channels.items():
            if not fields or fields.get("status") != "cells":
                continue
            if (fields.get("phase1") or {}).get("outcome") == "dead":
                out.append((pos_name, ch_name))
    return out


def select_gold_standard(verbose: bool = False) -> list[tuple[str, str]]:
    """Phase1 survivors with every phase1 division interval in
    [GOLD_MIN_H, GOLD_MAX_H], minus MANUAL_EXCLUDE."""
    out = []
    n_candidates = 0
    for pos, ch in list_phase1_survivors():
        if (pos, ch) in MANUAL_EXCLUDE:
            continue
        csv_path = find_lineage_csv(pos, ch)
        if csv_path is None:
            continue
        n_candidates += 1
        try:
            df = pd.read_csv(csv_path)
            clist_path = find_clist_csv(csv_path)
            clist = pd.read_csv(clist_path) if clist_path else None
        except Exception:
            continue
        divs = [d for d in rank1_division_frames(df, clist) if d < PHASE1_FRAMES]
        iv = intervals_h(divs)
        if iv.size == 0:
            continue
        if iv.min() >= GOLD_MIN_H and iv.max() <= GOLD_MAX_H:
            out.append((pos, ch))
    if verbose:
        print(f"phase1 survivors with lineage: {n_candidates}")
        print(f"gold-standard (all intervals in [{GOLD_MIN_H},{GOLD_MAX_H}] h): {len(out)}")
    return out


if __name__ == "__main__":
    gold = select_gold_standard(verbose=True)
    for pos, ch in gold:
        print(f"  {pos}_{ch}")
