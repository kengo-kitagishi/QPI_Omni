"""Quality check across ALL mother-revived candidates from the YAML.

Goal: find the "gold standard" subset — channels where the mother kept a steady
~3.25 h division interval throughout phase1 with NO deviation. Compare against
all mother-revived candidates (much larger than the 30 used in earlier overlays).

Reads lineage CSVs from the latest per_channel_figures inbox run for each
(Pos, ch). Skips channels with no inbox run.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from per_channel_figures import mother_division_frames  # noqa: E402
from overlay_mother_revived_vs_dead import find_latest_run_for_channel  # noqa: E402

YAML_PATH = Path("/Users/kitak/QPI_Omni/docs/channel_classification_260517.yaml")
OUT_CSV = Path("/Users/kitak/QPI_Omni/results/260517/quality_check_all_mother_revived.csv")
TIME_INTERVAL_MIN = 5.0
# Gold standard: every inter-division interval must be in [GOLD_MIN_H, GOLD_MAX_H]
# 2.5–5.0 h allows for stochastic variation around the ~3.25 h fission yeast cycle
# in 2% glucose without admitting clear pauses or premature blob splits.
GOLD_MIN_H = 2.5
GOLD_MAX_H = 5.0
PHASE1_FRAMES = 2019  # restrict division counts to phase1 window


def classify_mother_fate(phase2_outcome: str | None, phase2_notes: str | None) -> str:
    """Return 'revived' if mother is revived, 'dead' if not, 'unknown' otherwise."""
    if phase2_outcome == "revived":
        return "revived"
    if phase2_outcome in ("died_starvation", "never_revived"):
        return "dead"
    if phase2_outcome != "mixed":
        return "unknown"
    notes = (phase2_notes or "").lstrip()
    first_line = notes.split("\n")[0].strip()
    # match: "mother: revived" or "mother + 2番目: revived" etc.
    m = re.match(r"^mother[^:]*:\s*(revived|never_revived|dead)", first_line)
    if m:
        return "revived" if m.group(1) == "revived" else "dead"
    if first_line.startswith("全細胞 revived") or first_line.startswith("全 細胞 revived"):
        return "revived"
    # try last resort: "mother + xxx (全部生存)" patterns are too varied; mark unknown
    return "unknown"


def list_phase1_survivors() -> list[tuple[str, str]]:
    """Walk YAML, return (pos, ch) for status=cells & phase1.outcome=alive.

    Phase2 outcome is NOT filtered — the question is "did the mother live
    through phase1?", regardless of what happened later.
    """
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
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


def intervals_h(birth_frames: np.ndarray) -> np.ndarray:
    if len(birth_frames) < 2:
        return np.array([])
    return np.diff(np.sort(birth_frames)) * TIME_INTERVAL_MIN / 60.0


def main():
    candidates = list_phase1_survivors()
    print(f"Phase1 survivors (status=cells, phase1=alive) in YAML: {len(candidates)}")
    rows = []
    for pos, ch in candidates:
        result = find_latest_run_for_channel(pos, ch)
        if result is None:
            rows.append({"channel": f"{pos}_{ch}", "status": "no_inbox_run"})
            continue
        run_dir, _ = result
        clist_path = run_dir / "clist.csv"
        lineage_path = run_dir / "lineage_data3D.csv"
        if not (clist_path.exists() and lineage_path.exists()):
            rows.append({"channel": f"{pos}_{ch}", "status": "missing_csv"})
            continue
        clist = pd.read_csv(clist_path)
        lineage = pd.read_csv(lineage_path)
        # restrict to phase1
        divs_all = mother_division_frames(clist, lineage=lineage, exclude_bad_frames=True)
        divs = divs_all[divs_all < PHASE1_FRAMES]
        intervals = intervals_h(divs)
        if len(intervals) == 0:
            rows.append({"channel": f"{pos}_{ch}", "status": "no_divisions",
                         "n_divisions_phase1": int(len(divs))})
            continue
        # gold-standard: every interval inside [GOLD_MIN_H, GOLD_MAX_H]
        n_below = int(np.sum(intervals < GOLD_MIN_H))
        n_above = int(np.sum(intervals > GOLD_MAX_H))
        rows.append({
            "channel": f"{pos}_{ch}",
            "status": "ok",
            "n_divisions_phase1": int(len(divs)),
            "median_interval_h": float(np.median(intervals)),
            "min_interval_h": float(intervals.min()),
            "max_interval_h": float(intervals.max()),
            "n_intervals_below_min": n_below,
            "n_intervals_above_max": n_above,
            "is_gold_standard": (n_below == 0) and (n_above == 0),
        })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    ok = df[df["status"] == "ok"]
    print(f"\nOf {len(candidates)} candidates:")
    print(f"  - inbox run found and lineage usable: {len(ok)}")
    print(f"  - no inbox run / no divisions / missing csv: {len(df) - len(ok)}")

    if len(ok):
        print(f"\nAmong the {len(ok)} with usable lineage:")
        print(f"  - GOLD STANDARD (zero abnormal intervals): "
              f"{int(ok['is_gold_standard'].sum())}")
        print(f"  - has at least one abnormal interval: "
              f"{int((~ok['is_gold_standard']).sum())}")
        print(f"\nDivision count distribution (phase1):")
        print(f"  - median: {int(ok['n_divisions_phase1'].median())}")
        print(f"  - 25%-75%: [{int(ok['n_divisions_phase1'].quantile(0.25))}, "
              f"{int(ok['n_divisions_phase1'].quantile(0.75))}]")
        print(f"  - min..max: {ok['n_divisions_phase1'].min()}..{ok['n_divisions_phase1'].max()}")

        gold = ok[ok["is_gold_standard"]]
        if len(gold):
            print(f"\nGold-standard channels ({len(gold)}):")
            print(gold[["channel", "n_divisions_phase1", "median_interval_h",
                        "max_interval_h"]].to_string(index=False))

    print(f"\nFull report: {OUT_CSV}")


if __name__ == "__main__":
    main()
