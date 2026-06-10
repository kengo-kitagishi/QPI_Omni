"""Check tracking coverage of phase2-dead mother channels.

For each (Pos, ch) with phase1=alive AND mother died in phase2 AND inbox
lineage exists, measure how far the rank-1 mother trace actually persists.
A channel is "usable" if the lineage rank-1 row extends close to the end of
the timecourse (or close to the user-noted death time, if any).

This lets us split the 49 candidates into:
  - tracking ends near end of timecourse (good for full-trajectory analysis)
  - tracking stops mid-phase2 (questionable — mother may have actually died
    or the tracker dropped out)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from overlay_mother_revived_vs_dead import find_latest_run_for_channel  # noqa: E402

YAML_PATH = Path("/Users/kitak/QPI_Omni/docs/channel_classification_260517.yaml")
OUT_CSV = Path("/Users/kitak/QPI_Omni/results/260517/tracking_coverage_phase2_dead_mother.csv")

TIME_INTERVAL_MIN = 5.0
PHASE1_END_FRAME = 2019           # frame at end of 2% glucose growth
RECOVERY_FRAME = 2885             # 2% recovery starts here
EXPECTED_END_FRAME = 3748         # nominal last frame
TRACK_OK_FRAMES = 3500            # last rank-1 frame >= this → tracked to end


def mother_didnt_revive_in_phase2(fields) -> bool:
    p1 = (fields.get("phase1") or {}).get("outcome")
    if p1 != "alive":
        return False
    p2 = fields.get("phase2") or {}
    p2_out = p2.get("outcome")
    if p2_out in ("died_starvation", "never_revived"):
        return True
    if p2_out != "mixed":
        return False
    notes = (p2.get("notes") or "").lstrip()
    first = notes.split("\n")[0].strip()
    m = re.match(r"^mother[^:]*:\s*(revived|never_revived|dead)", first)
    return bool(m) and m.group(1) != "revived"


def main():
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    rows = []
    for pos_name, channels in (data.get("positions") or {}).items():
        if not channels:
            continue
        for ch_name, fields in channels.items():
            if not fields or fields.get("status") != "cells":
                continue
            if not mother_didnt_revive_in_phase2(fields):
                continue
            res = find_latest_run_for_channel(pos_name, ch_name)
            if res is None:
                rows.append({"channel": f"{pos_name}_{ch_name}",
                             "status": "no_inbox_run"})
                continue
            run_dir, _ = res
            lineage_path = run_dir / "lineage_data3D.csv"
            if not lineage_path.exists():
                rows.append({"channel": f"{pos_name}_{ch_name}",
                             "status": "no_lineage_csv"})
                continue
            df = pd.read_csv(lineage_path)
            m = df[df["rank"] == 1]
            if m.empty:
                rows.append({"channel": f"{pos_name}_{ch_name}",
                             "status": "no_rank1"})
                continue
            last_frame = int(m["frame"].max())
            n_rows = len(m)
            # bad-frame rate among rank-1 rows
            bad = m["is_outlier"].astype(bool) | m["touches_border"].astype(bool)
            bad_rate = float(bad.mean()) if len(m) else 0.0
            # mass sanity (mass < 10 pg flagged)
            low_mass_rate = float((m["mass_pg"] < 10.0).mean())
            rows.append({
                "channel": f"{pos_name}_{ch_name}",
                "status": "ok",
                "last_rank1_frame": last_frame,
                "last_rank1_time_h": last_frame * TIME_INTERVAL_MIN / 60.0,
                "n_rank1_rows": n_rows,
                "bad_rate": bad_rate,
                "low_mass_rate": low_mass_rate,
                "tracked_to_end": last_frame >= TRACK_OK_FRAMES,
                "tracked_past_recovery": last_frame >= RECOVERY_FRAME,
            })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    ok = df[df["status"] == "ok"]
    print(f"Phase1-alive, mother-not-revived candidates: {len(df)}")
    print(f"  with usable lineage: {len(ok)}")
    if len(ok):
        n_to_end = int(ok["tracked_to_end"].sum())
        n_past_rec = int(ok["tracked_past_recovery"].sum())
        n_mid_p2 = int(((~ok["tracked_to_end"]) & ok["tracked_past_recovery"]).sum())
        n_pre_rec = int((~ok["tracked_past_recovery"]).sum())
        print(f"  - tracked to end (last frame >= {TRACK_OK_FRAMES} ≈ "
              f"{TRACK_OK_FRAMES*TIME_INTERVAL_MIN/60:.0f} h): {n_to_end}")
        print(f"  - tracked past recovery (>= frame {RECOVERY_FRAME} ≈ "
              f"{RECOVERY_FRAME*TIME_INTERVAL_MIN/60:.0f} h) but stops "
              f"before end: {n_mid_p2}")
        print(f"  - tracking stops before recovery (< frame {RECOVERY_FRAME}): "
              f"{n_pre_rec}")
        print(f"\nDistribution of last_rank1_frame:")
        print(ok["last_rank1_frame"].describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

        print(f"\n=== top 20 channels by last_rank1_frame ===")
        print(ok.nlargest(20, "last_rank1_frame")[
            ["channel", "last_rank1_frame", "last_rank1_time_h",
             "n_rank1_rows", "bad_rate", "low_mass_rate"]].to_string(index=False))

        print(f"\n=== bottom 20 (where tracking stops earliest) ===")
        print(ok.nsmallest(20, "last_rank1_frame")[
            ["channel", "last_rank1_frame", "last_rank1_time_h",
             "n_rank1_rows", "bad_rate", "low_mass_rate"]].to_string(index=False))

    print(f"\nFull report: {OUT_CSV}")


if __name__ == "__main__":
    main()
