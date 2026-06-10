"""Split per-channel lineage CSVs into two analysis scopes.

For each (Pos, ch) with a latest run in the per_channel_figures inbox:
  - load lineage_data3D.csv and clist.csv from that run
  - filter rows by time window and save into two result subfolders:
      results/260517/phase1_2per_7days/per_cell_data/<Pos>_<ch>/
      results/260517/starvation_to_recovery/per_cell_data/<Pos>_<ch>/
  - the latter also exports a `time_h_shifted` column (= time_h - 120)

This gives a clean local CSV workspace for downstream analysis without needing
to re-run the upstream Windows pipeline.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from overlay_mother_revived_vs_dead import (  # noqa: E402
    INBOX_DIRS, GROUP_DEAD, GROUP_REVIVED,
    find_latest_run_for_channel,
)

YAML_PATH = Path("/Users/kitak/QPI_Omni/docs/channel_classification_260517.yaml")
DEATH_FRAME_RE = re.compile(r"frame\s*(\d+)(?:[\s\-–~–]+(\d+))?")

OUT_ROOT = Path("/Users/kitak/QPI_Omni/results/260517")
PHASE1_DIR = OUT_ROOT / "phase1_2per_7days"
LATE_DIR = OUT_ROOT / "starvation_to_recovery"

# time windows (in hours)
PHASE1_T_MIN, PHASE1_T_MAX = 0.0, 168.25     # 0 → frame 2019 (= wo_2 end)
LATE_T_MIN = 120.0                            # shift origin
TIME_ZERO_LATE = 120.0


def load_phase1_dead_group() -> list[tuple[str, str, int | None]]:
    """Scan the channel-classification YAML for (Pos, ch) with mother death in phase1.

    Returns list of (pos, ch, death_frame_or_None). death_frame is parsed from
    notes like '約 frame N で死亡' or 'frame N-M で死亡' (uses the lower bound).
    """
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    out = []
    for pos_name, channels in (data.get("positions") or {}).items():
        if not isinstance(channels, dict):
            continue
        for ch_name, fields in channels.items():
            if not isinstance(fields, dict):
                continue
            if fields.get("status") != "cells":
                continue
            phase1 = fields.get("phase1") or {}
            if phase1.get("outcome") != "dead":
                continue
            notes = phase1.get("notes") or ""
            m = DEATH_FRAME_RE.search(notes)
            death_frame = int(m.group(1)) if m else None
            out.append((pos_name, ch_name, death_frame))
    return out


def copy_run_metadata(run_dir: Path, out_dir: Path) -> None:
    """Copy lineage_run_params.json if present (used by per_channel_figures plot helpers)."""
    src = run_dir / "lineage_run_params.json"
    if src.exists():
        shutil.copy(src, out_dir / "lineage_run_params.json")


def split_one(
    pos: str,
    ch: str,
    run_dir: Path,
    group_label: str,
    death_frame: int | None = None,
    only_phase1: bool = False,
) -> dict:
    """Trim CSVs from a single run, write two scoped versions, return summary.

    For `phase1_dead` group, set `only_phase1=True` to skip the late-window output
    (mother already dead before phase2). Pass `death_frame` to also persist a
    `death_frame.txt` next to the CSVs.
    """
    lineage_csv = run_dir / "lineage_data3D.csv"
    clist_csv = run_dir / "clist.csv"
    if not (lineage_csv.exists() and clist_csv.exists()):
        return {"pos": pos, "ch": ch, "status": "missing_csv"}
    lineage = pd.read_csv(lineage_csv)
    clist = pd.read_csv(clist_csv)
    if lineage.empty:
        return {"pos": pos, "ch": ch, "status": "empty_lineage"}

    label = f"{pos}_{ch}"

    # --- phase1 ---
    p1 = lineage[(lineage["time_h"] >= PHASE1_T_MIN) & (lineage["time_h"] <= PHASE1_T_MAX)].copy()
    p1_clist = clist[clist["birth_time_h"] <= PHASE1_T_MAX].copy()
    out_p1 = PHASE1_DIR / "per_cell_data" / group_label / label
    out_p1.mkdir(parents=True, exist_ok=True)
    p1.to_csv(out_p1 / "lineage_data3D.csv", index=False)
    p1_clist.to_csv(out_p1 / "clist.csv", index=False)
    copy_run_metadata(run_dir, out_p1)
    if death_frame is not None:
        (out_p1 / "death_frame.txt").write_text(str(death_frame), encoding="utf-8")

    n_rows_late = n_cells_late = 0
    if not only_phase1:
        # --- late (120h → end) with shifted time ---
        late = lineage[lineage["time_h"] >= LATE_T_MIN].copy()
        late["time_h_shifted"] = late["time_h"] - TIME_ZERO_LATE
        late_clist = clist[clist["death_time_h"] >= LATE_T_MIN].copy()
        late_clist["birth_time_h_shifted"] = late_clist["birth_time_h"] - TIME_ZERO_LATE
        late_clist["death_time_h_shifted"] = late_clist["death_time_h"] - TIME_ZERO_LATE
        out_late = LATE_DIR / "per_cell_data" / group_label / label
        out_late.mkdir(parents=True, exist_ok=True)
        late.to_csv(out_late / "lineage_data3D.csv", index=False)
        late_clist.to_csv(out_late / "clist.csv", index=False)
        copy_run_metadata(run_dir, out_late)
        n_rows_late = len(late)
        n_cells_late = late_clist["cell_id"].nunique() if not late_clist.empty else 0

    return {
        "pos": pos, "ch": ch, "status": "ok",
        "n_rows_phase1": len(p1),
        "n_cells_phase1": p1_clist["cell_id"].nunique() if not p1_clist.empty else 0,
        "n_rows_late": n_rows_late,
        "n_cells_late": n_cells_late,
        "death_frame": death_frame,
        "run_dir": str(run_dir),
    }


def main():
    summary_rows = []
    # revived / never_revived(=dead) — populate both phase1 and late scopes
    for group_label, group in [("dead", GROUP_DEAD), ("revived", GROUP_REVIVED)]:
        for pos, ch in group:
            result = find_latest_run_for_channel(pos, ch)
            if result is None:
                summary_rows.append({"pos": pos, "ch": ch, "group": group_label,
                                     "status": "no_run"})
                continue
            run_dir, _ = result
            row = split_one(pos, ch, run_dir, group_label)
            row["group"] = group_label
            summary_rows.append(row)
            print(f"[{group_label}] {pos}_{ch}: {row.get('status')} "
                  f"(phase1 rows={row.get('n_rows_phase1', '-')}, "
                  f"late rows={row.get('n_rows_late', '-')})")

    # phase1_dead — YAML-derived group, phase1 scope only (mother died in phase1)
    phase1_dead = load_phase1_dead_group()
    print(f"\n=== phase1_dead group ({len(phase1_dead)} channels from YAML) ===")
    for pos, ch, death_frame in phase1_dead:
        result = find_latest_run_for_channel(pos, ch)
        if result is None:
            summary_rows.append({"pos": pos, "ch": ch, "group": "phase1_dead",
                                 "status": "no_run", "death_frame": death_frame})
            print(f"[phase1_dead] {pos}_{ch}: no inbox run found (death_frame={death_frame})")
            continue
        run_dir, _ = result
        row = split_one(pos, ch, run_dir, "phase1_dead",
                        death_frame=death_frame, only_phase1=True)
        row["group"] = "phase1_dead"
        summary_rows.append(row)
        print(f"[phase1_dead] {pos}_{ch}: {row.get('status')} "
              f"(phase1 rows={row.get('n_rows_phase1', '-')}, "
              f"death_frame={death_frame})")

    summary_df = pd.DataFrame(summary_rows)
    for d in (PHASE1_DIR, LATE_DIR):
        summary_df.to_csv(d / "split_summary.csv", index=False)

    # write minimal READMEs documenting scope
    (PHASE1_DIR / "README.md").write_text(
        "# phase1_2per_7days\n\n"
        f"Time window: t = {PHASE1_T_MIN} h .. {PHASE1_T_MAX} h "
        f"(frames 0 .. 2019, original time, no shift).\n\n"
        "- `per_cell_data/<group>/<Pos>_<ch>/lineage_data3D.csv`: lineage rows "
        "within phase1.\n"
        "- `per_cell_data/<group>/<Pos>_<ch>/clist.csv`: cells born by end of "
        "phase1.\n"
        "- `split_summary.csv`: row counts per channel.\n\n"
        "Group folders: `dead` (mother never_revived in phase2) and "
        "`revived` (mother revived).\n",
        encoding="utf-8")
    (LATE_DIR / "README.md").write_text(
        "# starvation_to_recovery\n\n"
        f"Time window: t = {LATE_T_MIN} h .. end (original time).\n"
        f"`time_h_shifted = time_h - {TIME_ZERO_LATE}` so the first row is at 0 h.\n\n"
        "Media schedule in shifted time:\n"
        "- 0.0055% starts at ~48.25 h\n"
        "- 0% starts at ~72.25 h\n"
        "- 2% recovery starts at ~120.4 h\n\n"
        "Group folders: `dead` and `revived`.\n",
        encoding="utf-8")

    print(f"\nWrote {len(summary_rows)} entries to {PHASE1_DIR} and {LATE_DIR}")


if __name__ == "__main__":
    main()
