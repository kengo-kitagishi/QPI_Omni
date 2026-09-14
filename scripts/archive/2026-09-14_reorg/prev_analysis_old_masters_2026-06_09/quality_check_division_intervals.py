"""Quick quality check: do mothers maintain ~3.2 h division interval?

For each well-tracked channel (phase1-dead with YAML death_frame, plus revived
controls), compute mother inter-division intervals and flag deviation from the
"normal" baseline (~3.2 h).

Three buckets of interest:
  A. phase1-survivors (`revived` group, mother stayed alive through phase1):
     baseline — how stable is the 3.2 h interval across the 30 channels?
  B. phase1-dead (mother died during phase1):
     pre-death intervals — are they normal up until death, or did they go
     abnormal before the user-reported death_frame?
  C. abnormal-before-death channels: candidates where the YAML death call
     and the division-interval evidence may disagree.

Definition of "abnormal interval": > 5 h (≈ 1.5x the median ~3.2 h).
This is a heuristic; the user trusts their YAML labels most, so this script
just reports the disagreement rate.

Reads CSVs from results/260517/phase1_2per_7days/per_cell_data/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from per_channel_figures import mother_division_frames  # noqa: E402

SCOPE_DIR = Path("/Users/kitak/QPI_Omni/results/260517/phase1_2per_7days/per_cell_data")
TIME_INTERVAL_MIN = 5.0
ABNORMAL_GAP_H = 5.0  # gap > 5 h flagged as abnormal


def intervals_h(birth_frames: np.ndarray) -> np.ndarray:
    if len(birth_frames) < 2:
        return np.array([])
    diffs_frames = np.diff(np.sort(birth_frames))
    return diffs_frames * TIME_INTERVAL_MIN / 60.0


def load_channel(group_dir: Path, label: str):
    d = group_dir / label
    clist = pd.read_csv(d / "clist.csv")
    lineage = pd.read_csv(d / "lineage_data3D.csv")
    death_frame = None
    df_path = d / "death_frame.txt"
    if df_path.exists():
        death_frame = int(df_path.read_text().strip())
    return clist, lineage, death_frame


def analyze_group(group_label: str) -> pd.DataFrame:
    rows = []
    group_dir = SCOPE_DIR / group_label
    if not group_dir.exists():
        return pd.DataFrame()
    for ch_dir in sorted(group_dir.iterdir()):
        if not ch_dir.is_dir():
            continue
        label = ch_dir.name
        clist, lineage, death_frame = load_channel(group_dir, label)
        divs = mother_division_frames(clist, lineage=lineage, exclude_bad_frames=True)
        intervals = intervals_h(divs)
        if len(intervals) == 0:
            rows.append({"channel": label, "group": group_label,
                         "n_divisions": int(len(divs)),
                         "death_frame": death_frame,
                         "median_interval_h": np.nan,
                         "max_interval_h": np.nan,
                         "n_abnormal_intervals": 0,
                         "abnormal_before_death": False,
                         "abnormal_after_death": False})
            continue

        # split intervals into "before death" / "after death" if death_frame given
        before_mask = np.ones(len(intervals), dtype=bool)
        after_mask = np.zeros(len(intervals), dtype=bool)
        if death_frame is not None:
            # interval i is the gap between divs[i] and divs[i+1]
            # consider the right-edge of each interval
            right_edges = divs[1:]
            before_mask = right_edges <= death_frame
            after_mask = right_edges > death_frame

        n_abnormal = int(np.sum(intervals > ABNORMAL_GAP_H))
        before_abnormal = bool(np.any(intervals[before_mask] > ABNORMAL_GAP_H))
        after_abnormal = bool(np.any(intervals[after_mask] > ABNORMAL_GAP_H))

        rows.append({
            "channel": label,
            "group": group_label,
            "n_divisions": int(len(divs)),
            "death_frame": death_frame,
            "median_interval_h": float(np.median(intervals)),
            "max_interval_h": float(intervals.max()),
            "n_abnormal_intervals": n_abnormal,
            "n_before_death_intervals": int(before_mask.sum()),
            "n_after_death_intervals": int(after_mask.sum()),
            "abnormal_before_death": before_abnormal,
            "abnormal_after_death": after_abnormal,
        })
    return pd.DataFrame(rows)


def main():
    revived_df = analyze_group("revived")
    dead_df = analyze_group("phase1_dead")

    print("\n=== Group: revived (phase1 survivors) ===")
    if not revived_df.empty:
        print(revived_df[["channel", "n_divisions", "median_interval_h",
                          "max_interval_h", "n_abnormal_intervals"]]
              .to_string(index=False))
        ok = (revived_df["n_abnormal_intervals"] == 0).sum()
        print(f"\nChannels with zero abnormal intervals: {ok}/{len(revived_df)}")
        print(f"Median of medians: "
              f"{revived_df['median_interval_h'].median():.2f} h")

    print("\n\n=== Group: phase1_dead (mother died in phase1) ===")
    if not dead_df.empty:
        cols = ["channel", "death_frame", "n_divisions", "median_interval_h",
                "max_interval_h", "n_before_death_intervals",
                "abnormal_before_death", "abnormal_after_death"]
        print(dead_df[cols].to_string(index=False))
        n_pre = dead_df["abnormal_before_death"].sum()
        n_only_post = ((~dead_df["abnormal_before_death"])
                       & (dead_df["abnormal_after_death"])).sum()
        n_clean = ((~dead_df["abnormal_before_death"])
                   & (~dead_df["abnormal_after_death"])).sum()
        print(f"\nPre-death abnormal intervals : {n_pre}/{len(dead_df)} "
              f"(potential YAML/data disagreement)")
        print(f"Only post-death abnormal     : {n_only_post}/{len(dead_df)} "
              f"(consistent with YAML death)")
        print(f"No abnormal at all           : {n_clean}/{len(dead_df)}")

    # Save full report
    out = SCOPE_DIR.parent / "quality_check_division_intervals.csv"
    pd.concat([revived_df, dead_df], ignore_index=True).to_csv(out, index=False)
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
