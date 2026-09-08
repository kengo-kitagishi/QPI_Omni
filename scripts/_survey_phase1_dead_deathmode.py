"""Survey phase1-dead channels by death mode (elongation vs normal).

For each phase1-dead channel (mother lineage ends in a death window before the
2% phase end, frame ~2018, instead of dividing), compute:
  - last division frame and the length at that last division
  - the post-last-cycle death window
  - the maximum long axis reached after the last division (filament length)
  - elongation ratio = death_maxlong / division_length

High ratio  -> elongation/filamentation death (the Pos20 ch06 keeper type).
Ratio ~1    -> "normal" death: cell stops near division length and lyses.

Output is a ranked table so we can pick representative NON-elongation deaths
as the "other phase1-dead" examples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _fig_panelA_cellcycle import (
    data_root, load_pixel_size, load_bad_frames, enumerate_all_cycles,
)

# the 26 phase1-dead channels (from the channel-all sheets in the inbox)
CHANNELS = [
    ("Pos1", "ch09"), ("Pos2", "ch03"), ("Pos2", "ch08"), ("Pos5", "ch08"),
    ("Pos6", "ch09"), ("Pos9", "ch04"), ("Pos9", "ch08"), ("Pos10", "ch07"),
    ("Pos11", "ch06"), ("Pos14", "ch08"), ("Pos17", "ch02"), ("Pos17", "ch09"),
    ("Pos18", "ch02"), ("Pos20", "ch06"), ("Pos26", "ch03"), ("Pos26", "ch08"),
    ("Pos30", "ch04"), ("Pos31", "ch06"), ("Pos32", "ch05"), ("Pos35", "ch05"),
    ("Pos37", "ch08"), ("Pos37", "ch11"), ("Pos39", "ch05"), ("Pos42", "ch01"),
    ("Pos45", "ch00"), ("Pos45", "ch02"),
]

MAX_FRAME = 2018      # cycles only count as "phase 1" if they end by here
DEATH_WIN = 400       # follow the dying cell this many frames past last division


def survey_one(pos: str, ch: str):
    ch_dir = data_root(pos) / ch
    csv = ch_dir / "inference_out" / "lineage_out" / "lineage_data3D.csv"
    if not csv.exists():
        return None
    dd = pd.read_csv(csv)
    mm = dd[dd["rank"] == 1].sort_values("frame").reset_index(drop=True)
    bad = load_bad_frames(ch_dir, pos)
    cycles, _ = enumerate_all_cycles(mm, bad, min_kept=6)
    if not cycles:
        return None
    last_cycle = cycles[-1]
    last_end = int(last_cycle["frame"].max())
    div_len = float(last_cycle["long_axis_um"].iloc[-1])  # length at last division

    # everything after the last division up to the phase-1 end
    after = mm[(mm["frame"] > last_end) & (mm["frame"] <= MAX_FRAME)]
    if "is_outlier" in after:
        after = after[~after["is_outlier"].to_numpy(dtype=bool)]
    after = after[~after["frame"].isin(bad)]
    if len(after) < 4:
        return None
    L = after["long_axis_um"].to_numpy()
    death_maxlong = float(np.nanmax(L))
    peak_frame = int(after["frame"].iloc[int(np.nanargmax(L))])
    peak_time = float(after["time_h"].iloc[int(np.nanargmax(L))])
    ratio = death_maxlong / max(div_len, 1e-6)

    # a death-window range good for a single strip: from just after the last
    # division to a little past the peak filament length
    f_lo = last_end + 1
    f_hi = min(peak_frame + 8, int(after["frame"].max()))
    return {
        "pos": pos, "ch": ch, "last_div_frame": last_end,
        "div_len_um": round(div_len, 1), "death_maxlong_um": round(death_maxlong, 1),
        "ratio": round(ratio, 2), "peak_frame": peak_frame,
        "peak_time_h": round(peak_time, 1), "n_after": len(after),
        "strip_range": f"{pos}:{ch}:{f_lo}-{f_hi}",
    }


def main():
    rows = []
    for pos, ch in CHANNELS:
        try:
            r = survey_one(pos, ch)
        except Exception as e:
            print(f"  [skip] {pos} {ch}: {e}")
            r = None
        if r:
            rows.append(r)
    df = pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
