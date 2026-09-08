"""Quick: list a channel's cycle rows (frames) and check a custom frame range."""
import sys
import numpy as np
import pandas as pd
from _fig_panelA_cellcycle import data_root, load_bad_frames, enumerate_all_cycles

pos, ch = sys.argv[1], sys.argv[2]
rng = sys.argv[3] if len(sys.argv) > 3 else None
cd = data_root(pos) / ch
dd = pd.read_csv(cd / "inference_out" / "lineage_out" / "lineage_data3D.csv")
mm = dd[dd["rank"] == 1].sort_values("frame").reset_index(drop=True)
bad = load_bad_frames(cd, pos)
cycles, sk = enumerate_all_cycles(mm, bad, min_kept=6)
print(f"{pos} {ch}  n_cycles={len(cycles)}")
for i, c in enumerate(cycles, 1):
    f0, f1 = int(c["frame"].min()), int(c["frame"].max())
    t0 = float(c["time_h"].iloc[0])
    print(f"row {i:2d}  frames {f0}-{f1}  ({f1 - f0 + 1} fr)  t0={t0:.1f}h")
if rng:
    a, b = (int(x) for x in rng.split("-"))
    seg = mm[(mm["frame"] >= a) & (mm["frame"] <= b)]
    if len(seg):
        print(f"range {a}-{b}: {len(seg)} frames present, "
              f"long_axis {seg['long_axis_um'].min():.1f}-{seg['long_axis_um'].max():.1f} um, "
              f"t {seg['time_h'].iloc[0]:.1f}-{seg['time_h'].iloc[-1]:.1f} h")
    else:
        print(f"range {a}-{b}: NO frames present")
