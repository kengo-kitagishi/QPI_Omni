"""List a channel's division-bounded segments and flag long ones (potential
too_long_merged). Helps pick a --max-cycle-len threshold."""
import sys
import numpy as np
import pandas as pd
from _fig_panelA_cellcycle import data_root, load_bad_frames, _division_indices

pos, ch = sys.argv[1], sys.argv[2]
cd = data_root(pos) / ch
dd = pd.read_csv(cd / "inference_out" / "lineage_out" / "lineage_data3D.csv")
mm = dd[dd["rank"] == 1].sort_values("frame").reset_index(drop=True)
L = mm["long_axis_um"].to_numpy()
div = _division_indices(L)
divset = set(div)
bounds = [0, *[d + 1 for d in div], len(mm)]
print(f"{pos} {ch}: segments (div-bounded), showing len>=55")
for a, b in zip(bounds[:-1], bounds[1:]):
    raw = mm.iloc[a:b]
    n = len(raw)
    if n < 55:
        continue
    f0, f1 = int(raw["frame"].min()), int(raw["frame"].max())
    div_end = (b - 1) in divset
    lmax = float(raw["long_axis_um"].max())
    print(f"  frames {f0}-{f1}  len={n}  maxL={lmax:.1f}um  div_terminated={div_end}")
