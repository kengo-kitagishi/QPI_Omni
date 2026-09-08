"""Downsampled mother-track dump: every STEP frames over [f0,f1], to see trajectory."""
import sys
import pandas as pd
from _fig_panelA_cellcycle import data_root, load_bad_frames

pos, ch, f0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
f1 = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
step = int(sys.argv[5]) if len(sys.argv) > 5 else 15
cd = data_root(pos) / ch
dd = pd.read_csv(cd / "inference_out" / "lineage_out" / "lineage_data3D.csv")
mm = dd[dd["rank"] == 1].sort_values("frame").reset_index(drop=True)
bad = load_bad_frames(cd, pos)
sub = mm[(mm["frame"] >= f0) & (mm["frame"] <= f1)]
print(f"{pos} {ch}  {f0}-{f1}  present={len(sub)}  "
      f"frame span {int(sub['frame'].min())}-{int(sub['frame'].max())}")
for _, r in sub.iloc[::step].iterrows():
    fr = int(r["frame"])
    out = "OUT" if ("is_outlier" in r and bool(r["is_outlier"])) else ""
    bd = "BAD" if fr in bad else ""
    print(f"  f{fr:5d}  L={r['long_axis_um']:5.1f}um  t={r['time_h']:6.1f}h  {out}{bd}")
