"""make_timelapse_pos_from_state.py

Writes a timelapse .pos with every Pos at the place a stopped timelapse last parked it, i.e. the base
position minus the cumulative drift correction in drift_state_zstack.txt (the BeanShell moves to
base - CUMULATIVE_D?_UM_i).  Pos0 (background, not drift-corrected) gets base minus the mean correction
of the others, as make_grid_pos_from_state.py does.  Use it to start a NEW session on a grid that was
taken around those corrected positions.

    python scripts/make_timelapse_pos_from_state.py --session drift_session_261121 \
        --pos-file D:/AquisitionData/Kitagishi/261121/timelapse.pos \
        --out D:/AquisitionData/Kitagishi/261121/timelapse_2_from_session1_T32.pos
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--state", default="drift_state_zstack.txt", help="state file name inside the session folder")
    ap.add_argument("--pos-file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    state = {}
    for ln in open(Path(args.session) / args.state, encoding="utf-8"):
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.strip().split("=", 1)
            state[k] = v
    data = json.load(open(args.pos_file, encoding="utf-8"))
    positions = data["POSITIONS"]
    cums = {i: (float(state[f"CUMULATIVE_DX_UM_{i}"]), float(state[f"CUMULATIVE_DY_UM_{i}"]))
            for i in range(len(positions)) if f"CUMULATIVE_DX_UM_{i}" in state}
    mean = tuple(np.mean(list(cums.values()), axis=0))
    out = []
    for i, orig in enumerate(positions):
        cdx, cdy = cums.get(i, mean)
        q = copy.deepcopy(orig)
        for dev in q["DEVICES"]:
            if dev["DEVICE"] == "XYStage":
                dev["X"] -= cdx
                dev["Y"] -= cdy
        out.append(q)
    d = dict(data)
    d["POSITIONS"] = out
    json.dump(d, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"state TIMEPOINT={state.get('TIMEPOINT')} TIMESTAMP={state.get('TIMESTAMP')}")
    print(f"{args.out}: {len(out)} positions; corrections applied to {len(cums)} Pos, "
          f"mean ({mean[0]:+.3f}, {mean[1]:+.3f}) um (also used for the {len(out) - len(cums)} Pos without one)")


if __name__ == "__main__":
    main()
