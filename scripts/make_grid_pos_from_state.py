"""make_grid_pos_from_state.py

Builds a grid .pos centred on where a RUNNING timelapse currently parks each Pos, i.e. the
timelapse.pos base position minus the cumulative drift correction in drift_state_zstack.txt
(the BeanShell moves to corrX = baseX - CUMULATIVE_DX_UM_i, corrY = baseY - CUMULATIVE_DY_UM_i;
Pos0 = the BG gets base minus the mean correction of the others).  The grid is then expanded
with generate_grid_pos.grid_order (2D stride) and split into chunks of Pos0 + N Pos.

Run it right after stopping the timelapse, so the state is frozen.

    python scripts/make_grid_pos_from_state.py --session drift_session_261112_3 \
        --pos-file D:/AquisitionData/Kitagishi/261112/timelapse.pos --step 0.03 --half 9 \
        --out D:/AquisitionData/Kitagishi/261112/grid_0per_0p03um_19x19_stride.pos --chunk 25
"""
import argparse
import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def load_grid_order():
    # generate_grid_pos.py runs at import, so pull grid_order out of its source instead
    src = (SCRIPT_DIR / "generate_grid_pos.py").read_text(encoding="utf-8")
    start = src.index("# ---- Visiting order ----")
    end = src.index('with open(INPUT_POS, "r") as f:')
    ns = {}
    exec(src[start:end], ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--pos-file", required=True)
    ap.add_argument("--step", type=float, default=0.03)
    ap.add_argument("--half", type=int, default=9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=25, help="sample Pos per chunk (Pos0 is added to every chunk)")
    args = ap.parse_args()

    state = {}
    for ln in open(Path(args.session) / "drift_state_zstack.txt", encoding="utf-8"):
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.strip().split("=", 1)
            state[k] = v
    print(f"drift state: TIMEPOINT={state.get('TIMEPOINT')} TIMESTAMP={state.get('TIMESTAMP')}")

    data = json.load(open(args.pos_file, encoding="utf-8"))
    positions = data["POSITIONS"]
    cums = {}
    for i, p in enumerate(positions):
        if f"CUMULATIVE_DX_UM_{i}" in state:
            cums[i] = (float(state[f"CUMULATIVE_DX_UM_{i}"]), float(state[f"CUMULATIVE_DY_UM_{i}"]))
    mean = np.mean(list(cums.values()), axis=0)
    print(f"per-Pos cumulative corrections: {len(cums)}; mean ({mean[0]:+.3f}, {mean[1]:+.3f}) um, "
          f"range x {min(c[0] for c in cums.values()):+.3f}..{max(c[0] for c in cums.values()):+.3f}, "
          f"y {min(c[1] for c in cums.values()):+.3f}..{max(c[1] for c in cums.values()):+.3f} um")

    ns = load_grid_order()
    ns["X_STEP"] = ns["Y_STEP"] = args.step
    ns["SNAKE"] = False
    order = ns["grid_order"](args.half, args.half)

    out_positions = []
    for i, orig in enumerate(positions):
        cdx, cdy = cums.get(i, tuple(mean))
        base_label = orig["LABEL"]
        bx = by = bz = None
        for dev in orig["DEVICES"]:
            if dev["DEVICE"] == "XYStage":
                bx, by = dev["X"] - cdx, dev["Y"] - cdy
            elif dev["DEVICE"] == "TIPFSOffset":
                bz = dev["X"]
        for xi, yi in order:
            q = copy.deepcopy(orig)
            q["LABEL"] = f"{base_label}_x{xi:+d}_y{yi:+d}"
            for dev in q["DEVICES"]:
                if dev["DEVICE"] == "XYStage":
                    dev["X"] = bx + xi * args.step
                    dev["Y"] = by + yi * args.step
                elif dev["DEVICE"] == "TIPFSOffset":
                    dev["X"] = bz
            out_positions.append(q)
    out = Path(args.out)
    d = dict(data)
    d["POSITIONS"] = out_positions
    json.dump(d, open(out, "w", encoding="utf-8"), indent=2)
    print(f"{out}: {len(out_positions)} points ({len(positions)} Pos x {len(order)})")

    base = lambda l: int(re.match(r"Pos(\d+)", l).group(1))
    labels = sorted({base(q["LABEL"]) for q in out_positions} - {0})
    bg = [q for q in out_positions if base(q["LABEL"]) == 0]
    chunks = [labels[k:k + args.chunk] for k in range(0, len(labels), args.chunk)]
    for n, c in enumerate(chunks, 1):
        sel = bg + [q for q in out_positions if c[0] <= base(q["LABEL"]) <= c[-1]]
        p = out.with_name(f"{out.stem}_chunk{n}_Pos{c[0]}-{c[-1]}.pos")
        dd = dict(data)
        dd["POSITIONS"] = sel
        json.dump(dd, open(p, "w", encoding="utf-8"), indent=2)
        print(f"  {p.name}: {len(sel)} points, ~{len(sel)*0.485/60:.0f} min")


if __name__ == "__main__":
    main()
