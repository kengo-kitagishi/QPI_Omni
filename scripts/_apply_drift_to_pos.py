"""Apply cumulative drift corrections from previous session to .pos file."""
import json
import re
from pathlib import Path

POS_IN = Path(r"D:\AquisitionData\Kitagishi\260504\timelapse.pos")
DRIFT_STATE = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_260504\drift_state_zstack.txt")
POS_OUT = Path(r"D:\AquisitionData\Kitagishi\260508\timelapse.pos")
BG_POS_INDEX = 0

pos = json.loads(POS_IN.read_text(encoding="utf-8"))

lines = DRIFT_STATE.read_text(encoding="utf-8").splitlines()
cumul = {}
for line in lines:
    m = re.match(r"CUMULATIVE_D([XY])_UM_(\d+)=([\d.e+-]+)", line)
    if m:
        axis, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
        cumul.setdefault(idx, {})[axis] = val

print(f"Positions in .pos: {len(pos['POSITIONS'])}")
print(f"Drift corrections available: Pos1 - Pos{max(cumul.keys())}")

for p in pos["POSITIONS"]:
    label = p["LABEL"]
    m = re.match(r"Pos(\d+)", label)
    if not m:
        continue
    idx = int(m.group(1))
    if idx == BG_POS_INDEX or idx not in cumul:
        continue
    for dev in p["DEVICES"]:
        if dev["DEVICE"] == "XYStage":
            old_x, old_y = dev["X"], dev["Y"]
            dev["X"] = old_x + cumul[idx]["X"]
            dev["Y"] = old_y + cumul[idx]["Y"]
            if idx <= 3 or idx == 104:
                print(f"  {label}: X {old_x:.3f} -> {dev['X']:.3f} (+{cumul[idx]['X']:.3f})")
                print(f"         Y {old_y:.3f} -> {dev['Y']:.3f} (+{cumul[idx]['Y']:.3f})")

POS_OUT.write_text(json.dumps(pos, indent=3, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved: {POS_OUT}")
print(f"Pos0 (BG) unchanged.")
