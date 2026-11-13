"""
Script to expand grid positions in a Micro-Manager .pos file.
Input: timelapse.pos (original positions)
Output: timelapse_grid.pos (grid expansion centered on each position)

Axis mapping:
  Stage X (xi) -> Image Y direction   X_HALF = 4 (+-0.4 um)
  Stage Y (yi) -> Image X direction   Y_HALF = 4 (+-0.4 um)
  Total: 9 x 9 = 81 points/Pos
  Scan: Snake scan (yi direction alternates for each row)
"""
import json
import copy

# ---- Parameters ----
INPUT_POS  = r"D:\AquisitionData\Kitagishi\261112\timelapse.pos"
OUTPUT_POS = r"D:/AquisitionData/Kitagishi/261112/grid_0p03um_19x19_uni.pos"  # 19x19 @ 0.03 um, unidirectional rows (grid.pos = the 11x11 @ 0.05 um snake used on 2026-11-13)
X_STEP = 0.03  # um
Y_STEP = 0.03  # um
X_HALF = 9  # Half-range -> total 19 points (stage X -> image Y, +-0.27 um coverage)
Y_HALF = 9  # Half-range -> total 19 points (stage Y -> image X, +-0.27 um coverage)
SNAKE = False  # False: every row scanned in the same direction (see loop below)
# --------------------

with open(INPUT_POS, "r") as f:
    data = json.load(f)
orig_positions = data["POSITIONS"]
new_positions  = []

for orig in orig_positions:
    base_label = orig["LABEL"]

    # Get XY and Z base coordinates
    base_x, base_y, base_z_offset = 0.0, 0.0, 0.0
    for dev in orig["DEVICES"]:
        if dev["DEVICE"] == "XYStage":
            base_x = dev["X"]
            base_y = dev["Y"]
        elif dev["DEVICE"] == "TIPFSOffset":
            base_z_offset = dev["X"]

    # Grid expansion. SNAKE=True: yi direction reverses for each xi row (shorter travel,
    # but the stage Y backlash puts alternate rows 0.3 px apart in image x -- measured on
    # the 261112 calibration). SNAKE=False: every row runs -Y_HALF..+Y_HALF, so all
    # points are approached from the same side and the measured lattice stays square.
    for xi in range(-X_HALF, X_HALF + 1):
        row = xi + X_HALF  # 0-indexed
        if SNAKE and row % 2 == 1:
            yi_range = range(Y_HALF, -Y_HALF - 1, -1)
        else:
            yi_range = range(-Y_HALF, Y_HALF + 1)
        for yi in yi_range:
            new_pos = copy.deepcopy(orig)
            new_pos["LABEL"] = f"{base_label}_x{xi:+d}_y{yi:+d}"

            for dev in new_pos["DEVICES"]:
                if dev["DEVICE"] == "XYStage":
                    dev["X"] = base_x + xi * X_STEP
                    dev["Y"] = base_y + yi * Y_STEP
                elif dev["DEVICE"] == "TIPFSOffset":
                    dev["X"] = base_z_offset  # Keep unchanged

            new_positions.append(new_pos)

data["POSITIONS"] = new_positions

with open(OUTPUT_POS, "w") as f:
    json.dump(data, f, indent=3)

n_orig = len(orig_positions)
n_new  = len(new_positions)
print(f"Original positions : {n_orig}")
print(f"After grid expand  : {n_new}  ({n_orig} x {2*X_HALF+1} x {2*Y_HALF+1})")
print(f"Output file        : {OUTPUT_POS}")
