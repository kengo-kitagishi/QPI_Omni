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
INPUT_POS  = r"D:\AquisitionData\Kitagishi\260416\timelapse.pos"
OUTPUT_POS = r"D:\AquisitionData\Kitagishi\260416\timelapse_grid_260416.pos"
X_STEP = 0.1   # um
Y_STEP = 0.1   # um
X_HALF = 4    # Half-range -> total 9 points (stage X -> image Y, +-0.4 um coverage)
Y_HALF = 4    # Half-range -> total 9 points (stage Y -> image X, +-0.4 um coverage)
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

    # Grid expansion (Snake scan: yi direction reverses for each xi row)
    for xi in range(-X_HALF, X_HALF + 1):
        row = xi + X_HALF  # 0-indexed
        yi_range = range(-Y_HALF, Y_HALF + 1) if row % 2 == 0 else range(Y_HALF, -Y_HALF - 1, -1)
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