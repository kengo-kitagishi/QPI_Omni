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
OUTPUT_POS = r"D:/AquisitionData/Kitagishi/261112/grid_0p03um_25x25_stride.pos"  # 25x25 @ 0.03 um, 2D stride order (half-range +-0.36 um = +-1.04 px)
X_STEP = 0.03  # um
Y_STEP = 0.03  # um
X_HALF = 12  # Half-range -> total 25 points (stage X -> image Y, +-0.36 um coverage)
Y_HALF = 12  # Half-range -> total 25 points (stage Y -> image X, +-0.36 um coverage)
SNAKE = False  # False: every row scanned in the same direction (see loop below)
# --------------------


# ---- Visiting order ----
# "rows":     every row -Y_HALF..+Y_HALF in the same direction (SNAKE=False) or snake.
#             Consecutive commands are one step apart; on 261112 the stage did not execute
#             30 nm steps one by one (0.01 px between neighbours, 0.5 px after 3 steps), so
#             the points piled up and a 19x19 @ 0.03 um grid was no denser than 11x11 @ 0.05.
# "stride2d": k -> (p*a + q*b, r*a + s*b) mod N with a = k mod N, b = k // N, chosen so
#             that the map is a bijection on the N x N lattice and BOTH stage axes move at
#             least MIN_MOVE_UM at every point (measured on 261112: moves >= 0.15 um spread
#             the points, nearest-neighbour distance 0.006 -> 0.029 px, residual p50
#             0.16 -> 0.046 px). Deterministic, any odd N.
ORDER = "stride2d"
MIN_MOVE_UM = 0.15


def _stride_coeffs(n, step_um, min_move_um):
    """Smallest (p, q, r, s) giving a bijection and per-point moves >= min_move_um on both axes."""
    from math import gcd
    kmin = int(-(-min_move_um // step_um))          # steps needed for min_move_um
    for p in range(kmin, n):
        for r in range(kmin, n):
            for q in range(1, n):
                for s_ in range(1, n):
                    if gcd(p * s_ - q * r, n) != 1:
                        continue
                    # per-a-step move (p, r) steps; row wrap (a: n-1 -> 0, b+1) moves (q - (n-1)*p, s - (n-1)*r) mod n
                    wrap = ((q - (n - 1) * p) % n, (s_ - (n - 1) * r) % n)
                    wrap = (min(wrap[0], n - wrap[0]), min(wrap[1], n - wrap[1]))
                    if min(p, n - p) >= kmin and min(r, n - r) >= kmin and wrap[0] >= kmin and wrap[1] >= kmin:
                        return p, q, r, s_
    raise RuntimeError(f"no stride coefficients for N={n}, step {step_um} um, min move {min_move_um} um")


def grid_order(x_half, y_half):
    if ORDER == "rows":
        out = []
        for xi in range(-x_half, x_half + 1):
            row = xi + x_half
            if SNAKE and row % 2 == 1:
                out += [(xi, yi) for yi in range(y_half, -y_half - 1, -1)]
            else:
                out += [(xi, yi) for yi in range(-y_half, y_half + 1)]
        return out
    if ORDER == "stride2d":
        assert x_half == y_half, "stride2d needs a square grid"
        n = 2 * x_half + 1
        p, q, r, s_ = _stride_coeffs(n, min(X_STEP, Y_STEP), MIN_MOVE_UM)
        out, seen = [], set()
        for k in range(n * n):
            a, b = k % n, k // n
            xi, yi = (p * a + q * b) % n, (r * a + s_ * b) % n
            assert (xi, yi) not in seen
            seen.add((xi, yi))
            out.append((xi - x_half, yi - y_half))
        print(f"stride2d order: N={n}, coefficients (p,q,r,s)=({p},{q},{r},{s_}), "
              f"per-point move >= {MIN_MOVE_UM} um on both axes")
        return out
    raise ValueError(ORDER)

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

    # Grid expansion. The visiting order is built first (see grid_order), then each
    # (xi, yi) is turned into a position.
    for xi, yi in grid_order(X_HALF, Y_HALF):
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
