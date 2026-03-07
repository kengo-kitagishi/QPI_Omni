"""
Micro-Manager .pos ファイルにグリッドを展開するスクリプト
入力: movetest.pos（元ポジション）
出力: movetest_grid.pos（各ポジションを中心に 41x21 グリッド展開）
"""
import json
import copy

# ---- パラメータ ----
INPUT_POS  = r"E:\Acuisition\kitagishi\260301\movetest.pos"
OUTPUT_POS = r"E:\Acuisition\kitagishi\260301\movetest_grid.pos"

X_STEP = 0.1   # μm
Y_STEP = 0.1   # μm
X_HALF = 20    # 片側 → 合計 41個
Y_HALF = 10    # 片側 → 合計 21個
# --------------------

with open(INPUT_POS, "r") as f:
    data = json.load(f)
orig_positions = data["POSITIONS"]
new_positions  = []

for orig in orig_positions:
    base_label = orig["LABEL"]

    # XY・Z の基準座標を取得
    base_x, base_y, base_z_offset = 0.0, 0.0, 0.0
    for dev in orig["DEVICES"]:
        if dev["DEVICE"] == "XYStage":
            base_x = dev["X"]
            base_y = dev["Y"]
        elif dev["DEVICE"] == "TIPFSOffset":
            base_z_offset = dev["X"]

    # グリッド展開
    for xi in range(-X_HALF, X_HALF + 1):
        for yi in range(-Y_HALF, Y_HALF + 1):
            new_pos = copy.deepcopy(orig)
            new_pos["LABEL"] = f"{base_label}_x{xi:+d}_y{yi:+d}"

            for dev in new_pos["DEVICES"]:
                if dev["DEVICE"] == "XYStage":
                    dev["X"] = base_x + xi * X_STEP
                    dev["Y"] = base_y + yi * Y_STEP
                elif dev["DEVICE"] == "TIPFSOffset":
                    dev["X"] = base_z_offset  # 変更しない

            new_positions.append(new_pos)

data["POSITIONS"] = new_positions

with open(OUTPUT_POS, "w") as f:
    json.dump(data, f, indent=3)

n_orig = len(orig_positions)
n_new  = len(new_positions)
print(f"元ポジション数  : {n_orig}")
print(f"グリッド展開後  : {n_new}  ({n_orig} x {2*X_HALF+1} x {2*Y_HALF+1})")
print(f"出力ファイル    : {OUTPUT_POS}")