"""
Micro-Manager .pos ファイルにグリッドを展開するスクリプト
入力: movetest.pos（元ポジション）
出力: movetest_grid.pos（各ポジションを中心にグリッド展開）

軸対応:
  ステージ X (xi) → 画像 Y 方向   観測ドリフト ±1 μm → X_HALF = 12 (±1.2 μm)
  ステージ Y (yi) → 画像 X 方向   観測ドリフト ±3 μm → Y_HALF = 35 (±3.5 μm)
  合計: 25 × 71 = 1775 点/Pos
"""
import json
import copy

# ---- パラメータ ----
INPUT_POS  = r"D:\AquisitionData\Kitagishi\260310\movetest.pos"
OUTPUT_POS = r"D:\AquisitionData\Kitagishi\260310\movetest_grid_12_30.pos"

X_STEP = 0.1   # μm
Y_STEP = 0.1   # μm
X_HALF = 12    # 片側 → 合計 25個（ステージX → 画像Y、±1.2 μm カバー）
Y_HALF = 30    # 片側 → 合計 61個（ステージY → 画像X、±3.0 μm カバー）
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