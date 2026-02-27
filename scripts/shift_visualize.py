# %%
import json
import numpy as np
import matplotlib.pyplot as plt
from figure_logger import save_figure

# === 表示モード選択 ===
# "pixel"   : ピクセル単位で表示（従来通り）
# "physical" : 実際の距離 [μm] に変換して表示
DISPLAY_MODE = "pixel"  # "pixel" or "physical"

# === 光学パラメータ（物理距離変換用） ===
SENSOR_PIXEL_SIZE = 3.45e-6  # センサーピクセルサイズ [m]
MAGNIFICATION = 40           # 対物レンズ倍率
ORIGINAL_DIM = 2048          # 元画像サイズ [px]
RECONSTRUCTED_DIM = 511      # reconstruct後のサイズ [px]（aperture size）

# 再構成画像の1ピクセルあたりの実距離 [μm]
PIXEL_SCALE_UM = SENSOR_PIXEL_SIZE / MAGNIFICATION * ORIGINAL_DIM / RECONSTRUCTED_DIM * 1e6

# JSON読み込み
JSON_PATH = r"E:\Acuisition\kitagishi\260218\move_test_2\Pos1\crop\alignment_transforms.json"
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# シフト量を取得
shift_x = np.array([res["shift_x"] for res in data["alignment_results"]])
shift_y = np.array([res["shift_y"] for res in data["alignment_results"]])
frames = list(range(len(shift_x)))

if DISPLAY_MODE == "physical":
    shift_x_plot = shift_x * PIXEL_SCALE_UM
    shift_y_plot = shift_y * PIXEL_SCALE_UM
    unit_label = "μm"
    print(f"Pixel scale: {PIXEL_SCALE_UM:.4f} μm/px")
else:
    shift_x_plot = shift_x
    shift_y_plot = shift_y
    unit_label = "pixels"

# XY方向のシフトを時間的にプロット
fig = plt.figure(figsize=(10,5))
plt.ylim(-1,1)
plt.plot(frames, shift_x_plot, label='Shift X', marker='o')
plt.plot(frames, shift_y_plot, label='Shift Y', marker='o')
plt.xlabel("Frame number")
plt.ylabel(f"Shift ({unit_label})")
plt.title(f"Alignment shifts for {data['pos_name']}")
plt.legend()
plt.grid(True)
save_figure(fig,
            params={"data_source": JSON_PATH, "display_mode": DISPLAY_MODE,
                    "sensor_pixel_size": SENSOR_PIXEL_SIZE, "magnification": MAGNIFICATION,
                    "pixel_scale_um": PIXEL_SCALE_UM, "n_frames": len(frames)},
            description=f"shift_timeseries {data['pos_name']}")
plt.show()

# 2Dトラジェクトリ（動きの軌跡）
fig2 = plt.figure(figsize=(6,6))
plt.plot(shift_x_plot, shift_y_plot, marker='o')
plt.xlabel(f"Shift X ({unit_label})")
plt.ylabel(f"Shift Y ({unit_label})")
plt.title(f"Trajectory of image shifts for {data['pos_name']}")
plt.grid(True)
plt.axis('equal')
save_figure(fig2,
            params={"data_source": JSON_PATH, "display_mode": DISPLAY_MODE,
                    "sensor_pixel_size": SENSOR_PIXEL_SIZE, "magnification": MAGNIFICATION,
                    "pixel_scale_um": PIXEL_SCALE_UM, "n_frames": len(frames)},
            description=f"shift_trajectory {data['pos_name']}")
plt.show()

# %%
