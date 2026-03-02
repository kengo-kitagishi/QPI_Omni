# %%
import json
import numpy as np
import matplotlib.pyplot as plt
from figure_logger import save_figure

# === 表示モード選択 ===
# "pixel"   : ピクセル単位で表示（従来通り）
# "physical" : 実際の距離 [μm] に変換して表示
DISPLAY_MODE = "physical"  # "pixel" or "physical"

# === 光学パラメータ（物理距離変換用） ===
SENSOR_PIXEL_SIZE = 3.45e-6  # センサーピクセルサイズ [m]
MAGNIFICATION = 40           # 対物レンズ倍率
ORIGINAL_DIM = 2048          # 元画像サイズ [px]
RECONSTRUCTED_DIM = 511      # reconstruct後のサイズ [px]（aperture size）

# === 時間軸設定 ===
# None にすると横軸はフレーム番号のまま。数値を入れると横軸が時間 [min] になる
TIME_INTERVAL_MIN = 5        # 5分間隔。None にすると横軸はフレーム番号

# JSON読み込み
JSON_PATH = r"E:\Acuisition\kitagishi\260218\move_test_2\Pos1\crop\alignment_transforms.json"


def visualize_shifts(
    json_path,
    display_mode=None,
    sensor_pixel_size=SENSOR_PIXEL_SIZE,
    magnification=MAGNIFICATION,
    original_dim=ORIGINAL_DIM,
    reconstructed_dim=RECONSTRUCTED_DIM,
    time_interval_min=TIME_INTERVAL_MIN,
):
    """
    alignment_transforms.json を読み込んでシフト時系列・軌跡をプロットし save_figure() で保存する。

    Parameters
    ----------
    json_path : str or Path
        alignment_transforms.json のパス
    display_mode : str or None
        "pixel" or "physical"。None のとき DISPLAY_MODE を使用。
    """
    mode = display_mode or DISPLAY_MODE
    pixel_scale_um = sensor_pixel_size / magnification * original_dim / reconstructed_dim * 1e6

    with open(json_path, "r") as f:
        data = json.load(f)

    shift_x = np.array([res["shift_x"] for res in data["alignment_results"]])
    shift_y = np.array([res["shift_y"] for res in data["alignment_results"]])
    frames = np.arange(len(shift_x))
    pos_name = data.get("pos_name", "unknown")

    if time_interval_min is not None:
        x_values = frames * time_interval_min / 60
        x_label = "Time (h)"
    else:
        x_values = frames
        x_label = "Frame number"

    if mode == "physical":
        shift_x_plot = shift_x * pixel_scale_um
        shift_y_plot = shift_y * pixel_scale_um
        unit_label = "μm"
        print(f"Pixel scale: {pixel_scale_um:.4f} μm/px")
    else:
        shift_x_plot = shift_x
        shift_y_plot = shift_y
        unit_label = "pixels"

    params = {
        "data_source": str(json_path),
        "display_mode": mode,
        "sensor_pixel_size": sensor_pixel_size,
        "magnification": magnification,
        "pixel_scale_um": pixel_scale_um,
        "n_frames": len(frames),
        "time_interval_min": time_interval_min,
    }

    # XY方向のシフトを時間的にプロット
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_ylim(-1, 1)
    ax.plot(x_values, shift_x_plot, label="Shift X", marker="o")
    ax.plot(x_values, shift_y_plot, label="Shift Y", marker="o")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Shift ({unit_label})")
    ax.set_title(f"Alignment shifts for {pos_name}")
    ax.legend()
    ax.grid(True)
    save_figure(fig, params=params, description=f"shift_timeseries {pos_name}")
    plt.show()

    # 2Dトラジェクトリ（動きの軌跡）
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(shift_x_plot, shift_y_plot, marker="o")
    ax2.set_xlabel(f"Shift X ({unit_label})")
    ax2.set_ylabel(f"Shift Y ({unit_label})")
    ax2.set_title(f"Trajectory of image shifts for {pos_name}")
    ax2.grid(True)
    ax2.set_aspect("equal")
    save_figure(fig2, params=params, description=f"shift_trajectory {pos_name}")
    plt.show()

    print(f"[shift_visualize] done: {pos_name}  (n={len(frames)} frames)")


# スタンドアロン実行時
if __name__ == "__main__" or True:
    visualize_shifts(JSON_PATH)

# %%
