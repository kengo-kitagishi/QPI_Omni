# %%
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from matplotlib.colors import TwoSlopeNorm
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

# JSON読み込み（単体実行時）
JSON_PATH = r"E:\Acuisition\kitagishi\260301\movetest_8\Pos4\cropped\alignment_transforms.json"


def _representative_shift_indices(shift_x, shift_y):
    """shift量が小・中・大の代表フレーム index を返す。"""
    n = len(shift_x)
    if n == 0:
        return []

    mags = np.sqrt(shift_x ** 2 + shift_y ** 2)
    sorted_idx = np.argsort(mags)

    if n == 1:
        return [("single", int(sorted_idx[0]))]
    if n == 2:
        return [("small", int(sorted_idx[0])), ("large", int(sorted_idx[-1]))]

    return [
        ("small", int(sorted_idx[0])),
        ("medium", int(sorted_idx[n // 2])),
        ("large", int(sorted_idx[-1])),
    ]


def _target_shift_vector_indices(shift_x, shift_y, target_vectors_px, used_indices=None):
    """指定シフトベクトル(target_x, target_y)に最も近い frame index を返す。"""
    n = len(shift_x)
    if n == 0:
        return []

    used = set(used_indices or [])
    out = []

    def _fmt(v):
        return f"{v:.2f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")

    for target_x, target_y in target_vectors_px:
        tx = float(target_x)
        ty = float(target_y)
        d = np.sqrt((shift_x - tx) ** 2 + (shift_y - ty) ** 2)
        order = np.argsort(d)
        chosen = None
        for idx in order:
            idx_i = int(idx)
            if idx_i not in used:
                chosen = idx_i
                break
        if chosen is None:  # 全て使用済みなら最短距離を許容
            chosen = int(order[0])
        used.add(chosen)

        label = f"target_x{_fmt(tx)}_y{_fmt(ty)}px"
        out.append((label, chosen, tx, ty, float(d[chosen])))
    return out


def visualize_shifts(
    json_path,
    display_mode=None,
    sensor_pixel_size=SENSOR_PIXEL_SIZE,
    magnification=MAGNIFICATION,
    original_dim=ORIGINAL_DIM,
    reconstructed_dim=RECONSTRUCTED_DIM,
    time_interval_min=TIME_INTERVAL_MIN,
    subtracted_vmin=-0.1,
    subtracted_vmax=1.7,
    subtracted_cmap="RdBu_r",
    target_shift_vectors_px=(
        (-0.5, 0.0),
        (-1.0, 0.0),
        (-1.5, 0.0),
        (-2.0, 0.0),
        (0.0, 0.5),
        (0.0, 1.0),
        (0.0, 1.5),
        (0.0, 2.0),
    ),
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
    ax.set_ylim(-5,1)
    ax.legend()
    ax.grid(True)
    _shift_data = {
        "x_values":    x_values,
        "shift_x":     shift_x_plot,
        "shift_y":     shift_y_plot,
    }

    save_figure(fig, params=params, description=f"shift_timeseries {pos_name}",
                data=_shift_data)
    plt.show(block=False)

    # 2Dトラジェクトリ（動きの軌跡）
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(shift_x_plot, shift_y_plot, marker="o")
    ax2.set_xlabel(f"Shift X ({unit_label})")
    ax2.set_ylim(-3,3)
    ax2.set_xlim(-5,1)
    ax2.set_title(f"Trajectory of image shifts for {pos_name}")
    ax2.grid(True)
    ax2.set_aspect("equal")
    save_figure(fig2, params=params, description=f"shift_trajectory {pos_name}",
                data=_shift_data)
    plt.show(block=False)

    # 引き算後(subtracted)画像の代表例を shift量に応じて個別保存（同一 run_id フォルダ）
    subtracted_dir = os.path.join(os.path.dirname(json_path), "subtracted")
    representatives = _representative_shift_indices(shift_x, shift_y)
    target_examples = _target_shift_vector_indices(
        shift_x,
        shift_y,
        target_vectors_px=target_shift_vectors_px,
        used_indices=[idx for _, idx in representatives],
    )

    if os.path.isdir(subtracted_dir) and (representatives or target_examples):
        mags = np.sqrt(shift_x ** 2 + shift_y ** 2)
        examples = []
        for level, idx in representatives:
            examples.append(
                {
                    "selector": "rank",
                    "label": level,
                    "idx": int(idx),
                    "actual_shift_mag_px": float(mags[int(idx)]),
                    "target_shift_x_px": None,
                    "target_shift_y_px": None,
                    "target_distance_px": None,
                }
            )
        for label, idx, target_x, target_y, target_distance in target_examples:
            examples.append(
                {
                    "selector": "target",
                    "label": label,
                    "idx": int(idx),
                    "actual_shift_mag_px": float(mags[int(idx)]),
                    "target_shift_x_px": float(target_x),
                    "target_shift_y_px": float(target_y),
                    "target_distance_px": float(target_distance),
                }
            )

        for ex in examples:
            level = ex["label"]
            idx = ex["idx"]
            result = data["alignment_results"][idx]
            filename = result["filename"]
            base = os.path.splitext(filename)[0]
            subtracted_path = os.path.join(subtracted_dir, f"{base}_subtracted.tif")

            if not os.path.exists(subtracted_path):
                print(f"[shift_visualize] skip (not found): {subtracted_path}")
                continue

            subtracted_img = tifffile.imread(subtracted_path).astype(np.float64)

            fig_sub, ax_sub = plt.subplots(figsize=(8, 6))
            norm = TwoSlopeNorm(vmin=subtracted_vmin, vcenter=0.0, vmax=subtracted_vmax)
            im_sub = ax_sub.imshow(subtracted_img, cmap=subtracted_cmap, norm=norm)
            ax_sub.axis("off")
            frame_no = idx + 1  # 1-indexed
            if ex["target_shift_x_px"] is None:
                title = (
                    f"{pos_name} | {level} shift | frame {frame_no} ({filename}) | "
                    f"mag={ex['actual_shift_mag_px']:.3f} px "
                    f"(x={result['shift_x']:.3f}, y={result['shift_y']:.3f})"
                )
            else:
                title = (
                    f"{pos_name} | {level} | frame {frame_no} ({filename}) | "
                    f"target=({ex['target_shift_x_px']:.2f}, {ex['target_shift_y_px']:.2f}) px, "
                    f"actual=({result['shift_x']:.3f}, {result['shift_y']:.3f}) px, "
                    f"dist={ex['target_distance_px']:.3f} px"
                )
            ax_sub.set_title(title)
            plt.colorbar(im_sub, ax=ax_sub, fraction=0.046, label="Subtracted phase (rad)")
            plt.tight_layout()

            tif_dst_name = f"{level}__{base}_subtracted.tif"
            save_figure(
                fig_sub,
                params={
                    **params,
                    "example_selector": ex["selector"],
                    "example_level": level,
                    "example_frame_index": int(idx),
                    "example_filename": filename,
                    "shift_x_px": float(result["shift_x"]),
                    "shift_y_px": float(result["shift_y"]),
                    "shift_mag_px": ex["actual_shift_mag_px"],
                    "target_shift_x_px": ex["target_shift_x_px"],
                    "target_shift_y_px": ex["target_shift_y_px"],
                    "target_distance_px": ex["target_distance_px"],
                    "colormap": subtracted_cmap,
                    "vmin": subtracted_vmin,
                    "vmax": subtracted_vmax,
                    "source_subtracted_path": subtracted_path,
                },
                description=f"subtracted_example_{level}shift {pos_name}",
                copy_files=[(subtracted_path, tif_dst_name)],
            )
            plt.close(fig_sub)
    else:
        print(f"[shift_visualize] subtracted dir not found or empty: {subtracted_dir}")

    print(f"[shift_visualize] done: {pos_name}  (n={len(frames)} frames)")


# スタンドアロン実行時
if __name__ == "__main__":
    visualize_shifts(JSON_PATH)

# %%
