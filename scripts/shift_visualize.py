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
#    ax.set_xlim(-1, 1)
    ax.plot(x_values, shift_x_plot, label="Shift X", marker="o")
    ax.plot(x_values, shift_y_plot, label="Shift Y", marker="o")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Shift ({unit_label})")
    ax.set_title(f"Alignment shifts for {pos_name}")
    ax.set_ylim(-0.5,0.5)
    ax.legend()
    ax.grid(True)
    _shift_data = {
        "x_values":    x_values,
        "shift_x":     shift_x_plot,
        "shift_y":     shift_y_plot,
    }

    save_figure(fig, params=params, description=f"shift_timeseries {pos_name}",
                data=_shift_data)
    plt.close(fig)

    # 2Dトラジェクトリ（動きの軌跡）
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(shift_x_plot, shift_y_plot, marker="o")
    ax2.set_xlabel(f"Shift X ({unit_label})")
    ax2.set_ylim(-0.5,0.5)
    ax2.set_xlim(-0.5,0.5)
    ax2.set_title(f"Trajectory of image shifts for {pos_name}")
    ax2.grid(True)
    ax2.set_aspect("equal")
    save_figure(fig2, params=params, description=f"shift_trajectory {pos_name}",
                data=_shift_data)
    plt.close(fig2)

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


def visualize_2pass_shifts(
    json_path,
    sensor_pixel_size=SENSOR_PIXEL_SIZE,
    magnification=MAGNIFICATION,
    original_dim=ORIGINAL_DIM,
    reconstructed_dim=RECONSTRUCTED_DIM,
    time_interval_min=TIME_INTERVAL_MIN,
    x_step_um=0.1,
    y_step_um=0.1,
    shift_sign_x=1,
    shift_sign_y=1,
):
    """
    USE_SECOND_PASS_ECC=True で生成した pos_shifts.json から
    pass1 / pass2 / 比較 / fine の4種類の時系列図を save_figure で保存する。

    出力図 (description でファイル名識別):
      shift_timeseries_pass1 {pos}  — 1回目ECC (full crop) のシフト [grid offset込み]
      shift_timeseries_pass2 {pos}  — 2回目ECC (half crop) のシフト [=最終値, grid offset込み]
      shift_timeseries_pass1_vs_pass2 {pos}  — 両者のオーバーレイ比較
      shift_timeseries_fine_ecc {pos}  — gridからのfine ECC残差シフト
    """
    pixel_scale_um = sensor_pixel_size / magnification * original_dim / reconstructed_dim * 1e6

    with open(json_path, "r") as f:
        data = json.load(f)

    pos_name = data.get("pos_name", "unknown")
    frame_results = data.get("frame_results", [])
    n_frames = len(frame_results)

    # pass2 (final) はそのまま frame_results から取得
    pass2_x = np.array([r["shift_x_avg"] if r["shift_x_avg"] is not None else np.nan
                        for r in frame_results])
    pass2_y = np.array([r["shift_y_avg"] if r["shift_y_avg"] is not None else np.nan
                        for r in frame_results])

    # pass1/pass2 の total・fine・grid_offset を per_channel から平均
    def _avg_ch_field(field):
        out = []
        for r in frame_results:
            vals = [ch[field] for ch in r.get("per_channel", [])
                    if not ch.get("excluded", True) and ch.get(field) is not None]
            out.append(float(np.mean(vals)) if vals else np.nan)
        return np.array(out)

    pass1_x = _avg_ch_field("pass1_shift_x")
    pass1_y = _avg_ch_field("pass1_shift_y")
    fine1_x  = _avg_ch_field("pass1_fine_x")
    fine1_y  = _avg_ch_field("pass1_fine_y")
    fine2_x  = _avg_ch_field("pass2_fine_x")
    fine2_y  = _avg_ch_field("pass2_fine_y")
    goff1_x  = _avg_ch_field("pass1_grid_offset_x")
    goff1_y  = _avg_ch_field("pass1_grid_offset_y")
    goff2_x  = _avg_ch_field("pass2_grid_offset_x")
    goff2_y  = _avg_ch_field("pass2_grid_offset_y")

    # fine フィールドが未保存（旧JSON）の場合: grid_xi/yi から名目オフセットを計算してfine を推定
    # _get_grid_offset の式: (SHIFT_SIGN_Y * yi * Y_STEP / pscale, SHIFT_SIGN_X * xi * X_STEP / pscale)
    def _avg_fine_fallback(shift_arr, xi_field, yi_field):
        out = []
        for r in frame_results:
            vals = []
            for ch in r.get("per_channel", []):
                if ch.get("excluded", True):
                    continue
                sx = ch.get(shift_arr[0])
                sy = ch.get(shift_arr[1])
                xi = ch.get(xi_field, 0) or 0
                yi = ch.get(yi_field, 0) or 0
                ox = shift_sign_y * yi * y_step_um / pixel_scale_um
                oy = shift_sign_x * xi * x_step_um / pixel_scale_um
                if sx is not None:
                    vals.append((sx - ox, sy - oy))
            if vals:
                out.append((float(np.mean([v[0] for v in vals])),
                            float(np.mean([v[1] for v in vals]))))
            else:
                out.append((np.nan, np.nan))
        return (np.array([v[0] for v in out]), np.array([v[1] for v in out]))

    if np.all(np.isnan(fine1_x)):
        fine1_x, fine1_y = _avg_fine_fallback(
            ("pass1_shift_x", "pass1_shift_y"), "pass1_grid_xi", "pass1_grid_yi")
        goff1_x = pass1_x - fine1_x
        goff1_y = pass1_y - fine1_y
    if np.all(np.isnan(fine2_x)):
        fine2_x, fine2_y = _avg_fine_fallback(
            ("pass2_shift_x", "pass2_shift_y"), "pass2_grid_xi", "pass2_grid_yi")
        goff2_x = pass2_x - fine2_x
        goff2_y = pass2_y - fine2_y

    # pass2 で選択された grid xi/yi (per frame の平均値を整数化)
    def _avg_ch_int(field):
        out = []
        for r in frame_results:
            vals = [ch[field] for ch in r.get("per_channel", [])
                    if not ch.get("excluded", True) and ch.get(field) is not None]
            out.append(int(round(float(np.mean(vals)))) if vals else 0)
        return np.array(out)

    p2_grid_xi = _avg_ch_int("pass2_grid_xi")
    p2_grid_yi = _avg_ch_int("pass2_grid_yi")

    frames = np.arange(n_frames)
    if time_interval_min is not None:
        x_values = frames * time_interval_min / 60
        x_label = "Time (h)"
    else:
        x_values = frames
        x_label = "Frame number"

    p1x = pass1_x * pixel_scale_um
    p1y = pass1_y * pixel_scale_um
    p2x = pass2_x * pixel_scale_um
    p2y = pass2_y * pixel_scale_um
    unit = "μm"

    base_params = {
        "data_source": str(json_path),
        "pixel_scale_um": pixel_scale_um,
        "n_frames": n_frames,
        "time_interval_min": time_interval_min,
    }

    YLIM = (-0.3, 0.3)  # 全図共通Y軸範囲 [μm]

    # ── Figure 1: pass1 のみ ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_values, p1x, label="Shift X  [pass1]", marker="o", markersize=3)
    ax.plot(x_values, p1y, label="Shift Y  [pass1]", marker="o", markersize=3)
    ax.set_xlabel(x_label); ax.set_ylabel(f"Shift ({unit})")
    ax.set_ylim(YLIM); ax.set_title(f"[pass1 / 1st ECC: grid(0,0) fixed]  {pos_name}")
    ax.legend(); ax.grid(True)
    save_figure(fig, params={**base_params, "pass": "pass1"},
                description=f"shift_timeseries_pass1 {pos_name}",
                data={"x_values": x_values, "shift_x": p1x, "shift_y": p1y})
    plt.close(fig)

    # ── Figure 2: pass2 のみ ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_values, p2x, label="Shift X  [pass2]", marker="o", markersize=3, color="tab:blue")
    ax.plot(x_values, p2y, label="Shift Y  [pass2]", marker="o", markersize=3, color="tab:orange")
    ax.set_xlabel(x_label); ax.set_ylabel(f"Shift ({unit})")
    ax.set_ylim(YLIM); ax.set_title(f"[pass2 / 2nd ECC: nearest grid, half crop]  {pos_name}  [= final]")
    ax.legend(); ax.grid(True)
    save_figure(fig, params={**base_params, "pass": "pass2"},
                description=f"shift_timeseries_pass2 {pos_name}",
                data={"x_values": x_values, "shift_x": p2x, "shift_y": p2y})
    plt.close(fig)

    # ── Figure 3: pass1 vs pass2 オーバーレイ ────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, p1, p2, label in zip(axes, [p1x, p1y], [p2x, p2y], ["X", "Y"]):
        ax.plot(x_values, p1, label="pass1 (grid(0,0), half crop)", alpha=0.7,
                linewidth=1, marker="o", markersize=2)
        ax.plot(x_values, p2, label="pass2 (nearest grid, half crop) [final]", alpha=0.9,
                linewidth=1.2, marker="o", markersize=2)
        ax.fill_between(x_values, p1, p2, alpha=0.15, label="diff (p2−p1)")
        ax.set_ylabel(f"Shift {label} ({unit})"); ax.set_ylim(YLIM)
        ax.legend(fontsize=7); ax.grid(True)
    axes[0].set_title(f"[pass1 vs pass2]  {pos_name}")
    axes[1].set_xlabel(x_label)
    fig.tight_layout()
    save_figure(fig, params={**base_params, "pass": "pass1_vs_pass2"},
                description=f"shift_timeseries_pass1_vs_pass2 {pos_name}",
                data={"x_values": x_values,
                      "pass1_shift_x": p1x, "pass1_shift_y": p1y,
                      "pass2_shift_x": p2x, "pass2_shift_y": p2y})
    plt.close(fig)

    # ── Figure 4: pass2 選択grid情報 + fine2 + 距離比較 ──────
    f2x = fine2_x * pixel_scale_um   # 選択gridからの距離 X
    f2y = fine2_y * pixel_scale_um   # 選択gridからの距離 Y
    # pass2_x/y = fine2 + grid_offset2 = grid(0,0)からの距離
    euclid2 = np.sqrt(f2x**2 + f2y**2)  # 選択gridからのユークリッド距離

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    # Row 0: X方向
    ax = axes[0]
    ax.step(x_values, p2_grid_xi * x_step_um, color="gray", alpha=0.4, linewidth=1, where="mid",
            label=f"grid xi × {x_step_um} μm (selected)")
    ax.plot(x_values, f2x, label="fine2 X (from selected grid)", color="tab:orange",
            linewidth=1, marker="o", markersize=2)
    ax.plot(x_values, p2x, label="pass2 X (from grid(0,0))", color="tab:blue",
            linewidth=1, linestyle="--", marker="o", markersize=2)
    ax.set_ylabel(f"X ({unit})"); ax.set_ylim(YLIM); ax.legend(fontsize=7); ax.grid(True)

    # Row 1: Y方向
    ax = axes[1]
    ax.step(x_values, p2_grid_yi * y_step_um, color="gray", alpha=0.4, linewidth=1, where="mid",
            label=f"grid yi × {y_step_um} μm (selected)")
    ax.plot(x_values, f2y, label="fine2 Y (from selected grid)", color="tab:orange",
            linewidth=1, marker="o", markersize=2)
    ax.plot(x_values, p2y, label="pass2 Y (from grid(0,0))", color="tab:blue",
            linewidth=1, linestyle="--", marker="o", markersize=2)
    ax.set_ylabel(f"Y ({unit})"); ax.set_ylim(YLIM); ax.legend(fontsize=7); ax.grid(True)

    # Row 2: ユークリッド距離
    ax = axes[2]
    ax.plot(x_values, euclid2, label="Euclidean dist fine2 (from selected grid)",
            color="tab:purple", linewidth=1, marker="o", markersize=2)
    ax.set_ylabel(f"Distance ({unit})"); ax.set_ylim(0, YLIM[1])
    ax.legend(fontsize=7); ax.grid(True); ax.set_xlabel(x_label)

    axes[0].set_title(f"[pass2 grid selection + fine2 vs pass2 distance]  {pos_name}")
    fig.tight_layout()
    save_figure(fig, params={**base_params, "pass": "fine_shifts"},
                description=f"shift_timeseries_fine_ecc {pos_name}",
                data={"x_values": x_values,
                      "fine2_x": f2x, "fine2_y": f2y,
                      "pass2_x": p2x, "pass2_y": p2y,
                      "euclid2": euclid2,
                      "p2_grid_xi": p2_grid_xi, "p2_grid_yi": p2_grid_yi})
    plt.close(fig)

    print(f"[shift_visualize] 2pass figures saved: {pos_name}")


def visualize_exclusion_summary(
    csv_path,
    json_path=None,
    time_interval_min=TIME_INTERVAL_MIN,
):
    """
    pos_shifts_exclusion_summary.csv を読み込んで除外チャネルの内訳を可視化する。

    上パネル: per-frame の n_used (有効チャネル数) 時系列
    下パネル: 除外チャネル数を理由別に積み上げ棒グラフ（low_ecc / mad / failed）
    出力ファイル: pos_shifts_exclusion_summary.png（save_figure 経由）
    """
    import csv as _csv

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rows.append({k: int(v) for k, v in row.items()})

    if not rows:
        print(f"[shift_visualize] exclusion_summary: データなし ({csv_path})")
        return

    frames = np.array([r["frame_index"] for r in rows])
    n_used       = np.array([r["n_used"]         for r in rows])
    n_total      = np.array([r["n_total"]         for r in rows])
    n_failed     = np.array([r["n_excl_failed"]   for r in rows])
    n_low_ecc    = np.array([r["n_excl_low_ecc"]  for r in rows])
    n_mad        = np.array([r["n_excl_mad"]       for r in rows])

    pos_name = "unknown"
    if json_path is not None:
        try:
            with open(json_path, "r") as f:
                pos_name = json.load(f).get("pos_name", "unknown")
        except Exception:
            pass

    if time_interval_min is not None:
        x_values = frames * time_interval_min / 60
        x_label = "Time (h)"
    else:
        x_values = frames
        x_label = "Frame number"

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 上パネル: n_used 時系列
    ax0 = axes[0]
    ax0.fill_between(x_values, n_used, alpha=0.4, color="tab:green", label="n_used")
    ax0.plot(x_values, n_used, color="tab:green", linewidth=1)
    ax0.plot(x_values, n_total, color="gray", linewidth=0.8, linestyle="--", label="n_total")
    ax0.set_ylabel("Channels used")
    ax0.set_ylim(0, int(n_total.max()) + 1)
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.4)
    ax0.set_title(f"Channel exclusion summary  {pos_name}")

    # 下パネル: 除外内訳積み上げ棒グラフ
    ax1 = axes[1]
    width = x_values[1] - x_values[0] if len(x_values) > 1 else 1.0
    ax1.bar(x_values, n_failed,  width=width * 0.8, label="alignment_failed",  color="tab:red",    alpha=0.8)
    ax1.bar(x_values, n_low_ecc, width=width * 0.8, label="low_ecc_score",      color="tab:orange", alpha=0.8,
            bottom=n_failed)
    ax1.bar(x_values, n_mad,     width=width * 0.8, label="channel_outlier_mad", color="tab:purple", alpha=0.8,
            bottom=n_failed + n_low_ecc)
    ax1.set_ylabel("Excluded channels")
    ax1.set_xlabel(x_label)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.4, axis="y")

    fig.tight_layout()

    params = {
        "data_source": str(csv_path),
        "n_frames": len(frames),
        "time_interval_min": time_interval_min,
    }
    save_figure(fig, params=params,
                description=f"channel_exclusion_summary {pos_name}")
    plt.close(fig)
    print(f"[shift_visualize] exclusion_summary saved: {pos_name}")


# スタンドアロン実行時
if __name__ == "__main__":
    visualize_shifts(JSON_PATH)

# %%
