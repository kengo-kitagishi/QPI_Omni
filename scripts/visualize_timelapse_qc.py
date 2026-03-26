"""
visualize_timelapse_qc.py
--------------------------
タイムラプスTIFシーケンスのQC可視化スクリプト。

対象データ:
  - 511×511 再構成位相画像 (output_phase/)
  - 40×440 crop_subtracted/*.tif (compute_pos_shifts.py 出力)
  - 40×440 output_phase_iarpls/*.tif (timelapse_iarpls_bgsub.py 出力)

生成する図:
  Fig1  Frame montage         : 等間隔サンプリングしたフレームのグリッド表示
  Fig2  Temporal profile map  : 列平均プロファイル × 時刻 ヒートマップ
  Fig3  Frame statistics      : mean/std/p5/p95 の時系列 + ジャンプ検出
  Fig4  Histogram evolution   : 等間隔フレームの輝度ヒストグラム重ね書き
  Fig5  Inter-frame MAD       : フレーム間差分の時系列 + 行/列ヒートマップ
  Fig6  Before/after compare  : TIF_DIR_B 指定時のみ (iarpls 前後比較)
"""

import sys
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

# ============================================================
# 設定パラメータ
# ============================================================

# --- Primary input ---
TIF_DIR_A = r"C:\ph\Pos1\output_phase_iarpls"        # 検査対象ディレクトリ
TIF_PATTERN = "img_*_ph_000_phase.tif"

# --- Optional comparison (None = skip) ---
TIF_DIR_B = r"C:\ph\Pos1\output_phase\channels\crop_subtracted"   # iarpls前との比較

# --- Optional spatial crop applied after loading ---
# 511×511 画像から 40×440 ROI を切り出す場合に設定。
# None = 切り出しなし（40×440 TIF を直接読む場合も None でよい）
# 形式: (cy, cx, crop_h, crop_w) — timelapse_iarpls_bgsub.py と同じ規約
ROI_CROP = None   # 40×440 TIF を直接読むので不要

# 光学パラメータ（x軸をμm表示にしたい場合）
PIXEL_SCALE_UM = 0.34568   # px → μm。None で px 表示
INTERVAL_MIN   = 5.0       # タイムポイント間の時間 [分]

# --- 表示 & サンプリング ---
VMIN, VMAX      = -1.0, 1.0  # フレーム表示のカラーマップ範囲
FRAME_STEP      = None       # None = auto (RAM ~60 frames)
N_MONTAGE_FRAMES = 16        # montage に表示するフレーム数
N_HIST_FRAMES   = 24         # histogram overlay のフレーム数
MAX_FRAMES      = None       # None = all
# ============================================================


def extract_rect_roi(img, cy, cx, crop_h, crop_w):
    """timelapse_iarpls_bgsub.py と同じ ROI crop ロジック。"""
    h, w = img.shape[-2], img.shape[-1]
    y1 = cy - crop_h // 2;  y2 = y1 + crop_h
    x1 = cx - crop_w // 2;  x2 = x1 + crop_w
    pad_y0 = max(0, -y1);   y1 = max(0, y1)
    pad_y1 = max(0, y2 - h); y2 = min(h, y2)
    pad_x0 = max(0, -x1);   x1 = max(0, x1)
    pad_x1 = max(0, x2 - w); x2 = min(w, x2)
    if img.ndim == 2:
        crop = img[y1:y2, x1:x2]
        if any([pad_y0, pad_y1, pad_x0, pad_x1]):
            crop = np.pad(crop, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant")
    else:
        crop = img[..., y1:y2, x1:x2]
        if any([pad_y0, pad_y1, pad_x0, pad_x1]):
            crop = np.pad(crop, ((0, 0), (pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant")
    return crop


def mad(arr):
    m = np.median(arr)
    return np.median(np.abs(arr - m))


def load_sequence(tif_dir, pattern, step, max_frames, roi_crop):
    """ソートされた TIF シーケンスを読み込む。返値: (stack: N×H×W, file_indices: list[int])"""
    files = sorted(Path(tif_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"TIF が見つかりません: {tif_dir}/{pattern}")
    if max_frames is not None:
        files = files[:max_frames]
    total = len(files)

    if step is None:
        step = max(1, total // 60)

    sampled = files[::step]
    indices = list(range(0, total, step))[:len(sampled)]

    frames = []
    for f in tqdm(sampled, desc=f"Loading {Path(tif_dir).name}"):
        img = tifffile.imread(str(f)).astype(np.float32)
        if roi_crop is not None:
            cy, cx, crop_h, crop_w = roi_crop
            img = extract_rect_roi(img, cy, cx, crop_h, crop_w)
        frames.append(img)

    return np.stack(frames, axis=0), indices, total


def set_pub_style():
    """Publication-quality matplotlib スタイル設定。"""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 150,
    })


def x_ticks(ax, n_cols, pixel_scale_um, axis="x"):
    """x or y 軸を px / μm 表示に設定。"""
    if pixel_scale_um is None:
        label = "x (px)" if axis == "x" else "y (px)"
        if axis == "x":
            ax.set_xlabel(label)
        else:
            ax.set_ylabel(label)
    else:
        ticks = np.linspace(0, n_cols - 1, min(6, n_cols))
        labels = [f"{t * pixel_scale_um:.1f}" for t in ticks]
        if axis == "x":
            ax.set_xticks(ticks); ax.set_xticklabels(labels)
            ax.set_xlabel("x (μm)")
        else:
            ax.set_yticks(ticks); ax.set_yticklabels(labels)
            ax.set_ylabel("y (μm)")


# ============================================================
# Figure 1: Frame montage
# ============================================================
def fig_montage(stack, indices, total, n_show=16, vmin=-1.0, vmax=1.0, interval_min=5.0):
    """等間隔サンプリングしたフレームを 4×N グリッドで表示。"""
    sel_idx = np.round(np.linspace(0, len(stack) - 1, n_show)).astype(int)
    n_cols = 4
    n_rows = int(np.ceil(n_show / n_cols))
    H, W = stack.shape[1], stack.shape[2]

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * W * 0.022 + 0.5, n_rows * H * 0.1 + 0.5),
                              squeeze=False)
    # Adjust figure size to reasonable bounds
    fig_w = max(8, min(20, n_cols * 3))
    fig_h = max(4, min(16, n_rows * max(1.0, H / W * 2.5)))
    fig.set_size_inches(fig_w, fig_h)

    for k, si in enumerate(sel_idx):
        r, c = divmod(k, n_cols)
        ax = axes[r][c]
        frame_no = indices[si]
        t_min = frame_no * interval_min
        im = ax.imshow(stack[si], cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"T={frame_no}  ({t_min:.0f} min)", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for k in range(n_show, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r][c].axis("off")

    fig.suptitle(f"Frame montage  (N={total} frames, showing {n_show})", fontsize=11)
    fig.tight_layout()
    return fig


# ============================================================
# Figure 2: Temporal profile heatmaps (col mean + row mean)
# ============================================================
def fig_profile_heatmap(stack, indices, total, pixel_scale_um=None, interval_min=5.0):
    """
    左: 列平均プロファイル (mean over rows) × 時刻 ヒートマップ
    右: 行平均プロファイル (mean over cols) × 時刻 ヒートマップ
    """
    col_mean = stack.mean(axis=1)   # (N, W)
    row_mean = stack.mean(axis=2)   # (N, H)
    times = np.array(indices) * interval_min

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 列平均プロファイルヒートマップ ---
    ax = axes[0]
    im = ax.imshow(col_mean, aspect="auto", origin="upper",
                   extent=[0, stack.shape[2] - 1, times[-1], times[0]],
                   cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Phase (rad)")
    ax.set_xlabel("x (px)" if pixel_scale_um is None else "x (μm)")
    ax.set_ylabel("Time (min)")
    ax.set_title("Column-mean profile over time")
    if pixel_scale_um is not None:
        W = stack.shape[2]
        ticks = np.linspace(0, W - 1, 6)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t * pixel_scale_um:.1f}" for t in ticks])

    # --- 行平均プロファイルヒートマップ ---
    ax = axes[1]
    im = ax.imshow(row_mean, aspect="auto", origin="upper",
                   extent=[0, stack.shape[1] - 1, times[-1], times[0]],
                   cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Phase (rad)")
    ax.set_xlabel("y (px)" if pixel_scale_um is None else "y (μm)")
    ax.set_ylabel("Time (min)")
    ax.set_title("Row-mean profile over time")
    if pixel_scale_um is not None:
        H = stack.shape[1]
        ticks = np.linspace(0, H - 1, 6)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t * pixel_scale_um:.1f}" for t in ticks])

    fig.suptitle("Temporal profile heatmaps", fontsize=11)
    fig.tight_layout()
    return fig, col_mean, row_mean


# ============================================================
# Figure 3: Frame statistics over time
# ============================================================
def fig_frame_stats(stack, indices, interval_min=5.0):
    """mean / std / p5 / p95 の時系列と外れ値フレームのハイライト。"""
    times = np.array(indices) * interval_min

    stat_funcs = {
        "Mean": lambda s: s.reshape(len(s), -1).mean(axis=1),
        "Std":  lambda s: s.reshape(len(s), -1).std(axis=1),
        "p5":   lambda s: np.percentile(s.reshape(len(s), -1), 5,  axis=1),
        "p95":  lambda s: np.percentile(s.reshape(len(s), -1), 95, axis=1),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes_flat = axes.flatten()

    for ax, (label, fn) in zip(axes_flat, stat_funcs.items()):
        vals = fn(stack)
        ax.plot(times, vals, lw=0.8, color="#2166ac")

        # ジャンプ検出: 差分の MAD ベース外れ値
        diff = np.abs(np.diff(vals))
        if len(diff) > 3:
            m = mad(diff)
            if m > 0:
                jump_mask = diff > 3 * m
                jump_t = times[1:][jump_mask]
                for jt in jump_t:
                    ax.axvline(jt, color="#d6604d", lw=0.8, alpha=0.7, linestyle="--")

        ax.set_ylabel(f"{label} (rad)")
        ax.set_title(label)
        ax.grid(True, lw=0.4, alpha=0.5)

    for ax in axes[1]:
        ax.set_xlabel("Time (min)")

    fig.suptitle("Frame statistics over time  (dashed = jump detected)", fontsize=11)
    fig.tight_layout()

    stats = {
        "mean": stat_funcs["Mean"](stack),
        "std":  stat_funcs["Std"](stack),
        "p5":   stat_funcs["p5"](stack),
        "p95":  stat_funcs["p95"](stack),
        "times_min": times,
    }
    return fig, stats


# ============================================================
# Figure 4: Histogram evolution
# ============================================================
def fig_histogram(stack, indices, n_frames=24, vmin=-1.0, vmax=1.0):
    """等間隔フレームのヒストグラムを時刻でカラーマップして重ね書き。"""
    sel_idx = np.round(np.linspace(0, len(stack) - 1, n_frames)).astype(int)
    cmap = plt.cm.coolwarm
    bins = np.linspace(vmin - 0.5, vmax + 0.5, 120)

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, si in enumerate(sel_idx):
        color = cmap(k / max(1, len(sel_idx) - 1))
        vals = stack[si].ravel()
        hist, edges = np.histogram(vals, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, hist, lw=0.7, color=color, alpha=0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=mcolors.Normalize(vmin=indices[0], vmax=indices[-1]))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Frame index")

    ax.set_xlabel("Phase (rad)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Histogram evolution  ({n_frames} frames, blue=early, red=late)")
    ax.grid(True, lw=0.4, alpha=0.5)
    fig.tight_layout()
    return fig


# ============================================================
# Figure 5: Inter-frame MAD
# ============================================================
def fig_interframe_mad(stack, indices, interval_min=5.0):
    """
    上: フレーム間 MAD の時系列
    中: 列ごとの inter-frame diff ヒートマップ (W × time)
    下: 行ごとの inter-frame diff ヒートマップ (H × time)
    """
    if len(stack) < 2:
        print("[skip] inter-frame MAD: フレーム数が 2 未満")
        return None, None

    diff = np.abs(np.diff(stack, axis=0))   # (N-1, H, W)
    mad_per_frame   = np.median(diff.reshape(len(diff), -1), axis=1)
    col_mad = diff.mean(axis=1)   # (N-1, W) — 行方向平均
    row_mad = diff.mean(axis=2)   # (N-1, H) — 列方向平均
    times_diff = (np.array(indices[1:]) + np.array(indices[:-1])) / 2 * interval_min

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.5, 1, 1])

    # -- 上: MAD 時系列 --
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(times_diff, mad_per_frame, lw=0.8, color="#2166ac")
    # スパイク検出
    m = mad(mad_per_frame)
    if m > 0:
        spike_mask = mad_per_frame > 3 * m
        ax0.scatter(times_diff[spike_mask], mad_per_frame[spike_mask],
                    s=20, color="#d6604d", zorder=5, label="spike (>3×MAD)")
        if spike_mask.any():
            ax0.legend(fontsize=8)
    ax0.set_xlabel("Time (min)")
    ax0.set_ylabel("Median |Δframe| (rad)")
    ax0.set_title("Inter-frame MAD over time")
    ax0.grid(True, lw=0.4, alpha=0.5)

    # -- 中: 列ごとヒートマップ --
    ax1 = fig.add_subplot(gs[1, :])
    im1 = ax1.imshow(col_mad.T, aspect="auto", origin="upper",
                     extent=[times_diff[0], times_diff[-1], stack.shape[2] - 1, 0],
                     cmap="hot")
    plt.colorbar(im1, ax=ax1, label="Mean |Δ| (rad)")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("x (px)")
    ax1.set_title("Column-wise inter-frame diff (mean over rows)")

    # -- 下: 行ごとヒートマップ --
    ax2 = fig.add_subplot(gs[2, :])
    im2 = ax2.imshow(row_mad.T, aspect="auto", origin="upper",
                     extent=[times_diff[0], times_diff[-1], stack.shape[1] - 1, 0],
                     cmap="hot")
    plt.colorbar(im2, ax=ax2, label="Mean |Δ| (rad)")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("y (px)")
    ax2.set_title("Row-wise inter-frame diff (mean over cols)")

    fig.suptitle("Inter-frame MAD analysis", fontsize=11)
    fig.tight_layout()

    ifd_data = {"mad_per_frame": mad_per_frame, "col_mad": col_mad,
                "row_mad": row_mad, "times_diff_min": times_diff}
    return fig, ifd_data


# ============================================================
# Figure 6: Before/after comparison (TIF_DIR_B != None)
# ============================================================
def fig_before_after(stack_a, stack_b, indices, vmin=-1.0, vmax=1.0, interval_min=5.0):
    """4フレームについて DIR_A / DIR_B の画像と列平均プロファイルを比較。"""
    n = min(len(stack_a), len(stack_b))
    sel = [0, n // 3, 2 * n // 3, n - 1]

    fig, axes = plt.subplots(3, 4, figsize=(16, 8))

    for col_idx, si in enumerate(sel):
        frame_no = indices[si]
        t_min = frame_no * interval_min

        # Row 0: DIR_A
        ax = axes[0][col_idx]
        ax.imshow(stack_a[si], cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"A  T={frame_no} ({t_min:.0f} min)", fontsize=8)
        ax.axis("off")

        # Row 1: DIR_B
        ax = axes[1][col_idx]
        ax.imshow(stack_b[si], cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"B  T={frame_no} ({t_min:.0f} min)", fontsize=8)
        ax.axis("off")

        # Row 2: column-mean profile overlay
        ax = axes[2][col_idx]
        prof_a = stack_a[si].mean(axis=0)
        prof_b = stack_b[si].mean(axis=0)
        ax.plot(prof_a, lw=0.9, color="#2166ac", label="A")
        ax.plot(prof_b, lw=0.9, color="#d6604d", label="B", linestyle="--")
        ax.set_xlabel("x (px)", fontsize=8)
        ax.set_ylabel("Phase (rad)", fontsize=8)
        if col_idx == 0:
            ax.legend(fontsize=7)
        ax.grid(True, lw=0.4, alpha=0.5)
        ax.tick_params(labelsize=7)

    axes[0][0].set_ylabel("DIR_A", fontsize=9)
    axes[1][0].set_ylabel("DIR_B", fontsize=9)
    fig.suptitle("Before / After comparison  (A = primary, B = comparison)", fontsize=11)
    fig.tight_layout()
    return fig


# ============================================================
# Main
# ============================================================
def main():
    set_pub_style()

    print(f"Loading DIR_A: {TIF_DIR_A}")
    step = FRAME_STEP
    stack_a, indices, total = load_sequence(TIF_DIR_A, TIF_PATTERN, step, MAX_FRAMES, ROI_CROP)
    actual_step = indices[1] - indices[0] if len(indices) > 1 else 1
    print(f"  Loaded {len(stack_a)} frames (step={actual_step}, total={total}), shape={stack_a.shape[1:]}")

    stack_b = None
    if TIF_DIR_B is not None:
        print(f"Loading DIR_B: {TIF_DIR_B}")
        stack_b, indices_b, _ = load_sequence(TIF_DIR_B, TIF_PATTERN, actual_step, MAX_FRAMES, ROI_CROP)
        # DIR_B のフレーム数を DIR_A に合わせる
        min_n = min(len(stack_a), len(stack_b))
        stack_a  = stack_a[:min_n]
        stack_b  = stack_b[:min_n]
        indices  = indices[:min_n]
        print(f"  Loaded {len(stack_b)} frames from DIR_B")

    params = {
        "tif_dir_a": str(TIF_DIR_A),
        "tif_dir_b": str(TIF_DIR_B) if TIF_DIR_B else None,
        "roi_crop": ROI_CROP,
        "frame_step": actual_step,
        "n_loaded": len(stack_a),
        "total_frames": total,
        "shape": list(stack_a.shape[1:]),
        "vmin": VMIN,
        "vmax": VMAX,
        "interval_min": INTERVAL_MIN,
        "pixel_scale_um": PIXEL_SCALE_UM,
    }

    # ----- Fig 1: montage -----
    print("Generating Fig1: montage...")
    fig1 = fig_montage(stack_a, indices, total, N_MONTAGE_FRAMES, VMIN, VMAX, INTERVAL_MIN)
    save_figure(fig1, params=params,
                description=f"Timelapse QC montage ({Path(TIF_DIR_A).name})",
                data={"stack_sample": stack_a[::max(1, len(stack_a)//N_MONTAGE_FRAMES)][:N_MONTAGE_FRAMES],
                      "frame_indices": np.array(indices)})
    plt.close(fig1)
    print("  saved")

    # ----- Fig 2: profile heatmaps -----
    print("Generating Fig2: profile heatmaps...")
    fig2, col_mean, row_mean = fig_profile_heatmap(
        stack_a, indices, total, PIXEL_SCALE_UM, INTERVAL_MIN)
    save_figure(fig2, params=params,
                description="Timelapse QC: temporal profile heatmaps",
                data={"col_mean_profiles": col_mean,
                      "row_mean_profiles": row_mean,
                      "frame_indices": np.array(indices)})
    plt.close(fig2)
    print("  saved")

    # ----- Fig 3: frame stats -----
    print("Generating Fig3: frame statistics...")
    fig3, stats = fig_frame_stats(stack_a, indices, INTERVAL_MIN)
    save_figure(fig3, params=params,
                description="Timelapse QC: frame statistics over time",
                data={k: np.array(v) for k, v in stats.items()})
    plt.close(fig3)
    print("  saved")

    # ----- Fig 4: histogram -----
    print("Generating Fig4: histogram evolution...")
    fig4 = fig_histogram(stack_a, indices, N_HIST_FRAMES, VMIN, VMAX)
    save_figure(fig4, params=params,
                description="Timelapse QC: histogram evolution")
    plt.close(fig4)
    print("  saved")

    # ----- Fig 5: inter-frame MAD -----
    print("Generating Fig5: inter-frame MAD...")
    fig5, ifd_data = fig_interframe_mad(stack_a, indices, INTERVAL_MIN)
    if fig5 is not None:
        save_figure(fig5, params=params,
                    description="Timelapse QC: inter-frame MAD analysis",
                    data={k: np.array(v) for k, v in ifd_data.items()})
        plt.close(fig5)
        print("  saved")

    # ----- Fig 6: before/after (optional) -----
    if stack_b is not None:
        print("Generating Fig6: before/after comparison...")
        fig6 = fig_before_after(stack_a, stack_b, indices, VMIN, VMAX, INTERVAL_MIN)
        save_figure(fig6, params={**params, "dir_a": str(TIF_DIR_A), "dir_b": str(TIF_DIR_B)},
                    description="Timelapse QC: before/after comparison")
        plt.close(fig6)
        print("  saved")

    print("\nDone.")


if __name__ == "__main__":
    main()
