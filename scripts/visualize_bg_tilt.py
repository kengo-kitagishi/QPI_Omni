"""
visualize_bg_tilt.py
---------------------
grid_subtracted TIFFスタックの列平均プロファイルに
pybaselines + scipy UnivariateSpline を適用し、
どれが背景をよく捉えているか目視確認するための比較図を出力する。

使い方:
  python visualize_bg_tilt.py

設定:
  STACK_PATH  : 対象の grid_sub TIF スタック
  N_FRAMES    : 均等サンプリングするフレーム数
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from pybaselines import Baseline
from scipy.interpolate import UnivariateSpline

_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

from figure_logger import save_figure

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
STACK_PATH = r"C:\ph_1\Pos1\output_phase\channels\grid_subtracted\channel_00_grid_sub.tif"
N_FRAMES        = 12    # 表示フレーム数
FRAME_START     = 0     # 開始フレーム（連続サンプリングの起点）
FRAME_STEP      = 5     # 連続フレームのステップ（1=毎フレーム、5=5枚おき）
ROWS_PER_FIG    = 4     # 1枚の図に入れる行数
CLIP_FRACTION   = 0.10  # 両端 10% ずつカット → 中央 80% を使用
IMG_VMIN        = -0.5
IMG_VMAX        = 2.0
# 比較するアルゴリズム: (ラベル, メソッド名, kwargs)
ALGORITHMS = [
    ("pspline_iarpls",    "pspline_iarpls",    dict(lam=1e5, num_knots=20)),
    ("derpsalsa",         "derpsalsa",         dict(lam=1e5)),
    ("pspline_aspls",     "pspline_aspls",     dict(lam=1e5, num_knots=20)),
    ("pspline_derpsalsa", "pspline_derpsalsa", dict(lam=1e5, num_knots=20)),
    ("penalized_poly",    "penalized_poly",    dict(poly_order=3)),
    ("mpspline",          "mpspline",          dict(lam=1e5, num_knots=20)),
]

_ALL_COLORS = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
COLORS = _ALL_COLORS[:len(ALGORITHMS)]

# ─────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────
def load_tiff_stack(path: str) -> np.ndarray:
    img = Image.open(path)
    frames = []
    for i in range(100000):
        try:
            img.seek(i)
            frames.append(np.array(img, dtype=np.float64))
        except EOFError:
            break
    return np.stack(frames, axis=0)


def pick_frame_indices(n_total: int, n_pick: int) -> np.ndarray:
    if n_total <= n_pick:
        return np.arange(n_total)
    return np.linspace(0, n_total - 1, n_pick, dtype=int)


def compute_baselines(profile: np.ndarray) -> dict:
    """1D プロファイルに各アルゴリズムを適用してベースラインを返す"""
    x = np.arange(len(profile))
    bl_fitter = Baseline(x_data=x)
    results = {}
    for label, method_name, kwargs in ALGORITHMS:
        try:
            if method_name == "univariate_spline":
                s_factor = kwargs.get("s_factor", 1.0)
                k = kwargs.get("k", 3)
                s = len(profile) * s_factor
                spl = UnivariateSpline(x, profile, s=s, k=k)
                results[label] = spl(x)
            else:
                bl_func = getattr(bl_fitter, method_name)
                baseline, _ = bl_func(profile, **kwargs)
                results[label] = baseline
        except Exception as e:
            print(f"  [{label}] failed: {e}")
            results[label] = np.full_like(profile, np.nan)
    return results


# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────
def main():
    stack_path = Path(STACK_PATH)
    if not stack_path.exists():
        print(f"ERROR: File not found: {stack_path}")
        sys.exit(1)

    print(f"Loading: {stack_path}")
    stack = load_tiff_stack(str(stack_path))
    n_frames, h, w = stack.shape
    print(f"  Shape: {stack.shape}")

    # 連続フレームサンプリング
    indices = np.arange(FRAME_START, FRAME_START + N_FRAMES * FRAME_STEP, FRAME_STEP)
    indices = indices[indices < n_frames]

    # 端アーティファクト除去: 中央 (1 - 2*CLIP_FRACTION) の範囲だけ使う
    clip_px  = int(w * CLIP_FRACTION)
    x_start  = clip_px
    x_end    = w - clip_px  # exclusive
    x        = np.arange(x_start, x_end)

    print(f"  Column mean (all {h} rows), frames={indices.tolist()}")
    print(f"  x range: [{x_start}, {x_end}) ({x_end - x_start} px, clipped {clip_px}px each side)")

    # ─── 全フレームのプロファイル・ベースラインを先に計算 ───
    all_profiles  = []
    all_baselines = []
    for frame_idx in indices:
        profile = stack[frame_idx, :, x_start:x_end].mean(axis=0)
        all_profiles.append(profile)
        all_baselines.append(compute_baselines(profile))

    global_left_ylim  = (-0.5, 1.0)
    global_right_ylim = (-0.25, 0.5)

    # ─── 図1系: ROWS_PER_FIG 行ごとに分割 ───────────────────
    figs1 = []
    chunks = [list(range(i, min(i + ROWS_PER_FIG, len(indices)))) for i in range(0, len(indices), ROWS_PER_FIG)]
    for chunk_idx, chunk_rows in enumerate(chunks):
        n_rows = len(chunk_rows)
        fig1, axes1 = plt.subplots(
            n_rows, 3,
            figsize=(18, 5 * n_rows),
            squeeze=False,
        )
        fig1.suptitle(
            f"BG baseline algorithms — column mean (all {h} rows) [{chunk_idx+1}/{len(chunks)}]\n{stack_path.name}",
            fontsize=11,
        )

        for row, fi in enumerate(chunk_rows):
            frame_idx = indices[fi]
            profile   = all_profiles[fi]
            bls       = all_baselines[fi]

            ax_left  = axes1[row, 0]
            ax_right = axes1[row, 1]
            ax_img   = axes1[row, 2]

            # ─ 左列: プロファイル + ベースライン ─
            ax_left.plot(x, profile, color="black", lw=1.0, alpha=0.6, label="profile")
            for (label, *_), color in zip(ALGORITHMS, COLORS):
                bl = bls[label]
                ax_left.plot(x, bl, color=color, lw=1.0, alpha=0.5, label=label)
            ax_left.set_ylim(global_left_ylim)
            ax_left.set_title(f"frame {frame_idx}", fontsize=10)
            ax_left.set_xlabel("x (pixel)", fontsize=8)
            ax_left.set_ylabel("phase (rad)", fontsize=8)
            ax_left.tick_params(labelsize=8)
            ax_left.grid(True, alpha=0.3)
            if row == 0:
                ax_left.legend(fontsize=7, loc="upper right", framealpha=0.8, ncol=2)

            # ─ 中列: 残差 (profile - baseline) ─
            for (label, *_), color in zip(ALGORITHMS, COLORS):
                bl = bls[label]
                ax_right.plot(x, profile - bl, color=color, lw=1.2, label=label)
            ax_right.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
            ax_right.set_ylim(global_right_ylim)
            ax_right.set_title(f"frame {frame_idx} — residual", fontsize=10)
            ax_right.set_xlabel("x (pixel)", fontsize=8)
            ax_right.set_ylabel("Δphase (rad)", fontsize=8)
            ax_right.tick_params(labelsize=8)
            ax_right.grid(True, alpha=0.3)
            if row == 0:
                ax_right.legend(fontsize=7, loc="upper right", framealpha=0.8, ncol=2)

            # ─ 右列: TIF画像（全幅）─
            im = ax_img.imshow(
                stack[frame_idx],
                cmap="RdBu_r", vmin=IMG_VMIN, vmax=IMG_VMAX,
                aspect="auto", origin="upper",
            )
            ax_img.axvline(x_start, color="white", lw=0.8, ls="--", alpha=0.7)
            ax_img.axvline(x_end,   color="white", lw=0.8, ls="--", alpha=0.7)
            ax_img.set_title(f"frame {frame_idx} — image", fontsize=10)
            ax_img.set_xlabel("x (pixel)", fontsize=8)
            ax_img.set_ylabel("y (pixel)", fontsize=8)
            ax_img.tick_params(labelsize=8)
            plt.colorbar(im, ax=ax_img, fraction=0.03, pad=0.02)

        fig1.tight_layout(rect=[0, 0, 1, 0.97])
        figs1.append(fig1)

    # ─── 安定性図: 引いた後の residual を全フレーム重ね書き ───────
    print(f"  Computing subtracted residuals over all {n_frames} frames...")
    all_residuals = []
    for fi in range(n_frames):
        prof = stack[fi, :, x_start:x_end].mean(axis=0)
        bl = compute_baselines(prof).get("pspline_iarpls", np.full(len(x), np.nan))
        all_residuals.append(prof - bl)

    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5))
    fig2.suptitle(
        f"pspline_iarpls: subtracted residual — all {n_frames} frames\n{stack_path.name}",
        fontsize=11,
    )
    for residual in all_residuals:
        ax2.plot(x, residual, color="steelblue", lw=0.3, alpha=0.08)
    ax2.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax2.set_ylim(-0.15, 0.15)
    ax2.set_ylabel("phase − baseline (rad)", fontsize=9)
    ax2.set_xlabel("x (pixel)", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    # ─── 保存 ───────────────────────────────────
    params = {
        "stack": str(stack_path),
        "profile_method": f"column_mean_all_{h}_rows",
        "n_frames": N_FRAMES,
        "frame_indices": indices.tolist(),
        "algorithms": [a[0] for a in ALGORITHMS],
    }

    try:
        for i, fig1 in enumerate(figs1):
            save_figure(
                fig1, params=params,
                description=f"iarpls profile+residual per frame (page {i+1})",
            )
        save_figure(
            fig2, params=params,
            description=f"iarpls stability: all {n_frames} frames, mean±1σ, delta_std",
        )
    except Exception as e:
        print(f"[warn] save_figure: {e}")
        out_dir = _script_dir.parent / "results" / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, fig1 in enumerate(figs1):
            fig1.savefig(out_dir / f"bg_tilt_iarpls_p{i+1}.png", dpi=150, bbox_inches="tight")
        fig2.savefig(out_dir / "bg_tilt_iarpls_frames.png", dpi=150, bbox_inches="tight")
        print(f"  Saved to {out_dir}")

    plt.show()


if __name__ == "__main__":
    main()
