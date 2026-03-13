"""
plot_shift_invariant_profiles.py
--------------------------------
シフト量が大きく異なる10枚のフレームを選び、その中心横線プロファイルを重ねて表示する。
「プロットの形がシフト量に依存せず一定である」ことを示す。

使い方:
  python plot_shift_invariant_profiles.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

from figure_logger import save_figure


SHIFT_NPZ = r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\figure-hub\inbox\2026-03-12\pipeline_full\pipeline_full_20260312T132504Z_ed7cca\pipeline_full__pipeline_full_20260312T132504Z_ed7cca__f012_data.npz"

TIFF_PATHS = [
    r"D:\AquisitionData\Kitagishi\260310\timelapse_11day_exp200ms_1pos_EMM2\Pos1\output_phase\channels\channel_02_bg_corr.tif",
    r"D:\AquisitionData\Kitagishi\260310\timelapse_11day_exp200ms_1pos_EMM2\Pos1\output_phase\channels\grid_subtracted\channel_02_grid_sub.tif",
]
TIFF_LABELS = ["channel_02_bg_corr", "channel_02_grid_sub"]

N_FRAMES = 2  # shift_x が最も離れている2枚のみ
# bg_corr=subtract前, grid_sub=subtract後
Y_SCALE_BG_CORR = (-5.0, 1.0)
Y_SCALE_GRID_SUB = (-1.0, 1.0)


def load_tiff_stack(path: str) -> np.ndarray:
    img = Image.open(path)
    frames = []
    for i in range(1000):
        try:
            img.seek(i)
            frames.append(np.array(img))
        except EOFError:
            break
    return np.stack(frames, axis=0)


def pick_diverse_shift_frames(npz_path: str, n_pick: int) -> np.ndarray:
    """shift_x が最も離れている n_pick 枚を選ぶ（最小と最大）"""
    data = np.load(npz_path)
    sx = data["pass2_shift_x"]
    sy = data["pass2_shift_y"]

    order_asc = np.argsort(sx)
    # 最小(マイナス最大)と最大(プラス最大)の2枚
    idx_min = order_asc[0]
    idx_max = order_asc[-1]
    return np.array([idx_min, idx_max])


def main():
    npz_path = Path(SHIFT_NPZ)
    if not npz_path.exists():
        print(f"ERROR: Shift npz not found: {npz_path}")
        sys.exit(1)

    indices = pick_diverse_shift_frames(str(npz_path), N_FRAMES)
    data = np.load(str(npz_path))
    sx, sy = data["pass2_shift_x"], data["pass2_shift_y"]
    print(f"Selected frames (shift-diverse): {indices.tolist()}")
    for i in indices:
        print(f"  frame {i}: shift_x={sx[i]:.4f}, shift_y={sy[i]:.4f}")

    out_dir = _script_dir.parent / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tiff_path, label in zip(TIFF_PATHS, TIFF_LABELS):
        path = Path(tiff_path)
        if not path.exists():
            print(f"SKIP (not found): {path}")
            continue

        y_scale = Y_SCALE_BG_CORR if "bg_corr" in label else Y_SCALE_GRID_SUB
        print(f"\nLoading: {path.name} (Y scale: {y_scale})")
        stack = load_tiff_stack(str(path))
        n_frames, h, w = stack.shape
        center_y = h // 2

        profiles = []
        for i in indices:
            if i >= n_frames:
                continue
            line = stack[i, center_y, :].astype(np.float64)
            profiles.append(line)

        if len(profiles) < N_FRAMES:
            print(f"  WARNING: only {len(profiles)} frames (expected {N_FRAMES})")

        x = np.arange(w)
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, prof in zip(indices, profiles):
            ax.plot(x, prof, label=f"f{idx} (Δx={sx[idx]:.2f}, Δy={sy[idx]:.2f})", alpha=0.7)

        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("pixel value")
        ax.set_ylim(y_scale)
        ax.set_title(f"Shift-invariant: center line profiles (y={center_y})\n{path.name}\n10 frames with diverse shifts")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        base = f"shift_invariant_profiles_{label}"
        try:
            save_figure(
                fig,
                params={"tiff": str(path), "frame_indices": indices.tolist(), "center_y": center_y},
                data={"x": x, "frame_indices": indices, "profiles": np.array(profiles)},
                description=f"Shift-invariant profiles {label}",
            )
        except Exception as e:
            print(f"[warn] save_figure: {e}")
        fig_path = out_dir / f"{base}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
