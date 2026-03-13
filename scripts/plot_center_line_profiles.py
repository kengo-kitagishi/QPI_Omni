"""
plot_center_line_profiles.py
----------------------------
スタックされたTIFFの各フレームについて、真ん中の横線（端から端まで）に沿った
ピクセル値をプロットし、データをCSVで保存する。

使い方:
  python plot_center_line_profiles.py

出力:
  - 図: results/figures/ に保存
  - データ: results/center_line_profiles.csv
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

from figure_logger import save_figure


TIFF_PATH = r"D:\AquisitionData\Kitagishi\260310\timelapse_11day_exp200ms_1pos_EMM2\Pos1\output_phase\channels\channel_02_bg_corr.tif"
N_PROFILES = 10
FRAME_OFFSET = 12

# (x範囲, y範囲, 出力サフィックス) のリスト。None は全体
# 全体:-5~1, x150_250:-1~1, x320_420:-5~-3
PLOT_REGIONS = [
    (None, (-5.0, 1.0), "full"),       # 全体
    ((150, 250), (-1.0, 1.0), "x150_250"),
    ((320, 420), (-5.0, -3.0), "x320_420"),
]
OFFSETS_TO_RUN = [0, 10, 20]  # 3セット


def load_tiff_stack(path: str) -> np.ndarray:
    """TIFFスタックを読み込み (T, H, W) または (H, W) を返す"""
    img = Image.open(path)
    frames = []
    for i in range(1000):  # 十分な枚数
        try:
            img.seek(i)
            frames.append(np.array(img))
        except EOFError:
            break
    if not frames:
        raise ValueError(f"No frames found in {path}")
    return np.stack(frames, axis=0)


def pick_frame_indices(n_total: int, n_pick: int, offset: int = 0) -> np.ndarray:
    """スタック全体から均等に n_pick 枚を選ぶ。offset で選択範囲をずらす"""
    if n_total <= n_pick:
        return np.arange(n_total)
    # offset でずらした範囲内で均等に選択
    start = min(offset, n_total - n_pick)
    end = n_total - 1
    indices = np.linspace(start, end, n_pick, dtype=int)
    return indices


def main():
    tiff_path = Path(TIFF_PATH)
    if not tiff_path.exists():
        print(f"ERROR: File not found: {tiff_path}")
        sys.exit(1)

    print(f"Loading: {tiff_path}")
    stack = load_tiff_stack(str(tiff_path))
    n_frames, h, w = stack.shape
    print(f"  Shape: {stack.shape} (frames, height, width)")

    center_y = h // 2
    data_dir = _script_dir.parent / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir = _script_dir.parent / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    offsets = OFFSETS_TO_RUN if OFFSETS_TO_RUN else [FRAME_OFFSET]
    for off in offsets:
        indices = pick_frame_indices(n_frames, N_PROFILES, off)
        print(f"\n--- offset={off} → frames {indices.tolist()} ---")

        profiles = []
        for i in indices:
            line = stack[i, center_y, :].astype(np.float64)
            profiles.append(line)

        file_tag = f"_offset{off}" if len(offsets) > 1 else ""

        for x_range, y_scale, suffix in PLOT_REGIONS:
            if x_range is None:
                x_start, x_end = 0, w - 1
            else:
                x_start, x_end = x_range
            x_slice = slice(x_start, x_end + 1)
            x_vals = np.arange(x_start, x_end + 1)

            fig, ax = plt.subplots(figsize=(8, 5))
            for idx, prof in zip(indices, profiles):
                ax.plot(x_vals, prof[x_slice], label=f"frame {idx}", alpha=0.8)

            ax.set_xlabel("x (pixel)")
            ax.set_ylabel("pixel value")
            ax.set_title(f"Center line profiles x={x_start}~{x_end} (y={center_y}) offset={off}\n{Path(TIFF_PATH).name}")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_start, x_end)
            ax.set_ylim(y_scale)
            fig.tight_layout()

            base = f"center_line_profiles_{suffix}{file_tag}"
            csv_path = data_dir / f"{base}.csv"
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("frame_index,x,pixel_value\n")
                for idx, prof in zip(indices, profiles):
                    for x_val, pv in zip(x_vals, prof[x_slice]):
                        f.write(f"{idx},{x_val},{pv:.6f}\n")
            print(f"  Data: {csv_path}")

            prof_arr = np.array([p[x_slice] for p in profiles])
            try:
                save_figure(
                    fig,
                    params={
                        "tiff": str(tiff_path),
                        "x_range": (x_start, x_end),
                        "y_range": y_scale,
                        "center_y": center_y,
                        "frame_offset": off,
                    },
                    data={
                        "x": x_vals,
                        "frame_indices": np.array(indices),
                        "profiles": prof_arr,
                    },
                    description=f"Center line profiles x{x_start}-{x_end} offset{off}",
                )
            except Exception as e:
                print(f"[warn] save_figure: {e}")
            fig_path = out_dir / f"{base}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"  Figure: {fig_path} (x={x_start}~{x_end}, y={y_scale[0]}~{y_scale[1]})")

    print(f"\nFigure saved to: {out_dir}")


if __name__ == "__main__":
    main()
