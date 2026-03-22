"""
quick_grid_sub.py — TL output_phase から grid 00 を引いて保存
"""
from pathlib import Path
import numpy as np
import tifffile
from tqdm import tqdm

# ── 設定 ──────────────────────────────────────────────
TL_DIR   = Path(r"C:\ph_1\Pos1")
GRID_REF = Path(r"E:\Acuisition\kitagishi\260317_0p0055\grid_0p5_0p5_0p1_exp60ms_allpos_EMM2_1\Pos1_x+0_y+0")
OUT_DIR  = Path(r"C:\ph_1\Pos1\output_phase_grid_sub")
TL_Z     = 0   # タイムラプスの ch インデックス (ph_000)
GRID_Z   = 2   # grid 00 の z インデックス (ph_002)
# ──────────────────────────────────────────────────────

def main():
    # grid 参照フレームを読み込む
    grid_path = GRID_REF / "output_phase" / f"img_000000000_ph_{GRID_Z:03d}_phase.tif"
    if not grid_path.exists():
        raise FileNotFoundError(f"grid ref not found: {grid_path}")
    grid_ref = tifffile.imread(str(grid_path)).astype(np.float32)
    print(f"grid ref: {grid_path.name}  shape={grid_ref.shape}  dtype={grid_ref.dtype}")

    # タイムラプスフレームリストを取得
    tl_frames = sorted((TL_DIR / "output_phase").glob(f"img_*_ph_{TL_Z:03d}_phase.tif"))
    if not tl_frames:
        raise FileNotFoundError(f"no TL frames found in {TL_DIR / 'output_phase'}")
    print(f"TL frames: {len(tl_frames)}")

    # 出力ディレクトリを作成
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 各フレームを処理
    for fp in tqdm(tl_frames, desc="grid subtract"):
        frame = tifffile.imread(str(fp)).astype(np.float32)
        sub = frame - grid_ref
        tifffile.imwrite(str(OUT_DIR / fp.name), sub)

    print(f"Done → {OUT_DIR}  ({len(tl_frames)} files)")


if __name__ == "__main__":
    main()
