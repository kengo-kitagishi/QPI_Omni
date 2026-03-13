# %%
"""
batch_reconstruction_grid.py
----------------------------
generate_grid_pos.py で取得したグリッドデータ（multipos_test_1 等）の
一括 QPI 再構成スクリプト。

ディレクトリ構造:
    GRID_DIR/
        Pos0_x+0_y+0/   ← BG (reconstruction 用)
            img_000000000_ph_000.tif
            img_000000000_ph_001.tif
            ...
            img_000000000_ph_010.tif
        Pos0_x-1_y+0/   ← BG
            ...
        Pos1_x+0_y+0/   ← 再構成対象
            img_000000000_ph_000.tif
            ...
        Pos2_x+0_y+0/   ← 再構成対象
            ...

対応関係:
    PosX_x{xi}_y{yi}  →  BGは Pos0_x{xi}_y{yi} の同じ z を使う

出力:
    GRID_DIR/PosX_x{xi}_y{yi}/output_phase/img_000000000_ph_{z:03d}.tif
"""
import re
import sys
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
from skimage.restoration import unwrap_phase
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ============================================================
# 設定パラメータ
# ============================================================
GRID_DIR = r"E:\Acuisition\kitagishi\260301\multipos_test_1"

# BG として使うベースラベル（通常 "Pos0"）
BG_BASE_LABEL = "Pos0"

# 再構成対象のベースラベル（None で Pos0 以外の全 PosX_x*_y* を処理）
# 特定のPosだけ回したい場合: TARGET_BASE_LABELS = ["Pos1", "Pos2"]
TARGET_BASE_LABELS = None

# QPI 光学パラメータ
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

# ---- Pos番号によるクロップ切り替え ----
# Pos番号 < POS_SPLIT → CROP_BEFORE / Pos番号 >= POS_SPLIT → CROP_AFTER
POS_SPLIT    = 3
CROP_BEFORE  = (0, 2048,   0, 2048)   # Pos0, Pos1, Pos2
CROP_AFTER   = (0, 2048, 400, 2448)   # Pos3 以降
# -----------------------------------------------

# 平均0調整の領域 (row_start, row_end, col_start, col_end)
# None で無効（mean adjustment しない）
MEAN_REGION = None   # 例: (1, 50, 1, 50)

# 再構成済み（output_phase/ が既存）の場合スキップするか
SKIP_IF_EXISTS = True

# PNG カラーマップも保存するか
SAVE_PNG = False
PNG_DPI  = 150
PNG_VMIN = -2.0
PNG_VMAX =  2.0
# 並列処理ワーカー数（None = cpu_count(), 1 = 逐次実行でデバッグ向き）
N_WORKERS = None
# ============================================================

# QPIインポート
try:
    from qpi import QPIParameters, get_field
except ImportError:
    print("ERROR: qpi モジュールが見つかりません。QPI_Omni/scripts が PYTHONPATH にあるか確認してください。")
    sys.exit(1)


def scan_grid_folders(grid_dir: Path):
    """
    grid_dir 内の {label}_x{xi:+d}_y{yi:+d} フォルダを全列挙し、
    {base_label: {(xi, yi): folder_path}} の dict を返す。
    """
    pattern = re.compile(r"^(.+)_x([+-]?\d+)_y([+-]?\d+)$")
    result = {}
    for d in sorted(grid_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            base  = m.group(1)
            xi    = int(m.group(2))
            yi    = int(m.group(3))
            result.setdefault(base, {})[(xi, yi)] = d
    return result


def get_z_files(pos_dir: Path):
    """img_000000000_ph_XXX.tif を z 番号順で返す。"""
    files = sorted(pos_dir.glob("img_*_ph_*.tif"))
    return files


def get_z_index(path: Path):
    """img_000000000_ph_010.tif → 10"""
    m = re.search(r"_ph_(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def get_crop(pos_number: int):
    """Pos番号からクロップ領域を返す。"""
    return CROP_BEFORE if pos_number < POS_SPLIT else CROP_AFTER


def reconstruct_image(img_path: Path, qpi_params, crop):
    """1枚の生画像を読み込んでクロップ→位相再構成し ndarray (float64) で返す。"""
    img = np.array(Image.open(str(img_path)))
    rs, re_, cs, ce = crop
    img = img[rs:re_, cs:ce]
    field = get_field(img, qpi_params)
    phase = unwrap_phase(np.angle(field))
    return phase


def make_qpi_params(sample_img_path: Path, crop):
    """1枚の画像からクロップサイズを取得して QPIParameters を作成。"""
    img = np.array(Image.open(str(sample_img_path)))
    rs, re_, cs, ce = crop
    cropped = img[rs:re_, cs:ce]
    return QPIParameters(
        wavelength=WAVELENGTH,
        NA=NA,
        img_shape=cropped.shape,
        pixelsize=PIXELSIZE,
        offaxis_center=OFFAXIS_CENTER,
    )


def _reconstruct_grid_point(args):
    """ProcessPoolExecutor ワーカー: 1グリッドポイントの全 z スライスを再構成して保存。"""
    xi, yi, target_dir_str, bg_dir_str, crop, pos_number = args
    target_dir = Path(target_dir_str)
    bg_dir     = Path(bg_dir_str)
    out_dir    = target_dir / "output_phase"

    z_files_target = {get_z_index(p): p for p in get_z_files(target_dir)}
    z_files_bg     = {get_z_index(p): p for p in get_z_files(bg_dir)}

    if not z_files_target:
        return xi, yi, False, "z画像なし"

    out_dir.mkdir(exist_ok=True)
    sample_path = next(iter(z_files_target.values()))
    try:
        qpi_params = make_qpi_params(sample_path, crop)
    except Exception as e:
        return xi, yi, False, f"QPIParams: {e}"

    folder_ok = True
    for z_idx, tgt_path in sorted(z_files_target.items()):
        out_path = out_dir / (tgt_path.stem + "_phase.tif")
        if SKIP_IF_EXISTS and out_path.exists():
            continue
        if z_idx not in z_files_bg:
            folder_ok = False
            continue
        try:
            phase_target = reconstruct_image(tgt_path, qpi_params, crop)
            phase_bg     = reconstruct_image(z_files_bg[z_idx], qpi_params, crop)
            phase_diff   = phase_target - phase_bg
            h, w = phase_diff.shape
            if pos_number < POS_SPLIT:
                region = phase_diff[1:h-1, 1:w//2]
            else:
                region = phase_diff[1:h-1, w//2:w-1]
            if region.size > 0:
                phase_diff -= np.mean(region)
            tifffile.imwrite(str(out_path), phase_diff.astype(np.float32))
            if SAVE_PNG:
                import matplotlib.pyplot as plt
                png_path = out_dir / (tgt_path.stem + ".png")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(phase_diff, cmap="RdBu_r", vmin=PNG_VMIN, vmax=PNG_VMAX)
                ax.axis("off")
                ax.set_title(f"{target_dir.name} z={z_idx}")
                plt.tight_layout()
                plt.savefig(str(png_path), dpi=PNG_DPI, bbox_inches="tight")
                plt.close()
        except Exception as e:
            folder_ok = False
    return xi, yi, folder_ok, None


def main():
    grid_dir = Path(GRID_DIR)
    if not grid_dir.exists():
        print(f"ERROR: GRID_DIR が見つかりません: {grid_dir}")
        sys.exit(1)

    # フォルダスキャン
    folders = scan_grid_folders(grid_dir)
    if BG_BASE_LABEL not in folders:
        print(f"ERROR: BG フォルダ '{BG_BASE_LABEL}_x*_y*' が見つかりません: {grid_dir}")
        sys.exit(1)

    bg_map = folders[BG_BASE_LABEL]  # {(xi, yi): Path}

    # 対象ベースラベルを決定
    if TARGET_BASE_LABELS is not None:
        target_labels = TARGET_BASE_LABELS
    else:
        target_labels = [k for k in sorted(folders.keys()) if k != BG_BASE_LABEL]

    print(f"BG ベースラベル: {BG_BASE_LABEL}  ({len(bg_map)} 座標点)")
    print(f"対象ベースラベル: {target_labels}  (計 {sum(len(folders[l]) for l in target_labels if l in folders)} フォルダ)")

    total_ok = 0
    total_skip = 0
    total_err = 0

    for base_label in target_labels:
        if base_label not in folders:
            print(f"  [WARN] '{base_label}_x*_y*' フォルダが見つかりません。スキップ。")
            continue

        target_map = folders[base_label]
        # Pos番号を base_label から取得（例: "Pos3" → 3）
        m = re.match(r"Pos(\d+)$", base_label)
        pos_number = int(m.group(1)) if m else 0
        crop = get_crop(pos_number)
        print(f"\n{'='*60}")
        print(f"  {base_label}  ({len(target_map)} フォルダ)  crop={crop}")
        print(f"{'='*60}")

        # タスクリスト構築（スキップ・BG欠損チェック）
        tasks = []
        for (xi, yi) in sorted(target_map.keys()):
            tgt_dir = target_map[(xi, yi)]
            out_dir = tgt_dir / "output_phase"
            if SKIP_IF_EXISTS and out_dir.exists() and any(out_dir.glob("*.tif")):
                total_skip += 1
                continue
            if (xi, yi) not in bg_map:
                print(f"  [WARN] BG が見つかりません: {BG_BASE_LABEL}_x{xi:+d}_y{yi:+d}  → スキップ")
                total_err += 1
                continue
            bg_d = bg_map[(xi, yi)]
            tasks.append((xi, yi, str(tgt_dir), str(bg_d), crop, pos_number))

        if not tasks:
            continue

        n_workers_display = N_WORKERS if N_WORKERS is not None else os.cpu_count()
        print(f"  並列処理: {len(tasks)} ポイント / {n_workers_display} ワーカー")

        if N_WORKERS == 1:
            results = [_reconstruct_grid_point(args) for args in tqdm(tasks, desc=base_label)]
        else:
            results = []
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = {executor.submit(_reconstruct_grid_point, args): args for args in tasks}
                for fut in tqdm(as_completed(futures), total=len(tasks), desc=base_label):
                    results.append(fut.result())

        for xi, yi, folder_ok, err_msg in results:
            if folder_ok:
                total_ok += 1
            else:
                total_err += 1
                if err_msg:
                    print(f"  [ERR] ({xi:+d},{yi:+d}): {err_msg}")

    print(f"\n{'='*60}")
    print(f"完了")
    print(f"  成功:   {total_ok} フォルダ")
    print(f"  スキップ: {total_skip} フォルダ（SKIP_IF_EXISTS=True）")
    print(f"  エラー:  {total_err} フォルダ")


if __name__ == "__main__":
    main()

# %%
