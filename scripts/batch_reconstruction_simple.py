"""
batch_reconstruction_simple.py
-------------------------------
Pos0 を BG として Pos1... を一括再構成するシンプルスクリプト。
（grid フォーマット不要: Pos0/Pos1/Pos2/... の flat 構造に対応）

使い方:
    DATA_DIR / BG_POS / TARGET_POS_LIST を設定して実行。
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
DATA_DIR = r"D:\AquisitionData\Kitagishi\260321\focus_test_2"

# BG Pos ラベル
BG_POS = "Pos0"

# 再構成対象（None で BG_POS 以外の全 PosX を自動検出）
TARGET_POS_LIST = None   # 例: ["Pos1", "Pos2", "Pos3"]

# クロップ (row_start, row_end, col_start, col_end)
CROP = (0, 2048, 400, 2448)   # 右チャンネル

# QPI 光学パラメータ
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

# 再構成済みスキップ
SKIP_IF_EXISTS = True

# 平均0調整領域（None で無効）
MEAN_REGION = (1, 50, 1, 50)

# 並列ワーカー数（None = CPU数, 1 = 逐次）
N_WORKERS = None
# ============================================================

try:
    from qpi import QPIParameters, get_field
except ImportError:
    print("ERROR: qpi モジュールが見つかりません。")
    sys.exit(1)


def get_z_files(pos_dir: Path):
    return sorted(pos_dir.glob("img_*_ph_*.tif"))


def get_z_index(path: Path):
    m = re.search(r"_ph_(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def make_qpi_params(sample_path: Path, crop):
    img = np.array(Image.open(str(sample_path)))
    rs, re_, cs, ce = crop
    cropped = img[rs:re_, cs:ce]
    return QPIParameters(
        wavelength=WAVELENGTH,
        NA=NA,
        img_shape=cropped.shape,
        pixelsize=PIXELSIZE,
        offaxis_center=OFFAXIS_CENTER,
    )


def reconstruct_image(img_path: Path, qpi_params, crop):
    img = np.array(Image.open(str(img_path)))
    rs, re_, cs, ce = crop
    img = img[rs:re_, cs:ce]
    field = get_field(img, qpi_params)
    phase = unwrap_phase(np.angle(field))
    return phase


def _reconstruct_pos(args):
    pos_label, target_dir_str, bg_dir_str, crop = args
    target_dir = Path(target_dir_str)
    bg_dir     = Path(bg_dir_str)
    out_dir    = target_dir / "output_phase"

    z_files_target = {get_z_index(p): p for p in get_z_files(target_dir)}
    z_files_bg     = {get_z_index(p): p for p in get_z_files(bg_dir)}

    if not z_files_target:
        return pos_label, False, "z画像なし"

    out_dir.mkdir(exist_ok=True)
    sample_path = next(iter(z_files_target.values()))
    try:
        qpi_params = make_qpi_params(sample_path, crop)
    except Exception as e:
        return pos_label, False, f"QPIParams: {e}"

    ok = True
    for z_idx, tgt_path in sorted(z_files_target.items()):
        out_path = out_dir / (tgt_path.stem + "_phase.tif")
        if SKIP_IF_EXISTS and out_path.exists():
            continue
        if z_idx not in z_files_bg:
            ok = False
            continue
        try:
            phase_target = reconstruct_image(tgt_path, qpi_params, crop)
            phase_bg     = reconstruct_image(z_files_bg[z_idx], qpi_params, crop)
            phase_diff   = phase_target - phase_bg
            if MEAN_REGION is not None:
                rs, re_, cs, ce = MEAN_REGION
                phase_diff -= np.mean(phase_diff[rs:re_, cs:ce])
            tifffile.imwrite(str(out_path), phase_diff.astype(np.float32))
        except Exception as e:
            ok = False
    return pos_label, ok, None


def main():
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        print(f"ERROR: DATA_DIR が見つかりません: {data_dir}")
        sys.exit(1)

    bg_dir = data_dir / BG_POS
    if not bg_dir.exists():
        print(f"ERROR: BG フォルダが見つかりません: {bg_dir}")
        sys.exit(1)

    # 対象 Pos を決定
    if TARGET_POS_LIST is not None:
        targets = TARGET_POS_LIST
    else:
        pattern = re.compile(r"^Pos(\d+)$")
        targets = sorted(
            [d.name for d in data_dir.iterdir() if d.is_dir() and pattern.match(d.name) and d.name != BG_POS],
            key=lambda x: int(re.search(r"\d+", x).group())
        )

    print(f"DATA_DIR : {data_dir}")
    print(f"BG       : {BG_POS}")
    print(f"対象     : {targets}")
    print(f"CROP     : {CROP}")
    print(f"OFFAXIS  : {OFFAXIS_CENTER}")

    tasks = []
    for pos in targets:
        tgt_dir = data_dir / pos
        if not tgt_dir.exists():
            print(f"  [WARN] {pos} が見つかりません。スキップ。")
            continue
        out_dir = tgt_dir / "output_phase"
        if SKIP_IF_EXISTS and out_dir.exists() and any(out_dir.glob("*.tif")):
            print(f"  [SKIP] {pos} (output_phase 既存)")
            continue
        tasks.append((pos, str(tgt_dir), str(bg_dir), CROP))

    if not tasks:
        print("処理対象なし（全スキップ）。")
        return

    n_workers_display = N_WORKERS if N_WORKERS is not None else os.cpu_count()
    print(f"\n並列処理: {len(tasks)} Pos / {n_workers_display} ワーカー")

    if N_WORKERS == 1:
        results = [_reconstruct_pos(args) for args in tqdm(tasks)]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(_reconstruct_pos, args): args for args in tasks}
            for fut in tqdm(as_completed(futures), total=len(tasks)):
                results.append(fut.result())

    print()
    for pos_label, ok, err_msg in sorted(results, key=lambda x: x[0]):
        status = "OK" if ok else "ERR"
        msg = f"  [{status}] {pos_label}"
        if err_msg:
            msg += f": {err_msg}"
        print(msg)
    print("完了。")


if __name__ == "__main__":
    main()
