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
import argparse
import re
import sys
import os
import numpy as np
from pathlib import Path

# Set CUDA_PATH before any cupy import (cupy reads it at import time)
if not os.environ.get("CUDA_PATH"):
    _nvrtc = Path(sys.prefix) / "Lib/site-packages/nvidia/cuda_nvrtc"
    if _nvrtc.exists():
        os.environ["CUDA_PATH"] = str(_nvrtc)

import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ============================================================
# 設定パラメータ
# ============================================================
# pipeline_full.py の GRID_DIR と揃えること
GRID_DIR = r"C:\grid_0pergluc_60ms_1"

# BG として使うベースラベル（pipeline_full: GRID_BG_BASE_LABEL）
BG_BASE_LABEL = "Pos0"

# 再構成対象のベースラベル（None で Pos0 以外の全 PosX_x*_y* を処理）
# pipeline_full: GRID_TARGET_BASE_LABELS と同じ意味
TARGET_BASE_LABELS = None

# 再構成対象の (xi, yi) 座標を絞る（None = すべて処理）
# 例: TARGET_COORDS = [(0, 0)]  → 中心点のみ
TARGET_COORDS = None

# QPI 光学パラメータ
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

# ---- Pos番号によるクロップ切り替え（pipeline_full.py と同一）----
# pos_number < POS_SPLIT → 右側 (400:2448)  センサー幅2448
# pos_number >= POS_SPLIT → 左側 (0:2048)
# ※ BG（Pos0）はターゲットの pos_number で決まる crop を使う（常に右ではない）
POS_SPLIT    = 33
CROP_BEFORE  = (0, 2048, 400, 2448)
CROP_AFTER   = (0, 2048,   0, 2048)

# 平均0調整の領域（pipeline GRID_MEAN_REGION と同じ。None で無効）
MEAN_REGION = None

# 再構成済み（output_phase に *_phase.tif が既存）の場合スキップするか（pipeline: GRID_SKIP_IF_EXISTS）
SKIP_IF_EXISTS = False

# PNG カラーマップも保存するか
SAVE_PNG = False
PNG_DPI  = 150
PNG_VMIN = -2.0
PNG_VMAX =  2.0
# 並列処理ワーカー数（pipeline_full: N_WORKERS_GRID と同じ）
N_WORKERS = 16
# ============================================================

# QPIインポート
try:
    from qpi import QPIParameters, get_field, set_backend, _HAS_CUPY
except ImportError:
    print("ERROR: qpi モジュールが見つかりません。QPI_Omni/scripts が PYTHONPATH にあるか確認してください。")
    sys.exit(1)

# GPU acceleration
_USE_GPU = False
if _HAS_CUPY:
    try:
        import cupy as cp
        cp.array([1.0]) * 2  # smoke test
        set_backend("cupy")
        _USE_GPU = True
        print(f"GPU mode: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    except Exception as e:
        print(f"GPU init failed, using CPU: {e}")
        set_backend("numpy")


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
    if hasattr(field, "get"):  # CuPy array → numpy
        angle = np.angle(field.get())
    else:
        angle = np.angle(field)
    phase = unwrap_phase(angle)
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


def reconstruct_from_holo(holo_path, crop):
    """One-shot: create QPIParameters and reconstruct unwrapped phase in one call.

    Convenience wrapper around make_qpi_params + reconstruct_image.
    """
    qpi = make_qpi_params(holo_path, crop)
    return reconstruct_image(holo_path, qpi, crop)


def _reconstruct_grid_point(args):
    """ProcessPoolExecutor ワーカー: 1グリッドポイントの全 z スライスを再構成して保存。

    Saves two outputs per z slice:
      - output_phase/     : BG-subtracted + region mean subtracted (for ECC)
      - output_phase_raw/ : raw phase, no BG subtraction (for grid_subtract / correct_0pergluc)
    """
    xi, yi, target_dir_str, bg_dir_str, crop, pos_number = args
    target_dir  = Path(target_dir_str)
    bg_dir      = Path(bg_dir_str)
    out_dir     = target_dir / "output_phase"
    raw_out_dir = target_dir / "output_phase_raw"

    z_files_target = {get_z_index(p): p for p in get_z_files(target_dir)}
    z_files_bg     = {get_z_index(p): p for p in get_z_files(bg_dir)}

    if not z_files_target:
        return xi, yi, False, "z画像なし"

    out_dir.mkdir(exist_ok=True)
    raw_out_dir.mkdir(exist_ok=True)
    sample_path = next(iter(z_files_target.values()))
    try:
        qpi_params = make_qpi_params(sample_path, crop)
    except Exception as e:
        return xi, yi, False, f"QPIParams: {e}"

    folder_ok = True
    for z_idx, tgt_path in sorted(z_files_target.items()):
        out_path     = out_dir     / (tgt_path.stem + "_phase.tif")
        raw_out_path = raw_out_dir / (tgt_path.stem + "_phase.tif")
        if SKIP_IF_EXISTS and out_path.exists() and raw_out_path.exists():
            continue
        if z_idx not in z_files_bg:
            folder_ok = False
            continue
        try:
            phase_target = reconstruct_image(tgt_path, qpi_params, crop)

            # Save raw phase (no BG subtraction) for grid_subtract / correct_0pergluc
            if not raw_out_path.exists():
                tifffile.imwrite(str(raw_out_path), phase_target.astype(np.float32))

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


def _parse_cli():
    p = argparse.ArgumentParser(
        description="グリッド各点の生ホロを BG 引き算付きで output_phase に再構成する。",
    )
    p.add_argument(
        "--grid-dir",
        type=str,
        default=None,
        help=f"GRID_DIR（既定: {GRID_DIR}）",
    )
    p.add_argument(
        "--bg-label",
        type=str,
        default=None,
        help=f"BG ベースラベル（既定: {BG_BASE_LABEL}）",
    )
    p.add_argument(
        "--targets",
        nargs="*",
        default=None,
        metavar="LABEL",
        help='再構成するベースラベル（例: Pos6）。未指定なら定数 TARGET_BASE_LABELS / 全 Pos（BG 以外）',
    )
    return p.parse_args()


def main():
    args = _parse_cli()
    grid_dir = Path(args.grid_dir or GRID_DIR)
    bg_label = args.bg_label or BG_BASE_LABEL
    if args.targets is not None:
        target_labels_override = args.targets if args.targets else None
    else:
        target_labels_override = TARGET_BASE_LABELS

    if not grid_dir.exists():
        print(f"ERROR: GRID_DIR が見つかりません: {grid_dir}")
        sys.exit(1)

    t_start = time.perf_counter()

    # フォルダスキャン
    folders = scan_grid_folders(grid_dir)
    if bg_label not in folders:
        print(f"ERROR: BG フォルダ '{bg_label}_x*_y*' が見つかりません: {grid_dir}")
        sys.exit(1)

    bg_map = folders[bg_label]  # {(xi, yi): Path}

    # 対象ベースラベルを決定
    if target_labels_override is not None:
        target_labels = target_labels_override
    else:
        target_labels = [k for k in sorted(folders.keys()) if k != bg_label]

    print(f"BG ベースラベル: {bg_label}  ({len(bg_map)} 座標点)")
    print(f"対象ベースラベル: {target_labels}  (計 {sum(len(folders[l]) for l in target_labels if l in folders)} フォルダ)")

    total_ok = 0
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
            if TARGET_COORDS is not None and (xi, yi) not in TARGET_COORDS:
                continue
            tgt_dir = target_map[(xi, yi)]
            # スキップは _reconstruct_grid_point 内で z ごとに行う（途中まで終わったフォルダを再開できる）
            if (xi, yi) not in bg_map:
                print(f"  [WARN] BG が見つかりません: {bg_label}_x{xi:+d}_y{yi:+d}  → スキップ")
                total_err += 1
                continue
            bg_d = bg_map[(xi, yi)]
            tasks.append((xi, yi, str(tgt_dir), str(bg_d), crop, pos_number))

        if not tasks:
            continue

        if _USE_GPU:
            print(f"  GPU sequential: {len(tasks)} points")
            results = [_reconstruct_grid_point(t) for t in tqdm(tasks, desc=base_label)]
        else:
            n_workers_display = N_WORKERS if N_WORKERS is not None else os.cpu_count()
            print(f"  CPU parallel: {len(tasks)} points / {n_workers_display} workers")
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

    elapsed = time.perf_counter() - t_start
    mode = "GPU" if _USE_GPU else f"CPU ({N_WORKERS} workers)"
    print(f"\n{'='*60}")
    print(f"Done  ({mode}, {elapsed:.1f}s)")
    print(f"  OK:    {total_ok} folders")
    print(f"  Error: {total_err} folders")
    print(f"  (SKIP_IF_EXISTS={SKIP_IF_EXISTS})")


if __name__ == "__main__":
    main()

# %%
