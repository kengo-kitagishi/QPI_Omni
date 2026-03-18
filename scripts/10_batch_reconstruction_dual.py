# %%
"""
新フォルダ構造対応の位相再構成スクリプト（Dual出力版）
背景引き算済み画像と生の位相画像の両方を保存
wo_0_EMM_1とwo_2_EMM_1の両方を処理し、各フォルダ内のPos0を背景として使用
"""
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

# 並列処理ワーカー数（None = cpu_count(), 1 = 逐次実行でデバッグ向き）
N_WORKERS = None

# ベースディレクトリ設定
BASE_DIRS = [
    r"E:\Acuisition\kitagishi\260301\movetest_9"
]

# Pos0 も output_phase / output_phase_raw を生成するか（True: Pos0 を自分自身をBGとして処理）
INCLUDE_POS0 = True

def _worker_batch_frame(args):
    """ProcessPoolExecutor ワーカー: 1フレームを再構成して保存（dual出力版）。"""
    filepath, bg_filepath, output_dir, output_dir_raw, output_dir_colormap, base_name, pos_number, pos_split = args

    if not os.path.exists(bg_filepath):
        return base_name, "no_bg", None

    try:
        bg_img = np.array(Image.open(bg_filepath))
        if pos_number < pos_split:
            bg_img = bg_img[0:2048, 400:2448]
        else:
            bg_img = bg_img[0:2048, 0:2048]

        params = QPIParameters(
            wavelength=WAVELENGTH, NA=NA,
            img_shape=bg_img.shape, pixelsize=PIXELSIZE,
            offaxis_center=OFFAXIS_CENTER
        )
        field_bg = get_field(bg_img, params)
        angle_bg = unwrap_phase(np.angle(field_bg))

        img = np.array(Image.open(filepath))
        if pos_number < pos_split:
            img = img[0:2048, 400:2448]
        else:
            img = img[0:2048, 0:2048]

        field = get_field(img, params)
        angle = unwrap_phase(np.angle(field))
        angle_nobg = angle - angle_bg

        h, w = angle_nobg.shape
        if pos_number < pos_split:
            center_region = angle_nobg[1:h-1, 1:w//2]
        else:
            center_region = angle_nobg[1:h-1, w//2:w-1]
        if center_region.size > 0:
            angle_nobg -= np.mean(center_region)

        outpath = os.path.join(output_dir, f"{base_name}_phase.tif")
        tifffile.imwrite(outpath, angle_nobg.astype(np.float32))

        outpath_raw = os.path.join(output_dir_raw, f"{base_name}_phase.tif")
        tifffile.imwrite(outpath_raw, angle.astype(np.float32))

        if True:  # PNG保存
            plt.figure(figsize=(6, 6))
            plt.imshow(angle_nobg, cmap='viridis', vmin=-4, vmax=2)
            plt.colorbar(label='Phase (rad)')
            plt.title(f"Phase: {base_name}")
            plt.axis('off')
            plt.tight_layout()
            png_outpath = os.path.join(output_dir_colormap, f"{base_name}_colormap.png")
            plt.savefig(png_outpath, dpi=150)
            plt.close()

        return base_name, "ok", None
    except Exception as e:
        return base_name, "err", str(e)


def process_folder(base_dir, pos_start=None, pos_end=None):
    """
    1つのベースディレクトリ（wo_0_EMM_1またはwo_2_EMM_1）を処理

    Parameters:
    -----------
    base_dir : str
        処理するベースディレクトリのパス
    pos_start : int, optional
        処理開始Pos番号（None=全て）
    pos_end : int, optional
        処理終了Pos番号（None=全て）
    """
    print(f"\n{'='*80}")
    print(f"処理中: {os.path.basename(base_dir)}")
    print(f"{'='*80}")

    # Pos0（背景）ディレクトリ
    bg_dir = os.path.join(base_dir, "Pos0")

    if not os.path.exists(bg_dir):
        print(f"❌ エラー: 背景ディレクトリが見つかりません: {bg_dir}")
        return

    # Posフォルダを検索（INCLUDE_POS0 時は Pos0 も、通常は Pos1 以降）
    pos_folders = []
    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        if not (os.path.isdir(item_path) and item.startswith("Pos")):
            continue
        if item == "Pos0":
            if INCLUDE_POS0:
                pos_folders.append(item)
            continue
        # Pos番号を取得
        try:
            pos_num = int(item.replace("Pos", ""))
        except:
            continue
        # 範囲チェック
        if pos_start is not None and pos_num < pos_start:
            continue
        if pos_end is not None and pos_num > pos_end:
            continue
        pos_folders.append(item)

    if len(pos_folders) == 0:
        print(f"⚠️ 警告: Pos1以降のフォルダが見つかりません")
        return

    print(f"検出されたPosフォルダ: {len(pos_folders)}個")
    if len(pos_folders) > 0:
        print(f"処理範囲: {pos_folders[0]} ~ {pos_folders[-1]}")
        if pos_start is not None or pos_end is not None:
            print(f"  （指定範囲: Pos{pos_start if pos_start else '1'} ~ Pos{pos_end if pos_end else '最後'}）")

    # 各Posフォルダを処理
    # pos_split: 前半・後半を分けるPos番号（Pos31未満が前半、Pos31以降が後半）
    pos_split = 3
    for pos_name in pos_folders:
        process_pos_folder(base_dir, bg_dir, pos_name, pos_split=pos_split)

    print(f"\n✅ {os.path.basename(base_dir)} の処理が完了しました")


def process_pos_folder(base_dir, bg_dir, pos_name, pos_split=44):
    """
    1つのPosフォルダを処理

    Parameters:
    -----------
    base_dir : str
        ベースディレクトリのパス
    bg_dir : str
        背景画像ディレクトリ（Pos0）のパス
    pos_name : str
        処理するPosフォルダ名（例: "Pos1"）
    pos_split : int
        前半・後半を分けるPos番号（この値未満が前半、以上が後半）
    """
    target_dir = os.path.join(base_dir, pos_name)

    # Pos番号を取得
    try:
        pos_number = int(pos_name.replace("Pos", ""))
    except:
        pos_number = 0

    if not os.path.exists(target_dir):
        print(f"⚠️ スキップ: {target_dir} が存在しません")
        return

    # 出力ディレクトリ作成
    output_dir = os.path.join(target_dir, "output_phase")
    output_dir_raw = os.path.join(target_dir, "output_phase_raw")  # 追加: 生の位相画像用
    output_dir_colormap = os.path.join(target_dir, "output_colormap")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_raw, exist_ok=True)  # 追加
    os.makedirs(output_dir_colormap, exist_ok=True)

    print(f"\n>> 処理中: {pos_name}")

    # TIFファイルを取得
    tif_files = []
    for filename in os.listdir(target_dir):
        if filename.startswith("._"):
            continue  # macOSの不要な隠しファイルをスキップ
        if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
            if "output" not in filename:  # 出力ファイルを除外
                tif_files.append(filename)

    tif_files = sorted(tif_files)

    if len(tif_files) == 0:
        print(f"  ⚠️ TIF画像が見つかりません")
        return

    print(f"  TIF画像: {len(tif_files)}枚")

    # タスクリスト構築
    tasks = []
    for filename in tif_files:
        filepath = os.path.join(target_dir, filename)
        bg_filepath = os.path.join(bg_dir, filename)
        base_name = os.path.splitext(filename)[0]
        tasks.append((filepath, bg_filepath, output_dir, output_dir_raw,
                      output_dir_colormap, base_name, pos_number, pos_split))

    n_ok = n_err = 0
    if N_WORKERS == 1:
        raw_results = [_worker_batch_frame(args) for args in tqdm(tasks, desc=f"  {pos_name}")]
    else:
        raw_results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(_worker_batch_frame, args): args for args in tasks}
            for fut in tqdm(as_completed(futures), total=len(tasks), desc=f"  {pos_name}"):
                raw_results.append(fut.result())

    for base_name, status, err_msg in raw_results:
        if status == "ok":
            n_ok += 1
        elif status == "err":
            n_err += 1
            print(f"\n  ❌ エラー: {base_name} - {err_msg}")
        # no_bg はサイレントスキップ

    print(f"  {pos_name}: 完了={n_ok}, エラー={n_err}")


def main():
    """メイン処理"""
    # ========================================
    # 処理範囲設定（必要に応じて変更）
    # ========================================
    POS_START = None  # 開始Pos番号（None=最初から）例: 1
    POS_END = None    # 終了Pos番号（None=最後まで）例: 10
    # ========================================

    print("="*80)
    print("新フォルダ構造対応の位相再構成処理（Dual出力版）")
    print("output_phase: 背景引き算済み / output_phase_raw: 生の位相画像")
    print("="*80)
    print(f"\n処理対象:")
    for base_dir in BASE_DIRS:
        print(f"  - {base_dir}")

    if POS_START is not None or POS_END is not None:
        print(f"\n処理Pos範囲: ", end="")
        if POS_START is not None and POS_END is not None:
            print(f"Pos{POS_START} ~ Pos{POS_END}")
        elif POS_START is not None:
            print(f"Pos{POS_START} ~ 最後")
        else:
            print(f"最初 ~ Pos{POS_END}")
    else:
        print(f"\n処理Pos範囲: 全て")

    # 各ベースディレクトリを処理
    for base_dir in BASE_DIRS:
        if os.path.exists(base_dir):
            process_folder(base_dir, pos_start=POS_START, pos_end=POS_END)
        else:
            print(f"\n⚠️ 警告: ディレクトリが見つかりません: {base_dir}")

    print("\n" + "="*80)
    print("全ての処理が完了しました！")
    print("="*80)


if __name__ == "__main__":
    main()



# %%

