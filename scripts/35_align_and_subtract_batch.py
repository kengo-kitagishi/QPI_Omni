# %%
"""
バッチ版アライメント・引き算スクリプト
Posフォルダ構造をイテレーションしつつ、ファイル取得はソート順（命名規則不問）。
基準画像はPos毎にREFERENCE_BASE内のN番目、または直接パス指定。
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
import json
import cv2
from tqdm import tqdm

plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_tif_image(path):
    """TIF画像を読み込んでfloatに変換"""
    img = io.imread(path)
    return img.astype(np.float64)


def to_uint8(img, vmin=-5.0, vmax=2.0):
    """固定範囲でuint8に変換（アライメント用）"""
    clipped = np.clip(img, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)
    return (normalized * 255).astype(np.uint8)


def get_tif_files(directory):
    """
    ディレクトリ内の.tif/.tiffファイルをソート順で取得

    Returns:
        list of str: フルパスのリスト（名前順ソート）
    """
    files = sorted([
        f for f in os.listdir(directory)
        if not f.startswith("._")
        and f.lower().endswith(('.tif', '.tiff'))
    ])
    return [os.path.join(directory, f) for f in files]


def get_pos_folders(base_dir, pos_start=None, pos_end=None):
    """
    ベースディレクトリ内のPosフォルダを列挙（Pos0除外）

    Returns:
        list of tuples: [(pos_name, pos_num), ...] ソート済み
    """
    pos_folders = []
    for item in sorted(os.listdir(base_dir)):
        if not (os.path.isdir(os.path.join(base_dir, item))
                and item.startswith("Pos") and item != "Pos0"):
            continue
        try:
            pos_num = int(item.replace("Pos", ""))
        except ValueError:
            continue
        if pos_start is not None and pos_num < pos_start:
            continue
        if pos_end is not None and pos_num > pos_end:
            continue
        pos_folders.append((item, pos_num))
    return pos_folders


def process_timelapse(pos_name, timelapse_dir, reference_img, tif_files,
                      method='ecc', save_png=True, vmin=-0.1, vmax=1.7,
                      cmap='RdBu_r', png_dpi=150, png_sample_interval=5):
    """
    タイムラプス画像群に対してアライメント＋引き算を実行

    Parameters
    ----------
    pos_name : str
        Posフォルダ名（表示用）
    timelapse_dir : str
        タイムラプス画像のディレクトリ
    reference_img : np.ndarray
        基準画像（float64）
    tif_files : list of str
        処理対象のtifファイルパスリスト
    method : str
        'ecc' or 'phase_correlation'
    save_png : bool
        カラーマップPNGを保存するか
    vmin, vmax : float
        カラーマップの範囲
    cmap : str
        カラーマップ名
    png_dpi : int
        PNG解像度
    png_sample_interval : int
        N枚ごとにPNG保存

    Returns
    -------
    dict or None: 処理結果
    """
    # 出力ディレクトリ作成
    aligned_dir = os.path.join(timelapse_dir, "aligned")
    subtracted_dir = os.path.join(timelapse_dir, "subtracted")
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(subtracted_dir, exist_ok=True)

    if save_png:
        colored_dir = os.path.join(timelapse_dir, "subtracted_colored")
        os.makedirs(colored_dir, exist_ok=True)

    alignment_results = []
    processed_count = 0
    skipped_count = 0
    png_saved_count = 0

    for tif_path in tqdm(tif_files, desc=f"  {pos_name}"):
        filename = os.path.basename(tif_path)
        try:
            timelapse_img = load_tif_image(tif_path)

            if timelapse_img.shape != reference_img.shape:
                print(f"\n  ⚠️ サイズ不一致、スキップ: {filename}")
                skipped_count += 1
                continue

            # アライメント計算
            if method == 'ecc':
                ref_u8 = to_uint8(reference_img)
                tl_u8 = to_uint8(timelapse_img)

                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-8)

                try:
                    correlation, warp_matrix = cv2.findTransformECC(
                        ref_u8, tl_u8, warp_matrix,
                        cv2.MOTION_TRANSLATION, criteria
                    )
                    shift_x = warp_matrix[0, 2]
                    shift_y = warp_matrix[1, 2]
                except Exception as e:
                    print(f"\n  ⚠️ アライメント失敗: {filename} - {e}")
                    skipped_count += 1
                    continue

            elif method == 'phase_correlation':
                from skimage import registration
                try:
                    shift, error, _ = registration.phase_cross_correlation(
                        reference_img, timelapse_img, upsample_factor=10
                    )
                    shift_y, shift_x = shift[0], shift[1]
                    correlation = 1.0 - error
                    warp_matrix = np.array([
                        [1.0, 0.0, shift_x],
                        [0.0, 1.0, shift_y]
                    ], dtype=np.float32)
                except Exception as e:
                    print(f"\n  ⚠️ アライメント失敗: {filename} - {e}")
                    skipped_count += 1
                    continue

            # アライメント適用
            h, w = timelapse_img.shape
            aligned_img = cv2.warpAffine(
                timelapse_img.astype(np.float32),
                warp_matrix, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            ).astype(np.float64)

            # 保存
            base_name = os.path.splitext(filename)[0]

            aligned_path = os.path.join(aligned_dir, filename)
            io.imsave(aligned_path, aligned_img.astype(np.float32))

            subtracted = aligned_img - reference_img
            subtracted_path = os.path.join(subtracted_dir, f"{base_name}_subtracted.tif")
            io.imsave(subtracted_path, subtracted.astype(np.float32))

            alignment_results.append({
                'filename': filename,
                'warp_matrix': warp_matrix.tolist(),
                'shift_x': float(shift_x),
                'shift_y': float(shift_y),
                'correlation': float(correlation)
            })
            processed_count += 1

            # PNG保存
            if save_png and (processed_count % png_sample_interval == 0):
                colored_path = os.path.join(colored_dir, f"{base_name}_subtracted.png")
                fig, ax = plt.subplots(figsize=(10, 8))
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                im = ax.imshow(subtracted, cmap=cmap, norm=norm)
                ax.axis('off')
                ax.set_title(f'{pos_name} - {base_name}\n'
                             f'平均: {np.mean(subtracted):.3f}, '
                             f'標準偏差: {np.std(subtracted):.3f}')
                plt.colorbar(im, ax=ax, fraction=0.046, label='差分 (rad)')
                plt.tight_layout()
                plt.savefig(colored_path, dpi=png_dpi, bbox_inches='tight')
                plt.close()
                png_saved_count += 1

        except Exception as e:
            print(f"\n  ❌ エラー: {filename} - {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue

    # JSON保存
    json_path = os.path.join(timelapse_dir, "alignment_transforms.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'pos_name': pos_name,
            'method': method,
            'num_processed': processed_count,
            'alignment_results': alignment_results
        }, f, indent=2, ensure_ascii=False)

    # シフト可視化
    if alignment_results:
        from shift_visualize import visualize_shifts
        visualize_shifts(json_path)

    # サマリー
    print(f"\n  ✅ {pos_name} 完了: {processed_count}枚", end="")
    if skipped_count > 0:
        print(f" (スキップ: {skipped_count}枚)", end="")
    print()

    if alignment_results:
        shifts_y = [r['shift_y'] for r in alignment_results]
        shifts_x = [r['shift_x'] for r in alignment_results]
        corrs = [r['correlation'] for r in alignment_results]
        print(f"     シフトY: [{np.min(shifts_y):.2f}, {np.max(shifts_y):.2f}]px, "
              f"シフトX: [{np.min(shifts_x):.2f}, {np.max(shifts_x):.2f}]px, "
              f"相関: {np.mean(corrs):.4f}")

    return {
        'pos_name': pos_name,
        'processed_count': processed_count,
        'skipped_count': skipped_count,
        'png_saved_count': png_saved_count,
        'alignment_results': alignment_results
    }


def main():
    # ========================================
    # 設定パラメータ
    # ========================================

    # ベースディレクトリ（Posフォルダが入っている場所）
    TIMELAPSE_BASE = r"E:\Acuisition\kitagishi\260216\move_test_1"
    REFERENCE_BASE = r"E:\Acuisition\kitagishi\260216\move_test_1"

    # Pos内のサブディレクトリ（tifが入っている場所）
    # 例: "output_phase", "large_crop", "" (Pos直下)
    TIMELAPSE_SUBDIR = "output_phase"
    REFERENCE_SUBDIR = "output_phase"

    # 処理範囲
    POS_START = None  # 開始Pos番号（None=全て）
    POS_END = None    # 終了Pos番号（None=全て）

    # 基準画像
    # 方法1: 直接パス指定（全Posで同じ画像を使用）
    REFERENCE_IMAGE_PATH = None
    # 方法2: 各PosのREFERENCE_SUBDIR内のN番目（1始まり）
    REFERENCE_INDEX = 1

    # 処理設定
    ALIGNMENT_METHOD = 'ecc'  # 'ecc' or 'phase_correlation'
    SAVE_PNG = True
    PNG_DPI = 150
    PNG_SAMPLE_INTERVAL = 1
    VMIN = -0.1
    VMAX = 1.7
    CMAP = 'RdBu_r'

    # ========================================

    print("=" * 80)
    print("バッチ版アライメント・引き算処理")
    print("=" * 80)

    # ディレクトリ確認
    if not os.path.exists(TIMELAPSE_BASE):
        print(f"\n❌ タイムラプスディレクトリが見つかりません: {TIMELAPSE_BASE}")
        return

    print(f"\nタイムラプス: {TIMELAPSE_BASE} / {{Pos}} / {TIMELAPSE_SUBDIR}")
    if REFERENCE_IMAGE_PATH is not None:
        print(f"基準画像（直接指定）: {REFERENCE_IMAGE_PATH}")
        if not os.path.exists(REFERENCE_IMAGE_PATH):
            print(f"\n❌ 基準画像が見つかりません: {REFERENCE_IMAGE_PATH}")
            return
    else:
        print(f"基準画像: {REFERENCE_BASE} / {{Pos}} / {REFERENCE_SUBDIR} の {REFERENCE_INDEX}番目")

    # Posフォルダ列挙
    pos_folders = get_pos_folders(TIMELAPSE_BASE, POS_START, POS_END)
    if not pos_folders:
        print(f"\n❌ Posフォルダが見つかりません")
        return

    print(f"対象Pos: {len(pos_folders)}個 ({pos_folders[0][0]}~{pos_folders[-1][0]})")
    print(f"アライメント方法: {ALIGNMENT_METHOD.upper()}")

    # 直接パス指定の場合、基準画像を先に読み込む
    shared_ref_img = None
    if REFERENCE_IMAGE_PATH is not None:
        shared_ref_img = load_tif_image(REFERENCE_IMAGE_PATH)
        print(f"基準画像サイズ: {shared_ref_img.shape}")

    # 各Posを処理
    results = []
    for pos_name, pos_num in pos_folders:
        print(f"\n{'='*60}")
        print(f"{pos_name}")
        print(f"{'='*60}")

        # タイムラプス画像ディレクトリ
        if TIMELAPSE_SUBDIR:
            tl_dir = os.path.join(TIMELAPSE_BASE, pos_name, TIMELAPSE_SUBDIR)
        else:
            tl_dir = os.path.join(TIMELAPSE_BASE, pos_name)

        if not os.path.exists(tl_dir):
            print(f"  ⚠️ ディレクトリなし、スキップ: {tl_dir}")
            continue

        tif_files = get_tif_files(tl_dir)
        if not tif_files:
            print(f"  ⚠️ tifファイルなし、スキップ: {tl_dir}")
            continue

        print(f"  tifファイル数: {len(tif_files)}")

        # 基準画像の解決
        if shared_ref_img is not None:
            ref_img = shared_ref_img
        else:
            if REFERENCE_SUBDIR:
                ref_dir = os.path.join(REFERENCE_BASE, pos_name, REFERENCE_SUBDIR)
            else:
                ref_dir = os.path.join(REFERENCE_BASE, pos_name)

            if not os.path.exists(ref_dir):
                print(f"  ⚠️ 基準画像ディレクトリなし、スキップ: {ref_dir}")
                continue

            ref_candidates = get_tif_files(ref_dir)
            if not ref_candidates:
                print(f"  ⚠️ 基準画像のtifなし、スキップ: {ref_dir}")
                continue
            if REFERENCE_INDEX < 1 or REFERENCE_INDEX > len(ref_candidates):
                print(f"  ⚠️ REFERENCE_INDEX={REFERENCE_INDEX} が範囲外（1~{len(ref_candidates)}）、スキップ")
                continue

            ref_path = ref_candidates[REFERENCE_INDEX - 1]
            print(f"  基準画像（{REFERENCE_INDEX}番目）: {os.path.basename(ref_path)}")
            ref_img = load_tif_image(ref_path)

        # 処理実行
        result = process_timelapse(
            pos_name, tl_dir, ref_img, tif_files,
            method=ALIGNMENT_METHOD,
            save_png=SAVE_PNG,
            vmin=VMIN, vmax=VMAX, cmap=CMAP,
            png_dpi=PNG_DPI,
            png_sample_interval=PNG_SAMPLE_INTERVAL
        )
        if result is not None:
            results.append(result)

    # 全体サマリー
    print(f"\n{'='*80}")
    print(f"全処理完了")
    print(f"{'='*80}")
    print(f"処理Pos数: {len(results)}")

    if results:
        total_processed = sum(r['processed_count'] for r in results)
        total_skipped = sum(r['skipped_count'] for r in results)
        print(f"処理画像総数: {total_processed}枚")
        if total_skipped > 0:
            print(f"スキップ: {total_skipped}枚")


if __name__ == "__main__":
    main()

# %%
