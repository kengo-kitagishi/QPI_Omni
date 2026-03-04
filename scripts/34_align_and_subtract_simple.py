# %%
"""
シンプルなアライメント・引き算スクリプト
指定ディレクトリ内の.tifファイルをソート順に処理し、
N番目の画像（または直接パス指定）を基準にアライメント・引き算を行う。
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


def process_timelapse(timelapse_dir, reference_img, tif_files,
                      method='ecc', save_png=True, vmin=-0.1, vmax=1.7,
                      cmap='RdBu_r', png_dpi=150, png_sample_interval=5):
    """
    タイムラプス画像群に対してアライメント＋引き算を実行

    Parameters
    ----------
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

    for tif_path in tqdm(tif_files, desc="処理中"):
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
                ax.set_title(f'{base_name}\n平均: {np.mean(subtracted):.3f}, '
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
            'method': method,
            'num_processed': processed_count,
            'alignment_results': alignment_results
        }, f, indent=2, ensure_ascii=False)

    # シフト可視化
    if alignment_results:
        from shift_visualize import visualize_shifts
        visualize_shifts(
            json_path,
            subtracted_vmin=vmin,
            subtracted_vmax=vmax,
            subtracted_cmap=cmap,
        )

    # サマリー
    print(f"\n✅ 処理完了")
    print(f"   成功: {processed_count}枚")
    if skipped_count > 0:
        print(f"   スキップ: {skipped_count}枚")
    print(f"   aligned/: {aligned_dir}")
    print(f"   subtracted/: {subtracted_dir}")
    if save_png:
        print(f"   subtracted_colored/: {colored_dir} ({png_saved_count}枚)")

    if alignment_results:
        shifts_y = [r['shift_y'] for r in alignment_results]
        shifts_x = [r['shift_x'] for r in alignment_results]
        corrs = [r['correlation'] for r in alignment_results]
        print(f"\n   シフトY: 平均={np.mean(shifts_y):.2f}px, "
              f"範囲=[{np.min(shifts_y):.2f}, {np.max(shifts_y):.2f}]")
        print(f"   シフトX: 平均={np.mean(shifts_x):.2f}px, "
              f"範囲=[{np.min(shifts_x):.2f}, {np.max(shifts_x):.2f}]")
        print(f"   相関: 平均={np.mean(corrs):.4f}, "
              f"範囲=[{np.min(corrs):.4f}, {np.max(corrs):.4f}]")


def main():
    # ========================================
    # 設定パラメータ
    # ========================================

    # タイムラプス画像のディレクトリ（.tifが入ったフォルダ）
    TIMELAPSE_DIR = r"E:\Acuisition\kitagishi\260218\move_test_2\Pos1\crop"
    # 基準画像の指定（優先度: REFERENCE_IMAGE_PATH > REFERENCE_DIR+INDEX > TIMELAPSE_DIR+INDEX）
    # 方法1: 直接パスを指定
    #   例: r"E:\Acuisition\kitagishi\260216\move_test_1\Pos1\img_000000200_ph_000.tif"
    REFERENCE_IMAGE_PATH = None
    # 方法2: 基準画像のディレクトリ + N番目（1始まり）
    #   REFERENCE_DIR=None の場合は TIMELAPSE_DIR 内のN番目を使用
    REFERENCE_DIR = None
    REFERENCE_INDEX = 20

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
    print("アライメント・引き算処理（シンプル版）")
    print("=" * 80)

    # ディレクトリ確認
    if not os.path.exists(TIMELAPSE_DIR):
        print(f"\n❌ ディレクトリが見つかりません: {TIMELAPSE_DIR}")
        return

    # tifファイル一覧取得
    tif_files = get_tif_files(TIMELAPSE_DIR)
    if not tif_files:
        print(f"\n❌ .tifファイルが見つかりません: {TIMELAPSE_DIR}")
        return

    print(f"\nディレクトリ: {TIMELAPSE_DIR}")
    print(f"tifファイル数: {len(tif_files)}")

    # 基準画像の解決
    if REFERENCE_IMAGE_PATH is not None:
        if not os.path.exists(REFERENCE_IMAGE_PATH):
            print(f"\n❌ 基準画像が見つかりません: {REFERENCE_IMAGE_PATH}")
            return
        ref_path = REFERENCE_IMAGE_PATH
        print(f"基準画像（直接指定）: {ref_path}")
    else:
        # REFERENCE_DIR が指定されていればそこから、なければ TIMELAPSE_DIR から選ぶ
        if REFERENCE_DIR is not None:
            if not os.path.exists(REFERENCE_DIR):
                print(f"\n❌ 基準画像ディレクトリが見つかりません: {REFERENCE_DIR}")
                return
            ref_candidates = get_tif_files(REFERENCE_DIR)
            ref_source = REFERENCE_DIR
        else:
            ref_candidates = tif_files
            ref_source = TIMELAPSE_DIR

        if not ref_candidates:
            print(f"\n❌ .tifファイルが見つかりません: {ref_source}")
            return
        if REFERENCE_INDEX < 1 or REFERENCE_INDEX > len(ref_candidates):
            print(f"\n❌ REFERENCE_INDEX={REFERENCE_INDEX} が範囲外です（1~{len(ref_candidates)}）")
            return
        ref_path = ref_candidates[REFERENCE_INDEX - 1]
        print(f"基準画像（{ref_source} の {REFERENCE_INDEX}番目）: {os.path.basename(ref_path)}")

    print(f"アライメント方法: {ALIGNMENT_METHOD.upper()}")

    # 基準画像読み込み
    print(f"\n基準画像を読み込み中...")
    reference_img = load_tif_image(ref_path)
    print(f"  サイズ: {reference_img.shape}")

    # 処理実行
    process_timelapse(
        TIMELAPSE_DIR, reference_img, tif_files,
        method=ALIGNMENT_METHOD,
        save_png=SAVE_PNG,
        vmin=VMIN, vmax=VMAX, cmap=CMAP,
        png_dpi=PNG_DPI,
        png_sample_interval=PNG_SAMPLE_INTERVAL
    )


if __name__ == "__main__":
    main()

# %%
