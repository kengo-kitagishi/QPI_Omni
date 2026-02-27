# %%
"""
タイムラプス画像のアライメント・引き算スクリプト
ph_1内のタイムラプス画像を、wocell_2_2の21番目の画像を基準にアライメント・引き算
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
from pathlib import Path
import json
import cv2
from tqdm import tqdm

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策


def load_tif_image(path):
    """TIF画像を読み込んでfloatに変換"""
    img = io.imread(path)
    return img.astype(np.float64)


def to_uint8(img, vmin=-5.0, vmax=2.0):
    """
    固定範囲でuint8に変換（アライメント用）
    """
    clipped = np.clip(img, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)
    return (normalized * 255).astype(np.uint8)


def get_image_number(filename):
    """
    ファイル名から画像番号（末尾3桁）を抽出
    例: img_000000000_Default_005_phase.tif -> 5
    """
    try:
        base = filename.replace("_phase.tif", "").replace("_phase.tiff", "")
        parts = base.split("_")
        return int(parts[-1])
    except:
        return None


def get_reference_image_path(phase_dir, reference_number):
    """
    output_phaseディレクトリ内から指定番号の画像を探す

    Parameters:
    -----------
    phase_dir : str
        output_phaseディレクトリのパス
    reference_number : int
        探す画像の番号（例: 21）

    Returns:
    --------
    str or None: 見つかった画像のフルパス、見つからない場合はNone
    """
    if not os.path.exists(phase_dir):
        return None

    # ディレクトリ内のファイルを取得
    files = sorted([f for f in os.listdir(phase_dir)
                   if not f.startswith("._") and (f.endswith("_phase.tif") or f.endswith("_phase.tiff"))])

    # 番号が一致するファイルを探す
    for filename in files:
        img_number = get_image_number(filename)
        if img_number == reference_number:
            return os.path.join(phase_dir, filename)

    # 見つからない場合、0-indexed/1-indexed両方を試す
    for filename in files:
        img_number = get_image_number(filename)
        if img_number == reference_number - 1:  # 0-indexedの可能性
            print(f"  ℹ️ 番号{reference_number}が見つからないため、{reference_number-1}（0-indexed）を使用します")
            return os.path.join(phase_dir, filename)

    return None


def find_timelapse_and_reference_pairs(timelapse_base, reference_base, reference_number=21,
                                       pos_start=None, pos_end=None):
    """
    タイムラプスデータと参照データの対応するPosフォルダを検索

    Parameters:
    -----------
    timelapse_base : str
        タイムラプスデータのベースディレクトリ（ph_1）
    reference_base : str
        参照データのベースディレクトリ（wocell_2_2）
    reference_number : int
        参照画像の番号（例: 21）
    pos_start : int, optional
        処理開始Pos番号（None=全て）
    pos_end : int, optional
        処理終了Pos番号（None=全て）

    Returns:
    --------
    list of tuples: [(pos_name, timelapse_phase_dir, reference_image_path), ...]
    """
    pairs = []

    # タイムラプスのPosフォルダを取得（Pos0を除く）
    timelapse_pos_folders = []
    for item in sorted(os.listdir(timelapse_base)):
        item_path = os.path.join(timelapse_base, item)
        if os.path.isdir(item_path) and item.startswith("Pos") and item != "Pos0":
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

            timelapse_pos_folders.append(item)

    # 各Posに対応する参照画像を探す
    for pos_name in timelapse_pos_folders:
        timelapse_pos_path = os.path.join(timelapse_base, pos_name)
        reference_pos_path = os.path.join(reference_base, pos_name)

        # output_phaseディレクトリの存在確認
        timelapse_phase_dir = os.path.join(timelapse_pos_path, "output_phase")
        reference_phase_dir = os.path.join(reference_pos_path, "output_phase")

        if not os.path.exists(timelapse_phase_dir):
            continue

        if not os.path.exists(reference_phase_dir):
            print(f"  ⚠️ 警告: {pos_name}の参照ディレクトリが見つかりません: {reference_phase_dir}")
            continue

        # 参照画像を探す
        reference_image_path = get_reference_image_path(reference_phase_dir, reference_number)

        if reference_image_path is None:
            print(f"  ⚠️ 警告: {pos_name}の参照画像（番号{reference_number}）が見つかりません")
            continue

        pairs.append((pos_name, timelapse_phase_dir, reference_image_path))

    return pairs


def process_pos_timelapse(pos_name, timelapse_phase_dir, reference_image_path,
                          method='ecc', save_png=True, vmin=-0.1, vmax=1.7, cmap='RdBu_r',
                          png_dpi=150, png_sample_interval=5):
    """
    1つのPosについて全タイムラプス画像を処理

    Parameters:
    -----------
    pos_name : str
        Posフォルダ名（例: "Pos1"）
    timelapse_phase_dir : str
        タイムラプス画像のディレクトリ（output_phase）
    reference_image_path : str
        参照画像のフルパス
    method : str
        アライメント方法（'ecc' または 'phase_correlation'）
    save_png : bool
        カラーマップPNG画像を保存するか
    vmin, vmax : float
        カラーマップの範囲
    cmap : str
        カラーマップの種類
    png_dpi : int
        PNG保存時の解像度
    png_sample_interval : int
        N枚ごとにPNG保存

    Returns:
    --------
    dict: 処理結果の情報
    """
    print(f"\n{'='*80}")
    print(f"処理中: {pos_name}")
    print(f"{'='*80}")

    # ===== ステップ1: 参照画像を読み込む =====
    print(f"\n[1/3] 参照画像読み込み中...")
    print(f"  参照画像: {os.path.basename(reference_image_path)}")

    try:
        reference_img = load_tif_image(reference_image_path)
        print(f"  画像サイズ: {reference_img.shape}")
    except Exception as e:
        print(f"  ❌ エラー: 参照画像の読み込みに失敗しました - {e}")
        return None

    # ===== ステップ2: タイムラプス画像リストを取得 =====
    print(f"\n[2/3] タイムラプス画像リスト取得中...")

    timelapse_files = sorted([f for f in os.listdir(timelapse_phase_dir)
                             if not f.startswith("._") and (f.endswith("_phase.tif") or f.endswith("_phase.tiff"))])

    if len(timelapse_files) == 0:
        print(f"  ❌ エラー: タイムラプス画像が見つかりません")
        return None

    print(f"  タイムラプス画像数: {len(timelapse_files)}枚")

    # ===== ステップ3: 出力ディレクトリ作成 =====
    aligned_dir = os.path.join(timelapse_phase_dir, "aligned")
    subtracted_dir = os.path.join(timelapse_phase_dir, "subtracted")
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(subtracted_dir, exist_ok=True)

    if save_png:
        colored_dir = os.path.join(timelapse_phase_dir, "subtracted_colored")
        os.makedirs(colored_dir, exist_ok=True)

    # ===== ステップ4: 各タイムラプス画像を処理 =====
    print(f"\n[3/3] アライメント + 引き算処理中...")

    alignment_results = []
    processed_count = 0
    skipped_count = 0
    png_saved_count = 0

    for timelapse_filename in tqdm(timelapse_files, desc=f"    {pos_name}"):
        try:
            # タイムラプス画像を読み込み
            timelapse_path = os.path.join(timelapse_phase_dir, timelapse_filename)
            timelapse_img = load_tif_image(timelapse_path)

            # サイズチェック
            if timelapse_img.shape != reference_img.shape:
                print(f"\n    ⚠️ 警告: 画像サイズが一致しません: {timelapse_filename}")
                skipped_count += 1
                continue

            # ECCアライメント計算
            if method == 'ecc':
                reference_uint8 = to_uint8(reference_img)
                timelapse_uint8 = to_uint8(timelapse_img)

                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-8)

                try:
                    correlation, warp_matrix = cv2.findTransformECC(
                        reference_uint8, timelapse_uint8, warp_matrix,
                        cv2.MOTION_TRANSLATION, criteria
                    )

                    shift_y = warp_matrix[1, 2]
                    shift_x = warp_matrix[0, 2]

                except Exception as e:
                    print(f"\n    ⚠️ 警告: {timelapse_filename}のアライメント計算失敗 - {e}")
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
                    print(f"\n    ⚠️ 警告: {timelapse_filename}のアライメント計算失敗 - {e}")
                    skipped_count += 1
                    continue

            # アライメント適用
            h, w = timelapse_img.shape
            aligned_img = cv2.warpAffine(
                timelapse_img.astype(np.float32),
                warp_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            ).astype(np.float64)

            # アライメント済み画像を保存
            aligned_path = os.path.join(aligned_dir, timelapse_filename)
            io.imsave(aligned_path, aligned_img.astype(np.float32))

            # 引き算: aligned_timelapse - reference
            subtracted = aligned_img - reference_img

            # TIF保存
            base_name = timelapse_filename.replace("_phase.tif", "").replace("_phase.tiff", "")
            subtracted_path = os.path.join(subtracted_dir, f"{base_name}_subtracted.tif")
            io.imsave(subtracted_path, subtracted.astype(np.float32))

            # アライメント情報を記録
            alignment_results.append({
                'filename': timelapse_filename,
                'warp_matrix': warp_matrix.tolist(),
                'shift_y': float(shift_y),
                'shift_x': float(shift_x),
                'correlation': float(correlation)
            })

            processed_count += 1

            # PNG保存（オプション）
            if save_png and (processed_count % png_sample_interval == 0):
                colored_path = os.path.join(colored_dir, f"{base_name}_subtracted.png")

                fig, ax = plt.subplots(figsize=(10, 8))
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                im = ax.imshow(subtracted, cmap=cmap, norm=norm)
                ax.axis('off')
                ax.set_title(f'{pos_name} - {base_name}\n平均: {np.mean(subtracted):.3f}, 標準偏差: {np.std(subtracted):.3f}')
                plt.colorbar(im, ax=ax, fraction=0.046, label='差分 (rad)')
                plt.tight_layout()
                plt.savefig(colored_path, dpi=png_dpi, bbox_inches='tight')
                plt.close()
                png_saved_count += 1

        except Exception as e:
            print(f"\n    ❌ エラー: {timelapse_filename} - {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue

    # ===== ステップ5: JSON保存 =====
    alignment_info = {
        'pos_name': pos_name,
        'reference_image': os.path.basename(reference_image_path),
        'reference_image_path': reference_image_path,
        'method': method,
        'num_processed': processed_count,
        'alignment_results': alignment_results
    }

    json_path = os.path.join(timelapse_phase_dir, "alignment_transforms.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(alignment_info, f, indent=2, ensure_ascii=False)

    # サマリー
    print(f"\n  ✅ 処理完了")
    print(f"     成功: {processed_count}枚")
    if skipped_count > 0:
        print(f"     スキップ: {skipped_count}枚")
    print(f"     保存先（アライメント済み）: {aligned_dir}")
    print(f"     保存先（引き算TIF）: {subtracted_dir}")
    if save_png:
        print(f"     保存先（引き算PNG）: {colored_dir} ({png_saved_count}枚)")
    print(f"     JSON: {json_path}")

    # アライメント統計
    if len(alignment_results) > 0:
        shifts_y = [r['shift_y'] for r in alignment_results]
        shifts_x = [r['shift_x'] for r in alignment_results]
        correlations = [r['correlation'] for r in alignment_results]

        print(f"\n  アライメント統計:")
        print(f"     シフト量Y: 平均={np.mean(shifts_y):.2f}px, 標準偏差={np.std(shifts_y):.2f}px, 範囲=[{np.min(shifts_y):.2f}, {np.max(shifts_y):.2f}]")
        print(f"     シフト量X: 平均={np.mean(shifts_x):.2f}px, 標準偏差={np.std(shifts_x):.2f}px, 範囲=[{np.min(shifts_x):.2f}, {np.max(shifts_x):.2f}]")
        print(f"     相関係数: 平均={np.mean(correlations):.4f}, 範囲=[{np.min(correlations):.4f}, {np.max(correlations):.4f}]")

    return {
        'pos_name': pos_name,
        'alignment_info': alignment_info,
        'processed_count': processed_count,
        'skipped_count': skipped_count,
        'png_saved_count': png_saved_count
    }


def main():
    """メイン処理"""
    # ========================================
    # 設定パラメータ（必要に応じて変更）
    # ========================================

    # ベースディレクトリ
    TIMELAPSE_BASE = r"E:\Acuisition\kitagishi\260216\move_test_1\Pos1"
    REFERENCE_BASE = r"E:\Acuisition\kitagishi\260216\move_test_1\Pos1"

    # 処理範囲
    POS_START = None  # 開始Pos番号（None=全て）例: 1
    POS_END = None    # 終了Pos番号（None=全て）例: 10

    # 参照画像
    # 方法1: REFERENCE_IMAGE_PATH に直接パスを指定（全Posで同じ画像を基準にする）
    #   例: r"E:\Acuisition\kitagishi\260216\move_test_1\Pos1\img_000000200_ph_000.tif"
    # 方法2: None にすると REFERENCE_BASE + REFERENCE_NUMBER から自動検索
    REFERENCE_IMAGE_PATH = r"E:\Acuisition\kitagishi\260216\move_test_1\Pos1\crop_for_precise_ali\output_phase0200.tif"
    REFERENCE_NUMBER = 1  # REFERENCE_IMAGE_PATH=None のときのみ使用

    # 処理設定
    ALIGNMENT_METHOD = 'ecc'  # 'ecc' or 'phase_correlation'
    SAVE_PNG = True           # PNG保存するか
    PNG_DPI = 150             # PNG解像度
    PNG_SAMPLE_INTERVAL = 1   # N枚ごとにPNG保存
    VMIN = -0.1               # カラーマップの最小値
    VMAX = 1.7                # カラーマップの最大値
    CMAP = 'RdBu_r'           # カラーマップ

    # ========================================

    print("="*80)
    print("タイムラプス画像のアライメント・引き算処理")
    print("="*80)

    print(f"\nタイムラプスデータ: {TIMELAPSE_BASE}")
    if REFERENCE_IMAGE_PATH is not None:
        print(f"参照画像（直接指定）: {REFERENCE_IMAGE_PATH}")
    else:
        print(f"参照データ: {REFERENCE_BASE}")
        print(f"参照画像番号: {REFERENCE_NUMBER}")

    # ディレクトリ存在確認
    if not os.path.exists(TIMELAPSE_BASE):
        print(f"\n❌ エラー: タイムラプスディレクトリが見つかりません: {TIMELAPSE_BASE}")
        return

    if REFERENCE_IMAGE_PATH is not None and not os.path.exists(REFERENCE_IMAGE_PATH):
        print(f"\n❌ エラー: 参照画像が見つかりません: {REFERENCE_IMAGE_PATH}")
        return

    if REFERENCE_IMAGE_PATH is None and not os.path.exists(REFERENCE_BASE):
        print(f"\n❌ エラー: 参照ディレクトリが見つかりません: {REFERENCE_BASE}")
        return

    # 対応するPosフォルダのペアを探す
    print("\n対応するPosフォルダを検索中...")

    if POS_START is not None or POS_END is not None:
        print(f"処理Pos範囲: ", end="")
        if POS_START is not None and POS_END is not None:
            print(f"Pos{POS_START} ~ Pos{POS_END}")
        elif POS_START is not None:
            print(f"Pos{POS_START} ~ 最後")
        else:
            print(f"最初 ~ Pos{POS_END}")
    else:
        print(f"処理Pos範囲: 全て")

    if REFERENCE_IMAGE_PATH is not None:
        # 直接パス指定: タイムラプスのPosフォルダを探して全て同じ参照画像を使う
        pairs = []
        for item in sorted(os.listdir(TIMELAPSE_BASE)):
            item_path = os.path.join(TIMELAPSE_BASE, item)
            if not (os.path.isdir(item_path) and item.startswith("Pos") and item != "Pos0"):
                continue
            try:
                pos_num = int(item.replace("Pos", ""))
            except ValueError:
                continue
            if POS_START is not None and pos_num < POS_START:
                continue
            if POS_END is not None and pos_num > POS_END:
                continue
            timelapse_phase_dir = os.path.join(item_path, "output_phase")
            if os.path.exists(timelapse_phase_dir):
                pairs.append((item, timelapse_phase_dir, REFERENCE_IMAGE_PATH))
    else:
        pairs = find_timelapse_and_reference_pairs(
            TIMELAPSE_BASE, REFERENCE_BASE,
            reference_number=REFERENCE_NUMBER,
            pos_start=POS_START,
            pos_end=POS_END
        )

    if len(pairs) == 0:
        print("\n❌ エラー: 対応するPosフォルダが見つかりません")
        print("   位相再構成処理（10_batch_reconstruction_new.py）を先に実行してください。")
        return

    print(f"検出されたPosペア: {len(pairs)}個")
    for pos_name, _, ref_path in pairs:
        print(f"  - {pos_name}: 参照={os.path.basename(ref_path)}")

    # 処理設定
    print(f"\n処理設定:")
    print(f"  アライメント方法: {ALIGNMENT_METHOD.upper()}")
    print(f"  PNG保存: {SAVE_PNG}")
    if SAVE_PNG:
        print(f"  PNG解像度: {PNG_DPI} dpi")
        print(f"  サンプリング間隔: {PNG_SAMPLE_INTERVAL}枚ごと")
        print(f"  カラーマップ範囲: [{VMIN}, {VMAX}]")
        print(f"  カラーマップ: {CMAP}")

    # 各Posペアを処理
    results = []

    for pos_name, timelapse_phase_dir, reference_image_path in pairs:
        result = process_pos_timelapse(
            pos_name, timelapse_phase_dir, reference_image_path,
            method=ALIGNMENT_METHOD,
            save_png=SAVE_PNG,
            vmin=VMIN,
            vmax=VMAX,
            cmap=CMAP,
            png_dpi=PNG_DPI,
            png_sample_interval=PNG_SAMPLE_INTERVAL
        )

        if result is not None:
            results.append(result)

    # 全体サマリー
    print("\n" + "="*80)
    print("全処理完了")
    print("="*80)
    print(f"\n処理されたPos数: {len(results)}")

    if len(results) > 0:
        total_processed = sum(r['processed_count'] for r in results)
        total_skipped = sum(r['skipped_count'] for r in results)
        print(f"処理された画像総数: {total_processed}枚")
        if total_skipped > 0:
            print(f"スキップされた画像: {total_skipped}枚")

        # 全体のアライメント統計
        all_shifts_y = []
        all_shifts_x = []
        all_correlations = []

        for r in results:
            for ar in r['alignment_info']['alignment_results']:
                all_shifts_y.append(ar['shift_y'])
                all_shifts_x.append(ar['shift_x'])
                all_correlations.append(ar['correlation'])

        if len(all_shifts_y) > 0:
            print(f"\n全体のアライメント統計:")
            print(f"  シフト量Y: 平均={np.mean(all_shifts_y):.2f}px, 標準偏差={np.std(all_shifts_y):.2f}px, 範囲=[{np.min(all_shifts_y):.2f}, {np.max(all_shifts_y):.2f}]")
            print(f"  シフト量X: 平均={np.mean(all_shifts_x):.2f}px, 標準偏差={np.std(all_shifts_x):.2f}px, 範囲=[{np.min(all_shifts_x):.2f}, {np.max(all_shifts_x):.2f}]")
            print(f"  相関係数: 平均={np.mean(all_correlations):.4f}, 範囲=[{np.min(all_correlations):.4f}, {np.max(all_correlations):.4f}]")


if __name__ == "__main__":
    main()

# %%
