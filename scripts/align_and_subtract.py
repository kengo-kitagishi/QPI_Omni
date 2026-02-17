# %%
"""
新フォルダ構造対応のアライメント・引き算統合スクリプト
wo_0を基準として、wo_2をアライメントし、引き算まで一気に処理
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


def find_matching_pos_folders(wo0_base, wo2_base, pos_start=None, pos_end=None):
    """
    wo_0とwo_2で対応するPosフォルダのペアを見つける
    
    Parameters:
    -----------
    wo0_base : str
        wo_0のベースディレクトリ
    wo2_base : str
        wo_2のベースディレクトリ
    pos_start : int, optional
        処理開始Pos番号（None=全て）
    pos_end : int, optional
        処理終了Pos番号（None=全て）
    
    Returns:
    --------
    list of tuples: [(pos_name, wo0_path, wo2_path), ...]
    """
    pos_pairs = []
    
    # wo_0のPosフォルダを取得（Pos0を除く）
    wo0_pos_folders = []
    for item in sorted(os.listdir(wo0_base)):
        item_path = os.path.join(wo0_base, item)
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
            
            wo0_pos_folders.append(item)
    
    # 各Posに対応するwo_2のフォルダがあるか確認
    for pos_name in wo0_pos_folders:
        wo0_pos_path = os.path.join(wo0_base, pos_name)
        wo2_pos_path = os.path.join(wo2_base, pos_name)
        
        # 両方の output_phase フォルダが存在するか確認
        wo0_phase_dir = os.path.join(wo0_pos_path, "output_phase")
        wo2_phase_dir = os.path.join(wo2_pos_path, "output_phase")
        
        if os.path.exists(wo0_phase_dir) and os.path.exists(wo2_phase_dir):
            pos_pairs.append((pos_name, wo0_phase_dir, wo2_phase_dir))
    
    return pos_pairs


def process_pos_complete(pos_name, wo0_phase_dir, wo2_phase_dir, 
                         reference_number=0, method='ecc',
                         save_png=True, vmin=-0.1, vmax=1.7, cmap='RdBu_r',
                         png_dpi=150, png_sample_interval=1):
    """
    1つのPosペアについてアライメント計算・適用・引き算を全て実行
    
    Parameters:
    -----------
    pos_name : str
        Posフォルダ名（例: "Pos1"）
    wo0_phase_dir : str
        wo_0の位相画像ディレクトリ
    wo2_phase_dir : str
        wo_2の位相画像ディレクトリ
    reference_number : int
        基準画像の番号（デフォルト: 0）
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
    
    # ===== ステップ1: アライメント計算 =====
    print(f"\n[1/3] アライメント計算中...")
    
    # wo_0とwo_2の画像リストを取得（._ファイルを除外）
    wo0_files = sorted([f for f in os.listdir(wo0_phase_dir) 
                        if not f.startswith("._") and (f.endswith("_phase.tif") or f.endswith("_phase.tiff"))])
    wo2_files = sorted([f for f in os.listdir(wo2_phase_dir) 
                        if not f.startswith("._") and (f.endswith("_phase.tif") or f.endswith("_phase.tiff"))])
    
    if len(wo0_files) == 0 or len(wo2_files) == 0:
        print(f"  ❌ エラー: 位相画像が見つかりません")
        return None
    
    # 基準画像を見つける
    wo0_reference_file = None
    wo2_reference_file = None
    
    for f in wo0_files:
        if get_image_number(f) == reference_number:
            wo0_reference_file = f
            break
    
    for f in wo2_files:
        if get_image_number(f) == reference_number:
            wo2_reference_file = f
            break
    
    if wo0_reference_file is None or wo2_reference_file is None:
        print(f"  ⚠️ 警告: 番号{reference_number}の画像が見つかりません。最初の画像を使用します。")
        wo0_reference_file = wo0_files[0]
        wo2_reference_file = wo2_files[0]
    
    print(f"  wo_0 基準画像: {wo0_reference_file}")
    print(f"  wo_2 ターゲット画像: {wo2_reference_file}")
    
    # 基準画像を読み込み
    wo0_reference_path = os.path.join(wo0_phase_dir, wo0_reference_file)
    wo2_reference_path = os.path.join(wo2_phase_dir, wo2_reference_file)
    
    wo0_reference_img = load_tif_image(wo0_reference_path)
    wo2_reference_img = load_tif_image(wo2_reference_path)
    
    print(f"  wo_0 画像サイズ: {wo0_reference_img.shape}")
    print(f"  wo_2 画像サイズ: {wo2_reference_img.shape}")
    
    # サイズチェック
    if wo0_reference_img.shape != wo2_reference_img.shape:
        print(f"  ❌ エラー: 画像サイズが一致しません")
        return None
    
    # ECCアライメント計算
    if method == 'ecc':
        wo0_uint8 = to_uint8(wo0_reference_img)
        wo2_uint8 = to_uint8(wo2_reference_img)
        
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-6)
        
        try:
            correlation, warp_matrix = cv2.findTransformECC(
                wo0_uint8, wo2_uint8, warp_matrix,
                cv2.MOTION_TRANSLATION, criteria
            )
            
            shift_y = warp_matrix[1, 2]
            shift_x = warp_matrix[0, 2]
            
            print(f"  ✅ アライメント成功")
            print(f"     シフト量: Y={shift_y:.2f}px, X={shift_x:.2f}px")
            print(f"     相関係数: {correlation:.4f}")
            
        except Exception as e:
            print(f"  ❌ アライメント計算エラー: {e}")
            return None
    
    elif method == 'phase_correlation':
        from skimage import registration
        
        try:
            shift, error, _ = registration.phase_cross_correlation(
                wo0_reference_img, wo2_reference_img, upsample_factor=10
            )
            
            shift_y, shift_x = shift[0], shift[1]
            correlation = 1.0 - error
            
            warp_matrix = np.array([
                [1.0, 0.0, shift_x],
                [0.0, 1.0, shift_y]
            ], dtype=np.float32)
            
            print(f"  ✅ アライメント成功")
            print(f"     シフト量: Y={shift_y:.2f}px, X={shift_x:.2f}px")
            print(f"     相関係数: {correlation:.4f}")
            
        except Exception as e:
            print(f"  ❌ アライメント計算エラー: {e}")
            return None
    
    # アライメント情報を保存
    alignment_info = {
        'pos_name': pos_name,
        'wo0_reference_file': wo0_reference_file,
        'wo2_reference_file': wo2_reference_file,
        'reference_number': reference_number,
        'warp_matrix': warp_matrix.tolist(),
        'shift_y': float(shift_y),
        'shift_x': float(shift_x),
        'correlation': float(correlation),
        'method': method
    }
    
    # ===== ステップ2: アライメント適用 + 引き算 =====
    print(f"\n[2/3] アライメント適用 + 引き算処理中...")
    
    # 出力ディレクトリ作成
    aligned_dir = os.path.join(wo2_phase_dir, "aligned")
    subtracted_dir = os.path.join(wo2_phase_dir, "subtracted")
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(subtracted_dir, exist_ok=True)
    
    if save_png:
        colored_dir = os.path.join(wo2_phase_dir, "subtracted_colored")
        os.makedirs(colored_dir, exist_ok=True)
    
    # 画像番号でwo_0の画像をマッピング
    wo0_file_map = {}
    for filename in wo0_files:
        img_number = get_image_number(filename)
        if img_number is not None:
            wo0_file_map[img_number] = filename
    
    # 処理カウンタ
    processed_count = 0
    skipped_count = 0
    png_saved_count = 0
    
    # 各wo_2画像を処理
    for wo2_filename in tqdm(wo2_files, desc=f"    {pos_name}"):
        try:
            # 画像番号を取得
            img_number = get_image_number(wo2_filename)
            
            if img_number is None:
                skipped_count += 1
                continue
            
            # 対応するwo_0の画像を探す
            if img_number not in wo0_file_map:
                if skipped_count < 3:
                    print(f"\n    ⚠️ 警告: wo_0に対応する画像が見つかりません（番号: {img_number}）")
                skipped_count += 1
                continue
            
            wo0_filename = wo0_file_map[img_number]
            
            # wo_2画像を読み込み
            wo2_path = os.path.join(wo2_phase_dir, wo2_filename)
            wo2_img = load_tif_image(wo2_path)
            
            # アライメント適用
            h, w = wo2_img.shape
            aligned_img = cv2.warpAffine(
                wo2_img.astype(np.float32),
                warp_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            ).astype(np.float64)
            
            # アライメント済み画像を保存
            aligned_path = os.path.join(aligned_dir, wo2_filename)
            io.imsave(aligned_path, aligned_img.astype(np.float32))
            
            # wo_0画像を読み込み
            wo0_path = os.path.join(wo0_phase_dir, wo0_filename)
            wo0_img = load_tif_image(wo0_path)
            
            # サイズチェック
            if aligned_img.shape != wo0_img.shape:
                print(f"\n    ⚠️ 警告: 画像サイズが一致しません: {wo2_filename}")
                skipped_count += 1
                continue
            
            # 引き算: wo_2(aligned) - wo_0
            subtracted = aligned_img - wo0_img
            
            # TIF保存
            base_name = wo2_filename.replace("_phase.tif", "").replace("_phase.tiff", "")
            subtracted_path = os.path.join(subtracted_dir, f"{base_name}_subtracted.tif")
            io.imsave(subtracted_path, subtracted.astype(np.float32))
            
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
            print(f"\n    ❌ エラー: {wo2_filename} - {e}")
            skipped_count += 1
            continue
    
    # ===== ステップ3: JSON保存 =====
    print(f"\n[3/3] 結果保存中...")
    
    json_path = os.path.join(wo2_phase_dir, "alignment_transform.json")
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
    # 処理範囲設定（必要に応じて変更）
    # ========================================
    POS_START = None  # 開始Pos番号（None=最初から）例: 1
    POS_END = None    # 終了Pos番号（None=最後まで）例: 10
    # ========================================
    
    print("="*80)
    print("新フォルダ構造対応のアライメント・引き算統合処理")
    print("="*80)
    
    # ベースディレクトリ設定
    WO0_BASE = r"F:\wo_0_EMM_1"
    WO2_BASE = r"F:\wo_2_EMM_1"
    
    print(f"\nwo_0: {WO0_BASE}")
    print(f"wo_2: {WO2_BASE}")
    
    # ディレクトリ存在確認
    if not os.path.exists(WO0_BASE):
        print(f"\n❌ エラー: wo_0ディレクトリが見つかりません: {WO0_BASE}")
        return
    
    if not os.path.exists(WO2_BASE):
        print(f"\n❌ エラー: wo_2ディレクトリが見つかりません: {WO2_BASE}")
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
    
    pos_pairs = find_matching_pos_folders(WO0_BASE, WO2_BASE, pos_start=POS_START, pos_end=POS_END)
    
    if len(pos_pairs) == 0:
        print("\n❌ エラー: 対応するPosフォルダが見つかりません")
        print("   位相再構成処理（batch_reconstruction_new.py）を先に実行してください。")
        return
    
    print(f"検出されたPosペア: {len(pos_pairs)}個")
    for pos_name, _, _ in pos_pairs:
        print(f"  - {pos_name}")
    
    # 処理設定
    SAVE_PNG = True  # PNG保存するか
    PNG_DPI = 150    # PNG解像度
    PNG_SAMPLE_INTERVAL = 1  # N枚ごとにPNG保存
    VMIN = -0.1  # カラーマップの最小値
    VMAX = 1.7   # カラーマップの最大値
    CMAP = 'RdBu_r'  # カラーマップ
    
    print(f"\n処理設定:")
    print(f"  アライメント方法: ECC")
    print(f"  PNG保存: {SAVE_PNG}")
    if SAVE_PNG:
        print(f"  PNG解像度: {PNG_DPI} dpi")
        print(f"  サンプリング間隔: {PNG_SAMPLE_INTERVAL}枚ごと")
        print(f"  カラーマップ範囲: [{VMIN}, {VMAX}]")
        print(f"  カラーマップ: {CMAP}")
    
    # 各Posペアを処理
    results = []
    
    for pos_name, wo0_phase_dir, wo2_phase_dir in pos_pairs:
        result = process_pos_complete(
            pos_name, wo0_phase_dir, wo2_phase_dir,
            reference_number=0,
            method='ecc',
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
        
        # アライメント統計
        shifts_y = [r['alignment_info']['shift_y'] for r in results]
        shifts_x = [r['alignment_info']['shift_x'] for r in results]
        correlations = [r['alignment_info']['correlation'] for r in results]
        
        print(f"\nアライメント統計:")
        print(f"  シフト量Y: 平均={np.mean(shifts_y):.2f}px, 標準偏差={np.std(shifts_y):.2f}px, 範囲=[{np.min(shifts_y):.2f}, {np.max(shifts_y):.2f}]")
        print(f"  シフト量X: 平均={np.mean(shifts_x):.2f}px, 標準偏差={np.std(shifts_x):.2f}px, 範囲=[{np.min(shifts_x):.2f}, {np.max(shifts_x):.2f}]")
        print(f"  相関係数: 平均={np.mean(correlations):.4f}, 範囲=[{np.min(correlations):.4f}, {np.max(correlations):.4f}]")


if __name__ == "__main__":
    main()

# %%

