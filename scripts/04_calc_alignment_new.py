# %%
"""
新フォルダ構造対応のアライメント計算スクリプト
wo_0を基準として、wo_2をそれに合わせるアライメント処理
各Posごとに1つのアライメント行列を計算し、そのPos内の全画像に適用
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
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
    
    Parameters:
    -----------
    vmin, vmax : float
        クリッピング範囲（位相画像のrad値）
    
    Returns:
    --------
    uint8 : 0-255の範囲に正規化された画像
    """
    # クリッピング（外れ値除去）
    clipped = np.clip(img, vmin, vmax)
    
    # 0-255に正規化
    normalized = (clipped - vmin) / (vmax - vmin)
    
    return (normalized * 255).astype(np.uint8)


def get_image_number(filename):
    """
    ファイル名から画像番号（末尾3桁）を抽出
    例: img_000000000_Default_005_phase.tif -> 5
    """
    try:
        # _phase.tif を除去
        base = filename.replace("_phase.tif", "").replace("_phase.tiff", "")
        # 最後のアンダースコア以降の数字を取得
        parts = base.split("_")
        return int(parts[-1])
    except:
        return None


def find_matching_pos_folders(wo0_base, wo2_base):
    """
    wo_0とwo_2で対応するPosフォルダのペアを見つける
    
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


def calculate_alignment_for_pos(pos_name, wo0_phase_dir, wo2_phase_dir, 
                                 reference_number=0, method='ecc'):
    """
    1つのPosペアについてアライメント計算を行う
    
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
    
    Returns:
    --------
    dict: アライメント情報
    """
    print(f"\n{'='*80}")
    print(f"処理中: {pos_name}")
    print(f"{'='*80}")
    
    # wo_0の基準画像を探す
    wo0_files = sorted([f for f in os.listdir(wo0_phase_dir) 
                        if f.endswith("_phase.tif") or f.endswith("_phase.tiff")])
    wo2_files = sorted([f for f in os.listdir(wo2_phase_dir) 
                        if f.endswith("_phase.tif") or f.endswith("_phase.tiff")])
    
    if len(wo0_files) == 0 or len(wo2_files) == 0:
        print(f"  ❌ エラー: 位相画像が見つかりません")
        return None
    
    # 基準画像を見つける（番号がreference_numberのもの）
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
    print(f"\n  アライメント計算中（方法: {method}）...")
    
    if method == 'ecc':
        # uint8に変換
        wo0_uint8 = to_uint8(wo0_reference_img)
        wo2_uint8 = to_uint8(wo2_reference_img)
        
        # ECC
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
            
            # warp_matrix形式に変換
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
        'method': method,
        'wo0_phase_dir': wo0_phase_dir,
        'wo2_phase_dir': wo2_phase_dir
    }
    
    return alignment_info


def apply_alignment_to_pos(alignment_info, save_aligned=True):
    """
    計算されたアライメント行列をwo_2の全画像に適用
    
    Parameters:
    -----------
    alignment_info : dict
        calculate_alignment_for_posの戻り値
    save_aligned : bool
        アライメント済み画像を保存するか
    """
    pos_name = alignment_info['pos_name']
    wo2_phase_dir = alignment_info['wo2_phase_dir']
    warp_matrix = np.array(alignment_info['warp_matrix'], dtype=np.float32)
    
    print(f"\n  アライメント適用中: {pos_name}")
    
    # wo_2の全画像を取得
    wo2_files = sorted([f for f in os.listdir(wo2_phase_dir) 
                        if f.endswith("_phase.tif") or f.endswith("_phase.tiff")])
    
    print(f"  対象画像数: {len(wo2_files)}枚")
    
    # 出力ディレクトリ作成
    if save_aligned:
        aligned_dir = os.path.join(wo2_phase_dir, "aligned")
        os.makedirs(aligned_dir, exist_ok=True)
    
    # 各画像にアライメント適用
    for filename in tqdm(wo2_files, desc=f"    {pos_name}"):
        try:
            # 画像読み込み
            img_path = os.path.join(wo2_phase_dir, filename)
            img = load_tif_image(img_path)
            
            # アライメント適用
            h, w = img.shape
            aligned_img = cv2.warpAffine(
                img.astype(np.float32),
                warp_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            ).astype(np.float64)
            
            # 保存
            if save_aligned:
                aligned_path = os.path.join(aligned_dir, filename)
                io.imsave(aligned_path, aligned_img.astype(np.float32))
        
        except Exception as e:
            print(f"\n    ❌ エラー: {filename} - {e}")
            continue
    
    print(f"  ✅ アライメント適用完了")
    
    if save_aligned:
        print(f"  保存先: {aligned_dir}")


def main():
    """メイン処理"""
    print("="*80)
    print("新フォルダ構造対応のアライメント計算")
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
    pos_pairs = find_matching_pos_folders(WO0_BASE, WO2_BASE)
    
    if len(pos_pairs) == 0:
        print("\n❌ エラー: 対応するPosフォルダが見つかりません")
        return
    
    print(f"検出されたPosペア: {len(pos_pairs)}個")
    for pos_name, _, _ in pos_pairs:
        print(f"  - {pos_name}")
    
    # 各Posペアを処理
    alignment_results = []
    
    for pos_name, wo0_phase_dir, wo2_phase_dir in pos_pairs:
        # アライメント計算
        alignment_info = calculate_alignment_for_pos(
            pos_name, wo0_phase_dir, wo2_phase_dir,
            reference_number=0,
            method='ecc'
        )
        
        if alignment_info is not None:
            # アライメント適用
            apply_alignment_to_pos(alignment_info, save_aligned=True)
            
            # JSON保存
            json_path = os.path.join(wo2_phase_dir, "alignment_transform.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(alignment_info, f, indent=2, ensure_ascii=False)
            
            print(f"  JSON保存: {json_path}")
            
            alignment_results.append(alignment_info)
    
    # サマリー表示
    print("\n" + "="*80)
    print("アライメント処理完了")
    print("="*80)
    print(f"\n処理されたPos数: {len(alignment_results)}")
    
    if len(alignment_results) > 0:
        shifts_y = [info['shift_y'] for info in alignment_results]
        shifts_x = [info['shift_x'] for info in alignment_results]
        correlations = [info['correlation'] for info in alignment_results]
        
        print(f"\nシフト量統計:")
        print(f"  Y: 平均={np.mean(shifts_y):.2f}px, 標準偏差={np.std(shifts_y):.2f}px, 範囲=[{np.min(shifts_y):.2f}, {np.max(shifts_y):.2f}]")
        print(f"  X: 平均={np.mean(shifts_x):.2f}px, 標準偏差={np.std(shifts_x):.2f}px, 範囲=[{np.min(shifts_x):.2f}, {np.max(shifts_x):.2f}]")
        print(f"\n相関係数:")
        print(f"  平均={np.mean(correlations):.4f}, 範囲=[{np.min(correlations):.4f}, {np.max(correlations):.4f}]")


if __name__ == "__main__":
    main()

# %%

