# %%
"""
ファイル名の数字部分（下4桁など）でマッチングして変換を適用
"""

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
from pathlib import Path
import json
import cv2
import re

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
        デフォルト: -5.0 ~ 2.0 rad（実測値域）
    
    Returns:
    --------
    uint8 : 0-255の範囲に正規化された画像
    
    Note:
    -----
    - 全画像で一貫した変換を保証
    - 外れ値は自動的にクリッピング
    - ECCアルゴリズムには十分な精度
    """
    # クリッピング（外れ値除去）
    clipped = np.clip(img, vmin, vmax)
    
    # 0-255に正規化
    normalized = (clipped - vmin) / (vmax - vmin)
    
    return (normalized * 255).astype(np.uint8)

def get_tif_files(folder):
    """TIFファイルを確実に取得"""
    tif_files = []
    extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    
    for file_path in Path(folder).iterdir():
        if file_path.is_file():
            if file_path.suffix in extensions:
                tif_files.append(file_path)
    
    return sorted(tif_files)

def extract_number_from_filename(filename):
    """
    ファイル名から数字部分を抽出
    例: "empty0001_bg_corr.tif" → "0001"
    例: "subtracted_by_maskmean_float320001_bg_corr.tif" → "0001"
    """
    # 最後の数字列を探す（拡張子の前）
    basename = Path(filename).stem  # 拡張子を除く
    
    # すべての数字列を見つける
    numbers = re.findall(r'\d+', basename)
    
    if numbers:
        # 最後の数字列を取得（通常はこれがファイル番号）
        last_number = numbers[-1]
        # 下4桁を返す（4桁未満の場合はそのまま）
        return last_number[-4:] if len(last_number) >= 4 else last_number
    else:
        return None

def create_transform_dict(transforms_list):
    """
    変換リストから、ファイル名の数字部分をキーとした辞書を作成
    重複する場合は最初のものを使用
    """
    transform_dict = {}
    
    for transform in transforms_list:
        filename = transform['filename']
        number_key = extract_number_from_filename(filename)
        
        if number_key:
            # 重複チェック
            if number_key in transform_dict:
                print(f"    ⚠️  重複検出: 数字{number_key}のファイル '{filename}' は既に登録済み")
                print(f"       既存: {transform_dict[number_key]['filename']}")
                print(f"       → 最初のものを使用します")
            else:
                transform_dict[number_key] = transform
    
    return transform_dict

def step2_apply_by_filename_number(target_folder, json_path, output_folder,
                                   vmin=-0.1, vmax=1.7, cmap='RdBu_r',
                                   save_png=False, png_dpi=150, png_sample_interval=1):
    """
    ステップ2: ファイル名の数字部分でマッチングして変換を適用
    
    Parameters:
    -----------
    save_png : bool, default=False
        カラーマップPNG画像を保存するかどうか（False=高速モード）
    png_dpi : int, default=150
        PNG保存時の解像度（低いほど高速、300は重い）
    png_sample_interval : int, default=1
        N枚ごとにPNG保存（1=全部、10=10枚に1枚）
    """
    print("=" * 80)
    print("ステップ2: ファイル名の数字部分でマッチング（改良版）")
    print("=" * 80)
    
    # 出力フォルダ作成
    os.makedirs(output_folder, exist_ok=True)
    aligned_folder = os.path.join(output_folder, "aligned")
    subtracted_folder = os.path.join(output_folder, "subtracted")
    colored_folder = os.path.join(output_folder, "colored")
    os.makedirs(aligned_folder, exist_ok=True)
    os.makedirs(subtracted_folder, exist_ok=True)
    os.makedirs(colored_folder, exist_ok=True)
    
    # JSON読み込み
    print(f"\n[1] 変換行列を読み込み: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"❌ エラー: JSONファイルが見つかりません")
        return 0
    
    with open(json_path, 'r', encoding='utf-8') as f:
        save_data = json.load(f)
    
    transforms_list = save_data['transforms']
    
    # 新しいJSONフォーマットから読み込み（後方互換性あり）
    alignment_reference_index = save_data.get('alignment_reference_index', save_data.get('reference_index', 0))
    subtraction_reference_index = save_data.get('subtraction_reference_index', save_data.get('reference_index', 0))
    
    print(f"    変換行列数: {len(transforms_list)}個")
    print(f"    アライメント基準インデックス: {alignment_reference_index}")
    print(f"    引き算基準インデックス: {subtraction_reference_index}")
    print(f"    方法: {save_data['method']}")
    
    # 重複チェックと辞書作成
    print(f"\n[2] 変換行列を数字キーでマッピング...")
    transform_dict = create_transform_dict(transforms_list)
    
    print(f"    ユニークな数字キー数: {len(transform_dict)}個")
    print(f"    重複削除数: {len(transforms_list) - len(transform_dict)}個")
    
    # サンプル表示
    print(f"\n    サンプル（最初の5個）:")
    for i, (number_key, transform) in enumerate(list(transform_dict.items())[:5]):
        print(f"      数字{number_key} → {transform['filename']}")
    
    # ターゲットフォルダのファイルリスト
    target_files = get_tif_files(target_folder)
    
    print(f"\n[3] ターゲットフォルダ: {target_folder}")
    print(f"    TIFファイル数: {len(target_files)}個")
    
    # ターゲットファイルの数字キーを抽出
    target_with_keys = []
    for f in target_files:
        number_key = extract_number_from_filename(f.name)
        target_with_keys.append({
            'path': f,
            'filename': f.name,
            'number_key': number_key
        })
    
    # マッチング確認
    print(f"\n[4] マッチング確認...")
    matched_count = 0
    unmatched_files = []
    
    for target_info in target_with_keys:
        if target_info['number_key'] in transform_dict:
            matched_count += 1
        else:
            unmatched_files.append(target_info)
    
    print(f"    マッチング成功: {matched_count}個")
    print(f"    マッチング失敗: {len(unmatched_files)}個")
    
    if len(unmatched_files) > 0:
        print(f"\n    ❌ 変換行列が見つからないファイル（最初の10個）:")
        for i, info in enumerate(unmatched_files[:10]):
            print(f"      {info['filename']} (数字: {info['number_key']})")
        if len(unmatched_files) > 10:
            print(f"      ... 他{len(unmatched_files)-10}個")
    
    # サンプルマッチング表示
    print(f"\n    ✅ マッチング例（最初の5個）:")
    matched_samples = [t for t in target_with_keys if t['number_key'] in transform_dict][:5]
    for info in matched_samples:
        transform = transform_dict[info['number_key']]
        print(f"      {info['filename']}")
        print(f"        ↓ 数字{info['number_key']}でマッチ")
        print(f"      {transform['filename']}")
        print(f"        シフト: Y={transform['shift_y']:.2f}, X={transform['shift_x']:.2f}")
        print()
    
    # 変換適用
    print(f"\n[5] 変換適用中（{matched_count}個を処理）...")
    
    aligned_images = []
    failed_files = []
    processed_count = 0
    
    for target_info in target_with_keys:
        number_key = target_info['number_key']
        target_path = target_info['path']
        
        # 変換行列がない場合はスキップ
        if number_key not in transform_dict:
            continue
        
        transform_data = transform_dict[number_key]
        
        if processed_count % 100 == 0 or processed_count < 5:
            print(f"\n  [{processed_count+1}/{matched_count}] {target_path.name}")
            print(f"    マッチ: {transform_data['filename']} (数字{number_key})")
        
        try:
            # 画像読み込み
            img = load_tif_image(str(target_path))
            img_uint8 = to_uint8(img)
            
            # 変換行列取得
            warp_matrix = np.array(transform_data['warp_matrix'], dtype=np.float32)
            
            # 変換適用（元画像を直接移動）
            h, w = img.shape
            aligned_img = cv2.warpAffine(
                img.astype(np.float32),
                warp_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            ).astype(np.float64)
            
            # アライメント済み画像を保存
            base_name = target_path.stem
            aligned_path = os.path.join(aligned_folder, f"{base_name}_aligned.tif")
            io.imsave(aligned_path, aligned_img.astype(np.float32))
            
            if processed_count % 100 == 0 or processed_count < 5:
                print(f"    シフト適用: Y={transform_data['shift_y']:.2f}, X={transform_data['shift_x']:.2f}")
            
            # 差分計算用に保存
            aligned_images.append({
                'filename': target_path.name,
                'aligned_img': aligned_img,
                'base_name': base_name,
                'number_key': number_key
            })
            
            processed_count += 1
            
        except Exception as e:
            print(f"\n    ❌ エラー: {target_path.name}: {e}")
            failed_files.append({
                'filename': target_path.name,
                'number_key': number_key,
                'reason': str(e)
            })
            continue
    
    if len(failed_files) > 0:
        print(f"\n⚠️  警告: {len(failed_files)}個のファイルの処理に失敗しました")
        for f in failed_files[:10]:
            print(f"    {f['filename']} - {f['reason']}")
    
    print(f"\n    ✅ アライメント完了: {len(aligned_images)}ファイル")
    
    # 差分計算
    if len(aligned_images) == 0:
        print("\n❌ エラー: アライメント済み画像がありません")
        return 0
    
    # 引き算基準画像を探す
    subtraction_transform = transforms_list[subtraction_reference_index]
    subtraction_key = extract_number_from_filename(subtraction_transform['filename'])
    subtraction_reference_img = None
    
    print(f"\n[6] 差分計算の基準画像を検索中...")
    print(f"    引き算基準インデックス: {subtraction_reference_index}")
    print(f"    引き算基準ファイル名: {subtraction_transform['filename']}")
    print(f"    抽出された数字キー: {subtraction_key}")
    
    for img_data in aligned_images:
        if img_data['number_key'] == subtraction_key:
            subtraction_reference_img = img_data['aligned_img']
            print(f"\n[7] 差分TIF計算・保存中")
            print(f"    引き算基準: {img_data['filename']} (数字{subtraction_key})")
            break
    
    if subtraction_reference_img is None:
        print(f"\n⚠️  警告: 引き算基準画像（数字{subtraction_key}）が見つかりません")
        print(f"    最初の画像を基準にします: {aligned_images[0]['filename']}")
        subtraction_reference_img = aligned_images[0]['aligned_img']
    
    # 差分データを保存するリスト
    subtracted_images = []
    
    for idx, img_data in enumerate(aligned_images):
        aligned_img = img_data['aligned_img']
        base_name = img_data['base_name']
        
        # 差分計算
        subtracted = aligned_img - subtraction_reference_img
        
        # 差分TIF保存
        subtracted_path = os.path.join(subtracted_folder, f"{base_name}_subtracted.tif")
        io.imsave(subtracted_path, subtracted.astype(np.float32))
        
        # PNG用にデータを保存
        subtracted_images.append({
            'subtracted': subtracted,
            'base_name': base_name
        })
        
        if idx % 500 == 0:
            print(f"    [{idx+1}/{len(aligned_images)}] 差分TIF保存中...")
    
    print(f"    ✅ 差分TIF保存完了: {len(aligned_images)}ファイル")
    
    # PNG保存（オプション）
    png_saved_count = 0
    if save_png:
        print(f"\n[8] カラーマップPNG保存中...")
        print(f"    サンプリング間隔: {png_sample_interval}枚ごと")
        print(f"    解像度: {png_dpi} dpi")
        
        # カラーマップ設定
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        for idx, data in enumerate(subtracted_images):
            # サンプリング
            if idx % png_sample_interval == 0:
                subtracted = data['subtracted']
                base_name = data['base_name']
                colored_path = os.path.join(colored_folder, f"{base_name}_colored.png")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(subtracted, cmap=cmap, norm=norm)
                ax.axis('off')
                ax.set_title(f'{base_name}\n平均: {np.mean(subtracted):.3f}, 標準偏差: {np.std(subtracted):.3f}')
                plt.colorbar(im, ax=ax, fraction=0.046, label='差分 (a.u.)')
                plt.tight_layout()
                plt.savefig(colored_path, dpi=png_dpi, bbox_inches='tight')
                plt.close()
                png_saved_count += 1
                
                if png_saved_count % 100 == 0:
                    print(f"    [{png_saved_count}枚保存] 進行中...")
        
        print(f"    ✅ PNG保存完了: {png_saved_count}枚")
    else:
        print(f"\n[8] PNG保存: スキップ（高速モード）")
    
    # 最終サマリー
    print("\n" + "=" * 80)
    print("ステップ2 完了")
    print("=" * 80)
    print(f"\n【処理サマリー】")
    print(f"  ターゲットファイル総数: {len(target_files)}個")
    print(f"  変換行列（重複あり）: {len(transforms_list)}個")
    print(f"  変換行列（ユニーク）: {len(transform_dict)}個")
    print(f"  マッチング成功: {matched_count}個")
    print(f"  処理成功: {len(aligned_images)}個")
    print(f"  処理失敗: {len(failed_files)}個")
    print(f"  マッチング失敗: {len(unmatched_files)}個")
    
    print(f"\n【出力フォルダ】")
    print(f"  - アライメント済み: {aligned_folder}")
    print(f"  - 差分TIF: {subtracted_folder}")
    if save_png:
        print(f"  - カラーマップPNG: {colored_folder} ({png_saved_count}枚)")
    else:
        print(f"  - カラーマップPNG: スキップ（save_png=Trueで有効化）")
    
    print(f"\n【カラーマップ設定】")
    print(f"  - vmin={vmin}, vmax={vmax}, vcenter=0")
    print(f"  - cmap={cmap}")
    
    return len(aligned_images)

# ================================================================
# メイン実行
# ================================================================
if __name__ == "__main__":
    
    count = step2_apply_by_filename_number(
        target_folder=r"F:\251212\ph_2\Pos10\ali_test",
        json_path=r"F:\251212\ph_1\Pos10\10_7\alignment_transforms.json",
        output_folder=r"F:\251212\ph_1\Pos10\10_7\bg_corr",
        vmin=-0.1,
        vmax=1.7,
        cmap='RdBu_r',
        save_png=False,           # PNG保存をスキップして高速化（必要な場合はTrueに変更）
        png_dpi=150,              # PNG保存時の解像度（150=軽い、300=重い）
        png_sample_interval=1     # サンプリング間隔（1=全保存、10=10枚に1枚）
    )
    
    if count > 0:
        print("\n✅ ステップ2が完了しました！")
        print("\n全処理が完了しました 🎉")
    else:
        print("\n❌ エラーが発生しました")

# %%