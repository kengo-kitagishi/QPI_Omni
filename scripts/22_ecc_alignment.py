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

def to_uint8(img):
    """uint8に変換（OpenCV用）"""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 0:
        normalized = (img - img_min) / (img_max - img_min)
    else:
        normalized = img
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
                                   vmin=-0.1, vmax=1.7, cmap='RdBu_r'):
    """
    ステップ2: ファイル名の数字部分でマッチングして変換を適用
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
    reference_index = save_data['reference_index']
    
    print(f"    変換行列数: {len(transforms_list)}個")
    print(f"    基準インデックス: {reference_index}")
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
            
            # 変換適用
            h, w = img.shape
            aligned_uint8 = cv2.warpAffine(
                img_uint8, warp_matrix, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            
            # float64に戻す
            aligned_img = aligned_uint8.astype(np.float64) / 255.0
            aligned_img = aligned_img * (np.max(img) - np.min(img)) + np.min(img)
            
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
    
    # 基準画像を探す（数字0001など最初のもの）
    reference_key = extract_number_from_filename(save_data['reference_filename'])
    reference_img = None
    
    for img_data in aligned_images:
        if img_data['number_key'] == reference_key:
            reference_img = img_data['aligned_img']
            print(f"\n[6] 差分計算中（基準: {img_data['filename']}, 数字{reference_key}）...")
            break
    
    if reference_img is None:
        print(f"\n⚠️  警告: 基準画像（数字{reference_key}）が見つかりません")
        print(f"    最初の画像を基準にします: {aligned_images[0]['filename']}")
        reference_img = aligned_images[0]['aligned_img']
    
    # カラーマップ設定
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    for idx, img_data in enumerate(aligned_images):
        aligned_img = img_data['aligned_img']
        base_name = img_data['base_name']
        
        # 差分計算
        subtracted = aligned_img - reference_img
        
        # 差分TIF保存
        subtracted_path = os.path.join(subtracted_folder, f"{base_name}_subtracted.tif")
        io.imsave(subtracted_path, subtracted.astype(np.float32))
        
        # カラーマップ画像保存
        colored_path = os.path.join(colored_folder, f"{base_name}_colored.png")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(subtracted, cmap=cmap, norm=norm)
        ax.axis('off')
        ax.set_title(f'{base_name}\n平均: {np.mean(subtracted):.3f}, 標準偏差: {np.std(subtracted):.3f}')
        plt.colorbar(im, ax=ax, fraction=0.046, label='差分 (a.u.)')
        plt.tight_layout()
        plt.savefig(colored_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if idx % 500 == 0:
            print(f"    [{idx+1}/{len(aligned_images)}] 差分計算中...")
    
    print(f"    ✅ 差分計算完了: {len(aligned_images)}ファイル")
    
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
    print(f"  - カラーマップPNG: {colored_folder}")
    
    print(f"\n【カラーマップ設定】")
    print(f"  - vmin={vmin}, vmax={vmax}, vcenter=0")
    print(f"  - cmap={cmap}")
    
    return len(aligned_images)

# ================================================================
# メイン実行
# ================================================================
if __name__ == "__main__":
    
    count = step2_apply_by_filename_number(
        target_folder=r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr",
        json_path=r"C:\Users\QPI\Desktop\align_demo\alignment_transforms.json",
        output_folder=r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr",
        vmin=-0.1,
        vmax=1.7,
        cmap='RdBu_r'
    )
    
    if count > 0:
        print("\n✅ ステップ2が完了しました！")
        print("\n全処理が完了しました 🎉")
    else:
        print("\n❌ エラーが発生しました")

# %%