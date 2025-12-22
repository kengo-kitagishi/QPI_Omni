# %%
"""
ステップ1: 空チャネルフォルダでアライメント計算（重複チェック付き）
"""

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
from pathlib import Path
import json
import cv2

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

def get_tif_files_unique(folder):
    """
    TIFファイルを取得（重複チェック付き）
    """
    tif_files = []
    seen_names = set()
    extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    
    all_files = []
    for file_path in Path(folder).iterdir():
        if file_path.is_file():
            if file_path.suffix in extensions:
                all_files.append(file_path)
    
    all_files = sorted(all_files)
    
    print(f"\n  📁 フォルダスキャン: {len(all_files)}個のTIFファイルを検出")
    
    # 重複チェック
    duplicates = []
    for file_path in all_files:
        if file_path.name in seen_names:
            duplicates.append(file_path.name)
            print(f"    ⚠️  重複スキップ: {file_path.name}")
        else:
            seen_names.add(file_path.name)
            tif_files.append(file_path)
    
    if len(duplicates) > 0:
        print(f"\n  ⚠️  重複ファイルが{len(duplicates)}個見つかりました")
        print(f"      重複を除いたファイル数: {len(tif_files)}個")
    else:
        print(f"  ✅ 重複なし: {len(tif_files)}個のユニークなファイル")
    
    return tif_files

def step1_calculate_and_subtract_fixed(empty_channel_folder, output_folder, output_json,
                                       reference_index=0, method='ecc',
                                       vmin=-0.1, vmax=1.7, cmap='RdBu_r'):
    """
    ステップ1: 空チャネルフォルダでアライメント計算（重複チェック付き）
    """
    print("=" * 80)
    print("ステップ1: 空チャネルフォルダでアライメント計算（重複チェック付き）")
    print("=" * 80)
    
    # 出力フォルダ作成
    os.makedirs(output_folder, exist_ok=True)
    aligned_folder = os.path.join(output_folder, "aligned")
    subtracted_folder = os.path.join(output_folder, "subtracted")
    colored_folder = os.path.join(output_folder, "colored")
    os.makedirs(aligned_folder, exist_ok=True)
    os.makedirs(subtracted_folder, exist_ok=True)
    os.makedirs(colored_folder, exist_ok=True)
    
    # ファイルリスト取得（重複チェック付き）
    print(f"\n[1] 空チャネルフォルダ: {empty_channel_folder}")
    tif_files = get_tif_files_unique(empty_channel_folder)
    
    if len(tif_files) == 0:
        print("❌ エラー: TIFファイルが見つかりません")
        return None
    
    print(f"\n    最終ファイル数: {len(tif_files)}")
    print(f"\n    ファイルリスト:")
    for i, f in enumerate(tif_files[:10]):  # 最初の10個
        marker = " ★基準" if i == reference_index else ""
        print(f"      {i}: {f.name}{marker}")
    if len(tif_files) > 10:
        print(f"      ...")
        print(f"      {len(tif_files)-1}: {tif_files[-1].name}")
    
    # 基準画像読み込み
    if reference_index >= len(tif_files):
        print(f"\n❌ エラー: 基準インデックス {reference_index} が範囲外です（0-{len(tif_files)-1}）")
        return None
    
    reference_path = tif_files[reference_index]
    print(f"\n[2] 基準画像: {reference_path.name}")
    print(f"    インデックス: {reference_index}")
    
    try:
        reference_img = load_tif_image(str(reference_path))
        print(f"    サイズ: {reference_img.shape}")
    except Exception as e:
        print(f"❌ エラー: 基準画像の読み込みに失敗しました")
        print(f"   {e}")
        return None
    
    if method == 'ecc':
        reference_uint8 = to_uint8(reference_img)
    
    # アライメント計算
    print(f"\n[3] アライメント計算中（方法: {method}）...")
    transforms_list = []
    aligned_images = []
    failed_files = []
    
    for i, tif_path in enumerate(tif_files):
        if i % 100 == 0 or i < 5:
            print(f"\n  [{i+1}/{len(tif_files)}] {tif_path.name}")
        
        try:
            target_img = load_tif_image(str(tif_path))
            
            if target_img.shape != reference_img.shape:
                print(f"    ⚠️  警告: サイズ不一致 - スキップ")
                failed_files.append({
                    'index': i,
                    'filename': tif_path.name,
                    'reason': 'サイズ不一致'
                })
                continue
            
            # 基準画像自身の場合
            if i == reference_index:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                shift_y, shift_x = 0.0, 0.0
                correlation = 1.0
                aligned_img = target_img.copy()
                if i < 5:
                    print(f"    基準画像（シフトなし）")
            else:
                if method == 'ecc':
                    # ECC
                    target_uint8 = to_uint8(target_img)
                    warp_matrix = np.eye(2, 3, dtype=np.float32)
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
                    
                    correlation, warp_matrix = cv2.findTransformECC(
                        reference_uint8, target_uint8, warp_matrix,
                        cv2.MOTION_TRANSLATION, criteria
                    )
                    
                    shift_y = warp_matrix[1, 2]
                    shift_x = warp_matrix[0, 2]
                    
                    # 変換適用
                    h, w = target_img.shape
                    aligned_uint8 = cv2.warpAffine(
                        target_uint8, warp_matrix, (w, h),
                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                    )
                    aligned_img = aligned_uint8.astype(np.float64) / 255.0
                    aligned_img = aligned_img * (np.max(target_img) - np.min(target_img)) + np.min(target_img)
                    
                elif method == 'phase_correlation':
                    # Phase Correlation
                    from skimage import registration
                    from scipy import ndimage
                    
                    shift, error, _ = registration.phase_cross_correlation(
                        reference_img, target_img, upsample_factor=10
                    )
                    
                    shift_y, shift_x = shift[0], shift[1]
                    correlation = 1.0 - error
                    
                    # 変換適用
                    aligned_img = ndimage.shift(target_img, shift, order=1)
                    
                    # warp_matrix形式に変換
                    warp_matrix = np.array([
                        [1.0, 0.0, shift_x],
                        [0.0, 1.0, shift_y]
                    ], dtype=np.float32)
                
                if i % 100 == 0 or i < 5:
                    print(f"    シフト: Y={shift_y:.2f}, X={shift_x:.2f}, 相関={correlation:.4f}")
            
            # アライメント済み画像を保存
            base_name = tif_path.stem
            aligned_path = os.path.join(aligned_folder, f"{base_name}_aligned.tif")
            io.imsave(aligned_path, aligned_img.astype(np.float32))
            
            # 変換情報を保存
            transforms_list.append({
                'index': i,
                'filename': tif_path.name,
                'warp_matrix': warp_matrix.tolist(),
                'shift_y': float(shift_y),
                'shift_x': float(shift_x),
                'correlation': float(correlation),
                'reference_index': reference_index,
                'reference_filename': reference_path.name
            })
            
            # 差分計算用に保存
            aligned_images.append({
                'index': i,
                'filename': tif_path.name,
                'aligned_img': aligned_img,
                'base_name': base_name
            })
            
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            failed_files.append({
                'index': i,
                'filename': tif_path.name,
                'reason': str(e)
            })
            import traceback
            traceback.print_exc()
            continue
    
    # 失敗したファイルのサマリー
    if len(failed_files) > 0:
        print(f"\n⚠️  警告: {len(failed_files)}個のファイルの処理に失敗しました")
        for f in failed_files[:10]:
            print(f"    {f['index']}: {f['filename']} - {f['reason']}")
    
    # JSON保存
    save_data = {
        'reference_index': reference_index,
        'reference_filename': reference_path.name,
        'method': method,
        'empty_channel_folder': str(empty_channel_folder),
        'total_files': len(tif_files),
        'successful_transforms': len(transforms_list),
        'failed_files': failed_files,
        'transforms': transforms_list
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("アライメント計算完了")
    print("=" * 80)
    print(f"全TIFファイル数: {len(tif_files)}個")
    print(f"成功: {len(transforms_list)}個")
    print(f"失敗: {len(failed_files)}個")
    print(f"保存先: {output_json}")
    
    # 統計表示
    if len(transforms_list) > 0:
        shifts_y = [d['shift_y'] for d in transforms_list]
        shifts_x = [d['shift_x'] for d in transforms_list]
        print(f"\nシフト量統計:")
        print(f"  Y: 平均={np.mean(shifts_y):.2f}px, 標準偏差={np.std(shifts_y):.2f}px, 範囲=[{np.min(shifts_y):.2f}, {np.max(shifts_y):.2f}]")
        print(f"  X: 平均={np.mean(shifts_x):.2f}px, 標準偏差={np.std(shifts_x):.2f}px, 範囲=[{np.min(shifts_x):.2f}, {np.max(shifts_x):.2f}]")
    
    # 差分計算
    if len(aligned_images) == 0:
        print("\n❌ エラー: アライメント済み画像がありません")
        return None
    
    reference_aligned = aligned_images[reference_index]['aligned_img']
    print(f"\n[4] 差分計算中（基準: {aligned_images[reference_index]['filename']}）...")
    
    # カラーマップ設定
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    for idx, img_data in enumerate(aligned_images):
        aligned_img = img_data['aligned_img']
        base_name = img_data['base_name']
        
        # 差分計算
        subtracted = aligned_img - reference_aligned
        
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
    
    print("\n" + "=" * 80)
    print("ステップ1 完了")
    print("=" * 80)
    print(f"\n出力フォルダ:")
    print(f"  - アライメント済み: {aligned_folder}")
    print(f"  - 差分TIF: {subtracted_folder}")
    print(f"  - カラーマップPNG: {colored_folder}")
    print(f"  - 変換行列JSON: {output_json}")
    print(f"\nカラーマップ設定:")
    print(f"  - vmin={vmin}, vmax={vmax}, vcenter=0")
    print(f"  - cmap={cmap}")
    
    return transforms_list

# ================================================================
# メイン実行
# ================================================================
if __name__ == "__main__":
    
    transforms = step1_calculate_and_subtract_fixed(
        empty_channel_folder=r"C:\Users\QPI\Desktop\align_demo\empty_channel\bg_corr",
        output_folder=r"C:\Users\QPI\Desktop\align_demo\empty_channel_aligned",
        output_json=r"C:\Users\QPI\Desktop\align_demo\alignment_transforms.json",
        reference_index=0,
        method='ecc',
        vmin=-0.1,
        vmax=1.7,
        cmap='RdBu_r'
    )
    
    if transforms is not None:
        print("\n✅ ステップ1が完了しました！")
        print("\n次に step2_apply_by_number.py を実行してください")
    else:
        print("\n❌ エラーが発生しました")

# %%