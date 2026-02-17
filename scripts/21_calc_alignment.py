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
                                       alignment_reference_index=0, 
                                       subtraction_reference_index=0,
                                       method='ecc',
                                       vmin=-0.1, vmax=1.7, cmap='RdBu_r',
                                       save_png=False, png_dpi=150, png_sample_interval=1):
    """
    ステップ1: 空チャネルフォルダでアライメント計算（重複チェック付き）
    
    Parameters:
    -----------
    alignment_reference_index : int, default=1200
        アライメント計算の基準画像インデックス
    subtraction_reference_index : int, default=0
        引き算の基準画像インデックス
    save_png : bool, default=False
        カラーマップPNG画像を保存するかどうか（False=高速モード）
    png_dpi : int, default=150
        PNG保存時の解像度（低いほど高速、300は重い）
    png_sample_interval : int, default=1
        N枚ごとにPNG保存（1=全部、10=10枚に1枚）
    """
    print("=" * 80)
    print("ステップ1: 空チャネルフォルダでアライメント計算（重複チェック付き）")
    print("=" * 80)
    
    # 出力フォルダ作成
    os.makedirs(output_folder, exist_ok=True)
    aligned_folder = os.path.join(output_folder, "aligned")
    subtracted_folder = os.path.join(output_folder, "subtracted")
    colored_folder = os.path.join(output_folder, "colored")
    
    try:
        os.makedirs(aligned_folder, exist_ok=True)
        os.makedirs(subtracted_folder, exist_ok=True)
        os.makedirs(colored_folder, exist_ok=True)
        print(f"\n✅ フォルダ作成成功:")
        print(f"    - {aligned_folder}")
        print(f"    - {subtracted_folder}")
        print(f"    - {colored_folder}")
    except Exception as e:
        print(f"\n❌ エラー: フォルダ作成に失敗しました")
        print(f"   {e}")
        return None
    
    # ファイルリスト取得（重複チェック付き）
    print(f"\n[1] 空チャネルフォルダ: {empty_channel_folder}")
    tif_files = get_tif_files_unique(empty_channel_folder)
    
    if len(tif_files) == 0:
        print("❌ エラー: TIFファイルが見つかりません")
        return None
    
    print(f"\n    最終ファイル数: {len(tif_files)}")
    print(f"\n    ファイルリスト:")
    for i, f in enumerate(tif_files[:10]):  # 最初の10個
        markers = []
        if i == alignment_reference_index:
            markers.append("★アライメント基準")
        if i == subtraction_reference_index:
            markers.append("●引き算基準")
        marker_str = " " + ", ".join(markers) if markers else ""
        print(f"      {i}: {f.name}{marker_str}")
    if len(tif_files) > 10:
        print(f"      ...")
        print(f"      {len(tif_files)-1}: {tif_files[-1].name}")
    
    # アライメント基準画像読み込み
    if alignment_reference_index >= len(tif_files):
        print(f"\n❌ エラー: アライメント基準インデックス {alignment_reference_index} が範囲外です（0-{len(tif_files)-1}）")
        return None
    
    if subtraction_reference_index >= len(tif_files):
        print(f"\n❌ エラー: 引き算基準インデックス {subtraction_reference_index} が範囲外です（0-{len(tif_files)-1}）")
        return None
    
    alignment_reference_path = tif_files[alignment_reference_index]
    subtraction_reference_path = tif_files[subtraction_reference_index]
    
    print(f"\n[2] アライメント基準画像: {alignment_reference_path.name}")
    print(f"    インデックス: {alignment_reference_index}")
    print(f"\n    引き算基準画像: {subtraction_reference_path.name}")
    print(f"    インデックス: {subtraction_reference_index}")
    
    try:
        alignment_reference_img = load_tif_image(str(alignment_reference_path))
        print(f"    サイズ: {alignment_reference_img.shape}")
    except Exception as e:
        print(f"❌ エラー: アライメント基準画像の読み込みに失敗しました")
        print(f"   {e}")
        return None
    
    if method == 'ecc':
        alignment_reference_uint8 = to_uint8(alignment_reference_img)
    
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
            
            if target_img.shape != alignment_reference_img.shape:
                print(f"    ⚠️  警告: サイズ不一致 - スキップ")
                failed_files.append({
                    'index': i,
                    'filename': tif_path.name,
                    'reason': 'サイズ不一致'
                })
                continue
            
            # アライメント基準画像自身の場合
            if i == alignment_reference_index:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                shift_y, shift_x = 0.0, 0.0
                correlation = 1.0
                aligned_img = target_img.copy()
                if i < 5:
                    print(f"    アライメント基準画像（シフトなし）")
            else:
                if method == 'ecc':
                    # ECC
                    target_uint8 = to_uint8(target_img)
                    warp_matrix = np.eye(2, 3, dtype=np.float32)
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-6)
                    
                    correlation, warp_matrix = cv2.findTransformECC(
                        alignment_reference_uint8, target_uint8, warp_matrix,
                        cv2.MOTION_TRANSLATION, criteria
                    )
                    
                    shift_y = warp_matrix[1, 2]
                    shift_x = warp_matrix[0, 2]
                    
                    # 変換適用（元画像を直接移動）
                    h, w = target_img.shape
                    aligned_img = cv2.warpAffine(
                        target_img.astype(np.float32),
                        warp_matrix,
                        (w, h),
                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                    ).astype(np.float64)
                    
                elif method == 'phase_correlation':
                    # Phase Correlation
                    from skimage import registration
                    from scipy import ndimage
                    
                    shift, error, _ = registration.phase_cross_correlation(
                        alignment_reference_img, target_img, upsample_factor=10
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
            # フォルダの存在を確認（念のため）
            os.makedirs(aligned_folder, exist_ok=True)
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
                'alignment_reference_index': alignment_reference_index,
                'alignment_reference_filename': alignment_reference_path.name
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
        'alignment_reference_index': alignment_reference_index,
        'alignment_reference_filename': alignment_reference_path.name,
        'subtraction_reference_index': subtraction_reference_index,
        'subtraction_reference_filename': subtraction_reference_path.name,
        'reference_index': alignment_reference_index,  # 後方互換性のため
        'reference_filename': alignment_reference_path.name,  # 後方互換性のため
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
    
    # 引き算の基準画像を取得
    subtraction_reference_aligned = aligned_images[subtraction_reference_index]['aligned_img']
    print(f"\n[4] 差分TIF計算・保存中")
    print(f"    引き算基準: {aligned_images[subtraction_reference_index]['filename']} (インデックス {subtraction_reference_index})")
    print(f"    アライメント基準: {alignment_reference_path.name} (インデックス {alignment_reference_index})")
    
    # 差分データを保存するリスト
    subtracted_images = []
    
    for idx, img_data in enumerate(aligned_images):
        aligned_img = img_data['aligned_img']
        base_name = img_data['base_name']
        
        # 差分計算
        subtracted = aligned_img - subtraction_reference_aligned
        
        # 差分TIF保存
        # フォルダの存在を確認（念のため）
        os.makedirs(subtracted_folder, exist_ok=True)
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
        print(f"\n[5] カラーマップPNG保存中...")
        print(f"    サンプリング間隔: {png_sample_interval}枚ごと")
        print(f"    解像度: {png_dpi} dpi")
        
        # フォルダの存在を確認（念のため）
        os.makedirs(colored_folder, exist_ok=True)
        
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
        print(f"\n[5] PNG保存: スキップ（高速モード）")
    
    print("\n" + "=" * 80)
    print("ステップ1 完了")
    print("=" * 80)
    print(f"\n出力フォルダ:")
    print(f"  - アライメント済み: {aligned_folder}")
    print(f"  - 差分TIF: {subtracted_folder}")
    if save_png:
        print(f"  - カラーマップPNG: {colored_folder} ({png_saved_count}枚)")
    else:
        print(f"  - カラーマップPNG: スキップ（save_png=Trueで有効化）")
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
        empty_channel_folder=r"F:\251212\ph_1\Pos10\ali_test\bg_corr",
        output_folder=r"F:\251212\ph_1\Pos10\ali_test\aligned",
        output_json=r"F:\251212\ph_1\Pos10\ali_test\alignment_transforms.json",
        alignment_reference_index=0,   # アライメント基準（中間フレーム）
        subtraction_reference_index=0,    # 引き算基準（最初のフレーム）
        method='ecc',
        vmin=-0.1,
        vmax=1.7,
        cmap='RdBu_r',
        save_png=True,           # PNG保存をスキップして高速化（必要な場合はTrueに変更）
        png_dpi=150,              # PNG保存時の解像度（150=軽い、300=重い）
        png_sample_interval=1     # サンプリング間隔（1=全保存、10=10枚に1枚）
    )
    
    if transforms is not None:
        print("\n✅ ステップ1が完了しました！")
        print("\n次に step2_apply_by_number.py を実行してください")
    else:
        print("\n❌ エラーが発生しました")

# %%