# %%
import numpy as np
from skimage import io, registration
from scipy import ndimage
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_tif_image(path):
    """TIF画像を読み込んでfloatに変換"""
    img = io.imread(path)
    return img.astype(np.float64)

def align_images_phase_correlation(reference_img, target_img, integer_only=True):
    """
    位相相関法を用いて画像を位置合わせ
    
    Parameters:
    -----------
    reference_img : ndarray
        参照画像（空チャネル）
    target_img : ndarray
        位置合わせする画像
    integer_only : bool
        Trueの場合、整数ピクセルのみでシフト（1ピクセル構造保持）
    
    Returns:
    --------
    aligned_img : ndarray
        位置合わせされた画像
    shift : tuple
        検出されたシフト量 (y, x)
    """
    if integer_only:
        # 整数ピクセルのみで検出
        shift, error, diffphase = registration.phase_cross_correlation(
            reference_img, target_img, upsample_factor=1
        )
        shift = np.round(shift).astype(int)
        
        # 整数シフト（補間なし）
        aligned_img = ndimage.shift(target_img, shift, order=0)
    else:
        # サブピクセル精度で検出
        shift, error, diffphase = registration.phase_cross_correlation(
            reference_img, target_img, upsample_factor=10
        )
        
        # サブピクセルシフト（補間あり）
        aligned_img = ndimage.shift(target_img, shift)
    
    return aligned_img, shift, error

def normalize_image(img):
    """画像を0-1に正規化"""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 0:
        return (img - img_min) / (img_max - img_min)
    return img

def process_folder(empty_channel_path, folder_path, output_folder, integer_only=True):
    """
    フォルダ内の全TIF画像をアライメントして差分を計算
    
    Parameters:
    -----------
    empty_channel_path : str
        空チャネル画像のパス
    folder_path : str
        処理対象フォルダのパス
    output_folder : str
        結果を保存するフォルダのパス
    integer_only : bool
        整数ピクセルのみでシフトするか
    """
    print("=" * 80)
    print("フォルダ内TIF画像の一括アライメントと差分処理")
    print("=" * 80)
    
    # 出力フォルダを作成
    os.makedirs(output_folder, exist_ok=True)
    aligned_folder = os.path.join(output_folder, "aligned")
    subtracted_folder = os.path.join(output_folder, "subtracted")
    os.makedirs(aligned_folder, exist_ok=True)
    os.makedirs(subtracted_folder, exist_ok=True)
    
    # 空チャネル画像を読み込み
    print(f"\n[1] 空チャネル画像を読み込み中...")
    print(f"    パス: {empty_channel_path}")
    empty_img = load_tif_image(empty_channel_path)
    print(f"    サイズ: {empty_img.shape}")
    empty_filename = Path(empty_channel_path).name
    
    # フォルダ内のTIFファイルを取得（空チャネル自身は除外）
    tif_files = []
    for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
        tif_files.extend(Path(folder_path).glob(ext))
    
    # 空チャネル自身を除外
    tif_files = [f for f in tif_files if f.name != empty_filename]
    tif_files = sorted(tif_files)
    
    if len(tif_files) == 0:
        print("\n警告: 処理対象のTIFファイルが見つかりませんでした")
        return
    
    print(f"\n[2] 処理対象ファイル: {len(tif_files)}個")
    for i, f in enumerate(tif_files[:5], 1):
        print(f"    {i}. {f.name}")
    if len(tif_files) > 5:
        print(f"    ... 他 {len(tif_files) - 5}ファイル")
    
    # 処理モード表示
    print(f"\n[3] アライメントモード: {'整数ピクセル（1ピクセル構造保持）' if integer_only else 'サブピクセル（補間あり）'}")
    
    # 各ファイルを処理
    print(f"\n[4] 処理開始...")
    results = []
    
    for i, tif_path in enumerate(tif_files, 1):
        print(f"\n  [{i}/{len(tif_files)}] {tif_path.name}")
        
        try:
            # 画像読み込み
            target_img = load_tif_image(str(tif_path))
            
            # サイズチェック
            if target_img.shape != empty_img.shape:
                print(f"    警告: サイズが異なります（{target_img.shape} vs {empty_img.shape}）- スキップ")
                continue
            
            # アライメント
            aligned_img, shift, error = align_images_phase_correlation(
                empty_img, target_img, integer_only
            )
            
            # 差分計算
            subtracted = empty_img - aligned_img
            
            # 結果を保存
            base_name = tif_path.stem
            aligned_path = os.path.join(aligned_folder, f"{base_name}_aligned.tif")
            subtracted_path = os.path.join(subtracted_folder, f"{base_name}_subtracted.tif")
            
            io.imsave(aligned_path, aligned_img.astype(np.float32))
            io.imsave(subtracted_path, subtracted.astype(np.float32))
            
            # 統計情報
            results.append({
                'filename': tif_path.name,
                'shift_y': shift[0],
                'shift_x': shift[1],
                'error': error,
                'subtracted_mean': np.mean(subtracted),
                'subtracted_std': np.std(subtracted)
            })
            
            print(f"    シフト: Y={shift[0]:.2f}, X={shift[1]:.2f}, 誤差={error:.4f}")
            print(f"    差分: 平均={np.mean(subtracted):.2f}, 標準偏差={np.std(subtracted):.2f}")
            
        except Exception as e:
            print(f"    エラー: {e}")
            continue
    
    # サマリーを作成
    print("\n" + "=" * 80)
    print("処理完了サマリー")
    print("=" * 80)
    print(f"処理成功: {len(results)}ファイル")
    print(f"出力フォルダ: {output_folder}")
    print(f"  - アライメント画像: {aligned_folder}")
    print(f"  - 差分画像: {subtracted_folder}")
    
    # 統計レポート
    if len(results) > 0:
        print("\n統計情報:")
        shifts_y = [r['shift_y'] for r in results]
        shifts_x = [r['shift_x'] for r in results]
        print(f"  シフト量 Y: 平均={np.mean(shifts_y):.2f}, 標準偏差={np.std(shifts_y):.2f}, 範囲=[{np.min(shifts_y):.2f}, {np.max(shifts_y):.2f}]")
        print(f"  シフト量 X: 平均={np.mean(shifts_x):.2f}, 標準偏差={np.std(shifts_x):.2f}, 範囲=[{np.min(shifts_x):.2f}, {np.max(shifts_x):.2f}]")
        
        # CSVレポート保存
        report_path = os.path.join(output_folder, "alignment_report.csv")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Filename,Shift_Y,Shift_X,Error,Subtracted_Mean,Subtracted_Std\n")
            for r in results:
                f.write(f"{r['filename']},{r['shift_y']:.4f},{r['shift_x']:.4f},"
                       f"{r['error']:.6f},{r['subtracted_mean']:.4f},{r['subtracted_std']:.4f}\n")
        print(f"\n詳細レポートを保存: {report_path}")
    
    # サンプル画像の可視化（最初の3枚）
    visualize_samples(empty_img, results[:3], aligned_folder, subtracted_folder, output_folder)
    
    return results

def visualize_samples(empty_img, sample_results, aligned_folder, subtracted_folder, output_folder):
    """最初の数枚をサンプル表示"""
    if len(sample_results) == 0:
        return
    
    n_samples = min(3, len(sample_results))
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    empty_norm = normalize_image(empty_img)
    
    for i, result in enumerate(sample_results[:n_samples]):
        base_name = Path(result['filename']).stem
        aligned_path = os.path.join(aligned_folder, f"{base_name}_aligned.tif")
        subtracted_path = os.path.join(subtracted_folder, f"{base_name}_subtracted.tif")
        
        aligned_img = load_tif_image(aligned_path)
        subtracted_img = load_tif_image(subtracted_path)
        
        aligned_norm = normalize_image(aligned_img)
        
        # 空チャネル
        axes[i, 0].imshow(empty_norm, cmap='gray')
        axes[i, 0].set_title(f'空チャネル')
        axes[i, 0].axis('off')
        
        # アライメント後
        axes[i, 1].imshow(aligned_norm, cmap='gray')
        axes[i, 1].set_title(f'{result["filename"]}\n(aligned)')
        axes[i, 1].axis('off')
        
        # オーバーレイ
        overlay = np.zeros((*empty_img.shape, 3))
        overlay[:, :, 0] = empty_norm
        overlay[:, :, 1] = aligned_norm
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'オーバーレイ\nシフト: ({result["shift_y"]:.1f}, {result["shift_x"]:.1f})')
        axes[i, 2].axis('off')
        
        # 差分
        im = axes[i, 3].imshow(subtracted_img, cmap='RdBu_r', 
                               vmin=-np.std(subtracted_img)*3, 
                               vmax=np.std(subtracted_img)*3)
        axes[i, 3].set_title(f'差分画像\n平均: {result["subtracted_mean"]:.1f}')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046)
    
    plt.tight_layout()
    sample_path = os.path.join(output_folder, "sample_results.png")
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    print(f"サンプル画像を保存: {sample_path}")
    plt.close()

# メイン実行
if __name__ == "__main__":
    # パス設定
    empty_channel_path = r"C:\Users\QPI\Desktop\align_demo\subtracted_by_maskmean_float320001.tif"
    folder_path = r"C:\Users\QPI\Desktop\align_demo"
    output_folder = r"C:\Users\QPI\Desktop\align_demo\output"
    
    # 処理実行
    # integer_only=True:  整数ピクセルのみ（1ピクセル構造保持、推奨）
    # integer_only=False: サブピクセル精度（補間あり）
    results = process_folder(empty_channel_path, folder_path, output_folder, integer_only=False)
    
    print("\n処理が完了しました！")
# %%
import numpy as np
from skimage import io, registration
from scipy import ndimage
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_tif_image(path):
    """TIF画像を読み込んでfloatに変換"""
    img = io.imread(path)
    return img.astype(np.float64)

def align_images_phase_correlation(reference_img, target_img, integer_only=True):
    """
    位相相関法を用いて画像を位置合わせ
    
    Parameters:
    -----------
    reference_img : ndarray
        参照画像（空チャネル）
    target_img : ndarray
        位置合わせする画像
    integer_only : bool
        Trueの場合、整数ピクセルのみでシフト（1ピクセル構造保持）
    
    Returns:
    --------
    aligned_img : ndarray
        位置合わせされた画像
    shift : tuple
        検出されたシフト量 (y, x)
    """
    if integer_only:
        # 整数ピクセルのみで検出
        shift, error, diffphase = registration.phase_cross_correlation(
            reference_img, target_img, upsample_factor=1
        )
        shift = np.round(shift).astype(int)
        
        # 整数シフト（補間なし）
        aligned_img = ndimage.shift(target_img, shift, order=0)
    else:
        # サブピクセル精度で検出
        shift, error, diffphase = registration.phase_cross_correlation(
            reference_img, target_img, upsample_factor=10
        )
        
        # サブピクセルシフト（補間あり）
        aligned_img = ndimage.shift(target_img, shift)
    
    return aligned_img, shift, error

def normalize_image(img):
    """画像を0-1に正規化"""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 0:
        return (img - img_min) / (img_max - img_min)
    return img

def process_folder(empty_channel_path, folder_path, output_folder, integer_only=True):
    """
    フォルダ内の全TIF画像をアライメントして差分を計算
    
    Parameters:
    -----------
    empty_channel_path : str
        空チャネル画像のパス
    folder_path : str
        処理対象フォルダのパス
    output_folder : str
        結果を保存するフォルダのパス
    integer_only : bool
        整数ピクセルのみでシフトするか
    """
    print("=" * 80)
    print("フォルダ内TIF画像の一括アライメントと差分処理")
    print("=" * 80)
    
    # 出力フォルダを作成
    os.makedirs(output_folder, exist_ok=True)
    aligned_folder = os.path.join(output_folder, "aligned")
    subtracted_folder = os.path.join(output_folder, "subtracted")
    os.makedirs(aligned_folder, exist_ok=True)
    os.makedirs(subtracted_folder, exist_ok=True)
    
    # 空チャネル画像を読み込み
    print(f"\n[1] 空チャネル画像を読み込み中...")
    print(f"    パス: {empty_channel_path}")
    empty_img = load_tif_image(empty_channel_path)
    print(f"    サイズ: {empty_img.shape}")
    empty_filename = Path(empty_channel_path).name
    
    # フォルダ内のTIFファイルを取得（空チャネル自身は除外）
    tif_files = []
    for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
        tif_files.extend(Path(folder_path).glob(ext))
    
    # 空チャネル自身を除外
    tif_files = [f for f in tif_files if f.name != empty_filename]
    tif_files = sorted(tif_files)
    
    if len(tif_files) == 0:
        print("\n警告: 処理対象のTIFファイルが見つかりませんでした")
        return
    
    print(f"\n[2] 処理対象ファイル: {len(tif_files)}個")
    for i, f in enumerate(tif_files[:5], 1):
        print(f"    {i}. {f.name}")
    if len(tif_files) > 5:
        print(f"    ... 他 {len(tif_files) - 5}ファイル")
    
    # 処理モード表示
    print(f"\n[3] アライメントモード: {'整数ピクセル（1ピクセル構造保持）' if integer_only else 'サブピクセル（補間あり）'}")
    
    # 各ファイルを処理
    print(f"\n[4] 処理開始...")
    results = []
    
    for i, tif_path in enumerate(tif_files, 1):
        print(f"\n  [{i}/{len(tif_files)}] {tif_path.name}")
        
        try:
            # 画像読み込み
            target_img = load_tif_image(str(tif_path))
            
            # サイズチェック
            if target_img.shape != empty_img.shape:
                print(f"    警告: サイズが異なります（{target_img.shape} vs {empty_img.shape}）- スキップ")
                continue
            
            # アライメント
            aligned_img, shift, error = align_images_phase_correlation(
                empty_img, target_img, integer_only
            )
            
            # 差分計算
            subtracted = empty_img - aligned_img
            
            # 結果を保存
            base_name = tif_path.stem
            aligned_path = os.path.join(aligned_folder, f"{base_name}_aligned.tif")
            subtracted_path = os.path.join(subtracted_folder, f"{base_name}_subtracted.tif")
            
            io.imsave(aligned_path, aligned_img.astype(np.float32))
            io.imsave(subtracted_path, subtracted.astype(np.float32))
            
            # 統計情報
            results.append({
                'filename': tif_path.name,
                'shift_y': shift[0],
                'shift_x': shift[1],
                'error': error,
                'subtracted_mean': np.mean(subtracted),
                'subtracted_std': np.std(subtracted)
            })
            
            print(f"    シフト: Y={shift[0]:.2f}, X={shift[1]:.2f}, 誤差={error:.4f}")
            print(f"    差分: 平均={np.mean(subtracted):.2f}, 標準偏差={np.std(subtracted):.2f}")
            
        except Exception as e:
            print(f"    エラー: {e}")
            continue
    
    # サマリーを作成
    print("\n" + "=" * 80)
    print("処理完了サマリー")
    print("=" * 80)
    print(f"処理成功: {len(results)}ファイル")
    print(f"出力フォルダ: {output_folder}")
    print(f"  - アライメント画像: {aligned_folder}")
    print(f"  - 差分画像: {subtracted_folder}")
    
    # 統計レポート
    if len(results) > 0:
        print("\n統計情報:")
        shifts_y = [r['shift_y'] for r in results]
        shifts_x = [r['shift_x'] for r in results]
        print(f"  シフト量 Y: 平均={np.mean(shifts_y):.2f}, 標準偏差={np.std(shifts_y):.2f}, 範囲=[{np.min(shifts_y):.2f}, {np.max(shifts_y):.2f}]")
        print(f"  シフト量 X: 平均={np.mean(shifts_x):.2f}, 標準偏差={np.std(shifts_x):.2f}, 範囲=[{np.min(shifts_x):.2f}, {np.max(shifts_x):.2f}]")
        
        # CSVレポート保存
        report_path = os.path.join(output_folder, "alignment_report.csv")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Filename,Shift_Y,Shift_X,Error,Subtracted_Mean,Subtracted_Std\n")
            for r in results:
                f.write(f"{r['filename']},{r['shift_y']:.4f},{r['shift_x']:.4f},"
                       f"{r['error']:.6f},{r['subtracted_mean']:.4f},{r['subtracted_std']:.4f}\n")
        print(f"\n詳細レポートを保存: {report_path}")
    
    # サンプル画像の可視化（最初の3枚）
    visualize_samples(empty_img, results[:3], aligned_folder, subtracted_folder, output_folder)
    
    return results

def save_colored_subtracted(subtracted_img, output_path, colorbar=True):
    """
    差分画像をカラーマップ付きで保存
    
    Parameters:
    -----------
    subtracted_img : ndarray
        差分画像
    output_path : str
        保存先パス
    colorbar : bool
        カラーバーを表示するか
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmin = -np.std(subtracted_img) * 3
    vmax = np.std(subtracted_img) * 3
    
    im = ax.imshow(subtracted_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.axis('off')
    
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_samples(empty_img, sample_results, aligned_folder, subtracted_folder, output_folder):
    """最初の数枚をサンプル表示"""
    if len(sample_results) == 0:
        return
    
    n_samples = min(3, len(sample_results))
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    empty_norm = normalize_image(empty_img)
    
    # カラー差分画像用フォルダ
    colored_folder = os.path.join(output_folder, "subtracted_colored")
    os.makedirs(colored_folder, exist_ok=True)
    
    for i, result in enumerate(sample_results[:n_samples]):
        base_name = Path(result['filename']).stem
        aligned_path = os.path.join(aligned_folder, f"{base_name}_aligned.tif")
        subtracted_path = os.path.join(subtracted_folder, f"{base_name}_subtracted.tif")
        
        aligned_img = load_tif_image(aligned_path)
        subtracted_img = load_tif_image(subtracted_path)
        
        aligned_norm = normalize_image(aligned_img)
        
        # 空チャネル
        axes[i, 0].imshow(empty_norm, cmap='gray')
        axes[i, 0].set_title(f'空チャネル')
        axes[i, 0].axis('off')
        
        # アライメント後
        axes[i, 1].imshow(aligned_norm, cmap='gray')
        axes[i, 1].set_title(f'{result["filename"]}\n(aligned)')
        axes[i, 1].axis('off')
        
        # オーバーレイ
        overlay = np.zeros((*empty_img.shape, 3))
        overlay[:, :, 0] = empty_norm
        overlay[:, :, 1] = aligned_norm
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'オーバーレイ\nシフト: ({result["shift_y"]:.1f}, {result["shift_x"]:.1f})')
        axes[i, 2].axis('off')
        
        # 差分
        im = axes[i, 3].imshow(subtracted_img, cmap='RdBu_r', 
                               vmin=-np.std(subtracted_img)*3, 
                               vmax=np.std(subtracted_img)*3)
        axes[i, 3].set_title(f'差分画像\n平均: {result["subtracted_mean"]:.1f}')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046)
        
        # カラー差分画像を個別に保存
        colored_path = os.path.join(colored_folder, f"{base_name}_subtracted_colored.png")
        save_colored_subtracted(subtracted_img, colored_path, colorbar=True)
    
    plt.tight_layout()
    sample_path = os.path.join(output_folder, "sample_results.png")
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    print(f"サンプル画像を保存: {sample_path}")
    print(f"カラー差分画像を保存: {colored_folder}")
    plt.close()

# メイン実行
if __name__ == "__main__":
    # パス設定
    empty_channel_path = r"C:\Users\QPI\Desktop\align_demo\subtracted_by_maskmean_float320001.tif"
    folder_path = r"C:\Users\QPI\Desktop\align_demo"
    output_folder = r"C:\Users\QPI\Desktop\align_demo\output"
    
    # 処理実行
    # integer_only=True:  整数ピクセルのみ（1ピクセル構造保持、推奨）
    # integer_only=False: サブピクセル精度（補間あり）
    results = process_folder(empty_channel_path, folder_path, output_folder, integer_only=False)
    
    print("\n処理が完了しました！")
# %%
import numpy as np
from skimage import io, registration
from scipy import ndimage
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_tif_image(path):
    """TIF画像を読み込んでfloatに変換"""
    img = io.imread(path)
    return img.astype(np.float64)

def align_images_phase_correlation(reference_img, target_img, integer_only=True):
    """
    位相相関法を用いて画像を位置合わせ
    
    Parameters:
    -----------
    reference_img : ndarray
        参照画像（空チャネル）
    target_img : ndarray
        位置合わせする画像
    integer_only : bool
        Trueの場合、整数ピクセルのみでシフト（1ピクセル構造保持）
    
    Returns:
    --------
    aligned_img : ndarray
        位置合わせされた画像
    shift : tuple
        検出されたシフト量 (y, x)
    """
    if integer_only:
        # 整数ピクセルのみで検出
        shift, error, diffphase = registration.phase_cross_correlation(
            reference_img, target_img, upsample_factor=1
        )
        shift = np.round(shift).astype(int)
        
        # 整数シフト（補間なし）
        aligned_img = ndimage.shift(target_img, shift, order=0)
    else:
        # サブピクセル精度で検出
        shift, error, diffphase = registration.phase_cross_correlation(
            reference_img, target_img, upsample_factor=10
        )
        
        # サブピクセルシフト（補間あり）
        aligned_img = ndimage.shift(target_img, shift)
    
    return aligned_img, shift, error

def normalize_image(img):
    """画像を0-1に正規化"""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 0:
        return (img - img_min) / (img_max - img_min)
    return img

def process_folder(empty_channel_path, folder_path, output_folder, integer_only=True):
    """
    フォルダ内の全TIF画像をアライメントして差分を計算
    
    Parameters:
    -----------
    empty_channel_path : str
        空チャネル画像のパス
    folder_path : str
        処理対象フォルダのパス
    output_folder : str
        結果を保存するフォルダのパス
    integer_only : bool
        整数ピクセルのみでシフトするか
    """
    print("=" * 80)
    print("フォルダ内TIF画像の一括アライメントと差分処理")
    print("=" * 80)
    
    # 出力フォルダを作成
    os.makedirs(output_folder, exist_ok=True)
    aligned_folder = os.path.join(output_folder, "aligned")
    subtracted_folder = os.path.join(output_folder, "subtracted")
    os.makedirs(aligned_folder, exist_ok=True)
    os.makedirs(subtracted_folder, exist_ok=True)
    
    # 空チャネル画像を読み込み
    print(f"\n[1] 空チャネル画像を読み込み中...")
    print(f"    パス: {empty_channel_path}")
    empty_img = load_tif_image(empty_channel_path)
    print(f"    サイズ: {empty_img.shape}")
    empty_filename = Path(empty_channel_path).name
    
    # フォルダ内のTIFファイルを取得（空チャネル自身は除外）
    tif_files = []
    for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
        tif_files.extend(Path(folder_path).glob(ext))
    
    # 空チャネル自身を除外
    tif_files = [f for f in tif_files if f.name != empty_filename]
    tif_files = sorted(tif_files)
    
    if len(tif_files) == 0:
        print("\n警告: 処理対象のTIFファイルが見つかりませんでした")
        return
    
    print(f"\n[2] 処理対象ファイル: {len(tif_files)}個")
    for i, f in enumerate(tif_files[:5], 1):
        print(f"    {i}. {f.name}")
    if len(tif_files) > 5:
        print(f"    ... 他 {len(tif_files) - 5}ファイル")
    
    # 処理モード表示
    print(f"\n[3] アライメントモード: {'整数ピクセル（1ピクセル構造保持）' if integer_only else 'サブピクセル（補間あり）'}")
    
    # 各ファイルを処理
    print(f"\n[4] 処理開始...")
    results = []
    
    for i, tif_path in enumerate(tif_files, 1):
        print(f"\n  [{i}/{len(tif_files)}] {tif_path.name}")
        
        try:
            # 画像読み込み
            target_img = load_tif_image(str(tif_path))
            
            # サイズチェック
            if target_img.shape != empty_img.shape:
                print(f"    警告: サイズが異なります（{target_img.shape} vs {empty_img.shape}）- スキップ")
                continue
            
            # アライメント
            aligned_img, shift, error = align_images_phase_correlation(
                empty_img, target_img, integer_only
            )
            
            # 差分計算
            subtracted = aligned_img - empty_img
            
            # 結果を保存
            base_name = tif_path.stem
            aligned_path = os.path.join(aligned_folder, f"{base_name}_aligned.tif")
            subtracted_path = os.path.join(subtracted_folder, f"{base_name}_subtracted.tif")
            
            io.imsave(aligned_path, aligned_img.astype(np.float32))
            io.imsave(subtracted_path, subtracted.astype(np.float32))
            
            # 統計情報
            results.append({
                'filename': tif_path.name,
                'shift_y': shift[0],
                'shift_x': shift[1],
                'error': error,
                'subtracted_mean': np.mean(subtracted),
                'subtracted_std': np.std(subtracted)
            })
            
            print(f"    シフト: Y={shift[0]:.2f}, X={shift[1]:.2f}, 誤差={error:.4f}")
            print(f"    差分: 平均={np.mean(subtracted):.2f}, 標準偏差={np.std(subtracted):.2f}")
            
        except Exception as e:
            print(f"    エラー: {e}")
            continue
    
    # サマリーを作成
    print("\n" + "=" * 80)
    print("処理完了サマリー")
    print("=" * 80)
    print(f"処理成功: {len(results)}ファイル")
    print(f"出力フォルダ: {output_folder}")
    print(f"  - アライメント画像: {aligned_folder}")
    print(f"  - 差分画像: {subtracted_folder}")
    
    # 統計レポート
    if len(results) > 0:
        print("\n統計情報:")
        shifts_y = [r['shift_y'] for r in results]
        shifts_x = [r['shift_x'] for r in results]
        print(f"  シフト量 Y: 平均={np.mean(shifts_y):.2f}, 標準偏差={np.std(shifts_y):.2f}, 範囲=[{np.min(shifts_y):.2f}, {np.max(shifts_y):.2f}]")
        print(f"  シフト量 X: 平均={np.mean(shifts_x):.2f}, 標準偏差={np.std(shifts_x):.2f}, 範囲=[{np.min(shifts_x):.2f}, {np.max(shifts_x):.2f}]")
        
        # CSVレポート保存
        report_path = os.path.join(output_folder, "alignment_report.csv")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Filename,Shift_Y,Shift_X,Error,Subtracted_Mean,Subtracted_Std\n")
            for r in results:
                f.write(f"{r['filename']},{r['shift_y']:.4f},{r['shift_x']:.4f},"
                       f"{r['error']:.6f},{r['subtracted_mean']:.4f},{r['subtracted_std']:.4f}\n")
        print(f"\n詳細レポートを保存: {report_path}")
    
    # 全ファイルのカラー差分画像を保存
    print("\n[5] カラー差分画像を保存中...")
    colored_folder = os.path.join(output_folder, "subtracted_colored")
    os.makedirs(colored_folder, exist_ok=True)
    
    for result in results:
        base_name = Path(result['filename']).stem
        subtracted_path = os.path.join(subtracted_folder, f"{base_name}_subtracted.tif")
        colored_path = os.path.join(colored_folder, f"{base_name}_subtracted_colored.png")
        
        subtracted_img = load_tif_image(subtracted_path)
        save_colored_subtracted(subtracted_img, colored_path, colorbar=True)
    
    print(f"    {len(results)}枚のカラー差分画像を保存: {colored_folder}")
    
    # サンプル画像の可視化（最初の3枚）
    visualize_samples(empty_img, results[:3], aligned_folder, subtracted_folder, output_folder)
    
    return results

def save_colored_subtracted(subtracted_img, output_path, colorbar=True):
    """
    差分画像をカラーマップ付きで保存
    
    Parameters:
    -----------
    subtracted_img : ndarray
        差分画像
    output_path : str
        保存先パス
    colorbar : bool
        カラーバーを表示するか
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmin = -np.std(subtracted_img) * 3
    vmax = np.std(subtracted_img) * 3
    
    im = ax.imshow(subtracted_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.axis('off')
    
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_samples(empty_img, sample_results, aligned_folder, subtracted_folder, output_folder):
    """最初の数枚をサンプル表示"""
    if len(sample_results) == 0:
        return
    
    n_samples = min(3, len(sample_results))
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    empty_norm = normalize_image(empty_img)
    
    for i, result in enumerate(sample_results[:n_samples]):
        base_name = Path(result['filename']).stem
        aligned_path = os.path.join(aligned_folder, f"{base_name}_aligned.tif")
        subtracted_path = os.path.join(subtracted_folder, f"{base_name}_subtracted.tif")
        
        aligned_img = load_tif_image(aligned_path)
        subtracted_img = load_tif_image(subtracted_path)
        
        aligned_norm = normalize_image(aligned_img)
        
        # 空チャネル
        axes[i, 0].imshow(empty_norm, cmap='gray')
        axes[i, 0].set_title(f'空チャネル')
        axes[i, 0].axis('off')
        
        # アライメント後
        axes[i, 1].imshow(aligned_norm, cmap='gray')
        axes[i, 1].set_title(f'{result["filename"]}\n(aligned)')
        axes[i, 1].axis('off')
        
        # オーバーレイ
        overlay = np.zeros((*empty_img.shape, 3))
        overlay[:, :, 0] = empty_norm
        overlay[:, :, 1] = aligned_norm
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'オーバーレイ\nシフト: ({result["shift_y"]:.1f}, {result["shift_x"]:.1f})')
        axes[i, 2].axis('off')
        
        # 差分
        im = axes[i, 3].imshow(subtracted_img, cmap='RdBu_r', 
                               vmin=-np.std(subtracted_img)*3, 
                               vmax=np.std(subtracted_img)*3)
        axes[i, 3].set_title(f'差分画像\n平均: {result["subtracted_mean"]:.1f}')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046)
    
    plt.tight_layout()
    sample_path = os.path.join(output_folder, "sample_results.png")
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    print(f"\nサンプル画像を保存: {sample_path}")
    plt.close()

# メイン実行
if __name__ == "__main__":
    # パス設定
    empty_channel_path = r"C:\Users\QPI\Desktop\align_demo\subtracted_by_maskmean_float320001.tif"
    folder_path = r"C:\Users\QPI\Desktop\align_demo"
    output_folder = r"C:\Users\QPI\Desktop\align_demo\output"
    
    # 処理実行
    # integer_only=True:  整数ピクセルのみ（1ピクセル構造保持、推奨）
    # integer_only=False: サブピクセル精度（補間あり）
    results = process_folder(empty_channel_path, folder_path, output_folder, integer_only=False)
    
    print("\n処理が完了しました！")
# %%
