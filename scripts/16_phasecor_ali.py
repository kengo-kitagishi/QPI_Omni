# %%
import numpy as np
from skimage import io, registration
from scipy import ndimage
import matplotlib.pyplot as plt

def load_tif_images(empty_channel_path, cell_channel_path):
    """
    TIF画像を読み込む
    
    Parameters:
    -----------
    empty_channel_path : str
        空チャネル画像のパス
    cell_channel_path : str
        細胞ありチャネル画像のパス
    
    Returns:
    --------
    empty_img, cell_img : ndarray
        読み込んだ画像
    """
    empty_img = io.imread(empty_channel_path)
    cell_img = io.imread(cell_channel_path)
    
    # 画像を float に変換して正規化
    empty_img = empty_img.astype(np.float64)
    cell_img = cell_img.astype(np.float64)
    
    return empty_img, cell_img

def align_images_phase_correlation(reference_img, target_img):
    """
    位相相関法を用いて画像を位置合わせ
    
    Parameters:
    -----------
    reference_img : ndarray
        参照画像（空チャネル）
    target_img : ndarray
        位置合わせする画像（細胞チャネル）
    
    Returns:
    --------
    aligned_img : ndarray
        位置合わせされた画像
    shift : tuple
        検出されたシフト量 (y, x)
    """
    # 位相相関法でシフトを検出
    shift, error, diffphase = registration.phase_cross_correlation(
        reference_img, target_img, upsample_factor=10
    )
    
    print(f"検出されたシフト量: Y={shift[0]:.2f}, X={shift[1]:.2f} pixels")
    print(f"相関エラー: {error:.4f}")
    
    # 画像をシフト
    aligned_img = ndimage.shift(target_img, shift)
    
    return aligned_img, shift

def normalize_image(img):
    """画像を0-1に正規化"""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 0:
        return (img - img_min) / (img_max - img_min)
    return img

def visualize_alignment(empty_img, cell_img, aligned_img, subtracted):
    """
    アライメント結果を可視化
    
    Parameters:
    -----------
    empty_img : ndarray
        空チャネル画像
    cell_img : ndarray
        元の細胞チャネル画像
    aligned_img : ndarray
        位置合わせされた細胞チャネル画像
    subtracted : ndarray
        差分画像
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 正規化
    empty_norm = normalize_image(empty_img)
    cell_norm = normalize_image(cell_img)
    aligned_norm = normalize_image(aligned_img)
    
    # 空チャネル
    axes[0, 0].imshow(empty_norm, cmap='gray')
    axes[0, 0].set_title('空チャネル（参照画像）')
    axes[0, 0].axis('off')
    
    # 元の細胞チャネル
    axes[0, 1].imshow(cell_norm, cmap='gray')
    axes[0, 1].set_title('細胞チャネル（元画像）')
    axes[0, 1].axis('off')
    
    # アライメント後の細胞チャネル
    axes[0, 2].imshow(aligned_norm, cmap='gray')
    axes[0, 2].set_title('細胞チャネル（位置合わせ後）')
    axes[0, 2].axis('off')
    
    # オーバーレイ（アライメント前）
    overlay_before = np.zeros((*empty_img.shape, 3))
    overlay_before[:, :, 0] = empty_norm  # 赤: 空チャネル
    overlay_before[:, :, 1] = cell_norm   # 緑: 細胞チャネル
    axes[1, 0].imshow(overlay_before)
    axes[1, 0].set_title('オーバーレイ（位置合わせ前）')
    axes[1, 0].axis('off')
    
    # オーバーレイ（アライメント後）
    overlay_after = np.zeros((*empty_img.shape, 3))
    overlay_after[:, :, 0] = empty_norm    # 赤: 空チャネル
    overlay_after[:, :, 1] = aligned_norm  # 緑: 細胞チャネル
    axes[1, 1].imshow(overlay_after)
    axes[1, 1].set_title('オーバーレイ（位置合わせ後）')
    axes[1, 1].axis('off')
    
    # 差分画像
    im = axes[1, 2].imshow(subtracted, cmap='RdBu_r', vmin=-np.std(subtracted)*3, vmax=np.std(subtracted)*3)
    axes[1, 2].set_title('差分画像（空 - 細胞）')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('alignment_result.png', dpi=150, bbox_inches='tight')
    print("\n結果を 'alignment_result.png' に保存しました")
    plt.show()

def main(empty_channel_path, cell_channel_path):
    """
    メイン処理
    
    Parameters:
    -----------
    empty_channel_path : str
        空チャネルTIF画像のパス
    cell_channel_path : str
        細胞チャネルTIF画像のパス
    """
    print("=" * 60)
    print("2チャネル画像アライメントと差分解析")
    print("=" * 60)
    
    # 画像読み込み
    print("\n[1/4] 画像を読み込んでいます...")
    empty_img, cell_img = load_tif_images(empty_channel_path, cell_channel_path)
    print(f"  空チャネル画像サイズ: {empty_img.shape}")
    print(f"  細胞チャネル画像サイズ: {cell_img.shape}")
    
    # アライメント実行
    print("\n[2/4] 位相相関法でアライメントを実行中...")
    aligned_img, shift = align_images_phase_correlation(empty_img, cell_img)
    
    # 差分計算
    print("\n[3/4] 差分画像を計算中...")
    subtracted = empty_img - aligned_img
    print(f"  差分の平均値: {np.mean(subtracted):.2f}")
    print(f"  差分の標準偏差: {np.std(subtracted):.2f}")
    
    # 可視化
    print("\n[4/4] 結果を可視化中...")
    visualize_alignment(empty_img, cell_img, aligned_img, subtracted)
    
    # アライメント済み画像と差分を保存
    io.imsave('aligned_cell_channel.tif', aligned_img.astype(np.float32))
    io.imsave('subtracted_image.tif', subtracted.astype(np.float32))
    print("\nアライメント済み画像を 'aligned_cell_channel.tif' に保存しました")
    print("差分画像を 'subtracted_image.tif' に保存しました")
    
    print("\n" + "=" * 60)
    print("処理完了！")
    print("=" * 60)
    
    return aligned_img, subtracted, shift

# 使用例
if __name__ == "__main__":
    # 画像ファイルのパスを指定
    empty_path =  r"C:\Users\QPI\Desktop\img_000000210_Default_000_phase.tif"
    cell_path = r"C:\Users\QPI\Desktop\img_000000001_Default_000_phase.tif"
   
    
    # 処理実行
    aligned, subtracted, shift = main(empty_path, cell_path)
    
    print(f"\n最終シフト量: Y={shift[0]:.2f}, X={shift[1]:.2f} pixels")
# %%
