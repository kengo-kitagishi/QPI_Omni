# %%
"""
TIF画像にカラーマップを適用してPNG保存
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
from tqdm import tqdm

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False


def save_colormap_images(input_dir, output_dir=None, vmin=-0.1, vmax=1.7, 
                         cmap='RdBu_r', dpi=150, figsize=(10, 8)):
    """
    TIF画像にカラーマップを適用してPNG保存
    
    Parameters:
    -----------
    input_dir : str
        入力TIF画像のディレクトリ
    output_dir : str, optional
        出力PNGディレクトリ（Noneの場合は入力ディレクトリに'_colored'を追加）
    vmin, vmax : float
        カラーマップの範囲
    cmap : str
        カラーマップ
    dpi : int
        PNG解像度
    figsize : tuple
        図のサイズ
    """
    print("="*80)
    print("カラーマップ可視化処理")
    print("="*80)
    
    # 入力ディレクトリチェック
    if not os.path.exists(input_dir):
        print(f"❌ エラー: 入力ディレクトリが見つかりません: {input_dir}")
        return
    
    # 出力ディレクトリ設定
    if output_dir is None:
        output_dir = input_dir + "_colored"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n入力: {input_dir}")
    print(f"出力: {output_dir}")
    print(f"\n設定:")
    print(f"  カラーマップ範囲: [{vmin}, {vmax}]")
    print(f"  カラーマップ: {cmap}")
    print(f"  解像度: {dpi} dpi")
    
    # TIFファイルを取得
    tif_files = []
    for filename in os.listdir(input_dir):
        if filename.startswith("._"):
            continue  # macOSの隠しファイルをスキップ
        if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
            tif_files.append(filename)
    
    tif_files = sorted(tif_files)
    
    if len(tif_files) == 0:
        print(f"\n❌ エラー: TIFファイルが見つかりません")
        return
    
    print(f"\n検出されたTIFファイル: {len(tif_files)}枚")
    
    # カラーマップ設定
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # 処理
    print(f"\nカラーマップ適用中...")
    saved_count = 0
    
    for filename in tqdm(tif_files, desc="処理中"):
        try:
            # 画像読み込み
            img_path = os.path.join(input_dir, filename)
            img = io.imread(img_path).astype(np.float64)
            
            # PNG保存
            base_name = filename.replace(".tif", "").replace(".tiff", "")
            output_path = os.path.join(output_dir, f"{base_name}_colored.png")
            
            # プロット
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(img, cmap=cmap, norm=norm)
            ax.axis('off')
            ax.set_title(f'{base_name}\n平均: {np.mean(img):.3f}, 標準偏差: {np.std(img):.3f}, 範囲: [{np.min(img):.3f}, {np.max(img):.3f}]')
            plt.colorbar(im, ax=ax, fraction=0.046, label='値')
            plt.tight_layout()
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            saved_count += 1
            
        except Exception as e:
            print(f"\n❌ エラー: {filename} - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # サマリー
    print(f"\n" + "="*80)
    print("処理完了")
    print("="*80)
    print(f"成功: {saved_count}/{len(tif_files)}枚")
    print(f"保存先: {output_dir}")


def main():
    """メイン処理"""
    # ========================================
    # 設定
    # ========================================
    INPUT_DIR = r"F:\t_bucksub\bg_corr"
    OUTPUT_DIR = None  # Noneの場合は入力ディレクトリに'_colored'を追加
    
    # カラーマップ範囲設定
    # データの実際の範囲に合わせて調整してください
    VMIN = -0.5  # データの最小値に合わせて調整（元: -0.1）
    VMAX = 1.7
    CMAP = 'RdBu_r'
    DPI = 150
    FIGSIZE = (10, 8)
    # ========================================
    
    save_colormap_images(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        vmin=VMIN,
        vmax=VMAX,
        cmap=CMAP,
        dpi=DPI,
        figsize=FIGSIZE
    )


if __name__ == "__main__":
    main()

# %%

