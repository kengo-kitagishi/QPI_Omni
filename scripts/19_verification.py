import numpy as np
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# ヒストグラムパラメータ（19_gaussian_backsub.pyと同じ設定）
hist_min = -1.1
hist_max = 1.5
n_bins = 512


def calculate_statistics(image):
    """画像の統計値を計算"""
    stats = {
        'mean': np.mean(image),
        'median': np.median(image),
        'std': np.std(image),
        'min': np.min(image),
        'max': np.max(image)
    }
    return stats


def create_verification_figure(image_path, output_dir):
    """補正後画像のヒストグラムと統計値を表示する検証図を生成"""
    # 画像読み込み
    image = tifffile.imread(str(image_path))
    
    if image is None:
        print(f"画像読み込み失敗: {image_path}")
        return False
    
    print(f"\n処理中: {image_path.name}")
    
    # 統計値計算
    stats = calculate_statistics(image)
    
    # ヒストグラム作成
    bin_edges = np.linspace(hist_min, hist_max, n_bins + 1)
    hist_counts, _ = np.histogram(image.flatten(), bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 図の作成
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # ヒストグラムをプロット
    ax.bar(bin_centers, hist_counts, width=(bin_centers[1] - bin_centers[0]), 
           color='steelblue', alpha=0.7, edgecolor='none')
    
    # 0 rad基準線
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='0 rad target')
    
    # 軸ラベルとタイトル
    ax.set_xlabel('Phase (rad)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_xlim(hist_min, hist_max)
    ax.set_title(f'{image_path.name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 統計値をテキストボックスで表示
    stats_text = (
        f"Mean: {stats['mean']:.4f} rad\n"
        f"Median: {stats['median']:.4f} rad\n"
        f"Std Dev: {stats['std']:.4f} rad\n"
        f"Min: {stats['min']:.4f} rad\n"
        f"Max: {stats['max']:.4f} rad"
    )
    
    # テキストボックスの配置（右上）
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props,
            family='monospace')
    
    # 保存
    output_path = output_dir / f"{image_path.stem}_verification.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存完了: {output_path.name}")
    print(f"  Mean: {stats['mean']:.4f} rad, Std: {stats['std']:.4f} rad")
    
    return True


def main():
    """メイン処理：bg_corrフォルダ内の全補正済み画像を処理"""
    # コマンドライン引数のパーサー設定
    parser = argparse.ArgumentParser(
        description='19_gaussian_backsub.pyで補正された画像の検証図を生成'
    )
    parser.add_argument(
        '--input_folder',
        type=str,
        default=r"C:\Users\QPI\Desktop\align_demo\bg_corr",
        help='補正済み画像が保存されているフォルダのパス'
    )
    
    args = parser.parse_args()
    
    # 入力フォルダの設定
    input_folder = Path(args.input_folder)
    
    if not input_folder.exists():
        print(f"エラー: フォルダが見つかりません: {input_folder}")
        return
    
    # 補正済みTIF画像を取得（*_bg_corr.tif）
    tif_files = sorted(input_folder.glob("*_bg_corr.tif"))
    
    if not tif_files:
        print(f"補正済み画像（*_bg_corr.tif）が見つかりません: {input_folder}")
        return
    
    print("="*60)
    print(f"{len(tif_files)}個の補正済み画像を処理します")
    print(f"入力フォルダ: {input_folder}")
    print(f"ヒストグラム範囲: {hist_min} ~ {hist_max} rad")
    print(f"ビン数: {n_bins}")
    print("="*60)
    
    # 各画像を処理
    success_count = 0
    for tif_file in tif_files:
        result = create_verification_figure(tif_file, input_folder)
        if result:
            success_count += 1
        print("-"*60)
    
    print("\n" + "="*60)
    print(f"処理完了: {success_count}/{len(tif_files)} 個の画像の検証図を作成しました")
    print("="*60)


if __name__ == "__main__":
    main()

