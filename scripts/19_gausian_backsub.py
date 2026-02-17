import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# パラメータ設定
minPhase = -1.1  # ピーク検出の下限（rad）
hist_min = -1.1  # ヒストグラムの最小値（rad）
hist_max = 1.5   # ヒストグラムの最大値（rad）
n_bins = 512     # ヒストグラムのビン数（少ないほど滑らか）
smooth_window = 20  # スムージングのウィンドウサイズ

def gaussian(x, amp, mean, std):
    """ガウス関数"""
    return amp * np.exp(-((x - mean)**2) / (2 * std**2))

def save_png_plots(data, output_dir, png_dpi=150):
    """
    PNG画像を保存（ヒストグラム + 補正前後の比較）
    
    Parameters:
    -----------
    data : dict
        process_image()から返されたデータ
    output_dir : Path
        出力ディレクトリ
    png_dpi : int, default=150
        PNG解像度
    """
    stem = data['stem']
    image_name = data['image_name']
    bin_centers = data['bin_centers']
    smoothed_histo = data['smoothed_histo']
    smoothed_corrected = data['smoothed_corrected']
    x_data = data['x_data']
    y_data = data['y_data']
    popt = data['popt']
    mean = data['mean']
    minPhase = data['minPhase']
    
    # ヒストグラムとガウスフィット
    plt.figure(figsize=(12, 6))
    plt.plot(bin_centers, smoothed_histo, 'b-', alpha=0.5, label='Smoothed Histogram')
    plt.plot(x_data, y_data, 'b-', linewidth=2, label='Fit Region')
    plt.plot(x_data, gaussian(x_data, *popt), 'r-', linewidth=2, label='Gaussian Fit')
    plt.axvline(0, color='g', linestyle='--', linewidth=2, label='0 rad target')
    plt.axvline(mean, color='orange', linestyle='--', linewidth=2, label=f'Peak ({mean:.3f} rad)')
    plt.axvline(minPhase, color='purple', linestyle=':', linewidth=1, label=f'Search limit ({minPhase} rad)')
    plt.xlabel('Phase (rad)')
    plt.ylabel('Count')
    plt.xlim(hist_min, hist_max)
    plt.legend()
    plt.title(f'{image_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_histogram.png", dpi=png_dpi, bbox_inches='tight')
    plt.close()
    
    # 補正前後の比較プロット
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 補正前
    axes[0].plot(bin_centers, smoothed_histo, 'b-', linewidth=2, label='Histogram')
    axes[0].axvline(0, color='g', linestyle='--', linewidth=2, label='0 rad target')
    axes[0].axvline(mean, color='orange', linestyle='--', linewidth=2, label=f'Peak ({mean:.3f} rad)')
    axes[0].set_xlabel('Phase (rad)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_xlim(hist_min, hist_max)
    axes[0].set_title('Before Correction', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 補正後
    axes[1].plot(bin_centers, smoothed_corrected, 'r-', linewidth=2, label='Histogram')
    axes[1].axvline(0, color='g', linestyle='--', linewidth=2, label='0 rad target')
    axes[1].set_xlabel('Phase (rad)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_xlim(hist_min, hist_max)
    axes[1].set_title('After Correction', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_comparison.png", dpi=png_dpi, bbox_inches='tight')
    plt.close()

def process_image(image_path, output_dir, save_png_data=False):
    """
    画像の背景補正を実行してTIF保存（ラジアン単位で処理）
    
    Parameters:
    -----------
    save_png_data : bool, default=False
        PNG作成用のデータを返すかどうか
    
    Returns:
    --------
    dict or bool : 成功時は処理データ（save_png_data=Trueの場合）またはTrue、失敗時はFalse
    """
    # 画像読み込み（32-bit float、ラジアン単位）
    raw_image = tifffile.imread(str(image_path))
    
    if raw_image is None:
        print(f"画像読み込み失敗: {image_path}")
        return False
    
    print(f"\n処理中: {image_path.name}")
    print(f"画像範囲: {raw_image.min():.3f} ~ {raw_image.max():.3f} rad")
    
    # 固定範囲でヒストグラム作成
    bin_edges = np.linspace(hist_min, hist_max, n_bins + 1)
    hist_counts, _ = np.histogram(raw_image.flatten(), bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0]
    
    print(f"ヒストグラム範囲: {hist_min} ~ {hist_max} rad (ビン幅: {bin_width:.4f} rad)")
    
    # スムージング
    smoothed_histo = uniform_filter1d(hist_counts, size=smooth_window, mode='nearest')
    smoothed_histo = uniform_filter1d(smoothed_histo, size=smooth_window, mode='nearest')
    
    # -1.1 rad以上の範囲でピーク検出
    valid_indices = np.where(bin_centers >= minPhase)[0]
    
    if len(valid_indices) == 0:
        print(f"警告: {minPhase} rad以上のデータがありません")
        return
    
    # 検索範囲を制限（最大値付近も除外、例えば上位95%まで）
    max_search_idx = int(len(bin_centers) * 0.95)
    search_indices = valid_indices[valid_indices < max_search_idx]
    
    if len(search_indices) == 0:
        print(f"警告: 有効な検索範囲がありません")
        return
    
    search_range = smoothed_histo[search_indices]
    peak_idx_relative = np.argmax(search_range)
    peak_idx = search_indices[peak_idx_relative]
    peak_value = bin_centers[peak_idx]
    
    print(f"ピーク位置: {peak_value:.3f} rad")
    
    # ガウスフィット用のデータ準備（ピーク周辺±300ビン）
    fit_width = 300
    start_idx = max(0, peak_idx - fit_width)
    end_idx = min(len(bin_centers), peak_idx + fit_width)
    
    x_data = bin_centers[start_idx:end_idx]
    y_data = smoothed_histo[start_idx:end_idx]
    
    # ガウスフィット
    try:
        # 初期推定値
        p0 = [np.max(y_data), peak_value, bin_width * 20]  # amp, mean, std
        
        popt, _ = curve_fit(gaussian, x_data, y_data, p0=p0, maxfev=5000)
        
        amp, mean, std = popt
        print(f"ガウスフィット - 平均: {mean:.4f} rad")
        print(f"              標準偏差: {std:.4f} rad")
        
        # 背景補正（背景ピークを0ラジアンにシフト）
        correction_rad = 0.0 - mean
        final_image_rad = raw_image + correction_rad
        
        print(f"補正値: {correction_rad:.4f} rad")
        print(f"補正後の範囲: {final_image_rad.min():.3f} ~ {final_image_rad.max():.3f} rad")
        
        # 32-bit float TIFFとして保存
        output_path = output_dir / f"{image_path.stem}_bg_corr.tif"
        tifffile.imwrite(str(output_path), final_image_rad.astype(np.float32))
        print(f"保存完了: {output_path.name}")
        
        # PNG用のデータを返す（オプション）
        if save_png_data:
            # 補正後のヒストグラムを計算
            hist_corrected, _ = np.histogram(final_image_rad.flatten(), bins=bin_edges)
            smoothed_corrected = uniform_filter1d(hist_corrected, size=smooth_window, mode='nearest')
            smoothed_corrected = uniform_filter1d(smoothed_corrected, size=smooth_window, mode='nearest')
            
            return {
                'image_name': image_path.name,
                'stem': image_path.stem,
                'bin_centers': bin_centers,
                'smoothed_histo': smoothed_histo,
                'smoothed_corrected': smoothed_corrected,
                'x_data': x_data,
                'y_data': y_data,
                'popt': popt,
                'mean': mean,
                'minPhase': minPhase
            }
        
        return True
        
    except Exception as e:
        print(f"ガウスフィット失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(save_png=False, png_dpi=150, png_sample_interval=1):
    """
    メイン処理：背景補正とオプションでPNG保存
    
    Parameters:
    -----------
    save_png : bool, default=False
        PNG画像を保存するかどうか（False=高速モード）
    png_dpi : int, default=150
        PNG保存時の解像度（低いほど高速）
    png_sample_interval : int, default=1
        N枚ごとにPNG保存（1=全部、10=10枚に1枚）
    """
    # 入力フォルダと出力フォルダの設定
    input_folder = Path(r"F:\t_bucksub")
    output_folder = input_folder / "bg_corr"
    
    # 出力フォルダ作成
    output_folder.mkdir(exist_ok=True)
    
    # TIF画像を取得
    tif_files = sorted(input_folder.glob("*.tif"))
    
    if not tif_files:
        print(f"TIF画像が見つかりません: {input_folder}")
        return
    
    print(f"{len(tif_files)}個の画像を処理します")
    print(f"出力フォルダ: {output_folder}")
    print(f"ヒストグラム範囲: {hist_min} ~ {hist_max} rad")
    print(f"ビン数: {n_bins} (ビン幅: {(hist_max-hist_min)/n_bins:.4f} rad)")
    print(f"ピーク検出範囲: {minPhase} rad以上")
    if save_png:
        print(f"PNG保存: 有効 (サンプリング間隔: {png_sample_interval}枚ごと, dpi={png_dpi})")
    else:
        print(f"PNG保存: スキップ（高速モード）")
    print("="*60)
    
    # フェーズ1: TIF保存
    print("\n[フェーズ1] TIF保存中...")
    print("-"*60)
    success_count = 0
    png_data_list = []
    
    for tif_file in tif_files:
        result = process_image(tif_file, output_folder, save_png_data=save_png)
        if result and result is not False:
            success_count += 1
            if save_png and isinstance(result, dict):
                png_data_list.append(result)
        print("-"*60)
    
    print(f"\n✅ TIF保存完了: {success_count}/{len(tif_files)} 個の画像を正常に処理しました")
    
    # フェーズ2: PNG保存（オプション）
    png_saved_count = 0
    if save_png and len(png_data_list) > 0:
        print("\n" + "="*60)
        print(f"[フェーズ2] PNG保存中...")
        print(f"サンプリング間隔: {png_sample_interval}枚ごと")
        print("-"*60)
        
        for idx, data in enumerate(png_data_list):
            if idx % png_sample_interval == 0:
                save_png_plots(data, output_folder, png_dpi)
                png_saved_count += 1
                if png_saved_count % 100 == 0:
                    print(f"  [{png_saved_count}枚保存] 進行中...")
        
        print(f"\n✅ PNG保存完了: {png_saved_count}枚")
    
    print("\n" + "="*60)
    print(f"全体完了:")
    print(f"  - TIF保存: {success_count}/{len(tif_files)} 個")
    if save_png:
        print(f"  - PNG保存: {png_saved_count}枚")
    else:
        print(f"  - PNG保存: スキップ")
    print("="*60)

if __name__ == "__main__":
    main(
        save_png=False,           # PNG保存をスキップして高速化（必要な場合はTrueに変更）
        png_dpi=150,              # PNG保存時の解像度（150=軽い、300=重い）
        png_sample_interval=1     # サンプリング間隔（1=全保存、10=10枚に1枚）
    )