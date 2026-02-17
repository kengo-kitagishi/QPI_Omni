#!/usr/bin/env python3
"""
フィルタリング済みデータの可視化を生成するスクリプト

filtered_*px ディレクトリのdensity_tiffから可視化画像を生成
"""
# %%
import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import measure
from tqdm import tqdm

def create_visualization(zstack_file, phase_file, ri_file, concentration_file, 
                         roi_str, output_dir, pixel_size_um=0.348, 
                         wavelength_nm=663, n_medium=1.333):
    """1つのROIの可視化を作成"""
    
    # データ読み込み
    zstack_map = tifffile.imread(zstack_file).astype(np.float32)
    phase_img = tifffile.imread(phase_file).astype(np.float32)
    ri_map = tifffile.imread(ri_file).astype(np.float32)
    concentration_map = tifffile.imread(concentration_file).astype(np.float32)
    
    mask = zstack_map > 0
    
    if not np.any(mask):
        print(f"  Warning: No valid pixels in {roi_str}")
        return False
    
    # フィルタリングされた領域（mask外）を明示的にクリア
    # ri_mapは培地の屈折率に、concentration_mapは0に設定
    ri_map[~mask] = n_medium
    concentration_map[~mask] = 0.0
    
    # 厚みをµm単位に変換
    thickness_um = zstack_map * pixel_size_um
    
    # 統計計算
    stats = {
        'volume_um3': np.sum(thickness_um[mask]) * (pixel_size_um ** 2),
        'ri_mean': np.mean(ri_map[mask]),
        'ri_median': np.median(ri_map[mask]),
        'ri_delta': np.mean(ri_map[mask]) - n_medium,
        'concentration_mean': np.mean(concentration_map[mask]),
        'zstack_max': np.max(zstack_map),
    }
    
    # フレーム番号を抽出
    frame_number = roi_str.split('_Frame_')[-1] if '_Frame_' in roi_str else 'Unknown'
    
    # プロット作成
    fig = plt.figure(figsize=(26, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    # 1. 元画像 + ROI + マスク輪郭線
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(phase_img, cmap='gray', vmin=-0.5, vmax=2.5)
    ax1.set_title(f'Original Image + Mask Contour\nFrame {frame_number}', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    
    # マスク輪郭線を描画
    binary_mask = (zstack_map > 0).astype(np.uint8)
    contours = measure.find_contours(binary_mask, 0.5)
    for contour in contours:
        ax1.plot(contour[:, 1], contour[:, 0], 'c-', linewidth=2, alpha=0.8)
    
    # 2. Z-stackマップ（厚みマップ）
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(thickness_um, cmap='viridis', vmin=0, vmax=12)
    ax2.set_title(f'Thickness Map\n(max={stats["zstack_max"]*pixel_size_um:.2f} µm)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    plt.colorbar(im2, ax=ax2, label='Thickness (µm)')
    
    # 3. 屈折率マップ（RI map）
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(ri_map, cmap='jet', vmin=1.3, vmax=1.39)
    ax3.set_title(f'Refractive Index Map\n(mean={stats["ri_mean"]:.6f})', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    cbar3 = plt.colorbar(im3, ax=ax3, label='RI')
    # 培地RIを示す線を追加
    cbar3.ax.axhline(y=n_medium, color='cyan', linestyle='--', linewidth=2, alpha=0.8)
    
    # 4. 質量濃度マップ（mg/ml）
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(concentration_map, cmap='hot', vmin=0, vmax=450)
    ax4.set_title(f'Protein Concentration Map\n(mean={stats["concentration_mean"]:.1f} mg/ml)', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    plt.colorbar(im4, ax=ax4, label='Concentration (mg/ml)')
    
    # 5. ΔRI マップ（培地との差）
    ax5 = fig.add_subplot(gs[1, 0])
    delta_ri = ri_map - n_medium
    im5 = ax5.imshow(delta_ri, cmap='plasma', vmin=-0.1, vmax=0.3)
    ax5.set_title(f'ΔRI Map (sample - medium)\n(mean ΔRI={stats["ri_delta"]:.6f})', 
                  fontsize=12, fontweight='bold')
    ax5.set_xlabel('X (pixels)')
    ax5.set_ylabel('Y (pixels)')
    plt.colorbar(im5, ax=ax5, label='ΔRI')
    
    # 6. RI vs Thickness
    ax6 = fig.add_subplot(gs[1, 1])
    if np.any(mask):
        thickness_masked = thickness_um[mask]
        ri_masked = ri_map[mask]
        ax6.scatter(thickness_masked, ri_masked, alpha=0.3, s=10, c='blue')
        
        # 培地RIの参照線
        ax6.axhline(y=n_medium, color='cyan', linestyle='--', linewidth=2,
                   label=f'Medium RI: {n_medium:.3f}')
        # 平均RIの参照線
        ax6.axhline(y=stats['ri_mean'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean RI: {stats["ri_mean"]:.6f}')
        
        ax6.set_xlabel('Thickness (µm)', fontsize=11)
        ax6.set_ylabel('Refractive Index', fontsize=11)
        ax6.set_title('RI vs Thickness', fontsize=12, fontweight='bold')
        ax6.set_xlim(0, 6)      # Thickness range: 0-6 µm
        ax6.set_ylim(1.3, 1.6)  # RI range: 1.3-1.6
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. RI分布
    ax7 = fig.add_subplot(gs[1, 2])
    if np.any(mask):
        ri_masked = ri_map[mask].flatten()
        ax7.hist(ri_masked, bins=50, alpha=0.7, color='purple', edgecolor='black', range=(1.30, 1.60))
        ax7.axvline(x=n_medium, color='cyan', linestyle='--', linewidth=2,
                   label=f'Medium: {n_medium:.3f}')
        ax7.axvline(x=stats['ri_mean'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {stats["ri_mean"]:.6f}')
        ax7.set_xlabel('Refractive Index', fontsize=11)
        ax7.set_ylabel('Frequency', fontsize=11)
        ax7.set_title('RI Distribution', fontsize=12, fontweight='bold')
        ax7.set_xlim(1.30, 1.60)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. 質量濃度分布
    ax8 = fig.add_subplot(gs[1, 3])
    if np.any(mask):
        concentration_masked = concentration_map[mask].flatten()
        ax8.hist(concentration_masked, bins=50, alpha=0.7, color='orange', edgecolor='black', range=(0, 450))
        ax8.axvline(x=stats['concentration_mean'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {stats["concentration_mean"]:.1f} mg/ml')
        ax8.set_xlabel('Concentration (mg/ml)', fontsize=11)
        ax8.set_ylabel('Frequency', fontsize=11)
        ax8.set_title('Protein Concentration Distribution', fontsize=12, fontweight='bold')
        ax8.set_xlim(0, 600)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    fig.suptitle(f'{roi_str} - QPI Analysis (Filtered) (V={stats["volume_um3"]:.1f}µm³, RI={stats["ri_mean"]:.4f}, C={stats["concentration_mean"]:.1f}mg/ml)', 
                 fontsize=16, fontweight='bold')
    
    viz_path = os.path.join(output_dir, f"{roi_str}_visualization.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def process_filtered_directory(filtered_dir, pixel_size_um=0.348, wavelength_nm=663, 
                               n_medium=1.333, alpha_ri=0.00018):
    """1つのフィルタリング済みディレクトリを処理"""
    dirname = os.path.basename(filtered_dir)
    print(f"\nProcessing: {dirname}")
    
    # density_tiffディレクトリ
    density_dir = os.path.join(filtered_dir, "density_tiff")
    if not os.path.exists(density_dir):
        print(f"  Error: density_tiff directory not found")
        return 0
    
    # visualizationsディレクトリを作成
    viz_dir = os.path.join(filtered_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # z-stackファイルを検索
    zstack_files = glob.glob(os.path.join(density_dir, "*_zstack.tif"))
    
    if len(zstack_files) == 0:
        print(f"  No z-stack files found")
        return 0
    
    print(f"  Found {len(zstack_files)} files to visualize")
    
    success_count = 0
    
    for zstack_file in tqdm(zstack_files, desc="  Creating visualizations"):
        roi_str = os.path.basename(zstack_file).replace('_zstack.tif', '')
        
        # 対応するファイルを探す
        phase_file = zstack_file.replace('_zstack.tif', '_phase.tif')
        ri_file = zstack_file.replace('_zstack.tif', '_ri.tif')
        concentration_file = zstack_file.replace('_zstack.tif', '_concentration.tif')
        
        # ファイルの存在確認
        if not all([os.path.exists(f) for f in [phase_file, ri_file, concentration_file]]):
            print(f"    Warning: Missing files for {roi_str}")
            continue
        
        try:
            success = create_visualization(
                zstack_file, phase_file, ri_file, concentration_file,
                roi_str, viz_dir,
                pixel_size_um=pixel_size_um,
                wavelength_nm=wavelength_nm,
                n_medium=n_medium
            )
            
            if success:
                success_count += 1
        except Exception as e:
            print(f"    Error creating visualization for {roi_str}: {e}")
    
    print(f"  Created {success_count} visualizations")
    print(f"  Output: {viz_dir}")
    
    return success_count

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(
        description='フィルタリング済みデータの可視化を生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全てのフィルタリング済みデータの可視化を生成
  python 31_create_filtered_visualizations.py
  
  # 特定のフィルタ閾値のみ
  python 31_create_filtered_visualizations.py --filter-pattern "*filtered_1.0px"
  
  # 基準ディレクトリを指定
  python 31_create_filtered_visualizations.py -d G:\\test_dens_est
  
  # パラメータを指定
  python 31_create_filtered_visualizations.py --pixel-size 0.348 --wavelength 663
  
  # 確認なしで実行
  python 31_create_filtered_visualizations.py -y
"""
    )
    
    parser.add_argument('-d', '--base-dir', type=str, default='.',
                        help='基準ディレクトリ（デフォルト: カレントディレクトリ）')
    parser.add_argument('--filter-pattern', type=str, default='*filtered_*',
                        help='フィルタディレクトリパターン（デフォルト: *filtered_*）')
    parser.add_argument('--pixel-size', type=float, default=0.348,
                        help='ピクセルサイズ（µm、デフォルト: 0.348）')
    parser.add_argument('--wavelength', type=float, default=663,
                        help='波長（nm、デフォルト: 663）')
    parser.add_argument('--n-medium', type=float, default=1.333,
                        help='培地の屈折率（デフォルト: 1.333）')
    parser.add_argument('--alpha-ri', type=float, default=0.00018,
                        help='比屈折率増分（ml/mg、デフォルト: 0.00018）')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='確認なしで実行')
    
    # Jupyter環境での実行に対応
    if 'ipykernel' in sys.modules:
        filtered_argv = [arg for arg in sys.argv if not arg.startswith('--f=') and not arg.startswith('-f=')]
        args = parser.parse_args(filtered_argv[1:] if len(filtered_argv) > 1 else [])
    else:
        args = parser.parse_args()
    
    print("="*80)
    print("Create Filtered Visualizations")
    print("="*80)
    print(f"\nBase directory: {os.path.abspath(args.base_dir)}")
    print(f"Filter pattern: {args.filter_pattern}")
    print(f"\nParameters:")
    print(f"  Pixel size: {args.pixel_size} µm")
    print(f"  Wavelength: {args.wavelength} nm")
    print(f"  Medium RI: {args.n_medium}")
    print(f"  Alpha RI: {args.alpha_ri} ml/mg")
    
    # フィルタリング済みディレクトリを検索
    pattern = os.path.join(args.base_dir, f'timeseries_density_output_{args.filter_pattern}')
    filtered_dirs = glob.glob(pattern)
    filtered_dirs = [d for d in filtered_dirs if os.path.isdir(d)]
    
    print(f"\nFound {len(filtered_dirs)} filtered directories")
    
    if len(filtered_dirs) == 0:
        print("\nNo filtered directories found!")
        print(f"Search pattern: {pattern}")
        return
    
    # リスト表示
    print("\nDirectories to process:")
    for i, d in enumerate(filtered_dirs, 1):
        dirname = os.path.basename(d)
        print(f"  {i}. {dirname}")
    
    # 確認
    if not args.yes:
        response = input(f"\nCreate visualizations for all {len(filtered_dirs)} directories? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    
    # 各ディレクトリを処理
    total_viz = 0
    
    for filtered_dir in filtered_dirs:
        try:
            count = process_filtered_directory(
                filtered_dir,
                pixel_size_um=args.pixel_size,
                wavelength_nm=args.wavelength,
                n_medium=args.n_medium,
                alpha_ri=args.alpha_ri
            )
            total_viz += count
        except Exception as e:
            print(f"  Error processing {filtered_dir}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("VISUALIZATION CREATION COMPLETE!")
    print("="*80)
    print(f"\nProcessed {len(filtered_dirs)} directories")
    print(f"Total visualizations created: {total_viz}")
    print(f"\nVisualization directories: */visualizations/")
    print("  - *_visualization.png")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

# %%

