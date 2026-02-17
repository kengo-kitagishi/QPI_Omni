#!/usr/bin/env python3
"""
フィルタリング済み条件の結果を個別にプロットするスクリプト

filtered_*px ディレクトリの結果を読み込んで、各条件ごとにプロット
"""
# %%
import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def parse_condition_name(dirname):
    """ディレクトリ名から条件を解析"""
    # timeseries_density_output_ellipse_subpixel5_discrete_round_filtered_1.0px
    
    # フィルタ情報を分離
    if '_filtered_' in dirname:
        main_part, filter_part = dirname.rsplit('_filtered_', 1)
        filter_threshold = filter_part.replace('px', '')
    else:
        main_part = dirname
        filter_threshold = None
    
    parts = main_part.replace('timeseries_density_output_', '').split('_')
    
    result = {'filter_threshold': filter_threshold}
    
    # interpolate処理されているか
    if 'interpolate' in parts:
        result['interpolate'] = True
        parts.remove('interpolate')
    else:
        result['interpolate'] = False
    
    if 'ellipse' in parts:
        result['shape'] = 'ellipse'
    elif 'feret' in parts:
        result['shape'] = 'feret'
    else:
        result['shape'] = 'unknown'
    
    # subpixel
    for part in parts:
        if part.startswith('subpixel'):
            result['subpixel'] = int(part.replace('subpixel', ''))
            break
    
    # thickness mode
    if 'discrete' in parts:
        result['thickness_mode'] = 'discrete'
        # discretize method
        for method in ['round', 'ceil', 'floor', 'pomegranate']:
            if method in parts:
                result['discretize_method'] = method
                break
    else:
        result['thickness_mode'] = 'continuous'
        result['discretize_method'] = None
    
    return result

def plot_single_condition(csv_file, output_base_dir='.'):
    """1つの条件をプロット"""
    dirname = os.path.basename(os.path.dirname(csv_file))
    condition = parse_condition_name(dirname)
    
    print(f"\nProcessing: {dirname}")
    
    try:
        df = pd.read_csv(csv_file)
        
        # roi_str からフレーム番号を抽出（例: ROI_0000_Frame_2438 -> 2438）
        if 'roi_str' in df.columns:
            df['frame_number'] = df['roi_str'].str.extract(r'Frame_(\d+)').astype(int)
            df['roi_id'] = df['roi_str'].str.extract(r'(ROI_\d+)_')
        
        # 時間を計算（frame_number / 12 = 時間[h]）
        df['time_h'] = df['frame_number'] / 12.0
        
        print(f"  Loaded {len(df)} data points")
        print(f"  Time range: {df['time_h'].min():.2f} - {df['time_h'].max():.2f} h")
        
        # 出力ディレクトリ名を作成（timeseries_density_output_ を timeseries_plots_ に置換）
        plot_dir = dirname.replace('timeseries_density_output_', 'timeseries_plots_')
        output_dir = os.path.join(output_base_dir, plot_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 時間ビンごとの統計を計算
        time_bin_h = 1.0  # 1時間ごと
        time_bins = np.arange(
            np.floor(df['time_h'].min()),
            np.ceil(df['time_h'].max()) + time_bin_h,
            time_bin_h
        )
        
        time_centers = []
        volume_means = []
        volume_stds = []
        mass_means = []
        mass_stds = []
        ri_means = []
        ri_stds = []
        
        for i in range(len(time_bins) - 1):
            mask = (df['time_h'] >= time_bins[i]) & (df['time_h'] < time_bins[i+1])
            if np.any(mask):
                time_centers.append((time_bins[i] + time_bins[i+1]) / 2)
                
                if 'volume_um3' in df.columns:
                    volume_means.append(df.loc[mask, 'volume_um3'].mean())
                    volume_stds.append(df.loc[mask, 'volume_um3'].std())
                else:
                    volume_means.append(np.nan)
                    volume_stds.append(np.nan)
                
                if 'total_mass_pg' in df.columns:
                    mass_means.append(df.loc[mask, 'total_mass_pg'].mean())
                    mass_stds.append(df.loc[mask, 'total_mass_pg'].std())
                else:
                    mass_means.append(np.nan)
                    mass_stds.append(np.nan)
                
                if 'ri_mean' in df.columns:
                    ri_means.append(df.loc[mask, 'ri_mean'].mean())
                    ri_stds.append(df.loc[mask, 'ri_mean'].std())
                else:
                    ri_means.append(np.nan)
                    ri_stds.append(np.nan)
        
        # プロット作成
        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        
        # Volume
        axes[0].plot(time_centers, volume_means, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Mean')
        axes[0].fill_between(time_centers, 
                            np.array(volume_means) - np.array(volume_stds), 
                            np.array(volume_means) + np.array(volume_stds), 
                            alpha=0.3, color='#2E86AB', label='±1 SD')
        axes[0].set_xlabel('Time [h]', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('Volume [µm³]', fontsize=13, fontweight='bold')
        axes[0].set_title(f'Volume vs Time\n{plot_dir}', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Mass
        axes[1].plot(time_centers, mass_means, 'o-', linewidth=2, markersize=6, color='#A23B72', label='Mean')
        axes[1].fill_between(time_centers, 
                            np.array(mass_means) - np.array(mass_stds), 
                            np.array(mass_means) + np.array(mass_stds), 
                            alpha=0.3, color='#A23B72', label='±1 SD')
        axes[1].set_xlabel('Time [h]', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('Total Mass [pg]', fontsize=13, fontweight='bold')
        axes[1].set_title('Mass vs Time', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # RI
        axes[2].plot(time_centers, ri_means, 'o-', linewidth=2, markersize=6, color='#F18F01', label='Mean')
        axes[2].fill_between(time_centers, 
                            np.array(ri_means) - np.array(ri_stds), 
                            np.array(ri_means) + np.array(ri_stds), 
                            alpha=0.3, color='#F18F01', label='±1 SD')
        axes[2].set_xlabel('Time [h]', fontsize=13, fontweight='bold')
        axes[2].set_ylabel('Mean RI', fontsize=13, fontweight='bold')
        axes[2].set_title('Refractive Index vs Time', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'timeseries_volume_ri_mass.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()
        
        # 統計サマリーをCSVに保存
        summary_stats = {
            'condition_dir': dirname,
            'shape': condition['shape'],
            'subpixel': condition.get('subpixel', None),
            'thickness_mode': condition['thickness_mode'],
            'discretize_method': condition.get('discretize_method', None),
            'filter_threshold': condition.get('filter_threshold', None),
            'interpolate': condition.get('interpolate', False),
            'n_data_points': len(df),
            'time_min_h': df['time_h'].min(),
            'time_max_h': df['time_h'].max(),
            'mean_volume': df['volume_um3'].mean() if 'volume_um3' in df.columns else np.nan,
            'std_volume': df['volume_um3'].std() if 'volume_um3' in df.columns else np.nan,
            'mean_mass': df['total_mass_pg'].mean() if 'total_mass_pg' in df.columns else np.nan,
            'std_mass': df['total_mass_pg'].std() if 'total_mass_pg' in df.columns else np.nan,
            'mean_ri': df['ri_mean'].mean() if 'ri_mean' in df.columns else np.nan,
            'std_ri': df['ri_mean'].std() if 'ri_mean' in df.columns else np.nan,
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_csv = os.path.join(output_dir, 'condition_statistics.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"  Saved: {summary_csv}")
        
        return summary_stats
        
    except Exception as e:
        print(f"  Error processing {dirname}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(
        description='フィルタリング済み条件の結果を個別にプロット',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全てのフィルタリング済みデータを個別にプロット
  python 30_plot_filtered_conditions.py
  
  # 特定のフィルタ閾値のみ
  python 30_plot_filtered_conditions.py --filter-pattern "*filtered_1.0px"
  
  # 基準ディレクトリを指定
  python 30_plot_filtered_conditions.py -d G:\\test_dens_est
"""
    )
    
    parser.add_argument('-d', '--base-dir', type=str, default='.',
                        help='基準ディレクトリ（デフォルト: カレントディレクトリ）')
    parser.add_argument('--filter-pattern', type=str, default='*filtered_*',
                        help='フィルタディレクトリパターン（デフォルト: *filtered_*）')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='確認なしで実行')
    
    # Jupyter環境での実行に対応
    if 'ipykernel' in sys.modules:
        # Jupyter環境の場合、sys.argvから--f引数を除外
        filtered_argv = [arg for arg in sys.argv if not arg.startswith('--f=') and not arg.startswith('-f=')]
        args = parser.parse_args(filtered_argv[1:] if len(filtered_argv) > 1 else [])
    else:
        args = parser.parse_args()
    
    print("="*80)
    print("Plot Filtered Conditions (Individual)")
    print("="*80)
    print(f"\nBase directory: {os.path.abspath(args.base_dir)}")
    print(f"Filter pattern: {args.filter_pattern}")
    
    # フィルタリング済みディレクトリを検索
    pattern = os.path.join(args.base_dir, f'timeseries_density_output_{args.filter_pattern}', 'filtering_summary.csv')
    csv_files = glob.glob(pattern)
    
    print(f"\nFound {len(csv_files)} filtered condition directories")
    
    if len(csv_files) == 0:
        print("\nNo filtered data found!")
        print(f"Search pattern: {pattern}")
        return
    
    # リスト表示
    print("\nConditions to plot:")
    for i, csv_file in enumerate(csv_files, 1):
        dirname = os.path.basename(os.path.dirname(csv_file))
        print(f"  {i}. {dirname}")
    
    # 確認
    if not args.yes:
        response = input(f"\nPlot all {len(csv_files)} conditions? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    
    # 各条件をプロット
    all_stats = []
    
    for csv_file in csv_files:
        stats = plot_single_condition(csv_file, args.base_dir)
        if stats is not None:
            all_stats.append(stats)
    
    # 全条件のサマリーを結合
    if len(all_stats) > 0:
        combined_summary = pd.DataFrame(all_stats)
        combined_csv = os.path.join(args.base_dir, 'all_filtered_conditions_summary.csv')
        combined_summary.to_csv(combined_csv, index=False)
        
        print("\n" + "="*80)
        print("PLOTTING COMPLETE!")
        print("="*80)
        print(f"\nProcessed {len(all_stats)} conditions")
        print(f"Combined summary: {combined_csv}")
        print(f"\nIndividual plot directories: timeseries_plots_*/")
        print("  - timeseries_volume_ri_mass.png")
        print("  - condition_statistics.csv")
        print("\n" + "="*80)
    else:
        print("\nNo plots generated!")

if __name__ == "__main__":
    main()

# %%
