#!/usr/bin/env python3
"""
全条件の結果を比較するスクリプト

各条件のall_rois_summary.csvを読み込んで、体積・質量・RIなどを比較
"""
# %%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

def parse_condition_name(dirname):
    """ディレクトリ名から条件を解析"""
    # timeseries_density_output_ellipse_subpixel5_discrete_round
    parts = dirname.replace('timeseries_density_output_', '').split('_')
    
    result = {}
    if 'ellipse' in parts:
        result['shape'] = 'ellipse'
        idx = parts.index('ellipse')
    elif 'feret' in parts:
        result['shape'] = 'feret'
        idx = parts.index('feret')
    else:
        result['shape'] = 'unknown'
        idx = 0
    
    # subpixel
    for i, part in enumerate(parts):
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

def load_all_conditions(base_dir='.'):
    """全条件のデータを読み込み"""
    pattern = os.path.join(base_dir, 'timeseries_density_output_*', 'all_rois_summary.csv')
    csv_files = glob.glob(pattern)
    
    print(f"Found {len(csv_files)} condition directories")
    
    all_data = []
    
    for csv_file in csv_files:
        dirname = os.path.basename(os.path.dirname(csv_file))
        condition = parse_condition_name(dirname)
        
        try:
            df = pd.read_csv(csv_file)
            df['condition_dir'] = dirname
            df['shape'] = condition['shape']
            df['subpixel'] = condition['subpixel']
            df['thickness_mode'] = condition['thickness_mode']
            df['discretize_method'] = condition.get('discretize_method', None)
            
            # 条件名を作成
            if condition['thickness_mode'] == 'discrete':
                df['condition_name'] = f"{condition['shape']}_sp{condition['subpixel']}_{condition['discretize_method']}"
            else:
                df['condition_name'] = f"{condition['shape']}_sp{condition['subpixel']}_continuous"
            
            all_data.append(df)
            print(f"  Loaded: {dirname} ({len(df)} ROIs)")
        except Exception as e:
            print(f"  Error loading {dirname}: {e}")
    
    if len(all_data) == 0:
        print("No data loaded!")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(combined_df)} ROIs across {len(all_data)} conditions")
    
    return combined_df

def plot_comparison_by_condition(df, output_dir='condition_comparison'):
    """条件ごとの比較プロット"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 時間を計算（frame_number / 12 = 時間[h]）
    if 'time_h' not in df.columns:
        df['time_h'] = df['frame_number'] / 12.0
    
    # 条件のユニークリストを取得
    conditions = df['condition_name'].unique()
    print(f"\nGenerating comparison plots for {len(conditions)} conditions...")
    
    # 1. 全条件の体積比較
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    for condition in conditions:
        subset = df[df['condition_name'] == condition]
        
        # 時間ビンごとの平均を計算
        time_bins = np.arange(0, subset['time_h'].max() + 1, 1.0)
        time_centers = []
        volume_means = []
        mass_means = []
        ri_means = []
        
        for i in range(len(time_bins) - 1):
            mask = (subset['time_h'] >= time_bins[i]) & (subset['time_h'] < time_bins[i+1])
            if np.any(mask):
                time_centers.append((time_bins[i] + time_bins[i+1]) / 2)
                
                if 'volume_um3' in subset.columns:
                    volume_means.append(subset.loc[mask, 'volume_um3'].mean())
                else:
                    volume_means.append(np.nan)
                
                if 'total_mass_pg' in subset.columns:
                    mass_means.append(subset.loc[mask, 'total_mass_pg'].mean())
                else:
                    mass_means.append(np.nan)
                
                if 'ri_mean' in subset.columns:
                    ri_means.append(subset.loc[mask, 'ri_mean'].mean())
                else:
                    ri_means.append(np.nan)
        
        # プロット
        axes[0].plot(time_centers, volume_means, '-', alpha=0.7, label=condition, linewidth=1.5)
        axes[1].plot(time_centers, mass_means, '-', alpha=0.7, label=condition, linewidth=1.5)
        axes[2].plot(time_centers, ri_means, '-', alpha=0.7, label=condition, linewidth=1.5)
    
    axes[0].set_xlabel('Time [h]', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Volume [µm³]', fontsize=12, fontweight='bold')
    axes[0].set_title('Volume vs Time (All Conditions)', fontsize=14, fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time [h]', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Total Mass [pg]', fontsize=12, fontweight='bold')
    axes[1].set_title('Mass vs Time (All Conditions)', fontsize=14, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Time [h]', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Mean RI', fontsize=12, fontweight='bold')
    axes[2].set_title('RI vs Time (All Conditions)', fontsize=14, fontweight='bold')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_conditions_timeseries.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: all_conditions_timeseries.png")
    plt.close()
    
    # 2. 条件ごとの平均値比較（バープロット）
    summary_stats = []
    
    for condition in conditions:
        subset = df[df['condition_name'] == condition]
        
        stats = {
            'condition': condition,
            'mean_volume': subset['volume_um3'].mean() if 'volume_um3' in subset.columns else np.nan,
            'std_volume': subset['volume_um3'].std() if 'volume_um3' in subset.columns else np.nan,
            'mean_mass': subset['total_mass_pg'].mean() if 'total_mass_pg' in subset.columns else np.nan,
            'std_mass': subset['total_mass_pg'].std() if 'total_mass_pg' in subset.columns else np.nan,
            'mean_ri': subset['ri_mean'].mean() if 'ri_mean' in subset.columns else np.nan,
            'std_ri': subset['ri_mean'].std() if 'ri_mean' in subset.columns else np.nan,
            'n_rois': len(subset)
        }
        
        # 条件パラメータを追加
        stats['shape'] = subset['shape'].iloc[0]
        stats['subpixel'] = subset['subpixel'].iloc[0]
        stats['thickness_mode'] = subset['thickness_mode'].iloc[0]
        stats['discretize_method'] = subset['discretize_method'].iloc[0] if 'discretize_method' in subset.columns else None
        
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.sort_values(['shape', 'subpixel', 'thickness_mode', 'discretize_method'])
    
    # CSVに保存
    summary_csv = os.path.join(output_dir, 'condition_summary_statistics.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Saved: condition_summary_statistics.csv")
    
    # バープロット
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    x_pos = np.arange(len(summary_df))
    
    axes[0].bar(x_pos, summary_df['mean_volume'], yerr=summary_df['std_volume'], alpha=0.7, capsize=5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(summary_df['condition'], rotation=90, ha='right', fontsize=8)
    axes[0].set_ylabel('Volume [µm³]', fontsize=12, fontweight='bold')
    axes[0].set_title('Mean Volume by Condition', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x_pos, summary_df['mean_mass'], yerr=summary_df['std_mass'], alpha=0.7, capsize=5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(summary_df['condition'], rotation=90, ha='right', fontsize=8)
    axes[1].set_ylabel('Total Mass [pg]', fontsize=12, fontweight='bold')
    axes[1].set_title('Mean Mass by Condition', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(x_pos, summary_df['mean_ri'], yerr=summary_df['std_ri'], alpha=0.7, capsize=5)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(summary_df['condition'], rotation=90, ha='right', fontsize=8)
    axes[2].set_ylabel('Mean RI', fontsize=12, fontweight='bold')
    axes[2].set_title('Mean RI by Condition', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'condition_comparison_bars.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: condition_comparison_bars.png")
    plt.close()
    
    # 3. ヒートマップ（shape × subpixel × method）
    plot_heatmaps(summary_df, output_dir)
    
    return summary_df

def plot_heatmaps(summary_df, output_dir):
    """条件パラメータごとのヒートマップ"""
    
    # Continuousモードのヒートマップ
    continuous_df = summary_df[summary_df['thickness_mode'] == 'continuous']
    if len(continuous_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for metric_idx, (metric, label) in enumerate([
            ('mean_volume', 'Volume [µm³]'),
            ('mean_mass', 'Mass [pg]'),
            ('mean_ri', 'RI')
        ]):
            pivot = continuous_df.pivot(index='shape', columns='subpixel', values=metric)
            
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis', ax=axes[metric_idx], cbar_kws={'label': label})
            axes[metric_idx].set_title(f'{label} (Continuous Mode)', fontsize=12, fontweight='bold')
            axes[metric_idx].set_xlabel('Subpixel Sampling', fontsize=11)
            axes[metric_idx].set_ylabel('Shape Type', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmap_continuous.png'), dpi=150, bbox_inches='tight')
        print(f"  Saved: heatmap_continuous.png")
        plt.close()
    
    # Discreteモードのヒートマップ（discretize_method × subpixel）
    discrete_df = summary_df[summary_df['thickness_mode'] == 'discrete']
    if len(discrete_df) > 0:
        for shape in discrete_df['shape'].unique():
            shape_df = discrete_df[discrete_df['shape'] == shape]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for metric_idx, (metric, label) in enumerate([
                ('mean_volume', 'Volume [µm³]'),
                ('mean_mass', 'Mass [pg]'),
                ('mean_ri', 'RI')
            ]):
                pivot = shape_df.pivot(index='discretize_method', columns='subpixel', values=metric)
                
                sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis', ax=axes[metric_idx], cbar_kws={'label': label})
                axes[metric_idx].set_title(f'{label} (Discrete Mode, {shape})', fontsize=12, fontweight='bold')
                axes[metric_idx].set_xlabel('Subpixel Sampling', fontsize=11)
                axes[metric_idx].set_ylabel('Discretize Method', fontsize=11)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'heatmap_discrete_{shape}.png'), dpi=150, bbox_inches='tight')
            print(f"  Saved: heatmap_discrete_{shape}.png")
            plt.close()

def main():
    """メイン実行"""
    print("="*80)
    print("Condition Comparison Analysis")
    print("="*80)
    
    # データ読み込み
    df = load_all_conditions()
    
    if df is None:
        print("No data found!")
        return
    
    # 比較プロット生成
    summary_df = plot_comparison_by_condition(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: condition_comparison/")
    print(f"  - all_conditions_timeseries.png")
    print(f"  - condition_comparison_bars.png")
    print(f"  - condition_summary_statistics.csv")
    print(f"  - heatmap_*.png")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

# %%





