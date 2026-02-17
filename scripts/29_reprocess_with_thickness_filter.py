#!/usr/bin/env python3
"""
既存の結果から1ピクセル未満の厚みをマスク外として再処理

既存のz-stack TIFFファイルとphase画像から、
min_thickness_px=1.0 でフィルタリングして再計算
"""
# %%
import os
import sys
import glob
import pandas as pd
import numpy as np
import tifffile
import argparse
from tqdm import tqdm

def reprocess_condition(condition_dir, min_thickness_px=1.0, 
                        pixel_size_um=0.348, wavelength_nm=663, n_medium=1.333, alpha_ri=0.00018,
                        voxel_z_um=0.3):
    """
    1つの条件ディレクトリを再処理
    
    Parameters
    ----------
    condition_dir : str
        timeseries_density_output_* ディレクトリ
    min_thickness_px : float
        最小厚み閾値（ピクセル単位）
    pixel_size_um : float
        ピクセルサイズ（µm）
    wavelength_nm : float
        波長（nm）
    n_medium : float
        培地の屈折率
    alpha_ri : float
        比屈折率増分（ml/mg）
    voxel_z_um : float
        Z方向のボクセルサイズ（µm、discreteモード用）
    """
    print(f"\nProcessing: {os.path.basename(condition_dir)}")
    
    # 出力ディレクトリ（元のディレクトリ名 + _filtered）
    output_dir = condition_dir + f"_filtered_{min_thickness_px}px"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "density_tiff"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "csv_data"), exist_ok=True)
    
    # z-stack TIFFファイルを検索
    zstack_files = glob.glob(os.path.join(condition_dir, "density_tiff", "*_zstack.tif"))
    phase_files = glob.glob(os.path.join(condition_dir, "density_tiff", "*_phase.tif"))
    
    if len(zstack_files) == 0:
        print(f"  No z-stack files found in {condition_dir}")
        return None
    
    print(f"  Found {len(zstack_files)} z-stack files")
    
    # thickness_modeを推定
    dirname = os.path.basename(condition_dir)
    if 'discrete' in dirname:
        thickness_mode = 'discrete'
        # voxel_z_umは引数から使用
    else:
        thickness_mode = 'continuous'
        # continuousモードではvoxel_z_umは使わないが、念のため
    
    wavelength_um = wavelength_nm / 1000.0
    
    summary_data = []
    
    for zstack_file in tqdm(zstack_files, desc="  Processing files"):
        roi_str = os.path.basename(zstack_file).replace('_zstack.tif', '')
        
        # 対応するphaseファイルを探す
        phase_file = zstack_file.replace('_zstack.tif', '_phase.tif')
        
        if not os.path.exists(phase_file):
            print(f"    Warning: Phase file not found for {roi_str}")
            continue
        
        # z-stackとphase画像を読み込み
        zstack_map_original = tifffile.imread(zstack_file).astype(np.float32)
        phase_img = tifffile.imread(phase_file).astype(np.float32)
        
        # 最小厚み閾値フィルタリング
        if thickness_mode == 'discrete':
            # スライス数をピクセル単位に換算して閾値判定
            thickness_px_for_threshold = zstack_map_original * (voxel_z_um / pixel_size_um)
        else:
            thickness_px_for_threshold = zstack_map_original
        
        # 閾値未満を0にする
        zstack_map_filtered = np.where(
            thickness_px_for_threshold >= min_thickness_px,
            zstack_map_original,
            0
        )
        
        # マスク
        mask_original = zstack_map_original > 0
        mask_filtered = zstack_map_filtered > 0
        
        pixels_before = np.count_nonzero(mask_original)
        pixels_after = np.count_nonzero(mask_filtered)
        
        if pixels_after == 0:
            print(f"    Warning: No pixels remain after filtering for {roi_str}")
            continue
        
        # 厚みをµm単位に変換
        if thickness_mode == 'discrete':
            thickness_um = zstack_map_filtered * voxel_z_um
        else:
            thickness_um = zstack_map_filtered * pixel_size_um
        
        # RI計算
        ri_map = np.full_like(phase_img, n_medium, dtype=np.float64)
        thickness_um_safe = np.where(thickness_um > 0, thickness_um, np.nan)
        ri_map[mask_filtered] = n_medium + (phase_img[mask_filtered] * wavelength_um) / (2 * np.pi * thickness_um_safe[mask_filtered])
        
        # 質量濃度計算
        concentration_map = np.zeros_like(ri_map)
        concentration_map[mask_filtered] = (ri_map[mask_filtered] - n_medium) / alpha_ri
        
        # フィルタリングされた領域（mask外）を明示的にクリア
        ri_map[~mask_filtered] = n_medium
        concentration_map[~mask_filtered] = 0.0
        
        # 体積計算
        pixel_area_um2 = pixel_size_um ** 2
        volume_um3 = np.sum(thickness_um[mask_filtered]) * pixel_area_um2
        
        # Total mass計算
        pixel_volumes = thickness_um[mask_filtered] * pixel_area_um2
        total_mass_pg = np.sum(concentration_map[mask_filtered] * pixel_volumes)
        
        # 統計情報
        stats = {
            'roi_str': roi_str,
            'pixels_before': pixels_before,
            'pixels_after': pixels_after,
            'pixels_filtered': pixels_before - pixels_after,
            'filter_ratio': (pixels_before - pixels_after) / pixels_before * 100 if pixels_before > 0 else 0,
            'zstack_max': float(np.max(zstack_map_filtered)),
            'zstack_mean': float(np.mean(zstack_map_filtered[mask_filtered])),
            'thickness_mean_um': float(np.mean(thickness_um[mask_filtered])),
            'thickness_max_um': float(np.max(thickness_um[mask_filtered])),
            'volume_um3': float(volume_um3),
            'total_mass_pg': float(total_mass_pg),
            'ri_mean': float(np.mean(ri_map[mask_filtered])),
            'ri_median': float(np.median(ri_map[mask_filtered])),
            'concentration_mean': float(np.mean(concentration_map[mask_filtered])),
        }
        
        summary_data.append(stats)
        
        # フィルタリング後のデータを保存
        tifffile.imwrite(os.path.join(output_dir, "density_tiff", f"{roi_str}_zstack.tif"), zstack_map_filtered.astype(np.float32))
        tifffile.imwrite(os.path.join(output_dir, "density_tiff", f"{roi_str}_ri.tif"), ri_map.astype(np.float32))
        tifffile.imwrite(os.path.join(output_dir, "density_tiff", f"{roi_str}_concentration.tif"), concentration_map.astype(np.float32))
        tifffile.imwrite(os.path.join(output_dir, "density_tiff", f"{roi_str}_phase.tif"), phase_img.astype(np.float32))
        
        # CSVデータを保存
        y_coords, x_coords = np.where(mask_filtered)
        
        if thickness_mode == 'discrete':
            z_column_name = 'Z_slice_count'
        else:
            z_column_name = 'Z_thickness_pixel'
        
        pixel_data = pd.DataFrame({
            'X_pixel': x_coords,
            'Y_pixel': y_coords,
            z_column_name: zstack_map_filtered[mask_filtered],
            'Thickness_um': thickness_um[mask_filtered],
            'Phase_value': phase_img[mask_filtered],
            'Refractive_Index': ri_map[mask_filtered],
            'Delta_RI': ri_map[mask_filtered] - n_medium,
            'Concentration_mg_ml': concentration_map[mask_filtered],
        })
        
        csv_path = os.path.join(output_dir, "csv_data", f"{roi_str}_pixel_data.csv")
        pixel_data.to_csv(csv_path, index=False)
    
    if len(summary_data) > 0:
        # サマリーCSVを保存
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(output_dir, "filtering_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"  Processed {len(summary_data)} files")
        print(f"  Mean filter ratio: {summary_df['filter_ratio'].mean():.2f}%")
        print(f"  Output: {output_dir}")
        
        return summary_df
    else:
        print(f"  No files processed")
        return None

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(
        description='既存の結果から1ピクセル未満の厚みをマスク外として再処理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # カレントディレクトリの全条件を処理
  python 29_reprocess_with_thickness_filter.py
  
  # 特定のディレクトリ内の全条件を処理
  python 29_reprocess_with_thickness_filter.py -d G:\\test_dens_est
  
  # 特定の条件のみ処理
  python 29_reprocess_with_thickness_filter.py -c timeseries_density_output_ellipse_subpixel5
  
  # 複数の条件を処理
  python 29_reprocess_with_thickness_filter.py -c *ellipse*subpixel5*
  
  # パラメータを指定
  python 29_reprocess_with_thickness_filter.py -t 0.5 --voxel-z 0.3
  
  # 確認なしで実行
  python 29_reprocess_with_thickness_filter.py -y
"""
    )
    
    parser.add_argument('-d', '--base-dir', type=str, default='.',
                        help='基準ディレクトリ（デフォルト: カレントディレクトリ）')
    parser.add_argument('-c', '--conditions', type=str, nargs='*', default=None,
                        help='処理する条件ディレクトリ（ワイルドカード可）。指定しない場合は全条件')
    parser.add_argument('-t', '--min-thickness', type=float, default=1.0,
                        help='最小厚み閾値（ピクセル単位、デフォルト: 1.0）')
    parser.add_argument('--pixel-size', type=float, default=0.348,
                        help='ピクセルサイズ（µm、デフォルト: 0.348）')
    parser.add_argument('--wavelength', type=float, default=663,
                        help='波長（nm、デフォルト: 663）')
    parser.add_argument('--n-medium', type=float, default=1.333,
                        help='培地の屈折率（デフォルト: 1.333）')
    parser.add_argument('--alpha-ri', type=float, default=0.00018,
                        help='比屈折率増分（ml/mg、デフォルト: 0.00018）')
    parser.add_argument('--voxel-z', type=float, default=0.3,
                        help='Z方向のボクセルサイズ（µm、デフォルト: 0.3）')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='確認なしで実行')
    parser.add_argument('--list-only', action='store_true',
                        help='条件リストのみ表示して終了')
    
    # Jupyter環境での実行に対応
    if 'ipykernel' in sys.modules:
        # Jupyter環境の場合、sys.argvから--f引数を除外
        filtered_argv = [arg for arg in sys.argv if not arg.startswith('--f=') and not arg.startswith('-f=')]
        args = parser.parse_args(filtered_argv[1:] if len(filtered_argv) > 1 else [])
    else:
        args = parser.parse_args()
    
    print("="*80)
    print("Re-process with Thickness Filter")
    print("="*80)
    
    # パラメータ
    MIN_THICKNESS_PX = args.min_thickness
    PIXEL_SIZE_UM = args.pixel_size
    WAVELENGTH_NM = args.wavelength
    N_MEDIUM = args.n_medium
    ALPHA_RI = args.alpha_ri
    VOXEL_Z_UM = args.voxel_z
    
    print(f"\nParameters:")
    print(f"  Base directory: {os.path.abspath(args.base_dir)}")
    print(f"  Min thickness: {MIN_THICKNESS_PX} px")
    print(f"  Pixel size: {PIXEL_SIZE_UM} µm")
    print(f"  Wavelength: {WAVELENGTH_NM} nm")
    print(f"  Medium RI: {N_MEDIUM}")
    print(f"  Alpha RI: {ALPHA_RI} ml/mg")
    print(f"  Voxel Z: {VOXEL_Z_UM} µm")
    
    # 条件ディレクトリを検索
    if args.conditions:
        # 指定された条件のみ
        condition_dirs = []
        for pattern in args.conditions:
            if not os.path.isabs(pattern):
                pattern = os.path.join(args.base_dir, pattern)
            matched = glob.glob(pattern)
            # ディレクトリのみ
            matched = [d for d in matched if os.path.isdir(d)]
            condition_dirs.extend(matched)
        # 重複削除
        condition_dirs = list(set(condition_dirs))
    else:
        # 全条件を自動検索
        pattern = os.path.join(args.base_dir, 'timeseries_density_output_*')
        condition_dirs = glob.glob(pattern)
    
    # すでにフィルタリング済みのディレクトリを除外
    condition_dirs = [d for d in condition_dirs if '_filtered_' not in d and os.path.isdir(d)]
    condition_dirs = sorted(condition_dirs)
    
    print(f"\nFound {len(condition_dirs)} condition directories")
    
    if len(condition_dirs) == 0:
        print("No condition directories found!")
        print(f"Search pattern: {pattern if not args.conditions else args.conditions}")
        return
    
    # 条件リスト表示
    print("\nConditions to process:")
    for i, d in enumerate(condition_dirs, 1):
        print(f"  {i}. {os.path.basename(d)}")
    
    if args.list_only:
        print("\n(--list-only mode: exiting)")
        return
    
    # ユーザーに確認
    if not args.yes:
        response = input(f"\nProcess all {len(condition_dirs)} conditions? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    
    # 全条件を処理
    all_summaries = []
    
    for condition_dir in condition_dirs:
        try:
            summary_df = reprocess_condition(
                condition_dir,
                min_thickness_px=MIN_THICKNESS_PX,
                pixel_size_um=PIXEL_SIZE_UM,
                wavelength_nm=WAVELENGTH_NM,
                n_medium=N_MEDIUM,
                alpha_ri=ALPHA_RI,
                voxel_z_um=VOXEL_Z_UM
            )
            
            if summary_df is not None:
                summary_df['condition_dir'] = os.path.basename(condition_dir)
                all_summaries.append(summary_df)
        except Exception as e:
            print(f"  Error processing {condition_dir}: {e}")
            import traceback
            traceback.print_exc()
    
    if len(all_summaries) > 0:
        # 全条件のサマリーを結合
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_summary.to_csv('reprocessed_all_conditions_summary.csv', index=False)
        
        print("\n" + "="*80)
        print("REPROCESSING COMPLETE!")
        print("="*80)
        print(f"\nProcessed {len(all_summaries)} conditions")
        print(f"Total files: {len(combined_summary)}")
        print(f"Mean filter ratio: {combined_summary['filter_ratio'].mean():.2f}%")
        print(f"\nOutput: reprocessed_all_conditions_summary.csv")
        print("\nFiltered directories: *_filtered_1.0px/")
    else:
        print("\nNo data processed!")

if __name__ == "__main__":
    main()

# %%

