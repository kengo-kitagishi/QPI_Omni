#!/usr/bin/env python3
"""
シンプルなmean RI計算: total_phase / ellipse理論体積

ROIのshape parameters（Feret直径、bounding box、面積など）から
ellipse/rod shapeの理論体積を計算し、全位相をその体積で割って
平均屈折率を求める単純な方法。

mean_RI = n_medium + (total_phase × λ) / (2π × volume)

ここで:
- total_phase: マスク内の全ピクセルの位相値の合計 (rad)
- volume: ellipse/rod shapeの理論体積 (µm³)
- λ: 波長 (nm → µm)
- n_medium: 培地の屈折率
"""
# %%
import os
import sys
import glob
import pandas as pd
import numpy as np
import tifffile
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import zipfile
import struct

# %%
def calculate_ellipse_volume(major, minor, pixel_size_um, shape_type='rod'):
    """
    楕円/rod shapeの理論体積を計算
    
    Parameters
    ----------
    major : float
        長軸（ピクセル単位）
    minor : float
        短軸（ピクセル単位）
    pixel_size_um : float
        ピクセルサイズ (µm)
    shape_type : str
        'rod': カプセル型（2つの半球 + 円柱）
        'ellipsoid': 回転楕円体
    
    Returns
    -------
    volume_um3 : float
        体積 (µm³)
    """
    # ピクセル単位からµmに変換
    length_um = major * pixel_size_um
    width_um = minor * pixel_size_um
    
    if shape_type == 'rod':
        # Rod shape: 2つの半球 + 円柱
        r_um = width_um / 2.0
        h_um = length_um - 2 * r_um
        
        if h_um < 0:
            # 円柱部分がない場合（球）
            volume_um3 = (4.0 / 3.0) * np.pi * (r_um ** 3)
        else:
            # 2つの半球（= 1つの球）+ 円柱
            volume_um3 = (4.0 / 3.0) * np.pi * (r_um ** 3) + np.pi * (r_um ** 2) * h_um
    else:  # ellipsoid
        # 回転楕円体: V = (4/3)π × a × b × c
        # 長軸周りの回転楕円体と仮定
        a_um = length_um / 2.0
        b_um = width_um / 2.0
        c_um = width_um / 2.0
        volume_um3 = (4.0 / 3.0) * np.pi * a_um * b_um * c_um
    
    return volume_um3

# %%
def create_simple_mask(roi_params, image_shape, pixel_size_um):
    """
    ROIパラメータから簡易的なマスクを作成
    
    Parameters
    ----------
    roi_params : dict
        ROIのパラメータ（X, Y, Major, Minor, Angleなど）
    image_shape : tuple
        画像サイズ (height, width)
    pixel_size_um : float
        ピクセルサイズ (µm)
    
    Returns
    -------
    mask : ndarray
        マスク（2D boolean array）
    """
    height, width = image_shape
    y_coords, x_coords = np.ogrid[:height, :width]
    
    # 楕円のパラメータ
    center_x = roi_params['X']
    center_y = roi_params['Y']
    major = roi_params.get('Major', roi_params.get('Feret'))
    minor = roi_params.get('Minor', roi_params.get('MinFeret'))
    angle_deg = roi_params.get('Angle', roi_params.get('FeretAngle', 0))
    
    # 角度をラジアンに変換
    angle_rad = np.deg2rad(angle_deg)
    
    # 座標変換
    dx = x_coords - center_x
    dy = y_coords - center_y
    
    # 回転を考慮
    x_rot = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
    y_rot = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    
    # 楕円内部判定
    a = major / 2.0
    b = minor / 2.0
    mask = ((x_rot / a) ** 2 + (y_rot / b) ** 2) <= 1.0
    
    return mask

# %%
def calculate_simple_mean_ri(phase_map, roi_params, pixel_size_um, 
                              wavelength_nm, n_medium, shape_type='rod'):
    """
    シンプルなmean RI計算: total_phase / ellipse理論体積
    
    Parameters
    ----------
    phase_map : ndarray
        位相マップ (rad)
    roi_params : dict
        ROIのパラメータ（Major, Minor, X, Y, Angleなど）
    pixel_size_um : float
        ピクセルサイズ (µm)
    wavelength_nm : float
        波長 (nm)
    n_medium : float
        培地の屈折率
    shape_type : str
        'rod': カプセル型（推奨）
        'ellipsoid': 回転楕円体
    
    Returns
    -------
    mean_ri : float
        平均屈折率
    volume_um3 : float
        ellipse理論体積 (µm³)
    total_phase : float
        全位相の合計 (rad)
    """
    # ellipse理論体積を計算
    major = roi_params.get('Major', roi_params.get('Feret'))
    minor = roi_params.get('Minor', roi_params.get('MinFeret'))
    volume_um3 = calculate_ellipse_volume(major, minor, pixel_size_um, shape_type)
    
    if volume_um3 == 0:
        return n_medium, 0.0, 0.0
    
    # マスクを作成
    mask = create_simple_mask(roi_params, phase_map.shape, pixel_size_um)
    
    # マスク内の全位相の合計
    total_phase = np.sum(phase_map[mask])
    
    # mean RI計算
    # n_sample = n_medium + (φ × λ) / (2π × thickness)
    # 全体では: mean_RI = n_medium + (total_φ × λ) / (2π × volume / pixel_area)
    #          = n_medium + (total_φ × λ × pixel_area) / (2π × volume)
    
    wavelength_um = wavelength_nm * 1e-3  # nm → µm
    pixel_area_um2 = pixel_size_um ** 2
    
    mean_ri = n_medium + (total_phase * wavelength_um * pixel_area_um2) / (2 * np.pi * volume_um3)
    
    return mean_ri, volume_um3, total_phase

# %%
def find_results_csv(base_dir):
    """
    Results.csvファイルを検索
    
    Parameters
    ----------
    base_dir : str
        検索基準ディレクトリ
    
    Returns
    -------
    results_csv : str or None
        Results.csvのパス
    """
    # いくつかの典型的な場所を検索
    candidates = [
        os.path.join(base_dir, 'Results.csv'),
        os.path.join(base_dir, '..', 'Results.csv'),
        os.path.join(base_dir, '..', '..', 'Results.csv'),
    ]
    
    # ワイルドカード検索
    patterns = [
        os.path.join(base_dir, 'Results*.csv'),
        os.path.join(base_dir, '..', 'Results*.csv'),
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None

# %%
def find_phase_image(image_dir, roi_label, frame_num):
    """
    対応する位相画像を検索
    
    Parameters
    ----------
    image_dir : str
        画像ディレクトリ
    roi_label : str
        ROIラベル（例: "ROI_0000"）
    frame_num : int
        フレーム番号
    
    Returns
    -------
    phase_file : str or None
        位相画像のパス
    """
    # 複数のパターンを試す
    patterns = [
        os.path.join(image_dir, f'*{frame_num:04d}*.tif'),
        os.path.join(image_dir, f'*Frame_{frame_num:04d}*.tif'),
        os.path.join(image_dir, f'{frame_num:04d}.tif'),
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None

# %%
def process_from_results_csv(results_csv, image_dir, pixel_size_um=0.348, 
                              wavelength_nm=663, n_medium=1.333, 
                              alpha_ri=0.00018, shape_type='rod'):
    """
    Results.csvから直接処理
    
    Parameters
    ----------
    results_csv : str
        Results.csvのパス
    image_dir : str
        位相画像ディレクトリ
    pixel_size_um : float
        ピクセルサイズ (µm)
    wavelength_nm : float
        波長 (nm)
    n_medium : float
        培地の屈折率
    alpha_ri : float
        比屈折率増分 (ml/mg)
    shape_type : str
        'rod': カプセル型（推奨）
        'ellipsoid': 回転楕円体
    
    Returns
    -------
    summary_df : DataFrame
        ROIごとのサマリー
    """
    print(f"\n  Reading: {os.path.basename(results_csv)}")
    
    # Results.csvを読み込み
    df = pd.read_csv(results_csv)
    
    # 必要なカラムをチェック
    required_cols = ['X', 'Y']
    shape_cols = [['Major', 'Minor'], ['Feret', 'MinFeret']]
    
    has_shape = False
    for cols in shape_cols:
        if all(col in df.columns for col in cols):
            has_shape = True
            break
    
    if not has_shape:
        print(f"  Error: Required shape columns not found in {results_csv}")
        print(f"  Available columns: {df.columns.tolist()}")
        return None
    
    # 結果を格納
    results = []
    
    # 画像リストを取得
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
    if len(image_files) == 0:
        print(f"  Error: No TIFF files found in {image_dir}")
        return None
    
    print(f"  Found {len(image_files)} phase images")
    print(f"  Found {len(df)} ROIs in CSV")
    
    # 最初の画像でサイズを取得
    first_image = Image.open(image_files[0])
    image_shape = (first_image.height, first_image.width)
    print(f"  Image size: {image_shape}")
    
    # 各ROIを処理
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Processing"):
        # ROI情報
        roi_params = row.to_dict()
        
        # Frameまたはラベル情報を取得
        if 'Label' in row:
            label = row['Label']
            # Frame番号を抽出
            import re
            frame_match = re.search(r'(\d{4})', label)
            if frame_match:
                frame_num = int(frame_match.group(1))
            else:
                frame_num = idx + 1
        elif 'Slice' in row:
            frame_num = int(row['Slice'])
        else:
            frame_num = idx + 1
        
        roi_name = f"ROI_{idx:04d}"
        frame_name = f"Frame_{frame_num:04d}"
        
        # 対応する画像を検索
        if frame_num <= len(image_files):
            phase_file = image_files[frame_num - 1]
        else:
            print(f"    Warning: Frame {frame_num} not found")
            continue
        
        # 位相画像を読み込み
        phase_map = np.array(Image.open(phase_file)).astype(np.float64)
        
        # シンプルなmean RI計算
        mean_ri, volume_um3, total_phase = calculate_simple_mean_ri(
            phase_map, roi_params, pixel_size_um, wavelength_nm, n_medium, shape_type
        )
        
        # 質量計算
        mean_concentration_mg_ml = (mean_ri - n_medium) / alpha_ri
        total_mass_pg = mean_concentration_mg_ml * volume_um3
        
        # 結果を保存
        results.append({
            'roi': roi_name,
            'frame': frame_name,
            'frame_num': frame_num,
            'mean_ri': mean_ri,
            'volume_um3': volume_um3,
            'total_phase_rad': total_phase,
            'mean_concentration_mg_ml': mean_concentration_mg_ml,
            'total_mass_pg': total_mass_pg,
            'major': roi_params.get('Major', roi_params.get('Feret')),
            'minor': roi_params.get('Minor', roi_params.get('MinFeret')),
            'shape_type': shape_type
        })
    
    # DataFrameに変換
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values(['roi', 'frame_num']).reset_index(drop=True)
    
    return summary_df

# %%
def plot_timeseries(summary_df, output_dir, condition_name):
    """
    時系列プロットを作成
    
    Parameters
    ----------
    summary_df : DataFrame
        サマリーデータ
    output_dir : str
        出力ディレクトリ
    condition_name : str
        条件名
    """
    # ROIごとにプロット
    rois = summary_df['roi'].unique()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Simple Mean RI Analysis\n{condition_name}', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(rois)))
    
    for roi, color in zip(rois, colors):
        roi_data = summary_df[summary_df['roi'] == roi].sort_values('frame_num')
        
        # Volume
        axes[0].plot(roi_data['frame_num'], roi_data['volume_um3'], 
                    marker='o', label=roi, color=color, linewidth=2, markersize=4)
        
        # Mean RI
        axes[1].plot(roi_data['frame_num'], roi_data['mean_ri'], 
                    marker='s', label=roi, color=color, linewidth=2, markersize=4)
        
        # Total Mass
        axes[2].plot(roi_data['frame_num'], roi_data['total_mass_pg'], 
                    marker='^', label=roi, color=color, linewidth=2, markersize=4)
    
    # 軸ラベルとグリッド
    axes[0].set_ylabel('Volume (µm³)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    axes[1].set_ylabel('Mean RI (simple)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    axes[2].set_xlabel('Frame Number', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Total Mass (pg)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # 保存
    plot_file = os.path.join(output_dir, 'timeseries_simple_mean_ri.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {plot_file}")

# %%
def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(
        description='シンプルなmean RI計算: total_phase / volume',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # カレントディレクトリの全条件を処理
  python 30_simple_mean_ri_analysis.py
  
  # 特定のディレクトリ内の全条件を処理
  python 30_simple_mean_ri_analysis.py -d G:\\test_dens_est
  
  # 特定の条件のみ処理
  python 30_simple_mean_ri_analysis.py -c timeseries_density_output_ellipse_subpixel5
  
  # パラメータを指定
  python 30_simple_mean_ri_analysis.py --wavelength 532 --n-medium 1.335
"""
    )
    
    parser.add_argument('-d', '--base-dir', type=str, default='.',
                        help='基準ディレクトリ（デフォルト: カレントディレクトリ）')
    parser.add_argument('-c', '--conditions', type=str, nargs='*', default=None,
                        help='処理する条件ディレクトリ（ワイルドカード可）。指定しない場合は全条件')
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
    parser.add_argument('--list-only', action='store_true',
                        help='条件リストのみ表示して終了')
    
    # Jupyter環境での実行に対応
    if 'ipykernel' in sys.modules:
        filtered_argv = [arg for arg in sys.argv if not arg.startswith('--f=') and not arg.startswith('-f=')]
        args = parser.parse_args(filtered_argv[1:] if len(filtered_argv) > 1 else [])
    else:
        args = parser.parse_args()
    
    print("="*80)
    print("Simple Mean RI Analysis: total_phase / volume")
    print("="*80)
    
    # パラメータ
    PIXEL_SIZE_UM = args.pixel_size
    WAVELENGTH_NM = args.wavelength
    N_MEDIUM = args.n_medium
    ALPHA_RI = args.alpha_ri
    VOXEL_Z_UM = args.voxel_z
    
    print(f"\nParameters:")
    print(f"  Base directory: {os.path.abspath(args.base_dir)}")
    print(f"  Pixel size: {PIXEL_SIZE_UM} µm")
    print(f"  Wavelength: {WAVELENGTH_NM} nm")
    print(f"  Medium RI: {N_MEDIUM}")
    print(f"  Alpha RI: {ALPHA_RI} ml/mg")
    print(f"  Voxel Z: {VOXEL_Z_UM} µm")
    
    # 条件ディレクトリを検索
    if args.conditions:
        condition_dirs = []
        for pattern in args.conditions:
            if not os.path.isabs(pattern):
                pattern = os.path.join(args.base_dir, pattern)
            matched = glob.glob(pattern)
            matched = [d for d in matched if os.path.isdir(d)]
            condition_dirs.extend(matched)
        condition_dirs = list(set(condition_dirs))
    else:
        pattern = os.path.join(args.base_dir, 'timeseries_density_output_*')
        condition_dirs = glob.glob(pattern)
    
    # フィルタリング済みディレクトリを除外
    condition_dirs = [d for d in condition_dirs if os.path.isdir(d)]
    condition_dirs = sorted(condition_dirs)
    
    print(f"\nFound {len(condition_dirs)} condition directories")
    
    if len(condition_dirs) == 0:
        print("No condition directories found!")
        return
    
    # 条件リスト表示
    print("\nConditions to process:")
    for i, d in enumerate(condition_dirs, 1):
        print(f"  {i}. {os.path.basename(d)}")
    
    if args.list_only:
        print("\n(--list-only mode: exiting)")
        return
    
    # 全条件を処理
    all_summaries = []
    
    for condition_dir in condition_dirs:
        try:
            summary_df = process_condition_simple(
                condition_dir,
                pixel_size_um=PIXEL_SIZE_UM,
                wavelength_nm=WAVELENGTH_NM,
                n_medium=N_MEDIUM,
                alpha_ri=ALPHA_RI,
                voxel_z_um=VOXEL_Z_UM
            )
            
            if summary_df is not None and len(summary_df) > 0:
                # プロット作成
                output_dir = condition_dir.replace('timeseries_density_output_', 'timeseries_plots_') + '_simple_mean_ri'
                plot_timeseries(summary_df, output_dir, os.path.basename(condition_dir))
                
                # 条件名を追加
                summary_df['condition'] = os.path.basename(condition_dir)
                all_summaries.append(summary_df)
                
        except Exception as e:
            print(f"  Error processing {condition_dir}: {e}")
            import traceback
            traceback.print_exc()
    
    # 全条件の統合サマリー
    if len(all_summaries) > 0:
        combined_df = pd.concat(all_summaries, ignore_index=True)
        
        output_file = os.path.join(args.base_dir, 'simple_mean_ri_all_conditions_summary.csv')
        combined_df.to_csv(output_file, index=False)
        print(f"\nSaved combined summary: {output_file}")
        
        print("\n" + "="*80)
        print("Processing complete!")
        print("="*80)
        print(f"\nProcessed {len(all_summaries)} conditions")
        print(f"Total ROIs: {combined_df['roi'].nunique()}")
        print(f"Total frames: {len(combined_df)}")
    else:
        print("\nNo data processed!")

# %%
if __name__ == '__main__':
    main()

