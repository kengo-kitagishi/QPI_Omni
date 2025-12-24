#!/usr/bin/env python3
"""
バッチ解析：全パラメータ組み合わせを網羅的に実行

実行する組み合わせ:
1. ellipse + subpixel1
2. ellipse + subpixel5
3. ellipse + subpixel10
4. feret + subpixel1
5. feret + subpixel5
6. feret + subpixel10
"""
# %%
import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from PIL import Image
import glob
import re
import tifffile
from scipy import ndimage

# 24_elip_volume.pyからTimeSeriesDensityMapperクラスをコピー
# （インポートの問題を避けるため、直接定義）

class TimeSeriesDensityMapper:
    """時系列画像とResults.csvから屈折率（RI）マップを生成"""
    
    # ... (24_elip_volume.pyのTimeSeriesDensityMapperクラス全体をコピー)
    # スペースの都合上、exec()を使用してインポート
    pass

def run_analysis(shape_type, subpixel_sampling, results_csv, image_directory, 
                 wavelength_nm, n_medium, pixel_size_um, alpha_ri, max_rois, csv_suffix=None):
    """
    指定されたパラメータで解析を実行
    
    Parameters:
    -----------
    csv_suffix : str, optional
        出力フォルダ名に追加するサフィックス。デフォルト: None
        Noneの場合、CSVファイル名から自動抽出
    """
    print(f"\n{'='*80}")
    print(f"Starting analysis:")
    print(f"  Shape type: {shape_type}")
    print(f"  Subpixel sampling: {subpixel_sampling}×{subpixel_sampling}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # 24_elip_volume.pyをexec()で実行
        # グローバル変数として渡す
        globals_dict = {
            '__name__': '__main__',
            'RESULTS_CSV': results_csv,
            'IMAGE_DIRECTORY': image_directory,
            'WAVELENGTH_NM': wavelength_nm,
            'N_MEDIUM': n_medium,
            'PIXEL_SIZE_UM': pixel_size_um,
            'ALPHA_RI': alpha_ri,
            'SHAPE_TYPE': shape_type,
            'SUBPIXEL_SAMPLING': subpixel_sampling,
            'MAX_ROIS': max_rois,
            'CSV_SUFFIX': csv_suffix,
        }
        
        # 24_elip_volume.pyの内容を読み込んで実行
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, '24_elip_volume.py')
        
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # if __name__ == "__main__"の部分を強制実行するため、置き換え
        code = code.replace('if __name__ == "__main__":', 'if True:')
        
        exec(code, globals_dict)
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✅ Completed: {shape_type} + subpixel{subpixel_sampling}")
        print(f"   Elapsed time: {elapsed_time/60:.1f} minutes")
        print(f"{'='*80}\n")
        success = True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ ERROR: {shape_type} + subpixel{subpixel_sampling}")
        print(f"   {str(e)}")
        print(f"   Elapsed time: {elapsed_time/60:.1f} minutes")
        import traceback
        traceback.print_exc()
        success = False
    
    return success


# ===== メイン実行 =====
if __name__ == "__main__":
    # 共通パラメータ
    # === 複数のCSVファイルを処理する場合はリストで指定 ===
    RESULTS_CSVS = [
        r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results_enlarge.csv",
        r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results_enlarge_interpolate.csv"
    ]
    
    IMAGE_DIRECTORY = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted"
    
    WAVELENGTH_NM = 663
    N_MEDIUM = 1.333
    PIXEL_SIZE_UM = 0.348
    ALPHA_RI = 0.00018
    
    MAX_ROIS = 5  # テスト実行（Noneで全ROI）
    
    # === CSVサフィックス（出力フォルダ名の識別用）===
    # オプション1: 自動抽出（Noneを指定）
    #   - CSVファイル名から自動で抽出されます
    #   - Results_enlarge.csv → 'enlarge'
    #   - Results_enlarge_interpolate.csv → 'enlarge_interpolate'
    # オプション2: 手動指定（文字列を指定）
    #   - 例: CSV_SUFFIX = 'my_custom_name'
    CSV_SUFFIX = None  # Noneで自動抽出、または手動で文字列を指定
    
    # パラメータの組み合わせ
    SHAPE_TYPES = ['ellipse', 'feret']
    SUBPIXEL_SAMPLINGS = [1, 5, 10]
    
    # バッチ実行開始
    total_combos = len(RESULTS_CSVS) * len(SHAPE_TYPES) * len(SUBPIXEL_SAMPLINGS)
    
    print(f"\n{'#'*80}")
    print(f"# BATCH ANALYSIS START")
    print(f"# Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Total combinations: {total_combos}")
    print(f"#   CSV files: {len(RESULTS_CSVS)}")
    print(f"#   Shape types: {len(SHAPE_TYPES)}")
    print(f"#   Subpixel samplings: {len(SUBPIXEL_SAMPLINGS)}")
    print(f"{'#'*80}\n")
    
    total_start_time = time.time()
    results = []
    combo_num = 0
    
    # 全組み合わせを実行
    for csv_idx, results_csv in enumerate(RESULTS_CSVS, 1):
        csv_name = os.path.basename(results_csv)
        print(f"\n{'='*80}")
        print(f"Processing CSV {csv_idx}/{len(RESULTS_CSVS)}: {csv_name}")
        print(f"{'='*80}\n")
        
        for i, shape_type in enumerate(SHAPE_TYPES, 1):
            for j, subpixel_sampling in enumerate(SUBPIXEL_SAMPLINGS, 1):
                combo_num += 1
                
                print(f"\n{'#'*80}")
                print(f"# Combination {combo_num}/{total_combos}")
                print(f"#   CSV: {csv_name}")
                print(f"#   Shape: {shape_type}")
                print(f"#   Subpixel: {subpixel_sampling}×{subpixel_sampling}")
                print(f"{'#'*80}")
                
                success = run_analysis(
                    shape_type=shape_type,
                    subpixel_sampling=subpixel_sampling,
                    results_csv=results_csv,
                    image_directory=IMAGE_DIRECTORY,
                    wavelength_nm=WAVELENGTH_NM,
                    n_medium=N_MEDIUM,
                    pixel_size_um=PIXEL_SIZE_UM,
                    alpha_ri=ALPHA_RI,
                    max_rois=MAX_ROIS,
                    csv_suffix=CSV_SUFFIX
                )
                
                results.append({
                    'csv_file': csv_name,
                    'shape_type': shape_type,
                    'subpixel_sampling': subpixel_sampling,
                    'success': success
                })
    
    # 最終サマリー
    total_elapsed = time.time() - total_start_time
    
    print(f"\n{'#'*80}")
    print(f"# BATCH ANALYSIS COMPLETE")
    print(f"# End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Total elapsed time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"{'#'*80}\n")
    
    print(f"Results summary:")
    print(f"{'='*80}")
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        csv_short = result['csv_file'].replace('Results_', '').replace('.csv', '')
        print(f"  {csv_short:20s} | {result['shape_type']:8s} | subpixel{result['subpixel_sampling']:2d} : {status}")
    print(f"{'='*80}\n")
    
    success_count = sum(1 for r in results if r['success'])
    print(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    print(f"\n{'#'*80}")
    print(f"# All output directories:")
    print(f"{'#'*80}")
    
    # CSVファイルごとにグループ化して表示
    for results_csv in RESULTS_CSVS:
        csv_name = os.path.basename(results_csv)
        csv_short = csv_name.replace('Results_', '').replace('.csv', '')
        print(f"\n  [{csv_short}]")
        
        for result in results:
            if result['csv_file'] == csv_name:
                # CSVファイル名から自動抽出されるサフィックスを推定
                csv_name_without_ext = os.path.splitext(csv_name)[0]
                if '_' in csv_name_without_ext:
                    parts = csv_name_without_ext.split('_', 1)
                    csv_suffix_auto = parts[1] if len(parts) > 1 and parts[1] else None
                else:
                    csv_suffix_auto = None
                
                if csv_suffix_auto:
                    dir_suffix = f"{result['shape_type']}_subpixel{result['subpixel_sampling']}_{csv_suffix_auto}"
                else:
                    dir_suffix = f"{result['shape_type']}_subpixel{result['subpixel_sampling']}"
                
                print(f"    - timeseries_density_output_{dir_suffix}/")
                print(f"    - timeseries_plots_{dir_suffix}/")
    print(f"\n{'#'*80}\n")

# %%
