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
                 wavelength_nm, n_medium, pixel_size_um, alpha_ri, max_rois):
    """
    指定されたパラメータで解析を実行
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
    RESULTS_CSV = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results_enlarge_interpolate.csv"
    IMAGE_DIRECTORY = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted"
    
    WAVELENGTH_NM = 663
    N_MEDIUM = 1.333
    PIXEL_SIZE_UM = 0.348
    ALPHA_RI = 0.00018
    
    MAX_ROIS = 5  # テスト実行（Noneで全ROI）
    
    # パラメータの組み合わせ
    SHAPE_TYPES = ['ellipse', 'feret']
    SUBPIXEL_SAMPLINGS = [1, 5, 10]
    
    # バッチ実行開始
    print(f"\n{'#'*80}")
    print(f"# BATCH ANALYSIS START")
    print(f"# Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Total combinations: {len(SHAPE_TYPES) * len(SUBPIXEL_SAMPLINGS)}")
    print(f"{'#'*80}\n")
    
    total_start_time = time.time()
    results = []
    
    # 全組み合わせを実行
    for i, shape_type in enumerate(SHAPE_TYPES, 1):
        for j, subpixel_sampling in enumerate(SUBPIXEL_SAMPLINGS, 1):
            combo_num = (i-1) * len(SUBPIXEL_SAMPLINGS) + j
            total_combos = len(SHAPE_TYPES) * len(SUBPIXEL_SAMPLINGS)
            
            print(f"\n{'#'*80}")
            print(f"# Combination {combo_num}/{total_combos}")
            print(f"{'#'*80}")
            
            success = run_analysis(
                shape_type=shape_type,
                subpixel_sampling=subpixel_sampling,
                results_csv=RESULTS_CSV,
                image_directory=IMAGE_DIRECTORY,
                wavelength_nm=WAVELENGTH_NM,
                n_medium=N_MEDIUM,
                pixel_size_um=PIXEL_SIZE_UM,
                alpha_ri=ALPHA_RI,
                max_rois=MAX_ROIS
            )
            
            results.append({
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
        print(f"  {result['shape_type']:8s} + subpixel{result['subpixel_sampling']:2d} : {status}")
    print(f"{'='*80}\n")
    
    success_count = sum(1 for r in results if r['success'])
    print(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    print(f"\n{'#'*80}")
    print(f"# All output directories:")
    print(f"{'#'*80}")
    for result in results:
        dir_suffix = f"{result['shape_type']}_subpixel{result['subpixel_sampling']}"
        print(f"  - timeseries_density_output_{dir_suffix}/")
        print(f"  - timeseries_plots_{dir_suffix}/")
    print(f"{'#'*80}\n")

# %%
