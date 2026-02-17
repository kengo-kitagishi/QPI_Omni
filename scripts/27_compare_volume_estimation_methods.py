#!/usr/bin/env python3
"""
バッチ解析：全パラメータ組み合わせを網羅的に実行

実行する組み合わせ:
【CSVファイル】
  - Results_enlarge.csv
  - Results_enlarge_interpolate.csv

【形状推定】
  - ellipse (楕円)
  - feret (Feret直径)

【サブピクセル精度】
  - 1×1
  - 5×5
  - 10×10

【厚みマップモード】
  - continuous (連続値) ※厚みマップをキャッシュに保存
  - discrete (離散値: round, ceil, floor, pomegranate) ※キャッシュを再利用して高速化

デフォルト設定で全パターン実行：
  2 (CSV) × 2 (形状) × 3 (サブピクセル) × (1 + 4) = 60パターン

【最適化機能】
continuousモードで計算した厚みマップをキャッシュに保存し、
discreteモードではそれを再利用することで計算時間を大幅短縮。
  - continuous: 全計算（位相差画像から厚みマップまで）
  - discrete: キャッシュ読込 → 離散化 → 体積再計算のみ（約10倍高速）
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

def check_if_completed(shape_type, subpixel_sampling, results_csv, thickness_mode, 
                       discretize_method, csv_suffix):
    """
    指定された条件の解析が既に完了しているかチェック
    
    Returns:
    --------
    bool : 完了している場合True
    """
    # 出力フォルダ名を生成
    csv_name = os.path.basename(results_csv)
    csv_name_without_ext = os.path.splitext(csv_name)[0]
    
    if csv_suffix is None:
        # CSVファイル名から自動抽出
        if '_' in csv_name_without_ext:
            parts = csv_name_without_ext.split('_', 1)
            csv_identifier = parts[1] if len(parts) > 1 and parts[1] else None
        else:
            csv_identifier = None
    else:
        csv_identifier = csv_suffix
    
    # フォルダ名生成
    if thickness_mode == 'discrete':
        mode_suffix = f"{thickness_mode}_{discretize_method}"
    else:
        mode_suffix = thickness_mode
    
    if csv_identifier:
        dir_suffix = f"{shape_type}_subpixel{subpixel_sampling}_{csv_identifier}_{mode_suffix}"
    else:
        dir_suffix = f"{shape_type}_subpixel{subpixel_sampling}_{mode_suffix}"
    
    output_dir = f"timeseries_density_output_{dir_suffix}"
    
    # 完了フラグファイルのパス
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flag_file = os.path.join(script_dir, output_dir, '.completed')
    
    return os.path.exists(flag_file)


def run_analysis(shape_type, subpixel_sampling, results_csv, image_directory, 
                 wavelength_nm, n_medium, pixel_size_um, alpha_ri, max_rois, 
                 thickness_mode='continuous', voxel_z_um=0.3, discretize_method='round',
                 min_thickness_px=0.0, csv_suffix=None, skip_if_completed=True):
    """
    指定されたパラメータで解析を実行
    
    Parameters:
    -----------
    thickness_mode : str
        厚みマップのモード。デフォルト: 'continuous'
    voxel_z_um : float
        Z方向のボクセルサイズ（µm）。デフォルト: 0.3
    discretize_method : str
        離散化の方法。デフォルト: 'round'
    min_thickness_px : float
        最小厚み閾値（ピクセル単位）。デフォルト: 0.0
    csv_suffix : str, optional
        出力フォルダ名に追加するサフィックス。デフォルト: None
        Noneの場合、CSVファイル名から自動抽出
    skip_if_completed : bool, optional
        完了済みの場合スキップする。デフォルト: True
    """
    # 完了チェック
    if skip_if_completed:
        if check_if_completed(shape_type, subpixel_sampling, results_csv, 
                             thickness_mode, discretize_method, csv_suffix):
            print(f"\n{'='*80}")
            print(f"⏭️  SKIPPED (already completed):")
            print(f"  Shape type: {shape_type}")
            print(f"  Subpixel sampling: {subpixel_sampling}×{subpixel_sampling}")
            print(f"  Thickness mode: {thickness_mode}")
            if thickness_mode == 'discrete':
                print(f"  Discretize method: {discretize_method}")
            print(f"{'='*80}\n")
            return True  # スキップしたが成功扱い
    
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
            'THICKNESS_MODE': thickness_mode,
            'VOXEL_Z_UM': voxel_z_um,
            'DISCRETIZE_METHOD': discretize_method,
            'MIN_THICKNESS_PX': min_thickness_px,
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
        #r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results_enlarge.csv",
        #r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results_enlarge_interpolate.csv",
        r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results.csv",
        r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results_interpolate.csv"
        ]
    
    IMAGE_DIRECTORY = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted"
    
    WAVELENGTH_NM = 663
    N_MEDIUM = 1.333
    PIXEL_SIZE_UM = 0.348
    ALPHA_RI = 0.00018
    
    MAX_ROIS = None  # テスト実行（Noneで全ROI）
    
    # === 厚みマップパラメータ（discreteモード用）===
    VOXEL_Z_UM = 0.3  # Z方向のボクセルサイズ（µm）
    MIN_THICKNESS_PX = 0.0  # 最小厚み閾値（ピクセル単位、0.0=閾値なし）
    
    # === CSVサフィックス（出力フォルダ名の識別用）===
    # オプション1: 自動抽出（Noneを指定）
    #   - CSVファイル名から自動で抽出されます
    #   - Results_enlarge.csv → 'enlarge'
    #   - Results_enlarge_interpolate.csv → 'enlarge_interpolate'
    # オプション2: 手動指定（文字列を指定）
    #   - 例: CSV_SUFFIX = 'my_custom_name'
    CSV_SUFFIX = None  # Noneで自動抽出、または手動で文字列を指定
    
    # === レジューム機能 ===
    # True: 完了済みの条件をスキップ（クラッシュから再開可能）
    # False: すべて再実行
    SKIP_IF_COMPLETED = True  # ✅ レジューム機能を有効化（推奨）
    
    # ===== パラメータの組み合わせ =====
    # 【形状推定方法】
    # - 'ellipse': 楕円フィッティング
    # - 'feret': Feret直径ベース
    SHAPE_TYPES = ['ellipse', 'feret']  # 両方試す
    # SHAPE_TYPES = ['ellipse']  # 楕円のみ
    # SHAPE_TYPES = ['feret']  # Feretのみ
    
    # 【サブピクセル精度】
    # サブピクセルサンプリング数（N×N）
    SUBPIXEL_SAMPLINGS = [1, 5, 10]  # 全部試す
    # SUBPIXEL_SAMPLINGS = [1]  # 高速テスト用
    # SUBPIXEL_SAMPLINGS = [5, 10]  # 高精度のみ
    
    # 【厚みマップモード】
    # - 'continuous': 連続値（実数値のまま）※先に実行して厚みマップをキャッシュ
    # - 'discrete': 離散値（ボクセル単位に丸める）※キャッシュを再利用して高速化
    # 
    # ⚠️ 重要: continuousを先に実行してください！
    #   discreteモードは、continuousで保存された厚みマップを再利用して高速化します
    THICKNESS_MODES = ['continuous', 'discrete']  # ✅ 全パターン（推奨、continuousが先）
    # THICKNESS_MODES = ['continuous']  # continuousのみ
    # THICKNESS_MODES = ['discrete']  # discreteのみ（要: 事前にcontinuous実行）
    
    # 【離散化方法】（discreteモードのみで使用）
    # - 'round': 四捨五入
    # - 'ceil': 切り上げ
    # - 'floor': 切り捨て
    # - 'pomegranate': ポメグラネート法
    DISCRETIZE_METHODS_FOR_DISCRETE = ['round', 'ceil', 'floor', 'pomegranate']  # 全部試す
    # DISCRETIZE_METHODS_FOR_DISCRETE = ['round']  # roundのみ
    
    # ===== 実行順序の最適化 =====
    # continuousを先に実行してキャッシュを生成、discreteはそれを再利用
    if 'continuous' in THICKNESS_MODES and 'discrete' in THICKNESS_MODES:
        # 両方含まれている場合、continuousを先に
        THICKNESS_MODES_SORTED = ['continuous', 'discrete']
        print("\n💡 最適化: continuousモードを先に実行してキャッシュを生成します")
        print("   discreteモードはキャッシュを再利用して高速化されます\n")
    else:
        THICKNESS_MODES_SORTED = THICKNESS_MODES
    
    # ===== パターン数の計算と表示 =====
    total_combos = 0
    for mode in THICKNESS_MODES_SORTED:
        if mode == 'discrete':
            total_combos += len(RESULTS_CSVS) * len(SHAPE_TYPES) * len(SUBPIXEL_SAMPLINGS) * len(DISCRETIZE_METHODS_FOR_DISCRETE)
        else:
            total_combos += len(RESULTS_CSVS) * len(SHAPE_TYPES) * len(SUBPIXEL_SAMPLINGS)
    
    # パターン数の内訳を表示
    print(f"\n{'='*80}")
    print(f"📊 実行パターン数の内訳")
    print(f"{'='*80}")
    print(f"  CSVファイル数: {len(RESULTS_CSVS)}")
    print(f"  形状推定方法: {SHAPE_TYPES} ({len(SHAPE_TYPES)}種類)")
    print(f"  サブピクセル: {SUBPIXEL_SAMPLINGS} ({len(SUBPIXEL_SAMPLINGS)}種類)")
    print(f"  厚みマップモード: {THICKNESS_MODES_SORTED} (実行順)")
    if 'continuous' in THICKNESS_MODES_SORTED and 'discrete' in THICKNESS_MODES_SORTED:
        continuous_combos = len(RESULTS_CSVS) * len(SHAPE_TYPES) * len(SUBPIXEL_SAMPLINGS)
        discrete_combos = len(RESULTS_CSVS) * len(SHAPE_TYPES) * len(SUBPIXEL_SAMPLINGS) * len(DISCRETIZE_METHODS_FOR_DISCRETE)
        print(f"    - continuous: {continuous_combos}パターン")
        print(f"    - discrete: {discrete_combos}パターン ({len(DISCRETIZE_METHODS_FOR_DISCRETE)}種類の離散化方法)")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  ✅ 合計実行数: {total_combos}パターン")
    print(f"{'='*80}\n")
    
    print(f"\n{'#'*80}")
    print(f"# BATCH ANALYSIS START")
    print(f"# Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Total combinations: {total_combos}")
    print(f"#   CSV files: {len(RESULTS_CSVS)}")
    print(f"#   Shape types: {len(SHAPE_TYPES)}")
    print(f"#   Subpixel samplings: {len(SUBPIXEL_SAMPLINGS)}")
    print(f"#   Thickness modes: {THICKNESS_MODES}")
    if 'discrete' in THICKNESS_MODES:
        print(f"#   Discretize methods (for discrete): {DISCRETIZE_METHODS_FOR_DISCRETE}")
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
        
        for thickness_mode in THICKNESS_MODES_SORTED:
            # thickness_modeに応じて離散化方法を設定
            if thickness_mode == 'discrete':
                discretize_methods = DISCRETIZE_METHODS_FOR_DISCRETE
            else:
                discretize_methods = [None]  # continuousモードでは1回だけ
            
            for i, shape_type in enumerate(SHAPE_TYPES, 1):
                for j, subpixel_sampling in enumerate(SUBPIXEL_SAMPLINGS, 1):
                    for k, discretize_method in enumerate(discretize_methods, 1):
                        combo_num += 1
                        
                        # continuousモードの場合はdiscretize_methodは使用しない
                        if thickness_mode == 'continuous':
                            actual_discretize_method = 'round'  # デフォルト値（使用されない）
                            method_str = ''
                        else:
                            actual_discretize_method = discretize_method
                            method_str = f"[{discretize_method}]"
                        
                        print(f"\n{'#'*80}")
                        print(f"# Combination {combo_num}/{total_combos}")
                        print(f"#   CSV: {csv_name}")
                        print(f"#   Shape: {shape_type}")
                        print(f"#   Subpixel: {subpixel_sampling}×{subpixel_sampling}")
                        print(f"#   Thickness mode: {thickness_mode} {method_str}")
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
                            thickness_mode=thickness_mode,
                            voxel_z_um=VOXEL_Z_UM,
                            discretize_method=actual_discretize_method,
                            min_thickness_px=MIN_THICKNESS_PX,
                            csv_suffix=CSV_SUFFIX,
                            skip_if_completed=SKIP_IF_COMPLETED
                        )
                        
                        results.append({
                            'csv_file': csv_name,
                            'shape_type': shape_type,
                            'subpixel_sampling': subpixel_sampling,
                            'thickness_mode': thickness_mode,
                            'discretize_method': actual_discretize_method if thickness_mode == 'discrete' else None,
                            'success': success,
                            'skipped': success and SKIP_IF_COMPLETED and check_if_completed(
                                shape_type, subpixel_sampling, results_csv,
                                thickness_mode, actual_discretize_method, CSV_SUFFIX
                            )
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
        if result.get('skipped', False):
            status = "⏭️  SKIPPED"
        elif result['success']:
            status = "✅ SUCCESS"
        else:
            status = "❌ FAILED"
        
        csv_short = result['csv_file'].replace('Results_', '').replace('.csv', '')
        
        # thickness_modeに応じて表示を変更
        if result['thickness_mode'] == 'discrete' and result['discretize_method']:
            mode_str = f"{result['thickness_mode']}[{result['discretize_method']}]"
        else:
            mode_str = result['thickness_mode']
        
        print(f"  {csv_short:20s} | {result['shape_type']:8s} | subpixel{result['subpixel_sampling']:2d} | {mode_str:20s} : {status}")
    print(f"{'='*80}\n")
    
    success_count = sum(1 for r in results if r['success'])
    skipped_count = sum(1 for r in results if r.get('skipped', False))
    failed_count = sum(1 for r in results if not r['success'])
    
    print(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    if skipped_count > 0:
        print(f"  - Completed new: {success_count - skipped_count}")
        print(f"  - Skipped (already done): {skipped_count}")
    if failed_count > 0:
        print(f"  - Failed: {failed_count}")
    
    print(f"\n{'#'*80}")
    print(f"# All output directories:")
    print(f"{'#'*80}")
    
    # キャッシュ機能の説明
    if 'continuous' in THICKNESS_MODES and 'discrete' in THICKNESS_MODES:
        print(f"\n💾 キャッシュ機能:")
        print(f"  continuousモードで生成された厚みマップは以下に保存されています:")
        print(f"  → timeseries_density_output_*/thickness_cache/")
        print(f"  discreteモードはこれを再利用して高速化されました")
    
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
