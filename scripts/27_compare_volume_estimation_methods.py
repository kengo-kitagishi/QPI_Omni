#!/usr/bin/env python3
"""
Batch analysis: exhaustive execution of all parameter combinations.

Combinations to execute:
[CSV files]
  - Results_enlarge.csv
  - Results_enlarge_interpolate.csv

[Shape estimation]
  - ellipse
  - feret (Feret diameter)

[Subpixel precision]
  - 1x1
  - 5x5
  - 10x10

[Thickness map mode]
  - continuous: saves thickness map to cache
  - discrete (round, ceil, floor, pomegranate): reuses cache for speedup

Default settings run all patterns:
  2 (CSV) x 2 (shape) x 3 (subpixel) x (1 + 4) = 60 patterns

[Optimization]
Thickness maps computed in continuous mode are cached,
and discrete mode reuses them for significant speedup.
  - continuous: full computation (phase image to thickness map)
  - discrete: cache load -> discretize -> volume recalculation only (~10x faster)
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

# Copy TimeSeriesDensityMapper class from 24_ellipse_volume.py
# (defined directly to avoid import issues)

class TimeSeriesDensityMapper:
    """Generate refractive index (RI) maps from time-series images and Results.csv"""

    # ... (copy entire TimeSeriesDensityMapper class from 24_ellipse_volume.py)
    # Using exec() for import due to space constraints
    pass

def check_if_completed(shape_type, subpixel_sampling, results_csv, thickness_mode,
                       discretize_method, csv_suffix):
    """
    Check if analysis for the specified conditions has already been completed.

    Returns:
    --------
    bool : True if already completed
    """
    # Generate output folder name
    csv_name = os.path.basename(results_csv)
    csv_name_without_ext = os.path.splitext(csv_name)[0]
    
    if csv_suffix is None:
        # Auto-extract from CSV filename
        if '_' in csv_name_without_ext:
            parts = csv_name_without_ext.split('_', 1)
            csv_identifier = parts[1] if len(parts) > 1 and parts[1] else None
        else:
            csv_identifier = None
    else:
        csv_identifier = csv_suffix
    
    # Generate folder name
    if thickness_mode == 'discrete':
        mode_suffix = f"{thickness_mode}_{discretize_method}"
    else:
        mode_suffix = thickness_mode
    
    if csv_identifier:
        dir_suffix = f"{shape_type}_subpixel{subpixel_sampling}_{csv_identifier}_{mode_suffix}"
    else:
        dir_suffix = f"{shape_type}_subpixel{subpixel_sampling}_{mode_suffix}"
    
    output_dir = f"timeseries_density_output_{dir_suffix}"
    
    # Path to completion flag file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flag_file = os.path.join(script_dir, output_dir, '.completed')
    
    return os.path.exists(flag_file)


def run_analysis(shape_type, subpixel_sampling, results_csv, image_directory,
                 wavelength_nm, n_medium, pixel_size_um, alpha_ri, max_rois,
                 thickness_mode='continuous', voxel_z_um=0.3, discretize_method='round',
                 min_thickness_px=0.0, csv_suffix=None, skip_if_completed=True):
    """
    Execute analysis with specified parameters.

    Parameters:
    -----------
    thickness_mode : str
        Thickness map mode. Default: 'continuous'
    voxel_z_um : float
        Z-direction voxel size (um). Default: 0.3
    discretize_method : str
        Discretization method. Default: 'round'
    min_thickness_px : float
        Minimum thickness threshold (in pixel units). Default: 0.0
    csv_suffix : str, optional
        Suffix appended to output folder name. Default: None
        If None, auto-extracted from CSV filename
    skip_if_completed : bool, optional
        Skip if already completed. Default: True
    """
    # Completion check
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
            return True  # Skipped but treated as success
    
    print(f"\n{'='*80}")
    print(f"Starting analysis:")
    print(f"  Shape type: {shape_type}")
    print(f"  Subpixel sampling: {subpixel_sampling}×{subpixel_sampling}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Execute 24_ellipse_volume.py via exec()
        # Pass as global variables
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
        
        # Read and execute the contents of 24_ellipse_volume.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, '24_ellipse_volume.py')
        
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Replace to force execution of the if __name__ == "__main__" block
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


# ===== Main execution =====
if __name__ == "__main__":
    # Common parameters
    # === Specify as list when processing multiple CSV files ===
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
    
    MAX_ROIS = None  # Test run (None for all ROIs)

    # === Thickness map parameters (for discrete mode) ===
    VOXEL_Z_UM = 0.3  # Z-direction voxel size (um)
    MIN_THICKNESS_PX = 0.0  # Minimum thickness threshold (pixels, 0.0=no threshold)

    # === CSV suffix (for identifying output folder name) ===
    # Option 1: Auto-extract (specify None)
    #   - Automatically extracted from CSV filename
    #   - Results_enlarge.csv -> 'enlarge'
    #   - Results_enlarge_interpolate.csv -> 'enlarge_interpolate'
    # Option 2: Manual specification (specify a string)
    #   - e.g.: CSV_SUFFIX = 'my_custom_name'
    CSV_SUFFIX = None  # None for auto-extract, or manually specify a string

    # === Resume functionality ===
    # True: Skip completed conditions (allows resuming after crash)
    # False: Re-run everything
    SKIP_IF_COMPLETED = True  # Resume functionality enabled (recommended)

    # ===== Parameter combinations =====
    # [Shape estimation method]
    # - 'ellipse': Ellipse fitting
    # - 'feret': Feret diameter based
    SHAPE_TYPES = ['ellipse', 'feret']  # Try both
    # SHAPE_TYPES = ['ellipse']  # Ellipse only
    # SHAPE_TYPES = ['feret']  # Feret only

    # [Subpixel precision]
    # Subpixel sampling count (NxN)
    SUBPIXEL_SAMPLINGS = [1, 5, 10]  # Try all
    # SUBPIXEL_SAMPLINGS = [1]  # Fast test
    # SUBPIXEL_SAMPLINGS = [5, 10]  # High precision only

    # [Thickness map mode]
    # - 'continuous': Continuous values (real-valued) *run first to cache thickness maps
    # - 'discrete': Discrete values (rounded to voxel units) *reuses cache for speedup
    #
    # IMPORTANT: Run continuous first!
    #   Discrete mode reuses thickness maps saved by continuous for speedup
    THICKNESS_MODES = ['continuous', 'discrete']  # All patterns (recommended, continuous first)
    # THICKNESS_MODES = ['continuous']  # Continuous only
    # THICKNESS_MODES = ['discrete']  # Discrete only (requires prior continuous run)

    # [Discretization method] (used only in discrete mode)
    # - 'round': Round to nearest integer
    # - 'ceil': Round up
    # - 'floor': Round down
    # - 'pomegranate': Pomegranate method
    DISCRETIZE_METHODS_FOR_DISCRETE = ['round', 'ceil', 'floor', 'pomegranate']  # Try all
    # DISCRETIZE_METHODS_FOR_DISCRETE = ['round']  # Round only
    
    # ===== Execution order optimization =====
    # Run continuous first to generate cache, discrete reuses it
    if 'continuous' in THICKNESS_MODES and 'discrete' in THICKNESS_MODES:
        # If both are included, run continuous first
        THICKNESS_MODES_SORTED = ['continuous', 'discrete']
        print("\nOptimization: Running continuous mode first to generate cache")
        print("   Discrete mode will reuse cache for speedup\n")
    else:
        THICKNESS_MODES_SORTED = THICKNESS_MODES
    
    # ===== Calculate and display pattern count =====
    total_combos = 0
    for mode in THICKNESS_MODES_SORTED:
        if mode == 'discrete':
            total_combos += len(RESULTS_CSVS) * len(SHAPE_TYPES) * len(SUBPIXEL_SAMPLINGS) * len(DISCRETIZE_METHODS_FOR_DISCRETE)
        else:
            total_combos += len(RESULTS_CSVS) * len(SHAPE_TYPES) * len(SUBPIXEL_SAMPLINGS)
    
    # Display pattern count breakdown
    print(f"\n{'='*80}")
    print(f"Execution pattern breakdown")
    print(f"{'='*80}")
    print(f"  CSV files: {len(RESULTS_CSVS)}")
    print(f"  Shape estimation methods: {SHAPE_TYPES} ({len(SHAPE_TYPES)} types)")
    print(f"  Subpixel: {SUBPIXEL_SAMPLINGS} ({len(SUBPIXEL_SAMPLINGS)} types)")
    print(f"  Thickness map modes: {THICKNESS_MODES_SORTED} (execution order)")
    if 'continuous' in THICKNESS_MODES_SORTED and 'discrete' in THICKNESS_MODES_SORTED:
        continuous_combos = len(RESULTS_CSVS) * len(SHAPE_TYPES) * len(SUBPIXEL_SAMPLINGS)
        discrete_combos = len(RESULTS_CSVS) * len(SHAPE_TYPES) * len(SUBPIXEL_SAMPLINGS) * len(DISCRETIZE_METHODS_FOR_DISCRETE)
        print(f"    - continuous: {continuous_combos} patterns")
        print(f"    - discrete: {discrete_combos} patterns ({len(DISCRETIZE_METHODS_FOR_DISCRETE)} discretization methods)")
    print(f"  {'='*24}")
    print(f"  Total executions: {total_combos} patterns")
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
    
    # Execute all combinations
    for csv_idx, results_csv in enumerate(RESULTS_CSVS, 1):
        csv_name = os.path.basename(results_csv)
        print(f"\n{'='*80}")
        print(f"Processing CSV {csv_idx}/{len(RESULTS_CSVS)}: {csv_name}")
        print(f"{'='*80}\n")
        
        for thickness_mode in THICKNESS_MODES_SORTED:
            # Set discretization method according to thickness_mode
            if thickness_mode == 'discrete':
                discretize_methods = DISCRETIZE_METHODS_FOR_DISCRETE
            else:
                discretize_methods = [None]  # Only once in continuous mode
            
            for i, shape_type in enumerate(SHAPE_TYPES, 1):
                for j, subpixel_sampling in enumerate(SUBPIXEL_SAMPLINGS, 1):
                    for k, discretize_method in enumerate(discretize_methods, 1):
                        combo_num += 1
                        
                        # discretize_method is not used in continuous mode
                        if thickness_mode == 'continuous':
                            actual_discretize_method = 'round'  # default value (not used)
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
    
    # Final summary
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
        
        # Change display according to thickness_mode
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
    
    # Cache functionality description
    if 'continuous' in THICKNESS_MODES and 'discrete' in THICKNESS_MODES:
        print(f"\nCache functionality:")
        print(f"  Thickness maps generated in continuous mode are saved at:")
        print(f"  -> timeseries_density_output_*/thickness_cache/")
        print(f"  Discrete mode reused these for speedup")
    
    # Display grouped by CSV file
    for results_csv in RESULTS_CSVS:
        csv_name = os.path.basename(results_csv)
        csv_short = csv_name.replace('Results_', '').replace('.csv', '')
        print(f"\n  [{csv_short}]")
        
        for result in results:
            if result['csv_file'] == csv_name:
                # Estimate the auto-extracted suffix from CSV filename
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
