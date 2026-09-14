#!/usr/bin/env python3
"""
Simple mean RI calculation: total_phase / theoretical ellipse volume

Compute theoretical volume from ROI shape parameters (Feret diameter, bounding box, area, etc.)
for ellipse/rod shapes, then divide total phase by that volume
to obtain mean refractive index.

mean_RI = n_medium + (total_phase * lambda) / (2*pi * volume)

Where:
- total_phase: sum of phase values of all pixels within the mask (rad)
- volume: theoretical ellipse/rod shape volume (um^3)
- lambda: wavelength (nm -> um)
- n_medium: refractive index of the medium
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
    Compute theoretical volume for ellipse/rod shape

    Parameters
    ----------
    major : float
        Major axis (in pixels)
    minor : float
        Minor axis (in pixels)
    pixel_size_um : float
        Pixel size (um)
    shape_type : str
        'rod': capsule shape (two hemispheres + cylinder)
        'ellipsoid': prolate spheroid

    Returns
    -------
    volume_um3 : float
        Volume (um^3)
    """
    # Convert from pixels to um
    length_um = major * pixel_size_um
    width_um = minor * pixel_size_um
    
    if shape_type == 'rod':
        # Rod shape: two hemispheres + cylinder
        r_um = width_um / 2.0
        h_um = length_um - 2 * r_um
        
        if h_um < 0:
            # No cylinder part (sphere)
            volume_um3 = (4.0 / 3.0) * np.pi * (r_um ** 3)
        else:
            # Two hemispheres (= one sphere) + cylinder
            volume_um3 = (4.0 / 3.0) * np.pi * (r_um ** 3) + np.pi * (r_um ** 2) * h_um
    else:  # ellipsoid
        # Prolate spheroid: V = (4/3)*pi * a * b * c
        # Assuming spheroid of revolution about the long axis
        a_um = length_um / 2.0
        b_um = width_um / 2.0
        c_um = width_um / 2.0
        volume_um3 = (4.0 / 3.0) * np.pi * a_um * b_um * c_um
    
    return volume_um3

# %%
def create_simple_mask(roi_params, image_shape, pixel_size_um):
    """
    Create a simple mask from ROI parameters

    Parameters
    ----------
    roi_params : dict
        ROI parameters (X, Y, Major, Minor, Angle, etc.)
    image_shape : tuple
        Image size (height, width)
    pixel_size_um : float
        Pixel size (um)

    Returns
    -------
    mask : ndarray
        Mask (2D boolean array)
    """
    height, width = image_shape
    y_coords, x_coords = np.ogrid[:height, :width]
    
    # Ellipse parameters
    center_x = roi_params['X']
    center_y = roi_params['Y']
    major = roi_params.get('Major', roi_params.get('Feret'))
    minor = roi_params.get('Minor', roi_params.get('MinFeret'))
    angle_deg = roi_params.get('Angle', roi_params.get('FeretAngle', 0))
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)
    
    # Coordinate transformation
    dx = x_coords - center_x
    dy = y_coords - center_y
    
    # Account for rotation
    x_rot = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
    y_rot = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    
    # Ellipse interior check
    a = major / 2.0
    b = minor / 2.0
    mask = ((x_rot / a) ** 2 + (y_rot / b) ** 2) <= 1.0
    
    return mask

# %%
def calculate_simple_mean_ri(phase_map, roi_params, pixel_size_um,
                              wavelength_nm, n_medium, shape_type='rod'):
    """
    Simple mean RI calculation: total_phase / theoretical ellipse volume

    Parameters
    ----------
    phase_map : ndarray
        Phase map (rad)
    roi_params : dict
        ROI parameters (Major, Minor, X, Y, Angle, etc.)
    pixel_size_um : float
        Pixel size (um)
    wavelength_nm : float
        Wavelength (nm)
    n_medium : float
        Refractive index of the medium
    shape_type : str
        'rod': capsule shape (recommended)
        'ellipsoid': prolate spheroid

    Returns
    -------
    mean_ri : float
        Mean refractive index
    volume_um3 : float
        Theoretical ellipse volume (um^3)
    total_phase : float
        Sum of all phase values (rad)
    """
    # Compute theoretical ellipse volume
    major = roi_params.get('Major', roi_params.get('Feret'))
    minor = roi_params.get('Minor', roi_params.get('MinFeret'))
    volume_um3 = calculate_ellipse_volume(major, minor, pixel_size_um, shape_type)
    
    if volume_um3 == 0:
        return n_medium, 0.0, 0.0
    
    # Create mask
    mask = create_simple_mask(roi_params, phase_map.shape, pixel_size_um)
    
    # Sum of all phase values within the mask
    total_phase = np.sum(phase_map[mask])
    
    # Mean RI calculation
    # n_sample = n_medium + (φ × λ) / (2π × thickness)
    # Overall: mean_RI = n_medium + (total_phase * lambda) / (2*pi * volume / pixel_area)
    #          = n_medium + (total_φ × λ × pixel_area) / (2π × volume)
    
    wavelength_um = wavelength_nm * 1e-3  # nm → µm
    pixel_area_um2 = pixel_size_um ** 2
    
    mean_ri = n_medium + (total_phase * wavelength_um * pixel_area_um2) / (2 * np.pi * volume_um3)
    
    return mean_ri, volume_um3, total_phase

# %%
def find_results_csv(base_dir):
    """
    Search for Results.csv file

    Parameters
    ----------
    base_dir : str
        Base directory for search

    Returns
    -------
    results_csv : str or None
        Path to Results.csv
    """
    # Search in typical locations
    candidates = [
        os.path.join(base_dir, 'Results.csv'),
        os.path.join(base_dir, '..', 'Results.csv'),
        os.path.join(base_dir, '..', '..', 'Results.csv'),
    ]
    
    # Wildcard search
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
    Search for the corresponding phase image

    Parameters
    ----------
    image_dir : str
        Image directory
    roi_label : str
        ROI label (e.g., "ROI_0000")
    frame_num : int
        Frame number

    Returns
    -------
    phase_file : str or None
        Path to the phase image
    """
    # Try multiple patterns
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
    Process directly from Results.csv

    Parameters
    ----------
    results_csv : str
        Path to Results.csv
    image_dir : str
        Phase image directory
    pixel_size_um : float
        Pixel size (um)
    wavelength_nm : float
        Wavelength (nm)
    n_medium : float
        Refractive index of the medium
    alpha_ri : float
        Specific refractive index increment (ml/mg)
    shape_type : str
        'rod': capsule shape (recommended)
        'ellipsoid': prolate spheroid

    Returns
    -------
    summary_df : DataFrame
        Summary per ROI
    """
    print(f"\n  Reading: {os.path.basename(results_csv)}")
    
    # Read Results.csv
    df = pd.read_csv(results_csv)
    
    # Check required columns
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
    
    # Store results
    results = []
    
    # Get image list
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
    if len(image_files) == 0:
        print(f"  Error: No TIFF files found in {image_dir}")
        return None
    
    print(f"  Found {len(image_files)} phase images")
    print(f"  Found {len(df)} ROIs in CSV")
    
    # Get size from first image
    first_image = Image.open(image_files[0])
    image_shape = (first_image.height, first_image.width)
    print(f"  Image size: {image_shape}")
    
    # Process each ROI
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Processing"):
        # ROI info
        roi_params = row.to_dict()
        
        # Get Frame or label information
        if 'Label' in row:
            label = row['Label']
            # Extract frame number
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
        
        # Search for corresponding image
        if frame_num <= len(image_files):
            phase_file = image_files[frame_num - 1]
        else:
            print(f"    Warning: Frame {frame_num} not found")
            continue
        
        # Load phase image
        phase_map = np.array(Image.open(phase_file)).astype(np.float64)
        
        # Simple mean RI calculation
        mean_ri, volume_um3, total_phase = calculate_simple_mean_ri(
            phase_map, roi_params, pixel_size_um, wavelength_nm, n_medium, shape_type
        )
        
        # Mass calculation
        mean_concentration_mg_ml = (mean_ri - n_medium) / alpha_ri
        total_mass_pg = mean_concentration_mg_ml * volume_um3
        
        # Save results
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
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values(['roi', 'frame_num']).reset_index(drop=True)
    
    return summary_df

# %%
def plot_timeseries(summary_df, output_dir, condition_name):
    """
    Create time-series plot

    Parameters
    ----------
    summary_df : DataFrame
        Summary data
    output_dir : str
        Output directory
    condition_name : str
        Condition name
    """
    # Plot per ROI
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
    
    # Axis labels and grid
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
    
    # Save
    plot_file = os.path.join(output_dir, 'timeseries_simple_mean_ri.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {plot_file}")

# %%
def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Simple mean RI calculation: total_phase / volume',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all conditions in the current directory
  python 30_simple_mean_ri_analysis.py

  # Process all conditions in a specific directory
  python 30_simple_mean_ri_analysis.py -d G:\\test_dens_est

  # Process a specific condition only
  python 30_simple_mean_ri_analysis.py -c timeseries_density_output_ellipse_subpixel5

  # Specify parameters
  python 30_simple_mean_ri_analysis.py --wavelength 532 --n-medium 1.335
"""
    )
    
    parser.add_argument('-d', '--base-dir', type=str, default='.',
                        help='Base directory (default: current directory)')
    parser.add_argument('-c', '--conditions', type=str, nargs='*', default=None,
                        help='Condition directories to process (wildcards allowed). If not specified, all conditions')
    parser.add_argument('--pixel-size', type=float, default=0.348,
                        help='Pixel size (um, default: 0.348)')
    parser.add_argument('--wavelength', type=float, default=663,
                        help='Wavelength (nm, default: 663)')
    parser.add_argument('--n-medium', type=float, default=1.333,
                        help='Refractive index of the medium (default: 1.333)')
    parser.add_argument('--alpha-ri', type=float, default=0.00018,
                        help='Specific refractive index increment (ml/mg, default: 0.00018)')
    parser.add_argument('--voxel-z', type=float, default=0.3,
                        help='Voxel size in Z direction (um, default: 0.3)')
    parser.add_argument('--list-only', action='store_true',
                        help='Show condition list only and exit')
    
    # Support execution in Jupyter environment
    if 'ipykernel' in sys.modules:
        filtered_argv = [arg for arg in sys.argv if not arg.startswith('--f=') and not arg.startswith('-f=')]
        args = parser.parse_args(filtered_argv[1:] if len(filtered_argv) > 1 else [])
    else:
        args = parser.parse_args()
    
    print("="*80)
    print("Simple Mean RI Analysis: total_phase / volume")
    print("="*80)
    
    # Parameters
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
    
    # Search for condition directories
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
    
    # Exclude filtered directories
    condition_dirs = [d for d in condition_dirs if os.path.isdir(d)]
    condition_dirs = sorted(condition_dirs)
    
    print(f"\nFound {len(condition_dirs)} condition directories")
    
    if len(condition_dirs) == 0:
        print("No condition directories found!")
        return
    
    # Display condition list
    print("\nConditions to process:")
    for i, d in enumerate(condition_dirs, 1):
        print(f"  {i}. {os.path.basename(d)}")
    
    if args.list_only:
        print("\n(--list-only mode: exiting)")
        return
    
    # Process all conditions
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
                # Create plot
                output_dir = condition_dir.replace('timeseries_density_output_', 'timeseries_plots_') + '_simple_mean_ri'
                plot_timeseries(summary_df, output_dir, os.path.basename(condition_dir))
                
                # Add condition name
                summary_df['condition'] = os.path.basename(condition_dir)
                all_summaries.append(summary_df)
                
        except Exception as e:
            print(f"  Error processing {condition_dir}: {e}")
            import traceback
            traceback.print_exc()
    
    # Combined summary across all conditions
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

