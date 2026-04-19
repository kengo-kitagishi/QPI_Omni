#!/usr/bin/env python3
"""
Extract boundaries from zstack.tif, convert density map to Refractive Index (RI), and visualize.

Features:
1. Load zstack.tif file (thickness map)
2. Create binary mask by thresholding
3. Extract boundary lines
4. Detect contours
5. Convert density map from phase to refractive index (RI)
6. Overlay contour lines on RI map for visualization
7. Save images showing mask coverage

Physical principle:
Phase: phi = (2*pi/lambda) x (n_sample - n_medium) x thickness
Refractive index: n_sample = n_medium + (phi x lambda) / (2*pi x thickness)
"""
# %% 
import os
import glob
import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

def phase_to_refractive_index(phase_map, thickness_map, wavelength_nm=663,
                               n_medium=1.333, pixel_size_um=0.348):
    """
    Calculate refractive index (RI) map from phase map.

    Parameters:
    -----------
    phase_map : numpy array
        Phase map (in radians)
    thickness_map : numpy array
        Thickness map (in pixel units)
    wavelength_nm : float
        Wavelength (nanometers). Default: 663nm (red laser)
    n_medium : float
        Medium refractive index. Default: 1.333 (aqueous medium)
    pixel_size_um : float
        Pixel size (micrometers). Default: 0.348 um
        For 507x507 reconstructed images

    Returns:
    --------
    ri_map : numpy array
        Refractive index map

    Formula:
    --------
    phi = (2*pi/lambda) x (n_sample - n_medium) x thickness
    n_sample = n_medium + (phi x lambda) / (2*pi x thickness)
    """
    # Unify units: convert everything to um
    wavelength_um = wavelength_nm / 1000.0  # nm -> um
    thickness_um = thickness_map * pixel_size_um  # pixel -> um

    # Mask: calculate only where thickness is non-zero
    mask = thickness_um > 0

    # Initialize RI map (with medium refractive index)
    ri_map = np.full_like(phase_map, n_medium, dtype=np.float64)

    # Calculate refractive index from phase
    # n_sample = n_medium + (φ × λ) / (2π × thickness)
    if np.any(mask):
        ri_map[mask] = n_medium + (phase_map[mask] * wavelength_um) / (2 * np.pi * thickness_um[mask])
    
    return ri_map


def ri_to_concentration(ri_map, n_medium=1.333, alpha_ri=0.0018):
    """
    Calculate mass concentration map from refractive index map.

    Parameters:
    -----------
    ri_map : numpy array
        Refractive index map
    n_medium : float
        Medium refractive index. Default: 1.333
    alpha_ri : float
        Specific refractive index increment [ml/mg]. Default: 0.0018

    Returns:
    --------
    concentration_map : numpy array
        Mass concentration map [mg/ml]

    Formula:
    --------
    C [mg/ml] = (RI - RI_medium) / alpha
    """
    concentration_map = (ri_map - n_medium) / alpha_ri
    return concentration_map


def visualize_mask_on_density(zstack_path, density_path=None, threshold=0, output_dir=None,
                               wavelength_nm=663, n_medium=1.333, pixel_size_um=0.348,
                               alpha_ri=0.0018):
    """
    Extract boundaries from zstack.tif, convert density map to RI, and visualize.

    Parameters:
    -----------
    zstack_path : str
        Path to zstack.tif file (thickness map)
    density_path : str or None
        Path to density map file (phase map, None for auto-search)
    threshold : float
        Threshold (values above this are treated as mask. Default: 0)
    output_dir : str or None
        Output directory (None for same location as input file)
    wavelength_nm : float
        Wavelength (nanometers). Default: 663nm
    n_medium : float
        Medium refractive index. Default: 1.333
    pixel_size_um : float
        Pixel size (micrometers). Default: 0.348 um
    alpha_ri : float
        Specific refractive index increment [ml/mg]. Default: 0.0018

    Returns:
    --------
    output_path : str
        Path to saved visualization image

    Note:
    -----
    - Contour lines indicate mask region boundaries
    - The interior of contour lines is the masked region
    - The density map is interpreted as phase and converted to RI and mass concentration (mg/ml)
    """
    # Load zstack (thickness map)
    print(f"Loading: {os.path.basename(zstack_path)}")
    zstack = tifffile.imread(zstack_path)
    print(f"  Zstack (thickness) shape: {zstack.shape}")
    print(f"  Thickness range: [{zstack.min():.4f}, {zstack.max():.4f}] pixels")
    print(f"  Thickness range: [{zstack.min()*pixel_size_um:.4f}, {zstack.max()*pixel_size_um:.4f}] µm")
    
    # Create binary mask (values above threshold = 1, else 0)
    binary_mask = (zstack > threshold).astype(np.uint8)
    print(f"  Threshold: > {threshold}")
    print(f"  Mask pixels: {np.count_nonzero(binary_mask)} / {binary_mask.size} ({100*np.count_nonzero(binary_mask)/binary_mask.size:.1f}%)")
    
    if np.count_nonzero(binary_mask) == 0:
        print("  WARNING: No pixels above threshold!")
        return None
    
    # Detect contours (find_contours)
    contours = measure.find_contours(binary_mask, 0.5)
    print(f"  Found {len(contours)} contour(s)")
    
    if len(contours) == 0:
        print("  WARNING: No contours found!")
        return None
    
    # Load density map (phase map)
    if density_path is None:
        # Search for density map with the same name as zstack.tif
        base_name = os.path.basename(zstack_path).replace('_zstack.tif', '')
        dir_name = os.path.dirname(zstack_path)
        
        # Possible filename patterns
        possible_patterns = [
            f"{base_name}_density.tif",
            f"{base_name}.tif",
            f"{base_name}_mean.tif",
        ]
        
        for pattern in possible_patterns:
            candidate_path = os.path.join(dir_name, pattern)
            if os.path.exists(candidate_path):
                density_path = candidate_path
                break
        
        if density_path is None:
            print("  ERROR: Could not find corresponding density map!")
            print(f"  Searched for: {possible_patterns}")
            return None
    
    # Load density map (interpreted as phase map)
    print(f"  Loading phase map: {os.path.basename(density_path)}")
    phase_map = tifffile.imread(density_path)
    print(f"  Phase map shape: {phase_map.shape}")
    print(f"  Phase range: [{phase_map.min():.4f}, {phase_map.max():.4f}] (arbitrary units)")
    
    # Convert phase to refractive index (RI)
    print(f"  Converting phase to refractive index...")
    print(f"    Wavelength: {wavelength_nm} nm")
    print(f"    Medium RI: {n_medium}")
    print(f"    Pixel size: {pixel_size_um} µm")
    
    ri_map = phase_to_refractive_index(
        phase_map, zstack, 
        wavelength_nm=wavelength_nm,
        n_medium=n_medium,
        pixel_size_um=pixel_size_um
    )
    
    mask = binary_mask > 0
    if np.any(mask):
        print(f"  RI range (masked region): [{ri_map[mask].min():.6f}, {ri_map[mask].max():.6f}]")
        print(f"  Mean RI (masked region): {ri_map[mask].mean():.6f}")
    
    # Calculate mass concentration map
    print(f"  Converting RI to protein concentration...")
    print(f"    Alpha (RI increment): {alpha_ri} ml/mg")
    concentration_map = ri_to_concentration(ri_map, n_medium=n_medium, alpha_ri=alpha_ri)
    
    if np.any(mask):
        print(f"  Concentration range (masked region): [{concentration_map[mask].min():.2f}, {concentration_map[mask].max():.2f}] mg/ml")
        print(f"  Mean concentration (masked region): {concentration_map[mask].mean():.2f} mg/ml")
    
    # Determine output path
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(zstack_path), "visualized")
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(zstack_path))[0]
    output_filename = f"{base_name}_RI_Conc_map.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Visualization (RI map and mass concentration map side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), dpi=150)
    
    # === Left: RI map ===
    # Set color scale range (physiologically reasonable range)
    # Cytoplasm: ~1.35-1.37
    # Nucleus: ~1.38-1.40
    # Protein aggregates: ~1.40-1.45
    vmin_ri = n_medium  # Medium refractive index
    vmax_ri = max(1.45, ri_map[mask].max() if np.any(mask) else 1.45)  # Set upper limit
    
    im1 = ax1.imshow(ri_map, cmap='jet', interpolation='nearest', 
                     vmin=vmin_ri, vmax=vmax_ri)
    
    # Draw all contours with white lines (for visibility)
    for contour in contours:
        ax1.plot(contour[:, 1], contour[:, 0], 'w-', linewidth=2.5, alpha=0.9)
        ax1.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5, alpha=0.8)

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Refractive Index (RI)', rotation=270, labelpad=25, fontsize=12)
    
    # Add reference values
    cbar1.ax.axhline(y=n_medium, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8)
    cbar1.ax.text(1.5, n_medium, f'Medium ({n_medium:.3f})', va='center', fontsize=9, color='cyan')
    
    if np.any(mask):
        mean_ri = ri_map[mask].mean()
        cbar1.ax.axhline(y=mean_ri, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
        cbar1.ax.text(1.5, mean_ri, f'Mean ({mean_ri:.3f})', va='center', fontsize=9, color='white')
    
    # Add info to title
    mask_percentage = 100 * np.count_nonzero(binary_mask) / binary_mask.size
    title1 = f'Refractive Index Map\n{base_name.replace("_zstack", "")}\n'
    title1 += f'λ={wavelength_nm}nm, n_medium={n_medium:.3f}\n'
    title1 += f'(Red line = mask boundary, {mask_percentage:.1f}% masked)'
    ax1.set_title(title1, fontsize=11)
    ax1.axis('off')
    
    # === Right: Mass concentration map ===
    # Set color scale range
    vmin_conc = 0
    vmax_conc = max(50, concentration_map[mask].max() if np.any(mask) else 50)
    
    im2 = ax2.imshow(concentration_map, cmap='hot', interpolation='nearest',
                     vmin=vmin_conc, vmax=vmax_conc)
    
    # Draw all contours with white lines (for visibility)
    for contour in contours:
        ax2.plot(contour[:, 1], contour[:, 0], 'w-', linewidth=2.5, alpha=0.9)
        ax2.plot(contour[:, 1], contour[:, 0], 'cyan-', linewidth=1.5, alpha=0.8)

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Concentration (mg/ml)', rotation=270, labelpad=25, fontsize=12)
    
    # Add mean value
    if np.any(mask):
        mean_conc = concentration_map[mask].mean()
        cbar2.ax.axhline(y=mean_conc, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8)
        cbar2.ax.text(1.5, mean_conc, f'Mean ({mean_conc:.1f})', va='center', fontsize=9, color='cyan')
    
    # Add info to title
    title2 = f'Protein Concentration Map\n{base_name.replace("_zstack", "")}\n'
    title2 += f'α={alpha_ri} ml/mg, pixel={pixel_size_um}µm\n'
    title2 += f'(Cyan line = mask boundary, {mask_percentage:.1f}% masked)'
    ax2.set_title(title2, fontsize=11)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"  Saved: {output_filename}")
    
    return output_path


def process_all_zstacks(input_dir, output_dir=None, threshold=0,
                        wavelength_nm=663, n_medium=1.333, pixel_size_um=0.348,
                        alpha_ri=0.0018):
    """
    Process and visualize all zstack.tif files in a directory.

    Parameters:
    -----------
    input_dir : str
        Directory containing zstack.tif files
    output_dir : str or None
        Output directory (None for auto-setting)
    threshold : float
        Threshold (values above this are treated as mask. Default: 0)
    wavelength_nm : float
        Wavelength (nanometers). Default: 663nm
    n_medium : float
        Medium refractive index. Default: 1.333
    pixel_size_um : float
        Pixel size (micrometers). Default: 0.348 um
    alpha_ri : float
        Specific refractive index increment [ml/mg]. Default: 0.0018

    Returns:
    --------
    output_paths : list
        List of paths to created visualization images
    """
    # Search for zstack.tif files
    zstack_files = sorted(glob.glob(os.path.join(input_dir, "*_zstack.tif")))
    
    if not zstack_files:
        raise FileNotFoundError(f"No *_zstack.tif files found in {input_dir}")
    
    print(f"\n{'='*70}")
    print(f"Found {len(zstack_files)} zstack files")
    print(f"{'='*70}\n")
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, "visualized")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each zstack file
    output_paths = []
    
    for i, zstack_path in enumerate(zstack_files, 1):
        print(f"[{i}/{len(zstack_files)}]")
        
        try:
            output_path = visualize_mask_on_density(
                zstack_path,
                threshold=threshold,
                output_dir=output_dir,
                wavelength_nm=wavelength_nm,
                n_medium=n_medium,
                pixel_size_um=pixel_size_um,
                alpha_ri=alpha_ri
            )
            
            if output_path:
                output_paths.append(output_path)
        
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    return output_paths


# ===== Main execution =====
if __name__ == "__main__":
    # Parameter settings
    INPUT_DIR = "/mnt/user-data/outputs/timeseries_density_output/density_tiff"
    OUTPUT_DIR = "/mnt/user-data/outputs/timeseries_density_output/visualized"
    THRESHOLD = 0  # Zstack value threshold (values above this are treated as mask)

    # QPI experiment parameters (same values as 01_QPI_analysis.py)
    WAVELENGTH_NM = 663      # Laser wavelength (nanometers)
                             # Experimental value: 663nm (red laser)
    N_MEDIUM = 1.333         # Medium refractive index
                             # Water: 1.333, PBS: 1.334, DMEM: ~1.335
    PIXEL_SIZE_UM = 0.348    # Pixel size (micrometers)
                             # For 507x507 reconstructed images
                             # Calculation: 0.08625 um x (2048/507) = 0.348 um/pixel
                             # Note: Original hologram 2048x2048 has 0.08625 um/pixel
    ALPHA_RI = 0.0018        # Specific refractive index increment [ml/mg]
                             # Typical value for proteins: 0.0018 ml/mg
                             # Reference: C [mg/ml] = (RI - RI_medium) / alpha
    
    print(f"\n{'='*70}")
    print(f"QPI Parameters:")
    print(f"  Wavelength: {WAVELENGTH_NM} nm")
    print(f"  Medium RI: {N_MEDIUM}")
    print(f"  Pixel size: {PIXEL_SIZE_UM} µm (for 507×507 reconstructed images)")
    print(f"  Alpha (RI increment): {ALPHA_RI} ml/mg")
    print(f"  Note: Concentration (mg/ml) = (RI - {N_MEDIUM}) / {ALPHA_RI}")
    print(f"{'='*70}\n")
    
    # Process all zstack files and create visualization images
    # Note: The interior of contour lines indicates the masked region
    output_paths = process_all_zstacks(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        threshold=THRESHOLD,
        wavelength_nm=WAVELENGTH_NM,
        n_medium=N_MEDIUM,
        pixel_size_um=PIXEL_SIZE_UM,
        alpha_ri=ALPHA_RI
    )
    
    print(f"\n{'#'*70}")
    print(f"# Processing completed!")
    print(f"# Created {len(output_paths)} RI & Concentration visualization images")
    print(f"# Output directory: {OUTPUT_DIR}")
    print(f"# Note: Images show Refractive Index (RI) and Protein Concentration maps")
    print(f"#       Red/Cyan contour = mask boundary (inside = masked region)")
    print(f"#       RI range: {N_MEDIUM:.3f} (medium) to ~1.40 (protein)")
    print(f"#       Concentration: C (mg/ml) = (RI - {N_MEDIUM:.3f}) / {ALPHA_RI}")
    print(f"{'#'*70}\n")
# %%
