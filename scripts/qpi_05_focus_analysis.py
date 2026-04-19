# %%
"""
With-wo focus analysis script (fast version)

- Crop region visualization
- Use Pos0 as background
- Gaussian background subtraction
- QPI reconstruction of with/wo images
- ECC alignment
- Managed by config file (focus_analysis_config.yaml)
"""

import os
import yaml
import numpy as np
import tifffile
import cv2
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm
import matplotlib.pyplot as plt
from qpi_common import (create_qpi_params, to_uint8, visualize_crop_region,
                        gaussian_background_subtraction)

# %%
# ==================== Load Config File ====================

# Config file path (modify as needed)
CONFIG_PATH = "/Users/kitak/QPI_Omni/scripts/focus_analysis_config.yaml"

print("=" * 80)
print("With-wo Focus Analysis Script (Gaussian background subtraction + ECC)")
print("=" * 80)
print(f"\nLoading config file: {CONFIG_PATH}")

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Expand settings
BASE_DIR = config['base_dir']
WITH_DIR = config['with_dir']
WO_DIR = config['wo_dir']
BG_POS = config['bg_pos']
POS_START = config['pos_start']
POS_END = config['pos_end']
FILE_PATTERN = config.get('file_pattern', 'img_000000000_Default_000.tif')

# Crop settings
crop_config = config['crop']
CROP_Y_START = crop_config['y_start']
CROP_Y_END = crop_config['y_end']
CROP_X_START = crop_config['x_start']
CROP_X_END = crop_config['x_end']
CROP_COORDS = (CROP_Y_START, CROP_Y_END, CROP_X_START, CROP_X_END)

# QPI parameters
qpi_config = config['qpi']
WAVELENGTH = qpi_config['wavelength']
NA = qpi_config['NA']
PIXELSIZE = qpi_config['pixelsize']
OFFAXIS_CENTER = tuple(qpi_config['offaxis_center'])

# Gaussian background subtraction settings
gauss_config = config['gaussian_backsub']
GAUSS_ENABLED = gauss_config['enabled']
HIST_MIN = gauss_config['hist_min']
HIST_MAX = gauss_config['hist_max']
N_BINS = gauss_config['n_bins']
SMOOTH_WINDOW = gauss_config['smooth_window']
MIN_PHASE = gauss_config['min_phase']

# Alignment settings
align_config = config['alignment']
WARP_MODE_STR = align_config.get('warp_mode', 'AFFINE')
WARP_MODE = getattr(cv2.MOTION_, WARP_MODE_STR, cv2.MOTION_AFFINE)
ECC_ITERATIONS = align_config.get('iterations', 100000)
ECC_EPS = align_config.get('eps', 1e-8)

# Output settings
output_config = config['output']
DIFF_VMIN, DIFF_VMAX = output_config['diff_range']
COLORMAP = output_config.get('colormap', 'JET')
SAVE_ALIGNED = output_config.get('save_aligned', True)
SAVE_INDIVIDUAL_PHASE = output_config.get('save_individual_phase', False)
SAVE_CROP_VIZ = output_config.get('save_crop_visualization', True)

print("\n[Settings]")
print(f"  BASE_DIR: {BASE_DIR}")
print(f"  WITH_DIR: {WITH_DIR}")
print(f"  WO_DIR: {WO_DIR}")
print(f"  BG_POS: Pos{BG_POS}")
print(f"  Position range: Pos{POS_START} ~ Pos{POS_END}")
print(f"  File pattern: {FILE_PATTERN}")
print(f"  Crop region: ({CROP_Y_START}:{CROP_Y_END}, {CROP_X_START}:{CROP_X_END})")
print(f"  Gaussian background subtraction: {'enabled' if GAUSS_ENABLED else 'disabled'}")
print(f"  Alignment: {WARP_MODE_STR} (iterations={ECC_ITERATIONS})")
print(f"  Difference display range: [{DIFF_VMIN}, {DIFF_VMAX}]")

# %%
# ==================== Crop Region Visualization ====================

print("\n" + "=" * 80)
print("Crop Region Visualization")
print("=" * 80)

# Load sample image for visualization
sample_path = os.path.join(BASE_DIR, WITH_DIR, f"Pos{POS_START}", FILE_PATTERN)
if not os.path.exists(sample_path):
    print(f"WARNING: Sample image not found: {sample_path}")
    print("   Using Pos0 instead...")
    sample_path = os.path.join(BASE_DIR, WITH_DIR, f"Pos{BG_POS}", FILE_PATTERN)

sample_img = np.array(Image.open(sample_path))
print(f"\nOriginal image size: {sample_img.shape}")
print(f"Cropped size: ({CROP_Y_END - CROP_Y_START}, {CROP_X_END - CROP_X_START})")

# Visualize crop region
fig = visualize_crop_region(sample_img, CROP_COORDS, 
                            title="Crop Region Visualization")
plt.show()

# Save
output_base = os.path.join(BASE_DIR, WITH_DIR, "focus_analysis_output")
os.makedirs(output_base, exist_ok=True)

if SAVE_CROP_VIZ:
    crop_viz_path = os.path.join(output_base, "crop_region_visualization.png")
    fig.savefig(crop_viz_path, dpi=150, bbox_inches='tight')
    print(f"\nCrop region visualization saved: {crop_viz_path}")
plt.close(fig)

# Also display cropped image
cropped_img = sample_img[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]
plt.figure(figsize=(10, 8))
plt.imshow(cropped_img, cmap='gray')
plt.title('Cropped Image')
plt.colorbar()
plt.tight_layout()
plt.show()

print("Verify the crop region. If OK, proceed to the next step.")

# %%
# ==================== Load Pos0 (BG) Image and Parameter Setup ====================

print("\n" + "=" * 80)
print("Loading Pos0 (BG) Image")
print("=" * 80)

# Load Pos0 from the with side
bg_path_with = os.path.join(BASE_DIR, WITH_DIR, f"Pos{BG_POS}", FILE_PATTERN)
print(f"\nBG image path: {bg_path_with}")

if not os.path.exists(bg_path_with):
    raise FileNotFoundError(f"BG image not found: {bg_path_with}")

# Load and crop BG image
bg_img = np.array(Image.open(bg_path_with))
bg_img = bg_img[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]

print(f"BG image size: {bg_img.shape}")

# QPI parameter setup
params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=bg_img.shape,
    pixelsize=PIXELSIZE,
    offaxis_center=OFFAXIS_CENTER
)

print(f"QPI parameter setup complete:")
print(f"  - Wavelength: {WAVELENGTH*1e9:.1f} nm")
print(f"  - NA: {NA}")
print(f"  - Offaxis center: {OFFAXIS_CENTER}")
print(f"  - Aperture size: {params.aperturesize}")

# QPI reconstruction of BG image
field_bg = get_field(bg_img, params)
angle_bg = unwrap_phase(np.angle(field_bg))

print("BG phase image obtained")

# %%
# ==================== Prepare Output Directories ====================

diff_dir = os.path.join(output_base, "diff_wo_minus_with")
cmap_dir = os.path.join(output_base, "diff_colormap")
os.makedirs(diff_dir, exist_ok=True)
os.makedirs(cmap_dir, exist_ok=True)

if SAVE_ALIGNED:
    aligned_with_dir = os.path.join(output_base, "aligned_with")
    aligned_wo_dir = os.path.join(output_base, "aligned_wo")
    os.makedirs(aligned_with_dir, exist_ok=True)
    os.makedirs(aligned_wo_dir, exist_ok=True)

if SAVE_INDIVIDUAL_PHASE:
    phase_with_dir = os.path.join(output_base, "phase_with")
    phase_wo_dir = os.path.join(output_base, "phase_wo")
    os.makedirs(phase_with_dir, exist_ok=True)
    os.makedirs(phase_wo_dir, exist_ok=True)

print(f"\nOutput directory: {output_base}")

# %%
# ==================== Process All Positions ====================

print("\n" + "=" * 80)
print("Processing with-wo images")
print("=" * 80)

success_count = 0
fail_count = 0
failed_positions = []

# Colormap settings
if COLORMAP == "JET":
    cmap_cv = cv2.COLORMAP_JET
elif COLORMAP == "viridis":
    cmap_cv = cv2.COLORMAP_VIRIDIS
elif COLORMAP == "hot":
    cmap_cv = cv2.COLORMAP_HOT
else:
    cmap_cv = cv2.COLORMAP_JET

for pos_idx in tqdm(range(POS_START, POS_END + 1), desc="Processing positions"):
    pos_name = f"Pos{pos_idx}"
    
    try:
        # ==================== Process with image ====================
        with_path = os.path.join(BASE_DIR, WITH_DIR, pos_name, FILE_PATTERN)
        
        if not os.path.exists(with_path):
            raise FileNotFoundError(f"with image not found: {with_path}")

        # Load and crop with image
        with_img = np.array(Image.open(with_path))
        with_img = with_img[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]
        
        # QPI reconstruction
        field_with = get_field(with_img, params)
        angle_with = unwrap_phase(np.angle(field_with))

        # Background subtraction
        angle_with_nobg = angle_with - angle_bg

        # Gaussian background subtraction
        if GAUSS_ENABLED:
            angle_with_nobg, corr_with = gaussian_background_subtraction(
                angle_with_nobg,
                hist_min=HIST_MIN,
                hist_max=HIST_MAX,
                n_bins=N_BINS,
                smooth_window=SMOOTH_WINDOW,
                min_phase=MIN_PHASE
            )
        
        # ==================== Process wo image ====================
        wo_path = os.path.join(BASE_DIR, WO_DIR, pos_name, FILE_PATTERN)
        
        if not os.path.exists(wo_path):
            raise FileNotFoundError(f"wo image not found: {wo_path}")

        # Load and crop wo image
        wo_img = np.array(Image.open(wo_path))
        wo_img = wo_img[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]
        
        # QPI reconstruction
        field_wo = get_field(wo_img, params)
        angle_wo = unwrap_phase(np.angle(field_wo))

        # Background subtraction
        angle_wo_nobg = angle_wo - angle_bg

        # Gaussian background subtraction
        if GAUSS_ENABLED:
            angle_wo_nobg, corr_wo = gaussian_background_subtraction(
                angle_wo_nobg,
                hist_min=HIST_MIN,
                hist_max=HIST_MAX,
                n_bins=N_BINS,
                smooth_window=SMOOTH_WINDOW,
                min_phase=MIN_PHASE
            )
        
        # ==================== ECC Alignment (wo -> with) ====================
        # Convert to uint8
        with_uint8 = to_uint8(angle_with_nobg)
        wo_uint8 = to_uint8(angle_wo_nobg)
        
        # ECC alignment
        if WARP_MODE == cv2.MOTION_AFFINE:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:  # TRANSLATION or EUCLIDEAN
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                   ECC_ITERATIONS, ECC_EPS)
        
        try:
            _, warp_matrix = cv2.findTransformECC(
                with_uint8, wo_uint8, warp_matrix, WARP_MODE,
                criteria, inputMask=None, gaussFiltSize=1
            )
        except cv2.error as e:
            # If ECC does not converge, keep identity matrix (no alignment)
            if pos_idx <= POS_START + 5:  # Show warning only for first few
                print(f"\nWARNING: {pos_name}: ECC did not converge (continuing without alignment)")
        
        # Align wo image
        h, w = angle_with_nobg.shape
        angle_wo_aligned = cv2.warpAffine(
            angle_wo_nobg, warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        
        # ==================== Difference Calculation ====================
        diff = angle_wo_aligned - angle_with_nobg
        
        # ==================== Save ====================
        # Save difference TIF
        diff_path = os.path.join(diff_dir, f"{pos_name}_diff.tif")
        tifffile.imwrite(diff_path, diff.astype(np.float32))
        
        # Save colormap image
        diff_clipped = np.clip(diff, DIFF_VMIN, DIFF_VMAX)
        diff_norm = ((diff_clipped - DIFF_VMIN) / (DIFF_VMAX - DIFF_VMIN) * 255).astype(np.uint8)
        color_mapped = cv2.applyColorMap(diff_norm, cmap_cv)
        
        cmap_path = os.path.join(cmap_dir, f"{pos_name}_diff_cmap.png")
        cv2.imwrite(cmap_path, color_mapped)
        
        # Save aligned images (optional)
        if SAVE_ALIGNED:
            tifffile.imwrite(
                os.path.join(aligned_with_dir, f"{pos_name}_with.tif"),
                angle_with_nobg.astype(np.float32)
            )
            tifffile.imwrite(
                os.path.join(aligned_wo_dir, f"{pos_name}_wo_aligned.tif"),
                angle_wo_aligned.astype(np.float32)
            )
        
        # Save individual phase images (optional)
        if SAVE_INDIVIDUAL_PHASE:
            tifffile.imwrite(
                os.path.join(phase_with_dir, f"{pos_name}_phase.tif"),
                angle_with_nobg.astype(np.float32)
            )
            tifffile.imwrite(
                os.path.join(phase_wo_dir, f"{pos_name}_phase.tif"),
                angle_wo_nobg.astype(np.float32)
            )
        
        success_count += 1
        
    except Exception as e:
        fail_count += 1
        failed_positions.append((pos_name, str(e)))
        print(f"\nERROR: {pos_name}: {e}")
        continue

# %%
# ==================== Results Summary ====================

print("\n" + "=" * 80)
print("Processing Complete")
print("=" * 80)

print(f"\n[Results]")
print(f"  Success: {success_count} positions")
print(f"  Failed: {fail_count} positions")
print(f"  Total: {POS_END - POS_START + 1} positions")

if failed_positions:
    print(f"\n[Failed positions]")
    for pos_name, error in failed_positions[:10]:  # Show up to first 10
        print(f"  - {pos_name}: {error}")
    if len(failed_positions) > 10:
        print(f"  ... and {len(failed_positions) - 10} more")

print(f"\n[Output directories]")
print(f"  - Difference images (float32 TIF): {diff_dir}")
print(f"  - Colormap (PNG): {cmap_dir}")
if SAVE_ALIGNED:
    print(f"  - Aligned with: {aligned_with_dir}")
    print(f"  - Aligned wo: {aligned_wo_dir}")
if SAVE_INDIVIDUAL_PHASE:
    print(f"  - with phase images: {phase_with_dir}")
    print(f"  - wo phase images: {phase_wo_dir}")

print(f"\n[Settings]")
print(f"  - Crop region: ({CROP_Y_START}:{CROP_Y_END}, {CROP_X_START}:{CROP_X_END})")
print(f"  - Gaussian background subtraction: {'enabled' if GAUSS_ENABLED else 'disabled'}")
print(f"  - Difference display range: [{DIFF_VMIN}, {DIFF_VMAX}]")
print(f"  - Colormap: {COLORMAP}")
print(f"  - Alignment: {WARP_MODE_STR}")

print("\nAll processing complete!")

# %%
















