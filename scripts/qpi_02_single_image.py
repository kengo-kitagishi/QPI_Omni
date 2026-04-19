# %%
"""
Single image QPI reconstruction and verification script

Perform QPI reconstruction from a single image and background, and inspect the results in detail
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE, CROP_REGION
from figure_logger import setup_autosave
setup_autosave()

# %%
# ==================== Parameter Settings ====================

# Image path (modify before use)
path = r"D:\AquisitionData\Kitagishi\basler_image_seq\Basler_acA2440-75um__25176370__20260228_182040326_1553.tiff"
path_bg = r"D:\AquisitionData\Kitagishi\basler_image_seq\Basler_acA2440-75um__25176370__20260228_182040326_1553.tiff"

# Crop region
crop_slice = np.s_[CROP_REGION[0]:CROP_REGION[1], CROP_REGION[2]:CROP_REGION[3]]

# QPI parameters
offaxis_center = OFFAXIS_CENTER  # Managed in optical_config.py

# Save settings
SAVE_OUTPUT = False  # Set to True to save results
output_path = "/Users/kitak/QPI/output/angle_nobg.tif"

print("=" * 80)
print("Single Image QPI Reconstruction")
print("=" * 80)
print(f"\nSample image: {path}")
print(f"BG image: {path_bg}")

# %%
# ==================== Load Images ====================

print("\nLoading images...")

img = Image.open(path)
img = np.array(img)
img = img[crop_slice]

img_bg = Image.open(path_bg)
img_bg = np.array(img_bg)
img_bg = img_bg[crop_slice]

print(f"Image size: {img.shape}")

# Display images
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Sample Image')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(img_bg, cmap='gray')
plt.title('Background Image')
plt.colorbar()
plt.tight_layout()
plt.show()

# %%
# ==================== QPI Parameter Setup ====================

params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=img.shape,
    pixelsize=PIXELSIZE,
    offaxis_center=offaxis_center,
)

print("\nQPI parameters:")
print(f"  - Wavelength: {WAVELENGTH*1e9:.1f} nm")
print(f"  - NA: {NA}")
print(f"  - Pixel size: {PIXELSIZE*1e6:.3f} um")
print(f"  - Off-axis center: {offaxis_center}")
print(f"  - Aperture size: {params.aperturesize} pixels")

# %%
# ==================== FFT Display ====================

img_fft = np.fft.fftshift(np.fft.fft2(img_bg))
fft_log = np.log(np.abs(img_fft) + 1)

radius = params.aperturesize // 2
circle_center = (offaxis_center[1], offaxis_center[0])  # matplotlib uses (x, y) = (col, row)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(fft_log, cmap='gray')
circle = plt.Circle(circle_center, radius, color='red', fill=False, linewidth=1.5)
ax.add_patch(circle)
ax.set_title(f'FFT (background)  offaxis={offaxis_center}  r={radius}px')
plt.tight_layout()
plt.show()

# %%
# ==================== QPI Reconstruction ====================

print("\nPerforming QPI reconstruction...")

# Get complex field
field = get_field(img, params)
field_bg = get_field(img_bg, params)

# Phase unwrapping
angle = unwrap_phase(np.angle(field))
angle_bg = unwrap_phase(np.angle(field_bg))

print("Reconstruction complete")

# %%
# ==================== Phase Image Display ====================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].imshow(angle, cmap='viridis')
axes[0].set_title('Sample Phase')
plt.colorbar(im1, ax=axes[0], label='Phase (rad)')

im2 = axes[1].imshow(angle_bg, cmap='viridis')
axes[1].set_title('Background Phase')
plt.colorbar(im2, ax=axes[1], label='Phase (rad)')

plt.tight_layout()
plt.show()

# %%
# ==================== Background Subtraction ====================

print("\nComputing background subtraction...")

angle_nobg = angle - angle_bg

# Zero-mean adjustment (specify background region)
bg_region = np.s_[1:100, 1:254]
mean_value = np.mean(angle_nobg[bg_region])
angle_nobg = angle_nobg - mean_value

print(f"Background region mean: {mean_value:.6f} rad")
print(f"Adjusted mean: {np.mean(angle_nobg):.6e} rad")

# Phase image after background subtraction
plt.figure(figsize=(10, 8))
plt.imshow(angle_nobg, vmin=-0.1, vmax=0.1, cmap='viridis')
plt.colorbar(label='Phase (rad)')
plt.title('Background-subtracted Phase Map')
plt.tight_layout()
plt.show()

# %%
# ==================== Statistics ====================

print("\n" + "=" * 80)
print("Phase Image Statistics")
print("=" * 80)

print(f"\n[Overall]")
print(f"  Mean: {np.mean(angle_nobg):.6e} rad")
print(f"  Std: {np.std(angle_nobg):.6f} rad")
print(f"  Min: {np.min(angle_nobg):.6f} rad")
print(f"  Max: {np.max(angle_nobg):.6f} rad")

# Histogram
plt.figure(figsize=(10, 5))
plt.hist(angle_nobg.flatten(), bins=100, alpha=0.7, edgecolor='black')
plt.xlabel('Phase (rad)')
plt.ylabel('Frequency')
plt.title('Phase Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# ==================== Phase Profile ====================

print("\nDisplaying phase profiles...")

# Horizontal profile (image center)
y_coord = img.shape[0] // 2
profile_horizontal = angle_nobg[y_coord, :]

# Vertical profile (image center)
x_coord = img.shape[1] // 2
profile_vertical = angle_nobg[:, x_coord]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Horizontal profile
axes[0].plot(profile_horizontal, linewidth=2)
axes[0].set_title(f'Horizontal Profile (y = {y_coord})')
axes[0].set_xlabel('X coordinate (pixels)')
axes[0].set_ylabel('Phase (rad)')
axes[0].grid(True, alpha=0.3)

# Vertical profile
axes[1].plot(profile_vertical, linewidth=2)
axes[1].set_title(f'Vertical Profile (x = {x_coord})')
axes[1].set_xlabel('Y coordinate (pixels)')
axes[1].set_ylabel('Phase (rad)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# ==================== Colormap Comparison ====================

print("\nDisplaying with various colormaps...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
cmaps = ['viridis', 'plasma', 'inferno', 'RdBu_r', 'seismic', 'coolwarm']

for idx, cmap in enumerate(cmaps):
    ax = axes[idx // 3, idx % 3]
    im = ax.imshow(angle_nobg, vmin=-0.1, vmax=0.1, cmap=cmap)
    ax.set_title(f'Colormap: {cmap}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.show()

# %%
# ==================== Amplitude Image (Optional) ====================

print("\nDisplaying amplitude images...")

amplitude = np.abs(field)
amplitude_bg = np.abs(field_bg)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].imshow(amplitude, cmap='gray')
axes[0].set_title('Sample Amplitude')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(amplitude_bg, cmap='gray')
axes[1].set_title('Background Amplitude')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# %%
# ==================== Save (Optional) ====================

if SAVE_OUTPUT:
    print(f"\nSaving results: {output_path}")
    tifffile.imwrite(output_path, angle_nobg.astype(np.float32))
    
    # Also save as PNG with colormap
    png_path = output_path.replace('.tif', '_colormap.png')
    plt.figure(figsize=(10, 8))
    plt.imshow(angle_nobg, cmap='viridis', vmin=-0.1, vmax=0.1)
    plt.colorbar(label='Phase (rad)')
    plt.title('Background-subtracted Phase Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  - TIF: {output_path}")
    print(f"  - PNG: {png_path}")
else:
    print("\nTo save results, set SAVE_OUTPUT = True")

# %%
# ==================== Done ====================

print("\n" + "=" * 80)
print("QPI Reconstruction Complete")
print("=" * 80)

print("\n[Checklist]")
print("  - Phase image is properly reconstructed")
print("  - Background subtraction is correct")
print("  - Profile is smooth")
print("  - No outliers or artifacts")

print("\n[Next steps]")
print("  - If OK -> proceed to batch processing (qpi_03_batch_reconstruction.py)")
print("  - If results are inadequate -> adjust focus (qpi_01_focus_setup.py)")

print("\nSingle image processing script completed")

# %%


