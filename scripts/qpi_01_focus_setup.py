# %%
"""
Focus adjustment and optical system setup script

Used for focus position adjustment, off-axis center verification,
and visibility evaluation before microscope acquisition
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from qpi_common import visibility, create_qpi_params, WAVELENGTH, NA, PIXELSIZE
from CursorVisualizer import CursorVisualizer

# %%
# ==================== Parameter Settings ====================

# Image path (modify before use)
path = "/Volumes/QPI3/250910_kk/ph_1/Pos1/img_000000000_Default_000.tif"
path_bg = "/Volumes/QPI3/250910_kk/ph_1/Pos1/img_000000000_Default_000.tif"

# Crop region (adjust as needed)
crop_slice = np.s_[8:2056, 208:2256]
# e.g.: crop_slice = np.s_[8:2056, 416:2464]
# e.g.: crop_slice = np.s_[516:1540, 500:1524]

# Initial off-axis center (adjust after FFT verification)
offaxis_center = (1504, 1708)

print("=" * 80)
print("Focus adjustment and optical system setup")
print("=" * 80)
print(f"\nImage path: {path}")
print(f"BG image path: {path_bg}")
print(f"Crop region: {crop_slice}")

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
# ==================== FFT Display and Off-axis Center Verification ====================

print("\nComputing FFT...")

img_fft = np.fft.fftshift(np.fft.fft2(img_bg))

plt.figure(figsize=(10, 8))
plt.imshow(np.log(np.abs(img_fft)))
plt.title('FFT (log scale)')
plt.colorbar()
plt.show()

# %%
# ==================== Detailed Inspection with CursorVisualizer ====================

print("\nLaunching CursorVisualizer...")
print("  - Move the mouse over the FFT image to display coordinates")
print("  - Verify the off-axis peak coordinates")

cb = CursorVisualizer(np.log(np.abs(img_fft)))
cb.run()

# Enter the coordinates verified with CursorVisualizer here
# offaxis_center = (y_coord, x_coord)
# e.g.: offaxis_center = (1504, 1708)

print(f"\nCurrent off-axis center: {offaxis_center}")
print("  * Modify the above value as needed")

# %%
# ==================== QPI Parameter Setup and Aperture Verification ====================

IMG_SHAPE = img.shape

params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=IMG_SHAPE,
    pixelsize=PIXELSIZE,
    offaxis_center=offaxis_center,
)

print("\nQPI parameters:")
print(f"  - Wavelength: {WAVELENGTH*1e9:.1f} nm")
print(f"  - NA: {NA}")
print(f"  - Pixel size: {PIXELSIZE*1e6:.3f} um")
print(f"  - Off-axis center: {offaxis_center}")
print(f"  - Aperture size: {params.aperturesize} pixels")

# Overlay aperture circle on FFT image
fft_image = np.log(np.abs(img_fft))

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(fft_image)

# Draw aperture circle
aperture_size = params.aperturesize
radius = aperture_size // 2
circle_center = (offaxis_center[1], offaxis_center[0])  # Convert to (x, y)
circle = plt.Circle(circle_center, radius, color='red', fill=False, linewidth=2)
ax.add_patch(circle)

ax.set_title("FFT with Aperture Circle (red)")
plt.tight_layout()
plt.show()

print("\nVerify that the red circle properly covers the off-axis peak")

# %%
# ==================== Phase Image Reconstruction ====================

print("\nReconstructing phase image...")

field = get_field(img, params)
field_bg = get_field(img_bg, params)

angle = unwrap_phase(np.angle(field))
angle_bg = unwrap_phase(np.angle(field_bg))

# Display phase images
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
# Specify the appropriate region before use
bg_region = np.s_[1:100, 1:254]
angle_nobg = angle_nobg - np.mean(angle_nobg[bg_region])

plt.figure(figsize=(10, 8))
plt.imshow(angle_nobg, vmin=-0.1, vmax=0.1, cmap='viridis')
plt.colorbar(label='Phase (rad)')
plt.title('Background-subtracted Phase Map')
plt.tight_layout()
plt.show()

print(f"Phase image statistics:")
print(f"  - Mean: {np.mean(angle_nobg):.6f} rad")
print(f"  - Std: {np.std(angle_nobg):.6f} rad")
print(f"  - Min: {np.min(angle_nobg):.6f} rad")
print(f"  - Max: {np.max(angle_nobg):.6f} rad")

# %%
# ==================== Phase Profile Inspection ====================

print("\nInspecting phase profile...")

# Get profile at an arbitrary x coordinate
x_coord = img.shape[1] // 2  # Center
profile = angle_nobg[:, x_coord]

plt.figure(figsize=(10, 6))
plt.plot(profile, linewidth=2)
plt.title(f'Phase Profile at x = {x_coord}')
plt.xlabel('Y coordinate (pixels)')
plt.ylabel('Phase (rad)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"  Displayed phase profile at x = {x_coord}")
print("  * When in focus, the profile should be smooth")

# %%
# ==================== Visibility Evaluation ====================

print("\nComputing visibility...")

vis_map = visibility(img, params)

plt.figure(figsize=(10, 8))
plt.imshow(vis_map, vmin=0.3, vmax=0.95, cmap='viridis')
plt.colorbar(label='Visibility')
plt.title('Visibility Map')
plt.tight_layout()
plt.show()

# Visibility histogram
plt.figure(figsize=(10, 5))
plt.hist(vis_map.flatten(), bins=100, range=(0.3, 1.0), alpha=0.7, edgecolor='black')
plt.xlabel('Visibility')
plt.ylabel('Frequency')
plt.title('Visibility Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

vis_mean = vis_map.mean()
print(f"\nVisibility statistics:")
print(f"  - Mean: {vis_mean:.4f}")
print(f"  - Std: {vis_map.std():.4f}")
print(f"  - Min: {vis_map.min():.4f}")
print(f"  - Max: {vis_map.max():.4f}")

if vis_mean > 0.7:
    print("\nVisibility is good (> 0.7)")
elif vis_mean > 0.5:
    print("\nVisibility is slightly low (0.5-0.7) - consider adjustment")
else:
    print("\nVisibility is low (< 0.5) - optical system adjustment required")

# %%
# ==================== Focus Check Summary ====================

print("\n" + "=" * 80)
print("Focus Check Summary")
print("=" * 80)

print("\n[Checklist]")
print(f"  1. Off-axis center: {offaxis_center}")
print(f"  2. Aperture size: {params.aperturesize} pixels")
print(f"  3. Mean visibility: {vis_mean:.4f}")
print(f"  4. Phase image std: {np.std(angle_nobg):.6f} rad")

print("\n[Next steps]")
print("  - If visibility is low -> adjust focus position")
print("  - If off-axis peak is unclear -> adjust optical axis")
print("  - If everything looks good -> start acquisition")

print("\nFocus adjustment script completed")

# %%
















