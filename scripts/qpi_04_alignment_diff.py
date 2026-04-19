# %%
"""
Alignment and difference analysis script

CSV-based alignment, difference image generation from the first frame, colormap creation
"""

import os
import numpy as np
import pandas as pd
import tifffile
import cv2
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# %%
# ==================== Parameter Settings ====================

# Path settings (modify before use)
csv_path = "/Volumes/QPI/ph_1/Pos3/output_phase/Results.csv"
image_dir = "/Volumes/QPI/ph_1/Pos3/output_phase"

# Alignment method selection
# "left_center": midpoint of the left edge
# "right_center": midpoint of the right edge
# "center": center of the bounding rectangle
alignment_point = "left_center"

# Colormap settings
vmin, vmax = -0.5, 3.0
colormap = "viridis"  # viridis, plasma, JET, RdBu_r, etc.

print("=" * 80)
print("Alignment and Difference Analysis")
print("=" * 80)
print(f"\nCSV path: {csv_path}")
print(f"Image directory: {image_dir}")
print(f"Alignment reference: {alignment_point}")

# %%
# ==================== Cell 1: CSV-based Alignment (Left Edge Midpoint) ====================

print("\n" + "=" * 80)
print("CSV-based Alignment (Left Edge Midpoint)")
print("=" * 80)

# Output directory
output_dir = os.path.join(image_dir, "aligned_left_center")
os.makedirs(output_dir, exist_ok=True)

# Load CSV and sort by Slice
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

print(f"\nCSV rows: {len(df)}")

# Get midpoint coordinates of the left edge
x = df["BX"]
y = df["BY"] + df["Height"] / 2

# Reference frame (first slice) coordinates
x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

print(f"Reference position: ({x0:.2f}, {y0:.2f})")
print(f"Max shift: dx={dx.abs().max():.2f}, dy={dy.abs().max():.2f}")

# Load images (natural sort)
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))

# Exclude files containing 'aligned_'
image_paths = [p for p in image_paths if 'aligned' not in os.path.basename(p)]

print(f"Number of images: {len(image_paths)}")

if len(image_paths) != len(dx):
    print(f"WARNING: Number of images ({len(image_paths)}) does not match CSV rows ({len(dx)})")
    print("   Processing will use the CSV row count")
    image_paths = image_paths[:len(dx)]

# Alignment processing
print("\nPerforming alignment...")
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print(f"Alignment complete: {output_dir}")

# %%
# ==================== Cell 2: CSV-based Alignment (Right Edge Midpoint) ====================

print("\n" + "=" * 80)
print("CSV-based Alignment (Right Edge Midpoint)")
print("=" * 80)

# Output directory
output_dir = os.path.join(image_dir, "aligned_right_center")
os.makedirs(output_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# Compute midpoint coordinates of the right edge
x = df["BX"] + df["Width"]
y = df["BY"] + df["Height"] / 2

x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

print(f"Reference position: ({x0:.2f}, {y0:.2f})")
print(f"Max shift: dx={dx.abs().max():.2f}, dy={dy.abs().max():.2f}")

# Load images
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
image_paths = [p for p in image_paths if 'aligned' not in os.path.basename(p)]
image_paths = image_paths[:len(dx)]

# Alignment processing
print("\nPerforming alignment...")
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print(f"Alignment complete: {output_dir}")

# %%
# ==================== Cell 3: Difference Image Generation from First Frame ====================

print("\n" + "=" * 80)
print("Difference Image Generation from First Frame")
print("=" * 80)

# Aligned image directory (modify before use)
aligned_dir = os.path.join(image_dir, "aligned_left_center")
output_diff_dir = os.path.join(aligned_dir, "diff_from_first")
os.makedirs(output_diff_dir, exist_ok=True)

# Load and sort files
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))

if len(image_paths) < 2:
    print("Fewer than 2 images available")
else:
    print(f"Number of images: {len(image_paths)}")

    # Load first frame as reference
    ref_img = tifffile.imread(image_paths[0]).astype(np.float32)
    print(f"Reference image: {os.path.basename(image_paths[0])}")

    # Save difference images
    print("\nComputing differences...")
    for i, path in enumerate(image_paths[1:], start=1):
        img = tifffile.imread(path).astype(np.float32)
        diff = img - ref_img
        fname = os.path.basename(path)
        tifffile.imwrite(os.path.join(output_diff_dir, fname), diff)
    
    print(f"Difference images saved: {output_diff_dir}")

# %%
# ==================== Cell 4: Colormap Difference Image Generation ====================

print("\n" + "=" * 80)
print("Colormap Difference Image Generation")
print("=" * 80)

# Output directory
output_colormap_dir = os.path.join(aligned_dir, "diff_colormap")
os.makedirs(output_colormap_dir, exist_ok=True)

# Load files
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

print(f"Difference display range: [{vmin}, {vmax}]")
print(f"Colormap: {colormap}")

# Colormap settings
if colormap.lower() == "jet":
    cmap_cv = cv2.COLORMAP_JET
elif colormap.lower() == "viridis":
    cmap_cv = cv2.COLORMAP_VIRIDIS
elif colormap.lower() == "hot":
    cmap_cv = cv2.COLORMAP_HOT
else:
    cmap_cv = cv2.COLORMAP_JET

# Difference + colormap processing
print("\nGenerating colormap images...")
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    
    # Clip to specified range and normalize (0-255)
    diff_clipped = np.clip(diff, vmin, vmax)
    diff_norm = ((diff_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    
    # Apply colormap
    color_mapped = cv2.applyColorMap(diff_norm, cmap_cv)
    
    # Save
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap.png"
    cv2.imwrite(os.path.join(output_colormap_dir, fname), color_mapped)

print(f"Colormap images saved: {output_colormap_dir}")

# %%
# ==================== Cell 5: Matplotlib Colormap (High Quality) ====================

print("\n" + "=" * 80)
print("Matplotlib Colormap (High Quality)")
print("=" * 80)

# Output directory
output_colormap_mpl_dir = os.path.join(aligned_dir, "diff_colormap_matplotlib")
os.makedirs(output_colormap_mpl_dir, exist_ok=True)

# Colormap settings
cmap_mpl = cm.get_cmap(colormap)
norm = Normalize(vmin=vmin, vmax=vmax)

# Load files
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

print(f"Colormap: {colormap}")
print(f"Difference display range: [{vmin}, {vmax}]")

# Difference + colormap processing
print("\nGenerating matplotlib colormap images...")
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    
    # Normalize and convert to colormap (RGBA)
    rgba_image = cmap_mpl(norm(diff))  # shape: (H, W, 4)
    rgb_image = (rgba_image[..., :3] * 255).astype(np.uint8)
    
    # Save TIF
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap_mpl.tif"
    tifffile.imwrite(os.path.join(output_colormap_mpl_dir, fname), rgb_image)

print(f"Matplotlib colormap images saved: {output_colormap_mpl_dir}")

# %%
# ==================== Done ====================

print("\n" + "=" * 80)
print("Alignment and Difference Analysis Script Complete")
print("=" * 80)

print("\n[Usage]")
print("  1. Set the CSV path and image directory in the parameter settings cell")
print("  2. Run the required cells:")
print("     - Cell 1: Alignment by left edge midpoint")
print("     - Cell 2: Alignment by right edge midpoint")
print("     - Cell 3: Difference from first frame")
print("     - Cell 4: OpenCV colormap")
print("     - Cell 5: Matplotlib colormap (high quality)")

print("\n[Next steps]")
print("  - Inspect difference images")
print("  - Time series analysis")
print("  - Statistical analysis")

# %%
















