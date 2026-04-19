#!/usr/bin/env python3
"""
Simplest mean RI calculation

Computed from Results.csv only
- IntDen = total phase
- Major/Minor -> rod shape volume
- mean RI = n_medium + (IntDen x lambda x pixel_area) / (2pi x volume)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figure_logger import setup_autosave
setup_autosave()

# =============================================================================
# Parameter settings (edit only this section)
# =============================================================================
RESULTS_CSV = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results.csv"
OUTPUT_FILE = "simplest_mean_ri.png"

PIXEL_SIZE_UM = 0.348
WAVELENGTH_NM = 663
N_MEDIUM = 1.333
ALPHA_RI = 0.00018

# =============================================================================
# Calculation
# =============================================================================

print("="*60)
print("Simplest Ellipse RI Analysis")
print("="*60)

# Read CSV
print(f"\nReading: {RESULTS_CSV}")
df = pd.read_csv(RESULTS_CSV)
print(f"  {len(df)} ROIs")
print(f"  Columns: {', '.join(df.columns[:10])}...")

# Find IntDen column name
intden_col = None
for col in ['IntDen', 'RawIntDen', 'Integrated Density', 'IntegratedDensity']:
    if col in df.columns:
        intden_col = col
        break

if intden_col is None:
    print("\nError: IntDen column not found!")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

print(f"  Using '{intden_col}' as total phase")

# Frame number
if 'Slice' in df.columns:
    df['frame'] = df['Slice']
elif 'Label' in df.columns:
    import re
    df['frame'] = df['Label'].str.extract(r'(\d{4})').astype(float)
else:
    df['frame'] = range(1, len(df) + 1)

# Rod shape volume calculation
def calc_rod_volume(major_px, minor_px, pixel_size_um):
    length = major_px * pixel_size_um
    width = minor_px * pixel_size_um
    r = width / 2.0
    h = length - 2 * r
    
    if h < 0:
        return (4.0 / 3.0) * np.pi * (r ** 3)
    else:
        return (4.0 / 3.0) * np.pi * (r ** 3) + np.pi * (r ** 2) * h

# Volume calculation
major_col = 'Major' if 'Major' in df.columns else 'Feret'
minor_col = 'Minor' if 'Minor' in df.columns else 'MinFeret'

df['volume_um3'] = df.apply(
    lambda row: calc_rod_volume(row[major_col], row[minor_col], PIXEL_SIZE_UM), 
    axis=1
)

# total phase
df['total_phase'] = df[intden_col]

# Mean RI calculation
wavelength_um = WAVELENGTH_NM * 1e-3
pixel_area_um2 = PIXEL_SIZE_UM ** 2

df['mean_ri'] = N_MEDIUM + (df['total_phase'] * wavelength_um * pixel_area_um2) / (2 * np.pi * df['volume_um3'])

# Mass calculation
df['mean_conc'] = (df['mean_ri'] - N_MEDIUM) / ALPHA_RI
df['mass_pg'] = df['mean_conc'] * df['volume_um3']

# Display results
print(f"\nResults:")
print(f"  Volume: {df['volume_um3'].min():.1f} - {df['volume_um3'].max():.1f} µm³")
print(f"  Mean RI: {df['mean_ri'].min():.4f} - {df['mean_ri'].max():.4f}")
print(f"  Mass: {df['mass_pg'].min():.1f} - {df['mass_pg'].max():.1f} pg")

# =============================================================================
# Plot
# =============================================================================

print(f"\nCreating plot...")

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle('Simplest Ellipse RI Analysis (from Results.csv only)', 
             fontsize=14, fontweight='bold')

# Volume
axes[0].plot(df['frame'], df['volume_um3'], color='#1f77b4', 
             linewidth=2)
axes[0].set_ylabel('Volume (µm³)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, linestyle='--')

# Mean RI
axes[1].plot(df['frame'], df['mean_ri'], color='#ff7f0e', 
             linewidth=2)
axes[1].set_ylabel('Mean RI', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, linestyle='--')

# Mass
axes[2].plot(df['frame'], df['mass_pg'], color='#2ca02c', 
             linewidth=2)
axes[2].set_xlabel('Frame', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Total Mass (pg)', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_FILE}")

# Save CSV
csv_out = OUTPUT_FILE.replace('.png', '.csv')
df[['frame', 'volume_um3', 'mean_ri', 'mass_pg', 'total_phase']].to_csv(csv_out, index=False)
print(f"  Saved: {csv_out}")

plt.show()

print("\nDone!")

