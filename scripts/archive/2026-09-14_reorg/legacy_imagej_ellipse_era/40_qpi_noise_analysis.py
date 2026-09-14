# %%
# ============================================================
# qpi_noise_analysis.py — QPI noise analysis
# ============================================================
# UC1: Arbitrary position profile from a single image
# UC2: Timelapse noise tracking
# UC3: Correlation between shift magnitude and noise
# ============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from figure_logger import save_figure

# ============================================================
# Common settings (can be overridden per UC)
# ============================================================

WAVELENGTH  = 658e-9        # m
NA          = 0.95
PIXELSIZE   = 3.45e-6 / 40  # m/px (3.45 um camera pixel, 40x objective)
OFFAXIS_CENTER = (1710, 644) # (row, col) — confirmed via CursorVisualizer

# ============================================================
# UC_SENSOR: Camera temporal noise measurement (sensor noise evaluation)
#
# Objective: Measure sigma_sensor [e-] (corresponds to the purple line
#            parameter in Fig 3.6 of the paper)
#
# Procedure:
#   1. Capture 100 frames with laser/illumination completely OFF (dark frames)
#   2. Compute temporal STD per pixel in ADU
#   3. Convert to e- using EMVA1288 conversion gain
#   4. Report sigma_sensor [e-] + visualize temporal STD map
#
# Requirements:
#   - Directory containing 100 dark frames (TIF or PNG)
#   - Conversion gain [e-/ADU] from Basler EMVA1288 report
# ============================================================

# %%
# --- UC_SENSOR settings ---

SENSOR_DARK_DIR = "path/to/dark_frames_dir"  # Directory containing cropped 2048x2048 dark frames (laser OFF)

# --- Basler acA2440-75um parameters (from EMVA1288 report) ---
# EMVA1288 report: Basler website -> camera search -> Downloads -> EMVA1288 Data Sheet
SENSOR_FULL_WELL_E  = 10340     # [e-] Full Well Capacity (EMVA1288 value)
SENSOR_BIT_DEPTH    = 12        # Effective bit depth (12 or 16)
SENSOR_GAIN         = 1.0       # Software gain setting (fixed at 0 dB = 1.0 in Pylon)
# Compute conversion gain from the above: ADU -> e-
# If EMVA1288 lists "Overall System Gain" [e-/DN] directly, use that value instead
SENSOR_CONVERSION_GAIN = SENSOR_FULL_WELL_E / (2 ** SENSOR_BIT_DEPTH) / SENSOR_GAIN
# e.g. 10340 / 4096 / 1.0 ≈ 2.52 e⁻/ADU

# Same as paper (Fig 3.6): report the average of 80x80 px in the temporal STD map
# "plot the average of 80 pixels × 80 pixels in the temporal STD map as the temporal OPD noise"
SENSOR_ROI_SIZE = 80   # Value matching the paper. Modify only here if needed
# ROI position: placed at image center (uniform in dark, so any position works, but avoid edges)
# Automatically computed from image size at runtime, so set to None here
SENSOR_ROI_CENTER = None   # None -> auto-use image center. Can specify (row, col) explicitly

SENSOR_N_FRAMES_EXPECTED = 100      # Max number of frames to load (None to use all)

# %%
# --- UC_SENSOR execution ---

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from figure_logger import save_figure

def _load_frames(directory, n_max=None):
    """Load images from directory in sorted order, return (H, W, N) array"""
    exts = {".tif", ".tiff", ".png"}
    files = sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in exts
    )
    if n_max is not None:
        files = files[:n_max]
    if not files:
        raise FileNotFoundError(f"No image files found: {directory}")

    frames = []
    for fname in files:
        path = os.path.join(directory, fname)
        try:
            img = tifffile.imread(path).astype(np.float64)
        except Exception:
            img = np.array(Image.open(path)).astype(np.float64)
        frames.append(img)

    stack = np.stack(frames, axis=-1)  # (H, W, N)
    print(f"  Loaded: {stack.shape[2]} frames, shape={stack.shape[:2]}, dtype=float64")
    return stack

print("=== UC_SENSOR: Camera temporal noise measurement ===")
print(f"  Conversion gain: {SENSOR_CONVERSION_GAIN:.4f} e⁻/ADU")
print(f"  (full_well={SENSOR_FULL_WELL_E} e⁻, bit_depth={SENSOR_BIT_DEPTH}, gain={SENSOR_GAIN})")

# Load frames
dark_stack = _load_frames(SENSOR_DARK_DIR, n_max=SENSOR_N_FRAMES_EXPECTED)  # (H, W, N)
H, W, N = dark_stack.shape
print(f"  Number of frames: {N}")

# Compute temporal STD per pixel [ADU]
std_map_adu = np.std(dark_stack, axis=2, ddof=1)   # (H, W)
mean_map_adu = np.mean(dark_stack, axis=2)          # (H, W)

# Convert to e-
std_map_e   = std_map_adu  * SENSOR_CONVERSION_GAIN
mean_map_e  = mean_map_adu * SENSOR_CONVERSION_GAIN

# Place 80x80 ROI at image center (aggregation method matching the paper)
half = SENSOR_ROI_SIZE // 2
if SENSOR_ROI_CENTER is None:
    cr, cc = H // 2, W // 2   # Image center
else:
    cr, cc = SENSOR_ROI_CENTER
rs, re = cr - half, cr + half
cs, ce = cc - half, cc + half
print(f"  ROI: rows {rs}:{re}, cols {cs}:{ce}  ({SENSOR_ROI_SIZE}x{SENSOR_ROI_SIZE} px, paper-consistent)")

roi_std_e   = std_map_e[rs:re, cs:ce]
roi_mean_e  = mean_map_e[rs:re, cs:ce]

sigma_sensor     = float(np.mean(roi_std_e))
sigma_sensor_std = float(np.std(roi_std_e))
mean_dark_e      = float(np.mean(roi_mean_e))

print(f"\n--- Results ---")
print(f"  sigma_sensor (ROI mean) = {sigma_sensor:.1f} +/- {sigma_sensor_std:.1f} e-")
print(f"  mean dark level         = {mean_dark_e:.1f} e-  (dark current + offset)")
print(f"  ROI: rows {rs}:{re}, cols {cs}:{ce}  ({(re-rs)*(ce-cs)} pixels)")

# --- Fig 1: temporal STD map ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].imshow(std_map_e, cmap="hot", vmax=np.percentile(std_map_e, 99))
axes[0].set_title("Temporal STD map [e⁻]")
axes[0].set_xlabel("Column")
axes[0].set_ylabel("Row")
plt.colorbar(im0, ax=axes[0], label="STD [e⁻]")

# Draw rectangle around ROI
from matplotlib.patches import Rectangle
rect = Rectangle((cs, rs), ce - cs, re - rs, linewidth=1.5, edgecolor="cyan", facecolor="none",
                 label=f"{SENSOR_ROI_SIZE}x{SENSOR_ROI_SIZE} ROI (paper-consistent)")
axes[0].add_patch(rect)

# STD histogram within ROI
axes[1].hist(roi_std_e.ravel(), bins=50, color="steelblue", edgecolor="white")
axes[1].axvline(sigma_sensor, color="red", linestyle="--", label=f"mean={sigma_sensor:.1f} e⁻")
axes[1].set_xlabel("Temporal STD [e⁻]")
axes[1].set_ylabel("Pixel count")
axes[1].set_title("STD distribution (ROI)")
axes[1].legend()

# Time series: ROI mean value trend (drift check)
roi_timeseries = dark_stack[rs:re, cs:ce, :].mean(axis=(0, 1)) * SENSOR_CONVERSION_GAIN
axes[2].plot(roi_timeseries, lw=0.8)
axes[2].set_xlabel("Frame index")
axes[2].set_ylabel("ROI mean [e⁻]")
axes[2].set_title("Dark level timeseries (drift check)")
axes[2].grid(True, alpha=0.4)

plt.suptitle(
    f"Camera sensor noise  |  σ_sensor = {sigma_sensor:.1f} e⁻  |  "
    f"conversion gain = {SENSOR_CONVERSION_GAIN:.3f} e⁻/ADU  |  N = {N} frames",
    fontsize=11
)
plt.tight_layout()

save_figure(
    fig,
    params={
        "n_frames":         N,
        "conversion_gain":  SENSOR_CONVERSION_GAIN,
        "full_well_e":      SENSOR_FULL_WELL_E,
        "bit_depth":        SENSOR_BIT_DEPTH,
        "gain_setting":     SENSOR_GAIN,
        "roi_size":         SENSOR_ROI_SIZE,
        "roi_center":       (int(cr), int(cc)),
        "sigma_sensor_e":   round(sigma_sensor, 1),
    },
    description=f"Camera sensor noise measurement: sigma_sensor={sigma_sensor:.1f} e- (dark frames, N={N})",
    data_source={
        "raw_files": [SENSOR_DARK_DIR],
        "notes":     "Basler acA2440-75um, laser OFF, Pylon gain=0dB",
    },
    data={
        "std_map_e":      std_map_e,
        "roi_timeseries": roi_timeseries,
    },
)

print(f"\nReference: Value to substitute into Var(z_sensor) in the OPD noise formula = {sigma_sensor:.1f}^2 = {sigma_sensor**2:.0f} e-^2")

# ============================================================
# UC1: Arbitrary position profile from a single image
# ============================================================

# %%
# --- UC1 settings ---

UC1_PATH    = "path/to/image.tif"       # TIFF path of the frame to analyze
UC1_PATH_BG = "path/to/background.tif" # Background TIFF path
UC1_CROP    = (8, 2056, 208, 2256)      # (row_start, row_end, col_start, col_end)
UC1_PROFILE_COORD = 200                 # Column index for profile extraction
UC1_PROFILE_AXIS  = "x"                # "x" = column-direction slice, "y" = row-direction slice

# %%
# --- UC1 execution ---

img    = np.array(Image.open(UC1_PATH))[UC1_CROP[0]:UC1_CROP[1], UC1_CROP[2]:UC1_CROP[3]]
img_bg = np.array(Image.open(UC1_PATH_BG))[UC1_CROP[0]:UC1_CROP[1], UC1_CROP[2]:UC1_CROP[3]]

# To load from npy instead, uncomment below:
# angle_nobg = np.load("path/to/angle_nobg.npy")

params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=img.shape,
    pixelsize=PIXELSIZE,
    offaxis_center=OFFAXIS_CENTER,
)

field    = get_field(img, params)
field_bg = get_field(img_bg, params)
angle_nobg = unwrap_phase(np.angle(field)) - unwrap_phase(np.angle(field_bg))
angle_nobg -= np.mean(angle_nobg[1:100, 1:254])

# Display phase map
fig, ax = plt.subplots()
im = ax.imshow(angle_nobg, cmap="viridis", vmin=-0.5, vmax=0.5)
plt.colorbar(im, ax=ax, label="Phase (rad)")
ax.set_title("Background-subtracted Phase Map")
plt.tight_layout()
save_figure(fig, params={"crop": UC1_CROP, "offaxis_center": OFFAXIS_CENTER},
            description="UC1 phase map",
            data={"phase_map": angle_nobg})

# %%
# Profile plot
# When UC1_PROFILE_AXIS == "x":
#   angle_nobg[:, x_coord] -> extract all rows at column x_coord
#   -> x-axis is row index (Y direction)
if UC1_PROFILE_AXIS == "x":
    profile = angle_nobg[:, UC1_PROFILE_COORD]
    xlabel  = "Row index (Y direction, pixels)"
    title   = f"Phase profile at column (X) = {UC1_PROFILE_COORD}"
else:
    # When "y": extract all columns at row y_coord
    profile = angle_nobg[UC1_PROFILE_COORD, :]
    xlabel  = "Column index (X direction, pixels)"
    title   = f"Phase profile at row (Y) = {UC1_PROFILE_COORD}"

fig, ax = plt.subplots()
ax.plot(profile)
ax.set_title(title)
ax.set_xlabel(xlabel)
ax.set_ylabel("Phase (rad)")
ax.grid(True)
save_figure(fig, params={"axis": UC1_PROFILE_AXIS, "coord": UC1_PROFILE_COORD},
            description="UC1 phase profile",
            data={"profile": profile})

# ============================================================
# UC2: Timelapse noise tracking
# ============================================================

# %%
# --- UC2 settings ---

UC2_TIFF_DIR    = "path/to/timelapse_dir"  # Directory containing TIFF files
UC2_BG_PATH     = "path/to/background.tif"
UC2_CROP        = (8, 2056, 208, 2256)
UC2_ROI         = (50, 100, 50, 100)       # Noise measurement ROI (row_start, row_end, col_start, col_end)
                                            # Select a background region (no cells)
UC2_OFFAXIS     = OFFAXIS_CENTER

# %%
# --- UC2 execution ---

bg_img = np.array(Image.open(UC2_BG_PATH))[UC2_CROP[0]:UC2_CROP[1], UC2_CROP[2]:UC2_CROP[3]]
_params = QPIParameters(wavelength=WAVELENGTH, NA=NA, img_shape=bg_img.shape,
                        pixelsize=PIXELSIZE, offaxis_center=UC2_OFFAXIS)
angle_bg = unwrap_phase(np.angle(get_field(bg_img, _params)))

tif_files = sorted(f for f in os.listdir(UC2_TIFF_DIR) if f.lower().endswith(".tif"))

frames, roi_mean, roi_std = [], [], []
for fname in tif_files:
    img_t = np.array(Image.open(os.path.join(UC2_TIFF_DIR, fname)))
    img_t = img_t[UC2_CROP[0]:UC2_CROP[1], UC2_CROP[2]:UC2_CROP[3]]
    a_nobg = unwrap_phase(np.angle(get_field(img_t, _params))) - angle_bg
    roi = a_nobg[UC2_ROI[0]:UC2_ROI[1], UC2_ROI[2]:UC2_ROI[3]]
    frames.append(len(frames))
    roi_mean.append(float(np.mean(roi)))
    roi_std.append(float(np.std(roi)))

frames   = np.array(frames)
roi_mean = np.array(roi_mean)
roi_std  = np.array(roi_std)

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(frames, roi_mean)
axes[0].set_ylabel("ROI mean phase (rad)")
axes[0].set_title("Timelapse noise tracking")
axes[0].grid(True)
axes[1].plot(frames, roi_std, color="orange")
axes[1].set_ylabel("ROI phase std (rad)")
axes[1].set_xlabel("Frame index")
axes[1].grid(True)
plt.tight_layout()
save_figure(fig, params={"roi": UC2_ROI, "n_frames": len(frames)},
            description="UC2 timelapse noise tracking",
            data={"frames": frames, "roi_mean": roi_mean, "roi_std": roi_std})

print(f"Overall noise std: {roi_std.mean():.4f} rad  ({roi_std.mean() * WAVELENGTH / (4 * np.pi) * 1e9:.2f} nm)")

# ============================================================
# UC3: Correlation between shift magnitude and noise
# ============================================================

# %%
# --- UC3 settings ---

UC3_TRANSFORMS_JSON = "path/to/alignment_transforms.json"
UC3_TIFF_DIR        = "path/to/timelapse_dir"   # Can be same as UC2
UC3_BG_PATH         = "path/to/background.tif"
UC3_CROP            = (8, 2056, 208, 2256)
UC3_ROI             = (50, 100, 50, 100)
UC3_OFFAXIS         = OFFAXIS_CENTER

# %%
# --- UC3 execution ---

with open(UC3_TRANSFORMS_JSON) as f:
    _raw = json.load(f)
# Format output by align_and_subtract_timelapse.py:
#   {"alignment_results": [{"filename": "xxx.tif", "shift_x": ..., "shift_y": ...}, ...]}
if "alignment_results" in _raw:
    transforms = {r["filename"]: r for r in _raw["alignment_results"]}
else:
    transforms = _raw  # Also supports legacy {filename: {shift_x, shift_y}} format

bg_img3 = np.array(Image.open(UC3_BG_PATH))[UC3_CROP[0]:UC3_CROP[1], UC3_CROP[2]:UC3_CROP[3]]
_p3 = QPIParameters(wavelength=WAVELENGTH, NA=NA, img_shape=bg_img3.shape,
                    pixelsize=PIXELSIZE, offaxis_center=UC3_OFFAXIS)
angle_bg3 = unwrap_phase(np.angle(get_field(bg_img3, _p3)))

tif_files3 = sorted(f for f in os.listdir(UC3_TIFF_DIR) if f.lower().endswith(".tif"))

shift_mags, phase_stds = [], []
for fname in tif_files3:
    t = transforms.get(fname)
    if t is None:
        continue
    shift_x = t.get("shift_x", 0)
    shift_y = t.get("shift_y", 0)
    shift_mag = np.sqrt(shift_x**2 + shift_y**2)

    img_t = np.array(Image.open(os.path.join(UC3_TIFF_DIR, fname)))
    img_t = img_t[UC3_CROP[0]:UC3_CROP[1], UC3_CROP[2]:UC3_CROP[3]]
    a_nobg = unwrap_phase(np.angle(get_field(img_t, _p3))) - angle_bg3
    roi_std = float(np.std(a_nobg[UC3_ROI[0]:UC3_ROI[1], UC3_ROI[2]:UC3_ROI[3]]))

    shift_mags.append(shift_mag)
    phase_stds.append(roi_std)

shift_mags = np.array(shift_mags)
phase_stds = np.array(phase_stds)

fig, ax = plt.subplots()
ax.scatter(shift_mags, phase_stds, alpha=0.6)
ax.set_xlabel("Shift magnitude (pixels)")
ax.set_ylabel("ROI phase std (rad)")
ax.set_title("Shift magnitude vs phase noise")
ax.grid(True)

# Linear fit
if len(shift_mags) > 1:
    coeffs = np.polyfit(shift_mags, phase_stds, 1)
    x_fit  = np.linspace(shift_mags.min(), shift_mags.max(), 100)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), "r--",
            label=f"slope={coeffs[0]:.4f}")
    ax.legend()

save_figure(fig,
            params={"n_frames": len(shift_mags), "roi": UC3_ROI,
                    "transforms_json": UC3_TRANSFORMS_JSON},
            description="UC3 shift magnitude vs phase noise scatter",
            data={"shift_mags": shift_mags, "phase_stds": phase_stds})

# ============================================================
# UC_DIFF: Noise measurement via adjacent frame differences
# ============================================================
# Objective: Estimate temporal noise from non-overlapping pair
#            differences (0&1, 2&3, ...)
#
# Procedure:
#   1. Load adjacent pairs sequentially (memory efficient)
#   2. Compute std within 80x80 ROI for each difference
#   3. std / sqrt(2) = per-frame noise
#   4. Plot against pair index
#      -> Result has half the number of frames (N//2)
# ============================================================

# %%
# --- UC_DIFF settings ---

UC_DIFF_DIR        = r"D:\AquisitionData\Kitagishi\basler_image_seq\exp60ms_int100ms_300frame_meanint_620\Pos0"
UC_DIFF_ROI_SIZE   = 80       # 80x80 ROI (paper-consistent)
UC_DIFF_ROI_CENTER = None     # None -> image center; can specify (row, col) explicitly
UC_DIFF_PAIR_START_1BASED = 50
UC_DIFF_PAIR_END_1BASED   = 99
UC_DIFF_READ_NOISE_E = None   # Measured read noise [e-] (None if unknown)

# %%
# --- UC_DIFF execution ---

print("=== UC_DIFF: Adjacent frame difference noise measurement ===")

_exts = {".tif", ".tiff", ".png"}
_files_diff = sorted(
    f for f in os.listdir(UC_DIFF_DIR)
    if os.path.splitext(f)[1].lower() in _exts
)
N_diff   = len(_files_diff)
n_pairs_total = N_diff // 2
pair_start_idx = UC_DIFF_PAIR_START_1BASED - 1
pair_end_idx = UC_DIFF_PAIR_END_1BASED - 1

if pair_start_idx < 0 or pair_end_idx < pair_start_idx:
    raise ValueError(
        "UC_DIFF pair range is invalid: "
        f"{UC_DIFF_PAIR_START_1BASED}-{UC_DIFF_PAIR_END_1BASED}"
    )
if pair_end_idx >= n_pairs_total:
    raise ValueError(
        "UC_DIFF pair range exceeds available pairs: "
        f"requested {UC_DIFF_PAIR_START_1BASED}-{UC_DIFF_PAIR_END_1BASED}, "
        f"available 1-{n_pairs_total}"
    )

selected_pair_idx = np.arange(pair_start_idx, pair_end_idx + 1, dtype=int)
n_pairs = len(selected_pair_idx)
print(f"  Total frames: {N_diff}  -> total pairs: {n_pairs_total}")
print(
    "  Analysis target pair range: "
    f"{UC_DIFF_PAIR_START_1BASED}-{UC_DIFF_PAIR_END_1BASED} (N={n_pairs})"
)

# Read one frame to get image size
_probe = tifffile.imread(os.path.join(UC_DIFF_DIR, _files_diff[0]))
H_d, W_d = _probe.shape[:2]

# ROI setup
_half = UC_DIFF_ROI_SIZE // 2
if UC_DIFF_ROI_CENTER is None:
    cr_d, cc_d = H_d // 2, W_d // 2
else:
    cr_d, cc_d = UC_DIFF_ROI_CENTER
rs_d, re_d = cr_d - _half, cr_d + _half
cs_d, ce_d = cc_d - _half, cc_d + _half
print(f"  ROI: rows {rs_d}:{re_d}, cols {cs_d}:{ce_d}  ({UC_DIFF_ROI_SIZE}×{UC_DIFF_ROI_SIZE} px)")

# Process pairs sequentially (no need to load all frames, saves memory)
pair_idx   = []
noise_vals = []  # std / sqrt(2) [ADU]
roi_mean_adu_vals = []
shot_limit_e_vals = []
theory_limit_e_vals = []

for _i in selected_pair_idx:
    _f0 = tifffile.imread(os.path.join(UC_DIFF_DIR, _files_diff[2 * _i    ])).astype(np.float64)
    _f1 = tifffile.imread(os.path.join(UC_DIFF_DIR, _files_diff[2 * _i + 1])).astype(np.float64)
    _roi0 = _f0[rs_d:re_d, cs_d:ce_d]
    _roi1 = _f1[rs_d:re_d, cs_d:ce_d]
    _diff_roi = _roi1 - _roi0

    # Theoretical limit (shot noise): sigma_shot = sqrt(N_e)
    _roi_mean_adu = float(0.5 * (_roi0.mean() + _roi1.mean()))
    _roi_mean_e = max(_roi_mean_adu * SENSOR_CONVERSION_GAIN, 0.0)
    _shot_limit_e = float(np.sqrt(_roi_mean_e))
    if UC_DIFF_READ_NOISE_E is None:
        _theory_limit_e = _shot_limit_e
    else:
        _theory_limit_e = float(np.sqrt(_shot_limit_e**2 + UC_DIFF_READ_NOISE_E**2))

    pair_idx.append(_i + 1)
    noise_vals.append(float(np.std(_diff_roi) / np.sqrt(2)))
    roi_mean_adu_vals.append(_roi_mean_adu)
    shot_limit_e_vals.append(_shot_limit_e)
    theory_limit_e_vals.append(_theory_limit_e)

pair_idx   = np.array(pair_idx)
noise_vals = np.array(noise_vals)
roi_mean_adu_vals = np.array(roi_mean_adu_vals)
shot_limit_e_vals = np.array(shot_limit_e_vals)
theory_limit_e_vals = np.array(theory_limit_e_vals)

# Convert to e-
noise_e_diff = noise_vals * SENSOR_CONVERSION_GAIN
shot_limit_adu_vals = shot_limit_e_vals / SENSOR_CONVERSION_GAIN
theory_limit_adu_vals = theory_limit_e_vals / SENSOR_CONVERSION_GAIN

noise_mean_adu = float(noise_vals.mean())
noise_std_adu = float(noise_vals.std())
noise_mean_e = float(noise_e_diff.mean())
noise_std_e = float(noise_e_diff.std())

roi_mean_adu = float(roi_mean_adu_vals.mean())
roi_mean_e = float(roi_mean_adu * SENSOR_CONVERSION_GAIN)
shot_limit_mean_adu = float(shot_limit_adu_vals.mean())
shot_limit_mean_e = float(shot_limit_e_vals.mean())
theory_limit_mean_adu = float(theory_limit_adu_vals.mean())
theory_limit_mean_e = float(theory_limit_e_vals.mean())

ratio_measured_to_shot = (
    float(noise_mean_e / shot_limit_mean_e) if shot_limit_mean_e > 0 else np.nan
)
ratio_measured_to_theory = (
    float(noise_mean_e / theory_limit_mean_e) if theory_limit_mean_e > 0 else np.nan
)

print(f"\n--- Results ---")
print(f"  noise per frame (ADU): mean={noise_mean_adu:.2f} +/- {noise_std_adu:.2f}")
print(f"  noise per frame (e-):  mean={noise_mean_e:.1f} +/- {noise_std_e:.1f}")
print(f"\n--- Theoretical limit (based on ROI mean intensity) ---")
print(f"  ROI mean intensity: {roi_mean_adu:.1f} ADU = {roi_mean_e:.0f} e-")
print(
    f"  shot-noise limit:  {shot_limit_mean_adu:.2f} ADU "
    f"= {shot_limit_mean_e:.1f} e⁻ / frame"
)
if UC_DIFF_READ_NOISE_E is None:
    print("  read-noise term:    not set (shot-noise only)")
else:
    print(f"  read-noise term:    {UC_DIFF_READ_NOISE_E:.1f} e⁻")
    print(
        f"  shot+read limit:    {theory_limit_mean_adu:.2f} ADU "
        f"= {theory_limit_mean_e:.1f} e⁻ / frame"
    )
print(
    f"  measured / shot limit: {ratio_measured_to_shot:.2f}x "
    f"({ratio_measured_to_shot * 100:.1f}%)"
)
if UC_DIFF_READ_NOISE_E is not None:
    print(
        f"  measured / shot+read limit: {ratio_measured_to_theory:.2f}x "
        f"({ratio_measured_to_theory * 100:.1f}%)"
    )

# --- Figure: noise time series ---
fig_diff, axes_diff = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes_diff[0].plot(pair_idx, noise_vals, lw=0.8, color="steelblue")
axes_diff[0].axhline(noise_mean_adu, color="red", ls="--",
                     label=f"measured mean = {noise_mean_adu:.2f} ADU")
axes_diff[0].axhline(shot_limit_mean_adu, color="green", ls=":",
                     label=f"shot limit = {shot_limit_mean_adu:.2f} ADU")
if UC_DIFF_READ_NOISE_E is not None:
    axes_diff[0].axhline(theory_limit_mean_adu, color="purple", ls="-.",
                         label=f"shot+read limit = {theory_limit_mean_adu:.2f} ADU")
axes_diff[0].set_ylabel("Noise per frame [ADU]")
axes_diff[0].set_title(
    f"Adjacent-frame diff noise  |  80×80 ROI center=({cr_d},{cc_d})  |  "
    f"measured/shot={ratio_measured_to_shot:.2f}x"
)
axes_diff[0].legend()
axes_diff[0].grid(True, alpha=0.4)

axes_diff[1].plot(pair_idx, noise_e_diff, lw=0.8, color="darkorange")
axes_diff[1].axhline(noise_mean_e, color="red", ls="--",
                     label=f"measured mean = {noise_mean_e:.1f} e⁻")
axes_diff[1].axhline(shot_limit_mean_e, color="green", ls=":",
                     label=f"shot limit = {shot_limit_mean_e:.1f} e⁻")
if UC_DIFF_READ_NOISE_E is not None:
    axes_diff[1].axhline(theory_limit_mean_e, color="purple", ls="-.",
                         label=f"shot+read limit = {theory_limit_mean_e:.1f} e⁻")
axes_diff[1].set_ylabel("Noise per frame [e⁻]")
axes_diff[1].set_xlabel(
    f"Pair number  (N = {n_pairs}, selected {UC_DIFF_PAIR_START_1BASED}-"
    f"{UC_DIFF_PAIR_END_1BASED} / total {n_pairs_total})"
)
axes_diff[1].legend()
axes_diff[1].grid(True, alpha=0.4)
axes_diff[1].set_xlim(0, n_pairs)

plt.suptitle(
    f"UC_DIFF: adjacent-frame diff noise  |  "
    f"conversion_gain = {SENSOR_CONVERSION_GAIN:.3f} e⁻/ADU  |  "
    f"N = {N_diff} frames → total {n_pairs_total} pairs, "
    f"selected {UC_DIFF_PAIR_START_1BASED}-{UC_DIFF_PAIR_END_1BASED}",
    fontsize=11
)
plt.tight_layout()

_uc_diff_data = {
    "pair_idx":           pair_idx,
    "noise_adu":          noise_vals,
    "noise_e":            noise_e_diff,
    "shot_limit_adu":     shot_limit_adu_vals,
    "shot_limit_e":       shot_limit_e_vals,
    "roi_mean_adu":       roi_mean_adu_vals,
}

save_figure(
    fig_diff,
    params={
        "data_dir":        UC_DIFF_DIR,
        "n_frames":        N_diff,
        "n_pairs_total":   n_pairs_total,
        "n_pairs":         n_pairs,
        "pair_start_1based": UC_DIFF_PAIR_START_1BASED,
        "pair_end_1based": UC_DIFF_PAIR_END_1BASED,
        "roi_size":        UC_DIFF_ROI_SIZE,
        "roi_center":      (int(cr_d), int(cc_d)),
        "noise_mean_adu":  round(noise_mean_adu, 2),
        "noise_std_adu":   round(noise_std_adu, 2),
        "noise_mean_e":    round(noise_mean_e, 1),
        "noise_std_e":     round(noise_std_e, 1),
        "roi_mean_adu":    round(roi_mean_adu, 1),
        "shot_limit_mean_adu": round(shot_limit_mean_adu, 2),
        "shot_limit_mean_e":   round(shot_limit_mean_e, 1),
        "theory_limit_mean_adu": round(theory_limit_mean_adu, 2),
        "theory_limit_mean_e":   round(theory_limit_mean_e, 1),
        "read_noise_e":    UC_DIFF_READ_NOISE_E,
        "measured_over_shot": round(ratio_measured_to_shot, 3),
        "measured_over_theory": round(ratio_measured_to_theory, 3),
        "conversion_gain": SENSOR_CONVERSION_GAIN,
    },
    description=(
        f"UC_DIFF adjacent-diff noise: measured={noise_mean_adu:.2f} ADU "
        f"({noise_mean_e:.1f} e⁻), shot_limit={shot_limit_mean_adu:.2f} ADU "
        f"({shot_limit_mean_e:.1f} e⁻), measured/shot={ratio_measured_to_shot:.2f}x "
        f"({n_pairs} pairs, selected {UC_DIFF_PAIR_START_1BASED}-{UC_DIFF_PAIR_END_1BASED})"
    ),
    data=_uc_diff_data,
)

# --- Figure: top panel (ADU) only ---
fig_diff_top, ax_top = plt.subplots(1, 1, figsize=(10, 3.8))
ax_top.plot(pair_idx, noise_vals, lw=0.8, color="steelblue")
ax_top.axhline(noise_mean_adu, color="red", ls="--",
               label=f"measured mean = {noise_mean_adu:.2f} ADU")
ax_top.axhline(shot_limit_mean_adu, color="green", ls=":",
               label=f"shot limit = {shot_limit_mean_adu:.2f} ADU")
if UC_DIFF_READ_NOISE_E is not None:
    ax_top.axhline(theory_limit_mean_adu, color="purple", ls="-.",
                   label=f"shot+read limit = {theory_limit_mean_adu:.2f} ADU")
ax_top.set_ylabel("Noise per frame [ADU]")
ax_top.set_xlabel(
    f"Pair number  (N = {n_pairs}, selected {UC_DIFF_PAIR_START_1BASED}-"
    f"{UC_DIFF_PAIR_END_1BASED} / total {n_pairs_total})"
)
ax_top.set_title(
    f"Adjacent-frame diff noise (ADU)  |  ROI 80×80 center=({cr_d},{cc_d})  |  "
    f"measured/shot={ratio_measured_to_shot:.2f}x"
)
ax_top.legend()
ax_top.grid(True, alpha=0.4)
fig_diff_top.tight_layout()

save_figure(
    fig_diff_top,
    params={
        "parent":            "UC_DIFF_combined",
        "panel":             "top_ADU",
        "n_frames":          N_diff,
        "n_pairs_total":     n_pairs_total,
        "n_pairs":           n_pairs,
        "pair_start_1based": UC_DIFF_PAIR_START_1BASED,
        "pair_end_1based":   UC_DIFF_PAIR_END_1BASED,
        "roi_size":          UC_DIFF_ROI_SIZE,
        "roi_center":        (int(cr_d), int(cc_d)),
        "noise_mean_adu":    round(noise_mean_adu, 2),
        "shot_limit_mean_adu": round(shot_limit_mean_adu, 2),
        "theory_limit_mean_adu": round(theory_limit_mean_adu, 2),
        "read_noise_e":      UC_DIFF_READ_NOISE_E,
    },
    description=(
        f"UC_DIFF top panel only (ADU): measured={noise_mean_adu:.2f} ADU, "
        f"shot_limit={shot_limit_mean_adu:.2f} ADU, "
        f"selected pairs={UC_DIFF_PAIR_START_1BASED}-{UC_DIFF_PAIR_END_1BASED}"
    ),
    data=_uc_diff_data,
)

# --- Figure: bottom panel (e-) only ---
fig_diff_bottom, ax_bottom = plt.subplots(1, 1, figsize=(10, 3.8))
ax_bottom.plot(pair_idx, noise_e_diff, lw=0.8, color="darkorange")
ax_bottom.axhline(noise_mean_e, color="red", ls="--",
                  label=f"measured mean = {noise_mean_e:.1f} e⁻")
ax_bottom.axhline(shot_limit_mean_e, color="green", ls=":",
                  label=f"shot limit = {shot_limit_mean_e:.1f} e⁻")
if UC_DIFF_READ_NOISE_E is not None:
    ax_bottom.axhline(theory_limit_mean_e, color="purple", ls="-.",
                      label=f"shot+read limit = {theory_limit_mean_e:.1f} e⁻")
ax_bottom.set_ylabel("Noise per frame [e⁻]")
ax_bottom.set_xlabel(
    f"Pair number  (N = {n_pairs}, selected {UC_DIFF_PAIR_START_1BASED}-"
    f"{UC_DIFF_PAIR_END_1BASED} / total {n_pairs_total})"
)
ax_bottom.set_title(
    f"Adjacent-frame diff noise (e⁻)  |  ROI 80×80 center=({cr_d},{cc_d})  |  "
    f"measured/shot={ratio_measured_to_shot:.2f}x"
)
ax_bottom.legend()
ax_bottom.grid(True, alpha=0.4)
fig_diff_bottom.tight_layout()

save_figure(
    fig_diff_bottom,
    params={
        "parent":            "UC_DIFF_combined",
        "panel":             "bottom_electron",
        "n_frames":          N_diff,
        "n_pairs_total":     n_pairs_total,
        "n_pairs":           n_pairs,
        "pair_start_1based": UC_DIFF_PAIR_START_1BASED,
        "pair_end_1based":   UC_DIFF_PAIR_END_1BASED,
        "roi_size":          UC_DIFF_ROI_SIZE,
        "roi_center":        (int(cr_d), int(cc_d)),
        "noise_mean_e":      round(noise_mean_e, 1),
        "shot_limit_mean_e": round(shot_limit_mean_e, 1),
        "theory_limit_mean_e": round(theory_limit_mean_e, 1),
        "read_noise_e":      UC_DIFF_READ_NOISE_E,
    },
    description=(
        f"UC_DIFF bottom panel only (e-): measured={noise_mean_e:.1f} e⁻, "
        f"shot_limit={shot_limit_mean_e:.1f} e⁻, "
        f"selected pairs={UC_DIFF_PAIR_START_1BASED}-{UC_DIFF_PAIR_END_1BASED}"
    ),
    data=_uc_diff_data,
)

print(
    f"\nDone: {n_pairs} pairs measured "
    f"(pair {UC_DIFF_PAIR_START_1BASED}-{UC_DIFF_PAIR_END_1BASED})"
)

# %%
