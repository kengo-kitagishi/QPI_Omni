# %%
#from paths import DATA, RESULTS
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from qpi import QPIParameters
from qpi import get_field, get_spectrum, make_disk
from PIL import Image
from skimage.restoration import unwrap_phase
from CursorVisualizer import CursorVisualizer
from figure_logger import setup_autosave
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE, CROP_REGION
setup_autosave()


#%%

path = 
path_bg = r"D:\AquisitionData\Kitagishi\basler_image_seq\Basler_acA2440-75um__25176370__20260228_193905664_1157.tiff"
img = Image.open(path)

img_bg = Image.open(path_bg)
img = np.array(img)
#img = img[8:2056,400:2448]
img = img[8:2056,208:2256]

#img= img[520:1544,512:1536]
#img = img[0:2048,0:2048]
#img = img[516:1540,500:1524]
#img = img[500:1000,500:1000]
img_bg = np.array(img_bg)
#img_bg = img_bg[8:2056,400:2448]
img_bg = img_bg[8:2056,208:2256]
#img_bg = img_bg[520:1544,512:1536]
#img_bg = img_bg[0:2048, 0:2048]
#img_bg = img_bg[0:1024, 0:1024]
#img_bg = img_bg[500:1000,500:1000]

plt.imshow(img)
plt.show()

plt.imshow(img_bg)
#plt.show()

# %%

img_fft = np.fft.fftshift(np.fft.fft2(img_bg))

plt.imshow(np.log(np.abs(img_fft)))
#plt.show()

#plt.imshow(np.log(np.abs(img_fft[820:850, 770:800])))
#plt.show()}

cb = CursorVisualizer(np.log(np.abs(img_fft)))
cb.run()

# %%

IMG_SHAPE = img.shape
offaxis_center = OFFAXIS_CENTER  # managed in optical_config.py

params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=IMG_SHAPE,
    pixelsize=PIXELSIZE,
    offaxis_center=offaxis_center,
)




# %%
# Prepare log-scale FFT
fft_image = np.log(np.abs(img_fft))

# Create Figure & Axes
fig, ax = plt.subplots()
ax.imshow(fft_image)

# Specify circle radius and center
aperture_size = params.aperturesize  # no () needed
radius = aperture_size // 2
circle_center = (offaxis_center[1], offaxis_center[0])  # convert to (x, y)
circle = plt.Circle(circle_center, radius, color='red', fill=False, linewidth=1)

# Add circle
ax.add_patch(circle)
#ax.set_title("FFT with Aperture Circle")
plt.show()

# %%
field = get_field(img, params)
field_bg = get_field(img_bg, params)

angle = unwrap_phase(np.angle(field))
angle_bg = unwrap_phase(np.angle(field_bg))

plt.imshow(angle)
plt.colorbar()
plt.show()

plt.imshow(angle_bg)
plt.colorbar()
plt.show()
# %%
angle_nobg = angle - angle_bg 
#angle_nobg = angle_nobg - np.mean(angle_nobg[244:254,1:254])
angle_nobg = angle_nobg - np.mean(angle_nobg[1:100,1:254])
plt.imshow(angle_nobg,vmin=-0.1,vmax=0.1)
plt.colorbar()
plt.show()


#angle_nobg = angle_nobg[0:256,80:256]
#angle_nobg = angle_nobg - np.mean(angle_nobg[1:10,160:170])
#angle_nobg = angle_nobg - np.mean(angle_bg)
#plt.imshow(angle_nobg,vmin=-2.5,vmax=-1)
#plt.colorbar()
#plt.show()
# %%

def visibility(array: np.array, params: QPIParameters) -> np.array:
    img = np.array(array)
    radius = params.aperturesize // 2
    # Acquire the 0th order image
    img_freq = np.fft.fftshift(np.fft.fft2(img))
    disk_0th = make_disk(params.img_center, radius, img.shape)
    img_freq_0th = img_freq * disk_0th
    img_0th = np.fft.ifft2(np.fft.ifftshift(img_freq_0th))
    img_0th_abs = np.abs(img_0th)
    # Acquire the 1th order image
    disk_1th = make_disk(params.offaxis_center, radius, img.shape)
    img_freq_interfere = img_freq * disk_1th
    # off_axis_vec = find_max_args(img_freq_interfere)
    # off_x = off_axis_vec[0]
    # off_y = off_axis_vec[1]
    # img_freq_interfere = np.roll(
    #     img_freq_interfere,
    #     (
    #         params.img_center[0] - off_x,
    #         params.img_center[1] - off_y,
    #     ),
    #     axis=(0, 1),
    # )
    img_interfere = np.fft.ifft2(np.fft.ifftshift(img_freq_interfere))
    img_interfere_abs = np.abs(img_interfere)
    img_interfere_vis = img_interfere_abs / img_0th_abs * 2
    return img_interfere_vis

visibility = visibility(img, params)

plt.imshow(visibility,vmin=0.70,vmax=0.95)
plt.colorbar()
plt.show()

plt.hist(visibility.flatten(), bins=100, range=(0.3, 1))
plt.show()
print(visibility.mean())


# %%
import numpy as np
import matplotlib.pyplot as plt

# Load file
#angle_nobg = np.load("/Users/kitak/QPI/angle_nobg.npy")

# Display
plt.imshow(angle_nobg, cmap='viridis')  # or 'jet', 'gray', etc.
plt.colorbar(label="Phase (rad)")
plt.title("Background-subtracted Phase Map")
plt.tight_layout()
plt.show()
# %%
x_coord = 200  # arbitrary x position
profile = angle_nobg[:, x_coord]
plt.ylim(-4,1)
plt.plot(profile)
plt.title(f'Phase at x = {x_coord}')
#plt.ylim(-1,4)
plt.xlabel('Y coordinate (pixels)')
plt.ylabel('Phase (rad)')
plt.grid(True)
plt.show()

# %% 250522 Feature-based alignment. Did not work well

# %%
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm

# Constants
WAVELENGTH = 663e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
IMG_SHAPE = (1024, 1024)  # placeholder, will be overwritten later
OFFAXIS_CENTER = (858, 759)
BG_PATH = "/Users/kitak/QPI/data/250522/bg.tif"
TARGET_DIR = "/Users/kitak/QPI/data/250522/test_timelapse"
OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_2,exist_ok=True)
# Load and crop background image
bg_img = Image.open(BG_PATH)
bg_img = np.array(bg_img)[516:1540,720:1744]
params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=bg_img.shape,
    pixelsize=PIXELSIZE,
    offaxis_center=OFFAXIS_CENTER
)
field_bg = get_field(bg_img, params)
angle_bg = unwrap_phase(np.angle(field_bg))

# Processing loop
for filename in tqdm(sorted(os.listdir(TARGET_DIR))):
    if filename.lower().endswith(".tif"):
        filepath = os.path.join(TARGET_DIR, filename)
        outpath = os.path.join(OUTPUT_DIR, filename.replace(".tif", "_phase.tif"))

        # Load and crop image
        img = Image.open(filepath)
        

        # QPI reconstruction
        field = get_field(img, params)
        angle = unwrap_phase(np.angle(field))

        # Background subtraction and zero-mean adjustment
        angle_nobg = angle - angle_bg
        angle_nobg -= np.mean(angle_nobg[1:10, 1:10])

        # Save
        tifffile.imwrite(outpath, angle_nobg.astype(np.float32))
        # Save (colormap PNG image)
        plt.figure(figsize=(6,6))
        plt.imshow(angle_nobg, cmap='viridis',vmin=-4,vmax=2)  # 'jet' or 'gray' also possible
        plt.colorbar(label='Phase (rad)')
        plt.title(f"Phase: {filename}")
        plt.axis('off')

# Change save path
        png_outpath = os.path.join(OUTPUT_DIR_2, filename.replace(".tif", "_colormap.png"))
        plt.tight_layout()
        plt.savefig(png_outpath, dpi=300)
        plt.close() 

        

# %%
import os
import numpy as np
import pandas as pd
import tifffile
import cv2
from glob import glob
from natsort import natsorted

# === Path settings ===
csv_path = "/Volumes/QPI/ph_1/Pos3/output_phase/Results.csv"
image_dir = "/Volumes/QPI/ph_1/Pos3/output_phase"
output_dir = os.path.join(image_dir, "aligned_left_center")
os.makedirs(output_dir, exist_ok=True)

# === Load CSV and sort by Slice ===
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# Get coordinates of the left edge midpoint
x = df["BX"]
y = df["BY"] + df["Height"] / 2

# Reference frame (first slice) coordinates
x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

# === Load images (natural sort order) ===
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
assert len(image_paths) == len(dx), f"Number of images ({len(image_paths)}) does not match CSV rows ({len(dx)})."

# === Alignment processing ===
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print("Alignment complete", output_dir)
# %%
aligned_dir = "/Users/kitak/QPI/data/250522/test_timelapse/output_phase/test_channel/aligned_left_center"
output_diff_dir = os.path.join(aligned_dir, "diff_from_first")
os.makedirs(output_diff_dir, exist_ok=True)

# === Load and sort files ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
assert len(image_paths) > 1, "Only one image found."

# === Load the first image as reference ===
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

# === Save difference images ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_diff_dir, fname), diff)

print("Done: saved difference images:", output_diff_dir)
# %%
output_colormap_dir = os.path.join(aligned_dir, "diff_colormap")
os.makedirs(output_colormap_dir, exist_ok=True)

# === Load and sort files ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

vmin, vmax = -0.5, 3.0

# === Difference + colormap processing ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img

    # Clip to specified range and normalize (0-255)
    diff_clipped = np.clip(diff, vmin, vmax)
    diff_norm = ((diff_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # Apply colormap (JET)
    color_mapped = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # Save
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap_fixed.png"
    cv2.imwrite(os.path.join(output_colormap_dir, fname), color_mapped)

print("Done: saved colormap images (fixed range: -1 to 3) ->", output_colormap_dir)
# %%
# === Path settings ===
csv_path = "/Users/kitak/QPI/data/250522/test_timelapse/250526_rectangle.csv"
image_dir = "/Users/kitak/QPI/data/250522/test_timelapse/output_phase/test_channel/output_phase_crop"
output_dir = os.path.join(image_dir, "aligned_left_center")
os.makedirs(output_dir, exist_ok=True)

# === Load CSV and sort by Slice ===
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# Get coordinates of the left edge midpoint
x = df["BX"]
y = df["BY"] + df["Height"] / 2

# Reference frame (first slice) coordinates
x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

# === Load images (natural sort order) ===
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
assert len(image_paths) == len(dx), f"Number of images ({len(image_paths)}) does not match CSV rows ({len(dx)})."

# === Alignment processing ===
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print("Alignment complete", output_dir)
# %%
aligned_dir = "/Users/kitak/QPI/data/250522/test_timelapse/output_phase/test_channel/output_phase_crop/aligned_left_center"
output_diff_dir = os.path.join(aligned_dir, "diff_from_first")
os.makedirs(output_diff_dir, exist_ok=True)

# === Load and sort files ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
assert len(image_paths) > 1, "Only one image found."

# === Load the first image as reference ===
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

# === Save difference images ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_diff_dir, fname), diff)

print("Done: saved difference images:", output_diff_dir)

# %%
output_colormap_dir = os.path.join(aligned_dir, "diff_colormap")
os.makedirs(output_colormap_dir, exist_ok=True)
output_colormap_tif_dir = os.path.join(aligned_dir, "diff_colormap_tif")
os.makedirs(output_colormap_tif_dir, exist_ok=True)


# === Load and sort files ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

vmin, vmax = -0.5, 3.0

# === Difference + colormap processing ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img

    # Clip to specified range and normalize (0-255)
    diff_clipped = np.clip(diff, vmin, vmax)
    diff_norm = ((diff_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # Apply colormap (JET)
    color_mapped = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # Save
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap_fixed.tif"
    tifffile.imwrite(os.path.join(output_colormap_tif_dir, fname), color_mapped)

print("Done: saved colormap images (fixed range: -1 to 3) ->", output_colormap_dir)
# %% 250528 Improve visibility. Segmentation

path = "/Volumes/KIOXIA/vis_test_1/Image__2025-05-28__22-57-33.bmp"
path_bg = "/Volumes/KIOXIA/vis_test_1/Image__2025-05-28__22-54-50.bmp"
# path = "/Users/kitak/QPI/holo_MM_wo_cell_250502.bmp"
img = Image.open(path)

img_bg = Image.open(path_bg)
img = np.array(img)

img = img[516:1540,500:1524]
#img = img[500:1000,500:1000]
img_bg = np.array(img_bg)
img_bg = img_bg[0:1024,0:1024]
#img_bg = img_bg[0:1024, 0:1024]
#img_bg = img_bg[500:1000,500:1000]

plt.imshow(img)
plt.show()

plt.imshow(img_bg)
plt.show()

# %%

img_fft = np.fft.fftshift(np.fft.fft2(img_bg))

plt.imshow(np.log(np.abs(img_fft)))
plt.show()

#plt.imshow(np.log(np.abs(img_fft[820:850, 770:800])))
#plt.show()

#cb = CursorVisualizer(np.log(np.abs(img_fft)))
#cb.run()

# %%

WAVELENGTH = 663 * 10 ** (-9)
NA = 0.95
IMG_SHAPE = img.shape
PIXELSIZE = 3.45 * 10 ** (-6) / 40
offaxis_center = (857, 759)

params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=IMG_SHAPE,
    pixelsize=PIXELSIZE,
    offaxis_center=offaxis_center,
)


# %%

# %%

field = get_field(img, params)
field_bg = get_field(img_bg, params)


angle = unwrap_phase(np.angle(field))
angle_bg = unwrap_phase(np.angle(field_bg))

plt.imshow(angle)
plt.colorbar()
plt.show()

plt.imshow(angle_bg)
plt.colorbar()
plt.show()
# %%
angle_nobg = angle - angle_bg 
angle_nobg = angle_nobg - np.mean(angle_nobg[1:10,1:10])
#angle_nobg = angle_nobg - np.mean(angle_bg)
plt.imshow(angle_nobg)
plt.colorbar()
plt.show()
tifffile.imwrite("/Users/kitak/QPI/data/250522/angle_nobg_withcell.tif", angle_nobg.astype(np.float32))


#angle_nobg = angle_nobg[0:256,80:256]
#angle_nobg = angle_nobg - np.mean(angle_nobg[1:10,160:170])
#angle_nobg = angle_nobg - np.mean(angle_bg)
#plt.imshow(angle_nobg,vmin=-2.5,vmax=-1)
#plt.colorbar()
#plt.show()
# %%

def visibility(array: np.array, params: QPIParameters) -> np.array:
    img = np.array(array)
    radius = params.aperturesize // 2
    # Acquire the 0th order image
    img_freq = np.fft.fftshift(np.fft.fft2(img))
    disk_0th = make_disk(params.img_center, radius, img.shape)
    img_freq_0th = img_freq * disk_0th
    img_0th = np.fft.ifft2(np.fft.ifftshift(img_freq_0th))
    img_0th_abs = np.abs(img_0th)
    # Acquire the 1th order image
    disk_1th = make_disk(params.offaxis_center, radius, img.shape)
    img_freq_interfere = img_freq * disk_1th
    # off_axis_vec = find_max_args(img_freq_interfere)
    # off_x = off_axis_vec[0]
    # off_y = off_axis_vec[1]
    # img_freq_interfere = np.roll(
    #     img_freq_interfere,
    #     (
    #         params.img_center[0] - off_x,
    #         params.img_center[1] - off_y,
    #     ),
    #     axis=(0, 1),
    # )
    img_interfere = np.fft.ifft2(np.fft.ifftshift(img_freq_interfere))
    img_interfere_abs = np.abs(img_interfere)
    img_interfere_vis = img_interfere_abs / img_0th_abs * 2
    return img_interfere_vis

visibility = visibility(img_bg, params)

plt.imshow(visibility,vmin=0.79,vmax=0.81)
plt.colorbar()
plt.show()

plt.hist(visibility.flatten(), bins=100, range=(0.8, 0.9))
plt.show()
# No spatial inhomogeneity and the wavefront center is captured to some extent, so is there an optical axis misalignment, or something else
# %%

# %% 250604_
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
WAVELENGTH = 663e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
OFFAXIS_CENTER = (841, 779)

# Directory settings
TARGET_DIR = "/Volumes/KIOXIA/timelapse_bin1_3/Pos3"
BG_DIR = "/Volumes/KIOXIA/timelapse_bin1_3/Pos0"
OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_2, exist_ok=True)

# Processing loop
for filename in tqdm(sorted(os.listdir(TARGET_DIR))):
    if filename.lower().endswith(".tif"):

        filepath = os.path.join(TARGET_DIR, filename)
        bg_filepath = os.path.join(BG_DIR, filename)  # Look for background image with same name

        if not os.path.exists(bg_filepath):
            print(f"Background image not found: {bg_filepath} - skipping.")
            continue

        # Load background image
        bg_img = Image.open(bg_filepath)
        bg_img = np.array(bg_img)

        # Set parameters (match shape to background image)
        params = QPIParameters(
            wavelength=WAVELENGTH,
            NA=NA,
            img_shape=bg_img.shape,
            pixelsize=PIXELSIZE,
            offaxis_center=OFFAXIS_CENTER
        )
        field_bg = get_field(bg_img, params)
        angle_bg = unwrap_phase(np.angle(field_bg))

        # Load target image
        img = Image.open(filepath)
        img = np.array(img)

        # QPI reconstruction
        field = get_field(img, params)
        angle = unwrap_phase(np.angle(field))

        # Background subtraction and zero-mean adjustment
        angle_nobg = angle - angle_bg
        angle_nobg -= np.mean(angle_nobg[1:10, 1:254])

        # Save (TIF)
        outpath = os.path.join(OUTPUT_DIR, filename.replace(".tif", "_phase.tif"))
        tifffile.imwrite(outpath, angle_nobg.astype(np.float32))

        # Save (colormap PNG)
        plt.figure(figsize=(6, 6))
        plt.imshow(angle_nobg, cmap='viridis', vmin=-4, vmax=2)
        plt.colorbar(label='Phase (rad)')
        plt.title(f"Phase: {filename}")
        plt.axis('off')
        plt.tight_layout()
        png_outpath = os.path.join(OUTPUT_DIR_2, filename.replace(".tif", "_colormap.png"))
        plt.savefig(png_outpath, dpi=300)
        plt.close()

# %%
import os
import numpy as np
import pandas as pd
import tifffile
import cv2
from glob import glob
from natsort import natsorted

# === Path settings ===
csv_path = "/Volumes/QPI/250604_kk/ph_3/Pos6/output_phase/3_Results.csv"
image_dir = "/Volumes/QPI/250604_kk/ph_3/Pos6/output_phase"
output_dir = os.path.join(image_dir, "aligned_left_center")
os.makedirs(output_dir, exist_ok=True)

# === Load CSV and sort by Slice ===
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# Get coordinates of the left edge midpoint
x = df["BX"]
y = df["BY"] + df["Height"] / 2

# Reference frame (first slice) coordinates
x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

# === Load images (natural sort order) ===
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
assert len(image_paths) == len(dx), f"Number of images ({len(image_paths)}) does not match CSV rows ({len(dx)})."

# === Alignment processing ===
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print("Alignment complete", output_dir)
# %%
aligned_dir = output_dir
output_diff_dir = os.path.join(aligned_dir, "diff_from_first")
os.makedirs(output_diff_dir, exist_ok=True)

# === Load and sort files ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
assert len(image_paths) > 1, "Only one image found."

# === Load the first image as reference ===
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

# === Save difference images ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_diff_dir, fname), diff)

print("Done: saved difference images:", output_diff_dir)
# %%
output_colormap_dir = os.path.join(aligned_dir, "diff_colormap")
os.makedirs(output_colormap_dir, exist_ok=True)

# === Load and sort files ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

vmin, vmax = -0.5, 3.0

# === Difference + colormap processing ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img

    # Clip to specified range and normalize (0-255)
    diff_clipped = np.clip(diff, vmin, vmax)
    diff_norm = ((diff_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # Apply colormap (JET)
    color_mapped = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # Save
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap_fixed.png"
    cv2.imwrite(os.path.join(output_colormap_dir, fname), color_mapped)

print("Done: saved colormap images (fixed range: -1 to 3) ->", output_colormap_dir)
# %% What is this
# === Path settings ===
import os
import pandas as pd
import numpy as np
import tifffile
import cv2
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image

csv_path = "/Volumes/QPI_2/250910_kk/with_focus_1/Pos15/output_phase/Results.csv"
image_dir = "/Volumes/QPI_2/250910_kk/with_focus_1/Pos15/output_phase"
output_dir = os.path.join(image_dir, "aligned_left_center")
os.makedirs(output_dir, exist_ok=True)

# === Load CSV and sort by Slice ===
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# Get coordinates of the left edge midpoint
x = df["BX"]
y = df["BY"] + df["Height"] / 2

# Reference frame (first slice) coordinates
x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

# === Load images (natural sort order) ===
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
assert len(image_paths) == len(dx), f"Number of images ({len(image_paths)}) does not match CSV rows ({len(dx)})."

# === Alignment processing ===
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print("Alignment complete", output_dir)
# %%
aligned_dir = output_dir
output_diff_dir = os.path.join(aligned_dir, "diff_from_first")
os.makedirs(output_diff_dir, exist_ok=True)

# === Load and sort files ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
assert len(image_paths) > 1, "Only one image found."

# === Load the first image as reference ===
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

# === Save difference images ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_diff_dir, fname), diff)

print("Done: saved difference images:", output_diff_dir)

# %%
output_colormap_dir = os.path.join(aligned_dir, "diff_colormap")
os.makedirs(output_colormap_dir, exist_ok=True)
output_colormap_tif_dir = os.path.join(aligned_dir, "diff_colormap_tif")
os.makedirs(output_colormap_tif_dir, exist_ok=True)


# === Load and sort files ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

vmin, vmax = -0.5, 2.0

# === Difference + colormap processing ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img

    # Clip to specified range and normalize (0-255)
    diff_clipped = np.clip(diff, vmin, vmax)
    diff_norm = ((diff_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # Apply colormap (JET)
    color_mapped = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # Save
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap_fixed.tif"
    tifffile.imwrite(os.path.join(output_colormap_tif_dir, fname), color_mapped)

print("Done: saved colormap images (fixed range: -1 to 3) ->", output_colormap_dir)

# %% 250618 Align by right midpoint
import os
import pandas as pd
import numpy as np
import tifffile
import cv2
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image

# === Path settings ===
image_dir = "/Volumes/QPI_2/250815_kk/ph_1/Pos2/output_phase"
csv_path = os.path.join(image_dir, "Results.csv")

aligned_dir = os.path.join(image_dir, "aligned_right_center")
diff_dir = os.path.join(aligned_dir, "diff_from_first")
cmap_tif_dir = os.path.join(aligned_dir, "diff_colormap_tif")

os.makedirs(aligned_dir, exist_ok=True)
os.makedirs(diff_dir, exist_ok=True)
os.makedirs(cmap_tif_dir, exist_ok=True)

# === Load CSV and sort ===
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# === Calculate right edge midpoint coordinates ===
x = df["BX"] + df["Width"]
y = df["BY"] + df["Height"] / 2

x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

# === Alignment processing ===
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
assert len(image_paths) == len(dx), f"Number of images ({len(image_paths)}) does not match CSV rows ({len(dx)})."

for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(aligned_dir, fname), aligned.astype(np.float32))

print("Done: alignment complete:", aligned_dir)

# === Difference images (reference is the first image) ===
aligned_image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(aligned_image_paths[0]).astype(np.float32)

for i, path in enumerate(aligned_image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(diff_dir, fname), diff.astype(np.float32))

print("Done: saved difference images:", diff_dir)

# === Colormap processing (fixed range: -1 to 3) ===
vmin, vmax = -1.0, 3.0
colormap = cm.viridis  # can be changed to any colormap (e.g., plasma, magma, cividis, etc.)

for i, path in enumerate(aligned_image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img

    # Normalize and convert to colormap (RGBA)
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba_image = colormap(norm(diff))  # shape: (H, W, 4)
    rgb_image = (rgba_image[..., :3] * 255).astype(np.uint8)

    # Save (TIF for consistency, though PNG is also fine)
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap_mpl.tif"
    tifffile.imwrite(os.path.join(cmap_tif_dir, fname), rgb_image)

print("Done: saved colormap images (fixed range: -1 to 3):", cmap_tif_dir)

# %% 250528 Improve visibility. Segmentation

path = "/Volumes/KIOXIA/vis_test_1/Image__2025-05-28__22-57-33.bmp"
path_bg = "/Volumes/KIOXIA/vis_test_1/Image__2025-05-28__22-54-50.bmp"
# path = "/Users/kitak/QPI/holo_MM_wo_cell_250502.bmp"
img = Image.open(path)

img_bg = Image.open(path_bg)
img = np.array(img)

img = img[516:1540,500:1524]
#img = img[500:1000,500:1000]
img_bg = np.array(img_bg)
img_bg = img_bg[0:1024,0:1024]
#img_bg = img_bg[0:1024, 0:1024]
#img_bg = img_bg[500:1000,500:1000]

plt.imshow(img)
plt.show()

plt.imshow(img_bg)
plt.show()

# %%

img_fft = np.fft.fftshift(np.fft.fft2(img_bg))

plt.imshow(np.log(np.abs(img_fft)))
plt.show()

#plt.imshow(np.log(np.abs(img_fft[820:850, 770:800])))
#plt.show()

#cb = CursorVisualizer(np.log(np.abs(img_fft)))
#cb.run()

# %%

WAVELENGTH = 663 * 10 ** (-9)
NA = 0.95
IMG_SHAPE = img.shape
PIXELSIZE = 3.45 * 10 ** (-6) / 40
offaxis_center = (857, 759)

params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=IMG_SHAPE,
    pixelsize=PIXELSIZE,
    offaxis_center=offaxis_center,
)


# %%

# %%

field = get_field(img, params)
field_bg = get_field(img_bg, params)


angle = unwrap_phase(np.angle(field))
angle_bg = unwrap_phase(np.angle(field_bg))

plt.imshow(angle)
plt.colorbar()
plt.show()

plt.imshow(angle_bg)
plt.colorbar()
plt.show()
# %%
angle_nobg = angle - angle_bg 
angle_nobg = angle_nobg - np.mean(angle_nobg[1:10,1:10])
#angle_nobg = angle_nobg - np.mean(angle_bg)
plt.imshow(angle_nobg)
plt.colorbar()
plt.show()
tifffile.imwrite("/Users/kitak/QPI/data/250522/angle_nobg_withcell.tif", angle_nobg.astype(np.float32))


#angle_nobg = angle_nobg[0:256,80:256]
#angle_nobg = angle_nobg - np.mean(angle_nobg[1:10,160:170])
#angle_nobg = angle_nobg - np.mean(angle_bg)
#plt.imshow(angle_nobg,vmin=-2.5,vmax=-1)
#plt.colorbar()
#plt.show()
# %%

def visibility(array: np.array, params: QPIParameters) -> np.array:
    img = np.array(array)
    radius = params.aperturesize // 2
    # Acquire the 0th order image
    img_freq = np.fft.fftshift(np.fft.fft2(img))
    disk_0th = make_disk(params.img_center, radius, img.shape)
    img_freq_0th = img_freq * disk_0th
    img_0th = np.fft.ifft2(np.fft.ifftshift(img_freq_0th))
    img_0th_abs = np.abs(img_0th)
    # Acquire the 1th order image
    disk_1th = make_disk(params.offaxis_center, radius, img.shape)
    img_freq_interfere = img_freq * disk_1th
    # off_axis_vec = find_max_args(img_freq_interfere)
    # off_x = off_axis_vec[0]
    # off_y = off_axis_vec[1]
    # img_freq_interfere = np.roll(
    #     img_freq_interfere,
    #     (
    #         params.img_center[0] - off_x,
    #         params.img_center[1] - off_y,
    #     ),
    #     axis=(0, 1),
    # )
    img_interfere = np.fft.ifft2(np.fft.ifftshift(img_freq_interfere))
    img_interfere_abs = np.abs(img_interfere)
    img_interfere_vis = img_interfere_abs / img_0th_abs * 2
    return img_interfere_vis

visibility = visibility(img_bg, params)

plt.imshow(visibility,vmin=0.79,vmax=0.81)
plt.colorbar()
plt.show()

plt.hist(visibility.flatten(), bins=100, range=(0.8, 0.9))
plt.show()
# No spatial inhomogeneity and the wavefront center is captured to some extent, so is there an optical axis misalignment, or something else
# %%
#250604_Batch processing of Pos1-Pos30 phase images, 250618_output images other than amp
#250604_Batch processing of Pos1-Pos30 phase images, 250618_output images other than amp (copy)
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
WAVELENGTH = 658e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
OFFAXIS_CENTER = (1664, 485) #250910 1623,1621 251017 1504,1710 251212 (1664, 485)

# Directory settings
BASE_DIR = r"F:\251212\wo_cell\ph_2"

BG_DIR = os.path.join(BASE_DIR, "Pos0")  # Directory containing background images

# Loop through Pos1 to Pos30
for pos_idx in range(24,47): #251017 46,92 250910 44,91
    pos_name = f"Pos{pos_idx}"
    TARGET_DIR = os.path.join(BASE_DIR, pos_name)
    
    if not os.path.exists(TARGET_DIR):
        print(f"{TARGET_DIR} does not exist. Skipping.")
        continue

    OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
    OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_2, exist_ok=True)

    print(f"\nProcessing: {pos_name}")

    for filename in tqdm(sorted(os.listdir(TARGET_DIR)), desc=pos_name):
        if filename.startswith("._"):
            continue  # Skip unnecessary macOS hidden files

        if filename.lower().endswith(".tif") and "output" not in filename:
            filepath = os.path.join(TARGET_DIR, filename)
            bg_filepath = os.path.join(BG_DIR, filename)  # Corresponding background image

            if not os.path.exists(bg_filepath):
                print(f"Background image not found: {bg_filepath} - skipping.")
                continue

            # Load background image
            bg_img = Image.open(bg_filepath)
            bg_img = np.array(bg_img)
            #bg_img = bg_img[8:2056,400:2448] #250712_crop #250801_crop
            bg_img = bg_img[0:2048,0:2048] #250815_crop
            
            

            # Parameter settings
            params = QPIParameters(
                wavelength=WAVELENGTH,
                NA=NA,
                img_shape=bg_img.shape,
                pixelsize=PIXELSIZE,
                offaxis_center=OFFAXIS_CENTER
            )
            field_bg = get_field(bg_img, params)
            angle_bg = unwrap_phase(np.angle(field_bg))

            # Load target image
            img = Image.open(filepath)
            img = np.array(img)
            #img = img[8:2056,400:2448] #250712_crop #250801_crop
            img = img[0:2048,0:2048] #250712_crop #250801_crop


            # QPI reconstruction
            field = get_field(img, params)
            angle = unwrap_phase(np.angle(field))

            # Background subtraction and zero-mean adjustment
            angle_nobg = angle - angle_bg
            angle_nobg -= np.mean(angle_nobg[1:507, 254:507])

            # Save TIF
            outpath = os.path.join(OUTPUT_DIR, filename.replace(".tif", "_phase.tif"))
            tifffile.imwrite(outpath, angle_nobg.astype(np.float32))

            # Save PNG (with colormap)
            plt.figure(figsize=(6, 6))
            plt.imshow(angle_nobg, cmap='viridis', vmin=-4, vmax=2)
            plt.colorbar(label='Phase (rad)')
            plt.title(f"Phase: {filename}")
            plt.axis('off')
            plt.tight_layout()
            png_outpath = os.path.join(OUTPUT_DIR_2, filename.replace(".tif", "_colormap.png"))
            plt.savefig(png_outpath, dpi=300)
            plt.close()

# %%
#250604_Batch processing of Pos1-Pos30 phase images, 250618_output images other than amp (copy)
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
WAVELENGTH = 658e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
#OFFAXIS_CENTER = (1504,1708) 251017
#OFFAXIS_CENTER = (1623,1621) # 250910
OFFAXIS_CENTER = (1664, 485) #251212
# Directory settings
BASE_DIR =r"F:\251212\wo_cell\ph_2"
BG_DIR = os.path.join(BASE_DIR, "Pos0")  # Directory containing background images

# Loop through Pos1 to Pos30
for pos_idx in range(1,24): #251212 1,24 #251019 1,46 #250910 1,44
    pos_name = f"Pos{pos_idx}"
    TARGET_DIR = os.path.join(BASE_DIR, pos_name)
    
    if not os.path.exists(TARGET_DIR):
        print(f"{TARGET_DIR} does not exist. Skipping.")
        continue

    OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
    OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_2, exist_ok=True)

    print(f"\nProcessing: {pos_name}")

    for filename in tqdm(sorted(os.listdir(TARGET_DIR)), desc=pos_name):
        if filename.startswith("._"):
            continue  # Skip unnecessary macOS hidden files

        if filename.lower().endswith(".tif") and "output" not in filename:
            filepath = os.path.join(TARGET_DIR, filename)
            bg_filepath = os.path.join(BG_DIR, filename)  # Corresponding background image

            if not os.path.exists(bg_filepath):
                print(f"Background image not found: {bg_filepath} - skipping.")
                continue

            # Load background image
            bg_img = Image.open(bg_filepath)
            bg_img = np.array(bg_img)
            bg_img = bg_img[0:2048,400:2448] #250712_crop #250801_crop
            

            # Parameter settings
            params = QPIParameters(
                wavelength=WAVELENGTH,
                NA=NA,
                img_shape=bg_img.shape,
                pixelsize=PIXELSIZE,
                offaxis_center=OFFAXIS_CENTER
            )
            field_bg = get_field(bg_img, params)
            angle_bg = unwrap_phase(np.angle(field_bg))

            # Load target image
            img = Image.open(filepath)
            img = np.array(img)
            img = img[0:2048,400:2448] #250712_crop #250801_crop

            # QPI reconstruction
            field = get_field(img, params)
            angle = unwrap_phase(np.angle(field))

            # Background subtraction and zero-mean adjustment
            angle_nobg = angle - angle_bg
            angle_nobg -= np.mean(angle_nobg[1:507, 1:253])

            # Save TIF
            outpath = os.path.join(OUTPUT_DIR, filename.replace(".tif", "_phase.tif"))
            tifffile.imwrite(outpath, angle_nobg.astype(np.float32))

            # Save PNG (with colormap)
            plt.figure(figsize=(6, 6))
            plt.imshow(angle_nobg, cmap='viridis', vmin=-4, vmax=2)
            plt.colorbar(label='Phase (rad)')
            plt.title(f"Phase: {filename}")
            plt.axis('off')
            plt.tight_layout()
            png_outpath = os.path.join(OUTPUT_DIR_2, filename.replace(".tif", "_colormap.png"))
            plt.savefig(png_outpath, dpi=300)
            plt.close()

# %%
#250604_Batch processing of Pos1-Pos30 phase images, 250618_Pos1-26,27-45 batch processing, 250630_focus_test
#make_disk is not defined error occurs. Use the code above instead
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
WAVELENGTH = 663e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
OFFAXIS_CENTER = (1619,1621)

# Directory settings
BASE_DIR = "/Volumes/QPI/ph_1"
BG_DIR = os.path.join(BASE_DIR, "Pos0")  # Directory containing background images

# Loop through Pos1 to Pos30
for pos_idx in range(1, 24):
    pos_name = f"Pos{pos_idx}"
    TARGET_DIR = os.path.join(BASE_DIR, pos_name)
    
    if not os.path.exists(TARGET_DIR):
        print(f"{TARGET_DIR} does not exist. Skipping.")
        continue

    OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
    OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
    OUTPUT_DIR_AMP = os.path.join(TARGET_DIR, "output_amplitude")  # For amplitude output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_2, exist_ok=True)
    os.makedirs(OUTPUT_DIR_AMP, exist_ok=True)

    print(f"\nProcessing: {pos_name}")

    for filename in tqdm(sorted(os.listdir(TARGET_DIR)), desc=pos_name):
        if filename.startswith("._"):
            continue  # Skip unnecessary macOS hidden files

        if filename.lower().endswith(".tif") and "output" not in filename:
            filepath = os.path.join(TARGET_DIR, filename)
            bg_filepath = os.path.join(BG_DIR, filename)  # Corresponding background image

            if not os.path.exists(bg_filepath):
                print(f"Background image not found: {bg_filepath} - skipping.")
                continue

            # Load background image
            bg_img = Image.open(bg_filepath)
            bg_img = np.array(bg_img)

            # Parameter settings
            params = QPIParameters(
                wavelength=WAVELENGTH,
                NA=NA,
                img_shape=bg_img.shape,
                pixelsize=PIXELSIZE,
                offaxis_center=OFFAXIS_CENTER
            )
            field_bg = get_field(bg_img, params)
            angle_bg = unwrap_phase(np.angle(field_bg))

            # Load target image
            img = Image.open(filepath)
            img = np.array(img)

            # QPI reconstruction
            field = get_field(img, params)
            angle = unwrap_phase(np.angle(field))

            # Background subtraction and zero-mean adjustment
            angle_nobg = angle - angle_bg
            angle_nobg -= np.mean(angle_nobg[1:80, 1:254])

            # Save TIF
            outpath = os.path.join(OUTPUT_DIR, filename.replace(".tif", "_phase.tif"))
            tifffile.imwrite(outpath, angle_nobg.astype(np.float32))
            img_fft = np.fft.fftshift(np.fft.fft2(img))
            radius = params.aperturesize // 2
            disk_1th = make_disk(params.offaxis_center, radius, img.shape)
            img_fft_1st = img_fft * disk_1th
            img_1st = np.fft.ifft2(np.fft.ifftshift(img_fft_1st))
            img_1st_abs = np.abs(img_1st)

            amp_outpath = os.path.join(OUTPUT_DIR_AMP, filename.replace(".tif", "_amplitude.tif"))
            tifffile.imwrite(amp_outpath, img_1st_abs.astype(np.float32))

            # Save PNG (with colormap)
            plt.figure(figsize=(6, 6))
            plt.imshow(angle_nobg, cmap='viridis', vmin=-4, vmax=2)
            plt.colorbar(label='Phase (rad)')
            plt.title(f"Phase: {filename}")
            plt.axis('off')
            plt.tight_layout()
            png_outpath = os.path.join(OUTPUT_DIR_2, filename.replace(".tif", "_colormap.png"))
            plt.savefig(png_outpath, dpi=300)
            plt.close()

# %%

# Align first-order light for Pos6
csv_path = "/Volumes/QPI/ph_1/Pos3/output_phase/Results.csv"
image_dir = "/Volumes/QPI/ph_1/Pos3/output_phase"
output_dir = os.path.join(image_dir, "aligned_left_center")
os.makedirs(output_dir, exist_ok=True)

# === Load CSV and sort by Slice ===
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# Get coordinates of the left edge midpoint
x = df["BX"]
y = df["BY"] + df["Height"] / 2

# Reference frame (first slice) coordinates
x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

# === Load images (natural sort order) ===
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
assert len(image_paths) == len(dx), f"Number of images ({len(image_paths)}) does not match CSV rows ({len(dx)})."

# === Alignment processing ===
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print("Alignment complete", output_dir)
# %% 250630_focus

# Basic constants
WAVELENGTH = 663e-9
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
offaxis_center = (1609, 1639)
crop_slice = np.s_[8:2056,400:2448]
x_coord = 200

# Output directories
output_base = "/Volumes/KIOXIA/250813_kk/vis_focus_2"
angle_dir = os.path.join(output_base, "angle_nobg")
visib_dir = os.path.join(output_base, "visibility")
profile_dir = os.path.join(output_base, "profile")

os.makedirs(angle_dir, exist_ok=True)
os.makedirs(visib_dir, exist_ok=True)
os.makedirs(profile_dir, exist_ok=True)

# Visibility calculation function
def visibility(img: np.ndarray, params: QPIParameters) -> np.ndarray:
    radius = params.aperturesize // 2
    img_freq = np.fft.fftshift(np.fft.fft2(img))
    disk_0th = make_disk(params.img_center, radius, img.shape)
    img_0th = np.fft.ifft2(np.fft.ifftshift(img_freq * disk_0th))
    disk_1th = make_disk(params.offaxis_center, radius, img.shape)
    img_interfere = np.fft.ifft2(np.fft.ifftshift(img_freq * disk_1th))
    return np.abs(img_interfere) / np.abs(img_0th) * 2

for pos in range(0, 106):
    pos_name = f"Pos{pos}"
    print(f"Processing {pos_name}...")

    # Load files
    path = f"/Volumes/ESD-USB/250801_focus/1/mm_1/{pos_name}/img_000000000_Default_000.tif"
    path_bg = f"/Volumes/ESD-USB/250801_vis/1/bg_1/Pos0/img_000000000_Default_000.tif"
    img = np.array(Image.open(path))[crop_slice]
    img_bg = np.array(Image.open(path_bg))[crop_slice]

    # Update parameters
    params = QPIParameters(
        wavelength=WAVELENGTH,
        NA=NA,
        img_shape=img.shape,
        pixelsize=PIXELSIZE,
        offaxis_center=offaxis_center
    )

    # Acquire phase image
    field = get_field(img, params)
    field_bg = get_field(img_bg, params)
    angle = unwrap_phase(np.angle(field))
    angle_bg = unwrap_phase(np.angle(field_bg))
    angle_nobg = angle - angle_bg
    angle_nobg -= np.mean(angle_nobg[145:254, 1:254])

    # Visibility map
    vis_map = visibility(img, params)

    # Profile image (vertical phase profile)
    profile = angle_nobg[:, x_coord]
    profile_img = np.tile(profile[:, np.newaxis], (1, 100))  # Replicate vertical vector to 100 pixels wide for visualization

    # Save
    tifffile.imwrite(os.path.join(angle_dir, f"{pos_name}.tif"), angle_nobg.astype(np.float32))
    tifffile.imwrite(os.path.join(visib_dir, f"{pos_name}.tif"), vis_map.astype(np.float32))
    tifffile.imwrite(os.path.join(profile_dir, f"{pos_name}.tif"), profile_img.astype(np.float32))

print("Done: processing and saving complete for all positions.")

# %%250630_Focus_Save colormap too
# Basic constants
WAVELENGTH = 663e-9
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
offaxis_center = (1623, 1621)
crop_slice = np.s_[8:2056,400:2448]
x_coord = 200

# Output directories
output_base = "/Volumes/KIOXIA/250813_kk/vis_focus_2"
angle_dir = os.path.join(output_base, "angle_nobg")
visib_dir = os.path.join(output_base, "visibility")
profile_dir = os.path.join(output_base, "profile")

os.makedirs(angle_dir, exist_ok=True)
os.makedirs(visib_dir, exist_ok=True)
os.makedirs(profile_dir, exist_ok=True)

def save_with_colormap(data, save_path, cmap="viridis", vmin=None, vmax=None, colorbar_label=""):
    """Save as PNG image with colormap"""
    plt.figure()
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=colorbar_label)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

for pos in range(1,2):
    pos_name = f"Pos{pos}"
    print(f"Processing {pos_name}...")

    # Load files
    path = f"/Volumes/ESD-USB/250801_focus/1/mm_1/{pos_name}/img_000000000_Default_000.tif"
    path_bg = f"/Volumes/ESD-USB/250801_vis/1/bg_1/Pos0/img_000000000_Default_000.tif"
    img = np.array(Image.open(path))[crop_slice]
    img_bg = np.array(Image.open(path_bg))[crop_slice]

    # Update parameters
    params = QPIParameters(
        wavelength=WAVELENGTH,
        NA=NA,
        img_shape=img.shape,
        pixelsize=PIXELSIZE,
        offaxis_center=offaxis_center
    )

    # Acquire phase image
    field = get_field(img, params)
    field_bg = get_field(img_bg, params)
    angle = unwrap_phase(np.angle(field))
    angle_bg = unwrap_phase(np.angle(field_bg))
    angle_nobg = angle - angle_bg
    angle_nobg -= np.mean(angle_nobg[145:254, 1:254])

    # Visibility map
    #vis_map = visibility(img, params)

    # Profile image (vertical phase profile)
    #profile = angle_nobg[:, x_coord]
    #profile_img = np.tile(profile[:, np.newaxis], (1, 100))  # Replicate vertical vector to 100 pixels wide for visualization

    # Save (quantitative data)
    tifffile.imwrite(os.path.join(angle_dir, f"{pos_name}.tif"), angle_nobg.astype(np.float32))
    #tifffile.imwrite(os.path.join(visib_dir, f"{pos_name}.tif"), vis_map.astype(np.float32))
    #tifffile.imwrite(os.path.join(profile_dir, f"{pos_name}.tif"), profile_img.astype(np.float32))

    # Save (visualization images)
    save_with_colormap(angle_nobg, os.path.join(angle_dir, f"{pos_name}.png"),
                       cmap='viridis', vmin=-4, vmax=2, colorbar_label='Phase (rad)')

print("Done: .tif + colormap image saving complete for all positions.")

# %%
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
import matplotlib.pyplot as plt

# Constants
WAVELENGTH = 663e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
OFFAXIS_CENTER = (1619,1621)

# Directory settings
BASE_DIR = "/Volumes/QPI_2/250815_kk/ph_1"
BG_DIR = os.path.join(BASE_DIR, "Pos0")  # Directory containing background images
TARGET_FILENAME = "img_000000000_Default_000.tif"  # Only process this single file

# Loop through Pos5 to Pos49
for pos_idx in range(1, 50):
    pos_name = f"Pos{pos_idx}"
    TARGET_DIR = os.path.join(BASE_DIR, pos_name)
    
    if not os.path.exists(TARGET_DIR):
        print(f"{TARGET_DIR} does not exist. Skipping.")
        continue

    filepath = os.path.join(TARGET_DIR, TARGET_FILENAME)
    bg_filepath = os.path.join(BG_DIR, TARGET_FILENAME)

    if not os.path.exists(filepath):
        print(f"{filepath} does not exist. Skipping.")
        continue
    if not os.path.exists(bg_filepath):
        print(f"Background image not found: {bg_filepath} - skipping.")
        continue

    OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
    OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_2, exist_ok=True)

    print(f"\nProcessing: {pos_name}/{TARGET_FILENAME}")

    # Load background image
    bg_img = np.array(Image.open(bg_filepath))
    bg_img = bg_img[8:2056,400:2448] # crop
    
    params = QPIParameters(
        wavelength=WAVELENGTH,
        NA=NA,
        img_shape=bg_img.shape,
        pixelsize=PIXELSIZE,
        offaxis_center=OFFAXIS_CENTER
    )
    field_bg = get_field(bg_img, params)
    angle_bg = unwrap_phase(np.angle(field_bg))

    # Load target image
    img = np.array(Image.open(filepath))
    img = img[8:2056,400:2448] # crop

    field = get_field(img, params)
    angle = unwrap_phase(np.angle(field))

    # Background subtraction and zero-mean adjustment
    angle_nobg = angle - angle_bg
    angle_nobg -= np.mean(angle_nobg[1:507, 1:253])

    # Save TIF
    outpath = os.path.join(OUTPUT_DIR, TARGET_FILENAME.replace(".tif", "_phase.tif"))
    tifffile.imwrite(outpath, angle_nobg.astype(np.float32))

    # Save PNG (with colormap)
    plt.figure(figsize=(6, 6))
    plt.imshow(angle_nobg, cmap='viridis', vmin=-4, vmax=2)
    plt.colorbar(label='Phase (rad)')
    plt.title(f"Phase: {TARGET_FILENAME}")
    plt.axis('off')
    plt.tight_layout()
    png_outpath = os.path.join(OUTPUT_DIR_2, TARGET_FILENAME.replace(".tif", "_colormap.png"))
    plt.savefig(png_outpath, dpi=300)
    plt.close()

# %% 250911 focus with - wo
# === Path settings (fixed) ===
import os
import numpy as np
import pandas as pd
import tifffile
import cv2

csv_path_align = "/Volumes/QPI3/251009_focus/focus_with_2/Results.csv"  # 2 rows only
dir_A = "/Volumes/QPI_2/250910_kk/vis_10/Pos43/output_phase"          # A: vis_10 (fixed)
dir_B = "/Volumes/QPI_2/250910_kk/with_focus_1/Pos43/output_phase"    # B: with_focus_1 (shifted to match A)

out_root = os.path.join(dir_B, "aligned_diff_vs_vis10_rowbased")
out_A_aligned = os.path.join(out_root, "A_vis10_fixed")
out_B_aligned = os.path.join(out_root, "B_withfocus_aligned_to_A")
out_diff = os.path.join(out_root, "diff_B_minus_A_float32")
out_cmap = os.path.join(out_root, "diff_B_minus_A_cmap_uint8")
for d in [out_root, out_A_aligned, out_B_aligned, out_diff, out_cmap]:
    os.makedirs(d, exist_ok=True)

def slice_fname(i: int) -> str:
    return f"img_000000000_Default_{i:03d}_phase.tif"

# === Alignment (2 rows, row 0 as A, row 1 as B) ===
df = pd.read_csv(csv_path_align).reset_index(drop=True)
assert len(df) == 2, f"Results.csv must have exactly 2 rows (found: {len(df)} rows)"

# Left edge midpoint
x0 = float(df.loc[0, "BX"]); y0 = float(df.loc[0, "BY"] + df.loc[0, "Height"]/2)  # A (reference)
x1 = float(df.loc[1, "BX"]); y1 = float(df.loc[1, "BY"] + df.loc[1, "Height"]/2)  # B (moving)

# Shift amount to align B to A (A coordinate system <- B)
dx_B2A = x1 - x0
dy_B2A = y1 - y0
# In warpAffine, specifying (-dx, -dy) shifts right(+x)/down(+y) by dx,dy
M_B = np.float32([[1, 0, dx_B2A], [0, 1, dy_B2A]])

print(f"[INFO] B->A shift: dx={dx_B2A:.3f}, dy={dy_B2A:.3f}")

# === Visualization range ===
vmin, vmax = -0.5, 2.0

missing = []
for i in range(0, 41):
    fname = slice_fname(i)
    path_A = os.path.join(dir_A, fname)
    path_B = os.path.join(dir_B, fname)

    if not (os.path.exists(path_A) and os.path.exists(path_B)):
        missing.append((i, os.path.exists(path_A), os.path.exists(path_B)))
        continue

    A = tifffile.imread(path_A).astype(np.float32)
    B = tifffile.imread(path_B).astype(np.float32)
    H, W = A.shape[:2]

    # A is fixed (no transform) / Only B is translated to match A
    A_fix = A
    B_aln = cv2.warpAffine(B, M_B, (W, H),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Save
    tifffile.imwrite(os.path.join(out_A_aligned, fname), A_fix, dtype=np.float32)
    tifffile.imwrite(os.path.join(out_B_aligned, fname), B_aln, dtype=np.float32)

    # Difference (B - A)
    diff = B_aln - A_fix
    tifffile.imwrite(os.path.join(out_diff, fname), diff, dtype=np.float32)

    # Colormap (fixed range)
    diff_clip = np.clip(diff, vmin, vmax)
    diff_u8 = ((diff_clip - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
    tifffile.imwrite(os.path.join(out_cmap, fname.replace(".tif", "_cmap_fixed.tif")), cmap)

print("Done: A fixed, B aligned to A, B-A difference saved:", out_root)
if missing:
    print("Warning: missing slices:", missing)

# %%
import os
import numpy as np
import pandas as pd
import tifffile
import cv2

# === Alignment CSV (Pos15 fixed, 2 rows) ===
csv_path_align = "/Volumes/QPI3/251017/with_focus_3/Results.csv"

df = pd.read_csv(csv_path_align).reset_index(drop=True)
assert len(df) == 2, f"Results.csv must have exactly 2 rows (found: {len(df)} rows)"

# Left edge midpoint coordinates
x0 = float(df.loc[0, "BX"]); y0 = float(df.loc[0, "BY"] + df.loc[0, "Height"]/2)
x1 = float(df.loc[1, "BX"]); y1 = float(df.loc[1, "BY"] + df.loc[1, "Height"]/2)

# Shift to align B to A
dx_B2A = -(x1 - x0)
dy_B2A = -(y1 - y0)
M_B = np.float32([[1, 0, dx_B2A], [0, 1, dy_B2A]])

print(f"[INFO] Common shift: dx={dx_B2A:.3f}, dy={dy_B2A:.3f}")

# === Utilities ===
def slice_fname(i: int) -> str:
    return f"img_000000000_Default_{i:03d}_phase.tif"

# Visualization range
vmin, vmax = -0.5, 2.0

# === Pos1 to Pos91 loop ===
for pos_idx in range(1, 4):
    pos_name = f"Pos{pos_idx}"
    dir_A = f"/Volumes/QPI3/251017/with_focus_3/{pos_name}/output_phase"       # A: vis_10
    dir_B = f"/Volumes/QPI3/251017/wo_focus_5/{pos_name}/output_phase" # B: with_focus_1

    if not (os.path.exists(dir_A) and os.path.exists(dir_B)):
        print(f"Warning: {pos_name}: folder not found. Skipping.")
        continue

    out_root = os.path.join(dir_B, "aligned_diff_vs_vis10_rowbased")
    out_A_aligned = os.path.join(out_root, "A_vis10_fixed")
    out_B_aligned = os.path.join(out_root, "B_withfocus_aligned_to_A")
    out_diff = os.path.join(out_root, "diff_B_minus_A_float32")
    out_cmap = os.path.join(out_root, "diff_B_minus_A_cmap_uint8")
    for d in [out_root, out_A_aligned, out_B_aligned, out_diff, out_cmap]:
        os.makedirs(d, exist_ok=True)

    missing = []
    for i in range(0, 41):
        fname = slice_fname(i)
        path_A = os.path.join(dir_A, fname)
        path_B = os.path.join(dir_B, fname)

        if not (os.path.exists(path_A) and os.path.exists(path_B)):
            missing.append((i, os.path.exists(path_A), os.path.exists(path_B)))
            continue

        A = tifffile.imread(path_A).astype(np.float32)
        B = tifffile.imread(path_B).astype(np.float32)
        H, W = A.shape[:2]

        # Keep A as-is / Shift only B
        A_fix = A
        B_aln = cv2.warpAffine(B, M_B, (W, H),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Save
        tifffile.imwrite(os.path.join(out_A_aligned, fname), A_fix, dtype=np.float32)
        tifffile.imwrite(os.path.join(out_B_aligned, fname), B_aln, dtype=np.float32)

        # Difference
        diff = B_aln - A_fix
        tifffile.imwrite(os.path.join(out_diff, fname), diff, dtype=np.float32)

        # Colormap
        diff_clip = np.clip(diff, vmin, vmax)
        diff_u8 = ((diff_clip - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        cmap = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
        tifffile.imwrite(os.path.join(out_cmap, fname.replace(".tif", "_cmap_fixed.tif")), cmap)

    print(f"Done: {pos_name}: complete ({len(missing)} missing) -> {out_root}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import os

#%% Folder settings
FOLDER = r"/Volumes/QPI_2/250815_kk/ph_1/Pos3/3_one_channel/"
SAVE_FOLDER = os.path.join(FOLDER, "phase_png")
os.makedirs(SAVE_FOLDER, exist_ok=True)

tif_files = sorted(glob(os.path.join(FOLDER, "*.tif")))

#%% Load phase images and save as PNG
for path in tif_files:
    print("Processing:", path)
    
    # Load image
    phase_img = np.array(Image.open(path))

    # Filename for PNG output
    filename = os.path.basename(path).replace(".tif", ".png")
    save_path = os.path.join(SAVE_FOLDER, filename)
    
    # Plot and save as PNG
    plt.figure(figsize=(6,6))
    plt.imshow(phase_img, cmap='viridis',vmin=0,vmax=1.9)
    plt.colorbar(label='Phase [rad]')
    plt.title(os.path.basename(path))
    plt.axis('off')  # Hide axes
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()  # Close to save memory

# %%
