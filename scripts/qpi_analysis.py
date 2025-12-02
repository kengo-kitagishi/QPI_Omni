# %%

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from qpi import QPIParameters
from qpi import get_field, get_spectrum, make_disk
from PIL import Image
from skimage.restoration import unwrap_phase
from CursorVisualizer import CursorVisualizer


#%%

path = "/Volumes/QPI3/250910_kk/ph_1/Pos1/img_000000000_Default_000.tif"
path_bg = "/Volumes/QPI3/250910_kk/ph_1/Pos1/img_000000000_Default_000.tif"
img = Image.open(path)

img_bg = Image.open(path_bg)
img = np.array(img)
#img = img[8:2056,416:2464]
img = img[8:2056,208:2256]
#img= img[520:1544,512:1536]
#img = img[0:2048,0:2048]
#img = img[516:1540,500:1524]
#img = img[500:1000,500:1000]
img_bg = np.array(img_bg)
#img_bg = img_bg[8:2056,416:2464]
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

WAVELENGTH = 663 * 10 ** (-9)
NA = 0.95
IMG_SHAPE = img.shape
PIXELSIZE = 3.45 * 10 ** (-6) / 40
offaxis_center = (1504, 1708)

params = QPIParameters(
    wavelength=WAVELENGTH,
    NA=NA,
    img_shape=IMG_SHAPE,
    pixelsize=PIXELSIZE,
    offaxis_center=offaxis_center,
)




# %%
# ログスケールの FFT を準備
fft_image = np.log(np.abs(img_fft))

# Figure & Axes を生成
fig, ax = plt.subplots()
ax.imshow(fft_image)

# 円の半径と中心を指定
aperture_size = params.aperturesize  # () は不要
radius = aperture_size // 2
circle_center = (offaxis_center[1], offaxis_center[0])  # (x, y)に変換
circle = plt.Circle(circle_center, radius, color='red', fill=False, linewidth=1)

# 円を追加
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

plt.imshow(visibility,vmin=0.3,vmax=0.95)
plt.colorbar()
plt.show()

plt.hist(visibility.flatten(), bins=100, range=(0.3, 1))
plt.show()
print(visibility.mean())


# %%
import numpy as np
import matplotlib.pyplot as plt

# ファイルの読み込み
#angle_nobg = np.load("/Users/kitak/QPI/angle_nobg.npy")

# 表示
plt.imshow(angle_nobg, cmap='viridis')  # または 'jet', 'gray' など
plt.colorbar(label="Phase (rad)")
plt.title("Background-subtracted Phase Map")
plt.tight_layout()
plt.show()

# %%
x_coord = 200  # 任意のx位置
profile = angle_nobg[:, x_coord]
plt.ylim(-4,1)
plt.plot(profile)
plt.title(f'Phase at x = {x_coord}')
#plt.ylim(-1,4)
plt.xlabel('Y coordinate (pixels)')
plt.ylabel('Phase (rad)')
plt.grid(True)
plt.show()

# %% 250522 特徴量検出による位置合わせ。うまくいかず

import cv2
import numpy as np
from skimage import morphology
import tifffile


# 1. Load images in grayscale
img = tifffile.imread("/Users/kitak/QPI/data/250522/angle_nobg_withcell.tif")
img = img[80:120,0:255]
bg = tifffile.imread('/Users/kitak/QPI/data/250522/angle_nobg_wocell.tif')
bg = bg[80:120,0:255]
plt.imshow(img)
plt.imshow(bg)
# float32 → uint8（0–255に正規化）
def to_uint8(image):
    image = np.nan_to_num(image)
    min_val, max_val = np.min(image), np.max(image)
    if max_val - min_val < 1e-5:
        return np.zeros_like(image, dtype=np.uint8)
    norm = (image - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8)

img_8bit = to_uint8(img)
bg_8bit  = to_uint8(bg)
# 2. Feature detection (ORB) and matching
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(bg_8bit, None)
kp2, des2 = orb.detectAndCompute(img_8bit, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:50]

# 3. Estimate homography from matches
src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 4. Warp the 'with-cell' image to align with background
h, w = bg.shape
aligned = cv2.warpPerspective(img, M, (w, h))

# 5. Subtract background
diff = cv2.absdiff(bg, aligned)
plt.imshow(diff)

_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# 6. Morphological cleaning
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

# 7. Remove tiny artifacts (scikit-image)
clean_bool = clean.astype(bool)
clean_bool = morphology.remove_small_objects(clean_bool, min_size=150)
clean_bool = morphology.remove_small_holes(clean_bool, area_threshold=150)
result_mask = (clean_bool.astype(np.uint8) * 255)

# 8. Save the final mask as TIFF
tifffile.imwrite('segmented_cell.tif', result_mask)

# %%
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm

# 定数設定
WAVELENGTH = 663e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
IMG_SHAPE = (1024, 1024)  # 後で上書きされるが仮置き
OFFAXIS_CENTER = (858, 759)
BG_PATH = "/Users/kitak/QPI/data/250522/bg.tif"
TARGET_DIR = "/Users/kitak/QPI/data/250522/test_timelapse"
OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_2,exist_ok=True)
# 背景画像読み込み・クロップ
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

# 処理ループ
for filename in tqdm(sorted(os.listdir(TARGET_DIR))):
    if filename.lower().endswith(".tif"):
        filepath = os.path.join(TARGET_DIR, filename)
        outpath = os.path.join(OUTPUT_DIR, filename.replace(".tif", "_phase.tif"))

        # 画像読み込み・クロップ
        img = Image.open(filepath)
        

        # QPI再構成
        field = get_field(img, params)
        angle = unwrap_phase(np.angle(field))

        # 背景差分と平均0調整
        angle_nobg = angle - angle_bg
        angle_nobg -= np.mean(angle_nobg[1:10, 1:10])

        # 保存
        tifffile.imwrite(outpath, angle_nobg.astype(np.float32))
        # 保存（カラーマップ付きPNG画像）
        plt.figure(figsize=(6,6))
        plt.imshow(angle_nobg, cmap='viridis',vmin=-4,vmax=2)  # 'jet' や 'gray' も可
        plt.colorbar(label='Phase (rad)')
        plt.title(f"Phase: {filename}")
        plt.axis('off')

# 保存パス変更
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

# === パス設定 ===
csv_path = "/Volumes/QPI/ph_1/Pos3/output_phase/Results.csv"
image_dir = "/Volumes/QPI/ph_1/Pos3/output_phase"
output_dir = os.path.join(image_dir, "aligned_left_center")
os.makedirs(output_dir, exist_ok=True)

# === CSV読み込みとSlice順ソート ===
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# 左辺の中点の座標を取得
x = df["BX"]
y = df["BY"] + df["Height"] / 2

# 参照フレーム（最初のスライス）の座標
x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

# === 画像読み込み（自然順ソート） ===
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
assert len(image_paths) == len(dx), f"画像数（{len(image_paths)}）とCSV行数（{len(dx)}）が一致しません。"

# === アライメント処理 ===
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print("アライメント完了", output_dir)
# %%
aligned_dir = "/Users/kitak/QPI/data/250522/test_timelapse/output_phase/test_channel/aligned_left_center"
output_diff_dir = os.path.join(aligned_dir, "diff_from_first")
os.makedirs(output_diff_dir, exist_ok=True)

# === ファイル読み込みとソート ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
assert len(image_paths) > 1, "画像が1枚しか見つかりませんでした。"

# === 1枚目を基準として読み込む ===
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

# === 差分画像を保存 ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_diff_dir, fname), diff)

print("✅ 差分画像を保存しました:", output_diff_dir)
# %%
output_colormap_dir = os.path.join(aligned_dir, "diff_colormap")
os.makedirs(output_colormap_dir, exist_ok=True)

# === ファイル読み込みとソート ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

vmin, vmax = -0.5, 3.0

# === 差分＋カラーマップ処理 ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img

    # 指定範囲でクリップして正規化（0-255）
    diff_clipped = np.clip(diff, vmin, vmax)
    diff_norm = ((diff_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # カラーマップ適用（JET）
    color_mapped = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # 保存
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap_fixed.png"
    cv2.imwrite(os.path.join(output_colormap_dir, fname), color_mapped)

print("✅ カラーマップ画像を保存しました（範囲固定: -1〜3）→", output_colormap_dir)
# %%
# === パス設定 ===
csv_path = "/Users/kitak/QPI/data/250522/test_timelapse/250526_rectangle.csv"
image_dir = "/Users/kitak/QPI/data/250522/test_timelapse/output_phase/test_channel/output_phase_crop"
output_dir = os.path.join(image_dir, "aligned_left_center")
os.makedirs(output_dir, exist_ok=True)

# === CSV読み込みとSlice順ソート ===
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# 左辺の中点の座標を取得
x = df["BX"]
y = df["BY"] + df["Height"] / 2

# 参照フレーム（最初のスライス）の座標
x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

# === 画像読み込み（自然順ソート） ===
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
assert len(image_paths) == len(dx), f"画像数（{len(image_paths)}）とCSV行数（{len(dx)}）が一致しません。"

# === アライメント処理 ===
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print("アライメント完了", output_dir)
# %%
aligned_dir = "/Users/kitak/QPI/data/250522/test_timelapse/output_phase/test_channel/output_phase_crop/aligned_left_center"
output_diff_dir = os.path.join(aligned_dir, "diff_from_first")
os.makedirs(output_diff_dir, exist_ok=True)

# === ファイル読み込みとソート ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
assert len(image_paths) > 1, "画像が1枚しか見つかりませんでした。"

# === 1枚目を基準として読み込む ===
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

# === 差分画像を保存 ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_diff_dir, fname), diff)

print("✅ 差分画像を保存しました:", output_diff_dir)

# %%
output_colormap_dir = os.path.join(aligned_dir, "diff_colormap")
os.makedirs(output_colormap_dir, exist_ok=True)
output_colormap_tif_dir = os.path.join(aligned_dir, "diff_colormap_tif")
os.makedirs(output_colormap_tif_dir, exist_ok=True)


# === ファイル読み込みとソート ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

vmin, vmax = -0.5, 3.0

# === 差分＋カラーマップ処理 ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img

    # 指定範囲でクリップして正規化（0-255）
    diff_clipped = np.clip(diff, vmin, vmax)
    diff_norm = ((diff_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # カラーマップ適用（JET）
    color_mapped = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # 保存
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap_fixed.tif"
    tifffile.imwrite(os.path.join(output_colormap_tif_dir, fname), color_mapped)

print("✅ カラーマップ画像を保存しました（範囲固定: -1〜3）→", output_colormap_dir)
# %%　250528 visibilityを向上させたい.segmentation

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
# 空間の不均一性がないし波面の中心もある程度捉えているのでってことはそもそもの光軸のずれがあるのか、それとも
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

# 定数設定
WAVELENGTH = 663e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
OFFAXIS_CENTER = (841, 779)

# ディレクトリ設定
TARGET_DIR = "/Volumes/KIOXIA/timelapse_bin1_3/Pos3"
BG_DIR = "/Volumes/KIOXIA/timelapse_bin1_3/Pos0"
OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_2, exist_ok=True)

# 処理ループ
for filename in tqdm(sorted(os.listdir(TARGET_DIR))):
    if filename.lower().endswith(".tif"):

        filepath = os.path.join(TARGET_DIR, filename)
        bg_filepath = os.path.join(BG_DIR, filename)  # 同じ名前の背景画像を探す

        if not os.path.exists(bg_filepath):
            print(f"背景画像が見つかりません: {bg_filepath} をスキップします。")
            continue

        # 背景画像読み込み
        bg_img = Image.open(bg_filepath)
        bg_img = np.array(bg_img)

        # パラメータ設定（背景画像のshapeに合わせる）
        params = QPIParameters(
            wavelength=WAVELENGTH,
            NA=NA,
            img_shape=bg_img.shape,
            pixelsize=PIXELSIZE,
            offaxis_center=OFFAXIS_CENTER
        )
        field_bg = get_field(bg_img, params)
        angle_bg = unwrap_phase(np.angle(field_bg))

        # 処理対象画像読み込み
        img = Image.open(filepath)
        img = np.array(img)

        # QPI再構成
        field = get_field(img, params)
        angle = unwrap_phase(np.angle(field))

        # 背景差分と平均0調整
        angle_nobg = angle - angle_bg
        angle_nobg -= np.mean(angle_nobg[1:10, 1:254])

        # 保存（TIF）
        outpath = os.path.join(OUTPUT_DIR, filename.replace(".tif", "_phase.tif"))
        tifffile.imwrite(outpath, angle_nobg.astype(np.float32))

        # 保存（カラーマップPNG）
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

# === パス設定 ===
csv_path = "/Volumes/QPI/250604_kk/ph_3/Pos6/output_phase/3_Results.csv"
image_dir = "/Volumes/QPI/250604_kk/ph_3/Pos6/output_phase"
output_dir = os.path.join(image_dir, "aligned_left_center")
os.makedirs(output_dir, exist_ok=True)

# === CSV読み込みとSlice順ソート ===
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# 左辺の中点の座標を取得
x = df["BX"]
y = df["BY"] + df["Height"] / 2

# 参照フレーム（最初のスライス）の座標
x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

# === 画像読み込み（自然順ソート） ===
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
assert len(image_paths) == len(dx), f"画像数（{len(image_paths)}）とCSV行数（{len(dx)}）が一致しません。"

# === アライメント処理 ===
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print("アライメント完了", output_dir)
# %%
aligned_dir = output_dir
output_diff_dir = os.path.join(aligned_dir, "diff_from_first")
os.makedirs(output_diff_dir, exist_ok=True)

# === ファイル読み込みとソート ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
assert len(image_paths) > 1, "画像が1枚しか見つかりませんでした。"

# === 1枚目を基準として読み込む ===
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

# === 差分画像を保存 ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_diff_dir, fname), diff)

print("✅ 差分画像を保存しました:", output_diff_dir)
# %%
output_colormap_dir = os.path.join(aligned_dir, "diff_colormap")
os.makedirs(output_colormap_dir, exist_ok=True)

# === ファイル読み込みとソート ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

vmin, vmax = -0.5, 3.0

# === 差分＋カラーマップ処理 ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img

    # 指定範囲でクリップして正規化（0-255）
    diff_clipped = np.clip(diff, vmin, vmax)
    diff_norm = ((diff_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # カラーマップ適用（JET）
    color_mapped = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # 保存
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap_fixed.png"
    cv2.imwrite(os.path.join(output_colormap_dir, fname), color_mapped)

print("✅ カラーマップ画像を保存しました（範囲固定: -1〜3）→", output_colormap_dir)
# %% なんだこれは
# === パス設定 ===
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

csv_path = r"G:\250910_0\Pos1\Results.csv"
image_dir = r"G:\250910_0\Pos4\output_phase"
output_dir = os.path.join(image_dir, "aligned_left_center")
os.makedirs(output_dir, exist_ok=True)

# === CSV読み込みとSlice順ソート ===
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# 左辺の中点の座標を取得
x = df["BX"]
y = df["BY"] + df["Height"] / 2

# 参照フレーム（最初のスライス）の座標
x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

# === 画像読み込み（自然順ソート） ===
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
assert len(image_paths) == len(dx), f"画像数（{len(image_paths)}）とCSV行数（{len(dx)}）が一致しません。"

# === アライメント処理 ===
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print("アライメント完了", output_dir)
# %%
aligned_dir = output_dir
output_diff_dir = os.path.join(aligned_dir, "diff_from_first")
os.makedirs(output_diff_dir, exist_ok=True)

# === ファイル読み込みとソート ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
assert len(image_paths) > 1, "画像が1枚しか見つかりませんでした。"

# === 1枚目を基準として読み込む ===
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

# === 差分画像を保存 ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_diff_dir, fname), diff)

print("✅ 差分画像を保存しました:", output_diff_dir)

# %%
output_colormap_dir = os.path.join(aligned_dir, "diff_colormap")
os.makedirs(output_colormap_dir, exist_ok=True)
output_colormap_tif_dir = os.path.join(aligned_dir, "diff_colormap_tif")
os.makedirs(output_colormap_tif_dir, exist_ok=True)


# === ファイル読み込みとソート ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

vmin, vmax = -0.5, 2.0

# === 差分＋カラーマップ処理 ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img

    # 指定範囲でクリップして正規化（0-255）
    diff_clipped = np.clip(diff, vmin, vmax)
    diff_norm = ((diff_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # カラーマップ適用（JET）
    color_mapped = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # 保存
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap_fixed.tif"
    tifffile.imwrite(os.path.join(output_colormap_tif_dir, fname), color_mapped)

print("✅ カラーマップ画像を保存しました（範囲固定: -1〜3）→", output_colormap_dir)

# %% 250618 右中点で揃える
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

# === パス設定 ===
image_dir = "/Volumes/QPI_2/250815_kk/ph_1/Pos2/output_phase"
csv_path = os.path.join(image_dir, "Results.csv")

aligned_dir = os.path.join(image_dir, "aligned_right_center")
diff_dir = os.path.join(aligned_dir, "diff_from_first")
cmap_tif_dir = os.path.join(aligned_dir, "diff_colormap_tif")

os.makedirs(aligned_dir, exist_ok=True)
os.makedirs(diff_dir, exist_ok=True)
os.makedirs(cmap_tif_dir, exist_ok=True)

# === CSV読み込みとソート ===
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# === 右辺の中点座標を計算 ===
x = df["BX"] + df["Width"]
y = df["BY"] + df["Height"] / 2

x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

# === アライメント処理 ===
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
assert len(image_paths) == len(dx), f"画像数（{len(image_paths)}）とCSV行数（{len(dx)}）が一致しません。"

for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(aligned_dir, fname), aligned.astype(np.float32))

print("✅ アライメント完了:", aligned_dir)

# === 差分画像（基準は最初の1枚） ===
aligned_image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(aligned_image_paths[0]).astype(np.float32)

for i, path in enumerate(aligned_image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(diff_dir, fname), diff.astype(np.float32))

print("✅ 差分画像を保存しました:", diff_dir)

# === カラーマップ処理（範囲 -1〜3 に固定） ===
vmin, vmax = -1.0, 3.0
colormap = cm.viridis  # ← 好きなものに変更可能（例: plasma, magma, cividis, etc.）

for i, path in enumerate(aligned_image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img

    # 正規化してカラーマップに変換（RGBA）
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba_image = colormap(norm(diff))  # shape: (H, W, 4)
    rgb_image = (rgba_image[..., :3] * 255).astype(np.uint8)

    # 保存（PNGでもOKだがTIFに統一）
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap_mpl.tif"
    tifffile.imwrite(os.path.join(cmap_tif_dir, fname), rgb_image)

print("✅ カラーマップ画像を保存しました（固定範囲 -1〜3）:", cmap_tif_dir)

# %%　250528 visibilityを向上させたい.segmentation

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
# 空間の不均一性がないし波面の中心もある程度捉えているのでってことはそもそもの光軸のずれがあるのか、それとも
# %%
#250604_Pos1~Pos30の位相画像のバッチ処理,250618_amp以外の画像を出力
#250604_Pos1~Pos30の位相画像のバッチ処理,250618_amp以外の画像を出力のコピー
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm
import matplotlib.pyplot as plt

# 定数設定
WAVELENGTH = 663e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
OFFAXIS_CENTER = (1623,1621) #250910 1623,1621 251017 1504,1710

# ディレクトリ設定
BASE_DIR = r"G:\vis_10"
BG_DIR = os.path.join(BASE_DIR, "Pos0")  # 背景画像があるディレクトリ

# Pos1〜Pos30 をループ
for pos_idx in range(44,91): #251017 46,92 250910 44,91
    pos_name = f"Pos{pos_idx}"
    TARGET_DIR = os.path.join(BASE_DIR, pos_name)
    
    if not os.path.exists(TARGET_DIR):
        print(f"{TARGET_DIR} が存在しません。スキップします。")
        continue

    OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
    OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_2, exist_ok=True)

    print(f"\n▶ 処理中: {pos_name}")

    for filename in tqdm(sorted(os.listdir(TARGET_DIR)), desc=pos_name):
        if filename.startswith("._"):
            continue  # macOSの不要な隠しファイルをスキップ

        if filename.lower().endswith(".tif") and "output" not in filename:
            filepath = os.path.join(TARGET_DIR, filename)
            bg_filepath = os.path.join(BG_DIR, filename)  # 対応する背景画像

            if not os.path.exists(bg_filepath):
                print(f"背景画像が見つかりません: {bg_filepath} をスキップします。")
                continue

            # 背景画像読み込み
            bg_img = Image.open(bg_filepath)
            bg_img = np.array(bg_img)
            #bg_img = bg_img[8:2056,416:2464] #250712_crop #250801_crop
            bg_img = bg_img[8:2056,0:2048] #250815_crop
            
            

            # パラメータ設定
            params = QPIParameters(
                wavelength=WAVELENGTH,
                NA=NA,
                img_shape=bg_img.shape,
                pixelsize=PIXELSIZE,
                offaxis_center=OFFAXIS_CENTER
            )
            field_bg = get_field(bg_img, params)
            angle_bg = unwrap_phase(np.angle(field_bg))

            # 対象画像読み込み
            img = Image.open(filepath)
            img = np.array(img)
            #img = img[8:2056,416:2464] #250712_crop #250801_crop
            img = img[8:2056,0:2048] #250712_crop #250801_crop


            # QPI再構成
            field = get_field(img, params)
            angle = unwrap_phase(np.angle(field))

            # 背景差分と平均0調整
            angle_nobg = angle - angle_bg
            angle_nobg -= np.mean(angle_nobg[1:507, 254:507])

            # TIF保存
            outpath = os.path.join(OUTPUT_DIR, filename.replace(".tif", "_phase.tif"))
            tifffile.imwrite(outpath, angle_nobg.astype(np.float32))

            # PNG保存（カラーマップ付き）
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
#250604_Pos1~Pos30の位相画像のバッチ処理,250618_amp以外の画像を出力のコピー
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm
import matplotlib.pyplot as plt

# 定数設定
WAVELENGTH = 663e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
#OFFAXIS_CENTER = (1504,1708) 251017
OFFAXIS_CENTER = (1623,1621) # 250910

# ディレクトリ設定
BASE_DIR = r"G:\vis_10"
BG_DIR = os.path.join(BASE_DIR, "Pos0")  # 背景画像があるディレクトリ

# Pos1〜Pos30 をループ
for pos_idx in range(1,44): #251019 1,46 #250910 1,44
    pos_name = f"Pos{pos_idx}"
    TARGET_DIR = os.path.join(BASE_DIR, pos_name)
    
    if not os.path.exists(TARGET_DIR):
        print(f"{TARGET_DIR} が存在しません。スキップします。")
        continue

    OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
    OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_2, exist_ok=True)

    print(f"\n▶ 処理中: {pos_name}")

    for filename in tqdm(sorted(os.listdir(TARGET_DIR)), desc=pos_name):
        if filename.startswith("._"):
            continue  # macOSの不要な隠しファイルをスキップ

        if filename.lower().endswith(".tif") and "output" not in filename:
            filepath = os.path.join(TARGET_DIR, filename)
            bg_filepath = os.path.join(BG_DIR, filename)  # 対応する背景画像

            if not os.path.exists(bg_filepath):
                print(f"背景画像が見つかりません: {bg_filepath} をスキップします。")
                continue

            # 背景画像読み込み
            bg_img = Image.open(bg_filepath)
            bg_img = np.array(bg_img)
            bg_img = bg_img[8:2056,416:2464] #250712_crop #250801_crop
            

            # パラメータ設定
            params = QPIParameters(
                wavelength=WAVELENGTH,
                NA=NA,
                img_shape=bg_img.shape,
                pixelsize=PIXELSIZE,
                offaxis_center=OFFAXIS_CENTER
            )
            field_bg = get_field(bg_img, params)
            angle_bg = unwrap_phase(np.angle(field_bg))

            # 対象画像読み込み
            img = Image.open(filepath)
            img = np.array(img)
            img = img[8:2056,416:2464] #250712_crop #250801_crop

            # QPI再構成
            field = get_field(img, params)
            angle = unwrap_phase(np.angle(field))

            # 背景差分と平均0調整
            angle_nobg = angle - angle_bg
            angle_nobg -= np.mean(angle_nobg[1:507, 1:253])

            # TIF保存
            outpath = os.path.join(OUTPUT_DIR, filename.replace(".tif", "_phase.tif"))
            tifffile.imwrite(outpath, angle_nobg.astype(np.float32))

            # PNG保存（カラーマップ付き）
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
#250604_Pos1~Pos30の位相画像のバッチ処理,250618_Pos1~26,27~45のバッチ処理,250630_focus_test
#make_disk is not difined errorが出ます。上のコードを使いましょう
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm
import matplotlib.pyplot as plt

# 定数設定
WAVELENGTH = 663e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
OFFAXIS_CENTER = (1619,1621)

# ディレクトリ設定
BASE_DIR = " F:\250910_kk\vis_10"
BG_DIR = os.path.join(BASE_DIR, "Pos0")  # 背景画像があるディレクトリ

# Pos1〜Pos30 をループ
for pos_idx in range(77, 80):
    pos_name = f"Pos{pos_idx}"
    TARGET_DIR = os.path.join(BASE_DIR, pos_name)
    
    if not os.path.exists(TARGET_DIR):
        print(f"{TARGET_DIR} が存在しません。スキップします。")
        continue

    OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
    OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
    OUTPUT_DIR_AMP = os.path.join(TARGET_DIR, "output_amplitude")  # 振幅保存用
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_2, exist_ok=True)
    os.makedirs(OUTPUT_DIR_AMP, exist_ok=True)

    print(f"\n▶ 処理中: {pos_name}")

    for filename in tqdm(sorted(os.listdir(TARGET_DIR)), desc=pos_name):
        if filename.startswith("._"):
            continue  # macOSの不要な隠しファイルをスキップ

        if filename.lower().endswith(".tif") and "output" not in filename:
            filepath = os.path.join(TARGET_DIR, filename)
            bg_filepath = os.path.join(BG_DIR, filename)  # 対応する背景画像

            if not os.path.exists(bg_filepath):
                print(f"背景画像が見つかりません: {bg_filepath} をスキップします。")
                continue

            # 背景画像読み込み
            bg_img = Image.open(bg_filepath)
            bg_img = np.array(bg_img)

            # パラメータ設定
            params = QPIParameters(
                wavelength=WAVELENGTH,
                NA=NA,
                img_shape=bg_img.shape,
                pixelsize=PIXELSIZE,
                offaxis_center=OFFAXIS_CENTER
            )
            field_bg = get_field(bg_img, params)
            angle_bg = unwrap_phase(np.angle(field_bg))

            # 対象画像読み込み
            img = Image.open(filepath)
            img = np.array(img)

            # QPI再構成
            field = get_field(img, params)
            angle = unwrap_phase(np.angle(field))

            # 背景差分と平均0調整
            angle_nobg = angle - angle_bg
            angle_nobg -= np.mean(angle_nobg[1:80, 1:254])

            # TIF保存
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

            # PNG保存（カラーマップ付き）
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

#Pos6の一次光をアライメント
csv_path = "/Volumes/QPI/ph_1/Pos3/output_phase/Results.csv"
image_dir = "/Volumes/QPI/ph_1/Pos3/output_phase"
output_dir = os.path.join(image_dir, "aligned_left_center")
os.makedirs(output_dir, exist_ok=True)

# === CSV読み込みとSlice順ソート ===
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# 左辺の中点の座標を取得
x = df["BX"]
y = df["BY"] + df["Height"] / 2

# 参照フレーム（最初のスライス）の座標
x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

# === 画像読み込み（自然順ソート） ===
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
assert len(image_paths) == len(dx), f"画像数（{len(image_paths)}）とCSV行数（{len(dx)}）が一致しません。"

# === アライメント処理 ===
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print("アライメント完了", output_dir)
# %% 250630_focus

# 基本定数
WAVELENGTH = 663e-9
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
offaxis_center = (1609, 1639)
crop_slice = np.s_[8:2056,416:2464]
x_coord = 200

# 出力ディレクトリ
output_base = "/Volumes/KIOXIA/250813_kk/vis_focus_2"
angle_dir = os.path.join(output_base, "angle_nobg")
visib_dir = os.path.join(output_base, "visibility")
profile_dir = os.path.join(output_base, "profile")

os.makedirs(angle_dir, exist_ok=True)
os.makedirs(visib_dir, exist_ok=True)
os.makedirs(profile_dir, exist_ok=True)

# 可視性計算関数
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

    # ファイル読み込み
    path = f"/Volumes/ESD-USB/250801_focus/1/mm_1/{pos_name}/img_000000000_Default_000.tif"
    path_bg = f"/Volumes/ESD-USB/250801_vis/1/bg_1/Pos0/img_000000000_Default_000.tif"
    img = np.array(Image.open(path))[crop_slice]
    img_bg = np.array(Image.open(path_bg))[crop_slice]

    # パラメータ更新
    params = QPIParameters(
        wavelength=WAVELENGTH,
        NA=NA,
        img_shape=img.shape,
        pixelsize=PIXELSIZE,
        offaxis_center=offaxis_center
    )

    # 位相画像取得
    field = get_field(img, params)
    field_bg = get_field(img_bg, params)
    angle = unwrap_phase(np.angle(field))
    angle_bg = unwrap_phase(np.angle(field_bg))
    angle_nobg = angle - angle_bg
    angle_nobg -= np.mean(angle_nobg[145:254, 1:254])

    # 可視性マップ
    vis_map = visibility(img, params)

    # プロファイル画像 (縦方向の位相プロファイル)
    profile = angle_nobg[:, x_coord]
    profile_img = np.tile(profile[:, np.newaxis], (1, 100))  # 縦ベクトルを可視化用に横に100ピクセル複製

    # 保存
    tifffile.imwrite(os.path.join(angle_dir, f"{pos_name}.tif"), angle_nobg.astype(np.float32))
    tifffile.imwrite(os.path.join(visib_dir, f"{pos_name}.tif"), vis_map.astype(np.float32))
    tifffile.imwrite(os.path.join(profile_dir, f"{pos_name}.tif"), profile_img.astype(np.float32))

print("✅ 全ポジションの処理と保存が完了しました。")

# %%250630_焦点_カラーマップも保存
# 基本定数
WAVELENGTH = 663e-9
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
offaxis_center = (1623, 1621)
crop_slice = np.s_[8:2056,416:2464]
x_coord = 200

# 出力ディレクトリ
output_base = "/Volumes/KIOXIA/250813_kk/vis_focus_2"
angle_dir = os.path.join(output_base, "angle_nobg")
visib_dir = os.path.join(output_base, "visibility")
profile_dir = os.path.join(output_base, "profile")

os.makedirs(angle_dir, exist_ok=True)
os.makedirs(visib_dir, exist_ok=True)
os.makedirs(profile_dir, exist_ok=True)

def save_with_colormap(data, save_path, cmap="viridis", vmin=None, vmax=None, colorbar_label=""):
    """カラーマップ付きPNG画像として保存"""
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

    # ファイル読み込み
    path = f"/Volumes/ESD-USB/250801_focus/1/mm_1/{pos_name}/img_000000000_Default_000.tif"
    path_bg = f"/Volumes/ESD-USB/250801_vis/1/bg_1/Pos0/img_000000000_Default_000.tif"
    img = np.array(Image.open(path))[crop_slice]
    img_bg = np.array(Image.open(path_bg))[crop_slice]

    # パラメータ更新
    params = QPIParameters(
        wavelength=WAVELENGTH,
        NA=NA,
        img_shape=img.shape,
        pixelsize=PIXELSIZE,
        offaxis_center=offaxis_center
    )

    # 位相画像取得
    field = get_field(img, params)
    field_bg = get_field(img_bg, params)
    angle = unwrap_phase(np.angle(field))
    angle_bg = unwrap_phase(np.angle(field_bg))
    angle_nobg = angle - angle_bg
    angle_nobg -= np.mean(angle_nobg[145:254, 1:254])

    # 可視性マップ
    #vis_map = visibility(img, params)

    # プロファイル画像 (縦方向の位相プロファイル)
    #profile = angle_nobg[:, x_coord]
    #profile_img = np.tile(profile[:, np.newaxis], (1, 100))  # 縦ベクトルを可視化用に横に100ピクセル複製

    # 保存（定量データ）
    tifffile.imwrite(os.path.join(angle_dir, f"{pos_name}.tif"), angle_nobg.astype(np.float32))
    #tifffile.imwrite(os.path.join(visib_dir, f"{pos_name}.tif"), vis_map.astype(np.float32))
    #tifffile.imwrite(os.path.join(profile_dir, f"{pos_name}.tif"), profile_img.astype(np.float32))

    # 保存（可視画像）
    save_with_colormap(angle_nobg, os.path.join(angle_dir, f"{pos_name}.png"),
                       cmap='viridis', vmin=-4, vmax=2, colorbar_label='Phase (rad)')

print("✅ 全ポジションの .tif + カラーマップ画像の保存が完了しました。")

# %%
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
import matplotlib.pyplot as plt

# 定数設定
WAVELENGTH = 663e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
OFFAXIS_CENTER = (1619,1621)

# ディレクトリ設定
BASE_DIR = "/Volumes/QPI_2/250815_kk/ph_1"
BG_DIR = os.path.join(BASE_DIR, "Pos0")  # 背景画像があるディレクトリ
TARGET_FILENAME = "img_000000000_Default_000.tif"  # ← この1枚だけ処理する

# Pos5〜Pos49 をループ
for pos_idx in range(1, 50):
    pos_name = f"Pos{pos_idx}"
    TARGET_DIR = os.path.join(BASE_DIR, pos_name)
    
    if not os.path.exists(TARGET_DIR):
        print(f"{TARGET_DIR} が存在しません。スキップします。")
        continue

    filepath = os.path.join(TARGET_DIR, TARGET_FILENAME)
    bg_filepath = os.path.join(BG_DIR, TARGET_FILENAME)

    if not os.path.exists(filepath):
        print(f"{filepath} が存在しません。スキップします。")
        continue
    if not os.path.exists(bg_filepath):
        print(f"背景画像が見つかりません: {bg_filepath} をスキップします。")
        continue

    OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
    OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_2, exist_ok=True)

    print(f"\n▶ 処理中: {pos_name}/{TARGET_FILENAME}")

    # 背景画像読み込み
    bg_img = np.array(Image.open(bg_filepath))
    bg_img = bg_img[8:2056,416:2464] # crop
    
    params = QPIParameters(
        wavelength=WAVELENGTH,
        NA=NA,
        img_shape=bg_img.shape,
        pixelsize=PIXELSIZE,
        offaxis_center=OFFAXIS_CENTER
    )
    field_bg = get_field(bg_img, params)
    angle_bg = unwrap_phase(np.angle(field_bg))

    # 対象画像読み込み
    img = np.array(Image.open(filepath))
    img = img[8:2056,416:2464] # crop

    field = get_field(img, params)
    angle = unwrap_phase(np.angle(field))

    # 背景差分と平均0調整
    angle_nobg = angle - angle_bg
    angle_nobg -= np.mean(angle_nobg[1:507, 1:253])

    # TIF保存
    outpath = os.path.join(OUTPUT_DIR, TARGET_FILENAME.replace(".tif", "_phase.tif"))
    tifffile.imwrite(outpath, angle_nobg.astype(np.float32))

    # PNG保存（カラーマップ付き）
    plt.figure(figsize=(6, 6))
    plt.imshow(angle_nobg, cmap='viridis', vmin=-4, vmax=2)
    plt.colorbar(label='Phase (rad)')
    plt.title(f"Phase: {TARGET_FILENAME}")
    plt.axis('off')
    plt.tight_layout()
    png_outpath = os.path.join(OUTPUT_DIR_2, TARGET_FILENAME.replace(".tif", "_colormap.png"))
    plt.savefig(png_outpath, dpi=300)
    plt.close()

# %% 250911 focus　with - wo
# === パス設定（固定） ===
import os
import numpy as np
import pandas as pd
import tifffile
import cv2

csv_path_align = "/Volumes/QPI3/251009_focus/focus_with_2/Results.csv"  # 2行のみ
dir_A = "/Volumes/QPI_2/250910_kk/vis_10/Pos43/output_phase"          # A: vis_10（固定）
dir_B = "/Volumes/QPI_2/250910_kk/with_focus_1/Pos43/output_phase"    # B: with_focus_1（Aに合わせて移動）

out_root = os.path.join(dir_B, "aligned_diff_vs_vis10_rowbased")
out_A_aligned = os.path.join(out_root, "A_vis10_fixed")
out_B_aligned = os.path.join(out_root, "B_withfocus_aligned_to_A")
out_diff = os.path.join(out_root, "diff_B_minus_A_float32")
out_cmap = os.path.join(out_root, "diff_B_minus_A_cmap_uint8")
for d in [out_root, out_A_aligned, out_B_aligned, out_diff, out_cmap]:
    os.makedirs(d, exist_ok=True)

def slice_fname(i: int) -> str:
    return f"img_000000000_Default_{i:03d}_phase.tif"

# === アライメント（行数=2、行0をA・行1をBとして扱う） ===
df = pd.read_csv(csv_path_align).reset_index(drop=True)
assert len(df) == 2, f"Results.csv は2行前提です（検出: {len(df)}行）"

# 左辺中点
x0 = float(df.loc[0, "BX"]); y0 = float(df.loc[0, "BY"] + df.loc[0, "Height"]/2)  # A(基準)
x1 = float(df.loc[1, "BX"]); y1 = float(df.loc[1, "BY"] + df.loc[1, "Height"]/2)  # B(移動)

# B を A に合わせるためのシフト量（A座標系 <- B）
dx_B2A = x1 - x0
dy_B2A = y1 - y0
# warpAffine では「-dx, -dy」を指定すると右(+x)/下(+y)へ dx,dy だけ移動
M_B = np.float32([[1, 0, dx_B2A], [0, 1, dy_B2A]])

print(f"[INFO] B→A シフト: dx={dx_B2A:.3f}, dy={dy_B2A:.3f}")

# === 可視化レンジ ===
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

    # Aは固定（無変換で保存）／Bのみ A に合わせて平行移動
    A_fix = A
    B_aln = cv2.warpAffine(B, M_B, (W, H),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 保存
    tifffile.imwrite(os.path.join(out_A_aligned, fname), A_fix, dtype=np.float32)
    tifffile.imwrite(os.path.join(out_B_aligned, fname), B_aln, dtype=np.float32)

    # 差分（B - A）
    diff = B_aln - A_fix
    tifffile.imwrite(os.path.join(out_diff, fname), diff, dtype=np.float32)

    # カラーマップ（固定範囲）
    diff_clip = np.clip(diff, vmin, vmax)
    diff_u8 = ((diff_clip - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
    tifffile.imwrite(os.path.join(out_cmap, fname.replace(".tif", "_cmap_fixed.tif")), cmap)

print("✅ 完了：A固定・BをAへ整列し、B-A差分を保存:", out_root)
if missing:
    print("⚠️ 欠損スライス:", missing)

# %%
import os
import numpy as np
import pandas as pd
import tifffile
import cv2

# === アライメント用CSV（Pos15固定, 行数=2） ===
csv_path_align = "/Volumes/QPI3/251017/with_focus_3/Results.csv"

df = pd.read_csv(csv_path_align).reset_index(drop=True)
assert len(df) == 2, f"Results.csv は2行前提です（検出: {len(df)}行）"

# 左辺中点の座標
x0 = float(df.loc[0, "BX"]); y0 = float(df.loc[0, "BY"] + df.loc[0, "Height"]/2)
x1 = float(df.loc[1, "BX"]); y1 = float(df.loc[1, "BY"] + df.loc[1, "Height"]/2)

# B を A に合わせるためのシフト
dx_B2A = -(x1 - x0)
dy_B2A = -(y1 - y0)
M_B = np.float32([[1, 0, dx_B2A], [0, 1, dy_B2A]])

print(f"[INFO] 共通シフト: dx={dx_B2A:.3f}, dy={dy_B2A:.3f}")

# === ユーティリティ ===
def slice_fname(i: int) -> str:
    return f"img_000000000_Default_{i:03d}_phase.tif"

# 可視化レンジ
vmin, vmax = -0.5, 2.0

# === Pos1〜Pos91 ループ ===
for pos_idx in range(1, 4):
    pos_name = f"Pos{pos_idx}"
    dir_A = f"/Volumes/QPI3/251017/with_focus_3/{pos_name}/output_phase"       # A: vis_10
    dir_B = f"/Volumes/QPI3/251017/wo_focus_5/{pos_name}/output_phase" # B: with_focus_1

    if not (os.path.exists(dir_A) and os.path.exists(dir_B)):
        print(f"⚠️ {pos_name}: フォルダが見つかりません。スキップします。")
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

        # Aはそのまま / Bのみシフト
        A_fix = A
        B_aln = cv2.warpAffine(B, M_B, (W, H),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # 保存
        tifffile.imwrite(os.path.join(out_A_aligned, fname), A_fix, dtype=np.float32)
        tifffile.imwrite(os.path.join(out_B_aligned, fname), B_aln, dtype=np.float32)

        # 差分
        diff = B_aln - A_fix
        tifffile.imwrite(os.path.join(out_diff, fname), diff, dtype=np.float32)

        # カラーマップ
        diff_clip = np.clip(diff, vmin, vmax)
        diff_u8 = ((diff_clip - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        cmap = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
        tifffile.imwrite(os.path.join(out_cmap, fname.replace(".tif", "_cmap_fixed.tif")), cmap)

    print(f"✅ {pos_name}: 完了 ({len(missing)} 枚欠損) → {out_root}")

# %%
