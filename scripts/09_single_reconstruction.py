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
"""
path = r"/Volumes/QPI3/250910_kk/ph_1/Pos1/img_000000000_Default_000.tif"
path_bg = r"/Volumes/QPI3/250910_kk/ph_1/Pos1/img_000000000_Default_000.tif"
"""
path = r"F:\250611_kk\ph_1\Pos1\img_000000001_Default_000.tif"
path_bg = r"F:\250611_kk\ph_1\Pos0\img_000000001_Default_000.tif"

img = Image.open(path)

img_bg = Image.open(path_bg)
img = np.array(img)
#img = img[8:2056,416:2464]
#img = img[8:2056,208:2256]
#img= img[520:1544,512:1536]
#img = img[0:2048,0:2048]
#img = img[516:1540,500:1524]
#img = img[500:1000,500:1000]
img_bg = np.array(img_bg)
#img_bg = img_bg[8:2056,416:2464]
#img_bg = img_bg[8:2056,208:2256]
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
#offaxis_center = (1504, 1708) #251017_0
offaxis_center = (845,772) #250611_0.01

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
