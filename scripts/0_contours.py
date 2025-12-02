# %% # Import dependencies
import os
import numpy as np
import omnipose
import cellpose_omni
from omnipose.plot import imshow
from cellpose_omni import io, transforms, models, core
from omnipose.utils import normalize99
import time
use_GPU = core.use_gpu()
# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
plt.style.use('dark_background')
# %matplotlib inline
# %%
img_path = r"G:\Pos2\output_phase\aligned_left_center\crop_dif_0_1.5\dif_0_10300.tif"
os.makedirs(out_dir, exist_ok=True)

# %%
img_raw = io.imread(img_path)
img = transforms.move_min_dim(img_raw)
img = normalize99(img)
# %%
model_name = 'bact_fluor_omni'
model = models.CellposeModel(
    gpu=use_GPU,
    model_type=model_name
)
params = {'channels':None, # always define this if using older models, e.g. [0,0] with bact_phase_omni
          'rescale': None, # upscale or downscale your images, None = no rescaling 
          'mask_threshold': -2, # erode or dilate masks with higher or lower values between -5 and 5 
          'flow_threshold': 0, # default is .4, but only needed if there are spurious masks to clean up; slows down output
          'transparency': True, # transparency in flow output
          'omni': True, # we can turn off Omnipose mask reconstruction, not advised 
          'cluster': True, # use DBSCAN clustering
          'resample': True, # whether or not to run dynamics on rescaled grid or original grid 
          'verbose': False, # turn on if you want to see more output 
          'tile': False, # average the outputs from flipped (augmented) images; slower, usually not needed 
          'niter': None, # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation 
          'augment': False, # Can optionally rotate the image and average network outputs, usually not needed 
          'affinity_seg': False #feature, stay tuned...
         }

imgs = [img]   # 1枚だけリスト化
nimg = len(imgs)
n = range(nimg)

tic = time.time()
masks, flows, styles = model.eval([imgs[i] for i in n],**params)
net_time = time.time() - tic

print('total segmentation time: {}s'.format(net_time))
# %%
from cellpose_omni import plot
import omnipose

for idx,i in enumerate(n):

    maski = masks[idx] # get masks
    bdi = flows[idx][-1] # get boundaries
    flowi = flows[idx][0] # get RGB flows 

    # set up the output figure to better match the resolution of the images 
    f = 15
    szX = maski.shape[-1]/mpl.rcParams['figure.dpi']*f
    szY = maski.shape[-2]/mpl.rcParams['figure.dpi']*f
    fig = plt.figure(figsize=(szY,szX*4), facecolor=[0]*4, frameon=False)
    
    plot.show_segmentation(fig, omnipose.utils.normalize99(imgs[i]), 
                           maski, flowi, bdi, channels=None, omni=True,
                           interpolation=None)

    plt.tight_layout()
    plt.show()

#  %%

io.save_masks(
    img, masks, flows,
    file_names=[img_path],
    savedir=out_dir,
    save_flows=True,
    save_outlines=True
)
print(f"Saved results to: {out_dir}")
 

# %% from chapgpt 要検証
import os, glob, time, shutil
import numpy as np
import tifffile
import omnipose
import cellpose_omni
from cellpose_omni import io, transforms,models,core
from omnipose.utils import normalize99
from cellpose_omni import utils as cp_utils

# %%

base_dir = r"G:\Pos2\output_phase\aligned_left_center\crop_dif_0_1.5"
outline_dir = os.path.join(base_dir, "outline")
model_name = r"C:\Users\QPI\Desktop\train\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_1_dim_2_train_2025_10_11_17_25_03.140315"

params = dict(
    channels=[0,0],
    rescale=None,
    mask_threshold=0,
    flow_threshold=0,
    transparency=True,
    omni=True,
    cluster=True,
    resample=True,
    verbose=False,
    tile=False,
    niter=None,
    augment=False,
    affinity_seg=False
)

os.makedirs(outline_dir,exist_ok=True)
use_GPU = core.use_gpu()
print(f"GPU available : {use_GPU}")

model = models.CellposeModel(gpu=use_GPU,model_type=model_name)

imgs = sorted(glob.glob(os.path.join(base_dir,"*tif")))
if not imgs:
    raise FileNotFoundError(f"No images found under {base_dir}")

def save_outline_tif(mask_array: np.ndarray, out_tif_path: str):
    """ masks -> binary outlines -> TIFF (uint8 0/255) """
    outline_bool = cp_utils.masks_to_outlines(mask_array.astype(np.int32))
    outline_img = (outline_bool.astype(np.uint8) * 255)
    tifffile.imwrite(out_tif_path, outline_img)

# ---------- Main loop ----------
t_start = time.time()
for i, img_path in enumerate(imgs, 1):
    print(f"[{i}/{len(imgs)}] Processing: {img_path}")
    try:
        # 読み込みと正規化
        img_raw = io.imread(img_path)
        img = transforms.move_min_dim(img_raw)
        img = normalize99(img)

        # 推論
        tic = time.time()
        masks, flows, styles = model.eval([img], **params)
        print(f"  - segmentation: {time.time()-tic:.2f}s")

        # 保存: *_seg.npy 等は元TIFFと同じフォルダ
        io.save_masks(
            img, masks, flows,
            file_names=[img_path],
            savedir=None,
            save_flows=False,
            save_outlines=False,
            save_plot=False
        )
        print("  - npy/masks saved next to the TIFF")

        # アウトラインをTIFF保存 (outlines_only/)
        maski = masks[0]
        base = os.path.splitext(os.path.basename(img_path))[0]
        outline_name = f"{base}_outlines.tif"
        outline_path = os.path.join(outline_dir, outline_name)
        save_outline_tif(maski, outline_path)
        print(f"  - outline saved: {outline_path}")

    except Exception as e:
        print(f"  ! ERROR on {img_path}: {e}")

print(f"Done. Total time: {time.time()-t_start:.1f}s")
print(f"Outlines collected in: {outline_dir}")

# %% 251010 argument
import os, numpy as np, tifffile as tiff

train_dir = r"C:\Users\QPI\Desktop\train"

def rot180(a):
    # 180度回転（補間なし、dtype不問）
    return a[::-1, ::-1]

for fn in os.listdir(train_dir):
    if fn.lower().endswith(".tif") and not fn.endswith("_rot180.tif"):
        base = os.path.splitext(fn)[0]
        img_p = os.path.join(train_dir, fn)
        # 対応マスク（_masks.tif / _seg.npy の両対応）
        m_tif = os.path.join(train_dir, base + "_masks.tif")
        m_npy = os.path.join(train_dir, base + "_seg.npy")

        if not (os.path.exists(m_tif) or os.path.exists(m_npy)):
            print(f"skip (no mask): {fn}"); continue

        # 画像読み込み→回転→保存
        img = tiff.imread(img_p)  # float32 でもそのままOK
        img_r = rot180(img)
        out_img = os.path.join(train_dir, base + "_rot180.tif")
        tiff.imwrite(out_img, img_r.astype(img.dtype))
        print("wrote", os.path.basename(out_img))

        # マスクが _masks.tif の場合
        if os.path.exists(m_tif):
            m = tiff.imread(m_tif)
            m_r = rot180(m)
            # マスクは 0=背景, 1..K の整数。uint16 に寄せると無難
            out_m = os.path.join(train_dir, base + "_rot180_masks.tif")
            tiff.imwrite(out_m, m_r.astype(np.uint16))
            print("wrote", os.path.basename(out_m))

        # マスクが _seg.npy の場合
        if os.path.exists(m_npy):
            d = np.load(m_npy, allow_pickle=True).item()
            m = d.get("masks", None)
            if m is None:
                print("  warn: masks is None in", m_npy)
            else:
                m_r = rot180(m)
                out_n = os.path.join(train_dir, base + "_rot180_seg.npy")
                np.save(out_n, {"masks": m_r.astype(np.uint16)})
                print("wrote", os.path.basename(out_n))

# %%
import tifffile as tiff, numpy as np, glob
for f in glob.glob(r"C:\Users\QPI\Desktop\train\*_masks.tif"):
    m = tiff.imread(f)
    print(f, m.dtype, m.min(), m.max(), np.unique(m)[:10])

# %%
# %% segmentation with your trained model (nchan=1, nclasses=2)
# %% segmentation with your trained model (nchan=1, nclasses=2)
import os, glob, time, re
import numpy as np
import tifffile
from cellpose_omni import io, transforms, models, core
from cellpose_omni import utils as cp_utils
from omnipose.utils import normalize99

# ====== paths ======
base_dir    = r"G:\Pos2\output_phase\aligned_left_center\crop_dif_0_1.5"
outline_dir = os.path.join(base_dir, "outline")
model_dir   = r"C:\Users\QPI\Desktop\train\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_1_dim_2_train_2025_10_11_18_06_05.120367"

# ====== pick latest checkpoint in model_dir ======
cands = [os.path.join(model_dir, f) for f in os.listdir(model_dir)]
ckpts = [p for p in cands if os.path.isfile(p) and (p.endswith((".pth",".pt")) or re.search(r"_\d+$", p))]
if not ckpts:
    raise FileNotFoundError(f"No checkpoint file found under: {model_dir}")
ckpt = sorted(ckpts, key=os.path.getmtime)[-1]
print(f"[ckpt] using: {ckpt}")

# ====== params ======
params = dict(
    channels=[0,0],      # グレースケールを両chに割当（nchan=1でも安全）
    rescale=None,
    mask_threshold=0,
    flow_threshold=0,
    transparency=True,
    omni=True,
    cluster=True,
    resample=True,
    verbose=False,
    tile=False,
    niter=None,
    augment=False,
    affinity_seg=False
)

os.makedirs(outline_dir, exist_ok=True)

use_GPU = core.use_gpu()
print(f"GPU available : {use_GPU}")

# 重要：pretrained_model= に ckpt パス、かつ nchan=1, nclasses=2, omni=True を明示
model = models.CellposeModel(
    gpu=use_GPU,
    pretrained_model=ckpt,
    nchan=1,
    nclasses=2,
    omni=True
)

# 入力画像（rot系を除外したい場合は下の1行を有効化）
imgs = sorted(glob.glob(os.path.join(base_dir, "*.tif")))
# imgs = [p for p in imgs if "rot" not in os.path.basename(p).lower()]

if not imgs:
    raise FileNotFoundError(f"No images found under {base_dir}")

def save_outline_tif(mask_array: np.ndarray, out_tif_path: str):
    outline_bool = cp_utils.masks_to_outlines(mask_array.astype(np.int32))
    outline_img = (outline_bool.astype(np.uint8) * 255)
    tifffile.imwrite(out_tif_path, outline_img)

def safe_normalize(im: np.ndarray) -> np.ndarray:
    im = np.asarray(im)
    im = np.nan_to_num(im, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize99(im)

# ---------- Main loop ----------
t_start = time.time()
for i, img_path in enumerate(imgs, 1):
    print(f"[{i}/{len(imgs)}] Processing: {img_path}")
    try:
        img_raw = io.imread(img_path)
        img_raw = np.squeeze(img_raw)   # (Y,X)
        img = safe_normalize(img_raw)

        tic = time.time()
        masks_list, flows_list, styles_list = model.eval([img], **params)
        print(f"  - segmentation: {time.time()-tic:.2f}s")

        maski = masks_list[0]
        flowi = flows_list[0]

        io.save_masks(
            img, maski, flowi,
            file_names=[img_path],
            savedir=None,            # 元フォルダに _seg.npy / _masks.tif
            save_flows=False,
            save_outlines=False,
            save_plot=False
        )
        base = os.path.splitext(os.path.basename(img_path))[0]
        outline_path = os.path.join(outline_dir, f"{base}_outlines.tif")
        save_outline_tif(maski, outline_path)
        print(f"  - outline saved: {outline_path}")

    except Exception as e:
        print(f"  ! ERROR on {img_path}: {e}")

print(f"Done. Total time: {time.time()-t_start:.1f}s")
print(f"Outlines collected in: {outline_dir}")

# %%
