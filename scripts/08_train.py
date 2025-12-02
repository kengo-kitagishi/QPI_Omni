# %%
from cellpose_omni import models, utils, io
import numpy as np
import os
import tifffile
import glob

lower = 5
upper = 95

def normalize_custom(img):
    return util.normalize99(img, lower=lower,upper=upper)

def normalize_none(img):
    return img.astype(np.float32)


# ==== パラメータ設定 ====
train_dir = r"C:\Users\QPI\Desktop\train"

use_gpu = True
nchan = 1
nclasses = 3  # ※nclassesはモデル作成時に使用するかどうか環境による
diameter = 0
optimizer = 'RAdam'   # ただし公式 train() 構成には optimizer= 引数が明記されていないので注意
learning_rate = 0.0001
batch_size = 1
save_every = 100
n_epochs = 3000
crop_size = (32, 64)  # tyx

# ==== ファイル収集 ====
# 画像とマスクを手動でペアリング
image_paths = sorted(glob.glob(os.path.join(train_dir, "*.tif")))
mask_paths  = [p.replace(".tif", "_masks.tif") for p in image_paths]
mask_paths  = [p if os.path.exists(p) else None for p in mask_paths]

# 存在するペアだけ残す
pairs = [(im, m) for im, m in zip(image_paths, mask_paths) if m is not None]
image_paths = [im for im, _ in pairs]
mask_paths  = [m for _, m in pairs]

print(f"Found {len(image_paths)} valid image-mask pairs.")

# ==== 読み込み ====
imgs  = [tifffile.imread(p) for p in image_paths]
masks = [tifffile.imread(p) for p in mask_paths]

# ==== モデル作成 ====
model = models.CellposeModel(
    gpu=use_gpu,
    pretrained_model=None,
    omni=True,
    nchan=nchan,
    nclasses=nclasses
)

# ==== 学習 ====
model.train(
    train_data=imgs,
    train_labels=masks,
    channels=None,
    normalize=False,          # 公式デフォルト True をオフにして前処理済にする
    save_path=None,
    save_every=save_every,
    learning_rate=learning_rate,
    n_epochs=n_epochs,
    batch_size=batch_size,
    tyx=crop_size,
    rescale=False
)

print("Done")
# %%
import os
train_dir = r"C:\Users\QPI\Desktop\train"
print("Train dir =", train_dir)
print("Exists =", os.path.exists(train_dir))
# %%
from cellpose_omni import io

res = io.get_image_files(train_dir, mask_filter='_masks')
print("Returned object:", type(res))
print("Length:", len(res))

for i, item in enumerate(res):
    print(f"[{i}] Type: {type(item)}  Example: {item[:3] if isinstance(item, list) else item}")

# %%
import os
import glob
import tifffile
import numpy as np
from cellpose_omni import models

lower = 5
upper = 95

def normalize_custom(img):
    return util.normalize99(img, lower=lower,upper=upper)

# === パラメータ（必要なら調整） ===
os.chdir(r"C:\Users\QPI\Desktop")           # 安全のため作業ディレクトリを固定
train_dir = r"C:\Users\QPI\Desktop\verti_flip_train" 
use_gpu = True
nchan = 1
nclasses = 3
learning_rate = 0.0001
diameter = 20
batch_size = 2
save_every = 100
n_epochs = 3000
crop_size = (32, 96)  # tyx
save_dir = r"C:\Users\QPI\Desktop\verti_flip_train\omni_model"

# === ファイル収集（確実） ===
# 画像 (*.tif) を集めて、対応する *_masks.tif が存在するものだけ採用
image_paths = sorted(glob.glob(os.path.join(train_dir, "*.tif")))
# ignore files that already are masks (in case both are in same folder)
image_paths = [p for p in image_paths if not p.endswith("_masks.tif")]

mask_paths = [p.replace(".tif", "_masks.tif") for p in image_paths]
pairs = [(im, m) for im, m in zip(image_paths, mask_paths) if os.path.exists(m)]
image_paths = [im for im, _ in pairs]
mask_paths  = [m for _, m in pairs]

print(f"Found {len(image_paths)} image-mask pairs in {train_dir}")
if len(image_paths) == 0:
    raise RuntimeError("No valid (image, mask) pairs found. Check filenames and '_masks.tif' naming.")

# === 読み込み・型・次元チェック ===
imgs = []
masks = []
bad = []
for im_path, mask_path in zip(image_paths, mask_paths):
    try:
        img = tifffile.imread(im_path)
    except Exception as e:
        bad.append((im_path, f"image read error: {e}"))
        continue
    try:
        msk = tifffile.imread(mask_path)
    except Exception as e:
        bad.append((mask_path, f"mask read error: {e}"))
        continue

    # 次元チェック（少なくとも2D）
    if getattr(img, "ndim", 0) < 2:
        bad.append((im_path, f"image ndim < 2: {getattr(img,'ndim',None)}"))
        continue
    if getattr(msk, "ndim", 0) < 2:
        bad.append((mask_path, f"mask ndim < 2: {getattr(msk,'ndim',None)}"))
        continue

    # マスクは整数型に（ラベリング画像であること）
    msk_arr = np.asarray(msk)
    if not np.issubdtype(msk_arr.dtype, np.integer):
        try:
            msk_arr = msk_arr.astype(np.int32)
        except Exception as e:
            bad.append((mask_path, f"mask dtype conversion failed: {e}"))
            continue

    # 画像は float32 に（network が想定する入力）
    img_arr = np.asarray(img).astype(np.float32)

    imgs.append(img_arr)
    masks.append(msk_arr)

if bad:
    print("Some files were skipped due to errors:")
    for p, reason in bad:
        print("  -", p, " =>", reason)

if len(imgs) == 0:
    raise RuntimeError("No valid images/masks loaded after checks. Fix issues first.")

print(f"Loaded {len(imgs)} images and {len(masks)} masks for training.")

# === train_links と train_files を明示的に作る（zipでのTypeError対策） ===
train_links = [None] * len(masks)     # Omnipose の train() が zip(train_labels, train_links) するため必須
train_files = image_paths[:len(masks)]  # optional: デバッグ/保存用にファイル名 list を渡す

# === モデル ===
model = models.CellposeModel(
    gpu=use_gpu,
    pretrained_model=r"C:\Users\QPI\Desktop\verti_flip_train\omni_model\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_omni_model_2025_11_05_18_13_10.220871",
    omni=True,
    nchan=nchan,
    nclasses=nclasses
)

# === 学習 ===
try:
    model.train(
        train_data=imgs,
        train_labels=masks,
        train_links=train_links,
        train_files=train_files,
        channels=None,
        normalize=False,          # 正規化OFF
        save_path=save_dir,
        save_every=save_every,
        learning_rate=learning_rate,
        min_train_masks=1,
        n_epochs=n_epochs,
        batch_size=batch_size,
        #tyx=crop_size,
        rescale=False
    )
    print("Training finished without exception.")
except Exception as e:
    print("model.train() raised an exception:", repr(e))
    raise

# %%
