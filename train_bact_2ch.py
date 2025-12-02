import os, glob
import numpy as np
import tifffile as tiff
from cellpose_omni import models

DATA_DIR = r"C:\Users\QPI\Desktop\train_dup2"
OUT_DIR  = os.path.join(DATA_DIR, "models_api")
os.makedirs(OUT_DIR, exist_ok=True)

img_paths = sorted([p for p in glob.glob(os.path.join(DATA_DIR, "*.tif")) if not p.endswith("_masks.tif")])
imgs, masks = [], []
for ip in img_paths:
    stem = os.path.splitext(os.path.basename(ip))[0]
    mp = os.path.join(DATA_DIR, stem + "_masks.tif")
    if not os.path.exists(mp):
        print(f"[skip] mask missing: {stem}")
        continue
    im = tiff.imread(ip)   # 期待: CYX (2,H,W)
    ms = tiff.imread(mp)   # 期待: (H,W)
    im = np.asarray(im)
    ms = np.squeeze(np.asarray(ms))
    if im.ndim != 3 or im.shape[0] != 2:
        raise ValueError(f"image must be CYX (2,H,W): {ip}, got {im.shape}")
    if im.shape[1:] != ms.shape:
        raise ValueError(f"shape mismatch: {ip} {im.shape} vs {ms.shape}")
    imgs.append(im)
    masks.append(ms.astype(np.int32))

print(f"[INFO] Loaded {len(imgs)} image/mask pairs")

# モデル（bact_fluor_omni: 2ch/3class）
model = models.CellposeModel(
    gpu=True,
    model_type='bact_fluor_omni',
    omni=True,
    nchan=2,
    nclasses=3
)

# ★ 追加：train_links をダミーで渡す（各サンプルに None）
train_links = [None] * len(masks)

loss = model.train(
    imgs, masks,
    channels=[0, 1],
    channel_axis=0,          # CYX
    n_epochs=400,
    learning_rate=0.002,
    batch_size=1,
    tyx=(128, 128),
    rescale=False,
    save_every=100,
    save_path=OUT_DIR,
    SGD=False,
    normalize=True,
    train_links=train_links   # ← これが重要
)

print(f"[DONE] Training complete. Models saved in: {OUT_DIR}")
