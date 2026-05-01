
# %%
import os
import sys
import re
import glob
import logging
import datetime
import tifffile
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cellpose_omni import models

sys.path.insert(0, r"C:\Users\QPI\Documents\QPI_omni\scripts")
from figure_logger import save_figure

torch.backends.cudnn.benchmark = True

# === Parameters (adjust as needed) ===
os.chdir(r"C:\Users\QPI\Desktop")
train_dir    = r"C:\Users\QPI\Desktop\train"
use_gpu      = True
nchan        = 1
nclasses     = 3
lr_value     = 0.01
diameter     = 20
batch_size   = 5
save_every   = 100
n_epochs     = 1500
crop_size    = (40, 128)   # tyx
save_dir     = r"C:\Users\QPI\Desktop\train\omni_model_d20"
pretrained_model = None
loss_log_path = r"C:\Users\QPI\Desktop\train\train_loss.log"

# Pass LR as an array to disable warmup (constant LR throughout)
learning_rate = np.full(n_epochs, lr_value)

# === Log settings (append mode with run separators) ===
os.makedirs(os.path.dirname(loss_log_path), exist_ok=True)
_pretrained_label = os.path.basename(pretrained_model) if pretrained_model else "scratch"
with open(loss_log_path, mode='a', encoding='utf-8') as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"=== Run started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    f.write(f"=== LR={lr_value}, n_epochs={n_epochs}, pretrained={_pretrained_label} ===\n")
    f.write(f"{'='*60}\n")

_loss_fh = logging.FileHandler(loss_log_path, mode='a', encoding='utf-8')
_loss_fh.setLevel(logging.DEBUG)
_cp_logger = logging.getLogger('cellpose_omni.core')
_cp_logger.addHandler(_loss_fh)
_cp_logger.setLevel(logging.DEBUG)

# === Collect files ===
image_paths = sorted(glob.glob(os.path.join(train_dir, "*.tif")))
image_paths = [p for p in image_paths if not p.endswith("_masks.tif")]
mask_paths  = [p.replace(".tif", "_masks.tif") for p in image_paths]
pairs       = [(im, m) for im, m in zip(image_paths, mask_paths) if os.path.exists(m)]
image_paths = [im for im, _ in pairs]
mask_paths  = [m  for _,  m in pairs]

print(f"Found {len(image_paths)} image-mask pairs in {train_dir}")
if len(image_paths) == 0:
    raise RuntimeError("No valid (image, mask) pairs found.")

# === Load data ===
imgs, masks, bad = [], [], []
for im_path, mask_path in zip(image_paths, mask_paths):
    try:
        img = tifffile.imread(im_path)
    except Exception as e:
        bad.append((im_path, f"image read error: {e}")); continue
    try:
        msk = tifffile.imread(mask_path)
    except Exception as e:
        bad.append((mask_path, f"mask read error: {e}")); continue
    if getattr(img, "ndim", 0) < 2:
        bad.append((im_path, f"ndim < 2")); continue
    if getattr(msk, "ndim", 0) < 2:
        bad.append((mask_path, f"ndim < 2")); continue
    msk_arr = np.asarray(msk)
    if not np.issubdtype(msk_arr.dtype, np.integer):
        msk_arr = msk_arr.astype(np.int32)
    imgs.append(np.asarray(img).astype(np.float32))
    masks.append(msk_arr)

if bad:
    print("Skipped files:")
    for p, r in bad: print("  -", p, "=>", r)
if len(imgs) == 0:
    raise RuntimeError("No valid images/masks loaded.")
print(f"Loaded {len(imgs)} images and {len(masks)} masks.")

train_links = [None] * len(masks)
train_files = image_paths[:len(masks)]

# === Model ===
model = models.CellposeModel(
    gpu=use_gpu,
    pretrained_model=pretrained_model,
    omni=True,
    nchan=nchan,
    nclasses=nclasses
)

# === Training ===
try:
    model.train(
        train_data=imgs,
        train_labels=masks,
        train_links=train_links,
        train_files=train_files,
        channels=None,
        normalize=True,
        save_path=save_dir,
        save_every=save_every,
        learning_rate=learning_rate,
        min_train_masks=1,
        n_epochs=n_epochs,
        batch_size=batch_size,
        dataloader=False,
        num_workers=0,
        do_autocast=True,
        tyx=crop_size,
        rescale=False
    )
    print("Training finished without exception.")
except Exception as e:
    print("model.train() raised an exception:", repr(e))
    raise

# === Save training curve via figure_logger ===
_pat = re.compile(r"Train epoch:\s*(\d+).*?<Epoch Loss>:\s*([\d.]+)")
_epochs, _losses = [], []
with open(loss_log_path, encoding='utf-8') as f:
    for line in f:
        m = _pat.search(line)
        if m:
            _epochs.append(int(m.group(1)))
            _losses.append(float(m.group(2)))

if _epochs:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(_epochs, _losses, color="steelblue", linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epoch Loss")
    ax.set_title(f"Omnipose Training Loss  (LR={lr_value}, n_epochs={n_epochs})")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()

    save_figure(
        fig,
        params={"lr": lr_value, "n_epochs": n_epochs, "normalize": True,
                "tyx": str(crop_size), "pretrained": os.path.basename(pretrained_model)},
        description="Omnipose training loss curve with 50-epoch moving average"
    )
    plt.close(fig)
    print("Loss curve saved via figure_logger.")

# %%
