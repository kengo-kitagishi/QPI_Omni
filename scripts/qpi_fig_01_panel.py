# %%
# ===========================================================
# [DEPRECATED] このスクリプトは使用しない。
# 後継: qpi_fig_01_reconstruction_procedure.py
#   → 6パネル・スネーク状レイアウト（a-b-c / f-e-d）に刷新。
# ===========================================================

"""
qpi_fig_01_panel.py

qpi_fig_01_reconstruction_overview の出力画像を組み合わせた修論用パネル図。

入力: Google Drive inbox の PNG (f001, f003, f005, f006)
出力: 4パネル横並び + 解析順矢印ラベル + スケールバー

パネル構成:
  Raw image → (FFT) → FFT + filter circles → (Crop) → Cropped FFT → (IFFT) → Phase
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, "/Users/kitak/QPI_Omni/scripts")
from figure_logger import save_figure

# ============================================================
# 入力パス (Drive inbox)
# ============================================================
INBOX = (
    "/Users/kitak/Library/CloudStorage/"
    "GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/"
    "共有ドライブ/wakamotolab_meeting/kitagishi/figure-hub/inbox/"
    "2026-02-28/qpi_fig_01_reconstruction_overview/"
    "qpi_fig_01_reconstruction_overview_20260228T125954Z_834482"
)
PRE = (
    "qpi_fig_01_reconstruction_overview__"
    "qpi_fig_01_reconstruction_overview_20260228T125954Z_834482__"
)


def load_png(fname, grayscale=True):
    path = os.path.join(INBOX, PRE + fname)
    img = Image.open(path)
    return np.array(img.convert("L") if grayscale else img.convert("RGB"))


img_raw   = load_png("f001.png", grayscale=True)
img_fft   = load_png("f003.png", grayscale=False)  # RGB (赤い円を保持)
img_crop  = load_png("f005.png", grayscale=True)
img_phase = load_png("f006.png", grayscale=True)

# ============================================================
# スケールバー設定
# ============================================================
PIXELSIZE_UM = 3.45 / 40   # 0.08625 µm/px (センサー 3.45 µm, 40x 対物)
ORIG_SIZE    = 2048         # 元データのピクセルサイズ
SCALEBAR_UM  = 10           # µm

H, W = img_raw.shape
scale_px = int(round(SCALEBAR_UM / PIXELSIZE_UM * (H / ORIG_SIZE)))


def add_scalebar(ax, img_shape, scale_px, label,
                 color="white", pad_frac=0.05, thickness_frac=0.015):
    """右下にスケールバーを追加"""
    h, w = img_shape
    pad = int(h * pad_frac)
    thickness = max(2, int(h * thickness_frac))
    x0 = w - pad - scale_px
    y0 = h - pad - thickness
    rect = mpatches.Rectangle(
        (x0, y0), scale_px, thickness,
        linewidth=0, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x0 + scale_px / 2, y0 - h * 0.012, label,
            color=color, ha="center", va="bottom",
            fontsize=9, fontweight="bold")


# ============================================================
# パネル図
# ============================================================
panels        = [img_raw,    img_fft,  img_crop,      img_phase]
titles        = ["Raw image", "FFT",   "Cropped FFT", "Phase"]
show_scalebar = [True,        False,   False,          True]
arrow_labels  = ["FFT",       "Crop",  "IFFT"]

# GridSpec: image(4) + arrow(3) = 7 cols
fig = plt.figure(figsize=(18, 5))
gs = GridSpec(
    1, 7,
    width_ratios=[10, 2, 10, 2, 10, 2, 10],
    left=0.01, right=0.99,
    top=0.88, bottom=0.03,
    wspace=0,
)

img_axes = [fig.add_subplot(gs[0, i * 2])     for i in range(4)]
arr_axes = [fig.add_subplot(gs[0, i * 2 + 1]) for i in range(3)]

for i, (ax, img, title, sb) in enumerate(zip(img_axes, panels, titles, show_scalebar)):
    if i == 1:  # FFT: RGB
        ax.imshow(img)
    else:
        ax.imshow(img, cmap="gray")
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=6)
    if sb:
        add_scalebar(ax, img.shape, scale_px, f"{SCALEBAR_UM} µm")

for ax, label in zip(arr_axes, arrow_labels):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.annotate(
        "",
        xy=(0.85, 0.5),    xycoords="axes fraction",
        xytext=(0.15, 0.5), textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->,head_width=0.4,head_length=0.3",
            color="black", lw=1.5,
        ),
    )
    ax.text(0.5, 0.67, label,
            ha="center", va="center",
            fontsize=12, transform=ax.transAxes)

save_figure(
    fig,
    params={
        "scalebar_um": SCALEBAR_UM,
        "source_run": "20260228T125954Z_834482",
        "panels": "f001_raw, f003_fft_red, f005_crop, f006_phase",
    },
    description=(
        "QPI再構成概要パネル図（修論用）: "
        "Raw → FFT → Cropped FFT → Phase, "
        "矢印ラベル・スケールバー付き"
    ),
)

# PDF（Affinity Designer編集・figure-hub登録用）・SVG も inbox に保存
import pathlib
inbox_dir = pathlib.Path(
    "/Users/kitak/Library/CloudStorage/"
    "GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/"
    "共有ドライブ/wakamotolab_meeting/kitagishi/figure-hub/inbox/2026-03-01/qpi_fig_01_panel"
)
inbox_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(inbox_dir / "qpi_fig_01_panel.pdf", bbox_inches="tight")
fig.savefig(inbox_dir / "qpi_fig_01_panel.svg", bbox_inches="tight")

# thesis/figure/ への直接保存は行わない。
# 修論への反映は figure-hub で管理する:
#   register → use → sync → git push

print(f"PNG: results/figures/")
print(f"PDF/SVG: {inbox_dir}")
