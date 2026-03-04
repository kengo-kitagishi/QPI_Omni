# %%
"""
qpi_fig_01_reconstruction_procedure.py

QPIホログラム再構成手順の6パネル図（修論用）。

パネル構成（スネーク状レイアウト）:
  上段 (左→右):  a: raw hologram  →  b: 2D FFT (with freq. annotations)  →  c: sideband-centered FFT
                                                                                      ↓ LP filtering
  下段 (右→左):  f: final (Amp/OPD)  ←  e: after 2D IFT  ←  d: LP-filtered

矢印ラベル:
  a→b : 2D discrete FT
  b→c : sideband centering
  c→d : LP filtering（右端縦ブラケット）
  d→e : 2D discrete IFT
  e→f : background subtraction

入力: Google Drive inbox の PNG
出力: figure-hub inbox → thesis/figure（figure-hubで管理）
"""

import sys
import os
import pathlib
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, "/Users/kitak/QPI_Omni/scripts")
from figure_logger import save_figure

# ============================================================
# 入力パス (Drive inbox)
# TODO: 日付・フォルダ名・プレフィックスを実際の run に合わせて変更
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


# ============================================================
# 画像読み込み（ファイルが存在しない場合はグレーのプレースホルダー）
# ============================================================
def load_png(fname, grayscale=True):
    path = os.path.join(INBOX, PRE + fname)
    if not os.path.exists(path):
        print(f"[WARNING] not found, using placeholder: {path}")
        size = (512, 512) if grayscale else (512, 512, 3)
        return np.full(size, 128, dtype=np.uint8)
    img = Image.open(path)
    return np.array(img.convert("L") if grayscale else img.convert("RGB"))


def crop_bottom_right_quarter_then_2x(img):
    """右下1/4を切り出し、2x2拡大（最近傍）して元サイズにそろえる。"""
    h, w = img.shape[:2]
    cropped = img[h // 2 :, w // 2 :]
    upscaled = np.repeat(np.repeat(cropped, 2, axis=0), 2, axis=1)
    if img.ndim == 2:
        return upscaled[:h, :w]
    return upscaled[:h, :w, :]


# TODO: 各パネルに対応するファイル名を確認・修正する
img_a = crop_bottom_right_quarter_then_2x(load_png("f001.png", grayscale=True))    # a
img_b = crop_bottom_right_quarter_then_2x(load_png("f003.png", grayscale=False))    # b
img_c = load_png("f005.png", grayscale=True)   # c: sideband-centered FFT  ← TODO: 要確認
img_d = load_png("f004.png", grayscale=True)   # d: LP-filtered (bright circle in dark)  ← TODO: ファイル名要確認
img_e = crop_bottom_right_quarter_then_2x(load_png("f006.png", grayscale=True))     # e
img_f = crop_bottom_right_quarter_then_2x(load_png("f007.png", grayscale=False))     # f

# ============================================================
# スケールバー設定
# ============================================================
PIXELSIZE_UM = 3.45 / 40  # 0.08625 µm/px（センサー 3.45 µm, 40x 対物）
ORIG_SIZE    = 2048
SCALEBAR_UM  = 10

H, W = img_a.shape[:2]
scale_px = int(round(SCALEBAR_UM / PIXELSIZE_UM * (H / ORIG_SIZE)))


def add_scalebar(ax, img_shape, scale_px, label,
                 color="white", pad_frac=0.05, thickness_frac=0.015):
    h, w = img_shape[:2]
    pad       = int(h * pad_frac)
    thickness = max(2, int(h * thickness_frac))
    x0 = pad
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
# GridSpec レイアウト
#
#  cols:  0=img    1=arrow  2=img    3=arrow  4=img    5=lp_bracket
#  row0:  img_a  → arr_ab → img_b  → arr_bc → img_c  |
#                                                      | LP filtering
#  row1:  img_f  ← arr_ef ← img_e  ← arr_de ← img_d  |
# ============================================================
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(
    2, 6,
    width_ratios=[10, 2, 10, 2, 10, 3],
    height_ratios=[10, 10],
    left=0.03, right=0.98,
    top=0.93, bottom=0.05,
    wspace=0.0, hspace=0.15,
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 2])
ax_c = fig.add_subplot(gs[0, 4])
ax_d = fig.add_subplot(gs[1, 4])
ax_e = fig.add_subplot(gs[1, 2])
ax_f = fig.add_subplot(gs[1, 0])

ax_arr_ab = fig.add_subplot(gs[0, 1])
ax_arr_bc = fig.add_subplot(gs[0, 3])
ax_arr_de = fig.add_subplot(gs[1, 3])
ax_arr_ef = fig.add_subplot(gs[1, 1])

ax_lp = fig.add_subplot(gs[:, 5])  # LP filtering ブラケット（2行にまたがる）

# ============================================================
# 画像表示
# ============================================================
panels = [
    # (ax,   img,   label, grayscale, scalebar)
    (ax_a, img_a, "a", True,  True),
    (ax_b, img_b, "b", False, False),
    (ax_c, img_c, "c", True,  False),
    (ax_d, img_d, "d", True,  False),
    (ax_e, img_e, "e", True,  False),
    (ax_f, img_f, "f", False, True),
]

for ax, img, label, gray, sb in panels:
    ax.imshow(img, cmap="gray" if gray else None)
    ax.axis("off")
    # パネルラベル（左上・画像外）
    ax.text(-0.04, 1.06, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="right")
    if sb:
        add_scalebar(ax, img.shape, scale_px, f"{SCALEBAR_UM} µm")


def save_panel_figure(img, label, gray, scalebar):
    panel_fig, panel_ax = plt.subplots(figsize=(4.2, 4.2))
    panel_ax.imshow(img, cmap="gray" if gray else None)
    panel_ax.axis("off")
    panel_ax.text(-0.04, 1.06, label, transform=panel_ax.transAxes,
                  fontsize=14, fontweight="bold", va="top", ha="right")
    if scalebar:
        add_scalebar(panel_ax, img.shape, scale_px, f"{SCALEBAR_UM} µm")
    save_figure(
        panel_fig,
        params={
            "parent_figure": "qpi_fig_01_reconstruction_procedure",
            "panel": label,
            "crop_mode": "right-bottom quarter -> 2x upsample (a,b,e,f only)",
        },
        description=f"QPI再構成手順パネル {label} 単体保存",
        data_source={
            "run_id_data": "qpi_fig_01_reconstruction_overview_20260228T125954Z_834482",
            "notes": f"single panel export for {label}",
        },
    )
    plt.close(panel_fig)

# ============================================================
# 水平矢印
# ============================================================
arrow_axes = [
    # (ax,        label,                  left_to_right)
    (ax_arr_ab, "2D discrete FT",         True),
    (ax_arr_bc, "sideband\ncentering",    True),
    (ax_arr_de, "2D\ndiscrete IFT",       False),   # 右→左
    (ax_arr_ef, "background\nsubtraction", False),  # 右→左
]

for ax, label, ltr in arrow_axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x_tail, x_head = (0.15, 0.85) if ltr else (0.85, 0.15)
    ax.annotate(
        "",
        xy=(x_head, 0.5), xycoords="axes fraction",
        xytext=(x_tail, 0.5), textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->,head_width=0.4,head_length=0.3",
            color="black", lw=1.5,
        ),
    )
    ax.text(0.5, 0.73, label,
            ha="center", va="center",
            fontsize=10, transform=ax.transAxes)

# ============================================================
# LP filtering 縦ブラケット（右端、c → d）
#   ┌─  (img_c の右端から)
#   │  LP filtering
#   └─  (img_d の右端へ)
# ============================================================
ax_lp.set_xlim(0, 1)
ax_lp.set_ylim(0, 1)
ax_lp.axis("off")

x_line  = 0.35  # 縦線の x 位置
x_label = 0.75  # ラベルの x 位置
y_top   = 0.94
y_bot   = 0.06

# 上の水平線（img_c 右端から縦線へ）
ax_lp.plot([0.05, x_line], [y_top, y_top], "k-", lw=1.5, transform=ax_lp.transAxes)
# 縦線
ax_lp.plot([x_line, x_line], [y_top, y_bot], "k-", lw=1.5, transform=ax_lp.transAxes)
# 下の矢印（縦線から img_d 左端へ）
ax_lp.annotate(
    "",
    xy=(0.05, y_bot), xycoords="axes fraction",
    xytext=(x_line, y_bot), textcoords="axes fraction",
    arrowprops=dict(
        arrowstyle="->,head_width=0.3,head_length=0.15",
        color="black", lw=1.5,
    ),
)
# ラベル
ax_lp.text(x_label, 0.5, "LP\nfiltering",
           ha="center", va="center",
           fontsize=10, transform=ax_lp.transAxes)

# ============================================================
# 保存
# ============================================================
save_figure(
    fig,
    params={
        "scalebar_um": SCALEBAR_UM,
        "panels":      "a=f001, b=f003(RGB), c=f005, d=TODO, e=TODO, f=TODO",
        "crop_mode":   "a,b,e,f = right-bottom quarter then 2x upsample",
        "layout":      "snake 2x3 (a-b-c top / f-e-d bottom)",
    },
    description=(
        "QPI再構成手順6パネル図（修論用）: "
        "raw → 2D FFT → sideband centering → LP filtering → 2D IFT → background subtraction, "
        "スネーク状レイアウト（上段a-b-c / 下段f-e-d）"
    ),
    data_source={
        "raw_files": [
            str(pathlib.Path(INBOX) / (PRE + "f001.png")),  # a: raw hologram
            str(pathlib.Path(INBOX) / (PRE + "f003.png")),  # b: FFT + red circles
            str(pathlib.Path(INBOX) / (PRE + "f005.png")),  # c/d: sideband-centered / LP-filtered
        ],
        "measured_on": "2026-02-28",
        "run_id_data": "qpi_fig_01_reconstruction_overview_20260228T125954Z_834482",
        "notes": "source panels generated by qpi_fig_01_reconstruction_overview.py on measurement PC",
    },
)

# 各パネルを同一 save_figure フォルダ(run_id)にも保存
for _, panel_img, panel_label, panel_gray, panel_scalebar in panels:
    save_panel_figure(panel_img, panel_label, panel_gray, panel_scalebar)

# figure-hub inbox にも保存
today = datetime.date.today().strftime("%Y-%m-%d")
inbox_out = pathlib.Path(
    "/Users/kitak/Library/CloudStorage/"
    "GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/"
    "共有ドライブ/wakamotolab_meeting/kitagishi/figure-hub/inbox/"
    f"{today}/qpi_fig_01_reconstruction_procedure"
)
inbox_out.mkdir(parents=True, exist_ok=True)
fig.savefig(inbox_out / "qpi_fig_01_reconstruction_procedure.pdf", bbox_inches="tight")
fig.savefig(inbox_out / "qpi_fig_01_reconstruction_procedure.svg", bbox_inches="tight")

print(f"PNG: results/figures/")
print(f"PDF/SVG: {inbox_out}")
