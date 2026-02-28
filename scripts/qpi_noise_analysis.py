# %%
# ============================================================
# qpi_noise_analysis.py — QPIノイズ解析
# ============================================================
# UC1: 単一画像の任意位置プロファイル
# UC2: タイムラプスノイズ追跡
# UC3: シフト量とノイズの相関
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
# 共通設定（各UCで上書き可）
# ============================================================

WAVELENGTH  = 658e-9        # m
NA          = 0.95
PIXELSIZE   = 3.45e-6 / 40  # m/px (3.45 µm カメラ画素, 40x対物)
OFFAXIS_CENTER = (1664, 485) # (row, col) — CursorVisualizerで確認した値

# ============================================================
# UC1: 単一画像の任意位置プロファイル
# ============================================================

# %%
# --- UC1 設定 ---

UC1_PATH    = "path/to/image.tif"       # 解析対象フレームのTIFFパス
UC1_PATH_BG = "path/to/background.tif" # 背景TIFFパス
UC1_CROP    = (8, 2056, 208, 2256)      # (row_start, row_end, col_start, col_end)
UC1_PROFILE_COORD = 200                 # プロファイルを取るX列インデックス
UC1_PROFILE_AXIS  = "x"                # "x" = 列方向スライス, "y" = 行方向スライス

# %%
# --- UC1 実行 ---

img    = np.array(Image.open(UC1_PATH))[UC1_CROP[0]:UC1_CROP[1], UC1_CROP[2]:UC1_CROP[3]]
img_bg = np.array(Image.open(UC1_PATH_BG))[UC1_CROP[0]:UC1_CROP[1], UC1_CROP[2]:UC1_CROP[3]]

# npy から読む場合はこちら（コメントを外す）:
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

# 位相マップ表示
fig, ax = plt.subplots()
im = ax.imshow(angle_nobg, cmap="viridis", vmin=-0.5, vmax=0.5)
plt.colorbar(im, ax=ax, label="Phase (rad)")
ax.set_title("Background-subtracted Phase Map")
plt.tight_layout()
save_figure(fig, params={"crop": UC1_CROP, "offaxis_center": OFFAXIS_CENTER},
            description="UC1 phase map")

# %%
# プロファイルプロット
# UC1_PROFILE_AXIS == "x" のとき:
#   angle_nobg[:, x_coord] → 列 x_coord のすべての行を取り出す
#   → 横軸は行インデックス（Y方向）
if UC1_PROFILE_AXIS == "x":
    profile = angle_nobg[:, UC1_PROFILE_COORD]
    xlabel  = "Row index (Y direction, pixels)"
    title   = f"Phase profile at column (X) = {UC1_PROFILE_COORD}"
else:
    # "y" のとき: 行 y_coord のすべての列を取り出す
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
            description="UC1 phase profile")

# ============================================================
# UC2: タイムラプスノイズ追跡
# ============================================================

# %%
# --- UC2 設定 ---

UC2_TIFF_DIR    = "path/to/timelapse_dir"  # TIFFが並ぶディレクトリ
UC2_BG_PATH     = "path/to/background.tif"
UC2_CROP        = (8, 2056, 208, 2256)
UC2_ROI         = (50, 100, 50, 100)       # ノイズ計測ROI (row_start, row_end, col_start, col_end)
                                            # 背景領域（細胞なし）を選ぶ
UC2_OFFAXIS     = OFFAXIS_CENTER

# %%
# --- UC2 実行 ---

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
            description="UC2 timelapse noise tracking")

print(f"Overall noise std: {roi_std.mean():.4f} rad  ({roi_std.mean() * WAVELENGTH / (4 * np.pi) * 1e9:.2f} nm)")

# ============================================================
# UC3: シフト量とノイズの相関
# ============================================================

# %%
# --- UC3 設定 ---

UC3_TRANSFORMS_JSON = "path/to/alignment_transforms.json"
UC3_TIFF_DIR        = "path/to/timelapse_dir"   # UC2と同じでも可
UC3_BG_PATH         = "path/to/background.tif"
UC3_CROP            = (8, 2056, 208, 2256)
UC3_ROI             = (50, 100, 50, 100)
UC3_OFFAXIS         = OFFAXIS_CENTER

# %%
# --- UC3 実行 ---

with open(UC3_TRANSFORMS_JSON) as f:
    transforms = json.load(f)  # リスト or dict — キーはファイル名を想定

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

# 線形フィット
if len(shift_mags) > 1:
    coeffs = np.polyfit(shift_mags, phase_stds, 1)
    x_fit  = np.linspace(shift_mags.min(), shift_mags.max(), 100)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), "r--",
            label=f"slope={coeffs[0]:.4f}")
    ax.legend()

save_figure(fig,
            params={"n_frames": len(shift_mags), "roi": UC3_ROI,
                    "transforms_json": UC3_TRANSFORMS_JSON},
            description="UC3 shift magnitude vs phase noise scatter")
