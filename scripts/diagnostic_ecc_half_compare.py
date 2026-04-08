# %%
"""
diagnostic_ecc_half_compare.py
-------------------------------
Pos1 の channel_02_bg_corr を例に、
  RIGHT half（現行・列後半）vs LEFT half（未使用・列前半）
の ECC を比較し、それぞれのシフトを full crop に適用して
grid を引いた結果を並べて可視化する。

列レイアウト（5列 × N_FRAMES 行）:
  [0] frame (raw)
  [1] grid ref (full)
  [2] RIGHT half ECC → subtracted
  [3] LEFT half ECC, 固定スケール → subtracted
  [4] LEFT half ECC, 自動スケール → subtracted
"""
import numpy as np
import tifffile
import cv2
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent))
from channel_crop import extract_rect_roi
from figure_logger import save_figure

# ============================================================
TL_CHANNEL = r"D:\AquisitionData\Kitagishi\260310\timelapse_11day_exp200ms_1pos_EMM2\Pos1\output_phase\channels\channel_02_bg_corr.tif"
GRID_DIR   = r"D:\AquisitionData\Kitagishi\260310\grid_0p5_0p5_0p1_exp200ms_1pos_EMM2_1"
GRID_LABEL = "Pos1"
GRID_Z     = 2
ROIS_JSON  = r"D:\AquisitionData\Kitagishi\260310\timelapse_11day_exp200ms_1pos_EMM2\Pos1\output_phase\channels\channel_rois.json"
CH_INDEX   = 2        # channel_02

# ECC 正規化（固定スケール）
VMIN_ECC = -5.0
VMAX_ECC =  2.0

# backsub パラメータ（grid ref に適用）
BACKSUB_HIST_MIN    = -1.1
BACKSUB_HIST_MAX    =  1.5
BACKSUB_N_BINS      = 512
BACKSUB_SMOOTH_WIN  = 20
BACKSUB_MIN_PHASE   = -1.1

# 表示設定
N_FRAMES   = 5
FRAME_STEP = 40       # 0, 40, 80, 120, 160 フレーム
DISP_VMIN  = -0.5
DISP_VMAX  =  1.5
# ============================================================


def to_uint8(img, vmin, vmax):
    return ((np.clip(img, vmin, vmax) - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def ecc_align(ref_u8, tl_u8):
    """(tx, ty, corr) を返す。失敗時 None。"""
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-8)
    try:
        corr, warp = cv2.findTransformECC(ref_u8, tl_u8, warp, cv2.MOTION_TRANSLATION, criteria)
        return float(warp[0, 2]), float(warp[1, 2]), float(corr)
    except Exception:
        return None


def apply_shift_and_subtract(frame, grid_full, tx, ty):
    """
    frame を (-tx, -ty) だけ平行移動してグリッド座標に合わせ、grid_full を引く。
    warpAffine の変換行列: W = [[1,0,-tx],[0,1,-ty]]
    """
    h, w = frame.shape
    M = np.float32([[1, 0, -tx], [0, 1, -ty]])
    aligned = cv2.warpAffine(frame.astype(np.float32), M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    return aligned - grid_full.astype(np.float32)


def compute_backsub_offset(img):
    """gaussian_backsub と同方式でバックグラウンドピークを推定し -peak を返す。"""
    bin_edges = np.linspace(BACKSUB_HIST_MIN, BACKSUB_HIST_MAX, BACKSUB_N_BINS + 1)
    hist, _ = np.histogram(img.flatten(), bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    smoothed = uniform_filter1d(hist.astype(float), size=BACKSUB_SMOOTH_WIN, mode='nearest')
    smoothed = uniform_filter1d(smoothed, size=BACKSUB_SMOOTH_WIN, mode='nearest')
    valid = np.where(centers >= BACKSUB_MIN_PHASE)[0]
    search = valid[valid < int(len(centers) * 0.95)]
    if len(search) == 0:
        return 0.0
    peak_idx = search[np.argmax(smoothed[search])]
    peak_val = centers[peak_idx]
    fw = 300
    s, e = max(0, peak_idx - fw), min(len(centers), peak_idx + fw)
    def gaussian(x, a, m, sig):
        return a * np.exp(-((x - m) ** 2) / (2 * sig ** 2))
    try:
        popt, _ = curve_fit(gaussian, centers[s:e], smoothed[s:e],
                            p0=[float(np.max(smoothed[s:e])), peak_val, 0.05], maxfev=5000)
        return float(-popt[1])
    except Exception:
        return float(-peak_val)


def main():
    # ROI 読み込み
    with open(ROIS_JSON, encoding='utf-8') as f:
        rois = json.load(f)
    roi = rois[CH_INDEX]
    cy, cx = roi['cy'], roi['cx']
    crop_w, crop_h = roi['crop_w'], roi['crop_h']
    crop_h_half = crop_h // 2          # 220
    cx_right    = cx + crop_h // 4    # 右半分の中心
    cx_left     = cx - crop_h // 4    # 左半分の中心
    print(f"ROI ch{CH_INDEX}: cy={cy}, cx={cx}, crop_w={crop_w}, crop_h={crop_h}")
    print(f"  RIGHT half: cx={cx_right}, width={crop_h_half}  (cols [{crop_h_half}:{crop_h}])")
    print(f"  LEFT  half: cx={cx_left},  width={crop_h_half}  (cols [0:{crop_h_half}])")

    # グリッド基準画像
    grid_path = (Path(GRID_DIR) / f"{GRID_LABEL}_x+0_y+0" / "output_phase"
                 / f"img_000000000_ph_{GRID_Z:03d}_phase.tif")
    grid_img  = tifffile.imread(str(grid_path)).astype(np.float64)

    grid_full  = extract_rect_roi(grid_img, cy, cx,       crop_w, crop_h)
    grid_right = extract_rect_roi(grid_img, cy, cx_right, crop_w, crop_h_half)
    grid_left  = extract_rect_roi(grid_img, cy, cx_left,  crop_w, crop_h_half)

    # grid ref に backsub 適用
    grid_full  = grid_full  + compute_backsub_offset(grid_full)
    grid_right = grid_right + compute_backsub_offset(grid_right)
    grid_left  = grid_left  + compute_backsub_offset(grid_left)

    # auto scale for left half（grid の left half の実データ分布から決める）
    left_vmin = float(np.percentile(grid_left, 1))
    left_vmax = float(np.percentile(grid_left, 99))
    print(f"  LEFT half auto scale: vmin={left_vmin:.3f}  vmax={left_vmax:.3f}")
    print(f"  Fixed scale:          vmin={VMIN_ECC:.3f}  vmax={VMAX_ECC:.3f}")

    # uint8 変換（grid ref）
    g_right_u8       = to_uint8(grid_right, VMIN_ECC, VMAX_ECC)
    g_left_fixed_u8  = to_uint8(grid_left,  -1.0, 1.0)
    g_left_auto_u8   = to_uint8(grid_left,  left_vmin, left_vmax)

    # タイムラプス stack
    stack = tifffile.imread(str(TL_CHANNEL)).astype(np.float64)
    if stack.ndim == 2:
        stack = stack[np.newaxis]
    frame_indices = [i * FRAME_STEP for i in range(N_FRAMES) if i * FRAME_STEP < stack.shape[0]]
    print(f"フレーム数: {stack.shape[0]}, 表示: {frame_indices}")

    # ── 描画 ──────────────────────────────────────────────────
    n_rows = len(frame_indices)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.2),
                              constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        'frame (raw)',
        'grid ref\n(full)',
        'RIGHT half ECC\n→ subtracted',
        'LEFT half ECC\n[-1, 1] fixed → subtracted',
        'LEFT half ECC\nauto scale → subtracted',
    ]
    for j, ttl in enumerate(col_titles):
        axes[0, j].set_title(ttl, fontsize=7, fontweight='bold')

    # Pre-compute ECC results for all frames in parallel
    def _ecc_for_frame(t):
        frame = stack[t]
        f_right = frame[:, crop_h_half:]
        f_left  = frame[:, :crop_h_half]
        res_r = ecc_align(g_right_u8, to_uint8(f_right, VMIN_ECC, VMAX_ECC))
        res_l_fix = ecc_align(g_left_fixed_u8, to_uint8(f_left, -1.0, 1.0))
        fl_vmin = float(np.percentile(f_left, 1))
        fl_vmax = float(np.percentile(f_left, 99))
        res_l_auto = ecc_align(g_left_auto_u8, to_uint8(f_left, fl_vmin, fl_vmax))
        return res_r, res_l_fix, res_l_auto, fl_vmin, fl_vmax

    with ThreadPoolExecutor(max_workers=None) as pool:
        ecc_results = list(pool.map(_ecc_for_frame, frame_indices))

    for row, t in enumerate(frame_indices):
        frame = stack[t]  # shape (40, 440)
        f_left  = frame[:, :crop_h_half]

        res_r, res_l_fix, res_l_auto, fl_vmin, fl_vmax = ecc_results[row]

        # [0] frame raw
        ax = axes[row, 0]
        ax.imshow(frame, vmin=DISP_VMIN, vmax=DISP_VMAX, cmap='gray', aspect='auto')
        ax.set_ylabel(f't={t}', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

        # [1] grid ref
        ax = axes[row, 1]
        ax.imshow(grid_full, vmin=DISP_VMIN, vmax=DISP_VMAX, cmap='gray', aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])

        # [2] RIGHT subtracted
        ax = axes[row, 2]
        if res_r:
            tx, ty, corr = res_r
            sub = apply_shift_and_subtract(frame, grid_full, tx, ty)
            ax.imshow(sub, vmin=DISP_VMIN, vmax=DISP_VMAX, cmap='gray', aspect='auto')
            ax.set_xlabel(f'tx={tx:.2f} ty={ty:.2f}  corr={corr:.4f}', fontsize=6)
        else:
            ax.text(0.5, 0.5, 'ECC failed', transform=ax.transAxes, ha='center', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        # [3] LEFT fixed subtracted
        ax = axes[row, 3]
        if res_l_fix:
            tx, ty, corr = res_l_fix
            sub = apply_shift_and_subtract(frame, grid_full, tx, ty)
            ax.imshow(sub, vmin=DISP_VMIN, vmax=DISP_VMAX, cmap='gray', aspect='auto')
            ax.set_xlabel(f'tx={tx:.2f} ty={ty:.2f}  corr={corr:.4f}', fontsize=6)
        else:
            ax.text(0.5, 0.5, 'ECC failed', transform=ax.transAxes, ha='center', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        # [4] LEFT auto subtracted
        ax = axes[row, 4]
        if res_l_auto:
            tx, ty, corr = res_l_auto
            sub = apply_shift_and_subtract(frame, grid_full, tx, ty)
            ax.imshow(sub, vmin=DISP_VMIN, vmax=DISP_VMAX, cmap='gray', aspect='auto')
            ax.set_xlabel(f'tx={tx:.2f} ty={ty:.2f}  corr={corr:.4f} [auto vmin={fl_vmin:.2f}]',
                          fontsize=6)
        else:
            ax.text(0.5, 0.5, 'ECC failed', transform=ax.transAxes, ha='center', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f'ECC half compare  ch{CH_INDEX:02d}  '
        f'crop_h={crop_h} → half={crop_h_half}  '
        f'RIGHT cols [{crop_h_half}:{crop_h}] / LEFT cols [0:{crop_h_half}]',
        fontsize=8)

    save_figure(fig,
                params={
                    "channel": CH_INDEX,
                    "crop_h": crop_h, "crop_h_half": crop_h_half,
                    "vmin_ecc": VMIN_ECC, "vmax_ecc": VMAX_ECC,
                    "left_vmin_auto": left_vmin, "left_vmax_auto": left_vmax,
                    "frame_indices": frame_indices,
                },
                description="RIGHT vs LEFT half ECC comparison: shift applied to full crop, grid subtracted")
    plt.close(fig)
    print("完了")


if __name__ == "__main__":
    main()

# %%
