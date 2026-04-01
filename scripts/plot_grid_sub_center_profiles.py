"""
plot_grid_sub_center_profiles.py
---------------------------------
各チャネル crop に ECC アライメントを適用してから grid_ref_crops を pixel-wise に引き、
中心横線プロファイル（Y=center_y の 1 行）を複数 TP 重ねて表示する。

処理:
  1. 各 TP の位相画像からチャネル crop (crop_w × crop_h) を取得
  2. backsub offset を引いて float 正規化
  3. to_uint8 → ecc_align で (tx, ty) を取得
  4. float crop に (-tx, -ty) の warp を適用（sample を ref 座標に揃える）
  5. grid_ref_crop を pixel-wise に引く
  6. 中心行プロファイル（Y=center_y, 長さ crop_h）を取得
  7. 全チャネルを 4 列で並べてプロット（TP ごとに色分け）
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from compute_drift_online import (
    compute_backsub_offset,
    extract_rect_roi,
    to_uint8,
    ecc_align,
)
from figure_logger import save_figure

# ============================================================
# 設定
# ============================================================
PHASE_DIR = Path(r"D:\AquisitionData\Kitagishi\basler_image_seq\ph_3\Pos0\output_phase")
ROIS_JSON = Path(
    r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1"
    r"\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"
)
GRID_REF_CROPS_TIF = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\grid_ref_crops.tif")

ECC_VMIN = -5.0
ECC_VMAX  =  2.0

# 表示する TP インデックス（0-origin）
PICK_TPS = [0, 4, 9, 14, 19]
# ============================================================


def main():
    phase_paths = sorted(PHASE_DIR.glob("img_*_phase.tif"))
    if not phase_paths:
        print(f"ERROR: 位相画像が見つかりません: {PHASE_DIR}")
        sys.exit(1)

    rois       = json.loads(ROIS_JSON.read_text(encoding="utf-8"))
    n_ch       = len(rois)
    grid_ref_f = tifffile.imread(str(GRID_REF_CROPS_TIF)).astype(np.float32)
    print(f"Channels: {n_ch}, Phase imgs: {len(phase_paths)}, grid_ref shape: {grid_ref_f.shape}")

    # grid_ref の backsub 済み float と uint8 を事前に作る
    cfg_dummy = {}
    ref_u8_list   = []
    ref_bs_list   = []
    for c in range(n_ch):
        ref_crop = grid_ref_f[c]                               # (crop_w, crop_h)
        off      = compute_backsub_offset(ref_crop, cfg_dummy)
        ref_bs   = ref_crop + off                              # backsub 済み float
        ref_u8   = to_uint8(ref_bs, ECC_VMIN, ECC_VMAX)
        ref_u8_list.append(ref_u8)
        ref_bs_list.append(ref_bs)

    # TP インデックスをクランプ
    max_tp   = len(phase_paths) - 2   # 0-origin の最大 TP index
    pick_tps = [min(t, max_tp) for t in PICK_TPS]
    n_show   = len(pick_tps)
    colors   = plt.cm.viridis(np.linspace(0.1, 0.9, n_show))

    # profiles[i_show][c_idx] = center-line (crop_h 点)
    profiles   = [[None] * n_ch for _ in range(n_show)]
    tx_log     = np.full((n_show, n_ch), np.nan)
    ty_log     = np.full((n_show, n_ch), np.nan)

    for i_show, tp_idx in enumerate(pick_tps):
        path     = phase_paths[tp_idx + 1]   # +1: インデックス 0 は参照フレーム
        phase_tp = tifffile.imread(str(path)).astype(np.float32)
        print(f"  TP={tp_idx}: {path.name}")

        for c_idx, roi in enumerate(rois):
            crop_w = roi["crop_w"]
            crop_h = roi["crop_h"]
            cy, cx = roi["cy"], roi["cx"]

            # 1. crop 取得 & backsub
            crop = extract_rect_roi(phase_tp, cy, cx, crop_w, crop_h).astype(np.float32)
            off  = compute_backsub_offset(crop, cfg_dummy)
            crop_bs = crop + off

            # 2. ECC で (tx, ty) を取得
            result = ecc_align(ref_u8_list[c_idx],
                               to_uint8(crop_bs, ECC_VMIN, ECC_VMAX))
            if result is None:
                print(f"    ch{c_idx}: ECC failed, skipping")
                continue
            tx, ty = result[0], result[1]
            tx_log[i_show, c_idx] = tx
            ty_log[i_show, c_idx] = ty

            # 3. float crop に (-tx, -ty) warp を適用（sample を ref に揃える）
            M = np.float32([[1, 0, -tx], [0, 1, -ty]])
            aligned = cv2.warpAffine(crop_bs, M, (crop_h, crop_w),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

            # 4. grid_ref を pixel-wise に引く
            diff = aligned - ref_bs_list[c_idx]

            # 4b. slope+intercept 補正: 左 1/3（背景領域）で線形フィット → 引く
            x_coords = np.arange(diff.shape[1], dtype=np.float32)
            x_prof   = diff.mean(axis=0)
            fit_n    = max(1, diff.shape[1] // 3)
            slope, intercept = np.polyfit(x_coords[:fit_n], x_prof[:fit_n], 1)
            diff = diff - (slope * x_coords + intercept)[np.newaxis, :]

            # 5. 中心行プロファイル
            center_y = crop_w // 2
            profiles[i_show][c_idx] = diff[center_y, :]   # (crop_h,) = 80点

    print(f"\ntx range: {np.nanmin(tx_log):.3f} to {np.nanmax(tx_log):.3f} px")
    print(f"ty range: {np.nanmin(ty_log):.3f} to {np.nanmax(ty_log):.3f} px")

    # ---- プロット ----
    n_cols = 4
    n_rows = (n_ch + n_cols - 1) // n_cols
    crop_h_plot = rois[0]["crop_h"]    # 80
    x_axis = np.arange(crop_h_plot)

    fig = plt.figure(figsize=(5 * n_cols, 3.5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, hspace=0.55, wspace=0.35,
                            left=0.06, right=0.98, top=0.93, bottom=0.06)

    for c_idx, roi in enumerate(rois):
        row, col = divmod(c_idx, n_cols)
        ax = fig.add_subplot(gs[row, col])

        for i_show, (tp_idx, color) in enumerate(zip(pick_tps, colors)):
            prof = profiles[i_show][c_idx]
            if prof is None:
                continue
            tx_v = tx_log[i_show, c_idx]
            ax.plot(x_axis, prof, color=color, lw=0.9, alpha=0.85,
                    label=f"TP={tp_idx} (tx={tx_v:.2f})")

        ax.axhline(0, color="#bbb", lw=0.5, ls="--")
        ax.set_title(f"ch{c_idx}", fontsize=9)
        ax.set_xlabel("X [px]", fontsize=7)
        ax.set_ylabel("Phase diff (rad)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if c_idx == 0:
            ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(
        "Grid-subtracted center-line profiles  (ECC aligned → grid_ref subtracted)\n"
        f"crop: {rois[0]['crop_w']}×{crop_h_plot} px  |  TPs: {pick_tps}",
        fontsize=10,
    )

    all_profiles = np.array([
        [profiles[i][c] if profiles[i][c] is not None else np.full(crop_h_plot, np.nan)
         for c in range(n_ch)]
        for i in range(n_show)
    ])   # (n_show, n_ch, crop_h)

    save_figure(
        fig,
        params={
            "pick_tps": pick_tps,
            "n_ch": n_ch,
            "crop_h": crop_h_plot,
        },
        description=(
            f"Grid-subtracted center-line profiles after ECC alignment: "
            f"{n_ch} channels, TPs={pick_tps}"
        ),
        data={
            "x": x_axis,
            "profiles": all_profiles,
            "pick_tps": np.array(pick_tps),
            "tx_log": tx_log,
            "ty_log": ty_log,
        },
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
