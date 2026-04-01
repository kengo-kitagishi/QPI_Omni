"""compare_subtract_crop.py
--------------------------
grid参照に対する ph_3 フレームの ECC アライメント後の引き算を
crop_h=80（現在値）vs crop_h=160（推奨値）で比較する。

warp ロジックは ecc_channel_inspect.py の warp_translate を import して使用
（新しい warpAffine コードは書かない）。

符号規則（feedback_opencv_ecc_sign.md より）:
  ecc_align(ref, sample) → (tx, ty) = ref→sample 方向
  sample を ref に揃えるには warp_translate(sample, -tx, -ty)

Usage:
    python scripts/compare_subtract_crop.py
"""

import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
from figure_logger import save_figure
from compute_drift_online import (
    compute_backsub_offset,
    extract_rect_roi,
    to_uint8,
    ecc_align,
    _remove_outliers_mad,
)

# ---- ecc_channel_inspect.py の warp_translate を import ----
def _import_warp_translate():
    spec = importlib.util.spec_from_file_location(
        "ecc_channel_inspect", SCRIPTS_DIR / "ecc_channel_inspect.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.warp_translate

warp_translate = _import_warp_translate()

# ---- 設定 ----
DEFAULT_CONFIG   = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json")
PHASE_DIR        = Path(r"D:\AquisitionData\Kitagishi\basler_image_seq\ph_3\Pos0\output_phase")
ROIS_JSON        = Path(r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\channels\channel_rois.json")
GRID_REF_PATH    = Path(r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\img_000000000_ph_000_phase.tif")

CROP_W = 40
CROP_H_LIST = [80, 160]          # 比較する crop_h
FRAMES_TO_SHOW = [0, 25, 50, 74, 99]  # 表示するフレームインデックス

# 大パネル表示用: 単一フレームの diff をフルサイズで並べる
LARGE_PANEL_FRAME = 50           # 代表フレーム（0-99）


def get_mean_shift(grid_ref, phase_tp, channels, crop_w, crop_h, cfg):
    """全チャンネルの ECC シフトを計算してMAD除去後の平均 (tx, ty) を返す。"""
    vmin = cfg.get("ecc_vmin", -5.0)
    vmax = cfg.get("ecc_vmax",  2.0)

    tx_list, ty_list = [], []
    ref_crops_u8 = []
    for cy, cx in channels:
        crop = extract_rect_roi(grid_ref, cy, cx, crop_w, crop_h).astype(np.float32)
        offset = compute_backsub_offset(crop, cfg)
        ref_crops_u8.append(to_uint8(crop + offset, vmin, vmax))

    for ch_idx, (cy, cx) in enumerate(channels):
        crop = extract_rect_roi(phase_tp, cy, cx, crop_w, crop_h).astype(np.float32)
        offset = compute_backsub_offset(crop, cfg)
        curr_u8 = to_uint8(crop + offset, vmin, vmax)
        result = ecc_align(ref_crops_u8[ch_idx], curr_u8)
        if result is not None:
            tx, ty, _ = result
            tx_list.append(tx); ty_list.append(ty)

    if not tx_list:
        return 0.0, 0.0

    tx_arr = np.array(tx_list)
    ty_arr = np.array(ty_list)
    if len(tx_arr) >= 3:
        out_x = _remove_outliers_mad(tx_arr.tolist())
        out_y = _remove_outliers_mad(ty_arr.tolist())
        is_out = out_x | out_y
        ok = [i for i, o in enumerate(is_out) if not o] or list(range(len(tx_arr)))
        tx_arr = tx_arr[ok]; ty_arr = ty_arr[ok]

    return float(np.mean(tx_arr)), float(np.mean(ty_arr))


def channel_bbox(channels, margin=60):
    """チャンネル群を囲む矩形（y1, y2, x1, x2）を返す。"""
    cy_list = [cy for cy, cx in channels]
    cx_list = [cx for cy, cx in channels]
    return (max(0, min(cy_list) - margin),
            min(9999, max(cy_list) + margin),
            max(0, min(cx_list) - margin),
            min(9999, max(cx_list) + margin))


def main():
    cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    # ---- データ読み込み ----
    grid_ref = tifffile.imread(str(GRID_REF_PATH)).astype(np.float64)
    print(f"grid_ref: {GRID_REF_PATH.name}  shape={grid_ref.shape}")

    channels = [(r["cy"], r["cx"])
                for r in json.loads(ROIS_JSON.read_text(encoding="utf-8"))]
    n_ch = len(channels)
    print(f"Channels: {n_ch}")

    phase_paths = sorted(PHASE_DIR.glob("img_*_phase.tif"))
    print(f"ph_3 frames: {len(phase_paths)}")

    # 表示フレームをクランプ
    frame_indices = [min(fi, len(phase_paths) - 1) for fi in FRAMES_TO_SHOW]
    n_frames = len(frame_indices)

    # チャンネル領域の bounding box（クロップ表示用）
    y1_bb, y2_bb, x1_bb, x2_bb = channel_bbox(channels, margin=80)
    y2_bb = min(y2_bb, grid_ref.shape[0])
    x2_bb = min(x2_bb, grid_ref.shape[1])

    # ---- ECC + align + diff の計算 ----
    # results[fi][crop_h] = {"shift_x", "shift_y", "diff"}
    results = []
    for fi in frame_indices:
        ph = tifffile.imread(str(phase_paths[fi])).astype(np.float64)
        row = {}
        for ch in CROP_H_LIST:
            tx, ty = get_mean_shift(grid_ref, ph, channels, CROP_W, ch, cfg)
            # sample を ref に揃える: warp_translate(ph, -tx, -ty)
            ph_aligned = warp_translate(ph, -tx, -ty).astype(np.float64)
            diff = ph_aligned - grid_ref
            row[ch] = {"tx": tx, "ty": ty, "diff": diff, "aligned": ph_aligned}
            print(f"  frame={fi:3d}  crop_h={ch}  shift=({tx:+.3f}, {ty:+.3f}) px")
        results.append(row)

    # ---- 図 ----
    n_cols = 2 + len(CROP_H_LIST) * 2  # [grid_ref | ph3_raw] + [aligned_80 | diff_80 | aligned_160 | diff_160]
    fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_frames))
    gs = gridspec.GridSpec(n_frames, n_cols, hspace=0.35, wspace=0.25,
                           left=0.04, right=0.97, top=0.93, bottom=0.04)

    vmin_diff, vmax_diff = -0.3, 0.3
    vmin_raw,  vmax_raw  = -5.0,  2.0
    cmap_diff = "RdBu_r"
    cmap_raw  = "gray"

    def crop_bb(img):
        return img[y1_bb:y2_bb, x1_bb:x2_bb]

    for row_idx, (fi, row) in enumerate(zip(frame_indices, results)):
        ph_raw = tifffile.imread(str(phase_paths[fi])).astype(np.float64)

        # col 0: grid_ref
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(crop_bb(grid_ref), cmap=cmap_raw, vmin=vmin_raw, vmax=vmax_raw, aspect="equal")
        ax.set_title(f"grid_ref\n(frame {fi})", fontsize=7)
        ax.axis("off")

        # col 1: ph3 raw
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.imshow(crop_bb(ph_raw), cmap=cmap_raw, vmin=vmin_raw, vmax=vmax_raw, aspect="equal")
        ax.set_title(f"ph3 frame {fi}\n(raw)", fontsize=7)
        ax.axis("off")

        # col 2,3 / 4,5: aligned + diff for each crop_h
        for col_offset, ch in zip([2, 4], CROP_H_LIST):
            tx = row[ch]["tx"]; ty = row[ch]["ty"]

            ax_aln = fig.add_subplot(gs[row_idx, col_offset])
            ax_aln.imshow(crop_bb(row[ch]["aligned"]), cmap=cmap_raw,
                          vmin=vmin_raw, vmax=vmax_raw, aspect="equal")
            ax_aln.set_title(f"aligned (crop_h={ch})\ntx={tx:+.3f} ty={ty:+.3f} px",
                             fontsize=7)
            ax_aln.axis("off")

            ax_diff = fig.add_subplot(gs[row_idx, col_offset + 1])
            im = ax_diff.imshow(crop_bb(row[ch]["diff"]), cmap=cmap_diff,
                                vmin=vmin_diff, vmax=vmax_diff, aspect="equal")
            ax_diff.set_title(f"diff (crop_h={ch})\nph3_aligned - grid_ref",
                              fontsize=7)
            ax_diff.axis("off")
            if col_offset == 4:  # rightmost のみ colorbar
                plt.colorbar(im, ax=ax_diff, fraction=0.05, pad=0.03,
                             label="phase diff (rad)")

    fig.suptitle(
        f"Background subtraction quality: crop_h=80 vs crop_h=160  "
        f"(crop_w={CROP_W}, grid ref, ph_3)\n"
        f"Channel region crop shown  [vmin_diff={vmin_diff}, vmax_diff={vmax_diff}]",
        fontsize=10
    )

    # ---- データ保存 ----
    data = {"grid_ref_crop": crop_bb(grid_ref)}
    for fi, row in zip(frame_indices, results):
        for ch in CROP_H_LIST:
            k = f"f{fi:03d}_ch{ch}"
            data[f"diff_{k}"]    = crop_bb(row[ch]["diff"])
            data[f"aligned_{k}"] = crop_bb(row[ch]["aligned"])

    save_figure(
        fig,
        params={
            "grid_ref": str(GRID_REF_PATH),
            "crop_w": CROP_W,
            "crop_h_list": CROP_H_LIST,
            "frames": frame_indices,
            "vmin_diff": vmin_diff,
            "vmax_diff": vmax_diff,
        },
        description=(
            "ECC subtract comparison: crop_h=80 vs crop_h=160. "
            "ph3 aligned to grid Pos1 ref. Channel region crop shown."
        ),
        data=data,
    )
    plt.close(fig)

    # ---- 大パネル図: 代表フレームのフルサイズ diff ----
    lp_idx = min(LARGE_PANEL_FRAME, len(phase_paths) - 1)
    lp_row_idx = next((i for i, fi in enumerate(frame_indices) if fi == lp_idx), None)

    # LARGE_PANEL_FRAME が FRAMES_TO_SHOW になければ改めて計算
    if lp_row_idx is None:
        ph_lp = tifffile.imread(str(phase_paths[lp_idx])).astype(np.float64)
        lp_result = {}
        for ch in CROP_H_LIST:
            tx, ty = get_mean_shift(grid_ref, ph_lp, channels, CROP_W, ch, cfg)
            ph_aligned = warp_translate(ph_lp, -tx, -ty).astype(np.float64)
            lp_result[ch] = {"tx": tx, "ty": ty,
                             "diff": ph_aligned - grid_ref,
                             "aligned": ph_aligned}
    else:
        ph_lp = tifffile.imread(str(phase_paths[lp_idx])).astype(np.float64)
        lp_result = results[lp_row_idx]

    n_large = 2 + len(CROP_H_LIST)  # [grid_ref | ph3_raw | diff_80 | diff_160]
    fig2, axes = plt.subplots(1, n_large,
                              figsize=(6 * n_large, 6),
                              constrained_layout=True)

    axes[0].imshow(grid_ref, cmap=cmap_raw, vmin=vmin_raw, vmax=vmax_raw)
    axes[0].set_title("grid_ref", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(ph_lp, cmap=cmap_raw, vmin=vmin_raw, vmax=vmax_raw)
    axes[1].set_title(f"ph3 frame {lp_idx} (raw)", fontsize=10)
    axes[1].axis("off")

    for col, ch in enumerate(CROP_H_LIST, start=2):
        tx = lp_result[ch]["tx"]; ty = lp_result[ch]["ty"]
        im2 = axes[col].imshow(lp_result[ch]["diff"], cmap=cmap_diff,
                               vmin=vmin_diff, vmax=vmax_diff)
        axes[col].set_title(
            f"diff  crop_h={ch}\ntx={tx:+.3f} ty={ty:+.3f} px", fontsize=10
        )
        axes[col].axis("off")
        plt.colorbar(im2, ax=axes[col], fraction=0.046, pad=0.03,
                     label="phase diff (rad)")

    fig2.suptitle(
        f"Subtraction diff (full frame): crop_h=80 vs crop_h=160  "
        f"frame={lp_idx}  [vmin={vmin_diff}, vmax={vmax_diff}]",
        fontsize=11
    )

    data2 = {
        "grid_ref": grid_ref,
        "ph3_raw": ph_lp,
        **{f"diff_ch{ch}": lp_result[ch]["diff"] for ch in CROP_H_LIST},
    }
    save_figure(
        fig2,
        params={
            "grid_ref": str(GRID_REF_PATH),
            "crop_w": CROP_W,
            "crop_h_list": CROP_H_LIST,
            "frame": lp_idx,
            "vmin_diff": vmin_diff,
            "vmax_diff": vmax_diff,
        },
        description=(
            f"Large-panel subtraction diff: crop_h=80 vs 160, frame={lp_idx}. Full 511x511."
        ),
        data=data2,
    )
    plt.close(fig2)
    print("Done.")


if __name__ == "__main__":
    main()
