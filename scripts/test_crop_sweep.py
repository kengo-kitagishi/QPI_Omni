"""test_crop_sweep.py — ECC crop サイズの最適化（grid参照版）

保存済み位相再構成画像（ph_3/Pos0/output_phase/）を使い、
複数の crop_w × crop_h の組み合わせで ECC を再実行する。

参照画像: grid_2pergluc_60ms_1/Pos1_x+0_y+0 の第0フレーム（実際のパイプラインと同条件）。
ph_3 はドリフトなし → 各チャンネルのシフトは「grid-ph3 固定オフセット + ECC ノイズ」。

評価指標:
  temporal_std  各チャンネルの TP 間シフト std の平均 = ECC 精度（純粋ノイズ）
  mean_shift    各チャンネルの平均シフト = 系統的バイアス（波面 tilt 影響の可能性）
  ch_corr       チャンネル間シフトの Pearson 相関

Usage:
    python scripts/test_crop_sweep.py
    python scripts/test_crop_sweep.py --config path/to/drift_config.json
    python scripts/test_crop_sweep.py --max-tp 100
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure
from compute_drift_online import (
    compute_backsub_offset,
    extract_rect_roi,
    to_uint8,
    ecc_align,
    _remove_outliers_mad,
)

# ---- デフォルト設定 ----
DEFAULT_CONFIG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json")
CROP_W_LIST = [30, 40, 50]            # Y 方向（チャネル長; 40 が現在値）
CROP_H_LIST = [30, 50, 80, 120, 160]  # X 方向（チャネル幅; 80 が現在値）
CURRENT_CROP_W = 40
CURRENT_CROP_H = 80

# ph_3 → grid ref 設定
PHASE_DIR_OVERRIDE = r"D:\AquisitionData\Kitagishi\basler_image_seq\ph_3\Pos0\output_phase"
ROIS_JSON_OVERRIDE = r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"
GRID_REF_PATH      = r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\img_000000000_ph_000_phase.tif"

PH3_INTERVAL_S = 0.1  # 100 ms/frame


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--max-tp", type=int, default=None, help="使用する最大 TP 数（テスト用）")
    return p.parse_args()


def load_channels(rois_path):
    """channel_rois.json からチャネル中心座標を読む。"""
    rois = json.loads(Path(rois_path).read_text(encoding="utf-8"))
    return [(r["cy"], r["cx"]) for r in rois]


def run_ecc_one_tp(ref_crops_u8, phase_tp, channels, crop_w, crop_h, cfg):
    """1 TP 分の per-channel ECC を実行。

    Returns
    -------
    ch_shifts : dict {ch_idx: (tx_px, ty_px)} — MAD 外れ値除去済み
    corr_mean : float
    """
    vmin = cfg.get("ecc_vmin", -5.0)
    vmax = cfg.get("ecc_vmax",  2.0)

    tx_list, ty_list, corr_list, valid_ch = [], [], [], []
    for ch_idx, (cy, cx) in enumerate(channels):
        crop = extract_rect_roi(phase_tp, cy, cx, crop_w, crop_h).astype(np.float32)
        offset = compute_backsub_offset(crop, cfg)
        curr_u8 = to_uint8(crop + offset, vmin, vmax)
        result = ecc_align(ref_crops_u8[ch_idx], curr_u8)
        if result is not None:
            tx, ty, corr = result
            tx_list.append(tx); ty_list.append(ty)
            corr_list.append(corr); valid_ch.append(ch_idx)

    if not tx_list:
        return None, np.nan

    tx_arr = np.array(tx_list)
    ty_arr = np.array(ty_list)
    corr_arr = np.array(corr_list)
    n = len(tx_arr)

    if n >= 3:
        out_x = _remove_outliers_mad(tx_arr.tolist())
        out_y = _remove_outliers_mad(ty_arr.tolist())
        is_out = out_x | out_y
        ok = [i for i, o in enumerate(is_out) if not o] or list(range(n))
    else:
        ok = list(range(n))

    ch_shifts = {valid_ch[i]: (float(tx_arr[i]), float(ty_arr[i])) for i in ok}
    return ch_shifts, float(np.mean(corr_arr[ok]))


def sweep_one_crop(crop_w, crop_h, phase_paths, tp_indices, channels, cfg):
    """(crop_w, crop_h) 1 組分の全 TP ECC を実行。"""
    vmin = cfg.get("ecc_vmin", -5.0)
    vmax = cfg.get("ecc_vmax",  2.0)
    pixel_scale = cfg.get("pixel_scale_um", 0.3462)

    # 参照: grid Pos1 の位相画像（実際のパイプラインと同じ）
    phase_ref = tifffile.imread(GRID_REF_PATH).astype(np.float32)
    ref_crops_u8 = []
    for cy, cx in channels:
        crop = extract_rect_roi(phase_ref, cy, cx, crop_w, crop_h).astype(np.float32)
        offset = compute_backsub_offset(crop, cfg)
        ref_crops_u8.append(to_uint8(crop + offset, vmin, vmax))

    n_ch = len(channels)
    shifts_x = [[] for _ in range(n_ch)]  # image X（列）方向 um
    shifts_y = [[] for _ in range(n_ch)]  # image Y（行）方向 um
    corr_series = []
    valid_tps = []

    for i, path in enumerate(phase_paths):
        try:
            phase_tp = tifffile.imread(str(path)).astype(np.float32)
        except Exception:
            continue
        ch_shifts, corr_mean = run_ecc_one_tp(
            ref_crops_u8, phase_tp, channels, crop_w, crop_h, cfg
        )
        if ch_shifts is None:
            continue
        for ch_idx, (tx, ty) in ch_shifts.items():
            shifts_x[ch_idx].append(tx * pixel_scale)
            shifts_y[ch_idx].append(ty * pixel_scale)
        corr_series.append(corr_mean)
        valid_tps.append(tp_indices[i] if i < len(tp_indices) else i)

    # ECC 精度 = 各チャンネルの temporal std → 平均
    ch_stds_x = [np.std(s) for s in shifts_x if len(s) > 1]
    ch_stds_y = [np.std(s) for s in shifts_y if len(s) > 1]
    temporal_std_x = float(np.mean(ch_stds_x)) if ch_stds_x else np.nan
    temporal_std_y = float(np.mean(ch_stds_y)) if ch_stds_y else np.nan

    # 系統的バイアス = 各チャンネルの平均シフト
    mean_shift_x = [float(np.mean(s)) if len(s) > 0 else np.nan for s in shifts_x]
    mean_shift_y = [float(np.mean(s)) if len(s) > 0 else np.nan for s in shifts_y]

    return {
        "crop_w": crop_w, "crop_h": crop_h,
        "tps": np.array(valid_tps),
        "shifts_x": shifts_x,
        "shifts_y": shifts_y,
        "mean_shift_x": mean_shift_x,
        "mean_shift_y": mean_shift_y,
        "temporal_std_x": temporal_std_x,
        "temporal_std_y": temporal_std_y,
        "corr_mean": float(np.mean(corr_series)) if corr_series else np.nan,
    }


def _corr_matrix(shifts_list):
    """shifts_list: per-channel list of lists。Pearson 相関行列を返す。"""
    valid = [s for s in shifts_list if len(s) > 1]
    n = len(shifts_list)
    if len(valid) < 2:
        return np.full((n, n), np.nan)
    min_len = min(len(s) for s in valid)
    arr_full = np.full((n, min_len), np.nan)
    for i, s in enumerate(shifts_list):
        if len(s) >= min_len:
            arr_full[i] = s[:min_len]
    mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            xi, xj = arr_full[i], arr_full[j]
            if not (np.any(np.isnan(xi)) or np.any(np.isnan(xj))):
                si, sj = np.std(xi), np.std(xj)
                if si > 0 and sj > 0:
                    mat[i, j] = float(np.corrcoef(xi, xj)[0, 1])
                else:
                    mat[i, j] = 1.0 if i == j else 0.0
    return mat


def _heatmap(ax, results, key, title, crop_w_list, crop_h_list, cmap="RdYlGn_r", invert=True):
    matrix = np.full((len(crop_w_list), len(crop_h_list)), np.nan)
    for r in results:
        wi = crop_w_list.index(r["crop_w"])
        hi = crop_h_list.index(r["crop_h"])
        val = r[key] if key != "corr2_mean" else r["corr_mean"]
        matrix[wi, hi] = val

    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   extent=[-0.5, len(crop_h_list) - 0.5,
                            len(crop_w_list) - 0.5, -0.5])
    ax.set_xticks(range(len(crop_h_list))); ax.set_xticklabels(crop_h_list)
    ax.set_yticks(range(len(crop_w_list))); ax.set_yticklabels(crop_w_list)
    ax.set_xlabel("crop_h (X-dir, px)")
    ax.set_ylabel("crop_w (Y-dir, px)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)

    for wi in range(len(crop_w_list)):
        for hi in range(len(crop_h_list)):
            v = matrix[wi, hi]
            if not np.isnan(v):
                ax.text(hi, wi, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if invert else "black")

    if CURRENT_CROP_W in crop_w_list and CURRENT_CROP_H in crop_h_list:
        wi = crop_w_list.index(CURRENT_CROP_W)
        hi = crop_h_list.index(CURRENT_CROP_H)
        ax.add_patch(plt.Rectangle((hi - 0.5, wi - 0.5), 1, 1,
                                    fill=False, edgecolor="white", lw=2.5))


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    phase_dir = Path(PHASE_DIR_OVERRIDE) if PHASE_DIR_OVERRIDE else Path(cfg["save_dir"]) / "Pos0" / "output_phase"
    rois_path = ROIS_JSON_OVERRIDE if ROIS_JSON_OVERRIDE else cfg["channel_rois_json"]

    phase_paths = sorted(phase_dir.glob("img_*_phase.tif"))
    if not phase_paths:
        print(f"ERROR: 位相画像が見つかりません: {phase_dir}"); sys.exit(1)
    print(f"Phase images: {len(phase_paths)}  ({phase_paths[0].name} .. {phase_paths[-1].name})")
    print(f"Grid ref: {GRID_REF_PATH}")

    if args.max_tp is not None:
        phase_paths = phase_paths[:args.max_tp]
        print(f"  -> max_tp={args.max_tp} で {len(phase_paths)} TP に制限")

    channels = load_channels(rois_path)
    n_ch = len(channels)
    print(f"Channels: {n_ch}")

    def tp_from_path(p):
        return int(p.stem.split("_")[1])
    tp_indices = [tp_from_path(p) for p in phase_paths]

    # ---- Sweep 実行 ----
    sweep_params = list(product(CROP_W_LIST, CROP_H_LIST))
    print(f"\nSweep: {len(sweep_params)} 組み合わせ × {len(phase_paths)} TP × {n_ch} ch")

    results = []
    for i, (cw, ch) in enumerate(sweep_params):
        print(f"  [{i+1}/{len(sweep_params)}] crop_w={cw} crop_h={ch} ...", end=" ", flush=True)
        res = sweep_one_crop(cw, ch, phase_paths, tp_indices, channels, cfg)
        results.append(res)
        print(f"temporal_std_x={res['temporal_std_x']:.4f}  "
              f"temporal_std_y={res['temporal_std_y']:.4f}  "
              f"corr2={res['corr_mean']:.4f}")

    # ---- サマリー ----
    print("\n=== Temporal ECC Precision (um) -- lower is better ===")
    print(f"{'crop_w':>7} {'crop_h':>7}  {'tmp_std_x':>10}  {'tmp_std_y':>10}  {'corr2':>8}")
    for r in sorted(results, key=lambda r: r["temporal_std_x"] + r["temporal_std_y"]):
        print(f"{r['crop_w']:>7} {r['crop_h']:>7}  "
              f"{r['temporal_std_x']:>10.4f}  {r['temporal_std_y']:>10.4f}  "
              f"{r['corr_mean']:>8.4f}")

    best = min(results, key=lambda r: r["temporal_std_x"] + r["temporal_std_y"])
    current = next(
        (r for r in results if r["crop_w"] == CURRENT_CROP_W and r["crop_h"] == CURRENT_CROP_H), None
    )
    print(f"\nBest: crop_w={best['crop_w']}  crop_h={best['crop_h']}")

    # ---- 図（3 行 × 3 列）----
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.55, wspace=0.45,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)

    # ---- Row 0: Heatmaps ----
    ax_hx = fig.add_subplot(gs[0, 0])
    _heatmap(ax_hx, results, "temporal_std_x", "Temporal ECC Std X (um)", CROP_W_LIST, CROP_H_LIST)

    ax_hy = fig.add_subplot(gs[0, 1])
    _heatmap(ax_hy, results, "temporal_std_y", "Temporal ECC Std Y (um)", CROP_W_LIST, CROP_H_LIST)

    ax_hc = fig.add_subplot(gs[0, 2])
    _heatmap(ax_hc, results, "corr2_mean", "Mean corr2",
             CROP_W_LIST, CROP_H_LIST, cmap="RdYlGn", invert=False)

    # ---- Row 1: Per-channel time series (best crop) ----
    cmap_ch = plt.cm.tab20
    t_s = np.arange(len(best["tps"])) * PH3_INTERVAL_S  # 秒

    for col, (axis_label, shifts_key, mean_key) in enumerate([
        ("X (col, um)", "shifts_x", "mean_shift_x"),
        ("Y (row, um)", "shifts_y", "mean_shift_y"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        shifts_data = best[shifts_key]
        mean_data   = best[mean_key]
        for ch_idx in range(n_ch):
            s = shifts_data[ch_idx]
            if len(s) < 2:
                continue
            color = cmap_ch(ch_idx / max(n_ch - 1, 1))
            t = t_s[:len(s)]
            ax.plot(t, s, color=color, lw=0.6, alpha=0.6)
            ax.axhline(mean_data[ch_idx], color=color, lw=1.0, ls="--", alpha=0.9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"ECC shift {axis_label}")
        ax.set_title(f"Per-channel shift {axis_label.split()[0]}\n"
                     f"best {best['crop_w']}×{best['crop_h']}  "
                     f"solid=series, dashed=mean (bias)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Row 1 col 2: mean shift scatter（X vs Y、1 点/channel、cy で色分け）
    ax_sc = fig.add_subplot(gs[1, 2])
    cy_vals = np.array([cy for cy, cx in channels])
    mx = np.array(best["mean_shift_x"])
    my = np.array(best["mean_shift_y"])
    valid = ~(np.isnan(mx) | np.isnan(my))
    if valid.sum() > 0:
        sc = ax_sc.scatter(mx[valid], my[valid], c=cy_vals[valid],
                           cmap="viridis", s=30, alpha=0.85, zorder=3)
        plt.colorbar(sc, ax=ax_sc, label="channel cy (px)")
    ax_sc.axhline(0, color="k", lw=0.5, ls="--")
    ax_sc.axvline(0, color="k", lw=0.5, ls="--")
    ax_sc.set_xlabel("Mean shift X (um)")
    ax_sc.set_ylabel("Mean shift Y (um)")
    ax_sc.set_title(f"Per-channel bias (mean shift)\n"
                    f"best {best['crop_w']}×{best['crop_h']}  color=cy pos")
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)

    # ---- Row 2: Channel correlation matrices (X and Y) ----
    tick_step = max(1, n_ch // 10)
    ticks = list(range(0, n_ch, tick_step))

    for col, (axis_label, shifts_key) in enumerate([("X", "shifts_x"), ("Y", "shifts_y")]):
        ax_cm = fig.add_subplot(gs[2, col])
        corr_mat = _corr_matrix(best[shifts_key])
        im2 = ax_cm.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im2, ax=ax_cm, shrink=0.8)
        ax_cm.set_title(f"Channel shift Pearson corr {axis_label}\n"
                        f"best {best['crop_w']}×{best['crop_h']}")
        ax_cm.set_xlabel("Channel index")
        ax_cm.set_ylabel("Channel index")
        ax_cm.set_xticks(ticks); ax_cm.set_xticklabels(ticks, fontsize=6)
        ax_cm.set_yticks(ticks); ax_cm.set_yticklabels(ticks, fontsize=6)

    # Row 2 col 2: Summary table
    ax_t = fig.add_subplot(gs[2, 2])
    ax_t.axis("off")
    lines_txt = ["crop_w  crop_h  tmp_sx  tmp_sy  corr2"]
    for r in sorted(results, key=lambda r: (r["temporal_std_x"] + r["temporal_std_y"]) / 2)[:8]:
        marker = " <--best" if (r["crop_w"] == best["crop_w"] and r["crop_h"] == best["crop_h"]) else ""
        curr_m = " [cur]"  if (r["crop_w"] == CURRENT_CROP_W and r["crop_h"] == CURRENT_CROP_H) else ""
        lines_txt.append(f"{r['crop_w']:>6}  {r['crop_h']:>6}  "
                         f"{r['temporal_std_x']:.3f}  {r['temporal_std_y']:.3f}  "
                         f"{r['corr_mean']:.3f}{marker}{curr_m}")
    ax_t.text(0.02, 0.95, "\n".join(lines_txt), va="top", ha="left",
              fontsize=7, family="monospace", transform=ax_t.transAxes)

    fig.suptitle("ECC Crop Sweep: Temporal Precision + Channel Bias (grid ref)", fontsize=12)

    # ---- 数値データ保存 ----
    data = {}
    for r in results:
        k = f"w{r['crop_w']}h{r['crop_h']}"
        data[f"tps_{k}"] = r["tps"]
        for ch_idx in range(n_ch):
            if r["shifts_x"][ch_idx]:
                data[f"sx_ch{ch_idx}_{k}"] = np.array(r["shifts_x"][ch_idx])
                data[f"sy_ch{ch_idx}_{k}"] = np.array(r["shifts_y"][ch_idx])
    data["channels_cy"] = np.array([cy for cy, cx in channels])
    data["channels_cx"] = np.array([cx for cy, cx in channels])
    data["best_mean_shift_x"] = np.array(best["mean_shift_x"])
    data["best_mean_shift_y"] = np.array(best["mean_shift_y"])
    data["best_corr_matrix_x"] = _corr_matrix(best["shifts_x"])
    data["best_corr_matrix_y"] = _corr_matrix(best["shifts_y"])

    save_figure(
        fig,
        params={
            "n_phase_images": len(phase_paths),
            "n_channels": n_ch,
            "grid_ref": GRID_REF_PATH,
            "ph3_interval_s": PH3_INTERVAL_S,
            "crop_w_list": CROP_W_LIST,
            "crop_h_list": CROP_H_LIST,
            "best_crop_w": best["crop_w"],
            "best_crop_h": best["crop_h"],
            "best_temporal_std_x": best["temporal_std_x"],
            "best_temporal_std_y": best["temporal_std_y"],
            **{f"tmp_std_x_w{r['crop_w']}h{r['crop_h']}": r["temporal_std_x"] for r in results},
            **{f"tmp_std_y_w{r['crop_w']}h{r['crop_h']}": r["temporal_std_y"] for r in results},
        },
        description=(
            f"ECC crop sweep (grid ref): temporal precision + per-channel bias + channel corr. "
            f"Best: {best['crop_w']}x{best['crop_h']}"
        ),
        data=data,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
