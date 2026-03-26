"""test_crop_sweep.py — ECC crop サイズの最適化

保存済み位相再構成画像（C:/ph/Pos1/output_phase/）を使い、
複数の crop_w × crop_h の組み合わせで ECC を再実行し、
drift_log.json の SavGol スムージング趨勢（近似 ground truth）と比較する。

評価指標: residual_std = std(cumulative_sweep - savgol_ground_truth)
これが最小の crop サイズが最良。

Usage:
    python scripts/test_crop_sweep.py
    python scripts/test_crop_sweep.py --config path/to/drift_config.json
    python scripts/test_crop_sweep.py --max-tp 100   # テスト用に TP 数を制限
"""

import argparse
import json
import sys
import concurrent.futures
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tifffile
from scipy.signal import savgol_filter

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
CROP_W_LIST = [30, 40, 50]            # Y方向（チャネル長; 40が現在値）
CROP_H_LIST = [30, 50, 80, 120, 160]  # X方向（チャネル幅; 80が現在値）
SAVGOL_WINDOW = 51                     # ~4 時間窓（奇数）
SAVGOL_ORDER  = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--max-tp", type=int, default=None, help="使用する最大TP数（テスト用）")
    return p.parse_args()


def load_ground_truth(log_path, max_tp=None):
    """drift_log.json から ground truth 累積ドリフトを読み込み SavGol スムージング。"""
    records = json.loads(Path(log_path).read_text(encoding="utf-8"))
    records = [r for r in records if "cumulative_dx_um" in r and "timepoint" in r]
    records.sort(key=lambda r: r["timepoint"])
    if max_tp is not None:
        records = [r for r in records if r["timepoint"] <= max_tp]

    tps     = np.array([r["timepoint"] for r in records])
    cum_dx  = np.array([r["cumulative_dx_um"] for r in records])
    cum_dy  = np.array([r["cumulative_dy_um"] for r in records])

    wl = min(SAVGOL_WINDOW, len(cum_dx) - 1)
    if wl % 2 == 0:
        wl -= 1
    gt_x = savgol_filter(cum_dx, wl, SAVGOL_ORDER)
    gt_y = savgol_filter(cum_dy, wl, SAVGOL_ORDER)

    return tps, cum_dx, cum_dy, gt_x, gt_y


def load_channels(rois_path):
    """channel_rois.json からチャネル中心座標を読む。"""
    rois = json.loads(Path(rois_path).read_text(encoding="utf-8"))
    # 形式: list of {cy, cx, crop_w, crop_h, ...}
    return [(r["cy"], r["cx"]) for r in rois]


def run_ecc_one_tp(ref_crops_u8, phase_tp, channels, crop_w, crop_h, cfg):
    """1 TP 分の per-channel ECC を実行して (tx_agg, ty_agg, corr_mean) を返す。"""
    vmin = cfg.get("ecc_vmin", -5.0)
    vmax = cfg.get("ecc_vmax",  2.0)

    tx_list, ty_list, corr_list = [], [], []
    for ch_idx, (cy, cx) in enumerate(channels):
        crop = extract_rect_roi(phase_tp, cy, cx, crop_w, crop_h).astype(np.float32)
        offset = compute_backsub_offset(crop, cfg)
        crop_bs = crop + offset
        curr_u8 = to_uint8(crop_bs, vmin, vmax)

        result = ecc_align(ref_crops_u8[ch_idx], curr_u8)
        if result is not None:
            tx, ty, corr = result
            tx_list.append(tx)
            ty_list.append(ty)
            corr_list.append(corr)

    if not tx_list:
        return None, None, None

    n = len(tx_list)
    if n >= 3:
        out_x = _remove_outliers_mad(tx_list)
        out_y = _remove_outliers_mad(ty_list)
        is_out = out_x | out_y
        used = [i for i, o in enumerate(is_out) if not o] or list(range(n))
    else:
        used = list(range(n))

    tx_arr = np.array(tx_list)
    ty_arr = np.array(ty_list)
    corr_arr = np.array(corr_list)
    return (float(np.mean(tx_arr[used])),
            float(np.mean(ty_arr[used])),
            float(np.mean(corr_arr[used])))


def sweep_one_crop(args):
    """(crop_w, crop_h) 1 組分の全 TP ECC を実行。マルチプロセス対応のトップレベル関数。"""
    crop_w, crop_h, phase_paths, tp_indices, channels, cfg = args

    vmin = cfg.get("ecc_vmin", -5.0)
    vmax = cfg.get("ecc_vmax",  2.0)
    pixel_scale = cfg.get("pixel_scale_um", 0.3462)
    sx_sign = cfg.get("shift_sign_x", 1)
    sy_sign = cfg.get("shift_sign_y", 1)

    # 参照 crops を TP=0 の位相画像から切り出す
    phase_ref = tifffile.imread(str(phase_paths[0])).astype(np.float32)
    ref_crops_u8 = []
    for cy, cx in channels:
        crop = extract_rect_roi(phase_ref, cy, cx, crop_w, crop_h).astype(np.float32)
        offset = compute_backsub_offset(crop, cfg)
        ref_crops_u8.append(to_uint8(crop + offset, vmin, vmax))

    cum_x = 0.0
    cum_y = 0.0
    cum_x_series = []
    cum_y_series = []
    corr_series  = []
    valid_tps    = []

    for i, path in enumerate(phase_paths[1:], start=1):
        try:
            phase_tp = tifffile.imread(str(path)).astype(np.float32)
        except (FileNotFoundError, Exception):
            continue
        tx_agg, ty_agg, corr_mean = run_ecc_one_tp(
            ref_crops_u8, phase_tp, channels, crop_w, crop_h, cfg
        )
        if tx_agg is None:
            continue

        # 軸変換: image X→stage Y, image Y→stage X（compute_drift_online.py と同じ）
        corr_x = sx_sign * ty_agg * pixel_scale
        corr_y = sy_sign * tx_agg * pixel_scale
        cum_x += corr_x
        cum_y += corr_y
        cum_x_series.append(cum_x)
        cum_y_series.append(cum_y)
        corr_series.append(corr_mean)
        valid_tps.append(tp_indices[i] if i < len(tp_indices) else i)

    return {
        "crop_w": crop_w, "crop_h": crop_h,
        "tps": np.array(valid_tps),
        "cum_x": np.array(cum_x_series),
        "cum_y": np.array(cum_y_series),
        "corr_mean": float(np.mean(corr_series)) if corr_series else np.nan,
    }


def residual_std_vs_gt(cum_series, gt_series, tps_sweep, tps_gt):
    """sweep の累積を ground truth の対応 TP と比較して残差 std を返す。"""
    # TP インデックスで対応付け
    gt_dict = dict(zip(tps_gt, gt_series))
    gt_at_sweep = np.array([gt_dict.get(tp, np.nan) for tp in tps_sweep])
    valid = ~np.isnan(gt_at_sweep)
    if valid.sum() < 5:
        return np.nan
    residual = cum_series[valid] - gt_at_sweep[valid]
    return float(np.std(residual))


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    phase_dir   = Path(cfg["save_dir"]) / "Pos1" / "output_phase"
    log_path    = cfg["log_file"]
    rois_path   = cfg["channel_rois_json"]

    phase_paths = sorted(phase_dir.glob("img_*_phase.tif"))
    if not phase_paths:
        print(f"ERROR: 位相画像が見つかりません: {phase_dir}")
        sys.exit(1)

    print(f"Phase images: {len(phase_paths)}  ({phase_paths[0].name} .. {phase_paths[-1].name})")

    # TP 数を制限
    if args.max_tp is not None:
        phase_paths = phase_paths[:args.max_tp + 1]
        print(f"  -> max_tp={args.max_tp} で {len(phase_paths)} TP に制限")

    channels = load_channels(rois_path)
    print(f"Channels: {len(channels)}")

    tps_gt, cum_dx_raw, cum_dy_raw, gt_x, gt_y = load_ground_truth(log_path, args.max_tp)
    print(f"Ground truth TPs: {len(tps_gt)} ({tps_gt[0]}..{tps_gt[-1]})")

    # TP インデックスを phase_paths のインデックスに対応させる
    # ファイル名から TP インデックスを抽出
    def tp_from_path(p):
        return int(p.stem.split("_")[1])

    tp_indices = [tp_from_path(p) for p in phase_paths]

    # ---- Sweep 実行 ----
    sweep_params = list(product(CROP_W_LIST, CROP_H_LIST))
    print(f"\nSweep: {len(sweep_params)} 組み合わせ × {len(phase_paths)-1} TP × {len(channels)} ch")
    print("crop_w × crop_h:", [(cw, ch) for cw, ch in sweep_params])

    sweep_args = [
        (cw, ch, phase_paths, tp_indices, channels, cfg)
        for cw, ch in sweep_params
    ]

    results = []
    for i, sa in enumerate(sweep_args):
        cw, ch = sa[0], sa[1]
        print(f"  [{i+1}/{len(sweep_args)}] crop_w={cw} crop_h={ch} ...", end=" ", flush=True)
        res = sweep_one_crop(sa)
        # ground truth との残差 std を計算
        res["std_x"] = residual_std_vs_gt(res["cum_x"], gt_x, res["tps"], tps_gt)
        res["std_y"] = residual_std_vs_gt(res["cum_y"], gt_y, res["tps"], tps_gt)
        results.append(res)
        print(f"std_x={res['std_x']:.4f}  std_y={res['std_y']:.4f}  corr2={res['corr_mean']:.4f}")

    # ---- サマリー ----
    print("\n=== Residual std vs Ground Truth (um) ===")
    print(f"{'crop_w':>7} {'crop_h':>7}  {'std_x':>8}  {'std_y':>8}  {'corr2':>8}")
    for r in sorted(results, key=lambda r: r["std_x"] + r["std_y"]):
        print(f"{r['crop_w']:>7} {r['crop_h']:>7}  {r['std_x']:>8.4f}  {r['std_y']:>8.4f}  {r['corr_mean']:>8.4f}")

    best = min(results, key=lambda r: (r["std_x"] + r["std_y"]) / 2)
    print(f"\nBest (X+Y avg): crop_w={best['crop_w']}  crop_h={best['crop_h']}")

    # 現在の設定（40×80）の結果
    current = next((r for r in results if r["crop_w"] == 40 and r["crop_h"] == 80), None)
    if current:
        print(f"Current (40x80): std_x={current['std_x']:.4f}  std_y={current['std_y']:.4f}")

    # ---- 図 ----
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.4,
                             left=0.08, right=0.97, top=0.93, bottom=0.08)

    # ヒートマップ (X)
    ax_hx = fig.add_subplot(gs[0, 0])
    _heatmap(ax_hx, results, "std_x", "Residual std X (um)", CROP_W_LIST, CROP_H_LIST)

    # ヒートマップ (Y)
    ax_hy = fig.add_subplot(gs[0, 1])
    _heatmap(ax_hy, results, "std_y", "Residual std Y (um)", CROP_W_LIST, CROP_H_LIST)

    # ヒートマップ (corr2)
    ax_hc = fig.add_subplot(gs[0, 2])
    _heatmap(ax_hc, results, "corr2_mean", "Mean corr2", CROP_W_LIST, CROP_H_LIST,
             cmap="RdYlGn", invert=False)

    # 最良 crop vs current vs ground truth の時系列比較
    time_h = tps_gt * 5.0 / 60  # 5 min interval → hours

    for col, (axis_name, gt_series, cum_key) in enumerate([
        ("X", gt_x, "cum_x"),
        ("Y", gt_y, "cum_y"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(time_h, gt_series, "k-", lw=2.0, label="Ground truth (SavGol)", zorder=5)

        if current:
            t_h_c = current["tps"] * 5.0 / 60
            ax.plot(t_h_c, current[cum_key], color="#2196F3", lw=0.8, alpha=0.7,
                    label=f"current 40x80 (std={current['std_' + axis_name.lower()]:.3f}um)")

        if best["crop_w"] != 40 or best["crop_h"] != 80:
            t_h_b = best["tps"] * 5.0 / 60
            ax.plot(t_h_b, best[cum_key], color="#F44336", lw=0.8, alpha=0.7,
                    label=f"best {best['crop_w']}x{best['crop_h']} (std={best['std_' + axis_name.lower()]:.3f}um)")

        ax.set_xlabel("Time (h)")
        ax.set_ylabel(f"Cumulative drift {axis_name} (um)")
        ax.set_title(f"Cumulative drift {axis_name}: best vs current vs GT")
        ax.legend(frameon=False, fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # サマリーテキスト
    ax_t = fig.add_subplot(gs[1, 2])
    ax_t.axis("off")
    lines = ["crop_w  crop_h  std_x  std_y  corr2"]
    for r in sorted(results, key=lambda r: (r["std_x"] + r["std_y"]) / 2)[:8]:
        marker = " <--best" if (r["crop_w"] == best["crop_w"] and r["crop_h"] == best["crop_h"]) else ""
        curr_m = " [cur]"  if (r["crop_w"] == 40 and r["crop_h"] == 80) else ""
        lines.append(f"{r['crop_w']:>6}  {r['crop_h']:>6}  "
                     f"{r['std_x']:.3f}  {r['std_y']:.3f}  "
                     f"{r['corr_mean']:.3f}{marker}{curr_m}")
    ax_t.text(0.02, 0.95, "\n".join(lines), va="top", ha="left",
              fontsize=7, family="monospace", transform=ax_t.transAxes)

    fig.suptitle("ECC Crop Sweep: Residual std vs Ground Truth", fontsize=12)

    # 数値データも保存
    data = {"tps_gt": tps_gt, "gt_x": gt_x, "gt_y": gt_y, "cum_dx_raw": cum_dx_raw, "cum_dy_raw": cum_dy_raw}
    for r in results:
        k = f"w{r['crop_w']}h{r['crop_h']}"
        data[f"tps_{k}"]  = r["tps"]
        data[f"cum_x_{k}"] = r["cum_x"]
        data[f"cum_y_{k}"] = r["cum_y"]

    save_figure(
        fig,
        params={
            "n_phase_images": len(phase_paths),
            "n_channels": len(channels),
            "crop_w_list": CROP_W_LIST,
            "crop_h_list": CROP_H_LIST,
            "best_crop_w": best["crop_w"],
            "best_crop_h": best["crop_h"],
            "best_std_x": best["std_x"],
            "best_std_y": best["std_y"],
            **{f"std_x_w{r['crop_w']}h{r['crop_h']}": r["std_x"] for r in results},
            **{f"std_y_w{r['crop_w']}h{r['crop_h']}": r["std_y"] for r in results},
        },
        description=f"ECC crop sweep: residual std vs ground truth. Best: {best['crop_w']}x{best['crop_h']}",
        data=data,
    )
    plt.close(fig)


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
    ax.set_xticks(range(len(crop_h_list)))
    ax.set_xticklabels(crop_h_list)
    ax.set_yticks(range(len(crop_w_list)))
    ax.set_yticklabels(crop_w_list)
    ax.set_xlabel("crop_h (X-dir, px)")
    ax.set_ylabel("crop_w (Y-dir, px)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 数値を書き込む
    for wi, cw in enumerate(crop_w_list):
        for hi, ch in enumerate(crop_h_list):
            v = matrix[wi, hi]
            if not np.isnan(v):
                ax.text(hi, wi, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if invert else "black")

    # 現在の設定 (40×80) をマーク
    if 40 in crop_w_list and 80 in crop_h_list:
        wi = crop_w_list.index(40)
        hi = crop_h_list.index(80)
        ax.add_patch(plt.Rectangle((hi - 0.5, wi - 0.5), 1, 1,
                                    fill=False, edgecolor="white", lw=2.5))


if __name__ == "__main__":
    main()
