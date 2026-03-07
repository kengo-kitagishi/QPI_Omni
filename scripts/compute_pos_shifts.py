# %%
"""
compute_pos_shifts.py
---------------------
1つのPosのチャネルスタック群（channel_XX*.tif）に対して
ECC or phase_correlationでフレームごとのシフト量を計算し、
チャネル間で外れ値除去しながら平均してpos_shifts.jsonに保存する。
"""
import numpy as np
import tifffile
import cv2
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import median_filter

# ============================================================
# 設定パラメータ
# ============================================================
CHANNELS_DIR = r"E:\Acuisition\kitagishi\260301\movetest_8\Pos4\channels"
CHANNEL_PATTERN = "channel_*.tif"      # backsub済みなら "channel_*_bg_corr.tif"

# --- 基準画像の選択 ---
# USE_GRID_REFERENCE = True  : グリッドの x+0_y+0 画像をcropして各チャネルの基準にする（推奨）
# USE_GRID_REFERENCE = False : タイムラプスの REFERENCE_FRAME 番目を基準にする（従来方式）
USE_GRID_REFERENCE  = True
GRID_DIR            = r"E:\Acuisition\kitagishi\260301\multipos_test_1"
GRID_BASE_LABEL     = "Pos4"           # PosX_x+0_y+0 の PosX 部分
GRID_Z_INDEX        = 5               # img_000000000_ph_{Z_INDEX:03d}.tif
CHANNEL_ROIS_JSON   = r"E:\Acuisition\kitagishi\260301\movetest_8\Pos4\channels\channel_rois.json"

REFERENCE_FRAME = 150                  # USE_GRID_REFERENCE=False の場合のみ使用（1始まり）

ALIGNMENT_METHOD = 'ecc'              # 'ecc' or 'phase_correlation'
VMIN = -5.0
VMAX = 2.0                            # to_uint8の正規化範囲（ECC精度に影響）
OUTLIER_MAD_THRESH = 2.5              # チャネル間外れ値除去のMAD閾値
OUTLIER_TIMESERIES_WINDOW = 11        # 時系列外れ値検出のメジアンフィルタ幅（奇数）
OUTLIER_TIMESERIES_THRESH = 3.0       # 時系列MAD閾値（0で無効）
OUTPUT_JSON = "pos_shifts.json"
# ============================================================


def to_uint8(img, vmin=VMIN, vmax=VMAX):
    clipped = np.clip(img, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)
    return (normalized * 255).astype(np.uint8)


def ecc_align(ref_u8, tl_u8):
    """ECC アライメントで (shift_x, shift_y, correlation) を返す。失敗時は None。"""
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-8)
    try:
        correlation, warp_matrix = cv2.findTransformECC(
            ref_u8, tl_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria
        )
        return float(warp_matrix[0, 2]), float(warp_matrix[1, 2]), float(correlation)
    except Exception:
        return None


def phase_align(ref_img, tl_img):
    """phase_cross_correlation で (shift_x, shift_y, correlation) を返す。失敗時は None。"""
    from skimage import registration
    try:
        shift, error, _ = registration.phase_cross_correlation(
            ref_img, tl_img, upsample_factor=10
        )
        return float(shift[1]), float(shift[0]), float(1.0 - error)
    except Exception:
        return None


def mad(arr):
    """Median Absolute Deviation"""
    m = np.median(arr)
    return np.median(np.abs(arr - m))


def remove_outliers_mad(values, thresh):
    """外れ値フラグを返す。values: list of float, thresh: MAD閾値倍率。"""
    arr = np.array(values, dtype=np.float64)
    m = np.median(arr)
    md = mad(arr)
    if md == 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(arr - m) > thresh * md


def detect_timeseries_outliers(shift_avg, window, thresh):
    """
    時系列シフトに対してrolling median basedの外れ値検出。
    window: メジアンフィルタ幅（奇数推奨）
    thresh: MAD倍率（0で全フラグfalse）
    Returns: bool array, shape=(n_frames,)
    """
    if thresh <= 0:
        return np.zeros(len(shift_avg), dtype=bool)
    arr = np.array(shift_avg, dtype=np.float64)
    smoothed = median_filter(arr, size=window, mode='reflect')
    residual = arr - smoothed
    md = mad(residual)
    if md == 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(residual) > thresh * md


def load_grid_refs(channels_dir, n_channels):
    """
    グリッドの x+0_y+0 画像を読み込み、各チャネルのROIでcropして
    per-channel基準画像リストを返す。
    """
    from channel_crop import extract_rect_roi

    # 候補を優先順に試す:
    #   1. output_phase/*_ph_ZZZ_phase.tif  (pipeline_full.py 再構成済み)
    #   2. output_phase/*_ph_ZZZ.tif        (旧命名)
    #   3. *_ph_ZZZ.tif                     (未再構成の生画像 fallback)
    base_dir = Path(GRID_DIR) / f"{GRID_BASE_LABEL}_x+0_y+0"
    z_str = f"ph_{GRID_Z_INDEX:03d}"

    candidates = [
        base_dir / "output_phase" / f"img_000000000_{z_str}_phase.tif",
        base_dir / "output_phase" / f"img_000000000_{z_str}.tif",
        base_dir / f"img_000000000_{z_str}.tif",
    ]
    grid_ref_path = next((p for p in candidates if p.exists()), None)
    if grid_ref_path is None:
        raise FileNotFoundError(
            f"グリッド基準画像が見つかりません: {base_dir}\n"
            f"  試したパス:\n" + "\n".join(f"    {p}" for p in candidates)
        )

    rois_path = Path(CHANNEL_ROIS_JSON)
    if not rois_path.exists():
        raise FileNotFoundError(f"channel_rois.json が見つかりません: {rois_path}")

    grid_img = tifffile.imread(str(grid_ref_path)).astype(np.float64)
    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)

    print(f"グリッド基準画像: {grid_ref_path}")
    print(f"  グリッド画像サイズ: {grid_img.shape}")

    refs = []
    for ch in range(n_channels):
        roi = rois[ch] if ch < len(rois) else rois[-1]
        cropped = extract_rect_roi(grid_img, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"])
        refs.append(cropped)
        print(f"  ch{ch:02d} ROI crop: {cropped.shape}")

    return refs, str(grid_ref_path)


def main():
    channels_dir = Path(CHANNELS_DIR)
    if not channels_dir.exists():
        print(f"ERROR: CHANNELS_DIR が見つかりません: {channels_dir}")
        sys.exit(1)

    # チャネルスタック一覧
    stacks_paths = sorted(channels_dir.glob(CHANNEL_PATTERN))
    if not stacks_paths:
        print(f"ERROR: {CHANNEL_PATTERN} に合うファイルが見つかりません: {channels_dir}")
        sys.exit(1)
    print(f"チャネル数: {len(stacks_paths)}")
    for p in stacks_paths:
        print(f"  {p.name}")

    # スタック読み込み
    stacks = []
    for p in stacks_paths:
        arr = tifffile.imread(str(p))
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # 1フレームの場合
        stacks.append(arr.astype(np.float64))
        print(f"  {p.name}: shape={arr.shape}")

    n_frames = stacks[0].shape[0]
    n_channels = len(stacks)
    print(f"\nフレーム数: {n_frames}")
    print(f"アライメント手法: {ALIGNMENT_METHOD}")
    print(f"外れ値MAD閾値: {OUTLIER_MAD_THRESH}")
    if OUTLIER_TIMESERIES_THRESH > 0:
        print(f"時系列外れ値: window={OUTLIER_TIMESERIES_WINDOW}, thresh={OUTLIER_TIMESERIES_THRESH}")
    else:
        print("時系列外れ値検出: 無効")

    # 基準画像の構築
    reference_info = {}
    if USE_GRID_REFERENCE:
        print(f"\n基準: グリッド x+0_y+0  ({GRID_BASE_LABEL})")
        try:
            refs, grid_ref_path_str = load_grid_refs(channels_dir, n_channels)
            reference_info = {
                "reference_type": "grid",
                "grid_dir": GRID_DIR,
                "grid_base_label": GRID_BASE_LABEL,
                "grid_z_index": GRID_Z_INDEX,
                "grid_reference_path": grid_ref_path_str,
            }
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        ref_idx = REFERENCE_FRAME - 1
        if ref_idx < 0 or ref_idx >= n_frames:
            print(f"ERROR: REFERENCE_FRAME={REFERENCE_FRAME} が範囲外 (1~{n_frames})")
            sys.exit(1)
        print(f"\n基準: タイムラプス フレーム {REFERENCE_FRAME} (0-indexed: {ref_idx})")
        refs = [stacks[ch][ref_idx] for ch in range(n_channels)]
        reference_info = {
            "reference_type": "timelapse_frame",
            "reference_frame": REFERENCE_FRAME,
        }

    if ALIGNMENT_METHOD == 'ecc':
        refs_u8 = [to_uint8(r) for r in refs]

    # フレームごとにアライメント計算
    frame_results = []

    for t in tqdm(range(n_frames), desc="フレーム処理"):
        per_channel = []
        for ch in range(n_channels):
            frame = stacks[ch][t]
            if ALIGNMENT_METHOD == 'ecc':
                result = ecc_align(refs_u8[ch], to_uint8(frame))
            else:
                result = phase_align(refs[ch], frame)

            if result is None:
                per_channel.append({
                    "channel": ch,
                    "shift_x": None,
                    "shift_y": None,
                    "correlation": None,
                    "excluded": True,
                    "exclude_reason": "alignment_failed"
                })
            else:
                sx, sy, corr = result
                per_channel.append({
                    "channel": ch,
                    "shift_x": sx,
                    "shift_y": sy,
                    "correlation": corr,
                    "excluded": False,
                    "exclude_reason": None
                })

        # 有効なシフト値のみ取り出す
        valid = [c for c in per_channel if not c["excluded"]]
        if len(valid) == 0:
            frame_results.append({
                "frame_index": t,
                "shift_x_avg": None,
                "shift_y_avg": None,
                "n_channels_used": 0,
                "n_channels_excluded_outlier": 0,
                "is_outlier_timeseries": False,
                "per_channel": per_channel
            })
            continue

        # チャネル間外れ値除去（MAD）
        xs = np.array([c["shift_x"] for c in valid])
        ys = np.array([c["shift_y"] for c in valid])

        if len(valid) >= 3:
            outlier_x = remove_outliers_mad(xs.tolist(), OUTLIER_MAD_THRESH)
            outlier_y = remove_outliers_mad(ys.tolist(), OUTLIER_MAD_THRESH)
            is_outlier = outlier_x | outlier_y
        else:
            is_outlier = np.zeros(len(valid), dtype=bool)

        for i, c in enumerate(valid):
            if is_outlier[i]:
                c["excluded"] = True
                c["exclude_reason"] = "channel_outlier_mad"
        # per_channel リストに反映されている（参照）

        used = [c for c in valid if not c["excluded"]]
        n_excluded = int(np.sum(is_outlier))

        if len(used) == 0:
            # 全チャネルが外れ値扱い → 外れ値除去なしで全平均
            sx_avg = float(np.mean(xs))
            sy_avg = float(np.mean(ys))
            n_used = len(valid)
            n_excluded = 0
            for c in valid:
                c["excluded"] = False
                c["exclude_reason"] = None
        else:
            sx_avg = float(np.mean([c["shift_x"] for c in used]))
            sy_avg = float(np.mean([c["shift_y"] for c in used]))
            n_used = len(used)

        frame_results.append({
            "frame_index": t,
            "shift_x_avg": sx_avg,
            "shift_y_avg": sy_avg,
            "n_channels_used": n_used,
            "n_channels_excluded_outlier": n_excluded,
            "is_outlier_timeseries": False,  # 後で更新
            "per_channel": per_channel
        })

    # 時系列外れ値検出
    avg_x = [r["shift_x_avg"] for r in frame_results]
    avg_y = [r["shift_y_avg"] for r in frame_results]
    # None を含む場合は線形補間してから検出
    def fill_none(arr):
        arr = np.array([np.nan if v is None else v for v in arr], dtype=np.float64)
        nans = np.isnan(arr)
        if nans.all():
            return arr
        xp = np.where(~nans)[0]
        arr[nans] = np.interp(np.where(nans)[0], xp, arr[xp])
        return arr

    avg_x_filled = fill_none(avg_x)
    avg_y_filled = fill_none(avg_y)
    ts_outlier_x = detect_timeseries_outliers(avg_x_filled, OUTLIER_TIMESERIES_WINDOW, OUTLIER_TIMESERIES_THRESH)
    ts_outlier_y = detect_timeseries_outliers(avg_y_filled, OUTLIER_TIMESERIES_WINDOW, OUTLIER_TIMESERIES_THRESH)
    ts_outlier = ts_outlier_x | ts_outlier_y

    for i, r in enumerate(frame_results):
        r["is_outlier_timeseries"] = bool(ts_outlier[i])

    n_ts_outlier = int(np.sum(ts_outlier))
    print(f"\n時系列外れ値フレーム数: {n_ts_outlier} / {n_frames}")

    # シフト統計
    valid_avg_x = [r["shift_x_avg"] for r in frame_results if r["shift_x_avg"] is not None]
    valid_avg_y = [r["shift_y_avg"] for r in frame_results if r["shift_y_avg"] is not None]
    if valid_avg_x:
        print(f"shift_x: 平均={np.mean(valid_avg_x):.3f}, 範囲=[{np.min(valid_avg_x):.3f}, {np.max(valid_avg_x):.3f}]")
        print(f"shift_y: 平均={np.mean(valid_avg_y):.3f}, 範囲=[{np.min(valid_avg_y):.3f}, {np.max(valid_avg_y):.3f}]")

    # JSON保存
    out = {
        "method": ALIGNMENT_METHOD,
        "n_channels": n_channels,
        "n_frames": n_frames,
        "outlier_mad_thresh": OUTLIER_MAD_THRESH,
        "outlier_timeseries_window": OUTLIER_TIMESERIES_WINDOW,
        "outlier_timeseries_thresh": OUTLIER_TIMESERIES_THRESH,
        "channels_dir": str(channels_dir),
        "channel_pattern": CHANNEL_PATTERN,
        **reference_info,
        # shift_visualize.py互換フィールド（平均シフト量を alignment_results 形式でも持つ）
        "alignment_results": [
            {
                "filename": f"frame_{r['frame_index']:06d}",
                "shift_x": r["shift_x_avg"] if r["shift_x_avg"] is not None else 0.0,
                "shift_y": r["shift_y_avg"] if r["shift_y_avg"] is not None else 0.0,
                "correlation": None,
                "warp_matrix": [[1.0, 0.0, r["shift_x_avg"] or 0.0],
                                 [0.0, 1.0, r["shift_y_avg"] or 0.0]],
                "is_outlier_timeseries": r["is_outlier_timeseries"]
            }
            for r in frame_results
        ],
        "frame_results": frame_results
    }

    out_path = channels_dir / OUTPUT_JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n保存完了: {out_path}")

    # shift_visualize で可視化
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from shift_visualize import visualize_shifts
        visualize_shifts(str(out_path))
    except Exception as e:
        print(f"[shift_visualize] スキップ: {e}")


if __name__ == "__main__":
    main()

# %%
