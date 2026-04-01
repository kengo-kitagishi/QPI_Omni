# %%
"""
visualize_grid_true_positions.py
---------------------------------
グリッドスキャン各点の「本当のピクセル位置」を ECC で実測し可視化する。
直接ECC と BFS チェーン の 2 方式を実行して比較する。

【ECC 手順】compute_pos_shifts.py と完全に揃えた:
  - 前処理: _tilt_correct (TILT_CROP_H=270 → ECC_CROP_H=80)
  - 正規化: to_uint8 固定 VMIN/VMAX (-5/2)
  - チャネル平均: MAD 外れ値除去 (OUTLIER_MAD_THRESH=2.5) → 残チャネル平均
  - ECC: MOTION_TRANSLATION, 100000 iter, 1e-8

【方式 1: 直接ECC】
  center (0,0) を基準に各点を直接 ECC。最大シフト ±1.45 px。

【方式 2: BFS チェーン】
  calibrate_grid_positions.py と同じ BFS 方式。同じ前処理で実行。

【出力】
  Figure 1: 直接ECC — Nominal vs Measured + Residual map
  Figure 2: BFS    — Nominal vs Measured + Residual map
  単位: um
"""
import sys
import re
import json
import numpy as np
import tifffile
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

# ============================================================
# 設定パラメータ
# ============================================================
GRID_DIR   = Path(r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1")
POS_PREFIX = "Pos1"
Z_IDX      = 9

# ECC パラメータ (compute_pos_shifts.py と同値)
VMIN, VMAX       = -5.0, 2.0
ECC_MAX_ITER     = 100000
ECC_EPSILON      = 1e-8

# Tilt 補正パラメータ (compute_pos_shifts.py と同値)
TILT_CROP_H = 270   # X 方向の big crop 幅 [px]
ECC_CROP_H  = 80    # ECC に使う中央 crop 幅 [px]

# チャネル外れ値除去 (compute_pos_shifts.py と同値)
OUTLIER_MAD_THRESH = 2.5

# 光学パラメータ
SENSOR_PIXEL_SIZE = 3.45e-6   # [m]
MAGNIFICATION     = 40
ORIGINAL_DIM      = 2048
RECONSTRUCTED_DIM = 511

# グリッドステップ
X_STEP       = 0.1   # [um]
Y_STEP       = 0.1   # [um]
SHIFT_SIGN_X = -1
SHIFT_SIGN_Y = -1

# channel_rois.json パス (None → Pos1_x+0_y+0/output_phase/channels/ から自動)
CHANNEL_ROIS_JSON = None
# ============================================================

pixel_scale_um = (SENSOR_PIXEL_SIZE / MAGNIFICATION) * (ORIGINAL_DIM / RECONSTRUCTED_DIM) * 1e6
STEP_PX = X_STEP / pixel_scale_um  # ~0.289 px/step
PX2UM   = pixel_scale_um


# ---- ユーティリティ関数 ----

def extract_rect_roi(img, cy, cx, crop_w, crop_h):
    """(cy, cx) 中心で (crop_w rows x crop_h cols) を切り出す（パディングあり）。"""
    h, w = img.shape
    y1 = cy - crop_w // 2; y2 = y1 + crop_w
    x1 = cx - crop_h // 2; x2 = x1 + crop_h
    pad_y0 = max(0, -y1); y1 = max(0, y1)
    pad_y1 = max(0, y2 - h); y2 = min(h, y2)
    pad_x0 = max(0, -x1); x1 = max(0, x1)
    pad_x1 = max(0, x2 - w); x2 = min(w, x2)
    crop = img[y1:y2, x1:x2]
    if any([pad_y0, pad_y1, pad_x0, pad_x1]):
        crop = np.pad(crop, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant")
    return crop


def _tilt_correct(img_f64, cy, cx, crop_w, crop_h_out):
    """
    compute_pos_shifts.py の _tilt_correct と同じ処理。
    big crop (TILT_CROP_H cols) → 左1/3 slope fit → 補正 → 中央 crop_h_out cols。
    """
    big   = extract_rect_roi(img_f64, cy, cx, crop_w, TILT_CROP_H).astype(np.float64)
    x     = np.arange(TILT_CROP_H, dtype=np.float64)
    prof  = big.mean(axis=0)
    fit_n = max(1, TILT_CROP_H // 3)
    a, b  = np.polyfit(x[:fit_n], prof[:fit_n], 1)
    corrected = big - (a * x + b)[np.newaxis, :]
    start = (TILT_CROP_H - crop_h_out) // 2
    return corrected[:, start : start + crop_h_out]


def to_uint8(img, vmin=VMIN, vmax=VMAX):
    clipped    = np.clip(img, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)
    return (normalized * 255).astype(np.uint8)


def ecc_align(ref_u8, tl_u8):
    """
    ECC で (tx, ty, corr) を返す。失敗時は None。
    findTransformECC(ref, tl) → actual_dx = -tx,  actual_dy = -ty
    """
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITER, ECC_EPSILON)
    try:
        corr, warp_matrix = cv2.findTransformECC(
            ref_u8, tl_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        return float(warp_matrix[0, 2]), float(warp_matrix[1, 2]), float(corr)
    except Exception:
        return None


def remove_outliers_mad(values, thresh):
    """MAD 外れ値フラグを返す。compute_pos_shifts.py と同じ実装。"""
    arr = np.array(values, dtype=np.float64)
    m   = np.median(arr)
    md  = np.median(np.abs(arr - m))
    if md == 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(arr - m) > thresh * md


def _channel_average(tx_list, ty_list, corr_list):
    """
    チャネルリストから MAD 外れ値除去後の平均を計算。
    compute_pos_shifts.py の _frame_result_from_per_channel と同じロジック。
    戻り値: (actual_dx, actual_dy, mean_corr)
    """
    if not tx_list:
        return None
    tx_arr   = np.array(tx_list,   dtype=np.float64)
    ty_arr   = np.array(ty_list,   dtype=np.float64)
    corr_arr = np.array(corr_list, dtype=np.float64)
    if len(tx_list) >= 3:
        out_x = remove_outliers_mad(tx_list, OUTLIER_MAD_THRESH)
        out_y = remove_outliers_mad(ty_list, OUTLIER_MAD_THRESH)
        mask  = ~(out_x | out_y)
        if mask.any():
            tx_arr   = tx_arr[mask]
            ty_arr   = ty_arr[mask]
            corr_arr = corr_arr[mask]
    return (float(-np.mean(tx_arr)),   # actual_dx = -tx
            float(-np.mean(ty_arr)),   # actual_dy = -ty
            float(np.mean(corr_arr)))


def scan_grid_positions(grid_dir, base_label):
    """(xi, yi) → folder_path のマップを返す。"""
    pattern = re.compile(rf"^{re.escape(base_label)}_x([+-]?\d+)_y([+-]?\d+)$")
    pos_map = {}
    for d in Path(grid_dir).iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            pos_map[(int(m.group(1)), int(m.group(2)))] = d
    return pos_map


def load_phase_image(pos_dir, z_idx):
    """output_phase/img_000000000_ph_{z_idx:03d}_phase.tif を float64 で返す。"""
    path = Path(pos_dir) / "output_phase" / f"img_000000000_ph_{z_idx:03d}_phase.tif"
    if not path.exists():
        raise FileNotFoundError(f"Phase image not found: {path}")
    return tifffile.imread(str(path)).astype(np.float64)


def make_crops_u8(img_f64, rois):
    """各チャネル ROI に _tilt_correct + to_uint8 を適用して crops リストを返す。"""
    return [to_uint8(_tilt_correct(img_f64, roi["cy"], roi["cx"], roi["crop_w"], ECC_CROP_H))
            for roi in rois]


# ---- 直接ECC 方式 ----

def run_direct_ecc(crops_cache, pos_map, rois, pixel_scale_um):
    """center (0,0) を基準に各グリッド位置を直接 ECC する。"""
    n_channels = len(rois)
    results = {(0, 0): {
        "actual_dx": 0.0, "actual_dy": 0.0,
        "nominal_dx": 0.0, "nominal_dy": 0.0,
        "corr": 1.0, "failed": False,
    }}

    if (0, 0) not in crops_cache:
        raise RuntimeError("center (0,0) のcropsがキャッシュにありません")
    ref_crops = crops_cache[(0, 0)]

    sorted_keys = sorted(pos_map.keys(), key=lambda k: abs(k[0]) + abs(k[1]))
    pbar = tqdm(total=len(pos_map), desc="Direct ECC")
    pbar.update(1)  # (0,0) はスキップ

    for (xi, yi) in sorted_keys:
        if (xi, yi) == (0, 0):
            continue
        nominal_dx = float(SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um)
        nominal_dy = float(SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um)

        if (xi, yi) not in crops_cache:
            results[(xi, yi)] = {
                "actual_dx": nominal_dx, "actual_dy": nominal_dy,
                "nominal_dx": nominal_dx, "nominal_dy": nominal_dy,
                "corr": None, "failed": True,
            }
            pbar.update(1)
            continue

        cur_crops = crops_cache[(xi, yi)]
        tx_list, ty_list, corr_list = [], [], []
        for ch in range(n_channels):
            res = ecc_align(ref_crops[ch], cur_crops[ch])
            if res is not None:
                tx_list.append(res[0]); ty_list.append(res[1]); corr_list.append(res[2])

        avg = _channel_average(tx_list, ty_list, corr_list)
        if avg is None:
            results[(xi, yi)] = {
                "actual_dx": nominal_dx, "actual_dy": nominal_dy,
                "nominal_dx": nominal_dx, "nominal_dy": nominal_dy,
                "corr": None, "failed": True,
            }
        else:
            results[(xi, yi)] = {
                "actual_dx": avg[0], "actual_dy": avg[1],
                "nominal_dx": nominal_dx, "nominal_dy": nominal_dy,
                "corr": avg[2], "failed": False,
            }
        pbar.update(1)

    pbar.close()
    return results


# ---- BFS チェーン方式 ----

def run_bfs(crops_cache, pos_map, rois, pixel_scale_um):
    """
    BFS チェーン方式: calibrate_grid_positions.py と同じロジック。
    前処理・チャネル平均は compute_pos_shifts.py に揃える。
    """
    n_channels = len(rois)
    calibrated = {(0, 0): (0.0, 0.0)}
    results    = {(0, 0): {
        "actual_dx": 0.0, "actual_dy": 0.0,
        "nominal_dx": 0.0, "nominal_dy": 0.0,
        "corr": 1.0, "failed": False,
    }}
    visited = {(0, 0)}
    queue   = deque()

    def enqueue_neighbors(xi, yi):
        for dxi, dyi in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (xi + dxi, yi + dyi)
            if nb in pos_map and nb not in visited:
                queue.append(nb)
                visited.add(nb)

    enqueue_neighbors(0, 0)
    pbar = tqdm(total=len(pos_map), desc="BFS ECC")
    pbar.update(1)

    while queue:
        xi, yi = queue.popleft()
        nominal_dx = float(SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um)
        nominal_dy = float(SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um)
        cal_nb = [(xi + d, yi + e) for d, e in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                  if (xi + d, yi + e) in calibrated]

        if (xi, yi) not in crops_cache or not cal_nb:
            ref_nb = cal_nb[0] if cal_nb else (0, 0)
            ref_dx, ref_dy = calibrated.get(ref_nb, (0.0, 0.0))
            step_dx = SHIFT_SIGN_Y * (yi - ref_nb[1]) * Y_STEP / pixel_scale_um
            step_dy = SHIFT_SIGN_X * (xi - ref_nb[0]) * X_STEP / pixel_scale_um
            act_dx  = ref_dx + step_dx
            act_dy  = ref_dy + step_dy
            calibrated[(xi, yi)] = (act_dx, act_dy)
            results[(xi, yi)] = {
                "actual_dx": act_dx, "actual_dy": act_dy,
                "nominal_dx": nominal_dx, "nominal_dy": nominal_dy,
                "corr": None, "failed": True,
            }
            pbar.update(1)
            enqueue_neighbors(xi, yi)
            continue

        cur_crops = crops_cache[(xi, yi)]
        dx_est, dy_est, corr_est = [], [], []

        for ref_nb in cal_nb:
            if ref_nb not in crops_cache:
                continue
            ref_crops_nb = crops_cache[ref_nb]
            ref_dx, ref_dy = calibrated[ref_nb]
            tx_list, ty_list, corr_list = [], [], []
            for ch in range(n_channels):
                res = ecc_align(ref_crops_nb[ch], cur_crops[ch])
                if res is not None:
                    tx_list.append(res[0]); ty_list.append(res[1]); corr_list.append(res[2])
            avg = _channel_average(tx_list, ty_list, corr_list)
            if avg is not None:
                dx_est.append(ref_dx + avg[0])
                dy_est.append(ref_dy + avg[1])
                corr_est.append(avg[2])

        if not dx_est:
            ref_nb = cal_nb[0]
            ref_dx, ref_dy = calibrated[ref_nb]
            step_dx = SHIFT_SIGN_Y * (yi - ref_nb[1]) * Y_STEP / pixel_scale_um
            step_dy = SHIFT_SIGN_X * (xi - ref_nb[0]) * X_STEP / pixel_scale_um
            act_dx  = ref_dx + step_dx
            act_dy  = ref_dy + step_dy
            calibrated[(xi, yi)] = (act_dx, act_dy)
            results[(xi, yi)] = {
                "actual_dx": act_dx, "actual_dy": act_dy,
                "nominal_dx": nominal_dx, "nominal_dy": nominal_dy,
                "corr": None, "failed": True,
            }
        else:
            act_dx = float(np.mean(dx_est))
            act_dy = float(np.mean(dy_est))
            calibrated[(xi, yi)] = (act_dx, act_dy)
            results[(xi, yi)] = {
                "actual_dx": act_dx, "actual_dy": act_dy,
                "nominal_dx": nominal_dx, "nominal_dy": nominal_dy,
                "corr": float(np.mean(corr_est)), "failed": False,
            }
        pbar.update(1)
        enqueue_neighbors(xi, yi)

    pbar.close()
    return results


# ---- 図の生成・保存 ----

def make_figure(results, method_label, pixel_scale_um):
    """2-panel 図を生成して (fig, arrays_dict) を返す。単位 um。"""
    sorted_keys = sorted(results.keys())
    xi_list  = np.array([k[0] for k in sorted_keys])
    yi_list  = np.array([k[1] for k in sorted_keys])
    nom_dx   = np.array([results[k]["nominal_dx"] for k in sorted_keys]) * PX2UM
    nom_dy   = np.array([results[k]["nominal_dy"] for k in sorted_keys]) * PX2UM
    act_dx   = np.array([results[k]["actual_dx"]  for k in sorted_keys]) * PX2UM
    act_dy   = np.array([results[k]["actual_dy"]  for k in sorted_keys]) * PX2UM
    corr_arr = np.array([results[k]["corr"] if results[k]["corr"] is not None else 0.0
                         for k in sorted_keys])
    res_um   = np.sqrt((act_dx - nom_dx)**2 + (act_dy - nom_dy)**2)

    n_failed = sum(1 for r in results.values() if r["failed"])
    success  = [r for r in results.values() if not r["failed"]
                and (r["nominal_dx"] != 0 or r["nominal_dy"] != 0)]
    if success:
        res_px_vals = np.array([np.sqrt((r["actual_dx"]-r["nominal_dx"])**2 +
                                        (r["actual_dy"]-r["nominal_dy"])**2)
                                for r in success])
        print(f"  [{method_label}] 成功: {len(results)-n_failed}/{len(results)}")
        print(f"  残差 [px]: mean={res_px_vals.mean():.4f}  std={res_px_vals.std():.4f}  "
              f"max={res_px_vals.max():.4f}")
        print(f"  残差 [um]: mean={res_px_vals.mean()*pixel_scale_um:.4f}  "
              f"max={res_px_vals.max()*pixel_scale_um:.4f}")
        corrs = [r["corr"] for r in success if r["corr"] is not None]
        if corrs:
            print(f"  ECC相関係数: mean={np.mean(corrs):.4f}  min={np.min(corrs):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Grid true position check: {POS_PREFIX}, z={Z_IDX}  [{method_label}]",
        fontsize=11,
    )

    # Panel 1: 名目格子点 vs 実測位置
    ax = axes[0]
    ax.scatter(nom_dx, nom_dy, s=80, facecolors="none", edgecolors="gray",
               linewidths=1.2, zorder=2, label="Nominal")
    sc = ax.scatter(act_dx, act_dy, s=30, c=corr_arr, cmap="RdYlGn",
                    vmin=0.97, vmax=1.0, zorder=3, label="Measured")
    plt.colorbar(sc, ax=ax, label="ECC correlation")
    ax.set_xlabel("dx (um)  [Stage Y / image X direction]")
    ax.set_ylabel("dy (um)  [Stage X / image Y direction]")
    ax.set_title("Nominal lattice vs Measured positions")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # Panel 2: 残差エラーマップ
    ax2 = axes[1]
    sc2 = ax2.scatter(xi_list, yi_list, c=res_um, s=100,
                      cmap="hot_r", vmin=0, vmax=max(float(res_um.max()), 0.05))
    plt.colorbar(sc2, ax=ax2, label="Position error (um)")
    ax2.set_xlabel("xi  (Stage X direction)")
    ax2.set_ylabel("yi  (Stage Y direction)")
    ax2.set_title(
        f"Residual |actual - nominal| (um)\n"
        f"mean={res_um.mean():.4f} um  max={res_um.max():.4f} um"
    )
    ax2.set_aspect("equal")
    ax2.grid(True, linewidth=0.3, alpha=0.5)
    for xi, yi, r in zip(xi_list, yi_list, res_um):
        if r > 0.05:
            ax2.text(xi, yi, f"{r:.2f}", fontsize=5,
                     ha="center", va="center", color="white", fontweight="bold")

    plt.tight_layout()

    arrays = {
        "nominal_dx_um": nom_dx, "nominal_dy_um": nom_dy,
        "actual_dx_um":  act_dx, "actual_dy_um":  act_dy,
        "xi": xi_list, "yi": yi_list,
        "residual_um": res_um, "corr": corr_arr,
    }
    return fig, arrays, res_um


# ---- メイン ----

def main():
    print(f"Pixel scale: {pixel_scale_um:.4f} um/px")
    print(f"Step: {X_STEP} um = {STEP_PX:.4f} px/step  "
          f"(+-5 steps = +-{5*STEP_PX:.3f} px max)")

    # ROI 読み込み
    if CHANNEL_ROIS_JSON is not None:
        rois_path = Path(CHANNEL_ROIS_JSON)
    else:
        rois_path = (GRID_DIR / f"{POS_PREFIX}_x+0_y+0"
                     / "output_phase" / "channels" / "channel_rois.json")
    if not rois_path.exists():
        print(f"ERROR: channel_rois.json が見つかりません: {rois_path}")
        sys.exit(1)
    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)
    n_channels = len(rois)
    print(f"チャネル数: {n_channels}")

    # グリッド Pos スキャン
    pos_map = scan_grid_positions(GRID_DIR, POS_PREFIX)
    if not pos_map:
        print(f"ERROR: グリッドPosが見つかりません: {GRID_DIR}/{POS_PREFIX}_x*_y*")
        sys.exit(1)
    xi_all = [k[0] for k in pos_map]
    yi_all = [k[1] for k in pos_map]
    print(f"グリッドPos数: {len(pos_map)}  "
          f"xi: [{min(xi_all)}, {max(xi_all)}]  yi: [{min(yi_all)}, {max(yi_all)}]")

    if (0, 0) not in pos_map:
        print("ERROR: center (0,0) が見つかりません")
        sys.exit(1)

    # 全 crops を一括ロード（直接ECC / BFS で共有）
    print("\n全グリッド画像をロード中...")
    crops_cache = {}
    n_load_failed = 0
    for (xi, yi), pos_dir in tqdm(sorted(pos_map.items()), desc="Loading crops"):
        try:
            img = load_phase_image(pos_dir, Z_IDX)
            crops_cache[(xi, yi)] = make_crops_u8(img, rois)
        except FileNotFoundError:
            n_load_failed += 1
    print(f"ロード完了: {len(crops_cache)}/{len(pos_map)}  "
          f"(crop shape: {crops_cache[(0,0)][0].shape})")
    if n_load_failed:
        print(f"  ロード失敗: {n_load_failed} 点 (output_phase なし)")

    # ---- 方式 1: 直接ECC ----
    print("\n=== 方式1: 直接ECC (center 基準) ===")
    results_direct = run_direct_ecc(crops_cache, pos_map, rois, pixel_scale_um)

    print("\n=== 方式2: BFS チェーン ===")
    results_bfs = run_bfs(crops_cache, pos_map, rois, pixel_scale_um)

    # ---- 図の生成・保存 ----
    print("\n=== 図の保存 ===")

    print("\n[Figure 1] 直接ECC")
    fig1, arr1, res1 = make_figure(results_direct, "Direct ECC", pixel_scale_um)
    save_figure(
        fig1,
        params={
            "pos_prefix": POS_PREFIX, "z_idx": Z_IDX,
            "method": "direct_ecc",
            "step_um": X_STEP, "step_px": float(STEP_PX),
            "tilt_crop_h": TILT_CROP_H, "ecc_crop_h": ECC_CROP_H,
            "outlier_mad_thresh": OUTLIER_MAD_THRESH,
        },
        description=(
            f"Grid true position check: {POS_PREFIX} z={Z_IDX}, "
            f"direct ECC vs nominal. "
            f"mean_residual={res1.mean():.4f} um"
        ),
        data=arr1,
    )
    plt.close(fig1)

    print("\n[Figure 2] BFS チェーン")
    fig2, arr2, res2 = make_figure(results_bfs, "BFS chain", pixel_scale_um)
    save_figure(
        fig2,
        params={
            "pos_prefix": POS_PREFIX, "z_idx": Z_IDX,
            "method": "bfs_chain",
            "step_um": X_STEP, "step_px": float(STEP_PX),
            "tilt_crop_h": TILT_CROP_H, "ecc_crop_h": ECC_CROP_H,
            "outlier_mad_thresh": OUTLIER_MAD_THRESH,
        },
        description=(
            f"Grid true position check: {POS_PREFIX} z={Z_IDX}, "
            f"BFS chain vs nominal. "
            f"mean_residual={res2.mean():.4f} um"
        ),
        data=arr2,
    )
    plt.close(fig2)

    print(f"\n比較: Direct ECC mean={res1.mean():.4f} um  |  BFS mean={res2.mean():.4f} um")
    if res2.mean() > res1.mean():
        print("  → BFS のほうが大きい: BFS 誤差蓄積の可能性")
    else:
        print("  → BFS と直接ECC が同程度: 誤差蓄積の影響は小さい")


if __name__ == "__main__":
    main()

# %%
