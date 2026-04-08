# %%
"""
focus_check_subtract.py
-----------------------
焦点確認用スクリプト。

grid/00（各 Pos の背景参照 = Pos{N}_x+0_y+0）と focus_test（サンプルの z スタック）の
対応する z フレームを引き算し、焦点位置を視覚的に確認する。

アライメントは Pos ごとに 1 回だけ ECC で計算して全 z に適用する。

出力: OUTPUT_DIR/Pos{N}/z{z:03d}.tif（float32）
ImageJ で File > Import > Image Sequence → Pos{N}/ を開けば z スタックとして閲覧可能。
"""

import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ===========================================================================
# パラメータ（ここを編集して使う）
# ===========================================================================

DO_RECONSTRUCTION = True
GRID_DIR    = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1"   # Pos{N}_x+0_y+0 を含む親ディレクトリ
FOCUS_DIR   = r"E:\Acuisition\kitagishi\260331\focus_check_6"  # Pos0..PosN を含む親ディレクトリ
GRID_SUFFIX = "x+0_y+0"           # 背景参照グリッド位置のサフィックス

POS_LABELS  = None     # None=自動検出, 例: ["Pos1", "Pos2"]（Pos0=BG は自動除外）
ALIGN_Z     = 10                   # ECC アライメントに使う z インデックス（z=0 = index 10）

CROP_OUTPUT = False                # True=channel ROI でクロップ, False=全体画像
OUTPUT_DIR  = r"E:\Acuisition\kitagishi\260331\focus_check_subtracted_6"

N_WORKERS   = 4                    # 再構成時の並列数（DO_RECONSTRUCTION=True 時のみ）

# 再構成パラメータ（DO_RECONSTRUCTION=True 時のみ使用）
# batch_reconstruction_grid.py と同じ規約で記述する
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

# Pos番号によるクロップ切り替え（batch_reconstruction_grid.py に合わせる）
# Pos番号 < POS_SPLIT → CROP_BEFORE;  Pos番号 >= POS_SPLIT → CROP_AFTER
# OFFAXIS_CENTER はクロップ後の座標で左右共通
# 物理的な対応:
#   小Pos番号（< POS_SPLIT）= 右チャンネル → col 400:2448
#   大Pos番号（>= POS_SPLIT）= 左チャンネル → col 0:2048
# [!] データセットによって左右が入れ替わる場合あり。必ず実データで確認すること。
POS_SPLIT    = 33
CROP_BEFORE  = (0, 2048, 400, 2448)   # pos < POS_SPLIT  → 右チャンネル（col 400-2448）
CROP_AFTER   = (0, 2048,   0, 2048)   # pos >= POS_SPLIT → 左チャンネル（col 0-2048）

FORCE_RECONSTRUCT = False            # True: 既存 output_phase を上書き再構成

# ECC 正規化範囲
ECC_VMIN = -5.0
ECC_VMAX =  2.0

# tilt 補正（compute_pos_shifts.py と同一設定）
USE_SLOPE_CORRECTION = True   # True: _tilt_correct を使う（backsub 不要）
TILT_CROP_H = 270             # 傾き補正用横幅 [px]
ECC_CROP_H  = 80              # ECC・フォーカスメトリクス用 crop 幅 [px]

# ===========================================================================


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def to_uint8(img: np.ndarray, vmin: float = ECC_VMIN, vmax: float = ECC_VMAX) -> np.ndarray:
    clipped    = np.clip(img, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)
    return (normalized * 255).astype(np.uint8)


def compute_ecc_warp(ref_img: np.ndarray, src_img: np.ndarray):
    """
    ECC (MOTION_TRANSLATION) でワープ行列を計算して返す。

    Parameters
    ----------
    ref_img : grid 側の z=ALIGN_Z フレーム（参照）
    src_img : focus 側の z=ALIGN_Z フレーム（補正対象）

    Returns
    -------
    warp_matrix : np.ndarray (2×3)  失敗時は None
    correlation : float             失敗時は None
    """
    ref_u8 = to_uint8(ref_img)
    src_u8 = to_uint8(src_img)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-8)
    try:
        correlation, warp_matrix = cv2.findTransformECC(
            ref_u8, src_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria
        )
        return warp_matrix, float(correlation)
    except Exception as e:
        print(f"  ECC 失敗: {e}")
        return None, None


def apply_warp(img: np.ndarray, warp_matrix: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpAffine(
        img.astype(np.float32),
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
    )


def _z_from_filename(fname: str):
    """ファイル名から z インデックスを抽出する。

    例:
        img_000000000_ph_003_phase.tif → 3
        img_000000000_ph_003.tif       → 3  (未再構成フォールバック)
    """
    m = re.search(r'_ph_(\d+)_phase\.tif$', fname, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'_ph_(\d+)\.tif$', fname, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _detect_pos_labels(focus_dir: Path):
    """Pos1 以降を返す。Pos0 は BG なので除外。"""
    candidates = [
        item for item in focus_dir.iterdir()
        if item.is_dir() and re.match(r'^Pos\d+$', item.name) and item.name != "Pos0"
    ]
    return [item.name for item in sorted(candidates, key=lambda p: int(re.search(r'\d+', p.name).group()))]


def _load_phase_stack(phase_dir: Path) -> dict:
    """output_phase/ から全 z フレームを読み込み {z_idx: ndarray} を返す。"""
    stack = {}
    for f in sorted(phase_dir.glob("*_phase.tif")):
        z = _z_from_filename(f.name)
        if z is None:
            continue
        stack[z] = tifffile.imread(str(f))
    return stack


# ---------------------------------------------------------------------------
# 再構成
# ---------------------------------------------------------------------------

def _reconstruct_one(raw_path: Path, out_dir: Path, params_dict: dict):
    """1 枚の生ホログラムを再構成して output_phase/ に保存。既存なら skip。

    batch_reconstruction_dual と同一処理:
    - bg_path が指定されていれば Pos0 を再構成して位相差分を取る
    - 中央領域の平均を引いてオフセット正規化
    """
    try:
        from PIL import Image
        from skimage.restoration import unwrap_phase
        from qpi import QPIParameters, get_field

        out_path = out_dir / (raw_path.stem + "_phase.tif")
        if out_path.exists() and not params_dict.get("force", False):
            return raw_path.name, "skip"

        pos_number = params_dict.get("pos_number", 0)
        pos_split  = params_dict.get("pos_split", 9999)
        crop = params_dict["crop_before"] if pos_number < pos_split else params_dict["crop_after"]
        y0, y1, x0, x1 = crop

        img = np.array(Image.open(str(raw_path)))
        img = img[y0:y1, x0:x1]

        params = QPIParameters(
            wavelength=params_dict["wavelength"],
            NA=params_dict["NA"],
            img_shape=img.shape,
            pixelsize=params_dict["pixelsize"],
            offaxis_center=params_dict["offaxis_center"],
        )
        field = get_field(img, params)
        angle = unwrap_phase(np.angle(field))

        # Pos0 背景引き算（batch_reconstruction_dual と同一処理）
        bg_path = params_dict.get("bg_path")
        if bg_path is not None and Path(bg_path).exists():
            bg_img = np.array(Image.open(str(bg_path)))
            bg_img = bg_img[y0:y1, x0:x1]
            field_bg = get_field(bg_img, params)
            angle_bg = unwrap_phase(np.angle(field_bg))
            angle = angle - angle_bg

            # 中央領域の平均を引いてオフセット正規化
            h, w = angle.shape
            if pos_number < pos_split:
                center_region = angle[1:h-1, 1:w//2]
            else:
                center_region = angle[1:h-1, w//2:w-1]
            if center_region.size > 0:
                angle -= np.mean(center_region)

        tifffile.imwrite(str(out_path), angle.astype(np.float32))
        return raw_path.name, "ok"
    except Exception as e:
        return raw_path.name, f"err: {e}"


def reconstruct_dir(raw_dir: Path, recon_params: dict, pos_number: int, n_workers: int = 4,
                    bg_dir: Path = None):
    """ディレクトリ内の生 TIFF を全て再構成して output_phase/ に保存。

    bg_dir が指定されていれば同名ファイルを Pos0 として引き算する（batch_reconstruction_dual と同一）。
    """
    out_dir = raw_dir / "output_phase"
    out_dir.mkdir(exist_ok=True)

    raw_files = [
        f for f in sorted(raw_dir.glob("*.tif"))
        if not f.name.startswith("._") and "output" not in f.name.lower()
    ]
    if not raw_files:
        print(f"  [!] 生 TIFF が見つかりません: {raw_dir}")
        return

    tasks = []
    for f in raw_files:
        task_params = {**recon_params, "pos_number": pos_number}
        if bg_dir is not None:
            task_params["bg_path"] = str(bg_dir / f.name)
        tasks.append((f, out_dir, task_params))
    n_ok = n_skip = n_err = 0

    if n_workers == 1:
        results = [_reconstruct_one(*t) for t in tqdm(tasks, desc=f"  recon {raw_dir.name}")]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_reconstruct_one, *t): t for t in tasks}
            results = []
            for fut in tqdm(as_completed(futures), total=len(tasks), desc=f"  recon {raw_dir.name}"):
                results.append(fut.result())

    for name, status in results:
        if status == "ok":
            n_ok += 1
        elif status == "skip":
            n_skip += 1
        else:
            n_err += 1
            print(f"  [x] {name}: {status}")

    print(f"  完了={n_ok}, スキップ={n_skip}, エラー={n_err}")


# ---------------------------------------------------------------------------
# Pos 処理
# ---------------------------------------------------------------------------

def process_pos(pos_label: str,
                focus_dir: Path,
                grid_dir: Path,
                align_z: int,
                crop_output: bool,
                output_dir: Path,
                pos_number: int = 1):
    """1 Pos の ECC アライメント + 引き算処理。

    Returns
    -------
    (z_list, diff_frames) or None（スキップ時）
    """
    focus_pos_dir = focus_dir / pos_label
    grid_pos_name = f"{pos_label}_{GRID_SUFFIX}"
    grid_pos_dir  = grid_dir / grid_pos_name

    if not focus_pos_dir.exists():
        print(f"  [!] スキップ: {focus_pos_dir} が存在しない")
        return None
    if not grid_pos_dir.exists():
        print(f"  [!] スキップ: {grid_pos_dir} が存在しない")
        return None

    focus_phase_dir = focus_pos_dir / "output_phase"
    grid_phase_dir  = grid_pos_dir  / "output_phase"

    if not focus_phase_dir.exists():
        print(f"  [!] output_phase が見つかりません: {focus_phase_dir}")
        return None
    if not grid_phase_dir.exists():
        print(f"  [!] output_phase が見つかりません: {grid_phase_dir}")
        return None

    # スタック読込
    print(f"  focus スタック読込: {focus_phase_dir}")
    focus_stack = _load_phase_stack(focus_phase_dir)
    print(f"  grid  スタック読込: {grid_phase_dir}")
    grid_stack  = _load_phase_stack(grid_phase_dir)

    if not focus_stack:
        print(f"  [!] focus phase ファイルが見つかりません")
        return None
    if not grid_stack:
        print(f"  [!] grid phase ファイルが見つかりません")
        return None

    z_list = sorted(focus_stack.keys())
    print(f"  z フレーム数: {len(z_list)}  (z={z_list[0]}..{z_list[-1]})")

    # ECC アライメント計算（ALIGN_Z フレームのみ）
    actual_align_z = align_z if align_z in focus_stack else z_list[0]
    actual_grid_z  = align_z if align_z in grid_stack  else min(grid_stack.keys())

    ref_img = grid_stack[actual_grid_z]
    src_img = focus_stack[actual_align_z]

    # channel_rois.json を読み込み（ECC + 出力クロップ共用）
    # grid の output_phase/channels/ に置かれている（focus_test 側ではなく grid 側）
    ecc_rois_path = grid_pos_dir / "output_phase" / "channels" / "channel_rois.json"
    if ecc_rois_path.exists():
        with open(ecc_rois_path) as fp:
            rois = json.load(fp)
    else:
        print(f"  channel_rois.json not found; falling back to full-frame ECC")
        rois = []

    print(f"  ECC alignment (focus z={actual_align_z}, grid z={actual_grid_z})...")

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    if rois:
        # チャンネルごとに tilt_correct → ECC → MAD 外れ値除去（compute_pos_shifts と同一パターン）
        from compute_pos_shifts import _tilt_correct, remove_outliers_mad

        fit_right = pos_number >= POS_SPLIT
        tx_list, ty_list, ch_names = [], [], []
        for ch_idx, roi_info in enumerate(rois):
            cy     = roi_info["cy"]
            cx     = roi_info["cx"]
            crop_w = roi_info.get("crop_w", 256)
            ref_crop = _tilt_correct(ref_img.astype(np.float64), cy, cx, crop_w, ECC_CROP_H, fit_right)
            src_crop = _tilt_correct(src_img.astype(np.float64), cy, cx, crop_w, ECC_CROP_H, fit_right)
            warp_ch, corr_ch = compute_ecc_warp(ref_crop, src_crop)
            if warp_ch is not None:
                tx_list.append(warp_ch[0, 2])
                ty_list.append(warp_ch[1, 2])
                ch_names.append(f"ch{ch_idx}")
                print(f"    ch{ch_idx}: corr={corr_ch:.4f}, tx={warp_ch[0,2]:.2f}, ty={warp_ch[1,2]:.2f}")

        if len(tx_list) == 0:
            print("  [!] 全チャンネルで ECC 失敗。アライメントなしで続行")
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:
            valid_mask = np.ones(len(tx_list), dtype=bool)
            if len(tx_list) >= 3:
                out_x = remove_outliers_mad(tx_list, 5.0)
                out_y = remove_outliers_mad(ty_list, 5.0)
                valid_mask = ~(out_x | out_y)
                removed = [ch_names[i] for i, v in enumerate(valid_mask) if not v]
                if removed:
                    print(f"  [!] outlier removed: {removed}")
            valid_tx = [tx_list[i] for i in range(len(tx_list)) if valid_mask[i]]
            valid_ty = [ty_list[i] for i in range(len(ty_list)) if valid_mask[i]]
            tx_mean = float(np.mean(valid_tx))
            ty_mean = float(np.mean(valid_ty))
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            warp_matrix[0, 2] = -tx_mean
            warp_matrix[1, 2] = -ty_mean
            print(f"  ECC done: tx={tx_mean:.2f} px, ty={ty_mean:.2f} px  ({len(valid_tx)}/{len(tx_list)} ch used)")
    else:
        warp_ff, corr_ff = compute_ecc_warp(ref_img, src_img)
        if warp_ff is not None:
            warp_matrix[0, 2] = -warp_ff[0, 2]
            warp_matrix[1, 2] = -warp_ff[1, 2]
            print(f"  ECC (full-frame): corr={corr_ff:.4f}, tx={warp_ff[0,2]:.2f}, ty={warp_ff[1,2]:.2f}")
        else:
            print("  [!] Full-frame ECC failed; using identity")

    rois_for_focus = rois  # フォーカス評価用に保持

    # 出力ディレクトリ
    pos_out_dir = output_dir / pos_label
    pos_out_dir.mkdir(parents=True, exist_ok=True)

    # 全 z フレーム処理
    diff_frames = []
    for z in tqdm(z_list, desc=f"  {pos_label}"):
        focus_frame = focus_stack[z]
        # grid は対応 z がなければ最近傍
        grid_z = z if z in grid_stack else min(grid_stack.keys(), key=lambda gz: abs(gz - z))
        grid_frame = grid_stack[grid_z]

        warped_focus = apply_warp(focus_frame, warp_matrix)
        diff = warped_focus - grid_frame
        diff_frames.append(diff)

        # 全体フレームを保存
        tifffile.imwrite(str(pos_out_dir / f"z{z:03d}.tif"), diff.astype(np.float32))

        # channel ROI crop を保存（tilt 補正済み crop_w × TILT_CROP_H = 40 × 270）
        if rois_for_focus:
            for ch_idx, roi_info in enumerate(rois_for_focus):
                cy     = roi_info["cy"]
                cx     = roi_info["cx"]
                crop_w = roi_info.get("crop_w", 256)
                cropped = _tilt_correct(diff.astype(np.float64), cy, cx, crop_w, TILT_CROP_H, fit_right)
                ch_dir = pos_out_dir / f"ch{ch_idx:02d}"
                ch_dir.mkdir(exist_ok=True)
                tifffile.imwrite(str(ch_dir / f"z{z:03d}.tif"), cropped.astype(np.float32))

    return z_list, diff_frames, rois_for_focus


# ---------------------------------------------------------------------------
# モンタージュ
# ---------------------------------------------------------------------------

def make_montage(z_list, diff_frames, pos_label: str):
    try:
        from figure_logger import save_figure
    except ImportError:
        print("  figure_logger が見つかりません。モンタージュをスキップ")
        return

    n = len(diff_frames)
    fig, axes = plt.subplots(1, n, figsize=(max(3 * n, 6), 3))
    if n == 1:
        axes = [axes]

    all_vals = np.concatenate([f.ravel() for f in diff_frames])
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 98))

    for ax, z, frame in zip(axes, z_list, diff_frames):
        ax.imshow(frame, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"z={z}", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"Focus check subtracted: {pos_label}", fontsize=10)
    fig.tight_layout()

    save_figure(
        fig,
        params={"pos": pos_label, "align_z": ALIGN_Z, "n_z": n,
                "focus_dir": FOCUS_DIR, "grid_dir": GRID_DIR},
        description=f"Focus check subtracted montage: {pos_label}",
    )
    plt.close(fig)
    print(f"  モンタージュ保存完了")


# ---------------------------------------------------------------------------
# フォーカス検出
# ---------------------------------------------------------------------------

def _focus_metrics(frames: list) -> tuple:
    """フレームリストから (lap_vars, stds) を計算して返す。"""
    lap_vars, stds = [], []
    for frame in frames:
        f32 = frame.astype(np.float32)
        lap = cv2.Laplacian(f32, cv2.CV_32F)
        lap_vars.append(float(np.var(lap)))
        stds.append(float(np.std(f32)))
    return lap_vars, stds


def _best_z_from_metrics(z_list, lap_vars, stds):
    lap_rank = np.argsort(np.argsort(lap_vars))
    std_rank  = np.argsort(np.argsort(stds))
    combined  = (lap_rank + std_rank).astype(float)
    return z_list[int(np.argmin(combined))], combined


def find_best_focus_z(z_list: list, diff_frames: list, pos_label: str,
                      rois: list = None, pos_number: int = 1) -> dict:
    """background引き算済み位相フレームから最良フォーカスzをチャンネル別に検出する。

    rois が指定された場合はチャンネルROIごとに評価し、指定がなければ全体フレームで評価する。

    指標: Laplacian分散（エッジの鮮鋭さ）+ 標準偏差（位相コントラスト量）の平均ランク。

    Returns
    -------
    results : dict   {"ch0": best_z, "ch1": best_z, ...}  または {"full": best_z}
    """
    try:
        from figure_logger import save_figure
    except ImportError:
        save_figure = None

    from compute_pos_shifts import _tilt_correct

    # チャンネルごとのフレームリストを作成（compute_pos_shifts と同一パターン）
    if rois:
        fit_right = pos_number >= POS_SPLIT
        channels = {}
        for ch_idx, roi_info in enumerate(rois):
            cy     = roi_info["cy"]
            cx     = roi_info["cx"]
            crop_w = roi_info.get("crop_w", 256)
            channels[f"ch{ch_idx}"] = [
                _tilt_correct(f.astype(np.float64), cy, cx, crop_w, ECC_CROP_H, fit_right)
                for f in diff_frames
            ]
    else:
        channels = {"full": diff_frames}

    results = {}
    all_metrics = {}  # プロット用

    print(f"\n  === フォーカス検出結果: {pos_label} ===")
    for ch_name, frames in channels.items():
        lap_vars, stds = _focus_metrics(frames)
        best_z, combined = _best_z_from_metrics(z_list, lap_vars, stds)
        results[ch_name] = best_z
        all_metrics[ch_name] = {"lap_vars": lap_vars, "stds": stds, "combined": combined}

        print(f"\n  [{ch_name}]")
        header = f"{'z':>4}  {'Lap分散':>12}  {'std':>10}  {'スコア':>6}"
        print(f"  {header}")
        best_idx = z_list.index(best_z)
        for i, z in enumerate(z_list):
            marker = " <-- best" if i == best_idx else ""
            print(f"  {z:>4}  {lap_vars[i]:>12.4f}  {stds[i]:>10.4f}  {combined[i]:>6.0f}{marker}")
        print(f"  => best z = {best_z}")

    # フォーカスカーブをプロット（チャンネル数 × 2 パネル）
    if save_figure is not None:
        n_ch = len(channels)
        fig, axes = plt.subplots(n_ch, 2, figsize=(9, 2.2 * n_ch), squeeze=False)

        # 列ごとの共通 ylim を事前計算
        all_lap = [v for m in all_metrics.values() for v in m["lap_vars"]]
        all_std = [v for m in all_metrics.values() for v in m["stds"]]
        margin = 0.05
        lap_ylim = (min(all_lap) * (1 - margin), max(all_lap) * (1 + margin))
        std_ylim = (min(all_std) * (1 - margin), max(all_std) * (1 + margin))
        col_ylims = [lap_ylim, std_ylim]

        for row, (ch_name, metrics) in enumerate(all_metrics.items()):
            best_z = results[ch_name]
            for col, (vals, ylabel) in enumerate([
                (metrics["lap_vars"], "Laplacian variance"),
                (metrics["stds"],     "Std dev (phase)"),
            ]):
                ax = axes[row][col]
                ax.plot(z_list, vals, marker="o", linewidth=1.5, markersize=5, color="steelblue")
                ax.axvline(best_z, color="tomato", linestyle="--", linewidth=1.5,
                           label=f"best z={best_z}")
                ax.set_xlabel("z index", fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_title(ch_name, fontsize=10)
                ax.set_ylim(col_ylims[col])
                ax.tick_params(direction="in", labelsize=10)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.legend(fontsize=9, frameon=False)
        fig.suptitle(f"Focus curve: {pos_label}", fontsize=12)
        fig.tight_layout()
        data_dict = {"z_list": np.array(z_list)}
        for ch_name, metrics in all_metrics.items():
            data_dict[f"{ch_name}_lap_vars"] = np.array(metrics["lap_vars"])
            data_dict[f"{ch_name}_stds"]     = np.array(metrics["stds"])
        save_figure(
            fig,
            params={"pos": pos_label, "best_z_per_ch": results,
                    "focus_dir": FOCUS_DIR, "grid_dir": GRID_DIR},
            description=f"Focus curve per channel: {pos_label}, best_z={results}",
            data=data_dict,
        )
        plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    focus_dir  = Path(FOCUS_DIR)
    grid_dir   = Path(GRID_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pos ラベル検出
    pos_labels = POS_LABELS if POS_LABELS is not None else _detect_pos_labels(focus_dir)
    if not pos_labels:
        print("[x] Pos ディレクトリが見つかりません")
        return
    print(f"対象 Pos: {pos_labels}")

    # 再構成フェーズ（DO_RECONSTRUCTION=True の場合）
    if DO_RECONSTRUCTION:
        print("\n=== 再構成フェーズ ===")
        recon_params = {
            "wavelength":     WAVELENGTH,
            "NA":             NA,
            "pixelsize":      PIXELSIZE,
            "offaxis_center": OFFAXIS_CENTER,
            "pos_split":      POS_SPLIT,
            "crop_before":    CROP_BEFORE,
            "crop_after":     CROP_AFTER,
            "force":          FORCE_RECONSTRUCT,
        }
        focus_bg_dir = focus_dir / "Pos0"
        grid_bg_dir  = grid_dir / f"Pos0_{GRID_SUFFIX}"
        if focus_bg_dir.exists():
            print(f"  Pos0 BG (focus): {focus_bg_dir}")
        else:
            print(f"  [!] Pos0 BG が見つかりません（引き算なし）: {focus_bg_dir}")
            focus_bg_dir = None
        if grid_bg_dir.exists():
            print(f"  Pos0 BG (grid ): {grid_bg_dir}")
        else:
            print(f"  [!] Pos0 BG が見つかりません（引き算なし）: {grid_bg_dir}")
            grid_bg_dir = None

        for pos_label in pos_labels:
            pos_number = int(re.search(r'\d+', pos_label).group())
            crop_label = "BEFORE" if pos_number < POS_SPLIT else "AFTER"
            print(f"\n[focus] {pos_label}  (pos={pos_number}, crop={crop_label})")
            reconstruct_dir(focus_dir / pos_label, recon_params, pos_number, N_WORKERS,
                            bg_dir=focus_bg_dir)
            grid_pos_name = f"{pos_label}_{GRID_SUFFIX}"
            print(f"\n[grid ] {grid_pos_name}")
            reconstruct_dir(grid_dir / grid_pos_name, recon_params, pos_number, N_WORKERS,
                            bg_dir=grid_bg_dir)

    # 引き算フェーズ
    print("\n=== 引き算フェーズ ===")
    for pos_label in pos_labels:
        print(f"\n[{pos_label}]")
        pos_number = int(re.search(r'\d+', pos_label).group())
        result = process_pos(pos_label, focus_dir, grid_dir, ALIGN_Z, CROP_OUTPUT, output_dir,
                             pos_number=pos_number)
        if result is None:
            continue
        z_list, diff_frames, rois = result

        # フォーカス検出（全 Pos、チャンネル別）
        best_z_per_ch = find_best_focus_z(z_list, diff_frames, pos_label, rois=rois,
                                          pos_number=pos_number)

        # モンタージュは先頭 Pos のみ
        if pos_label == pos_labels[0]:
            print("  モンタージュ生成...")
            make_montage(z_list, diff_frames, pos_label)

    print(f"\n完了: {output_dir}")


if __name__ == "__main__":
    main()


# %%
