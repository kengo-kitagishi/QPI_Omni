# %%
"""
pipeline_full.py
----------------
グリッド再構成 → タイムラプス再構成 → channel_crop → backsub
→ compute_pos_shifts → grid_subtract → shift_visualize
を一括実行する統合パイプライン。

実行順序:
  [Step 0] Grid reconstruction       : GRID_DIR の全 PosX_x*_y* を再構成
  [Step 1] Timelapse reconstruction  : 各 TIMELAPSE_DIR の全 PosX を再構成
  各 Pos に対して:
    [Step 2] channel_crop            : output_phase/ → output_phase/channels/
    [Step 3] gaussian_backsub        : channel_*.tif → channel_*_bg_corr.tif
    [Step 4] align_simple            : 確認用（オプション）
    [Step 5] compute_pos_shifts      : pos_shifts.json + shift_visualize
    [Step 6] grid_subtract           : grid_subtracted/ スタック生成
"""

import re
import sys
import os
import json
import importlib.util
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

# ============================================================
# ★ データパス
# ============================================================
TIMELAPSE_DIRS = [
    r"D:\AquisitionData\Kitagishi\260310\timelapse_11day_exp200ms_1pos_EMM2",
]
GRID_DIR = r"D:\AquisitionData\Kitagishi\260310\grid_0p5_0p5_0p1_exp200ms_1pos_EMM2_1"


# ============================================================
# ★ 実行ステップ（False でスキップ）
# ============================================================
STEP_GRID_RECONSTRUCTION     = False
STEP_TIMELAPSE_RECONSTRUCTION = False
STEP_CALIBRATE_GRID          = False   # True で各Posのgridキャリブレーションを実行
STEP_CHANNEL_CROP            = True   # 完了済み
STEP_GAUSSIAN_GRADIENT       = False   # NEW: large-sigma Gaussian gradient removal before channel_crop
STEP_GAUSSIAN_BACKSUB        = True   # 完了済み
STEP_ALIGN_SIMPLE            = False   # 確認用（時間がかかる）
STEP_COMPUTE_SHIFTS          = True    # shift_visualize も自動実行
STEP_GRID_SUBTRACT           = True

# テストラン: None で全フレーム、整数で先頭 N フレームのみ処理
TEST_N_FRAMES                = None

# 並列処理ワーカー数（None = cpu_count(), 1 = 逐次実行でデバッグ向き）
N_WORKERS_GRID = None   # Step 0 grid reconstruction 用
N_WORKERS_TL   = None   # Step 1 timelapse reconstruction 用

# ============================================================
# Pos フィルタ（タイムラプス側）
# ============================================================
# None で全 Pos。["Pos1", "Pos3"] のように指定も可
POS_FILTER = None

# ============================================================
# QPI 光学パラメータ
# ============================================================
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

# ============================================================
# Pos 番号によるクロップ切り替え（grid / timelapse 共通）
# ============================================================
POS_SPLIT   = 15
CROP_BEFORE = (0, 2048, 400, 2448)   # pos_number < POS_SPLIT → 右側 (400:2448)  センサー幅2448
CROP_AFTER  = (0, 2048,   0, 2048)   # pos_number >= POS_SPLIT → 左側 (0:2048)
# ※ BG（Pos0）はターゲットのpos_numberで決まるcropを使う（常に右ではない）

# ============================================================
# 再構成パラメータ
# ============================================================
# グリッド
GRID_BG_BASE_LABEL        = "Pos0"   # BG として使うグリッドの base_label
GRID_TARGET_BASE_LABELS   = None  # None で Pos0 以外を全処理
GRID_SKIP_IF_EXISTS       = False
GRID_MEAN_REGION          = None     # (r1, r2, c1, c2) or None

# タイムラプス
TL_BG_LABEL               = "Pos0"  # BG フォルダ名
TL_RAW_PATTERN            = "img_*_ph_000.tif"   # 生画像パターン
TL_SKIP_IF_EXISTS         = True
TL_MEAN_REGION            = None     # (r1, r2, c1, c2) or None

# ============================================================
# Gaussian gradient removal パラメータ（STEP_GAUSSIAN_GRADIENT=True 時のみ）
# ============================================================
# sigma はチャネル crop サイズ（CROP_W=40, CROP_H=120）より十分大きくすること。
# alignment 用: sigma >= 150 推奨。可視化用: 50〜100 でも可。
GRADIENT_SIGMA            = 150      # pixels; large-scale gradient removal
# False → *_phase_bgsub.tif として保存（元ファイルを保持）
# True  → *_phase.tif を上書き（元ファイルが消える・Step1から再生成可能）
GRADIENT_INPLACE          = False

# ============================================================
# channel_crop パラメータ
# ============================================================
CROP_DETECT               = True     # channel_rois.json がなければ自動 detect
CROP_FORCE_RECOMPUTE      = True    # True にすると channels/ を丸ごと削除して最初から再計算
CROP_FORCE_DETECT         = True    # True にすると既存 channel_rois.json のみ削除して再検出
CROP_APPLY                = True
# STEP_GAUSSIAN_GRADIENT=True かつ GRADIENT_INPLACE=False のとき自動で *_phase_bgsub.tif に切り替わる
CROP_IMG_PATTERN          = "*_phase.tif"  # output_phase/ 内の位相画像パターン
CROP_W                    = 40
CROP_H                    = 120
CROP_MIN_DIST             = 35
CROP_PROMINENCE           = 0.3
CROP_X_START              = 40
CROP_X_END                = 480

# ============================================================
# gaussian_backsub パラメータ
# ============================================================
BACKSUB_MIN_PHASE         = -1.1
BACKSUB_HIST_MIN          = -1.1
BACKSUB_HIST_MAX          =  1.5
BACKSUB_N_BINS            = 512
BACKSUB_SMOOTH_WINDOW     = 20
BACKSUB_SAVE_PNG          = False
BACKSUB_PNG_DPI           = 150

# ============================================================
# align_and_subtract_simple パラメータ（STEP_ALIGN_SIMPLE=True 時のみ）
# ============================================================
ALIGN_REFERENCE_FRAME     = 150
ALIGN_METHOD              = 'ecc'
ALIGN_SAVE_PNG            = True
ALIGN_PNG_SAMPLE          = 5
ALIGN_VMIN                = -0.1
ALIGN_VMAX                =  1.7

# ============================================================
# compute_pos_shifts パラメータ
# ============================================================
SHIFTS_CHANNEL_PATTERN    = "channel_*_bg_corr.tif"
SHIFTS_USE_GRID_REFERENCE = True     # False → タイムラプスの SHIFTS_REFERENCE_FRAME 基準
SHIFTS_REFERENCE_FRAME    = 150      # USE_GRID_REFERENCE=False 時のみ使用
SHIFTS_GRID_Z_INDEX       = 5        # グリッド基準画像の z 番号
SHIFTS_METHOD             = 'ecc'
SHIFTS_VMIN               = -5.0
SHIFTS_VMAX               =  2.0
SHIFTS_OUTLIER_MAD_THRESH = 3.0
SHIFTS_TIMESERIES_WINDOW  = 11
SHIFTS_TIMESERIES_THRESH  = 2.0
SHIFTS_ECC_MIN_CORR       = 0.9   # ECC スコアがこれ未満のチャネルを除外
SHIFTS_APPLY_BACKSUB_TO_GRID_REF = True  # グリッド基準画像にも gaussian_backsub を適用
SHIFTS_USE_INCREMENTAL_TRACKING  = True   # 逐次追跡モード
SHIFTS_X_STEP                    = 0.1   # グリッドステップ [μm]（GSUB_X_STEP と同値）
SHIFTS_Y_STEP                    = 0.1   # [μm]
SHIFTS_SHIFT_SIGN_X              = 1
SHIFTS_SHIFT_SIGN_Y              = 1
SHIFTS_JUMP_THRESH_UM            = 1.0   # 前フレームとのシフト差 [μm] を超えたら外れ値
# calibrate_grid_positions.py の出力 JSON (None で名目値を使用)
GRID_CALIBRATION_JSON            = None   # 例: r"E:\...\grid_calibration_Pos1.json"
# True → GRID_DIR/grid_calibration_{Pos}.json が存在すれば自動使用
# False → GRID_CALIBRATION_JSON の値のみ使用（JSON が存在しても無視）
SHIFTS_USE_PER_POS_CALIBRATION   = False

# ============================================================
# 2段階ECC パラメータ（SHIFTS_USE_INCREMENTAL_TRACKING=True 時のみ有効）
# ============================================================
SHIFTS_USE_SECOND_PASS_ECC = True    # True で2段階ECC有効化
SHIFTS_FIRST_PASS_HALF     = True   # True で1回目ECCもhalf cropで実施
# None → pos_number < POS_SPLIT なら 'right'、>= POS_SPLIT なら 'left'
SHIFTS_SECOND_PASS_HALF    = None
SHIFTS_USE_THIRD_PASS_ECC  = True   # True で3段階ECC有効化（pass2結果から最近傍grid再選択）

# ============================================================
# grid_subtract パラメータ
# ============================================================
GSUB_Z_INDEX              = 5        # グリッド画像のz番号（SHIFTS_GRID_Z_INDEX と同じでよい）
GSUB_X_STEP               = 0.1     # μm
GSUB_Y_STEP               = 0.1     # μm
GSUB_SENSOR_PIXEL_SIZE    = 3.45e-6
GSUB_MAGNIFICATION        = 40
GSUB_ORIGINAL_DIM         = 2048
GSUB_RECONSTRUCTED_DIM    = 511
GSUB_SHIFT_SIGN_X              = 1
GSUB_SHIFT_SIGN_Y              = 1
GSUB_APPLY_INVERSE_SHIFT       = False
GSUB_APPLY_BACKSUB_TO_GRID     = True   # グリッド画像に backsub を適用
GSUB_APPLY_SUBPIXEL_CORRECTION = True   # サブピクセル残差 warp をフレームに適用
# ============================================================


# -----------------------------------------------------------
# 共通ユーティリティ
# -----------------------------------------------------------
def _banner(msg):
    print(f"\n{'='*64}\n  {msg}\n{'='*64}")

def _section(msg):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")

def get_crop(pos_number: int):
    return CROP_BEFORE if pos_number < POS_SPLIT else CROP_AFTER

def pos_number_from_label(label: str) -> int:
    m = re.match(r"Pos(\d+)", label)
    return int(m.group(1)) if m else 0

def _load_backsub_module():
    backsub_path = _script_dir / "19_gaussian_backsub.py"
    spec = importlib.util.spec_from_file_location("gaussian_backsub", backsub_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.minPhase      = BACKSUB_MIN_PHASE
    mod.hist_min      = BACKSUB_HIST_MIN
    mod.hist_max      = BACKSUB_HIST_MAX
    mod.n_bins        = BACKSUB_N_BINS
    mod.smooth_window = BACKSUB_SMOOTH_WINDOW
    return mod


# -----------------------------------------------------------
# QPI 再構成コア（grid / timelapse 共用）
# -----------------------------------------------------------
def _make_qpi_params(img_path: Path, crop):
    from PIL import Image
    from qpi import QPIParameters
    img = np.array(Image.open(str(img_path)))
    rs, re_, cs, ce = crop
    cropped = img[rs:re_, cs:ce]
    return QPIParameters(
        wavelength=WAVELENGTH, NA=NA,
        img_shape=cropped.shape, pixelsize=PIXELSIZE,
        offaxis_center=OFFAXIS_CENTER,
    )

def _reconstruct(img_path: Path, qpi_params, crop) -> np.ndarray:
    from PIL import Image
    from qpi import get_field
    from skimage.restoration import unwrap_phase
    img = np.array(Image.open(str(img_path)))
    rs, re_, cs, ce = crop
    img = img[rs:re_, cs:ce]
    field = get_field(img, qpi_params)
    return unwrap_phase(np.angle(field))

def _get_z_index(path: Path) -> int:
    m = re.search(r"_ph_(\d+)", path.stem)
    return int(m.group(1)) if m else -1

def _scan_grid_folders(grid_dir: Path):
    pattern = re.compile(r"^(.+)_x([+-]?\d+)_y([+-]?\d+)$")
    result = {}
    for d in sorted(grid_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            base = m.group(1)
            xi, yi = int(m.group(2)), int(m.group(3))
            result.setdefault(base, {})[(xi, yi)] = d
    return result


def _worker_grid_point(args):
    """step_grid_reconstruction のワーカー: 1グリッドポイントを再構成して保存。"""
    xi, yi, tgt_dir_str, bg_dir_str, crop, pos_num = args
    tgt_dir = Path(tgt_dir_str)
    bg_dir  = Path(bg_dir_str)
    out_dir = tgt_dir / "output_phase"

    z_tgt = {_get_z_index(p): p for p in sorted(tgt_dir.glob("img_*_ph_*.tif"))}
    z_bg  = {_get_z_index(p): p for p in sorted(bg_dir.glob("img_*_ph_*.tif"))}
    if not z_tgt:
        return xi, yi, False, "z画像なし"

    out_dir.mkdir(exist_ok=True)
    sample = next(iter(z_tgt.values()))
    try:
        qpi = _make_qpi_params(sample, crop)
    except Exception as e:
        return xi, yi, False, f"QPIParams: {e}"

    folder_ok = True
    for z_idx, tgt_path in sorted(z_tgt.items()):
        out_path = out_dir / (tgt_path.stem + "_phase.tif")
        if GRID_SKIP_IF_EXISTS and out_path.exists():
            continue
        if z_idx not in z_bg:
            folder_ok = False
            continue
        try:
            phase = _reconstruct(tgt_path, qpi, crop) - _reconstruct(z_bg[z_idx], qpi, crop)
            h, w = phase.shape
            if pos_num < POS_SPLIT:
                region = phase[1:h-1, 1:w//2]
            else:
                region = phase[1:h-1, w//2:w-1]
            if region.size > 0:
                phase -= np.mean(region)
            tifffile.imwrite(str(out_path), phase.astype(np.float32))
        except Exception as e:
            folder_ok = False
    return xi, yi, folder_ok, None


def _worker_tl_frame(args):
    """step_timelapse_reconstruction のワーカー: 1フレームを再構成して保存。"""
    tif_str, bg_str, out_str, crop, pos_num, qpi, skip_if_exists = args
    tif_path = Path(tif_str)
    bg_path  = Path(bg_str)
    out_path = Path(out_str)

    if skip_if_exists and out_path.exists():
        return out_str, "skip", None

    try:
        phase = _reconstruct(tif_path, qpi, crop) - _reconstruct(bg_path, qpi, crop)
        h, w  = phase.shape
        if pos_num < POS_SPLIT:
            region = phase[1:h-1, 1:w//2]
        else:
            region = phase[1:h-1, w//2:w-1]
        if region.size > 0:
            phase -= np.mean(region)
        tifffile.imwrite(str(out_path), phase.astype(np.float32))
        return out_str, "ok", None
    except Exception as e:
        return out_str, "err", str(e)


# ===========================================================
# Step 0: Grid reconstruction
# ===========================================================
def step_grid_reconstruction():
    _banner("Step 0: Grid reconstruction")
    grid_dir = Path(GRID_DIR)
    if not grid_dir.exists():
        print(f"  [ERROR] GRID_DIR が見つかりません: {grid_dir}")
        return False

    folders = _scan_grid_folders(grid_dir)
    if GRID_BG_BASE_LABEL not in folders:
        print(f"  [ERROR] BG フォルダ '{GRID_BG_BASE_LABEL}_x*_y*' が見つかりません")
        return False

    bg_map = folders[GRID_BG_BASE_LABEL]
    if GRID_TARGET_BASE_LABELS is not None:
        target_labels = GRID_TARGET_BASE_LABELS
    else:
        target_labels = [k for k in sorted(folders) if k != GRID_BG_BASE_LABEL]

    print(f"  BG: {GRID_BG_BASE_LABEL}  対象: {target_labels}")
    ok = skip = err = 0

    for base_label in target_labels:
        if base_label not in folders:
            print(f"  [WARN] {base_label}_x*_y* なし、スキップ")
            continue
        target_map = folders[base_label]
        pos_num = pos_number_from_label(base_label)
        crop = get_crop(pos_num)
        print(f"\n  {base_label}  crop={crop}")

        # タスクリスト構築
        tasks = []
        for (xi, yi) in sorted(target_map):
            tgt_dir = target_map[(xi, yi)]
            out_dir = tgt_dir / "output_phase"
            if GRID_SKIP_IF_EXISTS and out_dir.exists() and any(out_dir.glob("*.tif")):
                skip += 1
                continue
            if (xi, yi) not in bg_map:
                print(f"  [WARN] BG {GRID_BG_BASE_LABEL}_x{xi:+d}_y{yi:+d} なし")
                err += 1
                continue
            bg_d = bg_map[(xi, yi)]
            z_tgt = {_get_z_index(p): p for p in sorted(tgt_dir.glob("img_*_ph_*.tif"))}
            if not z_tgt:
                err += 1
                continue
            tasks.append((xi, yi, str(tgt_dir), str(bg_d), crop, pos_num))

        if tasks:
            n_workers_display = N_WORKERS_GRID if N_WORKERS_GRID is not None else os.cpu_count()
            print(f"  並列処理: {len(tasks)} ポイント / {n_workers_display} ワーカー")
            if N_WORKERS_GRID == 1:
                results = [_worker_grid_point(a) for a in tqdm(tasks, desc=base_label)]
            else:
                results = []
                with ProcessPoolExecutor(max_workers=N_WORKERS_GRID) as executor:
                    futures = {executor.submit(_worker_grid_point, a): a for a in tasks}
                    for fut in tqdm(as_completed(futures), total=len(tasks), desc=base_label):
                        results.append(fut.result())
            for xi, yi, folder_ok, err_msg in results:
                if folder_ok:
                    ok += 1
                else:
                    err += 1
                    if err_msg:
                        print(f"  [ERR] ({xi:+d},{yi:+d}): {err_msg}")

    print(f"\n  Grid reconstruction 完了: 成功={ok}, スキップ={skip}, エラー={err}")
    return True


# ===========================================================
# Step 1: Timelapse reconstruction
# ===========================================================
def step_timelapse_reconstruction(tl_dir: Path):
    _banner(f"Step 1: Timelapse reconstruction  ({tl_dir.name})")

    bg_dir = tl_dir / TL_BG_LABEL
    if not bg_dir.exists():
        print(f"  [ERROR] BG フォルダが見つかりません: {bg_dir}")
        return []

    # BG ファイル辞書（filename → path）
    bg_files = {p.name: p for p in sorted(bg_dir.glob("*.tif"))
                if not p.name.startswith("._")}
    if not bg_files:
        print(f"  [ERROR] BG 画像が見つかりません: {bg_dir}")
        return []
    bg_fallback = next(iter(bg_files.values()))  # マッチしない場合の fallback

    # 処理対象 PosX（Pos0 除く）
    pos_dirs = sorted([d for d in tl_dir.iterdir()
                       if d.is_dir() and re.match(r"^Pos\d+$", d.name)
                       and d.name != TL_BG_LABEL])
    if POS_FILTER:
        pos_dirs = [d for d in pos_dirs if d.name in POS_FILTER]

    processed = []
    for pos_dir in pos_dirs:
        pos_num = pos_number_from_label(pos_dir.name)
        crop = get_crop(pos_num)
        out_dir = pos_dir / "output_phase"

        tif_files = sorted([p for p in pos_dir.glob("*.tif")
                            if not p.name.startswith("._")])
        if not tif_files:
            print(f"  [SKIP] {pos_dir.name}: tif なし")
            continue

        # スキップ判定（全フレーム再構成済みか）
        if TL_SKIP_IF_EXISTS and out_dir.exists():
            done = set(p.name for p in out_dir.glob("*.tif"))
            if all((p.stem + "_phase.tif") in done for p in tif_files):
                print(f"  [SKIP already] {pos_dir.name}")
                processed.append(pos_dir)
                continue

        out_dir.mkdir(exist_ok=True)
        print(f"\n  {pos_dir.name}  crop={crop}  ({len(tif_files)} フレーム)")

        # QPIParams（最初のフレームから作成）
        try:
            qpi = _make_qpi_params(tif_files[0], crop)
        except Exception as e:
            print(f"  [ERR] QPIParams {pos_dir.name}: {e}")
            continue

        bg_is_timelapse = len(bg_files) > 1

        # タスクリスト構築（BG path を各フレームに対応させる）
        tasks = []
        for tif_path in tif_files:
            out_path = out_dir / (tif_path.stem + "_phase.tif")
            bg_path = bg_files.get(tif_path.name, bg_fallback) if bg_is_timelapse else bg_fallback
            tasks.append((str(tif_path), str(bg_path), str(out_path), crop, pos_num, qpi, TL_SKIP_IF_EXISTS))

        n_ok = n_skip = n_err = 0
        if N_WORKERS_TL == 1:
            raw_results = [_worker_tl_frame(a) for a in tqdm(tasks, desc=f"  {pos_dir.name}")]
        else:
            raw_results = []
            with ProcessPoolExecutor(max_workers=N_WORKERS_TL) as executor:
                futures = {executor.submit(_worker_tl_frame, a): a for a in tasks}
                for fut in tqdm(as_completed(futures), total=len(tasks), desc=f"  {pos_dir.name}"):
                    raw_results.append(fut.result())

        for _, status, err_msg in raw_results:
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                if err_msg:
                    print(f"\n  [ERR] {err_msg}")

        print(f"  {pos_dir.name}: 完了={n_ok}, スキップ={n_skip}, エラー={n_err}")
        processed.append(pos_dir)

    return processed


# ===========================================================
# Step 1.5: Gaussian gradient removal
# ===========================================================
def step_gaussian_gradient(phase_dir: Path) -> str:
    """
    output_phase/*_phase.tif に大きな sigma のガウスブラーを引いて
    空間的バックグラウンドグラジェントを除去する。

    GRADIENT_INPLACE=False のとき *_phase_bgsub.tif として保存し、
    channel_crop が使うべきパターン文字列を返す。
    GRADIENT_INPLACE=True のとき *_phase.tif を上書きし、元パターンを返す。
    """
    from scipy.ndimage import gaussian_filter

    suffix = "_bgsub" if not GRADIENT_INPLACE else ""
    img_pattern = "*_phase.tif"
    tif_files = sorted(phase_dir.glob(img_pattern))
    if not tif_files:
        print(f"  [SKIP] *_phase.tif が見つかりません: {phase_dir}")
        return CROP_IMG_PATTERN

    already = 0
    processed = 0
    for path in tif_files:
        if GRADIENT_INPLACE:
            out_path = path
        else:
            out_path = phase_dir / (path.stem + "_bgsub.tif")
            if out_path.exists():
                already += 1
                continue
        img = tifffile.imread(str(path)).astype(np.float32)
        bg  = gaussian_filter(img, sigma=GRADIENT_SIGMA, mode='nearest')
        tifffile.imwrite(str(out_path), (img - bg).astype(np.float32))
        processed += 1

    print(f"  gaussian gradient removal: {processed} 処理, {already} スキップ  "
          f"(sigma={GRADIENT_SIGMA}px, inplace={GRADIENT_INPLACE})")

    return "*_phase_bgsub.tif" if not GRADIENT_INPLACE else "*_phase.tif"


# ===========================================================
# Step 2: channel_crop
# ===========================================================
def step_channel_crop(phase_dir: Path, img_pattern: str = None) -> bool:
    from channel_crop import run_detect, run_apply

    img_pattern = img_pattern or CROP_IMG_PATTERN
    img_files = sorted(phase_dir.glob(img_pattern))
    if not img_files:
        print(f"  [SKIP] {CROP_IMG_PATTERN} が見つかりません: {phase_dir}")
        return False

    out_dir = phase_dir / "channels"
    if CROP_FORCE_RECOMPUTE and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
        print(f"  [force] channels/ を削除して再計算")
    out_dir.mkdir(exist_ok=True)
    roi_path = out_dir / "channel_rois.json"

    if CROP_FORCE_DETECT and roi_path.exists():
        roi_path.unlink()

    rois = None
    if CROP_DETECT and not roi_path.exists():
        print(f"  [detect] {img_files[0].name}")
        rois = run_detect(
            img_files[0], CROP_W, CROP_H, out_dir,
            min_dist=CROP_MIN_DIST, prominence_sigma=CROP_PROMINENCE,
            x_start=CROP_X_START, x_end=CROP_X_END,
        )
    elif not roi_path.exists():
        print(f"  [ERROR] channel_rois.json なし・detect もスキップ")
        return False

    if CROP_APPLY:
        if rois is None:
            with open(roi_path, encoding="utf-8") as f:
                rois = json.load(f)
        print(f"  [apply] {len(img_files)} フレーム × {len(rois)} チャネル")
        run_apply(phase_dir, img_pattern, rois, out_dir)

    return True


# ===========================================================
# Step 3: gaussian_backsub
# ===========================================================
def step_gaussian_backsub(channels_dir: Path):
    backsub = _load_backsub_module()
    stack_files = [p for p in sorted(channels_dir.glob("channel_*.tif"))
                   if "_bg_corr" not in p.name]
    if not stack_files:
        print(f"  [SKIP] channel_*.tif が見つかりません: {channels_dir}")
        return

    for tif_path in stack_files:
        out_path = channels_dir / f"{tif_path.stem}_bg_corr.tif"
        if out_path.exists():
            print(f"  [SKIP already] {tif_path.name}")
            continue
        print(f"  処理: {tif_path.name}")
        result = backsub.process_image(tif_path, channels_dir, save_png_data=BACKSUB_SAVE_PNG)
        if BACKSUB_SAVE_PNG and isinstance(result, dict):
            backsub.save_png_plots(result, channels_dir, BACKSUB_PNG_DPI)


# ===========================================================
# Step 4: align_and_subtract_simple（確認用）
# ===========================================================
def step_align_simple(channels_dir: Path):
    from align_and_subtract_simple import process_timelapse
    import tempfile

    stack_files = sorted(channels_dir.glob(SHIFTS_CHANNEL_PATTERN))
    if not stack_files:
        print(f"  [SKIP] {SHIFTS_CHANNEL_PATTERN} が見つかりません")
        return

    for stack_path in stack_files:
        arr = tifffile.imread(str(stack_path))
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        ref_idx = min(ALIGN_REFERENCE_FRAME - 1, arr.shape[0] - 1)
        ref_img = arr[ref_idx].astype(np.float64)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frame_paths = []
            for i in range(arr.shape[0]):
                p = tmpdir / f"frame_{i:06d}.tif"
                tifffile.imwrite(str(p), arr[i].astype(np.float32))
                frame_paths.append(str(p))
            process_timelapse(
                str(channels_dir / stack_path.stem), ref_img, frame_paths,
                method=ALIGN_METHOD, save_png=ALIGN_SAVE_PNG,
                vmin=ALIGN_VMIN, vmax=ALIGN_VMAX,
                png_dpi=150, png_sample_interval=ALIGN_PNG_SAMPLE,
            )


# ===========================================================
# Step -1: calibrate_grid_positions（各Pos用グリッドキャリブレーション）
# ===========================================================
def step_calibrate_grid():
    _banner("Step -1: calibrate_grid_positions")

    # calibrate_grid_positions モジュールを importlib でロード
    cgp_path = _script_dir / "calibrate_grid_positions.py"
    spec = importlib.util.spec_from_file_location("calibrate_grid_positions", cgp_path)
    cgp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cgp)

    # 各 TIMELAPSE_DIR から Pos ラベルと channel_rois.json パスを収集
    pos_rois_map = {}   # label → channel_rois.json path（最初に見つかったものを使用）
    for tl_dir_str in TIMELAPSE_DIRS:
        tl_dir = Path(tl_dir_str)
        if not tl_dir.exists():
            continue
        pos_dirs = sorted([d for d in tl_dir.iterdir()
                           if d.is_dir() and re.match(r"^Pos\d+$", d.name)
                           and d.name != TL_BG_LABEL])
        if POS_FILTER:
            pos_dirs = [d for d in pos_dirs if d.name in POS_FILTER]
        for pos_dir in pos_dirs:
            label = pos_dir.name
            rois_json = pos_dir / "output_phase" / "channels" / "channel_rois.json"
            if label not in pos_rois_map and rois_json.exists():
                pos_rois_map[label] = rois_json

    if not pos_rois_map:
        print("  [ERROR] channel_rois.json が見つかりません（先に channel_crop を実行してください）")
        return

    for label in sorted(pos_rois_map):
        out_path = Path(GRID_DIR) / f"grid_calibration_{label}.json"
        if out_path.exists():
            print(f"  [SKIP already] {label}: {out_path.name}")
            continue
        _section(f"キャリブレーション: {label}")
        cgp.GRID_DIR          = GRID_DIR
        cgp.BASE_LABEL        = label
        cgp.GRID_Z_INDEX      = SHIFTS_GRID_Z_INDEX
        cgp.CHANNEL_ROIS_JSON = str(pos_rois_map[label])
        cgp.OUTPUT_JSON       = str(out_path)
        cgp.VMIN              = SHIFTS_VMIN
        cgp.VMAX              = SHIFTS_VMAX
        try:
            cgp.main()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [ERROR] {label}: {e}")


# ===========================================================
# Step 5: compute_pos_shifts（shift_visualize 含む）
# ===========================================================
def step_compute_shifts(channels_dir: Path, base_label: str):
    import compute_pos_shifts as cps
    cps.CHANNELS_DIR              = str(channels_dir)
    cps.CHANNEL_PATTERN           = SHIFTS_CHANNEL_PATTERN
    cps.USE_GRID_REFERENCE        = SHIFTS_USE_GRID_REFERENCE
    cps.GRID_DIR                  = GRID_DIR
    cps.GRID_BASE_LABEL           = base_label
    cps.GRID_Z_INDEX              = SHIFTS_GRID_Z_INDEX
    cps.CHANNEL_ROIS_JSON         = str(channels_dir / "channel_rois.json")
    cps.REFERENCE_FRAME           = SHIFTS_REFERENCE_FRAME
    cps.ALIGNMENT_METHOD          = SHIFTS_METHOD
    cps.VMIN                      = SHIFTS_VMIN
    cps.VMAX                      = SHIFTS_VMAX
    cps.OUTLIER_MAD_THRESH        = SHIFTS_OUTLIER_MAD_THRESH
    cps.OUTLIER_TIMESERIES_WINDOW = SHIFTS_TIMESERIES_WINDOW
    cps.OUTLIER_TIMESERIES_THRESH = SHIFTS_TIMESERIES_THRESH
    cps.ECC_MIN_CORR              = SHIFTS_ECC_MIN_CORR
    cps.APPLY_BACKSUB_TO_GRID_REF      = SHIFTS_APPLY_BACKSUB_TO_GRID_REF
    cps.USE_INCREMENTAL_TRACKING       = SHIFTS_USE_INCREMENTAL_TRACKING
    cps.X_STEP                         = SHIFTS_X_STEP
    cps.Y_STEP                         = SHIFTS_Y_STEP
    cps.SHIFT_SIGN_X                   = SHIFTS_SHIFT_SIGN_X
    cps.SHIFT_SIGN_Y                   = SHIFTS_SHIFT_SIGN_Y
    cps.JUMP_THRESH_UM                 = SHIFTS_JUMP_THRESH_UM
    # per-Pos キャリブレーション JSON の使用判断
    per_pos_cal = Path(GRID_DIR) / f"grid_calibration_{base_label}.json"
    if SHIFTS_USE_PER_POS_CALIBRATION and per_pos_cal.exists():
        cps.GRID_CALIBRATION_JSON = str(per_pos_cal)
        print(f"  [calibration] per-Pos キャリブレーション使用: {per_pos_cal.name}")
    else:
        cps.GRID_CALIBRATION_JSON = GRID_CALIBRATION_JSON
        if not SHIFTS_USE_PER_POS_CALIBRATION:
            print(f"  [calibration] 名目値を使用 (SHIFTS_USE_PER_POS_CALIBRATION=False)")
    # 2段階ECC
    cps.USE_SECOND_PASS_ECC            = SHIFTS_USE_SECOND_PASS_ECC
    cps.FIRST_PASS_HALF                = SHIFTS_FIRST_PASS_HALF
    cps.SECOND_PASS_HALF               = (
        SHIFTS_SECOND_PASS_HALF
        if SHIFTS_SECOND_PASS_HALF is not None
        else ('right' if pos_number_from_label(base_label) < POS_SPLIT else 'left')
    )
    cps.USE_THIRD_PASS_ECC             = SHIFTS_USE_THIRD_PASS_ECC
    cps.SENSOR_PIXEL_SIZE              = GSUB_SENSOR_PIXEL_SIZE
    cps.MAGNIFICATION                  = GSUB_MAGNIFICATION
    cps.ORIGINAL_DIM                   = GSUB_ORIGINAL_DIM
    cps.RECONSTRUCTED_DIM              = GSUB_RECONSTRUCTED_DIM
    cps.MAX_FRAMES                     = TEST_N_FRAMES
    cps.main()


# ===========================================================
# Step 6: grid_subtract
# ===========================================================
def step_grid_subtract(channels_dir: Path, base_label: str):
    import grid_subtract as gs
    shifts_json = channels_dir / "pos_shifts.json"
    rois_json   = channels_dir / "channel_rois.json"
    if not shifts_json.exists():
        print(f"  [SKIP] pos_shifts.json なし: {shifts_json}")
        return
    if not rois_json.exists():
        print(f"  [SKIP] channel_rois.json なし: {rois_json}")
        return

    # channels_dir = .../PosX/output_phase/channels → PosX dir = parent.parent
    gs.TIMELAPSE_DIR       = str(channels_dir.parent.parent)
    gs.SHIFTS_JSON         = str(shifts_json)
    gs.CHANNEL_ROIS_JSON   = str(rois_json)
    gs.GRID_DIR            = GRID_DIR
    gs.BASE_LABEL          = base_label
    gs.GRID_Z_INDEX        = GSUB_Z_INDEX
    gs.X_STEP              = GSUB_X_STEP
    gs.Y_STEP              = GSUB_Y_STEP
    gs.SENSOR_PIXEL_SIZE   = GSUB_SENSOR_PIXEL_SIZE
    gs.MAGNIFICATION       = GSUB_MAGNIFICATION
    gs.ORIGINAL_DIM        = GSUB_ORIGINAL_DIM
    gs.RECONSTRUCTED_DIM   = GSUB_RECONSTRUCTED_DIM
    gs.SHIFT_SIGN_X              = GSUB_SHIFT_SIGN_X
    gs.SHIFT_SIGN_Y              = GSUB_SHIFT_SIGN_Y
    gs.APPLY_INVERSE_SHIFT       = GSUB_APPLY_INVERSE_SHIFT
    gs.APPLY_BACKSUB_TO_GRID     = GSUB_APPLY_BACKSUB_TO_GRID
    gs.APPLY_SUBPIXEL_CORRECTION = GSUB_APPLY_SUBPIXEL_CORRECTION
    gs.GRID_CALIBRATION_JSON     = GRID_CALIBRATION_JSON
    gs.MAX_FRAMES                = TEST_N_FRAMES
    gs.main()


# ===========================================================
# メイン
# ===========================================================
def main():
    errors = []

    # ── Step -1: calibrate_grid_positions ────────────────────
    if STEP_CALIBRATE_GRID:
        try:
            step_calibrate_grid()
        except Exception as e:
            import traceback; traceback.print_exc()
            errors.append(f"calibrate_grid: {e}")

    # ── Step 0: Grid reconstruction ──────────────────────────
    if STEP_GRID_RECONSTRUCTION:
        try:
            step_grid_reconstruction()
        except Exception as e:
            import traceback; traceback.print_exc()
            errors.append(f"grid_reconstruction: {e}")

    # ── 各タイムラプスディレクトリを処理 ─────────────────────
    for tl_dir_str in TIMELAPSE_DIRS:
        tl_dir = Path(tl_dir_str)
        if not tl_dir.exists():
            errors.append(f"TIMELAPSE_DIR が見つかりません: {tl_dir}")
            continue

        _banner(f"タイムラプス: {tl_dir.name}")

        # ── Step 1: Timelapse reconstruction ─────────────────
        if STEP_TIMELAPSE_RECONSTRUCTION:
            try:
                pos_dirs = step_timelapse_reconstruction(tl_dir)
            except Exception as e:
                import traceback; traceback.print_exc()
                errors.append(f"{tl_dir.name} timelapse_reconstruction: {e}")
                continue
        else:
            # reconstruction スキップ時は既存 Pos フォルダを列挙
            pos_dirs = sorted([d for d in tl_dir.iterdir()
                               if d.is_dir() and re.match(r"^Pos\d+$", d.name)
                               and d.name != TL_BG_LABEL])
            if POS_FILTER:
                pos_dirs = [d for d in pos_dirs if d.name in POS_FILTER]

        # ── Steps 2-6: 各 Pos を処理 ─────────────────────────
        for pos_dir in pos_dirs:
            phase_dir    = pos_dir / "output_phase"
            channels_dir = phase_dir / "channels"
            base_label   = pos_dir.name  # e.g. "Pos4"

            if not phase_dir.exists():
                print(f"  [SKIP] output_phase/ なし: {pos_dir}")
                continue

            _section(f"{tl_dir.name} / {pos_dir.name}")

            # Step 1.5: Gaussian gradient removal（channel_crop の前）
            crop_pattern = CROP_IMG_PATTERN
            if STEP_GAUSSIAN_GRADIENT:
                try:
                    crop_pattern = step_gaussian_gradient(phase_dir)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    errors.append(f"{pos_dir.name}: gaussian_gradient: {e}")
                    # 失敗してもそのまま元パターンで続行

            # Step 2: channel_crop
            if STEP_CHANNEL_CROP:
                try:
                    ok = step_channel_crop(phase_dir, img_pattern=crop_pattern)
                    if not ok:
                        errors.append(f"{pos_dir.name}: channel_crop failed")
                        continue
                except Exception as e:
                    import traceback; traceback.print_exc()
                    errors.append(f"{pos_dir.name}: channel_crop: {e}")
                    continue

            # Step 3: gaussian_backsub
            if STEP_GAUSSIAN_BACKSUB:
                try:
                    step_gaussian_backsub(channels_dir)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    errors.append(f"{pos_dir.name}: gaussian_backsub: {e}")
                    continue

            # Step 4: align_simple（確認用）
            if STEP_ALIGN_SIMPLE:
                try:
                    step_align_simple(channels_dir)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    errors.append(f"{pos_dir.name}: align_simple: {e}")

            # Step 5: compute_pos_shifts + shift_visualize
            if STEP_COMPUTE_SHIFTS:
                try:
                    step_compute_shifts(channels_dir, base_label)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    errors.append(f"{pos_dir.name}: compute_shifts: {e}")
                    continue

            # Step 6: grid_subtract
            if STEP_GRID_SUBTRACT:
                try:
                    step_grid_subtract(channels_dir, base_label)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    errors.append(f"{pos_dir.name}: grid_subtract: {e}")

    # ── サマリー ─────────────────────────────────────────────
    _banner("完了")
    if errors:
        print(f"エラー ({len(errors)}件):")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("全ステップ正常終了")


if __name__ == "__main__":
    main()

# %%
