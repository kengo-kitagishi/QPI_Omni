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
import json
import importlib.util
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

# ============================================================
# ★ データパス
# ============================================================
TIMELAPSE_DIRS = [
    r"E:\Acuisition\kitagishi\260301\movetest_8",
]
GRID_DIR = r"E:\Acuisition\kitagishi\260301\multipos_test_1"

# ============================================================
# ★ 実行ステップ（False でスキップ）
# ============================================================
STEP_GRID_RECONSTRUCTION     = False
STEP_TIMELAPSE_RECONSTRUCTION = False
STEP_CHANNEL_CROP            = True
STEP_GAUSSIAN_BACKSUB        = True
STEP_ALIGN_SIMPLE            = False   # 確認用（時間がかかる）
STEP_COMPUTE_SHIFTS          = True    # shift_visualize も自動実行
STEP_GRID_SUBTRACT           = True

# ============================================================
# Pos フィルタ（タイムラプス側）
# ============================================================
# None で全 Pos。["Pos1", "Pos3"] のように指定も可
POS_FILTER = ["Pos1"]

# ============================================================
# QPI 光学パラメータ
# ============================================================
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

# ============================================================
# Pos 番号によるクロップ切り替え（grid / timelapse 共通）
# ============================================================
POS_SPLIT   = 3
CROP_BEFORE = (0, 2048, 416, 2464)   # Pos0, Pos1, Pos2  → 右側 (416:2464)
CROP_AFTER  = (0, 2048,   0, 2048)   # Pos3 以降        → 左側 (0:2048)

# ============================================================
# 再構成パラメータ
# ============================================================
# グリッド
GRID_BG_BASE_LABEL        = "Pos0"   # BG として使うグリッドの base_label
GRID_TARGET_BASE_LABELS   = ["Pos1"]  # None で Pos0 以外を全処理
GRID_SKIP_IF_EXISTS       = False
GRID_MEAN_REGION          = None     # (r1, r2, c1, c2) or None

# タイムラプス
TL_BG_LABEL               = "Pos0"  # BG フォルダ名
TL_RAW_PATTERN            = "img_*_ph_000.tif"   # 生画像パターン
TL_SKIP_IF_EXISTS         = True
TL_MEAN_REGION            = None     # (r1, r2, c1, c2) or None

# ============================================================
# channel_crop パラメータ
# ============================================================
CROP_DETECT               = True     # channel_rois.json がなければ自動 detect
CROP_FORCE_RECOMPUTE      = False    # True にすると channels/ を丸ごと削除して最初から再計算
CROP_FORCE_DETECT         = False    # True にすると既存 channel_rois.json のみ削除して再検出
CROP_APPLY                = True
CROP_IMG_PATTERN          = "*_phase.tif"  # output_phase/ 内の位相画像パターン
CROP_W                    = 30
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
SHIFTS_OUTLIER_MAD_THRESH = 2.5
SHIFTS_TIMESERIES_WINDOW  = 11
SHIFTS_TIMESERIES_THRESH  = 3.0
SHIFTS_APPLY_BACKSUB_TO_GRID_REF = True  # グリッド基準画像にも gaussian_backsub を適用

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
GSUB_SHIFT_SIGN_X         = 1
GSUB_SHIFT_SIGN_Y         = 1
GSUB_APPLY_INVERSE_SHIFT  = False
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

        for (xi, yi) in tqdm(sorted(target_map), desc=base_label):
            tgt_dir = target_map[(xi, yi)]
            out_dir = tgt_dir / "output_phase"

            if GRID_SKIP_IF_EXISTS and out_dir.exists() and any(out_dir.glob("*.tif")):
                skip += 1
                continue
            if (xi, yi) not in bg_map:
                print(f"  [WARN] BG {GRID_BG_BASE_LABEL}_x{xi:+d}_y{yi:+d} なし")
                err += 1
                continue

            bg_dir = bg_map[(xi, yi)]
            z_tgt = {_get_z_index(p): p for p in sorted(tgt_dir.glob("img_*_ph_*.tif"))}
            z_bg  = {_get_z_index(p): p for p in sorted(bg_dir.glob("img_*_ph_*.tif"))}
            if not z_tgt:
                err += 1
                continue

            out_dir.mkdir(exist_ok=True)
            sample = next(iter(z_tgt.values()))
            try:
                qpi = _make_qpi_params(sample, crop)
            except Exception as e:
                print(f"  [ERR] QPIParams {tgt_dir.name}: {e}")
                err += 1
                continue

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
                    print(f"  [ERR] {tgt_dir.name} z={z_idx}: {e}")
                    folder_ok = False
            (ok if folder_ok else err).__class__  # dummy
            if folder_ok: ok += 1
            else: err += 1

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
            if all(p.name in done for p in tif_files):
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

        # BG を1度だけロード（fallback: 最初のファイル）
        # BG がタイムラプスと同構造（同一ファイル名）なら都度ロードする
        bg_is_timelapse = len(bg_files) > 1

        bg_phase_cache = {}  # filename → phase（シングルBGモードではキャッシュ）

        n_ok = n_skip = n_err = 0
        for tif_path in tqdm(tif_files, desc=f"  {pos_dir.name}"):
            out_path = out_dir / tif_path.name
            if TL_SKIP_IF_EXISTS and out_path.exists():
                n_skip += 1
                continue

            # BG 選択
            if bg_is_timelapse:
                bg_path = bg_files.get(tif_path.name, bg_fallback)
            else:
                bg_path = bg_fallback

            try:
                if bg_path not in bg_phase_cache:
                    bg_phase_cache[bg_path] = _reconstruct(bg_path, qpi, crop)
                    if len(bg_phase_cache) > 10:
                        # メモリ節約：古いキャッシュを削除
                        oldest = next(iter(bg_phase_cache))
                        del bg_phase_cache[oldest]

                phase = _reconstruct(tif_path, qpi, crop) - bg_phase_cache[bg_path]
                h, w = phase.shape
                if pos_num < POS_SPLIT:
                    region = phase[1:h-1, 1:w//2]
                else:
                    region = phase[1:h-1, w//2:w-1]
                if region.size > 0:
                    phase -= np.mean(region)
                tifffile.imwrite(str(out_path), phase.astype(np.float32))
                n_ok += 1
            except Exception as e:
                print(f"\n  [ERR] {tif_path.name}: {e}")
                n_err += 1

        print(f"  {pos_dir.name}: 完了={n_ok}, スキップ={n_skip}, エラー={n_err}")
        processed.append(pos_dir)

    return processed


# ===========================================================
# Step 2: channel_crop
# ===========================================================
def step_channel_crop(phase_dir: Path) -> bool:
    from channel_crop import run_detect, run_apply

    img_files = sorted(phase_dir.glob(CROP_IMG_PATTERN))
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
        run_apply(phase_dir, CROP_IMG_PATTERN, rois, out_dir)

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
    cps.APPLY_BACKSUB_TO_GRID_REF = SHIFTS_APPLY_BACKSUB_TO_GRID_REF
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

    gs.CHANNELS_DIR        = str(channels_dir)
    gs.CHANNEL_PATTERN     = SHIFTS_CHANNEL_PATTERN
    gs.SHIFTS_JSON         = str(shifts_json)
    gs.CHANNEL_ROIS_JSON   = str(rois_json)
    gs.GRID_DIR            = GRID_DIR
    gs.BASE_LABEL          = base_label
    gs.Z_INDEX             = GSUB_Z_INDEX
    gs.X_STEP              = GSUB_X_STEP
    gs.Y_STEP              = GSUB_Y_STEP
    gs.SENSOR_PIXEL_SIZE   = GSUB_SENSOR_PIXEL_SIZE
    gs.MAGNIFICATION       = GSUB_MAGNIFICATION
    gs.ORIGINAL_DIM        = GSUB_ORIGINAL_DIM
    gs.RECONSTRUCTED_DIM   = GSUB_RECONSTRUCTED_DIM
    gs.SHIFT_SIGN_X        = GSUB_SHIFT_SIGN_X
    gs.SHIFT_SIGN_Y        = GSUB_SHIFT_SIGN_Y
    gs.APPLY_INVERSE_SHIFT = GSUB_APPLY_INVERSE_SHIFT
    gs.main()


# ===========================================================
# メイン
# ===========================================================
def main():
    errors = []

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

            # Step 2: channel_crop
            if STEP_CHANNEL_CROP:
                try:
                    ok = step_channel_crop(phase_dir)
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
