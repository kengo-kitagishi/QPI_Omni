# %%
"""
run_pipeline.py
---------------
ROOT_DIR 内の全 Pos* ディレクトリに対して以下を一括実行する：
  1. channel_crop     : チャネルcrop → channels/channel_XX.tif
  2. gaussian_backsub : 背景補正     → channels/channel_XX_bg_corr.tif
  3. align_simple     : 確認用アライメント（オプション）
  4. compute_shifts   : チャネル間平均シフト計算 → channels/pos_shifts.json
  5. grid_subtract    : グリッド画像を使ったsubtract → channels/grid_subtracted/
"""
import sys
import json
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

# ============================================================
# 実行対象
# ============================================================
ROOT_DIR = r"E:\Acuisition\kitagishi\260301\movetest_8"

# Posフィルタ: Noneで全Pos、["Pos4", "Pos5"] のように指定も可
POS_FILTER = ["Pos1"]

# 実行するステップ（Falseでスキップ）
STEP_CHANNEL_CROP_DETECT = True   # channel_rois.json がなければ自動でdetect
STEP_CHANNEL_CROP_APPLY  = True
STEP_GAUSSIAN_BACKSUB    = True
STEP_ALIGN_SIMPLE        = False  # 確認用（時間がかかる）
STEP_COMPUTE_SHIFTS      = True
STEP_GRID_SUBTRACT       = True

# ============================================================
# channel_crop パラメータ
# ============================================================
CROP_PATTERN   = "img_*_ph_000.tif"
CROP_W         = 40
CROP_H         = 270
MIN_DIST       = 35
PROMINENCE     = 0.3
X_START        = None   # int を入れると自動エッジ検出しない
X_END          = None

# ============================================================
# gaussian_backsub パラメータ
# ============================================================
BACKSUB_MIN_PHASE    = -1.1
BACKSUB_HIST_MIN     = -1.1
BACKSUB_HIST_MAX     = 1.5
BACKSUB_N_BINS       = 512
BACKSUB_SMOOTH_WINDOW = 20
BACKSUB_SAVE_PNG     = False
BACKSUB_PNG_DPI      = 150

# ============================================================
# align_and_subtract_simple パラメータ（STEP_ALIGN_SIMPLE=True の時のみ）
# ============================================================
ALIGN_REFERENCE_FRAME  = 150     # 1始まり
ALIGN_METHOD           = 'ecc'
ALIGN_SAVE_PNG         = True
ALIGN_PNG_SAMPLE       = 5
ALIGN_VMIN             = -0.1
ALIGN_VMAX             = 1.7

# ============================================================
# compute_pos_shifts パラメータ
# ============================================================
SHIFTS_CHANNEL_PATTERN      = "channel_*_bg_corr.tif"  # backsub後のパターン

# USE_GRID_REFERENCE=True: グリッドのx+0_y+0を基準にする（推奨）
# USE_GRID_REFERENCE=False: タイムラプスの SHIFTS_REFERENCE_FRAME 番目を基準にする
SHIFTS_USE_GRID_REFERENCE   = True
SHIFTS_REFERENCE_FRAME      = 150    # USE_GRID_REFERENCE=False の場合のみ使用

SHIFTS_METHOD               = 'ecc'
SHIFTS_VMIN                 = -5.0
SHIFTS_VMAX                 = 2.0
SHIFTS_OUTLIER_MAD_THRESH   = 5.0
SHIFTS_TIMESERIES_WINDOW    = 11
SHIFTS_TIMESERIES_THRESH    = 3.0

# ============================================================
# grid_subtract パラメータ
# ============================================================
GRID_DIR             = r"E:\Acuisition\kitagishi\260301\multipos_test_1"
Z_INDEX              = 2
X_STEP               = 0.1
Y_STEP               = 0.1
SENSOR_PIXEL_SIZE    = 3.45e-6
MAGNIFICATION        = 40
ORIGINAL_DIM         = 2048
RECONSTRUCTED_DIM    = 511
SHIFT_SIGN_X         = 1
SHIFT_SIGN_Y         = 1
APPLY_INVERSE_SHIFT  = False
# ============================================================


# ---- 各モジュールの関数をインポート ----
_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))


def _print_step(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ===========================================================
# Step 1: channel_crop
# ===========================================================
def run_channel_crop(pos_dir: Path):
    from channel_crop import run_detect, run_apply

    img_files = sorted(pos_dir.glob(CROP_PATTERN))
    if not img_files:
        print(f"  [SKIP] 画像が見つかりません: {CROP_PATTERN}")
        return False

    out_dir = pos_dir / "channels"
    out_dir.mkdir(exist_ok=True)
    roi_path = out_dir / "channel_rois.json"

    rois = None
    if STEP_CHANNEL_CROP_DETECT and not roi_path.exists():
        print(f"  [detect] {img_files[0].name} でチャネル検出")
        rois = run_detect(
            img_files[0], CROP_W, CROP_H, out_dir,
            min_dist=MIN_DIST,
            prominence_sigma=PROMINENCE,
            x_start=X_START,
            x_end=X_END,
        )
    elif not roi_path.exists():
        print(f"  [ERROR] channel_rois.json がなく detect もスキップ設定です")
        return False

    if STEP_CHANNEL_CROP_APPLY:
        if rois is None:
            with open(roi_path, encoding="utf-8") as f:
                rois = json.load(f)
        print(f"  [apply] {len(img_files)} フレーム × {len(rois)} チャネル")
        run_apply(pos_dir, CROP_PATTERN, rois, out_dir)

    return True


# ===========================================================
# Step 2: gaussian_backsub
# ===========================================================
def run_gaussian_backsub(pos_dir: Path):
    """channels/ 内の channel_XX.tif (スタック) を1ファイルずつ処理する。"""
    import importlib
    import sys as _sys

    # 19_gaussian_backsub.py を動的にインポート（数字始まりのため）
    import importlib.util
    backsub_path = _script_dir / "19_gaussian_backsub.py"
    spec = importlib.util.spec_from_file_location("gaussian_backsub", backsub_path)
    backsub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backsub)

    # パラメータを差し替え
    backsub.minPhase      = BACKSUB_MIN_PHASE
    backsub.hist_min      = BACKSUB_HIST_MIN
    backsub.hist_max      = BACKSUB_HIST_MAX
    backsub.n_bins        = BACKSUB_N_BINS
    backsub.smooth_window = BACKSUB_SMOOTH_WINDOW

    channels_dir = pos_dir / "channels"
    stack_files = sorted(channels_dir.glob("channel_*.tif"))
    # _bg_corr.tif はスキップ
    stack_files = [p for p in stack_files if "_bg_corr" not in p.name]

    if not stack_files:
        print(f"  [SKIP] channel_*.tif が見つかりません: {channels_dir}")
        return

    for tif_path in stack_files:
        out_path = channels_dir / f"{tif_path.stem}_bg_corr.tif"
        if out_path.exists():
            print(f"  [SKIP already] {tif_path.name}")
            continue
        print(f"  処理: {tif_path.name}")
        backsub.process_image(tif_path, channels_dir, save_png_data=False)
        if BACKSUB_SAVE_PNG:
            result = backsub.process_image(tif_path, channels_dir, save_png_data=True)
            if isinstance(result, dict):
                backsub.save_png_plots(result, channels_dir, BACKSUB_PNG_DPI)


# ===========================================================
# Step 3: align_and_subtract_simple（確認用）
# ===========================================================
def run_align_simple(pos_dir: Path):
    from align_and_subtract_simple import (
        get_tif_files, load_tif_image, process_timelapse
    )
    channels_dir = pos_dir / "channels"
    stack_files = sorted(channels_dir.glob(SHIFTS_CHANNEL_PATTERN))
    if not stack_files:
        print(f"  [SKIP] {SHIFTS_CHANNEL_PATTERN} が見つかりません")
        return

    for stack_path in stack_files:
        print(f"  align_simple: {stack_path.name}")
        # スタックを個別フレームとして扱う
        arr = tifffile.imread(str(stack_path))
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        ref_idx = min(ALIGN_REFERENCE_FRAME - 1, arr.shape[0] - 1)
        reference_img = arr[ref_idx].astype(np.float64)

        # 一時ディレクトリに個別フレームとして書き出してから process_timelapse を呼ぶ
        # → シンプルに process_timelapse の signature に合わせた仮tifリストを作る
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frame_paths = []
            for i in range(arr.shape[0]):
                p = tmpdir / f"frame_{i:06d}.tif"
                tifffile.imwrite(str(p), arr[i].astype(np.float32))
                frame_paths.append(str(p))

            out_base = channels_dir / stack_path.stem
            process_timelapse(
                str(out_base), reference_img, frame_paths,
                method=ALIGN_METHOD,
                save_png=ALIGN_SAVE_PNG,
                vmin=ALIGN_VMIN, vmax=ALIGN_VMAX,
                png_dpi=150,
                png_sample_interval=ALIGN_PNG_SAMPLE,
            )


# ===========================================================
# Step 4: compute_pos_shifts
# ===========================================================
def run_compute_shifts(pos_dir: Path):
    import compute_pos_shifts as cps

    channels_dir = pos_dir / "channels"

    base_label = pos_dir.name  # e.g. "Pos4"

    # パラメータをモジュール変数に差し込む
    cps.CHANNELS_DIR              = str(channels_dir)
    cps.CHANNEL_PATTERN           = SHIFTS_CHANNEL_PATTERN
    cps.USE_GRID_REFERENCE        = SHIFTS_USE_GRID_REFERENCE
    cps.GRID_DIR                  = GRID_DIR
    cps.GRID_BASE_LABEL           = base_label
    cps.GRID_Z_INDEX              = Z_INDEX
    cps.CHANNEL_ROIS_JSON         = str(channels_dir / "channel_rois.json")
    cps.REFERENCE_FRAME           = SHIFTS_REFERENCE_FRAME
    cps.ALIGNMENT_METHOD          = SHIFTS_METHOD
    cps.VMIN                      = SHIFTS_VMIN
    cps.VMAX                      = SHIFTS_VMAX
    cps.OUTLIER_MAD_THRESH        = SHIFTS_OUTLIER_MAD_THRESH
    cps.OUTLIER_TIMESERIES_WINDOW = SHIFTS_TIMESERIES_WINDOW
    cps.OUTLIER_TIMESERIES_THRESH = SHIFTS_TIMESERIES_THRESH

    cps.main()


# ===========================================================
# Step 5: grid_subtract
# ===========================================================
def run_grid_subtract(pos_dir: Path, base_label: str):
    import grid_subtract as gs

    channels_dir = pos_dir / "channels"
    shifts_json  = channels_dir / "pos_shifts.json"
    rois_json    = channels_dir / "channel_rois.json"

    if not shifts_json.exists():
        print(f"  [SKIP] pos_shifts.json が見つかりません: {shifts_json}")
        return
    if not rois_json.exists():
        print(f"  [SKIP] channel_rois.json が見つかりません: {rois_json}")
        return

    gs.CHANNELS_DIR       = str(channels_dir)
    gs.CHANNEL_PATTERN    = SHIFTS_CHANNEL_PATTERN
    gs.SHIFTS_JSON        = str(shifts_json)
    gs.CHANNEL_ROIS_JSON  = str(rois_json)
    gs.GRID_DIR           = GRID_DIR
    gs.BASE_LABEL         = base_label
    gs.Z_INDEX            = Z_INDEX
    gs.X_STEP             = X_STEP
    gs.Y_STEP             = Y_STEP
    gs.SENSOR_PIXEL_SIZE  = SENSOR_PIXEL_SIZE
    gs.MAGNIFICATION      = MAGNIFICATION
    gs.ORIGINAL_DIM       = ORIGINAL_DIM
    gs.RECONSTRUCTED_DIM  = RECONSTRUCTED_DIM
    gs.SHIFT_SIGN_X       = SHIFT_SIGN_X
    gs.SHIFT_SIGN_Y       = SHIFT_SIGN_Y
    gs.APPLY_INVERSE_SHIFT = APPLY_INVERSE_SHIFT

    gs.main()


# ===========================================================
# メイン
# ===========================================================
def main():
    root = Path(ROOT_DIR)
    if not root.exists():
        print(f"ERROR: ROOT_DIR が見つかりません: {root}")
        sys.exit(1)

    # Posディレクトリ一覧
    pos_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("Pos")])
    if POS_FILTER:
        pos_dirs = [d for d in pos_dirs if d.name in POS_FILTER]

    if not pos_dirs:
        print(f"ERROR: Pos* ディレクトリが見つかりません: {root}")
        sys.exit(1)

    print(f"対象Pos: {[d.name for d in pos_dirs]}")

    errors = []

    for pos_dir in pos_dirs:
        print(f"\n{'#'*60}")
        print(f"  {pos_dir.name}  ({pos_dir})")
        print(f"{'#'*60}")

        base_label = pos_dir.name  # e.g. "Pos4"

        # Step 1: channel_crop
        if STEP_CHANNEL_CROP_DETECT or STEP_CHANNEL_CROP_APPLY:
            _print_step(f"[1] channel_crop  ({pos_dir.name})")
            try:
                ok = run_channel_crop(pos_dir)
                if not ok:
                    errors.append(f"{pos_dir.name}: channel_crop failed")
                    continue
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                errors.append(f"{pos_dir.name}: channel_crop ERROR: {e}")
                continue

        # Step 2: gaussian_backsub
        if STEP_GAUSSIAN_BACKSUB:
            _print_step(f"[2] gaussian_backsub  ({pos_dir.name})")
            try:
                run_gaussian_backsub(pos_dir)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                errors.append(f"{pos_dir.name}: gaussian_backsub ERROR: {e}")
                continue

        # Step 3: align_simple (確認用)
        if STEP_ALIGN_SIMPLE:
            _print_step(f"[3] align_and_subtract_simple  ({pos_dir.name})")
            try:
                run_align_simple(pos_dir)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                errors.append(f"{pos_dir.name}: align_simple ERROR: {e}")

        # Step 4: compute_shifts
        if STEP_COMPUTE_SHIFTS:
            _print_step(f"[4] compute_pos_shifts  ({pos_dir.name})")
            try:
                run_compute_shifts(pos_dir)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                errors.append(f"{pos_dir.name}: compute_shifts ERROR: {e}")
                continue

        # Step 5: grid_subtract
        if STEP_GRID_SUBTRACT:
            _print_step(f"[5] grid_subtract  ({pos_dir.name})")
            try:
                run_grid_subtract(pos_dir, base_label)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                errors.append(f"{pos_dir.name}: grid_subtract ERROR: {e}")

    # サマリー
    print(f"\n{'='*60}")
    print(f"完了: {len(pos_dirs)} Pos 処理")
    if errors:
        print(f"\nエラー ({len(errors)}件):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("全Pos 正常終了")


if __name__ == "__main__":
    main()

# %%
