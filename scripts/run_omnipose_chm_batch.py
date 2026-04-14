# -*- coding: utf-8 -*-
"""
07_segmentation.py（251105）のバッチ版。ロジックは単一 Pos のスクリプトと同一。

各 Pos の ``output_phase/channels/crop_sub_rawraw`` 直下にある ``ch00``, ``ch01``, … を
（存在するものだけ）順に処理する。Pos ごとに ch の数が違ってもよい。

**パラメータは CLI ではなく、下の「バッチ設定」ブロックを編集する。**

``cellpose_omni.dynamics`` が連続 N 回 ``No cell pixels found.`` を出したら、その ch は
細胞なしチャネルとみなして残フレームをスキップする（``NO_CELL_PIXEL_STREAK``、0 で無効）。

例::

    python scripts/run_omnipose_chm_batch.py
"""
from __future__ import annotations

import logging
import os
import re
import sys

import numpy as np
import tifffile
import traceback
from cellpose_omni import io
from cellpose_omni.models import CellposeModel

EVAL_PARAMS = dict(
    channels=None,
    channel_axis=None,
    diameter=30,
    normalize=True,
    tile=False,
    net_avg=True,
    omni=True,
    verbose=False,
    flow_threshold=0.11,
    mask_threshold=0,
    min_size=10,
)

NCHAN = 1
NCLASSES = 3
USE_GPU = True

# ==== バッチ設定（必要に応じて変更） ====
TIMELAPSE_ROOT = r"F:\260405\ph_260405"
POS_START = 3
POS_END = 21
# 全 ch を回す: None  /  単一 ch のみ: 整数 M で chMM のみ（例: 1 → ch01）
CH_ONLY = None
MODEL_PATH = r"C:\Users\QPI\Desktop\train\omni_model\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_omni_model_2026_04_13_10_54_41.173761"
# 連続 N 回 No cell pixels で ch 打ち切り。0 で無効。
NO_CELL_PIXEL_STREAK = 20

_CH_DIR_PATTERN = re.compile(r"^ch\d+$")

# dynamics.py が INFO で出す文言（連続検出用）
_NO_CELL_PIXELS_LOG_HANDLER: logging.Handler | None = None


class _NoCellPixelsLogHandler(logging.Handler):
    """1 回の model.eval 内で 'No cell pixels found.' が出たか（emit のたびに True）。"""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.saw_no_cell_pixels = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if "No cell pixels found" in record.getMessage():
                self.saw_no_cell_pixels = True
        except Exception:
            pass


def _ensure_no_cell_pixels_handler() -> _NoCellPixelsLogHandler:
    global _NO_CELL_PIXELS_LOG_HANDLER
    if _NO_CELL_PIXELS_LOG_HANDLER is None:
        _NO_CELL_PIXELS_LOG_HANDLER = _NoCellPixelsLogHandler()
        lg = logging.getLogger("cellpose_omni.dynamics")
        lg.addHandler(_NO_CELL_PIXELS_LOG_HANDLER)
        if lg.level > logging.INFO:
            lg.setLevel(logging.INFO)
    return _NO_CELL_PIXELS_LOG_HANDLER


def list_ch_dirs(crop_sub_rawraw: str) -> list[str]:
    """crop_sub_rawraw 直下で名前が ch + 数字だけのディレクトリをソートして返す。"""
    if not os.path.isdir(crop_sub_rawraw):
        return []
    names = []
    for name in os.listdir(crop_sub_rawraw):
        path = os.path.join(crop_sub_rawraw, name)
        if os.path.isdir(path) and _CH_DIR_PATTERN.match(name):
            names.append(name)

    def sort_key(n: str) -> tuple[int, str]:
        # ch の後ろを数値として比較（ch2 と ch10 の順など）
        num = int(n[2:])
        return (num, n)

    return sorted(names, key=sort_key)


def _masks_effectively_empty(masks) -> bool:
    """eval の戻りで細胞ラベルが無い（全 0 または None）か。"""
    if masks is None:
        return True
    if isinstance(masks, (list, tuple)):
        if len(masks) == 0:
            return True
        arr = np.asarray(masks[0])
    else:
        arr = np.asarray(masks)
    return arr.size == 0 or np.max(arr) == 0


def run_one_indir(
    model: CellposeModel,
    indir: str,
    outdir: str,
    *,
    no_cell_pixel_streak_max: int = 100,
) -> None:
    """07_segmentation.py（251105）と同一の推論・保存ブロック。"""
    os.makedirs(outdir, exist_ok=True)
    logcap = _ensure_no_cell_pixels_handler()
    no_cell_streak = 0

    files = io.get_image_files(indir, mask_filter="_masks", look_one_level_down=False)

    if not isinstance(files, (list, tuple)):
        files = list(files)

    proc_files = []
    for f in files:
        if isinstance(f, (list, tuple)):
            if len(f) > 0:
                proc_files.append(f[0])
        else:
            proc_files.append(f)
    files = proc_files

    print(f"Found {len(files)} files for inference")
    assert files, "No images found."

    skipped = 0
    error_count = 0
    processed = 0
    aborted_no_cell_channel = False

    for i, f in enumerate(files, 1):
        try:
            img = tifffile.imread(f)
        except Exception as e:
            no_cell_streak = 0
            print(f"[{i}/{len(files)}] {os.path.basename(f)}  ❌ failed to read: {e}")
            traceback.print_exc()
            error_count += 1
            continue

        print(f"[{i}/{len(files)}] {os.path.basename(f)}")

        try:
            logcap.saw_no_cell_pixels = False
            masks, flows, _ = model.eval([img], **EVAL_PARAMS)
            # ログが届かない環境があるため、dynamics の INFO に加え
            # 「マスクが空（細胞ピクセルなし）」でも 1 連続と数える。
            if _masks_effectively_empty(masks) or logcap.saw_no_cell_pixels:
                no_cell_streak += 1
            else:
                no_cell_streak = 0
        except ValueError as e:
            no_cell_streak = 0
            print(f"  ⚠ Skipping due to ValueError: {e}")
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(
                os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")),
                empty_mask,
            )
            tifffile.imwrite(
                os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8),
            )
            skipped += 1
            continue
        except Exception as e:
            no_cell_streak = 0
            print(f"  ⚠ Skipped due to unexpected error: {e}")
            traceback.print_exc()
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(
                os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")),
                empty_mask,
            )
            tifffile.imwrite(
                os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8),
            )
            error_count += 1
            continue

        if (
            no_cell_pixel_streak_max > 0
            and no_cell_streak >= no_cell_pixel_streak_max
        ):
            print(
                f"  → [skip ch] 細胞なしが {no_cell_streak} 連続 "
                f"(空マスク / [INFO] No cell pixels) → "
                "no-cell channel とみなし、この ch の残りフレームは未処理です。"
            )
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(
                os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")),
                empty_mask,
            )
            tifffile.imwrite(
                os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8),
            )
            skipped += 1
            aborted_no_cell_channel = True
            break

        if _masks_effectively_empty(masks):
            print("  → No cells detected (saving empty mask).")
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(
                os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")),
                empty_mask,
            )
            tifffile.imwrite(
                os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8),
            )
            skipped += 1
            continue

        base = os.path.splitext(os.path.basename(f))[0]
        try:
            out_mask = masks[0].astype(np.uint16)
            tifffile.imwrite(os.path.join(outdir, f"{base}_masks.tif"), out_mask)
            m = out_mask
            border = (
                (m != np.roll(m, 1, 0))
                | (m != np.roll(m, -1, 0))
                | (m != np.roll(m, 1, 1))
                | (m != np.roll(m, -1, 1))
            ) & (m > 0)
            tifffile.imwrite(
                os.path.join(outdir, f"{base}_binary.tif"),
                np.where(border, 0, 255).astype(np.uint8),
            )
            processed += 1
        except Exception as e:
            print(f"  ❌ Failed to save outputs for {f}: {e}")
            traceback.print_exc()
            error_count += 1
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(
                os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")),
                empty_mask,
            )
            tifffile.imwrite(
                os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8),
            )

    print("=== Inference summary ===")
    print(f"Total files : {len(files)}")
    print(f"Processed   : {processed}")
    print(f"Skipped     : {skipped} (no cells or kNN issue)")
    print(f"Errors      : {error_count}")
    if aborted_no_cell_channel:
        print(
            f"Note        : 細胞なしが {no_cell_pixel_streak_max} 連続 → ch 打ち切り（残フレーム未処理）"
        )
    print("Saved results to:", outdir)


def main() -> int:
    model_path = MODEL_PATH
    if not os.path.isfile(model_path):
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    model = CellposeModel(
        gpu=USE_GPU,
        pretrained_model=model_path,
        omni=True,
        nchan=NCHAN,
        nclasses=NCLASSES,
        dim=2,
    )

    rel_crop = os.path.join("output_phase", "channels", "crop_sub_rawraw")

    for pos in range(POS_START, POS_END + 1):
        crop_base = os.path.join(TIMELAPSE_ROOT, f"Pos{pos}", rel_crop)
        print(f"\n=== Pos{pos} ===")
        print("crop_sub_rawraw:", crop_base)

        if not os.path.isdir(crop_base):
            print("  [skip] crop_sub_rawraw が存在しません")
            continue

        if CH_ONLY is not None:
            ch_dirs = [f"ch{CH_ONLY:02d}"]
        else:
            ch_dirs = list_ch_dirs(crop_base)

        if not ch_dirs:
            print("  [skip] 対象の ch フォルダがありません")
            continue

        print("  ch dirs:", ", ".join(ch_dirs))

        for ch_name in ch_dirs:
            indir = os.path.join(crop_base, ch_name)
            outdir = os.path.join(indir, "inference_out")

            print(f"\n  --- {ch_name} ---")
            print("  indir:", indir)
            print("  outdir:", outdir)

            if not os.path.isdir(indir):
                print("  [skip] indir が存在しません")
                continue

            run_one_indir(
                model,
                indir,
                outdir,
                no_cell_pixel_streak_max=NO_CELL_PIXEL_STREAK,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
