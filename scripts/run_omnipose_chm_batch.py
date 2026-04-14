# -*- coding: utf-8 -*-
"""
07_segmentation.py（251105）のバッチ版。ロジックは単一 Pos のスクリプトと同一。

各 Pos の ``output_phase/channels/crop_sub_rawraw`` 直下にある ``ch00``, ``ch01``, … を
（存在するものだけ）順に処理する。Pos ごとに ch の数が違ってもよい。

``--ch M`` を付けたときだけ ``chMM`` 1 本に限定できる。

例::

    python scripts/run_omnipose_chm_batch.py
    python scripts/run_omnipose_chm_batch.py --ch 1
"""
from __future__ import annotations

import argparse
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

_CH_DIR_PATTERN = re.compile(r"^ch\d+$")


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


def run_one_indir(model: CellposeModel, indir: str, outdir: str) -> None:
    """07_segmentation.py（251105）と同一の推論・保存ブロック。"""
    os.makedirs(outdir, exist_ok=True)

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

    for i, f in enumerate(files, 1):
        try:
            img = tifffile.imread(f)
        except Exception as e:
            print(f"[{i}/{len(files)}] {os.path.basename(f)}  ❌ failed to read: {e}")
            traceback.print_exc()
            empty_mask = np.zeros(
                (1 if img is None else img.shape[0], 1 if img is None else img.shape[1]),
                dtype=np.uint16,
            )
            try:
                tifffile.imwrite(
                    os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")),
                    empty_mask,
                )
            except Exception:
                pass
            error_count += 1
            continue

        print(f"[{i}/{len(files)}] {os.path.basename(f)}")

        try:
            masks, flows, _ = model.eval([img], **EVAL_PARAMS)
        except ValueError as e:
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

        if masks is None or (
            isinstance(masks, (list, tuple, np.ndarray)) and np.max(masks) == 0
        ):
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
    print("Saved results to:", outdir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="07_segmentation.py の Pos × ch バッチ")
    p.add_argument("--timelapse-root", type=str, default=r"F:\260405\ph_260405")
    p.add_argument("--pos-start", type=int, default=2)
    p.add_argument("--pos-end", type=int, default=21)
    p.add_argument(
        "--ch",
        type=int,
        default=None,
        help="省略時: 各 Pos の crop_sub_rawraw 内の ch* をすべて。指定時: chMM のみ。",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=r"C:\Users\QPI\Desktop\train\omni_model\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_omni_model_2026_04_13_10_54_41.173761",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path
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

    for pos in range(args.pos_start, args.pos_end + 1):
        crop_base = os.path.join(args.timelapse_root, f"Pos{pos}", rel_crop)
        print(f"\n=== Pos{pos} ===")
        print("crop_sub_rawraw:", crop_base)

        if not os.path.isdir(crop_base):
            print("  [skip] crop_sub_rawraw が存在しません")
            continue

        if args.ch is not None:
            ch_dirs = [f"ch{args.ch:02d}"]
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

            run_one_indir(model, indir, outdir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
