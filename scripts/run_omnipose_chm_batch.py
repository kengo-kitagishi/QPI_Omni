# -*- coding: utf-8 -*-
"""
Omnipose バッチ推論（複数 Pos・同一 ch 相対パス）

実行例::

    conda activate omnipose
    cd C:\\Users\\QPI\\Documents\\QPI_omni
    python scripts/run_omnipose_chm_batch.py

引数省略時の既定: timelapse-root=F:\\260405\\ph_260405, Pos2–21, --ch 9（ch09）。
別チャネルなら --ch 13 のように指定。

    python scripts/run_omnipose_chm_batch.py --ch 13

（--ch M の M はチャネル番号の整数。フォルダ名は ch00, ch01, … のように2桁ゼロ埋め）

モデルを明示する場合::

    python scripts/run_omnipose_chm_batch.py --model-path "C:\\...\\checkpoint_file"

--model-path を省略すると、--model-dir 内で更新日時が最新のファイルを使用する。
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import tifffile
from cellpose_omni import io
from cellpose_omni.models import CellposeModel

# 07_segmentation.py 末尾ブロック（251105）と同じ eval 設定
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


def resolve_latest_model(model_dir: Path) -> Path:
    """model_dir 直下の通常ファイルのうち、mtime が最大のものを返す。"""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    candidates = [p for p in model_dir.iterdir() if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No files in model directory: {model_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def normalize_file_list(raw) -> list:
    files = raw if isinstance(raw, (list, tuple)) else list(raw)
    out = []
    for f in files:
        if isinstance(f, (list, tuple)) and len(f) > 0:
            out.append(f[0])
        else:
            out.append(f)
    return out


def run_inference_for_indir(
    model: CellposeModel,
    indir: Path,
    outdir: Path,
) -> tuple[int, int, int]:
    """1 フォルダ分の推論。戻り値: (processed, skipped, error_count)"""
    outdir.mkdir(parents=True, exist_ok=True)
    raw_files = io.get_image_files(str(indir), mask_filter="_masks", look_one_level_down=False)
    files = normalize_file_list(raw_files)

    if not files:
        print(f"  [skip] No images in {indir}")
        return 0, 0, 0

    processed = 0
    skipped = 0
    error_count = 0
    n = len(files)

    for i, f in enumerate(files, 1):
        f = str(f)
        try:
            img = tifffile.imread(f)
        except Exception as e:
            print(f"  [{i}/{n}] {os.path.basename(f)}  failed to read: {e}")
            traceback.print_exc()
            error_count += 1
            continue

        print(f"  [{i}/{n}] {os.path.basename(f)}")

        try:
            masks, _flows, _ = model.eval([img], **EVAL_PARAMS)
        except ValueError as e:
            print(f"    Skipping due to ValueError: {e}")
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")), empty_mask)
            tifffile.imwrite(
                os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8),
            )
            skipped += 1
            continue
        except Exception as e:
            print(f"    Skipped due to unexpected error: {e}")
            traceback.print_exc()
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")), empty_mask)
            tifffile.imwrite(
                os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8),
            )
            error_count += 1
            continue

        if masks is None or (isinstance(masks, (list, tuple, np.ndarray)) and np.max(masks) == 0):
            print("    No cells detected (saving empty mask).")
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")), empty_mask)
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
            print(f"    Failed to save outputs for {f}: {e}")
            traceback.print_exc()
            error_count += 1
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")), empty_mask)
            tifffile.imwrite(
                os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8),
            )

    return processed, skipped, error_count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Omnipose batch inference for PosN and channel folder chMM (M = integer)."
    )
    p.add_argument(
        "--timelapse-root",
        type=Path,
        default=Path(r"F:\260405\ph_260405"),
        help="Dataset root containing Pos1, Pos2, ...",
    )
    p.add_argument("--pos-start", type=int, default=2)
    p.add_argument("--pos-end", type=int, default=21)
    p.add_argument(
        "--ch",
        type=int,
        metavar="M",
        default=9,
        help="Channel index M → .../crop_sub_rawraw/chMM (2-digit). Default: %(default)s.",
    )
    p.add_argument(
        "--ch-relative",
        type=str,
        default=None,
        help="Override: full relative path under each PosN to images (if set, --ch is ignored).",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path(r"C:\Users\QPI\Desktop\train\omni_model\models"),
        help="Used with newest mtime when --model-path is omitted.",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Explicit checkpoint file; overrides --model-dir resolution.",
    )
    p.add_argument("--no-gpu", action="store_true", help="Run on CPU (default: GPU).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    use_gpu = not args.no_gpu

    if args.model_path is not None:
        ckpt = args.model_path.resolve()
        if not ckpt.is_file():
            print(f"ERROR: --model-path is not a file: {ckpt}", file=sys.stderr)
            return 1
    else:
        ckpt = resolve_latest_model(args.model_dir.resolve())

    print(f"[model] {ckpt}")
    print(f"[gpu]   {use_gpu}")

    model = CellposeModel(
        gpu=use_gpu,
        pretrained_model=str(ckpt),
        omni=True,
        nchan=NCHAN,
        nclasses=NCLASSES,
        dim=2,
    )

    root = args.timelapse_root.resolve()
    if args.ch_relative is not None:
        ch_rel = Path(args.ch_relative)
    else:
        ch_rel = Path("output_phase") / "channels" / "crop_sub_rawraw" / f"ch{args.ch:02d}"
    print(f"[ch]    {ch_rel}  (M={args.ch})" if args.ch_relative is None else f"[ch]    {ch_rel}  (from --ch-relative)")

    total_p = total_sk = total_err = 0
    pos_summaries: list[tuple[int, int, int, int]] = []

    for pos in range(args.pos_start, args.pos_end + 1):
        indir = root / f"Pos{pos}" / ch_rel
        if not indir.is_dir():
            print(f"\n=== Pos{pos} ===\n  [skip] Missing directory: {indir}")
            continue
        outdir = indir / "inference_out"
        print(f"\n=== Pos{pos} ===\n  indir:  {indir}\n  outdir: {outdir}")
        p, sk, err = run_inference_for_indir(model, indir, outdir)
        pos_summaries.append((pos, p, sk, err))
        total_p += p
        total_sk += sk
        total_err += err
        print(f"  summary: processed={p}, skipped={sk}, errors={err}")

    print("\n=== Batch summary (all Pos) ===")
    for pos, p, sk, err in pos_summaries:
        print(f"  Pos{pos}: processed={p}, skipped={sk}, errors={err}")
    print(f"Total processed: {total_p}")
    print(f"Total skipped:   {total_sk}")
    print(f"Total errors:    {total_err}")
    return 0 if total_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
