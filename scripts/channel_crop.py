#!/usr/bin/env python3
"""
channel_crop.py - マイクロ流体チャネルをタイムラプス画像からチャネルごとに矩形cropする

使い方:
  # Step 1: チャネル検出・プレビュー → ROI設定を JSON に保存
  python channel_crop.py --dir "e:/Acuisition/kitagishi/260301/movetest_8/Pos4" --detect

  # JSON を手動編集して不要なチャネルを削除したり cy/cx を微調整する

  # Step 2: 全フレームに適用してチャネルごとの TIFF スタックを出力
  python channel_crop.py --dir "e:/Acuisition/kitagishi/260301/movetest_8/Pos4" --apply

  # 一括実行 (detect → apply)
  python channel_crop.py --dir "e:/Acuisition/kitagishi/260301/movetest_8/Pos4"

ROI JSON フォーマット (channels/channel_rois.json):
  [
    {"cy": 112, "cx": 1224, "crop_w": 30, "crop_h": 120},
    ...
  ]
  - cy     : チャネル中心 Y 座標 [px]
  - cx     : チャネル中心 X 座標 [px]  (crop の水平方向の中心)
  - crop_w : チャネル幅方向 (Y) のサイズ [px]
  - crop_h : チャネル長手方向 (X) のサイズ [px]

出力: <dir>/channels/channel_00.tif, channel_01.tif, ...
      各ファイルは shape (T, crop_w, crop_h) の uint16 TIFF スタック
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from figure_logger import save_figure
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

DEFAULT_CROP_W  = 30
DEFAULT_CROP_H  = 120
DEFAULT_PATTERN = "img_*_ph_000.tif"
ROI_FILENAME    = "channel_rois.json"


# ────────────────────────────────────────────────────────────────
#  I/O
# ────────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    return tifffile.imread(str(path))


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo + 1e-9), 0, 1)


# ────────────────────────────────────────────────────────────────
#  チャネル検出
# ────────────────────────────────────────────────────────────────

def detect_channels(img: np.ndarray, min_distance: int = 35,
                    prominence_sigma: float = 0.3):
    """行平均プロファイルの極小値からチャネル中心 Y 座標を検出。"""
    profile = np.mean(img.astype(np.float32), axis=1)  # shape: (H,)
    inv = profile.max() - profile
    peaks, _ = find_peaks(inv, distance=min_distance,
                           prominence=inv.std() * prominence_sigma)
    return peaks, profile


def detect_channel_edge_x(img: np.ndarray, cy: int, crop_w: int,
                           side: str = "left",
                           dark_threshold: float = 0.4) -> int:
    """チャネルのX方向エッジ位置を検出する。

    チャネル行（cy付近）の水平輝度プロファイルで暗い領域の端を探す。
    side='left'  → 暗い領域の左端（最小X）
    side='right' → 暗い領域の右端（最大X）
    戻り値が crop の cx（長方形の中心）になる。
    """
    y1 = max(0, cy - crop_w // 2)
    y2 = min(img.shape[0], cy + crop_w // 2)
    strip = img[y1:y2, :].astype(np.float32)
    col_profile = np.mean(strip, axis=0)  # shape: (W,)

    lo, hi = col_profile.min(), col_profile.max()
    norm = (col_profile - lo) / (hi - lo + 1e-9)
    is_dark = norm < dark_threshold

    dark_indices = np.where(is_dark)[0]
    if len(dark_indices) == 0:
        return img.shape[1] // 2  # フォールバック

    if side == "left":
        return int(dark_indices[0])
    else:
        return int(dark_indices[-1])


# ────────────────────────────────────────────────────────────────
#  矩形 crop（補間なし）
# ────────────────────────────────────────────────────────────────

def extract_rect_roi(img: np.ndarray, cy: int, cx: int,
                     crop_w: int, crop_h: int) -> np.ndarray:
    """(cx, cy) を中心とした crop_w × crop_h の矩形をそのまま切り出す。"""
    h, w = img.shape
    y1 = cy - crop_w // 2
    y2 = y1 + crop_w
    x1 = cx - crop_h // 2
    x2 = x1 + crop_h

    # 境界クリップ（不足分はゼロパディング）
    pad_y0 = max(0, -y1);  y1 = max(0, y1)
    pad_y1 = max(0, y2 - h); y2 = min(h, y2)
    pad_x0 = max(0, -x1);  x1 = max(0, x1)
    pad_x1 = max(0, x2 - w); x2 = min(w, x2)

    crop = img[y1:y2, x1:x2]
    if any([pad_y0, pad_y1, pad_x0, pad_x1]):
        crop = np.pad(crop, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant")
    return crop


# ────────────────────────────────────────────────────────────────
#  Step 1: 検出 & プレビュー
# ────────────────────────────────────────────────────────────────

def run_detect(img_path: Path, crop_w: int, crop_h: int, out_dir: Path,
               min_dist: int, prominence_sigma: float, side: str = "left",
               dark_threshold: float = 0.4,
               x_start: int = None, x_end: int = None):
    img = load_image(img_path)
    h, w = img.shape

    centers, profile = detect_channels(img, min_distance=min_dist,
                                        prominence_sigma=prominence_sigma)

    rois = []
    for cy in centers:
        if x_start is not None and x_end is not None:
            cx = (x_start + x_end) // 2
            ch = x_end - x_start
        else:
            cx = detect_channel_edge_x(img, int(cy), crop_w,
                                       side=side, dark_threshold=dark_threshold)
            ch = crop_h
        rois.append({"cy": int(cy), "cx": cx, "crop_w": crop_w, "crop_h": ch})


    print(f"検出チャネル数: {len(rois)}")
    for i, r in enumerate(rois):
        print(f"  ch{i:02d}: y={r['cy']}, x={r['cx']}")

    # ─── プレビュー ───
    fig, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(14, 7))
    disp = normalize_for_display(img)

    ax_img.imshow(disp, cmap="gray", aspect="equal")
    ax_img.set_title(f"検出チャネル ({len(rois)}本)", fontsize=12)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rois), 1)))
    for i, roi in enumerate(rois):
        actual_ch = roi["crop_h"]
        rect = mpatches.Rectangle(
            (roi["cx"] - actual_ch // 2, roi["cy"] - crop_w // 2),
            actual_ch, crop_w,
            linewidth=1.5, edgecolor=colors[i % 10], facecolor="none",
        )
        ax_img.add_patch(rect)
        ax_img.text(roi["cx"] - actual_ch // 2 + 2,
                    roi["cy"] - crop_w // 2 - 3,
                    f"ch{i:02d}", color=colors[i % 10], fontsize=6)
    ax_img.set_xlim(0, w)
    ax_img.set_ylim(h, 0)

    ax_prof.plot(profile, np.arange(len(profile)), color="steelblue", lw=0.8)
    ax_prof.scatter(profile[centers], centers, color="red", s=20, zorder=5,
                    label="検出位置")
    ax_prof.invert_xaxis()
    ax_prof.set_title("行平均プロファイル (Y 方向)", fontsize=12)
    ax_prof.set_xlabel("平均強度")
    ax_prof.set_ylabel("Y 座標 [px]")
    ax_prof.set_ylim(h, 0)
    ax_prof.legend(fontsize=8)

    plt.suptitle(str(img_path), fontsize=8)
    logged_path = save_figure(
        fig,
        params={
            "source_image": str(img_path),
            "n_channels": len(rois),
            "crop_w": int(crop_w),
            "crop_h_default": int(crop_h),
            "min_dist": int(min_dist),
            "prominence_sigma": float(prominence_sigma),
            "side": side,
            "dark_threshold": float(dark_threshold),
            "x_start": x_start,
            "x_end": x_end,
        },
        description=f"channel_detection_preview {img_path.parent.name}",
        publish=False,
        dpi=150,
        fmt="png",
    )
    preview_path = out_dir / "channel_detection_preview.png"
    shutil.copy2(logged_path, preview_path)
    print(f"プレビュー保存: {preview_path}")
    print(f"figure_logger 保存: {logged_path}")
    plt.close(fig)

    roi_path = out_dir / ROI_FILENAME
    with open(roi_path, "w", encoding="utf-8") as f:
        json.dump(rois, f, indent=2, ensure_ascii=False)
    print(f"ROI 設定保存: {roi_path}")
    print()
    print("不要なチャネルを JSON から削除したり cy/cx を調整してから --apply を実行してください。")

    return rois


# ────────────────────────────────────────────────────────────────
#  Step 2: 全フレームに適用
# ────────────────────────────────────────────────────────────────

def run_apply(img_dir: Path, pattern: str, rois: list, out_dir: Path):
    files = sorted(img_dir.glob(pattern))
    if not files:
        print(f"画像が見つかりません: {img_dir / pattern}")
        sys.exit(1)

    n_frames   = len(files)
    n_channels = len(rois)
    print(f"フレーム数: {n_frames}, チャネル数: {n_channels}")

    stacks = [[] for _ in range(n_channels)]

    for i, f in enumerate(files):
        if i % 50 == 0:
            print(f"  処理中: {i+1}/{n_frames}  ({f.name})")
        img = load_image(f)
        for ch_idx, roi in enumerate(rois):
            crop = extract_rect_roi(img, roi["cy"], roi["cx"],
                                    roi["crop_w"], roi["crop_h"])
            stacks[ch_idx].append(crop)

    print("保存中...")
    for ch_idx, stack in enumerate(stacks):
        arr = np.array(stack, dtype=np.float32)  # (T, crop_w, crop_h)
        out_path = out_dir / f"channel_{ch_idx:02d}.tif"
        tifffile.imwrite(str(out_path), arr)
        print(f"  ch{ch_idx:02d}: {out_path}  shape={arr.shape}")

    print("完了")


# ────────────────────────────────────────────────────────────────
#  main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="マイクロ流体チャネルをタイムラプス画像から矩形 crop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dir",     required=True, help="画像ディレクトリ")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="ファイル名パターン")
    parser.add_argument("--detect",  action="store_true", help="検出のみ（--apply なし）")
    parser.add_argument("--apply",   action="store_true", help="適用のみ（JSON が必要）")
    parser.add_argument("--crop-w",  type=int, default=DEFAULT_CROP_W,
                        help=f"チャネル幅方向 crop サイズ [px] (default={DEFAULT_CROP_W})")
    parser.add_argument("--crop-h",  type=int, default=DEFAULT_CROP_H,
                        help=f"チャネル長手方向 crop サイズ [px] (default={DEFAULT_CROP_H})")
    parser.add_argument("--min-dist",    type=int,   default=35,
                        help="チャネル検出の最小間隔 [px] (default=35)")
    parser.add_argument("--prominence",  type=float, default=0.3,
                        help="prominence 閾値 (std の倍数, default=0.3)")
    parser.add_argument("--side",        choices=["left", "right"], default="left",
                        help="チャネルのどちらの端を cx にするか (default=left)")
    parser.add_argument("--dark-threshold", type=float, default=0.4,
                        help="暗い領域とみなす正規化輝度の閾値 0-1 (default=0.4)")
    args = parser.parse_args()

    img_dir = Path(args.dir)
    out_dir = img_dir / "channels"
    out_dir.mkdir(exist_ok=True)

    files = sorted(img_dir.glob(args.pattern))
    if not files:
        print(f"画像が見つかりません: {img_dir / args.pattern}")
        sys.exit(1)

    do_detect = not args.apply
    do_apply  = not args.detect
    rois = None

    if do_detect:
        rois = run_detect(files[0], args.crop_w, args.crop_h, out_dir,
                          min_dist=args.min_dist,
                          prominence_sigma=args.prominence)

    if do_apply:
        if rois is None:
            roi_path = out_dir / ROI_FILENAME
            if not roi_path.exists():
                print(f"ROI ファイルが見つかりません: {roi_path}")
                print("先に --detect を実行してください。")
                sys.exit(1)
            with open(roi_path, encoding="utf-8") as f:
                rois = json.load(f)
        run_apply(img_dir, args.pattern, rois, out_dir)


if __name__ == "__main__":
    main()
