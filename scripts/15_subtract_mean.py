# %%
import os, glob
import numpy as np
import tifffile as tiff

# ====== 入力 ======
mask_path = r"G:\250910_0\Pos4\output_phase\aligned_left_center\mask.tif"   # 単一マスク（整列済み）
seq_dir   = r"G:\250910_0\Pos4\output_phase\aligned_left_center\diff_from_first"  # シーケンス画像

# ====== ROI (x, y, w, h) ======
x, y, w, h = 316, 163, 128, 256
x2, y2 = x + w, y + h

# ====== 出力 ======
out_dir = os.path.join(seq_dir, "roi_subtracted_only_float32")
os.makedirs(out_dir, exist_ok=True)

# ====== 単一マスクの読み込み＆2値化 ======
MSK_raw = tiff.imread(mask_path)
if MSK_raw.ndim > 2:
    MSK_raw = MSK_raw[..., 0]
MSK = (MSK_raw > 0)

# シーケンス対象（mask.tifは除外）
frames = sorted(glob.glob(os.path.join(seq_dir, "*.tif")))
frames = [f for f in frames if os.path.basename(f).lower() != "mask.tif"]
if not frames:
    raise FileNotFoundError(f"シーケンス画像が見つかりません: {seq_dir}")

# 形状チェック
test = tiff.imread(frames[0]).astype(np.float32)
if test.shape[:2] != MSK.shape[:2]:
    raise ValueError(f"形状不一致: image {test.shape} vs mask {MSK.shape}")

H, W = test.shape[:2]
if not (0 <= x < W and 0 <= y < H and x2 > x and y2 > y):
    raise ValueError(f"ROIが範囲外: ROI=({x},{y},{w},{h}), image_shape=({H},{W})")

# マスクROIを切り出して固定
msk_roi = MSK[y:y2, x:x2]
if not msk_roi.any():
    raise ValueError("ROI内にマスク画素がありません。")

# ====== 各画像を処理 ======
for f in frames:
    IMG = tiff.imread(f).astype(np.float32)

    # ROIクロップ
    img_roi = IMG[y:y2, x:x2]

    # マスク領域平均
    mean_in_mask = float(img_roi[msk_roi].mean())

    # 減算
    out_roi = img_roi - mean_in_mask

    # 保存
    base = os.path.splitext(os.path.basename(f))[0]
    out_path = os.path.join(out_dir, f"{base}_roi_subtracted.tif")
    tiff.imwrite(out_path, out_roi.astype(np.float32))

    print(f"[OK] {os.path.basename(f)} mean_in_mask={mean_in_mask:.6f}")

print(f"✅ 完了: {len(frames)} 枚 → {out_dir}")

# %%
