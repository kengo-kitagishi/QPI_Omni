# %%

import os, glob
import numpy as np
import tifffile as tiff

# ====== 入力 ======
#seq_dir  = r"G:\250910_0\Pos1\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\crop_128_256\diff_from_first"  #251101_0_Pos1
#mask_dir = r"G:\250910_0\Pos1\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\mask_for_2nd_backsub" #251101_0_Pos1
#seq_dir  = r"G:\250910_0\Pos3\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\crop_128_256\diff_from_first"  # 251101_0_Pos3
#mask_dir = r"G:\250910_0\Pos3\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\mask_for_2nd_backsub" # 251101_0_Pos3

#seq_dir = r"F:\250815_kk\ph_1\Pos1\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\crop_128_256\diff_from_first"
#mask_dir = r"F:\250815_kk\ph_1\Pos1\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\mask_for_2nd_backsub"

#seq_dir = r"G:\250815_kk\ph_1\Pos2\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\crop_128_256\diff_from_first"
#mask_dir = r"G:\250815_kk\ph_1\Pos2\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\mask_for_2nd_backsub"

#seq_dir = r"G:\250815_kk\ph_1\Pos3\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\crop_128_256\diff_from_first"
#mask_dir = r"G:\250815_kk\ph_1\Pos3\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\mask_for_2nd_backsub"

#seq_dir = r"H:\250910_0\Pos4\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\crop_128_256\diff_from_first"
#mask_dir = r"H:\250910_0\Pos4\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\mask_for_2nd_backsub"

#seq_dir = r"F:\250611_kk\ph_1\Pos1\output_phase\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\crop_128_256\diff_from_first"
#mask_dir = r"F:\250611_kk\ph_1\Pos1\output_phase\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\mask_for_2nd_backsub"

#seq_dir = r"F:\250611_kk\ph_1\Pos2\output_phase\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\crop_128_256\diff_from_first"
#mask_dir = r"F:\250611_kk\ph_1\Pos2\output_phase\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\mask_for_2nd_backsub"

#seq_dir = r"F:\250611_kk\ph_1\Pos3\output_phase\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\crop_128_256\diff_from_first"
#mask_dir = r"F:\250611_kk\ph_1\Pos3\output_phase\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\mask_for_2nd_backsub"

#seq_dir =r"G:\250815_kk\ph_1\Pos3\2_one_channel"
#mask_dir = r"G:\250815_kk\ph_1\Pos3\2_one_channel\mask_for_2_one_channel"

seq_dir =r"G:\250815_kk\ph_1\Pos3\3_one_channel"
mask_dir = r"G:\250815_kk\ph_1\Pos3\3_one_channel\mask_for_3_one_channel"

seq_dir =r"C:\Users\QPI\Desktop\0_train\for_sub_2"
mask_dir = r"C:\Users\QPI\Desktop\0_train\mask_for_2"

seq_dir =r"G:\250815_kk\ph_1\Pos1\mid_one_channel"
mask_dir = r"G:\250815_kk\ph_1\Pos1\mid_one_channel\mask_for_mid_one_channel"

seq_dir =r"F:\250611_kk\ph_1\Pos2\output_phase\1_one_channel"
mask_dir = r"F:\250611_kk\ph_1\Pos2\output_phase\1_one_channel\mask_for_1_one_channel"

out_dir  = os.path.join(seq_dir, "subtracted_by_maskmean_float32")
os.makedirs(out_dir, exist_ok=True)

# ====== ファイル対応 ======
frames = sorted(glob.glob(os.path.join(seq_dir, "*.tif")))
masks  = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

if not frames:
    raise FileNotFoundError(f"シーケンス画像が見つかりません: {seq_dir}")
if len(frames) != len(masks):
    print(f"⚠️ 警告: 枚数が一致しません → 画像={len(frames)}枚, マスク={len(masks)}枚")

# ====== 各ペアを処理 ======
processed = 0
skipped = 0

for f, m in zip(frames, masks):
    IMG = tiff.imread(f).astype(np.float32)
    MSK_raw = tiff.imread(m)

    if MSK_raw.ndim > 2:
        MSK_raw = MSK_raw[..., 0]
    MSK = (MSK_raw > 0)

    # サイズチェック
    if IMG.shape[:2] != MSK.shape[:2]:
        print(f"[SKIP] サイズ不一致: {os.path.basename(f)} {IMG.shape} vs {MSK.shape}")
        skipped += 1
        continue

    if not MSK.any():
        print(f"[SKIP] マスク領域なし: {os.path.basename(f)}")
        skipped += 1
        continue

    # マスク領域（白）の平均値を計算
    mean_in_mask = float(IMG[MSK].mean())

    # 平均値を画像全体から引く
    OUT = IMG - mean_in_mask

    # 保存
    base = os.path.splitext(os.path.basename(f))[0]
    out_path = os.path.join(out_dir, f"{base}_subtracted.tif")
    tiff.imwrite(out_path, OUT.astype(np.float32))

    processed += 1
    print(f"[OK] {os.path.basename(f)} mean_in_mask={mean_in_mask:.6f}")

print(f"✅ 完了: {processed}枚（スキップ: {skipped}枚） → {out_dir}")


# %%
