# %%
import os
import glob
import re
import numpy as np
import tifffile as tiff
# %%
# ====== パス ======
seq_dir  = r"G:\250815_kk\ph_1\Pos1\mid_one_channel"
mask_dir = r"G:\250815_kk\ph_1\Pos1\mid_one_channel\mask_for_mid_one_channel"  # backsubに用いたマスク
out_dir  = os.path.join(seq_dir, "subtracted_by_maskmean_float32")

# ====== ナチュラルソート（数字を数値として並べる） ======
_nsre = re.compile('([0-9]+)')
def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]

def list_tifs_sorted(folder, pattern="*.tif"):
    paths = glob.glob(os.path.join(folder, pattern))
    paths.sort(key=natural_key)
    return paths

# ====== 減算後画像、元フレーム、マスクを取得 ======
sub_imgs = list_tifs_sorted(out_dir, "*_subtracted.tif")
if not sub_imgs:
    raise FileNotFoundError(f"減算後画像が見つかりません: {out_dir}")

frames   = list_tifs_sorted(seq_dir, "*.tif")
masks    = list_tifs_sorted(mask_dir, "*.tif")

if not frames or not masks:
    raise FileNotFoundError(f"元フレーム or マスクが見つかりません: frames={len(frames)} masks={len(masks)}")

if len(frames) != len(masks):
    print(f"⚠️ 警告: 減算時と同じ対応を仮定しますが、枚数が一致しません → frames={len(frames)}, masks={len(masks)}")

# ====== 減算時と同じ対応 (frame_i ↔ mask_i) を辞書化 ======
def basename_wo_ext(p):
    return os.path.splitext(os.path.basename(p))[0]

frame_to_mask = {}
for f_path, m_path in zip(frames, masks):
    frame_to_mask[basename_wo_ext(f_path)] = m_path

# ====== 背景(mean over mask==0) を各フレームで計算 ======
bg_means = []  # (index_1based_in_subimgs, sub_path, bg_mean)
skipped_no_pair = 0
skipped_size = 0
skipped_nobg = 0

for idx_1, sub_path in enumerate(sub_imgs, start=1):
    base_sub = basename_wo_ext(sub_path)
    if not base_sub.endswith("_subtracted"):
        # 念のため
        continue
    base_orig = base_sub[:-len("_subtracted")]  # 減算前フレーム名

    mpath = frame_to_mask.get(base_orig)
    if mpath is None:
        # フレーム名と完全一致しないケースに備えて、近い名前の候補を探す（任意のフォールバック）
        # 例: "mid_one_channel_0005" に対して、前方一致するマスクを探す
        candidates = [p for p in masks if basename_wo_ext(p).startswith(base_orig)]
        if candidates:
            mpath = candidates[0]
        else:
            print(f"[SKIP] 対応マスクなし(対応表にも候補にも存在せず): {base_orig}")
            skipped_no_pair += 1
            continue

    img = tiff.imread(sub_path).astype(np.float32)
    msk_raw = tiff.imread(mpath)
    if msk_raw.ndim > 2:
        msk_raw = msk_raw[..., 0]
    msk = (msk_raw > 0)

    if img.shape[:2] != msk.shape[:2]:
        print(f"[SKIP] サイズ不一致: {os.path.basename(sub_path)} {img.shape} vs {msk.shape}")
        skipped_size += 1
        continue

    bg = ~msk  # mask==0
    if not bg.any():
        print(f"[SKIP] 背景なし(mask==0が無い): {os.path.basename(sub_path)}")
        skipped_nobg += 1
        continue

    bg_means.append((idx_1, sub_path, float(img[bg].mean())))

print(f"計算完了: {len(bg_means)}枚  | スキップ: 対応なし{skipped_no_pair}, サイズ不一致{skipped_size}, 背景なし{skipped_nobg}")

# ====== 範囲別の平均 ======
def mean_in_range(vals, lo, hi):
    sel = [v[2] for v in vals if lo <= v[0] <= hi]
    if not sel:
        return np.nan, 0
    return float(np.mean(sel)), len(sel)

ranges = [(100,1100), (1200,1400), (1500,1700),(2050,2500)]
for lo, hi in ranges:
    m, n = mean_in_range(bg_means, lo, hi)
    if np.isnan(m):
        print(f"[結果] {lo}〜{hi}枚目: 対象フレームなし (n={n})")
    else:
        print(f"[結果] {lo}〜{hi}枚目: 背景平均の平均 = {m:.6f} （n={n}）")

# %%
