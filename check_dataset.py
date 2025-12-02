import os, glob, numpy as np, tifffile as tiff

root = r"C:\Users\QPI\Desktop\train"
bad = []

for ip in sorted(glob.glob(os.path.join(root, "*.tif"))):
    if ip.endswith("_masks.tif"):
        continue
    stem = os.path.splitext(os.path.basename(ip))[0]
    mp = os.path.join(root, stem + "_masks.tif")

    # ---- 画像の読み込み ----
    try:
        im = tiff.imread(ip)
        im = np.squeeze(im).astype(np.float32)
    except Exception as e:
        bad.append((stem, f"img_read:{e}"))
        continue

    # ---- マスク存在確認 ----
    if not os.path.exists(mp):
        bad.append((stem, "mask_missing"))
        continue

    # ---- マスク読み込み ----
    try:
        ms = tiff.imread(mp)
        ms = np.squeeze(ms)
    except Exception as e:
        bad.append((stem, f"mask_read:{e}"))
        continue

    # ---- 構造チェック ----
    if im.ndim != 2:
        bad.append((stem, f"img_ndim_{im.ndim}"))
    if ms.ndim != 2:
        bad.append((stem, f"mask_ndim_{ms.ndim}"))
    if im.shape != ms.shape:
        bad.append((stem, f"shape_mismatch {im.shape}!={ms.shape}"))

    # ---- 画像レンジチェック ----
    lo, hi = np.percentile(im, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        bad.append((stem, "flat_or_bad_percentile"))

    # ---- マスク内容チェック ----
    if ms.dtype.kind in "fc":
        bad.append((stem, "mask_float"))
    if np.max(ms) < 1:
        bad.append((stem, "mask_empty"))

# ---- 結果出力 ----
print("BAD:", len(bad))
for b in bad:
    print(b)
