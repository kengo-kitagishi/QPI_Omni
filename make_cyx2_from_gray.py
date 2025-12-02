# make_cyx2_from_gray.py
# 単一チャネルの TIF (H×W) を、CYX (2×H×W) に複製して保存。
# マスク(_masks.tif)はそのままコピー（整数ID保持）。
# 失敗したTIFFは OpenCV フォールバックで読み直し。
import os, glob
import numpy as np
import tifffile as tiff
import cv2

# ====== 設定 ======
SRC = r"C:\Users\QPI\Desktop\train"           # 元データ（画像と*_masks.tif のペア）
DST = r"C:\Users\QPI\Desktop\train_dup2"      # 出力先（なければ作成）
DO_NORMALIZE = False                          # True: 1-99%で正規化してuint16保存
PCT_LOW, PCT_HIGH = 1, 99
# ===================

os.makedirs(DST, exist_ok=True)

def read_gray_safely(path):
    """tifffileで読んでダメならcv2でフォールバックし、2Dグレーにして返す"""
    try:
        arr = tiff.imread(path)
        if arr is None:
            raise IOError("tifffile returned None")
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            # 万一RGBならグレー化
            if arr.shape[-1] in (3,4):
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)  # tifffileはRGB順だがcv2はBGR前提。差は学習上ほぼ影響なし
            else:
                raise ValueError(f"unexpected 3D shape: {arr.shape}")
        if arr.ndim != 2:
            raise ValueError(f"not 2D after squeeze: {arr.shape}")
        return arr
    except Exception:
        # フォールバック
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError("cv2 read failed")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

def norm_1_99_to_uint16(im):
    im = im.astype(np.float32)
    im[np.isnan(im) | np.isinf(im)] = 0.0
    lo, hi = np.percentile(im, [PCT_LOW, PCT_HIGH])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # ほぼ一定など → そのまま返す（ダイナミックレンジが小さい場合は0除算回避）
        return np.clip(im, 0, np.max(im)).astype(np.uint16)
    x = (im - lo) / (hi - lo)
    x = np.clip(x, 0, 1)
    return (x * 65535.0 + 0.5).astype(np.uint16)

bad = []

images = sorted(glob.glob(os.path.join(SRC, "*.tif")))
for ipath in images:
    if ipath.endswith("_masks.tif"):
        continue
    stem = os.path.splitext(os.path.basename(ipath))[0]
    mpath = os.path.join(SRC, stem + "_masks.tif")
    if not os.path.exists(mpath):
        bad.append((stem, "mask_missing"))
        continue
    try:
        im = read_gray_safely(ipath)           # H×W
    except Exception as e:
        bad.append((stem, f"img_read:{e}"))
        continue

    # 正規化オプション
    if DO_NORMALIZE:
        im_u16 = norm_1_99_to_uint16(im)
    else:
        # そのままuint16へ（もともとuint16ならコピー、float等は丸め）
        im_u16 = im.astype(np.uint16, copy=False)

    # CYX = (2, H, W) に複製
    cyx = np.stack([im_u16, im_u16], axis=0)   # 2×H×W

    # マスク読み込み（整数ID保持）
    try:
        ms = tiff.imread(mpath)
        ms = np.squeeze(ms)
    except Exception as e:
        bad.append((stem, f"mask_read:{e}"))
        continue

    # 形状合わせ
    if cyx.shape[1:] != ms.shape:
        bad.append((stem, f"shape_mismatch img{cyx.shape[1:]} != mask{ms.shape}"))
        continue

    # 保存（非圧縮）
    out_img = os.path.join(DST, stem + ".tif")
    out_msk = os.path.join(DST, stem + "_masks.tif")
    # 画像はCYX (2,H,W) として保存
    tiff.imwrite(out_img, cyx, dtype=np.uint16)     # compression=None
    # マスクは整数IDのまま
    tiff.imwrite(out_msk, ms.astype(np.uint16))

# レポート
if bad:
    print("=== SKIPPED / PROBLEM FILES ===")
    for s, r in bad:
        print(f"{s} -> {r}")
else:
    print("All good.")
print(f"Output -> {DST}")
