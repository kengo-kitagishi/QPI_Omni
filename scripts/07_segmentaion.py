# 02_train.py (skeleton)
# %%
動かない
from cellpose_omni import io
from cellpose_omni.models import CellposeModel
import os

indir = r"G:\250910_0\Pos4\output_phase\aligned_left_center\diff_from_first\roi_subtracted_only_float32"
outdir = os.path.join(indir, "inference_out")
os.makedirs(outdir, exist_ok=True)

model_path = r"C:\Users\QPI\Desktop\train\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_train_2025_10_28_00_47_05.556252"

# 画像ファイルの取得と読込（listを渡す）
image_files = io.get_image_files(indir, mask_filter='_masks', look_one_level_down=False)
images = [io.imread(f) for f in image_files]

# 1chモデルをGPUでロード
model = CellposeModel(
    gpu=True,
    pretrained_model=model_path,
    omni=True,
    nchan=1,       # ★ モデルと一致
    nclasses=3,
    dim=2
)

# 正規化オフで推論（channels/channel_axisは触らない＝1chのまま）
masks, flows, styles = model.eval(
    images,
    channels=None,
    channel_axis=None,
    normalize=False,      # 正規化だけオフ（CLIでは指定できない部分）
    diameter=0,
    omni=True,
    net_avg=True,         # CLIデフォルトと合わせる
    tile=False,            # CLIデフォルトと合わせる
    verbose=True
)

# 保存
for img, msk, flw, fname in zip(images, masks, flows, image_files):
    io.save_masks(
        img, msk, flw, fname,
        tif=True,
        save_outlines=True,
        savedir=outdir,
        omni=True
    )

# %% 逐次処理で都度保存
from cellpose_omni import io
from cellpose_omni.models import CellposeModel
import os, numpy as np, tifffile

#indir = r"G:\250910_0\Pos1\output_phase\aligned_left_center\diff_from_first\roi_subtracted_only_float32"
indir = r"G:\250815_kk\ph_1\Pos1\1_one_channel\subtracted_by_maskmean_float32"
outdir = os.path.join(indir, "inference_out")
os.makedirs(outdir, exist_ok=True)

#model_path = r"C:\Users\QPI\Desktop\train\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_train_2025_10_27_20_39_47.098522"
#model_path = r"C:\Users\QPI\Desktop\train\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_train_2025_11_04_22_09_31.831434"
#model_path = r"C:\Users\QPI\Desktop\train\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_train_2025_11_04_23_32_45.400042"
model_path = r"C:\Users\QPI\Desktop\train\omni_model\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_omni_model_2025_11_05_13_17_19.582477"

files = io.get_image_files(indir, mask_filter='_masks', look_one_level_down=False)
print(f"{len(files)} files"); assert files, "No images found."

model = CellposeModel(gpu=True, pretrained_model=model_path, omni=True, nchan=1, nclasses=3, dim=2)

for i, f in enumerate(files, 1):
    img = io.imread(f)
    print(f"[{i}/{len(files)}] {os.path.basename(f)}")

    masks, flows, _ = model.eval(
        [img],
        channels=None,
        channel_axis=None,
        diameter=0,
        normalize=False,   # 正規化OFF
        tile=False,        # 揺らぎを避けたいなら False
        net_avg=True,
        omni=True,
        verbose=False
    )

    base = os.path.splitext(os.path.basename(f))[0]

    # 1) マスク本体（uint16）
    tifffile.imwrite(os.path.join(outdir, f"{base}_masks.tif"),
                     masks[0].astype(np.uint16))

    # 2) （任意）輪郭だけ二値（白=255, 黒=0）を保存
    m = masks[0]
    border = ((m != np.roll(m,  1, 0)) |
              (m != np.roll(m, -1, 0)) |
              (m != np.roll(m,  1, 1)) |
              (m != np.roll(m, -1, 1))) & (m > 0)
    tifffile.imwrite(os.path.join(outdir, f"{base}_binary.tif"),
                     np.where(border, 0, 255).astype(np.uint8))

# %%
import os, glob
import numpy as np
import tifffile
import cv2

# 入力（元画像のあるフォルダ）と出力（*_masks.tif があるフォルダ）
indir  = r"G:\250910_0\Pos4\output_phase\aligned_left_center\diff_from_first\roi_subtracted_only_float32"
outdir = os.path.join(indir, "inference_out")

# 固定表示スケール（必要に応じて調整）
VMIN, VMAX = -0.05, 0.05     # 例：差分画像の想定レンジ
THICKNESS  = 1               # 輪郭の太さ（px）

def to_uint8_fixed(x, vmin, vmax):
    y = (x - vmin) / (vmax - vmin)
    return np.clip(y, 0, 1).astype(np.float32) * 255

imgs = sorted(glob.glob(os.path.join(indir, "*.tif")))
for i, img_path in enumerate(imgs, 1):
    base = os.path.splitext(os.path.basename(img_path))[0]
    msk_path = os.path.join(outdir, f"{base}_masks.tif")
    if not os.path.exists(msk_path):
        continue  # 同名のマスクが無ければスキップ

    # 読み込み
    img = tifffile.imread(img_path)       # 2D想定
    msk = tifffile.imread(msk_path)       # 2Dラベル

    # 背景の固定スケール表示（グレースケール→BGR）
    disp = to_uint8_fixed(img, VMIN, VMAX).astype(np.uint8)
    bgr  = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

    # 輪郭抽出（1px幅）
    border = ((msk != np.roll(msk,  1, 0)) |
              (msk != np.roll(msk, -1, 0)) |
              (msk != np.roll(msk,  1, 1)) |
              (msk != np.roll(msk, -1, 1))) & (msk > 0)

    # オーバーレイ（緑の輪郭）
    overlay = bgr.copy()
    overlay[border] = (0, 255, 0)
    if THICKNESS > 1:
        # 太さを出すために1回膨張
        k = np.ones((THICKNESS, THICKNESS), np.uint8)
        border_thick = cv2.dilate(border.astype(np.uint8), k) > 0
        overlay[border_thick] = (0, 255, 0)

    # 保存
    out_path = os.path.join(outdir, f"{base}_overlay.tif")
    tifffile.imwrite(out_path, overlay, photometric="rgb")
    print(f"[{i}] saved {os.path.basename(out_path)}")

# %%
# 02_train.py (fixed, official-style saving)
# %%
from cellpose_omni import io
from cellpose_omni.models import CellposeModel
import os

indir = r"G:\250910_0\Pos4\output_phase\aligned_left_center\diff_from_first\roi_subtracted_only_float32"
outdir = os.path.join(indir, "inference_out")
os.makedirs(outdir, exist_ok=True)

model_path = r"C:\Users\QPI\Desktop\train\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_train_2025_10_28_00_47_05.556252"

# 画像ファイルの取得（list を返す）
image_files = io.get_image_files(indir, mask_filter='_masks', look_one_level_down=False)
print(f"[INFO] {len(image_files)} files found")
assert len(image_files) > 0, "No images found in indir."

# 画像を読み込み（list）
images = [io.imread(f) for f in image_files]

# 1chモデルをGPUでロード
model = CellposeModel(
    gpu=True,
    pretrained_model=model_path,
    omni=True,
    nchan=1,
    nclasses=3,
    dim=2
)

# 推論（正規化オフ、tileはFalseのまま）
masks, flows, styles = model.eval(
    images,
    channels=None,
    channel_axis=None,
    normalize=False,   # 1–99%正規化OFF
    diameter=0,
    omni=True,
    net_avg=True,
    tile=False,
    verbose=True

)

# 公式スタイルで一括保存（リストをそのまま渡す）
io.save_masks(
    images=images,
    masks=masks,
    flows=flows,
    file_names=image_files,
    tif=True,                # *_masks.tif, *_flows.tif など
    save_outlines=True,      # オーバーレイ画像も出力
    savedir=outdir,          # ここに保存
    omni=True,
    channel_axis=None        # 2D単一chなので None
)

print(f"[DONE] saved to: {outdir}")
# %%
# 251105
from cellpose_omni import io
from cellpose_omni.models import CellposeModel
import os, numpy as np, tifffile, traceback

# ==== 設定（必要に応じて変更） ====
# 入力ディレクトリ（推論対象画像）
indir = r"G:\250815_kk\ph_1\Pos3\3_one_channel\subtracted_by_maskmean_float32"
# 出力ディレクトリ（マスク等の保存先）
outdir = os.path.join(indir, "inference_out")
os.makedirs(outdir, exist_ok=True)

# 学習済みモデルパス
model_path = r"C:\Users\QPI\Desktop\verti_flip_train\omni_model\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_omni_model_2025_11_05_19_14_41.656097"
model_path = r"C:\Users\QPI\Desktop\verti_flip_train\omni_model\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_omni_model_2025_11_05_19_14_41.656097"
# 推論設定（チューニング可能）
USE_GPU = True
NCHAN = 1
NCLASSES = 3

# Omnipose eval のハイパラ（kNNエラー対策で緩めに）
EVAL_PARAMS = dict(
    channels=None,
    channel_axis=None,
    diameter=30,
    normalize=False,   # 正規化OFF
    tile=False,        # タイル処理はしない（揺らぎ抑制）
    net_avg=True,
    omni=True,
    verbose=False,
    # 以下を調整して "点が少なすぎる" 問題を回避
    flow_threshold=0.11,   # デフォルトより低めにして検出点を確保
    mask_threshold=0,   # 低めでマスクを多めに拾う
    min_size=10           # 小さなゴミは除去
)

# ==== ファイル取得（安全化） ====
files = io.get_image_files(indir, mask_filter='_masks', look_one_level_down=False)

# generator の可能性を潰す & 要素取り出し
if not isinstance(files, (list, tuple)):
    files = list(files)

# io.get_image_files が (path,) で返す可能性に対応（要素がタプルなら先頭を使う）
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

# ==== モデル読み込み ====
model = CellposeModel(gpu=USE_GPU, pretrained_model=model_path, omni=True, nchan=NCHAN, nclasses=NCLASSES, dim=2)

skipped = 0
error_count = 0
processed = 0

for i, f in enumerate(files, 1):
    try:
        # 画像読み込み（tifffileで堅牢に）
        img = tifffile.imread(f)
    except Exception as e:
        print(f"[{i}/{len(files)}] {os.path.basename(f)}  ❌ failed to read: {e}")
        traceback.print_exc()
        # 出力用に空マスクを残しておく（必要であれば削除）
        empty_mask = np.zeros((1 if img is None else img.shape[0], 1 if img is None else img.shape[1]), dtype=np.uint16)
        try:
            tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")), empty_mask)
        except Exception:
            pass
        error_count += 1
        continue

    print(f"[{i}/{len(files)}] {os.path.basename(f)}")

    try:
        masks, flows, _ = model.eval([img], **EVAL_PARAMS)
    except ValueError as e:
        # ここで kNN 周り等の ValueError が出ることがある（点が少なすぎる等）
        print(f"  ⚠ Skipping due to ValueError: {e}")
        # 空マスクを保存しておく（後で一括処理しやすいように）
        empty_mask = np.zeros_like(img, dtype=np.uint16)
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")), empty_mask)
        # もし輪郭画像も揃えたい場合は白背景で保存
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                         np.full_like(empty_mask, 255, dtype=np.uint8))
        skipped += 1
        continue
    except Exception as e:
        # その他の例外はログに残してスキップ
        print(f"  ⚠ Skipped due to unexpected error: {e}")
        traceback.print_exc()
        empty_mask = np.zeros_like(img, dtype=np.uint16)
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")), empty_mask)
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                         np.full_like(empty_mask, 255, dtype=np.uint8))
        error_count += 1
        continue

    # masks が None か、全てゼロ（細胞なし）をチェック
    if masks is None or (isinstance(masks, (list, tuple, np.ndarray)) and np.max(masks) == 0):
        print("  → No cells detected (saving empty mask).")
        empty_mask = np.zeros_like(img, dtype=np.uint16)
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")), empty_mask)
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                         np.full_like(empty_mask, 255, dtype=np.uint8))
        skipped += 1
        continue

    # 正常系：マスク保存
    base = os.path.splitext(os.path.basename(f))[0]
    try:
        out_mask = masks[0].astype(np.uint16)
        tifffile.imwrite(os.path.join(outdir, f"{base}_masks.tif"), out_mask)
        # 輪郭保存（白背景=255、線は0 にする既存形式を踏襲）
        m = out_mask
        border = ((m != np.roll(m,  1, 0)) |
                  (m != np.roll(m, -1, 0)) |
                  (m != np.roll(m,  1, 1)) |
                  (m != np.roll(m, -1, 1))) & (m > 0)
        tifffile.imwrite(os.path.join(outdir, f"{base}_binary.tif"),
                         np.where(border, 0, 255).astype(np.uint8))
        processed += 1
    except Exception as e:
        print(f"  ❌ Failed to save outputs for {f}: {e}")
        traceback.print_exc()
        error_count += 1
        # attempt to save empty mask to keep outputs consistent
        empty_mask = np.zeros_like(img, dtype=np.uint16)
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")), empty_mask)
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                         np.full_like(empty_mask, 255, dtype=np.uint8))

print("=== Inference summary ===")
print(f"Total files : {len(files)}")
print(f"Processed   : {processed}")
print(f"Skipped     : {skipped} (no cells or kNN issue)")
print(f"Errors      : {error_count}")
print("Saved results to:", outdir)

# %%
