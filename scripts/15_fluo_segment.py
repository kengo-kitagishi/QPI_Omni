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
indir = r"F:\251017_kk\BF50_GFP500_Bin2_3\Pos1\GFP"
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
    channels=None,       # GFP 1chなので None（=そのまま）
    rescale=None,        # リスケールしない
    mask_threshold=-2,   # 形をちょっと膨らませ気味（調整可）
    flow_threshold=0.0,  # 0〜0.4くらいで調整
    transparency=True,
    omni=True,
    cluster=True,        # DBSCANクラスタリング
    resample=True,
    verbose=False,
    tile=False,
    niter=None,
    augment=False,
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
model = CellposeModel(
    gpu=True,
    model_type="bact_fluor_omni",
    omni=True,
    nchan=1,
    nclasses=3,
    dim=2,
)
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
from cellpose_omni import io
from cellpose_omni.models import CellposeModel
import os, numpy as np, tifffile, traceback

# ==== ルートディレクトリ（Pos0, Pos1, ... が並んでいる場所）===
ROOT_DIR = r"F:\251017_kk\BF50_GFP500_Bin2_3"
#ROOT_DIR = r"F:\251108_kk\BF50_GFP500_Bin2_1"
ROOT_DIR = r"F:\251017_kk\BF50_GFP500_Bin2_1"
# 学習済みモデルパス（※今は未使用だが、使うなら CellposeModel(...) に pretrained_model= を渡す）
model_path = r"C:\Users\QPI\Desktop\verti_flip_train\omni_model\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_omni_model_2025_11_05_19_14_41.656097"

# 推論設定
USE_GPU = True
NCHAN = 1
NCLASSES = 3

# Omnipose eval のハイパラ（kNNエラー対策で緩めに）
EVAL_PARAMS = dict(
    channels=None,       # GFP 1chなので None（=そのまま）
    rescale=None,        # リスケールしない
    mask_threshold=-2,   # 形をちょっと膨らませ気味（調整可）
    flow_threshold=0.0,  # 0〜0.4くらいで調整
    transparency=True,
    omni=True,
    cluster=True,        # DBSCANクラスタリング
    resample=True,
    verbose=False,
    tile=False,
    niter=None,
    augment=False,
)

# ==== モデル読み込み（1回だけ）====
model = CellposeModel(
    gpu=USE_GPU,
    model_type="bact_fluor_omni",   # 自作モデルを使うなら model_type=None, pretrained_model=model_path に変更
    omni=True,
    nchan=NCHAN,
    nclasses=NCLASSES,
    dim=2,
)

# 全体集計用
total_files = 0
total_processed = 0
total_skipped = 0
total_errors = 0

# ==== Posフォルダの一覧取得 ====
pos_dirs = []
for name in os.listdir(ROOT_DIR):
    full = os.path.join(ROOT_DIR, name)
    if os.path.isdir(full) and name.startswith("Pos"):
        pos_dirs.append(name)

pos_dirs = sorted(pos_dirs)
print("Found Pos folders:", pos_dirs)

# ==== Posごとに処理 ====
for pos_name in pos_dirs:
    indir = os.path.join(ROOT_DIR, pos_name, "GFP")
    if not os.path.isdir(indir):
        print(f"[{pos_name}] GFP フォルダが見つかりませんでした: {indir}")
        continue

    outdir = os.path.join(indir, "inference_out")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n===== Processing {pos_name} ({indir}) =====")

    # ==== ファイル取得（安全化） ====
    files = io.get_image_files(indir, mask_filter='_masks', look_one_level_down=False)
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

    print(f"[{pos_name}] Found {len(files)} files for inference")
    if not files:
        continue

    skipped = 0
    error_count = 0
    processed = 0

    for i, f in enumerate(files, 1):
        fname = os.path.basename(f)

        try:
            img = tifffile.imread(f)
        except Exception as e:
            print(f"[{pos_name}][{i}/{len(files)}] {fname}  ❌ failed to read: {e}")
            traceback.print_exc()
            # 読めなかった場合は 1x1 の空マスクだけ残す（後処理の整合性用）
            empty_mask = np.zeros((1, 1), dtype=np.uint16)
            try:
                tifffile.imwrite(
                    os.path.join(outdir, fname.replace(".tif", "_masks.tif")),
                    empty_mask
                )
                tifffile.imwrite(
                    os.path.join(outdir, fname.replace(".tif", "_binary.tif")),
                    np.full_like(empty_mask, 255, dtype=np.uint8)
                )
            except Exception:
                pass
            error_count += 1
            continue

        print(f"[{pos_name}][{i}/{len(files)}] {fname}")

        try:
            # eval に渡す画像を整形
            img_eval = np.asarray(img)
            # 2次元 (Y, X) なら (1, Y, X) にして「1枚のスタック」として渡す
            if img_eval.ndim == 2:
                img_eval = img_eval[np.newaxis, ...]  # shape: (1, Y, X)

            masks, flows, _ = model.eval(img_eval, **EVAL_PARAMS)
        except ValueError as e:
            # kNN 周り等の ValueError（点が少なすぎる等）はスキップ扱い
            print(f"  ⚠ Skipping due to ValueError: {e}")
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(os.path.join(outdir, fname.replace(".tif", "_masks.tif")), empty_mask)
            tifffile.imwrite(
                os.path.join(outdir, fname.replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8)
            )
            skipped += 1
            continue
        except Exception as e:
            print(f"  ⚠ Skipped due to unexpected error: {e}")
            traceback.print_exc()
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(os.path.join(outdir, fname.replace(".tif", "_masks.tif")), empty_mask)
            tifffile.imwrite(
                os.path.join(outdir, fname.replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8)
            )
            error_count += 1
            continue

        # masks が None か、全ゼロ（細胞なし）をチェック
        if masks is None or (isinstance(masks, (list, tuple, np.ndarray)) and np.max(masks) == 0):
            print("  → No cells detected (saving empty mask).")
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(os.path.join(outdir, fname.replace(".tif", "_masks.tif")), empty_mask)
            tifffile.imwrite(
                os.path.join(outdir, fname.replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8)
            )
            skipped += 1
            continue

        # 正常系：マスク保存
        base = os.path.splitext(fname)[0]
        try:
            out_mask = masks[0].astype(np.uint16)
            tifffile.imwrite(os.path.join(outdir, f"{base}_masks.tif"), out_mask)

            # 輪郭画像作成（白背景=255, 境界=0）
            m = out_mask
            border = (
                (m != np.roll(m,  1, 0)) |
                (m != np.roll(m, -1, 0)) |
                (m != np.roll(m,  1, 1)) |
                (m != np.roll(m, -1, 1))
            ) & (m > 0)

            tifffile.imwrite(
                os.path.join(outdir, f"{base}_binary.tif"),
                np.where(border, 0, 255).astype(np.uint8)
            )
            processed += 1
        except Exception as e:
            print(f"  ❌ Failed to save outputs for {fname}: {e}")
            traceback.print_exc()
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(os.path.join(outdir, fname.replace(".tif", "_masks.tif")), empty_mask)
            tifffile.imwrite(
                os.path.join(outdir, fname.replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8)
            )
            error_count += 1

    # Posごとのサマリ
    print(f"\n=== Summary for {pos_name} ===")
    print(f"Files      : {len(files)}")
    print(f"Processed  : {processed}")
    print(f"Skipped    : {skipped} (no cells or kNN issue)")
    print(f"Errors     : {error_count}")
    print(f"Saved results under: {outdir}")

    total_files     += len(files)
    total_processed += processed
    total_skipped   += skipped
    total_errors    += error_count

# 全Posのサマリ
print("\n===== Global Inference Summary =====")
print(f"Total files : {total_files}")
print(f"Processed   : {total_processed}")
print(f"Skipped     : {total_skipped}")
print(f"Errors      : {total_errors}")
print("Root dir    :", ROOT_DIR)












# %%
from cellpose_omni import io
from cellpose_omni.models import CellposeModel
import os, numpy as np, tifffile, traceback

# ==== ルートディレクトリ（Pos0, Pos1, ... が並んでいる場所）===
ROOT_DIR = r"F:\251017_kk\BF50_GFP500_Bin2_1"
ROOT_DIR = r"F:\251108_kk\BF50_GFP500_Bin2_1"
#ROOT_DIR = r"F:\251017_kk\BF50_GFP500_Bin2_3"
# 学習済みモデルパス（※今は未使用だが、使うなら CellposeModel(...) に pretrained_model= を渡す）
model_path = r"C:\Users\QPI\Desktop\verti_flip_train\omni_model\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_omni_model_2025_11_05_19_14_41.656097"

# 推論設定
USE_GPU = True
NCHAN = 1
NCLASSES = 3

# Omnipose eval のハイパラ（kNNエラー対策で緩めに）
EVAL_PARAMS = dict(
    channels=None,       # GFP 1chなので None（=そのまま）
    rescale=None,        # リスケールしない
    mask_threshold=-2,   # 形をちょっと膨らませ気味（調整可）
    flow_threshold=0.0,  # 0〜0.4くらいで調整
    transparency=True,
    omni=True,
    cluster=True,        # DBSCANクラスタリング
    resample=True,
    verbose=False,
    tile=False,
    niter=None,
    augment=False,
)

# ==== モデル読み込み（1回だけ）====
model = CellposeModel(
    gpu=USE_GPU,
    model_type="bact_fluor_omni",   # 自作モデルを使うなら model_type=None, pretrained_model=model_path に変更
    omni=True,
    nchan=NCHAN,
    nclasses=NCLASSES,
    dim=2,
)

# 全体集計用
total_files = 0
total_processed = 0
total_skipped = 0
total_errors = 0

# ==== Posフォルダの一覧取得 ====
pos_dirs = []
for name in os.listdir(ROOT_DIR):
    full = os.path.join(ROOT_DIR, name)
    if os.path.isdir(full) and name.startswith("Pos"):
        pos_dirs.append(name)

pos_dirs = sorted(pos_dirs)
print("Found Pos folders:", pos_dirs)

# ==== Posごとに処理 ====
for pos_name in pos_dirs:
    indir = os.path.join(ROOT_DIR, pos_name, "GFP")
    if not os.path.isdir(indir):
        print(f"[{pos_name}] GFP フォルダが見つかりませんでした: {indir}")
        continue

    outdir = os.path.join(indir, "inference_out")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n===== Processing {pos_name} ({indir}) =====")

    # ==== ファイル取得（安全化） ====
    files = io.get_image_files(indir, mask_filter='_masks', look_one_level_down=False)
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

    print(f"[{pos_name}] Found {len(files)} files for inference")
    if not files:
        continue

    skipped = 0
    error_count = 0
    processed = 0

    for i, f in enumerate(files, 1):
        fname = os.path.basename(f)

        try:
            img = tifffile.imread(f)
        except Exception as e:
            print(f"[{pos_name}][{i}/{len(files)}] {fname}  ❌ failed to read: {e}")
            traceback.print_exc()
            # 読めなかった場合は 1x1 の空マスクだけ残す（後処理の整合性用）
            empty_mask = np.zeros((1, 1), dtype=np.uint16)
            try:
                tifffile.imwrite(
                    os.path.join(outdir, fname.replace(".tif", "_masks.tif")),
                    empty_mask
                )
                tifffile.imwrite(
                    os.path.join(outdir, fname.replace(".tif", "_binary.tif")),
                    np.full_like(empty_mask, 255, dtype=np.uint8)
                )
            except Exception:
                pass
            error_count += 1
            continue

        print(f"[{pos_name}][{i}/{len(files)}] {fname}")

        try:
            # ★ ここは素直に「リストで渡す」形に戻す
            masks, flows, _ = model.eval([img], **EVAL_PARAMS)
        except ValueError as e:
            # kNN 周り等の ValueError（点が少なすぎる等）はスキップ扱い
            print(f"  ⚠ Skipping due to ValueError: {e}")
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(os.path.join(outdir, fname.replace(".tif", "_masks.tif")), empty_mask)
            tifffile.imwrite(
                os.path.join(outdir, fname.replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8)
            )
            skipped += 1
            continue
        except Exception as e:
            print(f"  ⚠ Skipped due to unexpected error: {e}")
            traceback.print_exc()
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(os.path.join(outdir, fname.replace(".tif", "_masks.tif")), empty_mask)
            tifffile.imwrite(
                os.path.join(outdir, fname.replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8)
            )
            error_count += 1
            continue

        # masks が None か、全ゼロ（細胞なし）をチェック
        if masks is None or (isinstance(masks, (list, tuple, np.ndarray)) and np.max(masks) == 0):
            print("  → No cells detected (saving empty mask).")
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(os.path.join(outdir, fname.replace(".tif", "_masks.tif")), empty_mask)
            tifffile.imwrite(
                os.path.join(outdir, fname.replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8)
            )
            skipped += 1
            continue

        # 正常系：マスク保存
        base = os.path.splitext(fname)[0]
        try:
            # ★ ここから：必ず 2D (Y, X) に整形してから border を計算する
            m = np.asarray(masks[0])
            m = np.squeeze(m)  # 余分な次元を削る

            if m.ndim == 1:
                # 1Dベクトルだった場合、画像サイズに合わせて reshape を試みる
                if img.ndim >= 2:
                    h, w = img.shape[-2], img.shape[-1]
                    if m.size == h * w:
                        m = m.reshape(h, w)
                    else:
                        print(f"  ⚠ mask size {m.size} does not match image size {h*w}, saving empty mask instead.")
                        empty_mask = np.zeros_like(img, dtype=np.uint16)
                        tifffile.imwrite(os.path.join(outdir, f"{base}_masks.tif"), empty_mask)
                        tifffile.imwrite(
                            os.path.join(outdir, f"{base}_binary.tif"),
                            np.full_like(empty_mask, 255, dtype=np.uint8)
                        )
                        error_count += 1
                        continue

            if m.ndim != 2:
                # それでも2Dでなければ、マスクだけ諦めて空マスクを保存
                print(f"  ⚠ unexpected mask ndim={m.ndim}, saving empty mask instead.")
                empty_mask = np.zeros_like(img, dtype=np.uint16)
                tifffile.imwrite(os.path.join(outdir, f"{base}_masks.tif"), empty_mask)
                tifffile.imwrite(
                    os.path.join(outdir, f"{base}_binary.tif"),
                    np.full_like(empty_mask, 255, dtype=np.uint8)
                )
                error_count += 1
                continue

            out_mask = m.astype(np.uint16)
            tifffile.imwrite(os.path.join(outdir, f"{base}_masks.tif"), out_mask)

            # 輪郭画像作成（白背景=255, 境界=0）
            border = (
                (m != np.roll(m,  1, 0)) |
                (m != np.roll(m, -1, 0)) |
                (m != np.roll(m,  1, 1)) |
                (m != np.roll(m, -1, 1))
            ) & (m > 0)

            tifffile.imwrite(
                os.path.join(outdir, f"{base}_binary.tif"),
                np.where(border, 0, 255).astype(np.uint8)
            )
            processed += 1

        except Exception as e:
            print(f"Failed to save outputs for {fname}: {e}")
            traceback.print_exc()
            empty_mask = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(os.path.join(outdir, fname.replace(".tif", "_masks.tif")), empty_mask)
            tifffile.imwrite(
                os.path.join(outdir, fname.replace(".tif", "_binary.tif")),
                np.full_like(empty_mask, 255, dtype=np.uint8)
            )
            error_count += 1

    # Posごとのサマリ
    print(f"\n=== Summary for {pos_name} ===")
    print(f"Files      : {len(files)}")
    print(f"Processed  : {processed}")
    print(f"Skipped    : {skipped} (no cells or kNN issue)")
    print(f"Errors     : {error_count}")
    print(f"Saved results under: {outdir}")

    total_files     += len(files)
    total_processed += processed
    total_skipped   += skipped
    total_errors    += error_count

# 全Posのサマリ
print("\n===== Global Inference Summary =====")
print(f"Total files : {total_files}")
print(f"Processed   : {total_processed}")
print(f"Skipped     : {total_skipped}")
print(f"Errors      : {total_errors}")
print("Root dir    :", ROOT_DIR)



# %%
from cellpose_omni import io
from cellpose_omni.models import CellposeModel
import os, numpy as np, tifffile, traceback, re

# ==== 処理したい Pos 番号範囲 ====
START_POS = 50  # ← ここを書き換える
END_POS   = 51 # ← ここを書き換える

# ==== ルートディレクトリ ====
ROOT_DIR = r"F:\251108_kk\BF50_GFP500_Bin2_1"

# ==== モデル定義 ====
model = CellposeModel(
    gpu=True,
    model_type="bact_fluor_omni",
    omni=True,
    nchan=1,
    nclasses=3,
    dim=2,
)

EVAL_PARAMS = dict(
    channels=None,
    rescale=None,
    mask_threshold=-2,
    flow_threshold=0.0,
    transparency=True,
    omni=True,
    cluster=True,
    resample=True,
    verbose=False,
    tile=False,
    niter=None,
    augment=False,
)

# ==== Posフォルダ抽出 ====
pos_dirs = []
for name in os.listdir(ROOT_DIR):
    full = os.path.join(ROOT_DIR, name)
    if os.path.isdir(full) and name.startswith("Pos"):
        # ★ 数字部分を抽出
        m = re.search(r"Pos(\d+)", name)
        if m:
            num = int(m.group(1))
            if START_POS <= num <= END_POS:
                pos_dirs.append((num, name))

# 数値順に整列
pos_dirs = sorted(pos_dirs)
print("Selected Pos folders:", pos_dirs)

# ==== 全体集計 ====
total_files = total_processed = total_skipped = total_errors = 0

# ===== メインループ =====
for pos_num, pos_name in pos_dirs:

    indir = os.path.join(ROOT_DIR, pos_name, "GFP")
    if not os.path.isdir(indir):
        print(f"[{pos_name}] GFP フォルダが見つかりません")
        continue

    outdir = os.path.join(indir, "inference_out")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n===== Processing {pos_name} ({indir}) =====")

    files = io.get_image_files(indir, mask_filter='_masks', look_one_level_down=False)
    files = list(files) if not isinstance(files, list) else files

    print(f"[{pos_name}] Found {len(files)} files")
    if not files:
        continue

    skipped = error_count = processed = 0

    for i, f in enumerate(files, 1):

        fname = os.path.basename(f)
        print(f"[{pos_name}] [{i}/{len(files)}] {fname}")

        try:
            img = tifffile.imread(f)
        except Exception:
            error_count += 1
            continue

        # ==== Segmentation ====
        try:
            masks, flows, _ = model.eval([img], **EVAL_PARAMS)
        except:
            skipped += 1
            continue

        # ==== マスク存在チェック ====
        if masks is None or np.max(masks) == 0:
            skipped += 1
            continue

        # ==== マスク整形 ====
        m = np.squeeze(np.asarray(masks[0]))
        if m.ndim != 2:
            skipped += 1
            continue

        base = os.path.splitext(fname)[0]
        tifffile.imwrite(os.path.join(outdir, f"{base}_masks.tif"), m.astype(np.uint16))

        border = (
            (m != np.roll(m, 1, 0)) |
            (m != np.roll(m, -1, 0)) |
            (m != np.roll(m, 1, 1)) |
            (m != np.roll(m, -1, 1))
        ) & (m > 0)

        tifffile.imwrite(
            os.path.join(outdir, f"{base}_binary.tif"),
            np.where(border, 0, 255).astype(np.uint8)
        )

        processed += 1

    # ==== Pos ごとの結果 ====
    print(f"\n=== Summary {pos_name} ===")
    print("files   :", len(files))
    print("proc    :", processed)
    print("skip    :", skipped)
    print("errors  :", error_count)

    total_files     += len(files)
    total_processed += processed
    total_skipped   += skipped
    total_errors    += error_count

print("\n===== GLOBAL SUMMARY =====")
print("files   :", total_files)
print("proc    :", total_processed)
print("skip    :", total_skipped)
print("errors  :", total_errors)

# %%
