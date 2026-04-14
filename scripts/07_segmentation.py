
# %%
# 251105
from cellpose_omni import io
from cellpose_omni.models import CellposeModel
import os, numpy as np, tifffile, traceback

# ==== 設定（必要に応じて変更） ====
# 入力ディレクトリ（推論対象画像）
indir = r"F:\260405\ph_260405\Pos2\output_phase\channels\crop_sub_rawraw\ch01"
# 出力ディレクトリ（マスク等の保存先）
outdir = os.path.join(indir, "inference_out")
os.makedirs(outdir, exist_ok=True)

# 学習済みモデルパス
model_path = r"C:\Users\QPI\Desktop\train\omni_model\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_omni_model_2026_04_13_10_54_41.173761"
# 推論設定（チューニング可能）
USE_GPU = True
NCHAN = 1
NCLASSES = 3

# Omnipose eval のハイパラ（kNNエラー対策で緩めに）
EVAL_PARAMS = dict(
    channels=None,
    channel_axis=None,
    diameter=30,
    normalize=True,    # 学習時と同じ1st-99th percentile正規化
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
