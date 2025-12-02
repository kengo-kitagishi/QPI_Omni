# %%
import os
import shutil
from glob import glob

# ====== パス設定 ======
src_root = r"F:\250611_kk\channel_296.85_1"
dst_root = r"F:\250611_kk\ph_1"

# ====== ファイル名設定 ======
src_name = "img_000000001_Default_000.tif"
dst_name = "img_000000000_Default_000.tif"

# ====== Posフォルダをすべて探索 ======
pos_dirs = sorted(glob(os.path.join(src_root, "Pos*")))

if not pos_dirs:
    raise FileNotFoundError(f"Posフォルダが見つかりません: {src_root}")

for src_pos in pos_dirs:
    pos_name = os.path.basename(src_pos)
    dst_pos = os.path.join(dst_root, pos_name)

    src_file = os.path.join(src_pos, src_name)
    dst_file = os.path.join(dst_pos, dst_name)

    if not os.path.exists(src_file):
        print(f"[SKIP] {src_file} が存在しません。")
        continue

    if not os.path.exists(dst_pos):
        print(f"[WARN] {dst_pos} が存在しません。作成します。")
        os.makedirs(dst_pos, exist_ok=True)

    try:
        shutil.copy2(src_file, dst_file)
        print(f"[OK] {pos_name}: {os.path.basename(dst_file)} にコピー（上書き済）")
    except Exception as e:
        print(f"[ERR] {pos_name}: コピー失敗 -> {e}")

print("✅ 全Posフォルダの処理が完了しました。")

# %%
import os

# 対象ディレクトリ
root = r"C:\Users\QPI\Desktop\train"

for filename in os.listdir(root):
    if filename.endswith(".npy"):
        old_path = os.path.join(root, filename)
        new_name = filename.replace("_subtracted_subtracted_", "_subtracted_")
        new_path = os.path.join(root, new_name)

        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

print("Done.")

# %%
import os

# フォルダパス
folder = r"C:\Users\QPI\Desktop\0_train"

# 0～39までのファイルを順に処理
for i in range(40):
    old_name = f"1_0_train_masks{i:04d}.tif"
    new_name = f"1_0_train{i:04d}_subtracted_masks.tif"

    old_path = os.path.join(folder, old_name)
    new_path = os.path.join(folder, new_name)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"✅ {old_name} → {new_name}")
    else:
        print(f"⚠️ 見つかりません: {old_name}")

print("完了しました。")

# %%
