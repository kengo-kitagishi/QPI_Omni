# %%
import os

# 検索・削除対象のディレクトリを指定
root = r"C:\Users\QPI\Desktop\train"

# 削除対象のキーワードと拡張子
target_key = "subtracted_cp_masks"
extensions = ("cp_masks.tif", "outline.tif")

# 再帰的に探索
for dirpath, dirnames, filenames in os.walk(root):
    for name in filenames:
        if target_key in name and name.endswith(extensions):
            file_path = os.path.join(dirpath, name)
            print(f"削除: {file_path}")
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"  ⚠️ 削除失敗: {e}")

print("完了しました。")

# %%
