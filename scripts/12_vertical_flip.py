# %%
import os
import numpy as np
import tifffile as tiff

# 対象フォルダ
folder = r"C:\Users\QPI\Desktop\verti_flip_train"

# フォルダ内のファイルを走査
for filename in os.listdir(folder):
    if filename.lower().endswith(".tif"):
        in_path = os.path.join(folder, filename)
        out_path = os.path.join(folder, f"flippedud_{filename}")
        
        try:
            # 読み込み
            img = tiff.imread(in_path)
            
            # 上下反転
            flipped = np.flipud(img)
            
            # 保存（元のdtype保持）
            tiff.imwrite(out_path, flipped)
            print(f"[OK] {out_path}")
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

print("完了しました。")

# %%
