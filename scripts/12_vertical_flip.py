# %%
import os
import numpy as np
import tifffile as tiff

# Target folder
folder = r"C:\Users\QPI\Desktop\train"

# Scan files in folder
for filename in os.listdir(folder):
    if filename.lower().endswith(".tif"):
        in_path = os.path.join(folder, filename)
        out_path = os.path.join(folder, f"flippedud_{filename}")
        
        try:
            # Load
            img = tiff.imread(in_path)
            
            # Vertical flip
            flipped = np.flipud(img)
            
            # Save (preserve original dtype)
            tiff.imwrite(out_path, flipped)
            print(f"[OK] {out_path}")
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

print("Done.")

# %%
