# %%
import os

# Specify directory to search and delete from
root = r"C:\Users\QPI\Desktop\train"

# Target keyword and extensions for deletion
target_key = "subtracted_cp_masks"
extensions = ("cp_masks.tif", "outline.tif")

# Recursive search
for dirpath, dirnames, filenames in os.walk(root):
    for name in filenames:
        if target_key in name and name.endswith(extensions):
            file_path = os.path.join(dirpath, name)
            print(f"Delete: {file_path}")
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"  Delete failed: {e}")

print("Done.")

# %%
