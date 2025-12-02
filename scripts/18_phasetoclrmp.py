# %%
import os
import glob
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm   # ★ 追加

# === 設定 ===
input_dir = r"C:\Users\QPI\Desktop\align_demo\output\subtracted"
output_dir = os.path.join(input_dir, "colormap")
os.makedirs(output_dir, exist_ok=True)

# カラーマップ設定
vmin = -0.1
vmax = 1.7
cmap = "RdBu_r"   # ←希望のカラーマップ

# ★ 追加：0を中心（白）にする正規化
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# === 全 TIF を処理 ===
tif_files = glob.glob(os.path.join(input_dir, "*.tif"))

for path in tif_files:
    # 画像読み込み
    img = tifffile.imread(path).astype(float)

    # 保存名
    fname = os.path.splitext(os.path.basename(path))[0]
    out_png = os.path.join(output_dir, f"{fname}_RdBu.png")

    # 描画
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap=cmap, norm=norm)   # ★ ここも変更
    plt.colorbar(label="Intensity (a.u.)")
    plt.title(fname)
    plt.axis("off")

    # 保存
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out_png)

print("=== 完了しました ===")

# %%
