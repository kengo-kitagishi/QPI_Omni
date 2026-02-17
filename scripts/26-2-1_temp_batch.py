# %%
#250604_Pos1~Pos30の位相画像のバッチ処理,250618_amp以外の画像を出力
#250604_Pos1~Pos30の位相画像のバッチ処理,250618_amp以外の画像を出力のコピー
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm
import matplotlib.pyplot as plt

# 定数設定
WAVELENGTH = 658e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
OFFAXIS_CENTER = (1664, 485) #250910 1623,1621 251017 1504,1710 251212 (1664, 485)

# ディレクトリ設定
BASE_DIR = r"F:\251212\ph_2"

BG_DIR = os.path.join(BASE_DIR, "Pos0")  # 背景画像があるディレクトリ

# Pos1〜Pos30 をループ
for pos_idx in range(29,47): #251017 46,92 250910 44,91
    pos_name = f"Pos{pos_idx}"
    TARGET_DIR = os.path.join(BASE_DIR, pos_name)
    
    if not os.path.exists(TARGET_DIR):
        print(f"{TARGET_DIR} が存在しません。スキップします。")
        continue

    OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
    OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_2, exist_ok=True)

    print(f"\n▶ 処理中: {pos_name}")

    for filename in tqdm(sorted(os.listdir(TARGET_DIR)), desc=pos_name):
        if filename.startswith("._"):
            continue  # macOSの不要な隠しファイルをスキップ

        if filename.lower().endswith(".tif") and "output" not in filename:
            filepath = os.path.join(TARGET_DIR, filename)
            bg_filepath = os.path.join(BG_DIR, filename)  # 対応する背景画像

            if not os.path.exists(bg_filepath):
                print(f"背景画像が見つかりません: {bg_filepath} をスキップします。")
                continue

            # 背景画像読み込み
            bg_img = Image.open(bg_filepath)
            bg_img = np.array(bg_img)
            #bg_img = bg_img[8:2056,416:2464] #250712_crop #250801_crop
            bg_img = bg_img[0:2048,0:2048] #250815_crop
            
            

            # パラメータ設定
            params = QPIParameters(
                wavelength=WAVELENGTH,
                NA=NA,
                img_shape=bg_img.shape,
                pixelsize=PIXELSIZE,
                offaxis_center=OFFAXIS_CENTER
            )
            field_bg = get_field(bg_img, params)
            angle_bg = unwrap_phase(np.angle(field_bg))

            # 対象画像読み込み
            img = Image.open(filepath)
            img = np.array(img)
            #img = img[8:2056,416:2464] #250712_crop #250801_crop
            img = img[0:2048,0:2048] #250712_crop #250801_crop


            # QPI再構成
            field = get_field(img, params)
            angle = unwrap_phase(np.angle(field))

            # 背景差分と平均0調整
            angle_nobg = angle - angle_bg
            angle_nobg -= np.mean(angle_nobg[1:507, 254:507])

            # TIF保存
            outpath = os.path.join(OUTPUT_DIR, filename.replace(".tif", "_phase.tif"))
            tifffile.imwrite(outpath, angle_nobg.astype(np.float32))

            # PNG保存（カラーマップ付き）
            plt.figure(figsize=(6, 6))
            plt.imshow(angle_nobg, cmap='viridis', vmin=-4, vmax=2)
            plt.colorbar(label='Phase (rad)')
            plt.title(f"Phase: {filename}")
            plt.axis('off')
            plt.tight_layout()
            png_outpath = os.path.join(OUTPUT_DIR_2, filename.replace(".tif", "_colormap.png"))
            plt.savefig(png_outpath, dpi=300)
            plt.close()

# %%
#250604_Pos1~Pos30の位相画像のバッチ処理,250618_amp以外の画像を出力のコピー
import os
import numpy as np
import tifffile
from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field
from tqdm import tqdm
import matplotlib.pyplot as plt

# 定数設定
WAVELENGTH = 658e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40
#OFFAXIS_CENTER = (1504,1708) 251017
#OFFAXIS_CENTER = (1623,1621) # 250910
OFFAXIS_CENTER = (1664, 485) #251212
# ディレクトリ設定
BASE_DIR =r"F:\251212\ph_2"
BG_DIR = os.path.join(BASE_DIR, "Pos0")  # 背景画像があるディレクトリ

# Pos1〜Pos30 をループ
for pos_idx in range(14,24): #251212 1,24 #251019 1,46 #250910 1,44
    pos_name = f"Pos{pos_idx}"
    TARGET_DIR = os.path.join(BASE_DIR, pos_name)
    
    if not os.path.exists(TARGET_DIR):
        print(f"{TARGET_DIR} が存在しません。スキップします。")
        continue

    OUTPUT_DIR = os.path.join(TARGET_DIR, "output_phase")
    OUTPUT_DIR_2 = os.path.join(TARGET_DIR, "output_colormap")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_2, exist_ok=True)

    print(f"\n▶ 処理中: {pos_name}")

    for filename in tqdm(sorted(os.listdir(TARGET_DIR)), desc=pos_name):
        if filename.startswith("._"):
            continue  # macOSの不要な隠しファイルをスキップ

        if filename.lower().endswith(".tif") and "output" not in filename:
            filepath = os.path.join(TARGET_DIR, filename)
            bg_filepath = os.path.join(BG_DIR, filename)  # 対応する背景画像

            if not os.path.exists(bg_filepath):
                print(f"背景画像が見つかりません: {bg_filepath} をスキップします。")
                continue

            # 背景画像読み込み
            bg_img = Image.open(bg_filepath)
            bg_img = np.array(bg_img)
            bg_img = bg_img[0:2048,416:2464] #250712_crop #250801_crop
            

            # パラメータ設定
            params = QPIParameters(
                wavelength=WAVELENGTH,
                NA=NA,
                img_shape=bg_img.shape,
                pixelsize=PIXELSIZE,
                offaxis_center=OFFAXIS_CENTER
            )
            field_bg = get_field(bg_img, params)
            angle_bg = unwrap_phase(np.angle(field_bg))

            # 対象画像読み込み
            img = Image.open(filepath)
            img = np.array(img)
            img = img[0:2048,416:2464] #250712_crop #250801_crop

            # QPI再構成
            field = get_field(img, params)
            angle = unwrap_phase(np.angle(field))

            # 背景差分と平均0調整
            angle_nobg = angle - angle_bg
            angle_nobg -= np.mean(angle_nobg[1:507, 1:253])

            # TIF保存
            outpath = os.path.join(OUTPUT_DIR, filename.replace(".tif", "_phase.tif"))
            tifffile.imwrite(outpath, angle_nobg.astype(np.float32))

            # PNG保存（カラーマップ付き）
            plt.figure(figsize=(6, 6))
            plt.imshow(angle_nobg, cmap='viridis', vmin=-4, vmax=2)
            plt.colorbar(label='Phase (rad)')
            plt.title(f"Phase: {filename}")
            plt.axis('off')
            plt.tight_layout()
            png_outpath = os.path.join(OUTPUT_DIR_2, filename.replace(".tif", "_colormap.png"))
            plt.savefig(png_outpath, dpi=300)
            plt.close()

# %%
