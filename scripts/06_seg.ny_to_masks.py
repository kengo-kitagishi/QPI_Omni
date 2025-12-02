# %% 251025
import os, glob, numpy as np
from cellpose_omni import io

def convert_seg_to_masks(root_dir: str, savedir: str | None = None, verbose: bool = True):
    """
    Omnipose/Cellpose GUIで保存された *_seg.npy から *_masks.tif を生成する。

    Parameters
    ----------
    root_dir : str
        *_seg.npy ファイルが存在するディレクトリ（サブフォルダは探索しません）
    savedir : str or None, default=None
        出力先ディレクトリ。Noneの場合は元ファイルと同じ場所に保存。
    verbose : bool, default=True
        True の場合は進行ログを表示。
    """

    seg_files = sorted(glob.glob(os.path.join(root_dir, "*_seg.npy")))
    if not seg_files:
        print(f"[INFO] No _seg.npy files found in: {root_dir}")
        return

    for seg_path in seg_files:
        try:
            data = np.load(seg_path, allow_pickle=True).item()
            img   = data.get("img", None)
            masks = data.get("masks", None)
            flows = data.get("flows", None)
            base_name = data.get("filename", None)

            if masks is None or flows is None:
                if verbose:
                    print(f"[SKIP] {os.path.basename(seg_path)} → 'masks' or 'flows' missing")
                continue

            # 出力ベース名を決定
            if base_name:
                base_noext = os.path.splitext(base_name)[0]
            else:
                base_noext = os.path.splitext(seg_path)[0].replace("_seg", "")

            # API仕様に合わせてリスト形式で渡す
            images     = [img if img is not None else np.zeros_like(masks, dtype=np.uint8)]
            masks_list = [masks.astype(np.int32, copy=False)]
            flows_list = [flows]
            file_names = [base_noext]

            # 公式API：tif=True で _cp_masks.tif を生成
            io.save_masks(
                images, masks_list, flows_list, file_names,
                png=False, tif=True,
                save_flows=False, save_outlines=False,
                savedir=savedir, omni=True
            )

            # 出力後に "_cp_masks.tif" → "_masks.tif" にリネーム
            out_dir = savedir or os.path.dirname(seg_path)
            cp_mask_path = os.path.join(out_dir, os.path.basename(base_noext) + "_cp_masks.tif")
            mask_path = os.path.join(out_dir, os.path.basename(base_noext) + "_masks.tif")

            if os.path.exists(cp_mask_path):
                os.rename(cp_mask_path, mask_path)
                if verbose:
                    print(f"[OK] {os.path.basename(mask_path)}")
            else:
                print(f"[WARN] _cp_masks.tif not found for: {seg_path}")

        except Exception as e:
            print(f"[ERR] {seg_path} -> {e}")

    print("Done.")

# 指定フォルダ直下の *_seg.npy から *_mask.tif を生成
convert_seg_to_masks(r"C:\Users\QPI\Desktop\train")

# %%
