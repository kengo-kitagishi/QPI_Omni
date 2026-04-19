# %% 251025
import os, glob, numpy as np
from cellpose_omni import io

def convert_seg_to_masks(root_dir: str, savedir: str | None = None, verbose: bool = True):
    """
    Generate *_masks.tif from *_seg.npy files saved by the Omnipose/Cellpose GUI.

    Parameters
    ----------
    root_dir : str
        Directory containing *_seg.npy files (sub-folders are not searched)
    savedir : str or None, default=None
        Output directory. If None, saves in the same location as the source file.
    verbose : bool, default=True
        If True, print progress logs.
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

            # Determine output base name
            if base_name:
                base_noext = os.path.splitext(base_name)[0]
            else:
                base_noext = os.path.splitext(seg_path)[0].replace("_seg", "")

            # Pass as lists to match the API specification
            images     = [img if img is not None else np.zeros_like(masks, dtype=np.uint8)]
            masks_list = [masks.astype(np.int32, copy=False)]
            flows_list = [flows]
            file_names = [base_noext]

            # Official API: tif=True generates _cp_masks.tif
            io.save_masks(
                images, masks_list, flows_list, file_names,
                png=False, tif=True,
                save_flows=False, save_outlines=False,
                savedir=savedir, omni=True
            )

            # Rename "_cp_masks.tif" to "_masks.tif" after output
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

# Generate *_mask.tif from *_seg.npy files in the specified folder
convert_seg_to_masks(r"C:\Users\QPI\Desktop\train")

# %%
