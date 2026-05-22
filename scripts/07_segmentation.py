
# %%
# 251105
from cellpose_omni import io
from cellpose_omni.models import CellposeModel
import os, sys, argparse, numpy as np, tifffile, traceback

# ==== Settings (modify as needed) ====
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--indir", default=r"G:\マイドライブ\ch02")
_parser.add_argument("--model-path", default=None)
_parser.add_argument("--outdir", default=None)
_args, _ = _parser.parse_known_args()
indir = _args.indir
# Output directory (where masks etc. are saved)
outdir = _args.outdir if _args.outdir else os.path.join(indir, "inference_out")
os.makedirs(outdir, exist_ok=True)

# Trained model path
model_path = _args.model_path or r"C:\Users\QPI\Desktop\train\omni_model_d20\models\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_omni_model_d20_2026_05_01_19_01_52.321350"
# Inference settings (tunable)
USE_GPU = True
NCHAN = 1
NCLASSES = 3

# Early-exit: abort the whole channel if this many consecutive frames yield
# no cells (the mother machine channel has emptied out and won't refill).
# Set to None to disable.
NO_CELL_BREAK_AFTER = 100

# Omnipose eval hyperparameters (relaxed to mitigate kNN errors)
EVAL_PARAMS = dict(
    channels=None,
    channel_axis=None,
    diameter=20,
    normalize=True,    # Same 1st-99th percentile normalization as during training
    tile=False,        # Disable tiling (to suppress fluctuation)
    net_avg=True,
    omni=True,
    verbose=False,
    flow_threshold=0.4,
    mask_threshold=0,
    min_size=10
)

# ==== Get files (with safety handling) ====
files = io.get_image_files(indir, mask_filter='_masks', look_one_level_down=False)

# Convert generator to list if needed & extract elements
if not isinstance(files, (list, tuple)):
    files = list(files)

# Handle case where io.get_image_files returns (path,) tuples (use first element)
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

# ==== Load model ====
model = CellposeModel(gpu=USE_GPU, pretrained_model=model_path, omni=True, nchan=NCHAN, nclasses=NCLASSES, dim=2)

skipped = 0
error_count = 0
processed = 0
consecutive_empty = 0  # for NO_CELL_BREAK_AFTER early exit
aborted_early = False

for i, f in enumerate(files, 1):
    try:
        # Load image (robustly via tifffile)
        img = tifffile.imread(f)
    except Exception as e:
        print(f"[{i}/{len(files)}] {os.path.basename(f)}  ❌ failed to read: {e}")
        traceback.print_exc()
        # Leave an empty mask for output (remove if not needed)
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
        # ValueError can occur here from kNN etc. (too few points, etc.)
        print(f"  ⚠ Skipping due to ValueError: {e}")
        # Save an empty mask (to facilitate batch processing later)
        empty_mask = np.zeros_like(img, dtype=np.uint16)
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")), empty_mask)
        # Save with white background if contour images should also be aligned
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                         np.full_like(empty_mask, 255, dtype=np.uint8))
        skipped += 1
        continue
    except Exception as e:
        # Log other exceptions and skip
        print(f"  ⚠ Skipped due to unexpected error: {e}")
        traceback.print_exc()
        empty_mask = np.zeros_like(img, dtype=np.uint16)
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")), empty_mask)
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                         np.full_like(empty_mask, 255, dtype=np.uint8))
        error_count += 1
        continue

    # Check if masks is None or all zeros (no cells)
    if masks is None or (isinstance(masks, (list, tuple, np.ndarray)) and np.max(masks) == 0):
        print("  → No cells detected (saving empty mask).")
        empty_mask = np.zeros_like(img, dtype=np.uint16)
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_masks.tif")), empty_mask)
        tifffile.imwrite(os.path.join(outdir, os.path.basename(f).replace(".tif", "_binary.tif")),
                         np.full_like(empty_mask, 255, dtype=np.uint8))
        skipped += 1
        consecutive_empty += 1
        if NO_CELL_BREAK_AFTER is not None and consecutive_empty >= NO_CELL_BREAK_AFTER:
            print(f"  → {consecutive_empty} consecutive empty frames "
                  f">= NO_CELL_BREAK_AFTER ({NO_CELL_BREAK_AFTER}); "
                  f"aborting channel after frame {i}/{len(files)}.")
            aborted_early = True
            break
        continue
    consecutive_empty = 0  # got a real detection; reset the counter

    # Normal case: save masks
    base = os.path.splitext(os.path.basename(f))[0]
    try:
        out_mask = masks[0].astype(np.uint16)
        tifffile.imwrite(os.path.join(outdir, f"{base}_masks.tif"), out_mask)
        # Save contours (following existing format: white background=255, lines=0)
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
if aborted_early:
    print(f"Aborted early: yes (>= {NO_CELL_BREAK_AFTER} consecutive empty frames)")
print("Saved results to:", outdir)

# %%
