# %%
"""
Simple alignment and subtraction script.
Processes .tif files in a specified directory in sorted order,
performing alignment and subtraction using the Nth image (or a directly specified path) as reference.
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
import json
import cv2
from tqdm import tqdm

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_tif_image(path):
    """Load TIF image and convert to float"""
    img = io.imread(path)
    return img.astype(np.float64)


def to_uint8(img, vmin=-5.0, vmax=2.0):
    """Convert to uint8 with fixed range (for alignment)"""
    clipped = np.clip(img, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)
    return (normalized * 255).astype(np.uint8)


def get_tif_files(directory):
    """
    Get .tif/.tiff files in directory in sorted order.

    Returns:
        list of str: List of full paths (sorted by name)
    """
    files = sorted([
        f for f in os.listdir(directory)
        if not f.startswith("._")
        and f.lower().endswith(('.tif', '.tiff'))
    ])
    return [os.path.join(directory, f) for f in files]


def process_timelapse(timelapse_dir, reference_img, tif_files,
                      method='ecc', save_png=True, vmin=-0.1, vmax=1.7,
                      cmap='RdBu_r', png_dpi=150, png_sample_interval=5):
    """
    Perform alignment and subtraction on timelapse image series.

    Parameters
    ----------
    timelapse_dir : str
        Directory of timelapse images
    reference_img : np.ndarray
        Reference image (float64)
    tif_files : list of str
        List of tif file paths to process
    method : str
        'ecc' or 'phase_correlation'
    save_png : bool
        Whether to save colormap PNGs
    vmin, vmax : float
        Colormap range
    cmap : str
        Colormap name
    png_dpi : int
        PNG resolution
    png_sample_interval : int
        Save PNG every N frames
    """
    # Create output directories
    aligned_dir = os.path.join(timelapse_dir, "aligned")
    subtracted_dir = os.path.join(timelapse_dir, "subtracted")
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(subtracted_dir, exist_ok=True)

    if save_png:
        colored_dir = os.path.join(timelapse_dir, "subtracted_colored")
        os.makedirs(colored_dir, exist_ok=True)

    alignment_results = []
    processed_count = 0
    skipped_count = 0
    png_saved_count = 0

    for tif_path in tqdm(tif_files, desc="Processing"):
        filename = os.path.basename(tif_path)
        try:
            timelapse_img = load_tif_image(tif_path)

            if timelapse_img.shape != reference_img.shape:
                print(f"\n  WARNING: Size mismatch, skipping: {filename}")
                skipped_count += 1
                continue

            # Alignment computation
            if method == 'ecc':
                ref_u8 = to_uint8(reference_img)
                tl_u8 = to_uint8(timelapse_img)

                warp_matrix = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-8)

                try:
                    correlation, warp_matrix = cv2.findTransformECC(
                        ref_u8, tl_u8, warp_matrix,
                        cv2.MOTION_TRANSLATION, criteria
                    )
                    shift_x = warp_matrix[0, 2]
                    shift_y = warp_matrix[1, 2]
                except Exception as e:
                    print(f"\n  WARNING: Alignment failed: {filename} - {e}")
                    skipped_count += 1
                    continue

            elif method == 'phase_correlation':
                from skimage import registration
                try:
                    shift, error, _ = registration.phase_cross_correlation(
                        reference_img, timelapse_img, upsample_factor=10
                    )
                    shift_y, shift_x = shift[0], shift[1]
                    correlation = 1.0 - error
                    warp_matrix = np.array([
                        [1.0, 0.0, shift_x],
                        [0.0, 1.0, shift_y]
                    ], dtype=np.float32)
                except Exception as e:
                    print(f"\n  WARNING: Alignment failed: {filename} - {e}")
                    skipped_count += 1
                    continue

            # Apply alignment
            h, w = timelapse_img.shape
            aligned_img = cv2.warpAffine(
                timelapse_img.astype(np.float32),
                warp_matrix, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            ).astype(np.float64)

            # Save
            base_name = os.path.splitext(filename)[0]

            aligned_path = os.path.join(aligned_dir, filename)
            io.imsave(aligned_path, aligned_img.astype(np.float32))

            subtracted = aligned_img - reference_img
            subtracted_path = os.path.join(subtracted_dir, f"{base_name}_subtracted.tif")
            io.imsave(subtracted_path, subtracted.astype(np.float32))

            alignment_results.append({
                'filename': filename,
                'warp_matrix': warp_matrix.tolist(),
                'shift_x': float(shift_x),
                'shift_y': float(shift_y),
                'correlation': float(correlation)
            })
            processed_count += 1

            # PNG save
            if save_png and (processed_count % png_sample_interval == 0):
                colored_path = os.path.join(colored_dir, f"{base_name}_subtracted.png")
                fig, ax = plt.subplots(figsize=(10, 8))
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                im = ax.imshow(subtracted, cmap=cmap, norm=norm)
                ax.axis('off')
                ax.set_title(f'{base_name}\nMean: {np.mean(subtracted):.3f}, '
                             f'Std: {np.std(subtracted):.3f}')
                plt.colorbar(im, ax=ax, fraction=0.046, label='Difference (rad)')
                plt.tight_layout()
                plt.savefig(colored_path, dpi=png_dpi, bbox_inches='tight')
                plt.close()
                png_saved_count += 1

        except Exception as e:
            print(f"\n  ERROR: {filename} - {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue

    # Save JSON
    json_path = os.path.join(timelapse_dir, "alignment_transforms.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'method': method,
            'num_processed': processed_count,
            'alignment_results': alignment_results
        }, f, indent=2, ensure_ascii=False)

    # Shift visualization
    if alignment_results:
        from shift_visualize import visualize_shifts
        visualize_shifts(json_path)

    # Summary
    print(f"\nProcessing complete")
    print(f"   Succeeded: {processed_count} images")
    if skipped_count > 0:
        print(f"   Skipped: {skipped_count} images")
    print(f"   aligned/: {aligned_dir}")
    print(f"   subtracted/: {subtracted_dir}")
    if save_png:
        print(f"   subtracted_colored/: {colored_dir} ({png_saved_count} images)")

    if alignment_results:
        shifts_y = [r['shift_y'] for r in alignment_results]
        shifts_x = [r['shift_x'] for r in alignment_results]
        corrs = [r['correlation'] for r in alignment_results]
        print(f"\n   Shift Y: mean={np.mean(shifts_y):.2f}px, "
              f"range=[{np.min(shifts_y):.2f}, {np.max(shifts_y):.2f}]")
        print(f"   Shift X: mean={np.mean(shifts_x):.2f}px, "
              f"range=[{np.min(shifts_x):.2f}, {np.max(shifts_x):.2f}]")
        print(f"   Correlation: mean={np.mean(corrs):.4f}, "
              f"range=[{np.min(corrs):.4f}, {np.max(corrs):.4f}]")


def main():
    # ========================================
    # Configuration parameters
    # ========================================

    # Directory of timelapse images (folder containing .tif files)
    TIMELAPSE_DIR = r"E:\Acuisition\kitagishi\260301\movetest_8\Pos4\cropped"
    # Reference image specification (priority: REFERENCE_IMAGE_PATH > REFERENCE_DIR+INDEX > TIMELAPSE_DIR+INDEX)
    # Method 1: Specify path directly
    #   Example: r"E:\Acuisition\kitagishi\260216\move_test_1\Pos1\img_000000200_ph_000.tif"
    REFERENCE_IMAGE_PATH = None
    # Method 2: Reference image directory + Nth image (1-based)
    #   If REFERENCE_DIR=None, use the Nth image in TIMELAPSE_DIR
    REFERENCE_DIR = None
    REFERENCE_INDEX = 150

    # Processing settings
    ALIGNMENT_METHOD = 'ecc'  # 'ecc' or 'phase_correlation'
    SAVE_PNG = True
    PNG_DPI = 150
    PNG_SAMPLE_INTERVAL = 1
    VMIN = -0.1
    VMAX = 1.7
    CMAP = 'RdBu_r'

    # ========================================

    print("=" * 80)
    print("Alignment and subtraction (simple version)")
    print("=" * 80)

    # Check directory
    if not os.path.exists(TIMELAPSE_DIR):
        print(f"\nERROR: Directory not found: {TIMELAPSE_DIR}")
        return

    # Get tif file list
    tif_files = get_tif_files(TIMELAPSE_DIR)
    if not tif_files:
        print(f"\nERROR: No .tif files found: {TIMELAPSE_DIR}")
        return

    print(f"\nDirectory: {TIMELAPSE_DIR}")
    print(f"Number of tif files: {len(tif_files)}")

    # Resolve reference image
    if REFERENCE_IMAGE_PATH is not None:
        if not os.path.exists(REFERENCE_IMAGE_PATH):
            print(f"\nERROR: Reference image not found: {REFERENCE_IMAGE_PATH}")
            return
        ref_path = REFERENCE_IMAGE_PATH
        print(f"Reference image (direct path): {ref_path}")
    else:
        # Use REFERENCE_DIR if specified, otherwise select from TIMELAPSE_DIR
        if REFERENCE_DIR is not None:
            if not os.path.exists(REFERENCE_DIR):
                print(f"\nERROR: Reference image directory not found: {REFERENCE_DIR}")
                return
            ref_candidates = get_tif_files(REFERENCE_DIR)
            ref_source = REFERENCE_DIR
        else:
            ref_candidates = tif_files
            ref_source = TIMELAPSE_DIR

        if not ref_candidates:
            print(f"\nERROR: No .tif files found: {ref_source}")
            return
        if REFERENCE_INDEX < 1 or REFERENCE_INDEX > len(ref_candidates):
            print(f"\nERROR: REFERENCE_INDEX={REFERENCE_INDEX} out of range (1~{len(ref_candidates)})")
            return
        ref_path = ref_candidates[REFERENCE_INDEX - 1]
        print(f"Reference image ({ref_source}, #{REFERENCE_INDEX}): {os.path.basename(ref_path)}")

    print(f"Alignment method: {ALIGNMENT_METHOD.upper()}")

    # Load reference image
    print(f"\nLoading reference image...")
    reference_img = load_tif_image(ref_path)
    print(f"  Size: {reference_img.shape}")

    # Execute processing
    process_timelapse(
        TIMELAPSE_DIR, reference_img, tif_files,
        method=ALIGNMENT_METHOD,
        save_png=SAVE_PNG,
        vmin=VMIN, vmax=VMAX, cmap=CMAP,
        png_dpi=PNG_DPI,
        png_sample_interval=PNG_SAMPLE_INTERVAL
    )


if __name__ == "__main__":
    main()

# %%
