"""
ecc_channel_inspect.py
----------------------
Visual inspection tool for per-channel ECC quality.

For each channel:
  - Recompute ECC shift (tx, ty) and correlation (pass 1 only)
  - Display the difference image warped with the channel-specific shift
  - Verify correspondence between MAD outlier detection and corr

Usage:
    python ecc_channel_inspect.py \\
        --sample "C:\\ph_1\\Pos1\\output_phase\\img_000000010_ph_000_phase.tif" \\
        --config "C:\\Users\\QPI\\Documents\\QPI_Omni\\drift_session\\drift_config.json"

Arguments:
    --sample   : Phase-reconstructed TIF (output_phase/*.tif). Assumes BG-subtracted.
    --config   : Path to drift_config.json
    --vmin     : Display lower limit (defaults to config ecc_vmin, or -2.0 if absent)
    --vmax     : Display upper limit (defaults to config ecc_vmax, or  2.0 if absent)
    --diff-vmin: Difference image display lower limit (default -1.0)
    --diff-vmax: Difference image display upper limit (default  3.0)
    --no-save  : Skip saving via figure_logger (plt.show() only)
"""

import sys
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import cv2
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---- Import functions from compute_drift_online.py ----
def _import_cdo(script_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "compute_drift_online",
        script_dir / "compute_drift_online.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(description="Per-channel ECC quality inspection")
    p.add_argument("--sample",   required=True, help="output_phase/*.tif (BG-subtracted phase)")
    p.add_argument("--config",   default=r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json")
    p.add_argument("--vmin",     type=float, default=None, help="ECC crop display lower limit")
    p.add_argument("--vmax",     type=float, default=None, help="ECC crop display upper limit")
    p.add_argument("--diff-vmin", type=float, default=-1.0, help="Difference image display lower limit")
    p.add_argument("--diff-vmax", type=float, default=3.0,  help="Difference image display upper limit")
    p.add_argument("--no-save",  action="store_true", help="Skip figure_logger saving")
    return p.parse_args()


def warp_translate(img: np.ndarray, tx: float, ty: float) -> np.ndarray:
    """Translate image by (tx, ty) and return."""
    h, w = img.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img.astype(np.float32), M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def main():
    args = parse_args()

    # ---- Load config ----
    cfg_path = Path(args.config)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    script_dir = Path(cfg.get("script_dir", cfg_path.parent.parent / "scripts"))
    cdo = _import_cdo(script_dir)

    vmin = args.vmin if args.vmin is not None else cfg.get("ecc_vmin", -5.0)
    vmax = args.vmax if args.vmax is not None else cfg.get("ecc_vmax",  2.0)
    diff_vmin = args.diff_vmin
    diff_vmax = args.diff_vmax

    # ---- ROI & reference crops ----
    rois_path = Path(cfg["channel_rois_json"])
    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)
    n_channels = len(rois)

    ref_crops_path = Path(cfg["grid_ref_crops_tif"])
    grid_ref_crops = tifffile.imread(str(ref_crops_path)).astype(np.float64)
    if grid_ref_crops.ndim == 2:
        grid_ref_crops = grid_ref_crops[np.newaxis, ...]
    print(f"grid_ref_crops: shape={grid_ref_crops.shape}  n_ch={n_channels}")

    # ---- Sample phase image ----
    sample_path = Path(args.sample)
    phase = tifffile.imread(str(sample_path)).astype(np.float64)
    print(f"sample: {sample_path.name}  shape={phase.shape}")

    # ---- Per-channel ECC (pass 1: grid(0,0) full-crop) ----
    results = []   # list of dict per channel
    tx_list, ty_list = [], []

    # Get crop size from grid_ref_crops shape (match actual reference image, not JSON crop_h)
    ref_crop_w, ref_crop_h = grid_ref_crops.shape[1], grid_ref_crops.shape[2]

    for ch_idx, roi in enumerate(rois):
        # Sample crop + per-channel backsub (extract at same size as reference image)
        sample_crop = cdo.extract_rect_roi(phase, roi["cy"], roi["cx"],
                                           ref_crop_w, ref_crop_h)
        offset = cdo.compute_backsub_offset(sample_crop, cfg)
        sample_crop = sample_crop + offset

        # Reference crop (grid_ref_crops[ch])
        ref_idx = min(ch_idx, len(grid_ref_crops) - 1)
        ref_crop = grid_ref_crops[ref_idx]

        # ECC
        ref_u8    = cdo.to_uint8(ref_crop,    vmin, vmax)
        sample_u8 = cdo.to_uint8(sample_crop, vmin, vmax)
        ecc_result = cdo.ecc_align(ref_u8, sample_u8)

        if ecc_result is not None:
            tx, ty, corr = ecc_result
            tx_list.append(tx)
            ty_list.append(ty)
            print(f"  ch{ch_idx:02d}: tx={tx:+.3f}px  ty={ty:+.3f}px  corr={corr:.4f}")
        else:
            tx, ty, corr = 0.0, 0.0, 0.0
            print(f"  ch{ch_idx:02d}: ECC failed")

        results.append({
            "ch": ch_idx,
            "roi": roi,
            "sample_crop": sample_crop,
            "ref_crop": ref_crop,
            "tx": tx, "ty": ty, "corr": corr,
            "ecc_ok": ecc_result is not None,
        })

    # ---- MAD outlier detection ----
    is_outlier = np.zeros(len(tx_list), dtype=bool)
    if len(tx_list) >= 3:
        out_x = cdo._remove_outliers_mad(tx_list)
        out_y = cdo._remove_outliers_mad(ty_list)
        is_outlier = out_x | out_y

    used_idx = [i for i, o in enumerate(is_outlier) if not o]
    excl_idx = [i for i, o in enumerate(is_outlier) if o]
    tx_avg = float(np.mean(np.array(tx_list)[used_idx])) if used_idx else 0.0
    ty_avg = float(np.mean(np.array(ty_list)[used_idx])) if used_idx else 0.0

    print(f"\nMAD excluded: ch{excl_idx} ({len(excl_idx)}/{len(tx_list)}ch)")
    print(f"Mean shift: tx={tx_avg:+.4f}px  ty={ty_avg:+.4f}px  ({len(used_idx)}ch used)")

    # Assign outlier flag to results
    for i, r in enumerate(results):
        r["is_outlier"] = bool(is_outlier[i]) if i < len(is_outlier) else False

    # ---- Compute difference images ----
    # ecc_diff    = warp(sample_crop, -tx, -ty) - ref_crop (channel-specific shift)
    # avg_diff    = warp(sample_crop, -tx_avg, -ty_avg) - ref_crop (mean shift)
    for r in results:
        # ECC's findTransformECC(ref, sample) returns the warp from ref to sample
        # To align sample back to ref, reverse the sign
        r["ecc_diff"]   = warp_translate(r["sample_crop"], -r["tx"], -r["ty"]) - r["ref_crop"]
        r["avg_diff"]   = warp_translate(r["sample_crop"], -tx_avg, -ty_avg)   - r["ref_crop"]

    # ---- Save ECC difference as 32-bit float TIF ----
    ecc_diff_stack = np.stack(
        [r["ecc_diff"].astype(np.float32) for r in results], axis=0
    )  # shape: (n_channels, H, W)
    out_dir = sample_path.parent.parent / "ecc_inspect"
    out_dir.mkdir(exist_ok=True)
    tif_out = out_dir / f"{sample_path.stem}_ecc_sub.tif"
    tifffile.imwrite(str(tif_out), ecc_diff_stack, photometric="minisblack")
    print(f"ECC difference TIF saved: {tif_out}")

    # ---- Rendering ----
    # Layout: n_channels rows x 3 columns
    #   col 0: sample crop
    #   col 1: ecc diff (channel-own shift)
    #   col 2: avg diff (MAD-averaged shift)
    n_rows = n_channels
    n_cols = 3
    fig_w = n_cols * 3.5
    fig_h = n_rows * 3.0 + 1.5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    col_titles = ["sample crop",
                  "ECC diff\n(per-channel shift)", "avg diff\n(MAD-averaged)"]
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=9, fontweight="bold")

    for row, r in enumerate(results):
        ch = r["ch"]
        tx, ty, corr = r["tx"], r["ty"], r["corr"]
        is_out = r["is_outlier"]
        color = "#d32f2f" if is_out else "#388e3c"   # red=excluded, green=used
        label = "EXCL" if is_out else "USED"

        images = [r["sample_crop"], r["ecc_diff"], r["avg_diff"]]
        vmins  = [vmin,       diff_vmin, diff_vmin]
        vmaxs  = [vmax,       diff_vmax, diff_vmax]

        for col, (img, _vmin, _vmax) in enumerate(zip(images, vmins, vmaxs)):
            ax = axes[row, col]
            ax.imshow(img, cmap="viridis", vmin=_vmin, vmax=_vmax,
                      interpolation="bilinear", origin="upper")
            ax.set_xticks([]); ax.set_yticks([])

            # Add channel number + stats as row label at col=0
            if col == 0:
                row_label = (f"ch{ch:02d}  tx={tx:+.2f}  ty={ty:+.2f}"
                             f"\ncorr={corr:.4f}  [{label}]")
                ax.set_ylabel(row_label, fontsize=7.5, color=color, rotation=0,
                              labelpad=90, ha="left", va="center")

            # Display corr value at top-right of ECC diff
            if col == 1:
                ax.text(0.97, 0.97, f"corr={corr:.3f}", transform=ax.transAxes,
                        fontsize=7, color=color, ha="right", va="top",
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

            # Show channel verdict via border color
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.0 if (col == 1) else 0.8)

    # Legend
    patch_used = mpatches.Patch(color="#388e3c", label=f"USED ({len(used_idx)}ch)")
    patch_excl = mpatches.Patch(color="#d32f2f", label=f"EXCL ({len(excl_idx)}ch)")
    fig.legend(handles=[patch_used, patch_excl], loc="upper right",
               fontsize=9, framealpha=0.9)

    # Title
    fig.suptitle(
        f"{sample_path.name}\n"
        f"avg shift: tx={tx_avg:+.4f}px  ty={ty_avg:+.4f}px  "
        f"({len(used_idx)}/{len(results)} ch used)",
        fontsize=10, y=1.002
    )
    fig.tight_layout()

    # ---- Save ----
    if args.no_save:
        out_path = out_dir / f"{sample_path.stem}_inspect.png"
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        print(f"\nSaved: {out_path}")
    else:
        try:
            sys.path.insert(0, str(script_dir))
            from figure_logger import save_figure
            frame_label = sample_path.stem.replace("img_", "t").replace("_ph_000_phase", "")
            save_figure(fig, params={
                "sample": str(sample_path),
                "n_channels": n_channels,
                "n_used": len(used_idx),
                "n_excl": len(excl_idx),
                "tx_avg_px": tx_avg,
                "ty_avg_px": ty_avg,
                "diff_vmin": diff_vmin,
                "diff_vmax": diff_vmax,
            }, description=f"ECC channel inspection: {frame_label}",
               copy_files=[str(tif_out)], dpi=300)
        except Exception as e:
            print(f"[WARNING] figure_logger failed ({e}) -> saving locally")
            out_path = sample_path.parent.parent.parent / "ecc_inspect" / f"{sample_path.stem}_inspect.png"
            out_path.parent.mkdir(exist_ok=True)
            fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
            print(f"Saved: {out_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
