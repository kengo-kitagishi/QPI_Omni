"""
eval_tilt_correct_ecc_precision.py
-----------------------------------
Evaluate the effect of X-tilt correction on ECC precision
using ph_3 (static sample) as ground truth.

Issues with the old script (analyze_ch_xtilt_corrected_ecc.py):
  1. Used grid_ref_crops.tif (_tilt_correct applied, 30x80) as "Raw ref"
     -> test images are 30x120, size mismatch -> findTransformECC fails on all frames
  2. Grid ref and ph_3 have different FOVs -> ECC does not converge
  3. Tilt correction function in the analysis differs from pipeline's _tilt_correct
  4. Grid ref z index and dataset do not match the pipeline

This script's approach:
  - **Self-reference**: Use ph_3 frame 0 as ref (same FOV ensures ECC convergence)
  - channel_rois = 260331 Pos1 (same ROI definition as pipeline)
  - Raw condition: neither ref nor test has tilt correction (simple crop + backsub)
  - Corr condition: both ref and test use _tilt_correct (same function as pipeline)
  - Static sample, so true shift = 0 -> ECC output = measurement error directly

Metrics:
  A) per-channel mean |shift| [um] -- ECC precision per channel
  B) per-frame inter-channel std [um] -- inter-channel variability
  C) per-frame channel-mean shift [um] -- systematic error
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from compute_drift_online import (
    _tilt_correct,
    compute_backsub_offset,
    extract_rect_roi,
    to_uint8,
    ecc_align,
)
from figure_logger import save_figure

# ============================================================
# Configuration (must match pipeline exactly)
# ============================================================
PHASE_DIR = Path(r"D:\AquisitionData\Kitagishi\basler_image_seq\ph_3\Pos0\output_phase")
ROIS_JSON = Path(
    r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1"
    r"\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"
)

PIXEL_SCALE_UM = 0.34567514677103717
ECC_VMIN       = -5.0
ECC_VMAX       =  2.0

# Tilt correction parameters (must match drift_config.json)
TILT_CROP_H = 270
ECC_CROP_H  = 80
FIT_RIGHT   = False   # pos_split > center -> left side is BG
CROP_W      = 40       # Y-direction crop width (overrides ROI crop_w)

N_MAX_TP = None   # None for all frames (99 TP)
# ============================================================


def main():
    # ---- Load data ----
    phase_paths = sorted(PHASE_DIR.glob("img_*_phase.tif"))
    if not phase_paths:
        print(f"ERROR: {PHASE_DIR}")
        sys.exit(1)
    if N_MAX_TP is not None:
        phase_paths = phase_paths[:N_MAX_TP + 1]

    rois = json.loads(ROIS_JSON.read_text(encoding="utf-8"))
    n_ch = len(rois)
    n_tp = len(phase_paths) - 1
    print(f"Phase images: {len(phase_paths)}, Channels: {n_ch}, TPs: {n_tp}")
    print(f"Self-reference: frame 0 of ph_3")
    print(f"CROP_W={CROP_W}, TILT_CROP_H={TILT_CROP_H}, ECC_CROP_H={ECC_CROP_H}, FIT_RIGHT={FIT_RIGHT}")

    # ---- Use frame 0 as reference ----
    ref_img = tifffile.imread(str(phase_paths[0])).astype(np.float64)
    cfg_dummy = {}

    # Raw ref: simple crop + backsub (no tilt correction) -> (CROP_W, ECC_CROP_H)
    ref_raw_u8 = []
    for roi in rois:
        crop = extract_rect_roi(
            ref_img, roi["cy"], roi["cx"], CROP_W, ECC_CROP_H
        ).astype(np.float64)
        offset = compute_backsub_offset(crop, cfg_dummy)
        ref_raw_u8.append(to_uint8(crop + offset, ECC_VMIN, ECC_VMAX))

    # Corr ref: _tilt_correct (same as pipeline) -> (CROP_W, ECC_CROP_H)
    ref_corr_u8 = []
    for roi in rois:
        crop = _tilt_correct(
            ref_img, roi["cy"], roi["cx"], CROP_W,
            TILT_CROP_H, ECC_CROP_H, fit_right=FIT_RIGHT
        )
        ref_corr_u8.append(to_uint8(crop, ECC_VMIN, ECC_VMAX))

    print(f"\nRef crops: raw u8 shape={ref_raw_u8[0].shape}, "
          f"corr u8 shape={ref_corr_u8[0].shape}")

    # ---- ECC alignment ----
    # tx = image-X shift, ty = image-Y shift
    tx_raw  = np.full((n_ch, n_tp), np.nan)
    ty_raw  = np.full((n_ch, n_tp), np.nan)
    tx_corr = np.full((n_ch, n_tp), np.nan)
    ty_corr = np.full((n_ch, n_tp), np.nan)

    for t_idx, path in enumerate(phase_paths[1:]):
        try:
            phase_tp = tifffile.imread(str(path)).astype(np.float64)
        except Exception:
            continue

        for c_idx, roi in enumerate(rois):
            # --- Raw: simple crop + backsub (CROP_W x ECC_CROP_H) ---
            crop_r = extract_rect_roi(
                phase_tp, roi["cy"], roi["cx"], CROP_W, ECC_CROP_H
            ).astype(np.float64)
            off_r = compute_backsub_offset(crop_r, cfg_dummy)
            res_r = ecc_align(ref_raw_u8[c_idx],
                              to_uint8(crop_r + off_r, ECC_VMIN, ECC_VMAX))
            if res_r is not None:
                tx_raw[c_idx, t_idx] = res_r[0]
                ty_raw[c_idx, t_idx] = res_r[1]

            # --- Corr: _tilt_correct (same as pipeline) ---
            crop_c = _tilt_correct(
                phase_tp, roi["cy"], roi["cx"], CROP_W,
                TILT_CROP_H, ECC_CROP_H, fit_right=FIT_RIGHT
            )
            res_c = ecc_align(ref_corr_u8[c_idx],
                              to_uint8(crop_c, ECC_VMIN, ECC_VMAX))
            if res_c is not None:
                tx_corr[c_idx, t_idx] = res_c[0]
                ty_corr[c_idx, t_idx] = res_c[1]

        if (t_idx + 1) % 20 == 0:
            print(f"  TP {t_idx+1}/{n_tp}", flush=True)

    # Convert to um
    tx_raw_um  = tx_raw  * PIXEL_SCALE_UM
    ty_raw_um  = ty_raw  * PIXEL_SCALE_UM
    tx_corr_um = tx_corr * PIXEL_SCALE_UM
    ty_corr_um = ty_corr * PIXEL_SCALE_UM

    # ---- Statistics ----
    # A) per-channel mean |shift| [um] -- direct ECC precision metric
    abs_tx_raw  = np.nanmean(np.abs(tx_raw_um),  axis=1)  # (n_ch,)
    abs_ty_raw  = np.nanmean(np.abs(ty_raw_um),  axis=1)
    abs_tx_corr = np.nanmean(np.abs(tx_corr_um), axis=1)
    abs_ty_corr = np.nanmean(np.abs(ty_corr_um), axis=1)

    # B) per-frame inter-channel std [um] -- inter-channel variability
    std_tx_raw  = np.nanstd(tx_raw_um,  axis=0)   # (n_tp,)
    std_ty_raw  = np.nanstd(ty_raw_um,  axis=0)
    std_tx_corr = np.nanstd(tx_corr_um, axis=0)
    std_ty_corr = np.nanstd(ty_corr_um, axis=0)

    # C) per-frame channel-mean shift [um] -- systematic error
    mean_tx_raw  = np.nanmean(tx_raw_um,  axis=0)
    mean_ty_raw  = np.nanmean(ty_raw_um,  axis=0)
    mean_tx_corr = np.nanmean(tx_corr_um, axis=0)
    mean_ty_corr = np.nanmean(ty_corr_um, axis=0)

    # Overall statistics
    print("\n========== ECC precision (static sample ph_3) ==========")
    print(f"Grid ref: 260331 Pos1 z=10 (pipeline-consistent)")
    print(f"Conditions: Raw=crop+backsub, Corr=_tilt_correct(270→80)")
    print()
    print(f"{'Metric':<40} {'Raw':>8} {'Corr':>8} {'Unit'}")
    print("-" * 65)
    print(f"{'Mean |ECC X shift| (all ch×TP)':<40} "
          f"{np.nanmean(np.abs(tx_raw_um)):.4f} {np.nanmean(np.abs(tx_corr_um)):.4f} um")
    print(f"{'Mean |ECC Y shift| (all ch×TP)':<40} "
          f"{np.nanmean(np.abs(ty_raw_um)):.4f} {np.nanmean(np.abs(ty_corr_um)):.4f} um")
    print(f"{'Std of ECC X shift (all ch×TP)':<40} "
          f"{np.nanstd(tx_raw_um):.4f} {np.nanstd(tx_corr_um):.4f} um")
    print(f"{'Std of ECC Y shift (all ch×TP)':<40} "
          f"{np.nanstd(ty_raw_um):.4f} {np.nanstd(ty_corr_um):.4f} um")
    print(f"{'Mean inter-ch std (X)':<40} "
          f"{np.nanmean(std_tx_raw):.4f} {np.nanmean(std_tx_corr):.4f} um")
    print(f"{'Mean inter-ch std (Y)':<40} "
          f"{np.nanmean(std_ty_raw):.4f} {np.nanmean(std_ty_corr):.4f} um")
    print()

    # per-channel summary
    print("Per-channel mean |shift| [um]:")
    print(f"  {'ch':<4} {'|tx| raw':>8} {'|tx| corr':>10} {'|ty| raw':>8} {'|ty| corr':>10}")
    for c in range(n_ch):
        print(f"  {c:<4} {abs_tx_raw[c]:>8.4f} {abs_tx_corr[c]:>10.4f} "
              f"{abs_ty_raw[c]:>8.4f} {abs_ty_corr[c]:>10.4f}")

    # ---- Figure ----
    tp_axis = np.arange(n_tp)
    COLORS  = plt.cm.tab20(np.linspace(0, 1, n_ch))

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 3, hspace=0.50, wspace=0.38,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    # ---- Row 0: ECC X ----
    # (A) per-channel ECC X — Raw
    ax00 = fig.add_subplot(gs[0, 0])
    for c in range(n_ch):
        ax00.plot(tp_axis, tx_raw_um[c], lw=0.5, alpha=0.7, color=COLORS[c])
    ax00.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax00.set_xlabel("TP"); ax00.set_ylabel("ECC X shift (um)")
    ax00.set_title("Per-ch ECC X — Raw (crop+backsub)", fontsize=9)
    ax00.spines["top"].set_visible(False); ax00.spines["right"].set_visible(False)

    # (B) per-channel ECC X — Corrected
    ax01 = fig.add_subplot(gs[0, 1])
    for c in range(n_ch):
        ax01.plot(tp_axis, tx_corr_um[c], lw=0.5, alpha=0.7, color=COLORS[c])
    ax01.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax01.set_xlabel("TP"); ax01.set_ylabel("ECC X shift (um)")
    ax01.set_title("Per-ch ECC X — Tilt-corrected", fontsize=9)
    ax01.spines["top"].set_visible(False); ax01.spines["right"].set_visible(False)

    # (C) inter-channel std X
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.plot(tp_axis, std_tx_raw,  color="#E53935", lw=0.9, label="Raw")
    ax02.plot(tp_axis, std_tx_corr, color="#1E88E5", lw=0.9, label="Tilt-corr")
    ax02.set_xlabel("TP"); ax02.set_ylabel("Inter-ch std X (um)")
    ax02.set_title(f"Inter-ch std X: raw={np.nanmean(std_tx_raw):.4f}, "
                   f"corr={np.nanmean(std_tx_corr):.4f}", fontsize=9)
    ax02.legend(fontsize=8)
    ax02.spines["top"].set_visible(False); ax02.spines["right"].set_visible(False)

    # ---- Row 1: ECC Y ----
    ax10 = fig.add_subplot(gs[1, 0])
    for c in range(n_ch):
        ax10.plot(tp_axis, ty_raw_um[c], lw=0.5, alpha=0.7, color=COLORS[c])
    ax10.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax10.set_xlabel("TP"); ax10.set_ylabel("ECC Y shift (um)")
    ax10.set_title("Per-ch ECC Y — Raw (crop+backsub)", fontsize=9)
    ax10.spines["top"].set_visible(False); ax10.spines["right"].set_visible(False)

    ax11 = fig.add_subplot(gs[1, 1])
    for c in range(n_ch):
        ax11.plot(tp_axis, ty_corr_um[c], lw=0.5, alpha=0.7, color=COLORS[c])
    ax11.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax11.set_xlabel("TP"); ax11.set_ylabel("ECC Y shift (um)")
    ax11.set_title("Per-ch ECC Y — Tilt-corrected", fontsize=9)
    ax11.spines["top"].set_visible(False); ax11.spines["right"].set_visible(False)

    ax12 = fig.add_subplot(gs[1, 2])
    ax12.plot(tp_axis, std_ty_raw,  color="#E53935", lw=0.9, label="Raw")
    ax12.plot(tp_axis, std_ty_corr, color="#1E88E5", lw=0.9, label="Tilt-corr")
    ax12.set_xlabel("TP"); ax12.set_ylabel("Inter-ch std Y (um)")
    ax12.set_title(f"Inter-ch std Y: raw={np.nanmean(std_ty_raw):.4f}, "
                   f"corr={np.nanmean(std_ty_corr):.4f}", fontsize=9)
    ax12.legend(fontsize=8)
    ax12.spines["top"].set_visible(False); ax12.spines["right"].set_visible(False)

    # ---- Row 2: Per-channel precision bar chart ----
    ch_labels = [f"ch{c}" for c in range(n_ch)]
    x_pos = np.arange(n_ch)
    w = 0.35

    ax20 = fig.add_subplot(gs[2, 0])
    ax20.bar(x_pos - w/2, abs_tx_raw,  w, color="#E53935", alpha=0.7, label="Raw")
    ax20.bar(x_pos + w/2, abs_tx_corr, w, color="#1E88E5", alpha=0.7, label="Tilt-corr")
    ax20.set_xticks(x_pos); ax20.set_xticklabels(ch_labels, fontsize=7)
    ax20.set_ylabel("Mean |ECC X| (um)")
    ax20.set_title("Per-channel ECC X precision", fontsize=9)
    ax20.legend(fontsize=8)
    ax20.spines["top"].set_visible(False); ax20.spines["right"].set_visible(False)

    ax21 = fig.add_subplot(gs[2, 1])
    ax21.bar(x_pos - w/2, abs_ty_raw,  w, color="#E53935", alpha=0.7, label="Raw")
    ax21.bar(x_pos + w/2, abs_ty_corr, w, color="#1E88E5", alpha=0.7, label="Tilt-corr")
    ax21.set_xticks(x_pos); ax21.set_xticklabels(ch_labels, fontsize=7)
    ax21.set_ylabel("Mean |ECC Y| (um)")
    ax21.set_title("Per-channel ECC Y precision", fontsize=9)
    ax21.legend(fontsize=8)
    ax21.spines["top"].set_visible(False); ax21.spines["right"].set_visible(False)

    # (F) histogram of all shifts
    ax22 = fig.add_subplot(gs[2, 2])
    valid_r = ~np.isnan(tx_raw_um)
    valid_c = ~np.isnan(tx_corr_um)
    all_r = np.sqrt(tx_raw_um[valid_r]**2 + ty_raw_um[valid_r]**2)
    all_c = np.sqrt(tx_corr_um[valid_c]**2 + ty_corr_um[valid_c]**2)
    bins = np.linspace(0, max(np.nanmax(all_r), np.nanmax(all_c)) * 1.05, 40)
    ax22.hist(all_r, bins=bins, alpha=0.6, color="#E53935",
              label=f"Raw μ={np.mean(all_r):.4f}")
    ax22.hist(all_c, bins=bins, alpha=0.6, color="#1E88E5",
              label=f"Corr μ={np.mean(all_c):.4f}")
    ax22.set_xlabel("ECC shift magnitude (um)")
    ax22.set_ylabel("Count")
    ax22.set_title("Shift magnitude distribution (all ch×TP)", fontsize=9)
    ax22.legend(fontsize=8)
    ax22.spines["top"].set_visible(False); ax22.spines["right"].set_visible(False)

    # ---- Row 3: Mean shift time series (systematic error) ----
    ax30 = fig.add_subplot(gs[3, 0])
    ax30.plot(tp_axis, mean_tx_raw,  color="#E53935", lw=0.9, label="Raw")
    ax30.plot(tp_axis, mean_tx_corr, color="#1E88E5", lw=0.9, label="Tilt-corr")
    ax30.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax30.set_xlabel("TP"); ax30.set_ylabel("Ch-mean ECC X (um)")
    ax30.set_title("Systematic error X (ch-mean)", fontsize=9)
    ax30.legend(fontsize=8)
    ax30.spines["top"].set_visible(False); ax30.spines["right"].set_visible(False)

    ax31 = fig.add_subplot(gs[3, 1])
    ax31.plot(tp_axis, mean_ty_raw,  color="#E53935", lw=0.9, label="Raw")
    ax31.plot(tp_axis, mean_ty_corr, color="#1E88E5", lw=0.9, label="Tilt-corr")
    ax31.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax31.set_xlabel("TP"); ax31.set_ylabel("Ch-mean ECC Y (um)")
    ax31.set_title("Systematic error Y (ch-mean)", fontsize=9)
    ax31.legend(fontsize=8)
    ax31.spines["top"].set_visible(False); ax31.spines["right"].set_visible(False)

    # (H) ref crops comparison: raw vs corr u8
    ax32 = fig.add_subplot(gs[3, 2])
    mid_ch = n_ch // 2
    img_raw  = ref_raw_u8[mid_ch]
    img_corr = ref_corr_u8[mid_ch]
    # Stack side by side
    combined = np.hstack([img_raw, np.full((img_raw.shape[0], 3), 128, dtype=np.uint8), img_corr])
    ax32.imshow(combined, cmap="RdBu_r", aspect="auto", interpolation="nearest",
                vmin=0, vmax=255)
    ax32.axvline(img_raw.shape[1] + 1.5, color="yellow", lw=1.5)
    ax32.set_title(f"Grid ref ch{mid_ch}: Raw(left) vs Tilt-corr(right)", fontsize=9)
    ax32.set_xlabel("X [px]"); ax32.set_ylabel("Y [px]")
    ax32.tick_params(labelsize=7)

    fig.suptitle(
        f"ECC precision: tilt correction effect (ph_3 static, grid ref 260331 z=10)\n"
        f"Raw = crop+backsub | Corr = _tilt_correct(270→80) — ref & test both consistent\n"
        f"|X|: raw={np.nanmean(np.abs(tx_raw_um)):.4f}, corr={np.nanmean(np.abs(tx_corr_um)):.4f} um  |  "
        f"|Y|: raw={np.nanmean(np.abs(ty_raw_um)):.4f}, corr={np.nanmean(np.abs(ty_corr_um)):.4f} um",
        fontsize=10,
    )

    save_figure(
        fig,
        params={
            "crop_w": CROP_W, "tilt_crop_h": TILT_CROP_H, "ecc_crop_h": ECC_CROP_H,
            "fit_right": FIT_RIGHT, "ref": "self (ph_3 frame 0)",
            "n_ch": n_ch, "n_tp": n_tp,
            "grid_ref": "260331_Pos1_z10",
            "mean_abs_tx_raw_um":  float(np.nanmean(np.abs(tx_raw_um))),
            "mean_abs_tx_corr_um": float(np.nanmean(np.abs(tx_corr_um))),
            "mean_abs_ty_raw_um":  float(np.nanmean(np.abs(ty_raw_um))),
            "mean_abs_ty_corr_um": float(np.nanmean(np.abs(ty_corr_um))),
        },
        description=(
            f"ECC precision on static ph_3: raw vs _tilt_correct. "
            f"|X| raw={np.nanmean(np.abs(tx_raw_um)):.4f} corr={np.nanmean(np.abs(tx_corr_um)):.4f}, "
            f"|Y| raw={np.nanmean(np.abs(ty_raw_um)):.4f} corr={np.nanmean(np.abs(ty_corr_um)):.4f} um"
        ),
        data={
            "tx_raw_um": tx_raw_um, "ty_raw_um": ty_raw_um,
            "tx_corr_um": tx_corr_um, "ty_corr_um": ty_corr_um,
            "abs_tx_raw": abs_tx_raw, "abs_ty_raw": abs_ty_raw,
            "abs_tx_corr": abs_tx_corr, "abs_ty_corr": abs_ty_corr,
            "std_tx_raw": std_tx_raw, "std_ty_raw": std_ty_raw,
            "std_tx_corr": std_tx_corr, "std_ty_corr": std_ty_corr,
        },
    )
    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
