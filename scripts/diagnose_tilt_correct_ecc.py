"""
diagnose_tilt_correct_ecc.py
-----------------------------
Diagnostic script to visualize _tilt_correct behavior.
For frames with strong and weak X-tilt, displays:
  - 270px X crop (raw)
  - 270px X crop (tilt-corrected)
  - 80px ECC crop (test raw / corr)
  - 80px ECC crop (grid ref raw / corr)
  - diff: test - ref (raw & corr)
  - X profile overlay
side by side to visually verify what the correction does to ECC.

Selected frames:
  Strong tilt: frame 54 (mean slope ~ -1.514 mrad/px)
  Weak tilt:   frame 19 (mean slope ~ -0.729 mrad/px)
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
PHASE_DIR = Path(r"D:\AquisitionData\Kitagishi\basler_image_seq\ph_3\Pos0\output_phase")
ROIS_JSON = Path(r"F:\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\channels\channel_rois.json")
GRID_REF_PHASE = Path(r"F:\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\img_000000000_ph_009_phase.tif")

PIXEL_SCALE_UM = 0.34567514677103717
ECC_VMIN = -5.0
ECC_VMAX  =  2.0
TILT_CROP_H = 270
ECC_CROP_H  = 80
FIT_RIGHT   = False

# Diagnostic frames and representative channels
FRAME_STRONG = 54   # mean slope ~ -1.514 mrad/px
FRAME_WEAK   = 19   # mean slope ~ -0.729 mrad/px
VIS_CHANNELS = [2, 5, 8, 11]   # 4 representative channels
# ============================================================


def tilt_correct_full(img_f64, cy, cx, crop_w):
    """Return both the 270px crop and tilt-corrected crop (80px)."""
    big = extract_rect_roi(img_f64, cy, cx, crop_w, TILT_CROP_H).astype(np.float64)
    x = np.arange(TILT_CROP_H, dtype=np.float64)
    prof = big.mean(axis=0)
    fit_n = max(1, TILT_CROP_H // 3)
    if FIT_RIGHT:
        a, b = np.polyfit(x[-fit_n:], prof[-fit_n:], 1)
    else:
        a, b = np.polyfit(x[:fit_n], prof[:fit_n], 1)
    corrected_big = big - (a * x + b)[np.newaxis, :]
    start = (TILT_CROP_H - ECC_CROP_H) // 2
    corrected_80 = corrected_big[:, start:start + ECC_CROP_H]
    return big, corrected_big, corrected_80, a, b


def get_raw_80(img_f64, cy, cx, crop_w, crop_h):
    """80px raw crop + backsub."""
    crop = extract_rect_roi(img_f64, cy, cx, crop_w, crop_h).astype(np.float64)
    cfg_dummy = {}
    offset = compute_backsub_offset(crop, cfg_dummy)
    return crop, crop + offset


def main():
    rois = json.loads(ROIS_JSON.read_text(encoding="utf-8"))
    n_ch = len(rois)
    phase_paths = sorted(PHASE_DIR.glob("img_*_ph_000_phase.tif"))

    # Load frames
    path_strong = phase_paths[FRAME_STRONG]
    path_weak   = phase_paths[FRAME_WEAK]
    print(f"Strong tilt frame: {path_strong.name}")
    print(f"Weak   tilt frame: {path_weak.name}")

    ph_strong = tifffile.imread(str(path_strong)).astype(np.float64)
    ph_weak   = tifffile.imread(str(path_weak)).astype(np.float64)
    ph_ref    = tifffile.imread(str(GRID_REF_PHASE)).astype(np.float64)

    # ============================================================
    # Fig 1: Full crop view for 4 representative channels
    #   Rows: [strong-tilt frame]  [weak-tilt frame]  [grid ref]
    #   Cols: per channel (4)
    #   Each cell: imshow of 270px big crop (raw)
    #   + X profile overlay (raw / corrected / fit line)
    # ============================================================
    fig1, axes1 = plt.subplots(3, len(VIS_CHANNELS), figsize=(4 * len(VIS_CHANNELS), 10))
    fig1.subplots_adjust(hspace=0.45, wspace=0.30,
                         left=0.06, right=0.98, top=0.92, bottom=0.06)
    row_labels = [
        f"Strong tilt (frame {FRAME_STRONG}, slope≈-1.51 mrad/px)",
        f"Weak tilt   (frame {FRAME_WEAK}, slope≈-0.73 mrad/px)",
        "Grid ref (F: Pos1_x+0_y+0, z=9)",
    ]
    phases = [ph_strong, ph_weak, ph_ref]

    for col, ch_idx in enumerate(VIS_CHANNELS):
        roi = rois[ch_idx]
        cy, cx, crop_w, crop_h = roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]

        for row, (ph, rlabel) in enumerate(zip(phases, row_labels)):
            big, corrected_big, corrected_80, a, b = tilt_correct_full(ph, cy, cx, crop_w)
            x_ax = np.arange(TILT_CROP_H, dtype=float)
            prof_raw  = big.mean(axis=0)
            prof_corr = corrected_big.mean(axis=0)
            fit_line  = a * x_ax + b
            fit_n     = TILT_CROP_H // 3

            # imshow: raw 270px crop
            ax = axes1[row, col]
            vmin_v, vmax_v = np.nanpercentile(big, [2, 98])
            ax.imshow(big, cmap="RdBu_r", aspect="auto",
                      interpolation="nearest", vmin=vmin_v, vmax=vmax_v)
            # ECC crop region (center 80px)
            ecc_s = (TILT_CROP_H - crop_h) // 2
            ax.axvline(ecc_s,           color="cyan", lw=1.0, ls="--")
            ax.axvline(ecc_s + crop_h,  color="cyan", lw=1.0, ls="--")
            ax.axvline(fit_n,            color="yellow", lw=0.8, ls=":")
            ax.set_xlabel("X [px]"); ax.set_ylabel("Y [px]")
            title_str = f"ch{ch_idx}"
            if row == 0:
                ax.set_title(f"ch{ch_idx}  cy={cy}\n"
                             f"slope={a*1000:.3f} mrad/px", fontsize=8)
            else:
                ax.set_title(f"ch{ch_idx}  slope={a*1000:.3f} mrad/px", fontsize=8)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(rlabel.split("(")[0].strip() + "\nY [px]", fontsize=7)

    fig1.suptitle(
        "270px X crop (raw) — cyan: ECC crop region, yellow: BG fit boundary\n"
        "Strong tilt vs Weak tilt vs Grid ref  |  4 representative channels",
        fontsize=10,
    )
    save_figure(
        fig1,
        params={"frame_strong": FRAME_STRONG, "frame_weak": FRAME_WEAK,
                "vis_channels": VIS_CHANNELS},
        description="Diagnostic: 270px raw X crop for strong/weak tilt frames vs grid ref",
        data={"vis_channels": np.array(VIS_CHANNELS)},
    )
    plt.close(fig1)

    # ============================================================
    # Fig 2: X profile comparison (raw vs tilt-corrected vs ref)
    #   3 rows: strong / weak / ref  x  4 cols: channel
    #   Each panel: 3 X profiles (raw red, corr blue, ref gray)
    # ============================================================
    fig2, axes2 = plt.subplots(3, len(VIS_CHANNELS), figsize=(4 * len(VIS_CHANNELS), 9))
    fig2.subplots_adjust(hspace=0.50, wspace=0.35,
                         left=0.07, right=0.98, top=0.91, bottom=0.07)

    for col, ch_idx in enumerate(VIS_CHANNELS):
        roi = rois[ch_idx]
        cy, cx, crop_w, crop_h = roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]
        ecc_s = (TILT_CROP_H - crop_h) // 2
        fit_n = TILT_CROP_H // 3

        # ref profile (for overlay)
        _, ref_corr_big, _, a_ref, b_ref = tilt_correct_full(ph_ref, cy, cx, crop_w)
        ref_prof_raw  = extract_rect_roi(ph_ref, cy, cx, crop_w, TILT_CROP_H).astype(np.float64).mean(axis=0)
        ref_prof_corr = ref_corr_big.mean(axis=0)
        x_ax = np.arange(TILT_CROP_H, dtype=float)

        for row, (ph, rlabel) in enumerate(zip(phases[:2], row_labels[:2])):
            big, corrected_big, _, a, b = tilt_correct_full(ph, cy, cx, crop_w)
            prof_raw  = big.mean(axis=0)
            prof_corr = corrected_big.mean(axis=0)
            fit_line  = a * x_ax + b

            ax = axes2[row, col]
            ax.plot(x_ax, prof_raw,  color="#E53935", lw=1.0, alpha=0.85, label="Test raw")
            ax.plot(x_ax, prof_corr, color="#1E88E5", lw=1.0, alpha=0.85, label="Test corr")
            ax.plot(x_ax, ref_prof_raw,  color="#888", lw=0.8, ls="--", alpha=0.7, label="Ref raw")
            ax.plot(x_ax, ref_prof_corr, color="#4CAF50", lw=0.8, ls="--", alpha=0.7, label="Ref corr")
            ax.plot(x_ax[:fit_n], fit_line[:fit_n],
                    color="#FF6F00", lw=1.5, ls=":", label=f"Fit (slope={a*1000:.3f} mrad)")
            ax.axvline(ecc_s,          color="#00ACC1", lw=0.8, ls="--")
            ax.axvline(ecc_s + crop_h, color="#00ACC1", lw=0.8, ls="--")
            ax.axvline(fit_n,           color="#999", lw=0.7, ls=":")
            ax.set_xlabel("X [px]"); ax.set_ylabel("Phase (rad)")
            title = f"ch{ch_idx}  {'Strong' if row==0 else 'Weak'} tilt"
            ax.set_title(title, fontsize=8)
            if col == 0 and row == 0:
                ax.legend(fontsize=6, loc="upper right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Row 2: difference profile (test - ref)  raw vs corr
        ax_d = axes2[2, col]
        big_s, corr_big_s, _, _, _ = tilt_correct_full(ph_strong, cy, cx, crop_w)
        big_w, corr_big_w, _, _, _ = tilt_correct_full(ph_weak,   cy, cx, crop_w)
        diff_raw_s  = big_s.mean(axis=0)  - ref_prof_raw
        diff_corr_s = corr_big_s.mean(axis=0) - ref_prof_corr
        diff_raw_w  = big_w.mean(axis=0)  - ref_prof_raw
        diff_corr_w = corr_big_w.mean(axis=0) - ref_prof_corr

        ax_d.plot(x_ax, diff_raw_s,  color="#E53935", lw=1.0, label="Strong: raw diff")
        ax_d.plot(x_ax, diff_corr_s, color="#E53935", lw=1.0, ls="--", label="Strong: corr diff")
        ax_d.plot(x_ax, diff_raw_w,  color="#1E88E5", lw=1.0, label="Weak: raw diff")
        ax_d.plot(x_ax, diff_corr_w, color="#1E88E5", lw=1.0, ls="--", label="Weak: corr diff")
        ax_d.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax_d.axvline(ecc_s,          color="#00ACC1", lw=0.8, ls="--")
        ax_d.axvline(ecc_s + crop_h, color="#00ACC1", lw=0.8, ls="--")
        ax_d.set_xlabel("X [px]"); ax_d.set_ylabel("Phase diff (rad)")
        ax_d.set_title(f"ch{ch_idx}  diff (test−ref)", fontsize=8)
        if col == 0:
            ax_d.legend(fontsize=6)
        ax_d.spines["top"].set_visible(False)
        ax_d.spines["right"].set_visible(False)

    fig2.suptitle(
        "X profiles: test (strong/weak tilt) vs grid ref  |  raw & tilt-corrected\n"
        "cyan: ECC crop (80px), orange dot: BG fit region  |  "
        "Row 3: difference profile (test − ref)",
        fontsize=10,
    )
    save_figure(
        fig2,
        params={"frame_strong": FRAME_STRONG, "frame_weak": FRAME_WEAK,
                "vis_channels": VIS_CHANNELS, "tilt_crop_h": TILT_CROP_H},
        description="Diagnostic: X profiles raw/corrected for strong vs weak tilt frames vs grid ref",
        data={"vis_channels": np.array(VIS_CHANNELS)},
    )
    plt.close(fig2)

    # ============================================================
    # Fig 3: 80px ECC crop imshow comparison
    #   2 rows x 4 cols: raw crop u8 / tilt-corr crop u8
    #   Each column triplet: [strong-test | weak-test | ref]
    # ============================================================
    fig3, axes3 = plt.subplots(3, 3 * len(VIS_CHANNELS),
                               figsize=(3.5 * len(VIS_CHANNELS), 9),
                               gridspec_kw={"wspace": 0.15, "hspace": 0.5})
    cfg_dummy = {}
    labels_col = ["Strong", "Weak", "Ref"]

    for col_g, ch_idx in enumerate(VIS_CHANNELS):
        roi = rois[ch_idx]
        cy, cx, crop_w, crop_h = roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]

        for col_l, (ph, lbl) in enumerate(zip([ph_strong, ph_weak, ph_ref], labels_col)):
            col = col_g * 3 + col_l

            # Row 0: raw 80px crop (backsub) u8
            crop_r, crop_r_bs = get_raw_80(ph, cy, cx, crop_w, crop_h)
            u8_raw = to_uint8(crop_r_bs, ECC_VMIN, ECC_VMAX)

            # Row 1: tilt-corrected 80px crop u8
            crop_c = _tilt_correct(ph, cy, cx, crop_w, TILT_CROP_H, ECC_CROP_H, fit_right=FIT_RIGHT)
            u8_corr = to_uint8(crop_c, ECC_VMIN, ECC_VMAX)

            # Row 2: diff (test − ref)
            # ref values already computed below
            _, _, ref_corr_80, _, _ = tilt_correct_full(ph_ref, cy, cx, crop_w)
            ref_raw_80, _ = get_raw_80(ph_ref, cy, cx, crop_w, crop_h)

            for row, (u8, title) in enumerate([
                (u8_raw,  f"ch{ch_idx} {lbl}\nraw (backsub) u8"),
                (u8_corr, f"ch{ch_idx} {lbl}\ntilt-corr u8"),
            ]):
                ax = axes3[row, col]
                ax.imshow(u8, cmap="RdBu_r", aspect="auto",
                          interpolation="nearest", vmin=0, vmax=255)
                ax.set_title(title, fontsize=7)
                ax.set_xlabel("X [px]"); ax.set_ylabel("Y [px]")
                ax.tick_params(labelsize=5)

            # Row 2: raw diff u8
            ax_d = axes3[2, col]
            if lbl != "Ref":
                diff_raw  = crop_r_bs.astype(float) - (get_raw_80(ph_ref, cy, cx, crop_w, crop_h)[1]).astype(float)
                ax_d.imshow(diff_raw, cmap="RdBu_r", aspect="auto",
                            interpolation="nearest",
                            vmin=-30, vmax=30)
                ax_d.set_title(f"ch{ch_idx} {lbl}\ndiff raw (test−ref)", fontsize=7)
            else:
                # ref row: show corr-raw difference for ref
                diff_ref = to_uint8(ref_corr_80, ECC_VMIN, ECC_VMAX).astype(float) - u8_raw.astype(float)
                ax_d.imshow(diff_ref, cmap="RdBu_r", aspect="auto",
                            interpolation="nearest",
                            vmin=-30, vmax=30)
                ax_d.set_title(f"ch{ch_idx} Ref\ncorr−raw", fontsize=7)
            ax_d.set_xlabel("X [px]"); ax_d.set_ylabel("Y [px]")
            ax_d.tick_params(labelsize=5)

    fig3.suptitle(
        "80px ECC crop: raw (backsub) vs tilt-corrected — 4 channels × [Strong | Weak | Ref]\n"
        "Row 3: raw diff (test − ref u8)  /  Ref: corr − raw",
        fontsize=10,
    )
    save_figure(
        fig3,
        params={"frame_strong": FRAME_STRONG, "frame_weak": FRAME_WEAK,
                "vis_channels": VIS_CHANNELS},
        description="Diagnostic: 80px ECC crop (raw vs tilt-corr) for strong/weak tilt frames vs ref",
        data={"vis_channels": np.array(VIS_CHANNELS)},
    )
    plt.close(fig3)

    print("\nAll 3 figures saved via figure_logger.")


if __name__ == "__main__":
    main()
