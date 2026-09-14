"""test_ecc_precision.py -- ECC measurement noise characterization

Extract per-channel per-TP raw shifts from Pos1 phase images and analyze:

1. Per-channel residual (shift_ch - SavGol_trend) distribution
2. Autocorrelation (ACF) -> Is it random?
3. Cross-channel correlation -> Are channels independent?
4. sigma vs N curve -> How many channels/frames averaged to get below 0.01 um?

Usage:
    python scripts/test_ecc_precision.py
    python scripts/test_ecc_precision.py --config drift_session/drift_config.json --max-tp 100
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tifffile
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import savgol_filter
from scipy.stats import norm as sp_norm

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure
from compute_drift_online import (
    compute_backsub_offset,
    extract_rect_roi,
    to_uint8,
    ecc_align,
)

DEFAULT_CONFIG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json")

# Override for ph_3 precision measurement (set to None to use drift_config.json)
PHASE_DIR_OVERRIDE = r"D:\AquisitionData\Kitagishi\basler_image_seq\ph_3\Pos0\output_phase"
ROIS_JSON_OVERRIDE = r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"

# Specify to use grid reference image (None defaults to phase_paths[0])
# ph_3 is at the same stage position, so grid->ph_3 offset is constant across all frames/channels
# -> residual = each measurement - grand_mean = ECC noise (SavGol trend removal not needed)
GRID_REF_OVERRIDE = r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\img_000000000_ph_000_phase.tif"

SAVGOL_WINDOW = 51
SAVGOL_ORDER  = 2
ACF_MAX_LAG   = 30   # Maximum lag to display [TP]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--max-tp", type=int, default=None)
    p.add_argument(
        "--mode",
        choices=["fixed", "sliding"],
        default="fixed",
        help=(
            "fixed: frame 0 as fixed reference (current default). "
            "sliding: consecutive frame pairs (frame_i -> frame_{i+1}), "
            "measures single-step noise independently of long-term drift."
        ),
    )
    p.add_argument(
        "--residual",
        choices=["channel-mean", "zero", "savgol", "instant"],
        default="channel-mean",
        help=(
            "channel-mean (default): subtract each channel's temporal mean "
            "(mean over all TPs). Residual = deviation of each (ch, TP) "
            "measurement from its channel's fixed offset. std(residual) = ECC std. "
            "zero: no subtraction, raw shifts. "
            "savgol: SavGol temporal trend. "
            "instant: per-TP cross-channel mean."
        ),
    )
    return p.parse_args()


def load_channels(rois_path):
    rois = json.loads(Path(rois_path).read_text(encoding="utf-8"))
    return rois  # list of {cy, cx, crop_w, crop_h}


def compute_acf(series, max_lag):
    """Return normalized autocorrelation (lag=0 equals 1)."""
    x = series - np.mean(series)
    n = len(x)
    full = np.correlate(x, x, mode="full")
    acf = full[n - 1:] / full[n - 1]  # Normalize so lag=0 is 1
    return acf[:max_lag + 1]


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    phase_dir  = Path(PHASE_DIR_OVERRIDE) if PHASE_DIR_OVERRIDE else Path(cfg["save_dir"]) / "Pos1" / "output_phase"
    rois_path  = ROIS_JSON_OVERRIDE if ROIS_JSON_OVERRIDE else cfg["channel_rois_json"]
    pixel_scale = cfg.get("pixel_scale_um", 0.3462)
    vmin = cfg.get("ecc_vmin", -5.0)
    vmax = cfg.get("ecc_vmax",  2.0)

    phase_paths = sorted(phase_dir.glob("img_*_phase.tif"))
    if not phase_paths:
        print(f"ERROR: no phase images in {phase_dir}")
        sys.exit(1)

    if args.max_tp is not None:
        phase_paths = phase_paths[:args.max_tp + 1]

    rois = load_channels(rois_path)
    n_ch = len(rois)
    n_tp = len(phase_paths) - 1  # Exclude ref frame (index 0)
    print(f"Phase images: {len(phase_paths)} ({n_tp} TPs),  Channels: {n_ch}")

    # ---- Get per-channel per-TP raw shifts ----
    # shape: (n_ch, n_tp),  units: px (image coords)
    tx_raw = np.full((n_ch, n_tp), np.nan)
    ty_raw = np.full((n_ch, n_tp), np.nan)

    def make_crops_u8(phase_img):
        crops = []
        for roi in rois:
            crop = extract_rect_roi(phase_img, roi["cy"], roi["cx"],
                                    roi["crop_w"], roi["crop_h"]).astype(np.float32)
            offset = compute_backsub_offset(crop, cfg)
            crops.append(to_uint8(crop + offset, vmin, vmax))
        return crops

    def _ecc_one_tp(ref_crops, smp_crops):
        """Run ECC for all channels between ref and sample crops. Returns (tx_arr, ty_arr)."""
        tx_arr = np.full(n_ch, np.nan)
        ty_arr = np.full(n_ch, np.nan)
        def _run_ch(c_idx):
            return c_idx, ecc_align(ref_crops[c_idx], smp_crops[c_idx])
        with ThreadPoolExecutor(max_workers=None) as pool:
            for c_idx, result in pool.map(lambda c: _run_ch(c), range(n_ch)):
                if result is not None:
                    tx_arr[c_idx] = result[0]
                    ty_arr[c_idx] = result[1]
        return tx_arr, ty_arr

    if args.mode == "fixed":
        # ------ Fixed reference (grid image if GRID_REF_OVERRIDE set, otherwise frame 0) ------
        ref_path = GRID_REF_OVERRIDE if GRID_REF_OVERRIDE else str(phase_paths[0])
        phase_ref = tifffile.imread(ref_path).astype(np.float32)
        print(f"Reference: {ref_path}")
        ref_crops_u8 = make_crops_u8(phase_ref)

        for t_idx, path in enumerate(phase_paths[1:]):
            try:
                phase_tp = tifffile.imread(str(path)).astype(np.float32)
            except Exception:
                continue
            smp_crops_u8 = make_crops_u8(phase_tp)
            tx_arr, ty_arr = _ecc_one_tp(ref_crops_u8, smp_crops_u8)
            tx_raw[:, t_idx] = tx_arr
            ty_raw[:, t_idx] = ty_arr
            if (t_idx + 1) % 20 == 0:
                print(f"  TP {t_idx+1}/{n_tp}", flush=True)

    else:
        # ------ Sliding pairs (frame_i -> frame_{i+1}) ------
        # Each step is an independent noise measurement, independent of long-term drift trend.
        prev_img = tifffile.imread(str(phase_paths[0])).astype(np.float32)
        prev_crops_u8 = make_crops_u8(prev_img)

        for t_idx, path in enumerate(phase_paths[1:]):
            try:
                curr_img = tifffile.imread(str(path)).astype(np.float32)
            except Exception:
                prev_crops_u8 = prev_crops_u8  # keep previous
                continue
            curr_crops_u8 = make_crops_u8(curr_img)
            tx_arr, ty_arr = _ecc_one_tp(prev_crops_u8, curr_crops_u8)
            tx_raw[:, t_idx] = tx_arr
            ty_raw[:, t_idx] = ty_arr
            prev_crops_u8 = curr_crops_u8
            if (t_idx + 1) % 20 == 0:
                print(f"  Pair {t_idx+1}/{n_tp}", flush=True)

    print(f"ECC done. (mode={args.mode})")

    # ---- Convert to um (image X/Y -> stage shift, simple scaling) ----
    # No axis conversion here, only image px -> um.
    # Analyzing relative scatter (residuals), so axis conversion has no effect.
    tx_um = tx_raw * pixel_scale
    ty_um = ty_raw * pixel_scale

    # ---- Trend estimation and residual calculation ----
    tp_axis = np.arange(n_tp)

    def smooth_mean(arr2d):
        """Compute mean excluding NaN, then apply SavGol smoothing."""
        mean_series = np.nanmean(arr2d, axis=0)
        wl = min(SAVGOL_WINDOW, int(np.sum(~np.isnan(mean_series))) - 1)
        if wl < 3:
            return mean_series
        if wl % 2 == 0:
            wl -= 1
        smoothed = np.full_like(mean_series, np.nan)
        valid_idx = ~np.isnan(mean_series)
        smoothed[valid_idx] = savgol_filter(mean_series[valid_idx], wl, SAVGOL_ORDER)
        return smoothed

    if args.residual == "channel-mean":
        # Subtract each channel's temporal mean.
        # mean_ch[c] = mean over all TPs of tx_um[c, :] -> channel fixed offset estimate
        # res[c, t]  = tx_um[c, t] - mean_ch[c]         -> ECC measurement deviation
        mean_ch_x = np.nanmean(tx_um, axis=1, keepdims=True)  # (n_ch, 1)
        mean_ch_y = np.nanmean(ty_um, axis=1, keepdims=True)
        res_x = tx_um - mean_ch_x
        res_y = ty_um - mean_ch_y
        # Define trend for time series plot (constant per channel)
        trend_x = np.nanmean(tx_um, axis=0)  # cross-channel mean (for plot only)
        trend_y = np.nanmean(ty_um, axis=0)
        residual_label = "Per-channel temporal mean subtracted"
    elif args.residual == "zero":
        trend_x = np.zeros(n_tp)
        trend_y = np.zeros(n_tp)
        res_x = tx_um
        res_y = ty_um
        residual_label = "No subtraction (raw shifts)"
    elif args.residual == "savgol":
        trend_x = smooth_mean(tx_um)
        trend_y = smooth_mean(ty_um)
        res_x = tx_um - trend_x[np.newaxis, :]
        res_y = ty_um - trend_y[np.newaxis, :]
        residual_label = "SavGol temporal trend subtracted"
    else:  # instant
        trend_x = np.nanmean(tx_um, axis=0)
        trend_y = np.nanmean(ty_um, axis=0)
        res_x = tx_um - trend_x[np.newaxis, :]
        res_y = ty_um - trend_y[np.newaxis, :]
        residual_label = "Per-TP cross-channel mean subtracted"

    n_samples = int(np.sum(~np.isnan(res_x)))
    print(f"Residual: {args.residual}  ({n_samples} samples = {n_ch} ch x ~{n_samples//n_ch} TP)")

    # ---- Statistics ----
    sigma_x = np.nanstd(res_x)    # std over all channels and all TPs combined
    sigma_y = np.nanstd(res_y)
    print(f"\nSigma per channel per TP:")
    print(f"  X: {sigma_x:.4f} um")
    print(f"  Y: {sigma_y:.4f} um")

    # Per-channel individual sigma
    sigma_x_ch = np.nanstd(res_x, axis=1)   # shape (n_ch,)
    sigma_y_ch = np.nanstd(res_y, axis=1)
    print(f"\nPer-channel sigma X: mean={np.mean(sigma_x_ch):.4f}  "
          f"min={np.min(sigma_x_ch):.4f}  max={np.max(sigma_x_ch):.4f} um")
    print(f"Per-channel sigma Y: mean={np.mean(sigma_y_ch):.4f}  "
          f"min={np.min(sigma_y_ch):.4f}  max={np.max(sigma_y_ch):.4f} um")

    # SEM when averaging N channels
    N_arr = np.arange(1, n_ch + 1)
    sem_x = sigma_x / np.sqrt(N_arr)
    sem_y = sigma_y / np.sqrt(N_arr)
    n_need_x = int(np.ceil((sigma_x / 0.01) ** 2))
    n_need_y = int(np.ceil((sigma_y / 0.01) ** 2))
    print(f"\nChannels needed for SEM < 0.01 um:")
    print(f"  X: {n_need_x}  (current: {n_ch})")
    print(f"  Y: {n_need_y}  (current: {n_ch})")

    # ---- ACF ----
    # Compute ACF for each channel's residuals, then average
    acf_x_all, acf_y_all = [], []
    for c in range(n_ch):
        valid_c = ~np.isnan(res_x[c])
        if valid_c.sum() > ACF_MAX_LAG * 2:
            acf_x_all.append(compute_acf(res_x[c][valid_c], ACF_MAX_LAG))
            acf_y_all.append(compute_acf(res_y[c][valid_c], ACF_MAX_LAG))

    acf_x_mean = np.mean(acf_x_all, axis=0) if acf_x_all else None
    acf_y_mean = np.mean(acf_y_all, axis=0) if acf_y_all else None

    n_eff = np.sum(~np.isnan(res_x[0]))  # Number of valid TPs (representative)
    acf_ci = 2.0 / np.sqrt(n_eff)   # 95% CI for white noise

    # ---- Cross-correlation matrix ----
    # Compute Pearson correlation matrix after excluding NaN
    cross_x = np.full((n_ch, n_ch), np.nan)
    cross_y = np.full((n_ch, n_ch), np.nan)
    for i in range(n_ch):
        for j in range(n_ch):
            valid_ij = ~(np.isnan(res_x[i]) | np.isnan(res_x[j]))
            if valid_ij.sum() > 5:
                xi, xj = res_x[i][valid_ij], res_x[j][valid_ij]
                yi, yj = res_y[i][valid_ij], res_y[j][valid_ij]
                sx, sy = np.std(xi), np.std(xj)
                if sx > 0 and sy > 0:
                    cross_x[i, j] = np.mean((xi - xi.mean()) * (xj - xj.mean())) / (sx * sy)
                sx, sy = np.std(yi), np.std(yj)
                if sx > 0 and sy > 0:
                    cross_y[i, j] = np.mean((yi - yi.mean()) * (yj - yj.mean())) / (sx * sy)

    # off-diagonal mean (= average correlation between different channels)
    off_diag_mask = ~np.eye(n_ch, dtype=bool)
    mean_cross_x = np.nanmean(cross_x[off_diag_mask])
    mean_cross_y = np.nanmean(cross_y[off_diag_mask])
    print(f"\nMean off-diagonal cross-correlation:")
    print(f"  X: {mean_cross_x:.3f}  Y: {mean_cross_y:.3f}")
    print(f"  (0 = fully independent, 1 = fully correlated)")

    # "Effective number of channels" correction for non-independence
    # Effective N from rho_avg: N_eff = N / (1 + (N-1)*rho)
    def effective_n(N, rho):
        return N / (1 + (N - 1) * max(rho, 0))

    neff_x = effective_n(n_ch, mean_cross_x)
    neff_y = effective_n(n_ch, mean_cross_y)
    sem_x_corr = sigma_x / np.sqrt(neff_x)
    sem_y_corr = sigma_y / np.sqrt(neff_y)
    print(f"\nWith cross-channel correlation corrected:")
    print(f"  Effective N  X: {neff_x:.1f}  Y: {neff_y:.1f}  "
          f"(from {n_ch} channels)")
    print(f"  SEM_x = {sem_x_corr:.4f} um  SEM_y = {sem_y_corr:.4f} um")

    # ---- Figure ----
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 3, hspace=0.55, wspace=0.4,
                             left=0.07, right=0.97, top=0.93, bottom=0.07)
    time_h = np.arange(n_tp) * 5.0 / 60.0

    # --- [0,0] Per-channel shift time series (X) ---
    ax00 = fig.add_subplot(gs[0, 0])
    for c in range(n_ch):
        ax00.plot(time_h, tx_um[c], lw=0.5, alpha=0.4)
    ax00.plot(time_h, trend_x, "k-", lw=1.5, label="Mean trend (SavGol)")
    ax00.set_xlabel("Time (h)")
    ax00.set_ylabel("Shift X (um)")
    ax00.set_title("Per-channel shift X")
    ax00.legend(frameon=False, fontsize=7)
    ax00.spines[["top", "right"]].set_visible(False)

    # --- [0,1] Per-channel shift time series (Y) ---
    ax01 = fig.add_subplot(gs[0, 1])
    for c in range(n_ch):
        ax01.plot(time_h, ty_um[c], lw=0.5, alpha=0.4)
    ax01.plot(time_h, trend_y, "k-", lw=1.5, label="Mean trend (SavGol)")
    ax01.set_xlabel("Time (h)")
    ax01.set_ylabel("Shift Y (um)")
    ax01.set_title("Per-channel shift Y")
    ax01.legend(frameon=False, fontsize=7)
    ax01.spines[["top", "right"]].set_visible(False)

    # --- [0,2] Residual histogram ---
    ax02 = fig.add_subplot(gs[0, 2])
    res_x_flat = res_x[~np.isnan(res_x)].ravel()
    res_y_flat = res_y[~np.isnan(res_y)].ravel()
    bins = np.linspace(-0.15, 0.15, 50)
    ax02.hist(res_x_flat, bins=bins, alpha=0.6, label=f"X  sigma={sigma_x:.3f}um")
    ax02.hist(res_y_flat, bins=bins, alpha=0.6, label=f"Y  sigma={sigma_y:.3f}um")
    # Gaussian fit
    for sigma, color in [(sigma_x, "C0"), (sigma_y, "C1")]:
        x_fit = np.linspace(-0.15, 0.15, 300)
        ax02.plot(x_fit, sp_norm.pdf(x_fit, 0, sigma) * len(res_x_flat) * (bins[1]-bins[0]),
                  color=color, lw=1.5)
    ax02.set_xlabel("Residual (um)")
    ax02.set_ylabel("Count")
    ax02.set_title("Residual distribution (all channels, all TPs)")
    ax02.legend(frameon=False, fontsize=7)
    ax02.spines[["top", "right"]].set_visible(False)

    # --- [1,0] ACF (X) ---
    ax10 = fig.add_subplot(gs[1, 0])
    if acf_x_mean is not None:
        lags = np.arange(len(acf_x_mean))
        ax10.bar(lags[1:], acf_x_mean[1:], width=0.7, color="C0", alpha=0.7)
        ax10.axhline( acf_ci, color="gray", lw=1, ls="--", label="95% CI (white noise)")
        ax10.axhline(-acf_ci, color="gray", lw=1, ls="--")
    ax10.axhline(0, color="k", lw=0.5)
    ax10.set_xlabel("Lag (TP)")
    ax10.set_ylabel("ACF")
    ax10.set_title("Autocorrelation of residuals X")
    ax10.legend(frameon=False, fontsize=7)
    ax10.set_xlim(0, ACF_MAX_LAG)
    ax10.spines[["top", "right"]].set_visible(False)

    # --- [1,1] ACF (Y) ---
    ax11 = fig.add_subplot(gs[1, 1])
    if acf_y_mean is not None:
        lags = np.arange(len(acf_y_mean))
        ax11.bar(lags[1:], acf_y_mean[1:], width=0.7, color="C1", alpha=0.7)
        ax11.axhline( acf_ci, color="gray", lw=1, ls="--", label="95% CI (white noise)")
        ax11.axhline(-acf_ci, color="gray", lw=1, ls="--")
    ax11.axhline(0, color="k", lw=0.5)
    ax11.set_xlabel("Lag (TP)")
    ax11.set_ylabel("ACF")
    ax11.set_title("Autocorrelation of residuals Y")
    ax11.legend(frameon=False, fontsize=7)
    ax11.set_xlim(0, ACF_MAX_LAG)
    ax11.spines[["top", "right"]].set_visible(False)

    # --- [1,2] Cross-channel correlation (X) ---
    ax12 = fig.add_subplot(gs[1, 2])
    im = ax12.imshow(cross_x, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax12.set_xlabel("Channel index")
    ax12.set_ylabel("Channel index")
    ax12.set_title(f"Cross-channel corr X\n(off-diag mean = {mean_cross_x:.3f})")
    plt.colorbar(im, ax=ax12, shrink=0.8)

    # --- [2,0-1] SEM vs N (assuming independence) ---
    ax20 = fig.add_subplot(gs[2, :2])
    ax20.plot(N_arr, sem_x, "C0-o", ms=4, label=f"X  sigma={sigma_x:.3f}um")
    ax20.plot(N_arr, sem_y, "C1-o", ms=4, label=f"Y  sigma={sigma_y:.3f}um")
    # Correlation-corrected version
    N_cont = np.linspace(1, n_ch + 10, 200)
    sem_x_corr_curve = sigma_x / np.sqrt(N_cont / (1 + (N_cont - 1) * max(mean_cross_x, 0)))
    sem_y_corr_curve = sigma_y / np.sqrt(N_cont / (1 + (N_cont - 1) * max(mean_cross_y, 0)))
    ax20.plot(N_cont, sem_x_corr_curve, "C0--", lw=1, alpha=0.7,
              label=f"X corr-corrected (rho={mean_cross_x:.2f})")
    ax20.plot(N_cont, sem_y_corr_curve, "C1--", lw=1, alpha=0.7,
              label=f"Y corr-corrected (rho={mean_cross_y:.2f})")
    ax20.axhline(0.01, color="k", ls=":", lw=1.5, label="0.01 um threshold")
    ax20.axvline(n_ch, color="gray", ls="--", lw=1, label=f"Current ({n_ch} ch)")
    ax20.set_xlabel("Number of channels averaged (N)")
    ax20.set_ylabel("SEM (um)")
    ax20.set_title("Precision vs number of channels averaged\n(solid = independent, dashed = with cross-channel correlation)")
    ax20.legend(frameon=False, fontsize=7, loc="upper right")
    ax20.set_ylim(0, sigma_x * 1.1)
    ax20.spines[["top", "right"]].set_visible(False)

    # --- [2,2] Summary text ---
    ax22 = fig.add_subplot(gs[2, 2])
    ax22.axis("off")
    summary = (
        f"sigma per channel:\n"
        f"  X: {sigma_x:.4f} um\n"
        f"  Y: {sigma_y:.4f} um\n\n"
        f"N channels to SEM < 0.01 um\n"
        f"  (assuming independence):\n"
        f"  X: {n_need_x},  Y: {n_need_y}\n\n"
        f"Off-diag cross-correlation:\n"
        f"  X: {mean_cross_x:.3f}  Y: {mean_cross_y:.3f}\n\n"
        f"Effective N (with correlation):\n"
        f"  X: {neff_x:.1f}  Y: {neff_y:.1f}\n\n"
        f"Current SEM ({n_ch} ch):\n"
        f"  X: {sem_x[-1]:.4f} um  (indep)\n"
        f"     {sem_x_corr:.4f} um  (corr-corrected)\n"
        f"  Y: {sem_y[-1]:.4f} um  (indep)\n"
        f"     {sem_y_corr:.4f} um  (corr-corrected)"
    )
    ax22.text(0.05, 0.97, summary, va="top", ha="left",
              fontsize=8, family="monospace", transform=ax22.transAxes)

    mode_label = "fixed ref (frame 0)" if args.mode == "fixed" else "sliding pairs (frame_i → frame_{i+1})"
    fig.suptitle(
        f"ECC Precision: Noise Characterization\n"
        f"[{mode_label}  |  residual: {residual_label}  |  {n_samples} samples]",
        fontsize=11,
    )

    save_figure(
        fig,
        params={
            "n_tp": n_tp,
            "n_channels": n_ch,
            "sigma_x_um": float(sigma_x),
            "sigma_y_um": float(sigma_y),
            "n_need_x": n_need_x,
            "n_need_y": n_need_y,
            "mean_cross_corr_x": float(mean_cross_x),
            "mean_cross_corr_y": float(mean_cross_y),
            "effective_n_x": float(neff_x),
            "effective_n_y": float(neff_y),
            "sem_x_at_n_ch": float(sem_x[-1]),
            "sem_y_at_n_ch": float(sem_y[-1]),
            "sem_x_corr_corrected": float(sem_x_corr),
            "sem_y_corr_corrected": float(sem_y_corr),
        },
        description=(
            f"ECC precision characterization: sigma_x={sigma_x:.4f}um, "
            f"sigma_y={sigma_y:.4f}um, cross_corr_x={mean_cross_x:.3f}"
        ),
        data={
            "tx_um": tx_um,
            "ty_um": ty_um,
            "res_x": res_x,
            "res_y": res_y,
            "trend_x": trend_x,
            "trend_y": trend_y,
            "acf_x": np.array(acf_x_all) if acf_x_all else np.array([]),
            "acf_y": np.array(acf_y_all) if acf_y_all else np.array([]),
            "cross_x": cross_x,
            "cross_y": cross_y,
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
