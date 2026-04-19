"""
analyze_stage_repeatability.py
-------------------------------
Analyze images collected by test_stage_repeatability.bsh and
quantitatively compare stage repositioning accuracy vs. ECC precision.

Output:
  <test-dir>/repeatability_results.json  numerical results
  figure saved via figure_logger (inbox / results/figure_inbox)

Interpretation:
  std_repositioning_tx_nm  <- stage Y repositioning accuracy (image X direction)
  std_ecc_precision_tx_nm  <- ECC single-measurement precision (inter-channel sigma, image X)
  ratio_tx                 <- std_repositioning / std_ecc_precision
    ratio ~ 1  -> stage is accurate, ECC precision is the main source of residual
    ratio >> 1 -> stage positioning noise is the main source of residual
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

# ---- import helpers from compute_drift_online ----
_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))
from compute_drift_online import (
    load_config,
    reconstruct_phase,
    compute_backsub_offset,
    extract_rect_roi,
    _tilt_correct,
    to_uint8,
    ecc_align,
    _remove_outliers_mad,
)

try:
    import tifffile
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tifffile"])
    import tifffile


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir",  required=True)
    p.add_argument("--config",    required=True)
    p.add_argument("--n-reps",    type=int, required=True)
    return p.parse_args()


def run_ecc_one_rep(phase: np.ndarray, rois: list, grid_ref_crops: np.ndarray,
                    cfg: dict) -> dict:
    """ECC computation for one frame. Pass1 only (absolute shift measurement).
    Returns: {tx_avg, ty_avg, corr_avg, tx_per_ch, ty_per_ch, corr_per_ch}
    """
    vmin = cfg.get("ecc_vmin", -5.0)
    vmax = cfg.get("ecc_vmax",  2.0)
    tilt_crop_h = cfg.get("tilt_crop_h", 270)
    ecc_crop_h  = cfg.get("ecc_crop_h",  80)
    n_ch = min(len(rois), len(grid_ref_crops))

    tx_list, ty_list, corr_list = [], [], []
    for ch_idx in range(n_ch):
        roi = rois[ch_idx]
        crop = _tilt_correct(phase, roi["cy"], roi["cx"], roi["crop_w"],
                             tilt_crop_h, ecc_crop_h)

        ref_u8 = to_uint8(grid_ref_crops[ch_idx], vmin, vmax)
        cur_u8 = to_uint8(crop, vmin, vmax)

        result = ecc_align(ref_u8, cur_u8)
        if result is not None:
            tx, ty, corr = result
            tx_list.append(tx)
            ty_list.append(ty)
            corr_list.append(corr)

    if not tx_list:
        return None

    n_raw = len(tx_list)
    if n_raw >= 3:
        out_x = _remove_outliers_mad(tx_list)
        out_y = _remove_outliers_mad(ty_list)
        is_out = out_x | out_y
        used = [i for i, o in enumerate(is_out) if not o] or list(range(n_raw))
    else:
        used = list(range(n_raw))

    tx_arr = np.array(tx_list)
    ty_arr = np.array(ty_list)
    cr_arr = np.array(corr_list)

    return {
        "tx_avg":    float(np.mean(tx_arr[used])),
        "ty_avg":    float(np.mean(ty_arr[used])),
        "corr_avg":  float(np.mean(cr_arr[used])),
        "tx_per_ch": tx_arr.tolist(),
        "ty_per_ch": ty_arr.tolist(),
        "n_ch_used": len(used),
        "n_ch_raw":  n_raw,
    }


def main():
    args  = parse_args()
    cfg   = load_config(args.config)
    n_reps   = args.n_reps
    test_dir = Path(args.test_dir)
    pixel_scale = cfg["pixel_scale_um"]

    # ---- Load channel_rois ----
    rois_path = Path(cfg["channel_rois_json"])
    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)
    print(f"ROIs: {len(rois)} channels")

    # ---- Load grid_ref_crops (pass1 reference) ----
    ref_crops_path = Path(cfg["grid_ref_crops_tif"])
    grid_ref_crops = tifffile.imread(str(ref_crops_path)).astype(np.float64)
    if grid_ref_crops.ndim == 2:
        grid_ref_crops = grid_ref_crops[np.newaxis, ...]
    print(f"grid_ref_crops: shape={grid_ref_crops.shape}")

    # ---- Process each rep ----
    print(f"\nProcessing {n_reps} reps...")
    results_per_rep = []
    for rep in range(n_reps):
        fname = f"img_{rep:09d}_ph_000.tif"
        raw_path = test_dir / "Pos_ref" / fname
        bg_path  = test_dir / "Pos_bg"  / fname

        if not raw_path.exists():
            print(f"  rep {rep:3d}: MISSING {raw_path.name} — skip")
            continue

        try:
            phase = reconstruct_phase(raw_path, cfg,
                                      bg_path if bg_path.exists() else None)
        except Exception as e:
            print(f"  rep {rep:3d}: recon failed: {e}")
            continue

        # Mean removal
        h, w = phase.shape
        region = phase[1:h-1, 1:w//2]
        if region.size > 0:
            phase -= np.mean(region)

        res = run_ecc_one_rep(phase, rois, grid_ref_crops, cfg)
        if res is None:
            print(f"  rep {rep:3d}: ECC failed all channels")
            continue

        results_per_rep.append({"rep": rep, **res})
        print(f"  rep {rep:3d}: tx={res['tx_avg']:+.4f}px  ty={res['ty_avg']:+.4f}px"
              f"  corr={res['corr_avg']:.4f}  ({res['n_ch_used']}/{res['n_ch_raw']}ch)")

    if len(results_per_rep) < 3:
        print("ERROR: too few valid reps")
        sys.exit(1)

    # ---- Statistics ----
    tx_avgs = np.array([r["tx_avg"] for r in results_per_rep])
    ty_avgs = np.array([r["ty_avg"] for r in results_per_rep])

    # Temporal std = stage repositioning noise
    std_repo_tx_nm = float(np.std(tx_avgs) * pixel_scale * 1000)
    std_repo_ty_nm = float(np.std(ty_avgs) * pixel_scale * 1000)

    # Within-frame ECC precision = std across channels, averaged over reps
    per_ch_tx_stds, per_ch_ty_stds = [], []
    for r in results_per_rep:
        if len(r["tx_per_ch"]) >= 2:
            per_ch_tx_stds.append(np.std(r["tx_per_ch"]) * pixel_scale * 1000)
            per_ch_ty_stds.append(np.std(r["ty_per_ch"]) * pixel_scale * 1000)
    std_ecc_tx_nm = float(np.mean(per_ch_tx_stds)) if per_ch_tx_stds else float("nan")
    std_ecc_ty_nm = float(np.mean(per_ch_ty_stds)) if per_ch_ty_stds else float("nan")

    ratio_tx = std_repo_tx_nm / std_ecc_tx_nm if std_ecc_tx_nm > 0 else float("nan")
    ratio_ty = std_repo_ty_nm / std_ecc_ty_nm if std_ecc_ty_nm > 0 else float("nan")

    # ---- Verdict ----
    def verdict(ratio):
        if np.isnan(ratio):  return "Cannot determine"
        if ratio < 1.5:      return "Stage is accurate (ECC precision is the main source)"
        if ratio < 3.0:      return "Mixed stage noise (moderate)"
        return                      "Stage positioning noise is dominant"

    # ---- Print summary ----
    print("\n" + "="*55)
    print(" Stage Repositioning Test Results")
    print("="*55)
    print(f" Reps analyzed: {len(results_per_rep)} / {n_reps}")
    print()
    print(f"                       image X (stage Y)  image Y (stage X)")
    print(f"  temporal std [nm]    {std_repo_tx_nm:8.1f}          {std_repo_ty_nm:8.1f}")
    print(f"  ECC precision [nm]   {std_ecc_tx_nm:8.1f}          {std_ecc_ty_nm:8.1f}")
    print(f"  ratio                {ratio_tx:8.2f}x         {ratio_ty:8.2f}x")
    print()
    print(f"  image X (stage Y): {verdict(ratio_tx)}")
    print(f"  image Y (stage X): {verdict(ratio_ty)}")
    print()
    timelapse_residual_nm = 112.0  # from previous drift_log analysis
    print(f"  [Reference] Timelapse residual std ~= {timelapse_residual_nm:.0f} nm")
    print(f"  Current temporal std     = {std_repo_tx_nm:.1f} nm")
    if std_repo_tx_nm > 80:
        print(f"  -> Consistent. Stage noise confirmed as main cause of timelapse oscillation.")
    elif std_repo_tx_nm < 40:
        print(f"  -> Inconsistent. KF velocity feedforward or biological motion is the main cause.")
    else:
        print(f"  -> Partial match. Combination of stage noise + KF.")
    print("="*55)

    # ---- Save JSON results ----
    summary = {
        "n_reps_analyzed":      len(results_per_rep),
        "pixel_scale_um":       pixel_scale,
        "std_repositioning_tx_nm": std_repo_tx_nm,
        "std_repositioning_ty_nm": std_repo_ty_nm,
        "std_ecc_precision_tx_nm": std_ecc_tx_nm,
        "std_ecc_precision_ty_nm": std_ecc_ty_nm,
        "ratio_tx":             ratio_tx,
        "ratio_ty":             ratio_ty,
        "verdict_tx":           verdict(ratio_tx),
        "verdict_ty":           verdict(ratio_ty),
        "per_rep":              results_per_rep,
    }
    results_path = test_dir / "repeatability_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_path}")

    # ---- Figure ----
    _save_figure(tx_avgs, ty_avgs, pixel_scale, std_repo_tx_nm, std_repo_ty_nm,
                 std_ecc_tx_nm, std_ecc_ty_nm, cfg)


def _save_figure(tx_avgs, ty_avgs, pixel_scale, std_repo_tx, std_repo_ty,
                 std_ecc_tx, std_ecc_ty, cfg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(_script_dir))
        from figure_logger import save_figure

        reps = np.arange(len(tx_avgs))
        tx_nm = tx_avgs * pixel_scale * 1000
        ty_nm = ty_avgs * pixel_scale * 1000

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle("Stage Repositioning Test", fontsize=11)

        for ax, vals_nm, std_repo, std_ecc, label, color in [
            (axes[0], tx_nm, std_repo_tx, std_ecc_tx,
             "image X → stage Y", "#e05c2e"),
            (axes[1], ty_nm, std_repo_ty, std_ecc_ty,
             "image Y → stage X", "#2e7de0"),
        ]:
            ax.plot(reps, vals_nm, "o-", color=color, ms=4, lw=1, label="ECC shift")
            ax.axhline(np.mean(vals_nm), ls="--", color="gray", lw=0.8, label="mean")
            ax.fill_between(
                [-0.5, len(reps)-0.5],
                np.mean(vals_nm) - std_repo, np.mean(vals_nm) + std_repo,
                alpha=0.15, color=color, label=f"reposition σ={std_repo:.0f}nm"
            )
            ax.fill_between(
                [-0.5, len(reps)-0.5],
                np.mean(vals_nm) - std_ecc, np.mean(vals_nm) + std_ecc,
                alpha=0.25, color="green", label=f"ECC prec. σ={std_ecc:.0f}nm"
            )
            ax.set_xlabel("Rep #")
            ax.set_ylabel("ECC shift (nm)")
            ax.set_title(label)
            ax.legend(fontsize=7)
            ax.set_xlim(-0.5, len(reps)-0.5)

        plt.tight_layout()

        save_figure(
            fig,
            params={
                "n_reps": len(tx_avgs),
                "std_repositioning_tx_nm": round(std_repo_tx, 1),
                "std_repositioning_ty_nm": round(std_repo_ty, 1),
                "std_ecc_precision_tx_nm": round(std_ecc_tx, 1),
                "std_ecc_precision_ty_nm": round(std_ecc_ty, 1),
                "ratio_tx": round(std_repo_tx / std_ecc_tx, 2) if std_ecc_tx > 0 else None,
                "ratio_ty": round(std_repo_ty / std_ecc_ty, 2) if std_ecc_ty > 0 else None,
            },
            description="Stage repositioning test: temporal std vs. ECC within-frame precision",
            data={
                "tx_nm_per_rep": tx_nm,
                "ty_nm_per_rep": ty_nm,
            },
        )
        print("Figure saved via figure_logger.")
    except Exception as e:
        print(f"[WARN] Figure skipped: {e}")


if __name__ == "__main__":
    main()
