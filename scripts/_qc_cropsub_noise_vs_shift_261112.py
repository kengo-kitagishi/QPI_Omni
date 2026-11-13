"""_qc_cropsub_noise_vs_shift_261112.py

What separates a smooth crop_sub tile from one covered in high-frequency noise, once the
end traps and the dirty traps are set aside?  One frame (default 7) of the 261112 timelapse,
every channel of every Pos:

  hf_noise   robust high-frequency residual of the crop_sub tile: the tile minus its
             Gaussian blur (sigma 1.5 px), std taken over 8x8 blocks, 30th percentile of
             the blocks (cells make a few blocks large; the 30th percentile ignores them)
  shift_px   |(tx3, ty3)| from the online drift log: how far the frame still sits from the
             grid point that was subtracted (the grid step is 0.145 px, so anything above
             ~0.1 px means the frame is outside the 11x11 grid patch or the shift search
             did not converge)
  corr3      ECC correlation of the frame against that grid point (texture match)
  ref_hf     the same hf metric on the grid reference crop (PosN_x+0_y+0 output_phase,
             the channel ROI): how much fine texture there is to leak when misaligned
  aniso      orientation of the residual's power: fraction of high-frequency power along
             the trap axis vs across it (fringes parallel to the walls -> defocus / wall
             mismatch; isotropic -> speckle / shot noise)

Then: Spearman correlations and a rank regression of hf_noise on shift_px, corr3, ref_hf,
ch position, and a scatter figure.  Nothing is changed on disk except the figure.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_logger  # noqa: F401  (applies paper.mplstyle)
from figure_logger import save_figure
import matplotlib.pyplot as plt

CROP_ROOT = Path(r"D:\AquisitionData\Kitagishi\261112\online_crop_sub_2per_5min_31d_1")
GRID_ROOT = Path(r"D:\AquisitionData\Kitagishi\261112\recon_2per_grid_1")
LOG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261112\drift_log_zstack.json")
OUT_CSV = Path(r"C:\QPI_analysis\261112\_qc")


def hf_metric(a, sigma=1.5, block=8, q=30):
    r = a - ndimage.gaussian_filter(a, sigma)
    h, w = r.shape
    hb, wb = h // block, w // block
    blocks = r[:hb * block, :wb * block].reshape(hb, block, wb, block).std(axis=(1, 3))
    return float(np.percentile(blocks, q)), r


def anisotropy(r):
    """Power of the residual at spatial frequencies > 0.15 cyc/px, split by direction.
    Returns P(along trap axis) / (P(along) + P(across)); the long axis of the tile is the
    trap axis.  0.5 = isotropic."""
    f = np.abs(np.fft.fftshift(np.fft.fft2(r - r.mean()))) ** 2
    h, w = f.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    rad = np.sqrt(fx ** 2 + fy ** 2)
    hi = rad > 0.15
    long_axis = 0 if h >= w else 1          # axis index of the trap (long) direction
    # frequency "along the trap" varies along the long axis -> fringes ACROSS the trap
    if long_axis == 0:
        along = hi & (np.abs(fy) > np.abs(fx))
        across = hi & (np.abs(fx) > np.abs(fy))
    else:
        along = hi & (np.abs(fx) > np.abs(fy))
        across = hi & (np.abs(fy) > np.abs(fx))
    pa, pc = f[along].sum(), f[across].sum()
    return float(pa / (pa + pc)) if pa + pc > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=7)
    args = ap.parse_args()
    fr = args.frame

    log = json.load(open(LOG, encoding="utf-8"))
    entry = [e for e in log if e["timepoint"] == fr][0]
    rows = []
    for p in entry["positions"]:
        label = p["pos_label"]
        pos = int(label[3:])
        rois_p = GRID_ROOT / f"{label}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json"
        rois = json.load(open(rois_p, encoding="utf-8"))
        ref = tifffile.imread(GRID_ROOT / f"{label}_x+0_y+0" / "output_phase"
                              / "img_000000000_ph_000_phase.tif").astype(np.float32)
        n_ch = len(rois)
        for cd in p.get("channel_details", []):
            ch = cd["ch"]
            tif = (CROP_ROOT / label / "output_phase" / "channels" / "crop_sub_rawraw" / "z000"
                   / f"ch{ch:02d}" / f"img_{fr:09d}_ph_000.tif")
            if not tif.exists():
                continue
            a = tifffile.imread(tif).astype(np.float32)
            hf, r = hf_metric(a)
            an = anisotropy(r)
            roi = rois[ch]
            cy, cx = int(roi["cy"]), int(roi["cx"])
            hh, hw = roi["crop_w"] // 2, roi["crop_h"] // 2
            sub = ref[max(0, cy - hh):cy + hh, max(0, cx - hw):cx + hw]
            ref_hf = hf_metric(sub)[0] if sub.size > 64 * 8 else np.nan
            tx, ty = cd.get("tx3", cd.get("tx2", cd.get("tx1"))), cd.get("ty3", cd.get("ty2", cd.get("ty1")))
            corr = cd.get("corr3", cd.get("corr2", cd.get("corr1")))
            xi, yi = cd.get("xi3", cd.get("xi")), cd.get("yi3", cd.get("yi"))
            shift = float(np.hypot(tx, ty)) if tx is not None else np.nan
            on_border = (xi is not None and (abs(xi) == 5 or abs(yi) == 5))
            rows.append(dict(pos=pos, ch=ch, n_ch=n_ch, end_trap=(ch == 0 or ch == n_ch - 1),
                             status=cd.get("status", ""), outlier=bool(cd.get("outlier", False)),
                             hf=hf, aniso=an, ref_hf=ref_hf, shift=shift,
                             corr=float(corr) if corr is not None else np.nan,
                             xi=xi, yi=yi, border=on_border, cy=cy, cx=cx,
                             tile_mean=float(np.median(a)), tile_std=float(a.std())))
    import pandas as pd
    df = pd.DataFrame(rows)
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV / f"cropsub_noise_vs_shift_f{fr:03d}.csv", index=False)
    print(f"frame {fr}: {len(df)} channel tiles; statuses: {df.status.value_counts().to_dict()}")

    good = df[(~df.end_trap) & (~df.outlier) & df['shift'].notna() & df.ref_hf.notna()].copy()
    print(f"after dropping end traps and outlier channels: {len(good)}")
    print(f"hf_noise: median {good.hf.median():.4f}  p10 {good.hf.quantile(.1):.4f}  p90 {good.hf.quantile(.9):.4f} rad")
    print(f"shift_px: median {good['shift'].median():.3f}  p90 {good['shift'].quantile(.9):.3f}  "
          f"max {good['shift'].max():.2f};  on grid border (|xi| or |yi| = 5): {good.border.mean()*100:.0f}%")
    print(f"corr3   : median {good['corr'].median():.5f}  p10 {good['corr'].quantile(.1):.5f}")
    print("\nSpearman rho with hf_noise (all good tiles):")
    for col in ["shift", "corr", "ref_hf", "aniso", "cy", "cx", "pos", "tile_std"]:
        rho, pv = stats.spearmanr(good[col], good.hf, nan_policy="omit")
        print(f"  {col:8s} rho={rho:+.3f}  p={pv:.1e}")
    # within small-shift tiles, what still explains the noise?
    small = good[good['shift'] < good['shift'].quantile(.5)]
    print(f"\nTiles with shift below the median ({good['shift'].quantile(.5):.3f} px), n={len(small)}:")
    for col in ["corr", "ref_hf", "aniso", "cy", "cx", "pos"]:
        rho, pv = stats.spearmanr(small[col], small.hf, nan_policy="omit")
        print(f"  {col:8s} rho={rho:+.3f}  p={pv:.1e}")
    # rank regression
    cols = ["shift", "corr", "ref_hf", "aniso"]
    sub_ = good.dropna(subset=cols + ["hf"])
    X = np.column_stack([np.ones(len(sub_))] + [sub_[c].rank().values / len(sub_) for c in cols])
    y = sub_.hf.rank().values / len(sub_)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    tvals = beta / np.sqrt(np.diag(cov))
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    print("\nrank regression of hf_noise on (shift, corr, ref_hf, aniso):")
    print(f"  R2 = {r2:.3f}  (n={len(y)})")
    for k, v, p in zip(["const"] + cols, beta, pvals):
        print(f"  {k:8s} beta={v:+.3f}  p={p:.1e}")
    # per-Pos: is it Pos-level (optics/illumination) or ch-level?
    pos_med = good.groupby("pos").hf.median()
    within = good.groupby("pos").hf.transform(lambda s: s - s.median())
    print(f"\nvariance split: between-Pos {pos_med.var():.2e}  within-Pos {within.var():.2e}  "
          f"(ratio between/(between+within) = {pos_med.var()/(pos_med.var()+within.var()):.2f})")
    worst = good.sort_values("hf", ascending=False).head(12)[["pos", "ch", "hf", "shift", "corr", "ref_hf", "aniso", "status"]]
    print("\n12 noisiest non-end, non-outlier tiles:")
    print(worst.to_string(index=False))
    best = good.sort_values("hf").head(6)[["pos", "ch", "hf", "shift", "corr", "ref_hf", "aniso"]]
    print("\n6 smoothest:")
    print(best.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 60 / 25.4))
    ax = axes[0]
    sc = ax.scatter(good['shift'], good.hf, c=good['corr'], cmap="cividis", s=6, alpha=.6,
                    linewidths=0, rasterized=True, vmin=good['corr'].quantile(.05), vmax=1.0)
    ax.set_xlabel("residual shift to subtracted grid point [px]")
    ax.set_ylabel("high-frequency residual [rad]")
    cb = fig.colorbar(sc, ax=ax, pad=.02)
    cb.set_label("ECC corr")
    ax = axes[1]
    ax.scatter(good.ref_hf, good.hf, c="0.3", s=6, alpha=.5, linewidths=0, rasterized=True)
    ax.set_xlabel("grid reference texture [rad]")
    ax.set_ylabel("high-frequency residual [rad]")
    ax = axes[2]
    ax.scatter(good.aniso, good.hf, c="0.3", s=6, alpha=.5, linewidths=0, rasterized=True)
    ax.axvline(0.5, color="0.6", lw=0.6)
    ax.set_xlabel("fringe orientation (0.5 = isotropic, >0.5 across trap)")
    ax.set_ylabel("high-frequency residual [rad]")
    fig.suptitle(f"261112 frame {fr}: crop_sub high-frequency residual, {len(good)} tiles "
                 f"(end traps and outlier channels excluded)", fontsize=7)
    fig.tight_layout()
    save_figure(fig, params={"frame": fr, "n_tiles": int(len(good)), "sigma_px": 1.5, "block_px": 8,
                             "percentile": 30, "dataset": "261112", "crop_root": str(CROP_ROOT)},
                description=("Per-channel high-frequency residual of the online crop_sub (tile minus Gaussian "
                             "blur sigma 1.5 px, 30th percentile of 8x8-block std) against the residual "
                             "sub-grid shift, the ECC correlation, the texture of the grid reference crop and "
                             "the orientation of the residual power. One frame, every trap except end traps "
                             "and channels flagged outlier by the drift log."),
                data={"pos": good.pos.values, "ch": good.ch.values, "hf": good.hf.values,
                      "shift_px": good['shift'].values, "corr": good['corr'].values,
                      "ref_hf": good.ref_hf.values, "aniso": good.aniso.values},
                fmt="png")
    plt.close(fig)


if __name__ == "__main__":
    main()
