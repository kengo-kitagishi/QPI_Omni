"""_fig_shotnoise_pos0_261121.py

Figure: phase noise per pixel against the shot-noise prediction, on a structure-free field.

Data: Pos0 (background position, no channels) of the 261121 11x11 grid: 121 single-shot holograms taken
consecutively (about 65 s) at commanded stage offsets within +-0.15 um, reconstructed by the pipeline
(output_phase_raw, 511 x 511 px).

Definitions
  q_k(x)            phase image k minus its Gaussian blur (sigma 8 px)  [removes the slow phase background,
                    which changes from shot to shot; keeps 99.8 % of white noise]
  temporal sigma    sigma_t(x) = standard deviation over k = 1..121 of q_k(x), per pixel (ddof 1)
  prediction        sigma_p(x) = sqrt(2 f / (N(x) V(x)^2))
                    N = electrons per raw pixel (local mean of the hologram x gain), V = local fringe visibility
                    (2|sideband|/DC), both low-passed over ~34 raw px; f = area of the reconstruction filter
                    disk / area of the spectrum.  From the raw holograms (_qc_shotnoise_source_261121.py).
  spatial sigma     s_j = standard deviation over the pixels of the region of (q_k - q_k') for consecutive
                    shots k, k' (one number per pair).  If the noise is independent between pixels and the
                    field carries no structure, s^2 = 2 x mean over pixels of sigma_t(x)^2.

    python scripts/_fig_shotnoise_pos0_261121.py
"""
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from figure_logger import save_figure
import matplotlib.pyplot as plt

RECON = r"D:\AquisitionData\Kitagishi\261121\recon_0p03um_11x11"
SRC = r"D:\AquisitionData\Kitagishi\261121\shotnoise_source\pos0_pairs_shotnoise.npz"
GAIN = 2.6               # e-/DN, acA2440-75um (10.7 ke- / 4095 DN); photon transfer on these images: 2.4-2.6
BORDER = 30
# scripts/paper.mplstyle is missing on the microscope PC; same settings, set explicitly
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none"})


def main():
    dirs = sorted(glob.glob(os.path.join(RECON, "Pos0_x*")), key=os.path.getmtime)
    files = [glob.glob(os.path.join(d, "output_phase_raw", "img_*_phase.tif"))[0] for d in dirs]
    q = np.array([(lambda p: p - gaussian_filter(p, 8))(tifffile.imread(f).astype(np.float64)) for f in files])
    n = len(q)
    c = (slice(BORDER, -BORDER), slice(BORDER, -BORDER))
    sig_t = q.std(axis=0, ddof=1)
    z = np.load(SRC)
    pred_single = np.sqrt(np.median(z["pred_var"], axis=0) / 2 * float(z["gain_e_per_dn"]) / GAIN)
    w = np.random.default_rng(0).standard_normal((511, 511))
    keep = float((w - gaussian_filter(w, 8)).std() / w.std())
    sig_p = pred_single * keep
    ratio = sig_t / sig_p
    st, sp, rr = sig_t[c].ravel(), sig_p[c].ravel(), ratio[c].ravel()
    # spatial sigma of the difference of consecutive shots
    s_pair = np.array([(q[k + 1] - q[k])[c].std() for k in range(0, n - 1, 2)])
    s_expected = float(np.sqrt(2 * np.mean(st ** 2)))
    # independence between neighbouring pixels: correlation of the temporal fluctuations
    d = q - q.mean(axis=0)
    corr1 = float(np.mean(d[:, c[0], c[1]][:, :, 1:] * d[:, c[0], c[1]][:, :, :-1]) / np.mean(d[:, c[0], c[1]] ** 2))
    # sampling scatter of sigma_t for n frames
    se = 1 / np.sqrt(2 * (n - 1))
    print(f"{n} Pos0 frames, {st.size} pixels (border {BORDER} px dropped)")
    print(f"temporal sigma per pixel : median {np.median(st)*1e3:.2f} mrad (IQR {np.percentile(st,25)*1e3:.2f}-{np.percentile(st,75)*1e3:.2f})")
    print(f"shot-noise prediction    : median {np.median(sp)*1e3:.2f} mrad (IQR {np.percentile(sp,25)*1e3:.2f}-{np.percentile(sp,75)*1e3:.2f})")
    print(f"ratio temporal/prediction: median {np.median(rr):.3f} (IQR {np.percentile(rr,25):.3f}-{np.percentile(rr,75):.3f}); "
          f"expected pixel-to-pixel scatter from {n} frames alone: +-{se*100:.1f} %")
    print(f"spatial sigma of one difference image: median {np.median(s_pair)*1e3:.2f} mrad over {len(s_pair)} pairs "
          f"(range {s_pair.min()*1e3:.2f}-{s_pair.max()*1e3:.2f}); sqrt(2 x mean sigma_t^2) = {s_expected*1e3:.2f} mrad; "
          f"ratio {np.median(s_pair)/s_expected:.3f}")
    print(f"correlation of the fluctuation between horizontally adjacent pixels: {corr1:+.3f}")

    fig = plt.figure(figsize=(183 / 25.4, 105 / 25.4))
    gsp = fig.add_gridspec(2, 3, left=0.06, right=0.97, top=0.92, bottom=0.11, wspace=0.42, hspace=0.45)
    vmax = float(np.percentile(np.r_[st, sp], 99.5) * 1e3)
    vmin = float(np.percentile(np.r_[st, sp], 0.5) * 1e3)
    ext = [0, 511 * 0.3457, 511 * 0.3457, 0]
    for k, (img, title, cmap, lim, lab) in enumerate((
            (sig_t * 1e3, "measured: temporal σ per pixel", "cividis", (vmin, vmax), "σ [mrad]"),
            (sig_p * 1e3, "predicted: shot noise per pixel", "cividis", (vmin, vmax), "σ [mrad]"),
            (ratio, "measured / predicted", "RdBu_r", (0.7, 1.3), "ratio"))):
        ax = fig.add_subplot(gsp[0, k])
        h = ax.imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1], extent=ext, interpolation="nearest", rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("x [µm]"); ax.set_ylabel("y [µm]")
        cb = fig.colorbar(h, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(lab); cb.outline.set_linewidth(0.5)
        ax.text(-0.22, 1.08, "abc"[k], transform=ax.transAxes, fontweight="bold", fontsize=9)

    ax = fig.add_subplot(gsp[1, 0])
    lo, hi = vmin * 0.9, vmax * 1.05
    ax.hist2d(sp * 1e3, st * 1e3, bins=120, range=[[lo, hi], [lo, hi]], cmap="Greys", norm="log", rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="#D55E00", lw=0.8)
    ax.set_xlabel("predicted shot-noise σ [mrad]"); ax.set_ylabel("measured temporal σ [mrad]")
    ax.set_aspect("equal")
    ax.text(0.05, 0.95, f"{st.size:,} pixels\nmedian ratio {np.median(rr):.2f}", transform=ax.transAxes, va="top", fontsize=6)
    ax.text(-0.22, 1.08, "d", transform=ax.transAxes, fontweight="bold", fontsize=9)

    ax = fig.add_subplot(gsp[1, 1])
    ax.hist(rr, bins=np.linspace(0.6, 1.4, 81), color="#7f7f7f", density=True)
    xx = np.linspace(0.6, 1.4, 400)
    ax.plot(xx, np.exp(-0.5 * ((xx - np.median(rr)) / se) ** 2) / (se * np.sqrt(2 * np.pi)), color="#0072B2", lw=0.8)
    ax.axvline(1, color="#D55E00", lw=0.8)
    ax.set_xlabel("measured / predicted (per pixel)"); ax.set_ylabel("probability density")
    ax.text(-0.22, 1.08, "e", transform=ax.transAxes, fontweight="bold", fontsize=9)

    ax = fig.add_subplot(gsp[1, 2])
    rng = np.random.default_rng(1)
    ax.scatter(rng.uniform(-0.12, 0.12, len(s_pair)), s_pair * 1e3, s=5, color="#7f7f7f", lw=0, rasterized=True)
    ax.plot([-0.25, 0.25], [s_expected * 1e3] * 2, color="#0072B2", lw=1)
    ax.plot([-0.25, 0.25], [np.sqrt(2 * np.mean(sp ** 2)) * 1e3] * 2, color="#D55E00", lw=1, ls="--")
    ax.set_xlim(-0.6, 1.6); ax.set_xticks([])
    ax.set_ylim(s_expected * 1e3 * 0.85, s_expected * 1e3 * 1.15)
    ax.set_ylabel("σ of one difference image [mrad]")
    ax.text(0.32, s_expected * 1e3, "from temporal σ", color="#0072B2", va="center", fontsize=6)
    ax.text(0.32, np.sqrt(2 * np.mean(sp ** 2)) * 1e3 * 0.965, "from prediction", color="#D55E00", va="center", fontsize=6)
    ax.text(0, s_expected * 1e3 * 0.875, f"spatial σ, {len(s_pair)} pairs", ha="center", va="bottom", fontsize=6, color="#555555")
    ax.text(-0.22, 1.08, "f", transform=ax.transAxes, fontweight="bold", fontsize=9)

    caption = (
        "The phase noise of a single reconstructed image equals the shot-noise prediction pixel by pixel. "
        f"Data: {n} single-shot holograms of a field without channels (background position Pos0), taken consecutively in "
        "about 65 s at commanded stage offsets within +-0.15 um, reconstructed to 511 x 511 px phase images (0.3457 um/px). "
        "Before any statistics, the Gaussian blur (sigma 8 px) of each phase image is subtracted to remove the slow phase "
        "background, which changes from shot to shot; this keeps 99.8% of white noise. "
        f"(a) Measured temporal noise: for every pixel, the standard deviation of its value over the {n} images. "
        "(b) Predicted shot noise per pixel, sigma = sqrt(2f / (N V^2)): N = electrons per raw camera pixel (local mean of the "
        f"hologram x {GAIN} e-/DN), V = local fringe visibility (2 x sideband amplitude / mean signal), both low-passed over "
        f"about 34 raw px; f = {float(z['filter_fraction']):.4f} = area of the reconstruction filter disk (radius NA/lambda) / area "
        "of the spectrum. N and V from the raw holograms of the same acquisition (median over 30 shots). Same colour scale as (a). "
        "(c) Ratio (a)/(b). "
        f"(d) Two-dimensional histogram of (a) against (b) over the {st.size:,} pixels at least {BORDER} px from the border "
        "(grey, logarithmic counts); orange line: identity. "
        "(e) Distribution of the per-pixel ratio (grey); blue curve: scatter expected from estimating a standard deviation "
        f"from {n} samples (+-{se*100:.1f}%), centred on the median; orange line: 1. "
        "(f) Standard deviation over the pixels of the difference of two consecutive images (one grey dot per pair, "
        f"{len(s_pair)} pairs, horizontal jitter for display), compared with sqrt(2 x mean over pixels of the temporal "
        "variance) (blue) and the same from the predicted variance (orange dashed). Agreement of dots and blue line means that "
        "the spatial standard deviation of a difference image measures the temporal noise (noise independent between pixels, "
        "no structure left in the difference). "
        "No statistical test. Sample: PDMS device, EMM 2% glucose, 1 mL/h; off-axis QPI at 658 nm, 40x NA 0.95, 60 ms "
        "exposure, Basler acA2440-75um, 12 bit."
    )
    out = save_figure(fig, params={"recon": RECON, "source": SRC, "gain_e_per_dn": GAIN, "hp_sigma_px": 8, "border_px": BORDER, "n_frames": n},
                      description="Per-pixel temporal phase noise vs shot-noise prediction on a structure-free field (Pos0)",
                      data={"sigma_temporal_rad": sig_t, "sigma_predicted_rad": sig_p, "ratio": ratio,
                            "spatial_sigma_pairs_rad": s_pair, "spatial_sigma_expected_rad": s_expected,
                            "neighbour_correlation": corr1, "n_frames": n},
                      caption=caption, fmt="pdf")
    fig.savefig(Path(out).with_suffix(".png"), dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main()
