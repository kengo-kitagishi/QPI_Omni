"""Pos58 ch02 (cell-free trap, 261030 YE timelapse): measured phase noise vs the shot-noise limit.

Shot-noise limit per reconstructed pixel (qpi._get_phase_noise):
    sigma_shot = sqrt(2 * S_aperture / (V^2 * I_DC[e-] * N_sensor_px))
with V and I_DC taken from a raw hologram of this Pos inside the ch02 ROI.

Measured (same ROI, plane removed first):
    single   : spatial std of the reconstructed phase of one frame (output_phase)
    diff     : std(phi_t+1 - phi_t) / sqrt(2) over consecutive output_phase frames (static terms cancel)
    crop_sub : spatial std of the online crop_sub (drift-corrected, grid-subtracted, tilt-corrected) frames
A grid-subtracted image holds two independent shots, so its shot-noise limit is sqrt(2) * sigma_shot.

    python pos58_ch02_noise_vs_shot_261030.py <raw hologram dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from figure_logger import save_figure  # noqa: E402  (applies paper.mplstyle)
import matplotlib.pyplot as plt  # noqa: E402
from optical_config import WAVELENGTH, NA, OFFAXIS_CENTER  # noqa: E402
from qpi import QPIParameters, _get_dc_ac, _get_visibility, _get_phase_noise  # noqa: E402

POS, CH = 58, 2
POS_SPLIT = 59
CROP_BEFORE, CROP_AFTER = (0, 2048, 400, 2448), (0, 2048, 0, 2048)
PIXELSIZE = 3.45e-6 / 40
SENSOR_FULL_WELL_E = 10340          # same sensor constants as the noise analyses (41_phase_diff_noise.py)
SENSOR_BIT_DEPTH = 12
E_PER_ADU = SENSOR_FULL_WELL_E / 2 ** SENSOR_BIT_DEPTH

TL = Path(rf"E:\261030\ph_YE_5min_31d\Pos{POS}\z000")
CROP_SUB = Path(r"D:\AquisitionData\Kitagishi\261030\online_crop_sub_YE_5min_31d"
                rf"\Pos{POS}\output_phase\channels\crop_sub_rawraw\z000\ch{CH:02d}")
ROIS = Path(rf"D:\AquisitionData\Kitagishi\261029\recon_YE_grid\Pos{POS}_x+0_y+0\output_phase\channels\channel_rois.json")


def deplane(a):
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(a.size)]
    coef, *_ = np.linalg.lstsq(A, a.ravel(), rcond=None)
    return a - (A @ coef).reshape(a.shape)


def roi_slice(shape):
    r = json.loads(ROIS.read_text(encoding="utf-8"))[CH]
    h, w = r["crop_w"], r["crop_h"]          # channels are stacked along y; the trap runs along x
    y0, x0 = r["cy"] - h // 2, r["cx"] - w // 2
    y0, x0 = max(y0, 0), max(x0, 0)
    return slice(y0, min(y0 + h, shape[0])), slice(x0, min(x0 + w, shape[1]))


def main():
    raw_dir = Path(sys.argv[1])
    raws = sorted(raw_dir.glob("img_*_ph_000.tif"))
    if not raws:
        raise FileNotFoundError(f"no raw hologram in {raw_dir}")
    crop = CROP_BEFORE if POS < POS_SPLIT else CROP_AFTER

    shot, vis_med, dc_med_e, raw_max = [], [], [], 0
    for p in raws:
        holo = tifffile.imread(str(p)).astype(np.float64)
        raw_max = max(raw_max, float(holo.max()))
        holo = holo[crop[0]:crop[1], crop[2]:crop[3]]
        params = QPIParameters(wavelength=WAVELENGTH, NA=NA, img_shape=holo.shape,
                               pixelsize=PIXELSIZE, offaxis_center=OFFAXIS_CENTER)
        dc, ac = _get_dc_ac(holo, params)
        dc, ac = np.asarray(getattr(dc, "get", lambda: dc)()), np.asarray(getattr(ac, "get", lambda: ac)())
        vis = np.asarray(_get_visibility(dc, ac))
        dc_e = np.abs(dc) * E_PER_ADU
        sig = np.asarray(_get_phase_noise(vis, params.aperturesize, dc_e, holo.shape))
        sy, sx = roi_slice(sig.shape)
        shot.append(float(np.median(sig[sy, sx])))
        vis_med.append(float(np.median(vis[sy, sx])))
        dc_med_e.append(float(np.median(dc_e[sy, sx])))
    if raw_max > 2 ** SENSOR_BIT_DEPTH:
        raise ValueError(f"raw max {raw_max:.0f} > {2 ** SENSOR_BIT_DEPTH}: not 12-bit counts, e-/ADU would be wrong")
    sigma_shot = float(np.mean(shot))

    phases = sorted((TL / "output_phase").glob("img_*_ph_000_phase.tif"))
    ph = [tifffile.imread(str(p)).astype(np.float64) for p in phases]
    sy, sx = roi_slice(ph[0].shape)
    single = [float(np.std(deplane(a[sy, sx]))) for a in ph]
    diff = [float(np.std(deplane(b[sy, sx] - a[sy, sx])) / np.sqrt(2)) for a, b in zip(ph, ph[1:])]
    cs = [tifffile.imread(str(p)).astype(np.float64) for p in sorted(CROP_SUB.glob("img_*_ph_000.tif"))]
    crop_sub = [float(np.std(deplane(a))) for a in cs]

    res = {
        "visibility_median": float(np.mean(vis_med)), "dc_e_median": float(np.mean(dc_med_e)),
        "sigma_shot_mrad": sigma_shot * 1e3, "sigma_shot_sub_mrad": sigma_shot * np.sqrt(2) * 1e3,
        "single_frame_mrad": float(np.mean(single)) * 1e3,
        "frame_diff_mrad": float(np.mean(diff)) * 1e3 if diff else float("nan"),
        "crop_sub_mrad": float(np.mean(crop_sub)) * 1e3,
        "n_raw": len(raws), "n_phase": len(ph), "n_crop_sub": len(cs), "roi_px": [sy.stop - sy.start, sx.stop - sx.start],
    }
    for k, v in res.items():
        print(f"{k:22s} {v}")

    fig, ax = plt.subplots(figsize=(89 / 25.4, 60 / 25.4))
    # the single-frame std is the static trap structure (~1 rad), not noise: printed, not plotted
    groups = {"shot\n1 shot": [sigma_shot * 1e3], "frame\ndiff/√2": [d * 1e3 for d in diff],
              "shot\n×√2": [sigma_shot * np.sqrt(2) * 1e3], "crop\nsub": [c * 1e3 for c in crop_sub]}
    for i, (name, vals) in enumerate(groups.items()):
        vals = np.asarray(vals)
        ax.bar(i, vals.mean(), color="0.75" if "shot" in name else "#0072B2", width=0.6)
        ax.plot(np.full(vals.size, i), vals, "o", ms=3, color="k")
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_ylabel("phase std (mrad)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Pos{POS} ch{CH:02d}, 261030 YE (V={res['visibility_median']:.2f})")
    save_figure(
        fig,
        params={"pos": POS, "ch": CH, "e_per_adu": E_PER_ADU, "crop": crop, "raws": [p.name for p in raws]},
        description=(f"Pos{POS} ch{CH:02d} cell-free trap: measured phase std (plane removed in the ch ROI) vs "
                     "shot-noise limit sqrt(2 S_ap / (V^2 I_DC N_px)) from the raw hologram. Bars: mean; dots: frames. "
                     "frame diff = std(phi_t+1 - phi_t)/sqrt2; crop_sub = grid-subtracted, compare to sqrt2 x limit."),
        data={"sigma_shot_per_raw": np.array(shot), "single": np.array(single), "diff": np.array(diff),
              "crop_sub": np.array(crop_sub), "visibility": np.array(vis_med), "dc_e": np.array(dc_med_e)},
        extra_meta=res,
        caption=(f"Phase noise in the cell-free trap Pos{POS} ch{CH:02d} (261030 YE timelapse, 5 min interval). "
                 "Grey: shot-noise limit sqrt(2 S_ap/(V^2 I_DC N_px)) from the raw hologram in the ch ROI "
                 f"(V={res['visibility_median']:.2f}, I_DC={res['dc_e_median']:.0f} e-/px, {E_PER_ADU:.3f} e-/ADU), "
                 "for one shot and x sqrt2 for a grid-subtracted image. Blue: frame diff = std of the plane-removed "
                 "difference of consecutive reconstructed phases / sqrt2 (n=%d pairs); crop sub = spatial std of the "
                 "plane-removed online crop_sub (drift-corrected, grid-subtracted; n=%d frames). Bars: mean; dots: "
                 "individual frames. No statistical test." % (len(diff), len(cs))),
    )


if __name__ == "__main__":
    main()
