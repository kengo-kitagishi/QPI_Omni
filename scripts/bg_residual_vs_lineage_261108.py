"""Does the outside_quad background leave a per-trap offset that explains between-lineage differences? (261108 run 3)

Per trap: background residual = median phase of the crop_sub pixels at least DILATE px away from any segmented cell,
taken every STEP frames and averaged over time; its spread over frames (SD) is reported too.
Per lineage (mother): mean concentration, mean birth concentration and mean birth volume over its cycles
(fig_density_returnmap_261024_vs_260517 definitions, imported).
Panels: lineage means vs the trap's background residual, with Pearson r.
"""
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import fig_autocorr_261108 as s  # noqa: E402
import fig_density_returnmap_261024_vs_260517 as base  # noqa: E402
from figure_logger import save_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RAW = Path("D:/AquisitionData/Kitagishi/261108/online_crop_sub_YE_5min_31d_3")
MASK = Path("C:/QPI_analysis/261108/seg")
REL = Path("output_phase/channels/crop_sub_rawraw/z000")
STEP = 10
DILATE = 4


def trap_residual(pos, ch):
    d_raw = RAW / pos / REL / ch
    d_mask = MASK / pos / REL / ch / "inference_out"
    vals = []
    for f in sorted(d_raw.glob("img_*_ph_000.tif"))[2::STEP]:
        img = tifffile.imread(str(f)).astype(np.float64)
        mp = d_mask / (f.stem + "_masks.tif")
        cell = tifffile.imread(str(mp)) > 0 if mp.exists() else np.zeros(img.shape, bool)
        if cell.shape != img.shape:
            continue
        bg = ~binary_dilation(cell, iterations=DILATE)
        if bg.sum() > 200:
            vals.append(float(np.median(img[bg])))
    return (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)


def main():
    base.MIN_MOTHER_FRAMES = s.MIN_MOTHER_FRAMES_261030
    f, dv = s.load()
    m = base.mothers(f, None, s.DS)
    cyc = base.cycles(m, dv, s.DS)[0]
    lin = m[m["valid"]].groupby(["pos", "ch"])["conc"].mean().rename("conc")
    cy = cyc.groupby(["pos", "ch"])[["rhob", "Vb", "Mb"]].mean()
    tab = cy.join(lin)
    res = np.array([trap_residual(p, c) for p, c in tab.index])
    tab["bg_mean"], tab["bg_sd"] = res[:, 0], res[:, 1]
    print(tab.round(3).to_string())
    fig, axes = plt.subplots(1, 4, figsize=(183 / 25.4, 50 / 25.4), constrained_layout=True)
    out = {}
    for ax, (col, lab) in zip(axes, [("conc", "lineage mean conc. [mg/mL]"), ("rhob", "mean birth conc. [mg/mL]"),
                                     ("Mb", "mean birth mass [pg]"), ("Vb", "mean birth volume [µm³]")]):
        x, y = tab["bg_mean"].to_numpy() * 1e3, tab[col].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        out[col] = round(r, 3)
        ax.plot(x[ok], y[ok], "o", ms=3, color="#0072B2")
        ax.set_xlabel("trap background residual [mrad]")
        ax.set_ylabel(lab)
        ax.set_title(f"r = {r:.2f}, n = {ok.sum()}", fontsize=7)
    print("Pearson r vs background residual:", out)
    print("between-trap SD of background residual [mrad]:", round(float(np.nanstd(tab['bg_mean']) * 1e3), 1),
          "| mean within-trap SD over time [mrad]:", round(float(np.nanmean(tab['bg_sd']) * 1e3), 1))
    save_figure(
        fig, fmt="png", dpi=300,
        params={"step": STEP, "dilate_px": DILATE, "r": out},
        description="261108 run 3: per-trap background residual after outside_quad vs lineage means.",
        data={"bg_mean": tab["bg_mean"].to_numpy(), "bg_sd": tab["bg_sd"].to_numpy(),
              **{c: tab[c].to_numpy() for c in ("conc", "rhob", "Mb", "Vb")}},
        extra_meta={"r": out},
        caption=("Each point is one trap / mother lineage (261108 YE run 3, n = traps). x: background residual of the "
                 "crop_sub after the outside_quad background fit = median phase of the pixels at least "
                 f"{DILATE} px from any segmented cell, every {STEP}th frame, averaged over time. y: lineage means over "
                 "QC-validated cycles (birth values = median of the first 3 valid frames after division) or over valid "
                 "mother frames (concentration). r: Pearson correlation. No statistical test beyond r."),
    )


if __name__ == "__main__":
    main()
