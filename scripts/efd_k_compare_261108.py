"""EFD smoothing order: does K=3 instead of the adopted K=6 change the between-lineage width/volume differences?

261108 YE run 3, the 17 analysed mother lineages (excluded traps dropped). For the mother (cell 0) every STEP frames,
the Omnipose mask label of the cell is re-measured with mask_volume_schematic.efd_section_geometry at K=6 and K=3
(same 0.5 px inward offset, same centerline). Per lineage: mean width (median section width), mean volume, and
concentration = phase-integral mass / volume. Between-lineage CV and the correlation with concentration are compared.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import fig_autocorr_261108 as s  # noqa: E402
from mask_volume_schematic import efd_section_geometry, PIXEL_SIZE_UM_DEFAULT as PX  # noqa: E402
from figure_logger import save_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

MASK = Path("C:/QPI_analysis/261108/seg")
REL = Path("output_phase/channels/crop_sub_rawraw/z000")
STEP = 10
KS = (6, 3)


def main():
    f = pd.read_csv(s.CONS / "all_cells_lineage_data3D.csv.gz",
                    usecols=["pos", "ch", "cell_id", "frame", "mask_label", "mass_pg", "is_outlier", "touches_border"])
    f = f[(f.cell_id == 0) & f.frame.between(2, 769) & ~f.is_outlier.astype(bool) & ~f.touches_border.astype(bool)]
    ex = set(map(tuple, pd.read_csv(s.EXCLUDED)[["pos", "ch"]].values))
    f = f[[(p, c) not in ex for p, c in zip(f.pos, f.ch)]]
    f = f[f.groupby(["pos", "ch"]).frame.transform("size") >= s.MIN_MOTHER_FRAMES_261030]
    rows = []
    for (pos, ch), g in f.groupby(["pos", "ch"]):
        d = MASK / pos / REL / ch / "inference_out"
        for _, r in g.iloc[::STEP].iterrows():
            mp = d / f"img_{int(r.frame):09d}_ph_000_masks.tif"
            if not mp.exists():
                continue
            cell = tifffile.imread(str(mp)) == int(r.mask_label)
            out = {"pos": pos, "ch": ch, "frame": int(r.frame), "mass": r.mass_pg}
            for k in KS:
                geo = efd_section_geometry(cell, efd_k=k)
                if geo is None:
                    out[f"vol{k}"] = out[f"w{k}"] = np.nan
                    continue
                w = geo.w_perp_px[geo.w_perp_px > 0]
                out[f"vol{k}"] = geo.volume_px3 * PX ** 3
                out[f"w{k}"] = float(np.median(w)) * PX if w.size else np.nan
            rows.append(out)
    t = pd.DataFrame(rows).dropna()
    for k in KS:
        t[f"conc{k}"] = t["mass"] / t[f"vol{k}"] * 1000
    lin = t.groupby(["pos", "ch"]).mean(numeric_only=True)
    res = {"n_lineages": int(len(lin)), "n_frames": int(len(t))}
    for k in KS:
        res[f"K{k}"] = {
            "mean_width_um": round(float(lin[f"w{k}"].mean()), 3),
            "between_CV_width_pct": round(float(lin[f"w{k}"].std() / lin[f"w{k}"].mean() * 100), 2),
            "between_SD_width_px": round(float(lin[f"w{k}"].std() / PX), 3),
            "mean_vol_um3": round(float(lin[f"vol{k}"].mean()), 2),
            "between_CV_vol_pct": round(float(lin[f"vol{k}"].std() / lin[f"vol{k}"].mean() * 100), 2),
            "between_CV_conc_pct": round(float(lin[f"conc{k}"].std() / lin[f"conc{k}"].mean() * 100), 2),
            "r_conc_width": round(float(np.corrcoef(lin[f"conc{k}"], lin[f"w{k}"])[0, 1]), 2),
            "frame_SD_width_px": round(float((t[f"w{k}"] - t.groupby(['pos', 'ch'])[f"w{k}"].transform('mean')).std() / PX), 3),
        }
    res["vol3_over_vol6"] = round(float((t["vol3"] / t["vol6"]).median()), 4)
    res["r_lineage_width_K6_K3"] = round(float(np.corrcoef(lin["w6"], lin["w3"])[0, 1]), 3)
    for k, v in res.items():
        print(k, v)
    fig, axes = plt.subplots(1, 2, figsize=(120 / 25.4, 55 / 25.4), constrained_layout=True)
    axes[0].plot(lin["w6"], lin["w3"], "o", ms=3, color="#0072B2")
    lo, hi = float(min(lin[["w6", "w3"]].min())), float(max(lin[["w6", "w3"]].max()))
    axes[0].plot([lo, hi], [lo, hi], "--", lw=0.6, color="0.5")
    axes[0].set_xlabel("lineage mean width, K=6 [µm]")
    axes[0].set_ylabel("lineage mean width, K=3 [µm]")
    axes[1].plot(lin["vol6"], lin["vol3"], "o", ms=3, color="#0072B2")
    lo, hi = float(min(lin[["vol6", "vol3"]].min())), float(max(lin[["vol6", "vol3"]].max()))
    axes[1].plot([lo, hi], [lo, hi], "--", lw=0.6, color="0.5")
    axes[1].set_xlabel("lineage mean volume, K=6 [µm³]")
    axes[1].set_ylabel("lineage mean volume, K=3 [µm³]")
    save_figure(fig, fmt="png", dpi=300, params={"step": STEP, "ks": KS}, extra_meta=res,
                description="261108 run 3: EFD order K=3 vs K=6, lineage mean width and volume.",
                data={k: t[k].to_numpy() for k in ("mass", "vol6", "vol3", "w6", "w3", "frame")},
                caption=("Mother cells of the 17 analysed 261108 YE run 3 lineages, every 10th frame; each point is one "
                         "lineage mean. Width = median perpendicular section width of the EFD-smoothed contour "
                         "(0.5 px inward offset), volume = solid of revolution (mask_volume_schematic."
                         "efd_section_geometry) with K=6 (adopted) or K=3 harmonics. Dashed: y = x. No statistical test."))


if __name__ == "__main__":
    main()
