"""Mean dry-mass concentration of living mother cells: 261030 (YE) with 261024 (2% EMM) for reference.

Mother = cell_id 0 of a non-edge channel (ch00/ch11 excluded). Valid frame = not is_outlier, not touches_border,
finite density. Concentration [mg/mL] = density_pg_um3 (efd volume) x 1000.
Living window of a mother = first frame .. its last QC-validated mother division; if that division is within
CENSOR_GAP_H of the last analysed frame, the mother counts as alive to the end (observation censored, not dead).
Mothers with fewer than MIN_DIV validated divisions are left out (alive/dead cannot be told).
Per-cell value = mean concentration over the valid frames of the living window.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from figure_logger import save_figure  # noqa: E402  (applies paper.mplstyle)
import matplotlib.pyplot as plt  # noqa: E402

DATASETS = {   # name: (consolidated dir, last analysed frame, colour)
    "261030 YE": (Path(r"C:\QPI_analysis\261030\seg\_lineage_consolidated"), 430, "#D55E00"),
    "261024 2% EMM": (Path(r"D:\261024_seg\_lineage_consolidated"), 666, "#0072B2"),
}
EDGE = {"ch00", "ch11"}
FRAME_H = 5 / 60
CENSOR_GAP_H = 6.0       # ~2 doubling times (median 2.7 h)
MIN_DIV = 2
MIN_FRAMES = 20          # valid frames in the living window


def living_means(cdir, last_frame):
    f = pd.read_csv(cdir / "all_cells_lineage_data3D.csv.gz",
                    usecols=["pos", "ch", "cell_id", "frame", "density_pg_um3", "is_outlier", "touches_border"])
    f = f[(f["cell_id"] == 0) & ~f["ch"].isin(EDGE) & (f["frame"] <= last_frame)]
    f = f[~f["is_outlier"].astype(bool) & ~f["touches_border"].astype(bool) & np.isfinite(f["density_pg_um3"])]
    dv = pd.read_csv(cdir / "all_cells_divisions_qc.csv.gz",
                     usecols=["pos", "ch", "frame", "is_mother_division", "validated"])
    dv = dv[dv["is_mother_division"].astype(bool) & dv["validated"].astype(bool) & (dv["frame"] <= last_frame)]
    divs = dv.groupby(["pos", "ch"])["frame"].agg(["max", "size"])
    rows, frames_alive = [], []
    for (pos, ch), g in f.groupby(["pos", "ch"]):
        if (pos, ch) not in divs.index or divs.loc[(pos, ch), "size"] < MIN_DIV:
            continue
        last_div = int(divs.loc[(pos, ch), "max"])
        censored = (last_frame - last_div) * FRAME_H <= CENSOR_GAP_H
        end = last_frame if censored else last_div
        alive = g[g["frame"] <= end]["density_pg_um3"].to_numpy() * 1000
        if alive.size < MIN_FRAMES:
            continue
        rows.append((pos, ch, float(alive.mean()), alive.size, censored, last_div))
        frames_alive.append(alive)
    t = pd.DataFrame(rows, columns=["pos", "ch", "mean_conc", "n_frames", "alive_to_end", "last_div"])
    return t, np.concatenate(frames_alive) if frames_alive else np.array([])


def main():
    res, data = {}, {}
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 65 / 25.4), constrained_layout=True)
    bins = np.arange(150, 451, 5)
    for name, (cdir, last, col) in DATASETS.items():
        t, pooled = living_means(cdir, last)
        key = name.split()[0]
        n_dead = int((~t["alive_to_end"]).sum())
        res[key] = {"n_mothers": int(len(t)), "n_alive_to_end": int(t["alive_to_end"].sum()), "n_stopped": n_dead,
                    "median_cell_mean": float(t["mean_conc"].median()), "mean_cell_mean": float(t["mean_conc"].mean()),
                    "sd_cell_mean": float(t["mean_conc"].std()), "median_frames": float(np.median(pooled)),
                    "n_frames": int(pooled.size)}
        data[f"{key}_cell_mean"] = t["mean_conc"].to_numpy()
        data[f"{key}_alive_to_end"] = t["alive_to_end"].to_numpy()
        data[f"{key}_frames"] = pooled
        axes[0].hist(t["mean_conc"], bins=bins, density=True, histtype="step", lw=1.0, color=col,
                     label=f"{name}: median {res[key]['median_cell_mean']:.0f}, n={len(t)} mothers")
        axes[1].hist(pooled, bins=bins, density=True, histtype="step", lw=1.0, color=col,
                     label=f"{name}: median {res[key]['median_frames']:.0f}, {pooled.size} frames")
        print(name, res[key])
    axes[0].set_xlabel("per-cell mean concentration [mg/mL]")
    axes[1].set_xlabel("concentration, living frames [mg/mL]")
    for a, lab in zip(axes, "ab"):
        a.set_ylabel("probability density [1/(mg/mL)]")
        a.legend(frameon=False, fontsize=6, loc="upper right")
        a.set_xlim(200, 450)
        a.set_ylim(0, a.get_ylim()[1] * 1.3)    # headroom for the legend
        a.text(-0.16, 1.02, lab, transform=a.transAxes, fontweight="bold")
    save_figure(
        fig, fmt="png", dpi=300,
        params={"censor_gap_h": CENSOR_GAP_H, "min_div": MIN_DIV, "min_frames": MIN_FRAMES,
                "last_frame": {k: v[1] for k, v in DATASETS.items()}},
        description="Dry-mass concentration of living mother cells, 261030 YE vs 261024 2% EMM.",
        data=data, extra_meta=res,
        caption=("Mother cells (cell 0, non-edge traps) while alive: from the first frame to the last QC-validated "
                 f"mother division, or to the last analysed frame if that division is within {CENSOR_GAP_H:.0f} h of it "
                 f"(censored). Mothers with < {MIN_DIV} validated divisions or < {MIN_FRAMES} valid frames excluded. "
                 "Concentration = phase-integral dry mass / efd volume (yellow contour). a, per-cell mean over the living "
                 "window (n = mothers); b, all living frames pooled. 261030 YE: frames 2-430 (contamination after 430); "
                 "261024 2% EMM: frames 2-666. No statistical test."),
    )


if __name__ == "__main__":
    main()
