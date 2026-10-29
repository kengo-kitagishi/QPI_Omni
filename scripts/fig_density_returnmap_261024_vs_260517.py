"""Mother-cell dry-mass density histogram and generation return maps: 261024 vs 260517 (2% glucose).

Data
  261024  D:/261024_seg/_lineage_consolidated (EMM 2%, img 2-666, 55 h), n_medium 1.335029
  260517  data_master/260517/<LATEST>/derived/phase1_img0002-2017 (EMM 2%, img 2-2017, 7 d)

Mother = cell_id 0 of a channel. Valid frame = not is_outlier, not touches_border, finite density.
Concentration [mg/mL] = density_pg_um3 (yellow-contour efd volume) * 1000.
Cycle n = mother between two consecutive QC-validated mother divisions (divisions*.validated).
  T_n    = interdivision time [h]
  rho_n  = mean concentration over the valid frames of the cycle [mg/mL]
  Mb_n   = birth mass: median mass_pg of the first 3 valid frames after the division [pg]
  rhob_n = birth concentration: median of the same first 3 valid frames [mg/mL]
  Vb_n, Vmean_n = birth volume (same 3 frames) / cycle-mean volume [um3, efd]
  dMdt_n = mass production rate: slope of a linear fit of mass_pg vs time over the cycle [pg/h]
Return map = (x_n, x_n+1) over consecutive cycles of the same mother.

Channels: 261024 non-edge channels whose mother has >= MIN_MOTHER_FRAMES valid frames;
260517 channels.csv analysis_recommended.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from figure_logger import save_figure  # noqa: E402

D261024 = Path(r"D:\261024_seg\_lineage_consolidated")
M260517_ROOT = Path(r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\data_master\260517")
EDGE = {"ch00", "ch11"}
MIN_MOTHER_FRAMES = 600          # same selection as the 261024 lineage gallery
FRAME_WINDOWS = {"261024": (2, 666), "260517": (2, 2017)}
COLORS = {"261024": "#0072B2", "260517": "#D55E00"}   # Okabe-Ito blue / vermillion
LABELS = {"261024": "261024 (2% EMM, 55 h)", "260517": "260517 (2% EMM, 7 d)"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def load_261024():
    f = pd.read_csv(D261024 / "all_cells_lineage_data3D.csv.gz",
                    usecols=["pos", "ch", "cell_id", "frame", "time_h", "mass_pg", "volume_um3_efd",
                             "density_pg_um3", "is_outlier", "touches_border"])
    f = f.rename(columns={"density_pg_um3": "dens", "volume_um3_efd": "vol"})
    dv = pd.read_csv(D261024 / "all_cells_divisions_qc.csv.gz")
    return f, dv, None


def load_260517():
    tag = (M260517_ROOT / "LATEST.txt").read_text(encoding="utf-8").strip()
    pkg = M260517_ROOT / tag / "derived" / "phase1_img0002-2017"
    f = pd.read_csv(pkg / "cells_frames.csv.gz",
                    usecols=["pos", "ch", "cell_id", "frame", "time_h", "mass_pg_efd", "volume_um3_efd",
                             "density_pg_um3_efd", "is_outlier", "touches_border"])
    f = f.rename(columns={"density_pg_um3_efd": "dens", "mass_pg_efd": "mass_pg", "volume_um3_efd": "vol"})
    dv = pd.read_csv(pkg / "divisions.csv.gz")
    ch = pd.read_csv(pkg / "channels.csv")
    keep = set(map(tuple, ch.loc[ch["analysis_recommended"].astype(bool), ["pos", "ch"]].values))
    return f, dv, (keep, tag)


def mothers(f, keep, ds):
    lo, hi = FRAME_WINDOWS[ds]
    m = f[(f["cell_id"] == 0) & f["frame"].between(lo, hi)].copy()
    m = m[~m["ch"].isin(EDGE)]
    m["valid"] = (~m["is_outlier"].astype(bool)) & (~m["touches_border"].astype(bool)) & np.isfinite(m["dens"])
    if keep is not None:
        m = m[[(p, c) in keep for p, c in zip(m["pos"], m["ch"])]]
    else:
        nvalid = m[m["valid"]].groupby(["pos", "ch"]).size()
        good = set(nvalid[nvalid >= MIN_MOTHER_FRAMES].index)
        m = m[[(p, c) in good for p, c in zip(m["pos"], m["ch"])]]
    m["conc"] = m["dens"] * 1000.0
    return m


def cycles(m, dv, ds):
    lo, hi = FRAME_WINDOWS[ds]
    d = dv[dv["is_mother_division"].astype(bool) & dv["validated"].astype(bool)
           & dv["frame"].between(lo, hi)]
    rows = []
    for (pos, ch), g in m[m["valid"]].groupby(["pos", "ch"]):
        fr = np.sort(d.loc[(d["pos"] == pos) & (d["ch"] == ch), "frame"].unique())
        g = g.sort_values("frame")
        for i, (a, b) in enumerate(zip(fr[:-1], fr[1:])):
            seg = g[(g["frame"] >= a) & (g["frame"] < b)]
            if len(seg) < 5:
                continue
            slope = np.polyfit(seg["time_h"].values, seg["mass_pg"].values, 1)[0]
            rows.append({"pos": pos, "ch": ch, "k": i, "start": a, "end": b,
                         "T_h": (b - a) * 5.0 / 60.0, "rho": seg["conc"].mean(),
                         "rhob": seg["conc"].iloc[:3].median(),
                         "Mb": seg["mass_pg"].iloc[:3].median(),
                         "Vb": seg["vol"].iloc[:3].median(), "Vmean": seg["vol"].mean(),
                         "dMdt": slope})          # mass production rate: linear fit over the cycle [pg/h]
    c = pd.DataFrame(rows)
    pairs = []
    for (pos, ch), g in c.groupby(["pos", "ch"]):
        g = g.sort_values("k")
        nxt = g.shift(-1)
        ok = (nxt["k"] - g["k"]) == 1
        for _, r in pd.concat([g[ok], nxt[ok].add_suffix("_next")], axis=1).iterrows():
            pairs.append(r)
    return c, pd.DataFrame(pairs)


def main():
    data = {}
    f, dv, _ = load_261024()
    m = mothers(f, None, "261024")
    data["261024"] = (m, *cycles(m, dv, "261024"))
    f, dv, (keep, tag) = load_260517()
    m = mothers(f, keep, "260517")
    data["260517"] = (m, *cycles(m, dv, "260517"))

    fig, axes = plt.subplots(1, 4, figsize=(183 / 25.4, 50 / 25.4), constrained_layout=True)
    bins = np.arange(150, 451, 5)
    ax = axes[0]
    for ds, (m, c, p) in data.items():
        v = m.loc[m["valid"], "conc"].values
        ax.hist(v, bins=bins, density=True, histtype="step", lw=0.9, color=COLORS[ds],
                label=f"{LABELS[ds]}\nmedian {np.median(v):.0f}, n={m.groupby(['pos','ch']).ngroups} mothers")
    ax.set_xlabel("dry-mass concentration [mg/mL]")
    ax.set_ylabel("probability density [1/(mg/mL)]")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)          # headroom so the legend sits above the peaks
    ax.legend(fontsize=5.5, loc="upper left", borderaxespad=0.1)
    ax.text(-0.3, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=8)

    specs = [("rho", "cycle-mean concentration $\\rho_n$ [mg/mL]", (200, 380)),
             ("Mb", "birth mass $M_{b,n}$ [pg]", (0, 45)),
             ("T_h", "interdivision time $T_n$ [h]", (0, 8))]
    rstats = {}
    for ax, (key, lab, lim), letter in zip(axes[1:], specs, "bcd"):
        for ds in ("260517", "261024"):              # 261024 drawn last so it is not hidden
            m, c, p = data[ds]
            x, y = p[key].astype(float).values, p[key + "_next"].astype(float).values
            ok = np.isfinite(x) & np.isfinite(y)
            r = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 2 else np.nan
            rstats[f"{ds}_{key}"] = {"r": float(r), "n_pairs": int(ok.sum())}
            ax.scatter(x[ok], y[ok], s=1.5, alpha=0.15, lw=0, color=COLORS[ds], rasterized=True,
                       label=f"{ds}: r={r:.2f}, n={ok.sum()}")
        ax.plot(lim, lim, color="0.5", lw=0.5, ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(lab.replace("_n", "_n").replace("_{b,n}", "_{b,n}"))
        ax.set_ylabel(lab.replace("$\\rho_n$", "$\\rho_{n+1}$").replace("$M_{b,n}$", "$M_{b,n+1}$")
                      .replace("$T_n$", "$T_{n+1}$"))
        leg = ax.legend(fontsize=5.5, loc="upper left", markerscale=4, handletextpad=0.2,
                        borderaxespad=0.1)
        for h in leg.legend_handles:
            h.set_alpha(1)
        ax.text(-0.36, 1.02, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)

    caption = (
        "Mother-cell dry-mass concentration and generation return maps, 261024 vs 260517 (both 2% glucose EMM). "
        "a, distribution of concentration over all valid mother frames (concentration = dry mass / volume, "
        "dry mass = (mean RI - n_MilliQ) / alpha * V with alpha = 0.00018 mL/mg, V = yellow-contour efd volume; "
        "valid = not tracker outlier, not touching the crop border). "
        "b-d, return maps over consecutive cycles of the same mother (cycle = between two QC-validated mother "
        "divisions): b cycle-mean concentration, c birth mass (median of the first 3 valid frames after division), "
        "d interdivision time. r = Pearson correlation of (x_n, x_n+1); n = number of cycle pairs; dashed line y = x. "
        f"261024: img 2-666, n_medium 1.335029 carried over from 260517 (not measured on this chip), mothers with "
        f">= {MIN_MOTHER_FRAMES} valid frames, edge traps excluded. 260517: master {tag}, phase-1 window img 2-2017, "
        "channels.csv analysis_recommended. No statistical test; descriptive only."
    )
    arrays = {}
    for ds, (m, c, p) in data.items():
        arrays[f"{ds}_conc_valid"] = m.loc[m["valid"], "conc"].values
        for key in ("rho", "Mb", "T_h"):
            arrays[f"{ds}_{key}_n"] = p[key].astype(float).values
            arrays[f"{ds}_{key}_n1"] = p[key + "_next"].astype(float).values
    out = save_figure(
        fig, fmt="png", dpi=300,
        params={"min_mother_frames": MIN_MOTHER_FRAMES, "frame_windows": FRAME_WINDOWS,
                "master_260517": tag, "bins_mg_ml": [150, 450, 5], "return_map_r": rstats},
        description="Mother dry-mass concentration histogram and generation return maps (rho, birth mass, "
                    "interdivision time): 261024 vs 260517, 2% glucose",
        data=arrays, caption=caption,
    )
    for ds, (m, c, p) in data.items():
        print(ds, "mothers", m.groupby(["pos", "ch"]).ngroups, "valid frames", int(m["valid"].sum()),
              "cycles", len(c), "pairs", len(p),
              "conc median", round(float(np.median(m.loc[m["valid"], "conc"])), 1))
    print(rstats)
    print("saved", out)


if __name__ == "__main__":
    main()
