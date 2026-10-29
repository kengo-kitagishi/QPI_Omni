"""Daughter cells followed over their whole generation vs their mother: 261024 vs 260517.

Same channel selection, frame windows and concentration definition as
fig_density_returnmap_261024_vs_260517.py (imported from there).

For every QC-validated mother division at frame f (daughter d = divisions.daughter_id):
  daughter cycle = d from f to its own first QC-validated division (parent_id == d) at frame g.
                   A daughter that leaves the channel / the window before dividing has no complete
                   cycle and is dropped (counted as "incomplete").
  sister cycle   = the mother's cycle that starts at the same division f.
  previous cycle = the mother's cycle that ends at f (the generation that produced d).
Per cycle (valid frames only, as for the mother cycles):
  rho  = cycle-mean concentration, rhob = birth concentration (median of the first 3 valid frames),
  Mb   = birth mass (same frames), T = interdivision time.
A cycle needs >= 5 valid frames.

a  cycle-mean concentration: mother cycles vs complete daughter cycles
b  sisters: rho        c  sisters: rhob        d  previous mother cycle -> daughter: rho
e  sisters: Mb         f  sisters: T
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
import fig_density_returnmap_261024_vs_260517 as base  # noqa: E402  (also sets rcParams)
from figure_logger import save_figure  # noqa: E402

MIN_CYCLE_VALID = 5


def valid_frames(f, ds, chans):
    lo, hi = base.FRAME_WINDOWS[ds]
    v = f[f["frame"].between(lo, hi)].copy()
    v = v[[(p, c) in chans for p, c in zip(v["pos"], v["ch"])]]
    v = v[(~v["is_outlier"].astype(bool)) & (~v["touches_border"].astype(bool)) & np.isfinite(v["dens"])]
    v["conc"] = v["dens"] * 1000.0
    if base.OUTLIER_Z is not None:
        # same obvious-outlier bounds as the mothers of this dataset (set by base.mothers)
        b = base.OUTLIER_LOG[ds]
        v = v[v["conc"].between(b["lo"], b["hi"])]
    return v


def cycle_stats(seg, a, b):
    return {"T_h": (b - a) * 5.0 / 60.0, "rho": seg["conc"].mean(),
            "rhob": seg["conc"].iloc[:3].median(), "Mb": seg["mass_pg"].iloc[:3].median()}


def daughter_cycles(v, dv, ds):
    lo, hi = base.FRAME_WINDOWS[ds]
    ok = dv["validated"].astype(bool) & dv["frame"].between(lo, hi)
    d = dv[ok]
    md = d[d["is_mother_division"].astype(bool)]
    # first validated division of every (pos, ch, parent) -> that cell's own division frame
    first_div = d.groupby(["pos", "ch", "parent_id"])["frame"].apply(lambda s: np.sort(s.values))
    by = {k: g.sort_values("frame").set_index("frame") for k, g in v.groupby(["pos", "ch", "cell_id"])}
    rows, n_incomplete = [], 0
    for (pos, ch), g in md.groupby(["pos", "ch"]):
        fr = np.sort(g["frame"].unique())
        mo = by.get((pos, ch, 0))
        if mo is None:
            continue
        for r in g.itertuples(index=False):
            f = int(r.frame)
            dau = int(r.daughter_id)
            own = first_div.get((pos, ch, dau))
            own = own[own > f] if own is not None else np.array([])
            da = by.get((pos, ch, dau))
            if len(own) == 0 or da is None:
                n_incomplete += 1
                continue
            gd = int(own[0])
            dseg = da.loc[(da.index >= f) & (da.index < gd)]
            nxt, prv = fr[fr > f], fr[fr < f]
            if len(dseg) < MIN_CYCLE_VALID or len(nxt) == 0 or len(prv) == 0:
                n_incomplete += 1
                continue
            sis = mo.loc[(mo.index >= f) & (mo.index < nxt[0])]
            pre = mo.loc[(mo.index >= prv[-1]) & (mo.index < f)]
            if len(sis) < MIN_CYCLE_VALID or len(pre) < MIN_CYCLE_VALID:
                n_incomplete += 1
                continue
            row = {"pos": pos, "ch": ch, "frame": f, "daughter_id": dau}
            for tag_, seg, a, b in (("d", dseg, f, gd), ("s", sis, f, nxt[0]), ("p", pre, prv[-1], f)):
                for k, val in cycle_stats(seg.reset_index(), a, b).items():
                    row[f"{tag_}_{k}"] = val
            rows.append(row)
    return pd.DataFrame(rows), n_incomplete, len(rows) + n_incomplete   # divisions in the selected channels


def main():
    out = {}
    f, dv, _ = base.load_261024()
    m = base.mothers(f, None, "261024")
    v = valid_frames(f, "261024", set(zip(m["pos"], m["ch"])))
    out["261024"] = (base.cycles(m, dv, "261024")[0], *daughter_cycles(v, dv, "261024"))
    f, dv, (keep, tag) = base.load_260517()
    m = base.mothers(f, keep, "260517")
    v = valid_frames(f, "260517", set(zip(m["pos"], m["ch"])))
    out["260517"] = (base.cycles(m, dv, "260517")[0], *daughter_cycles(v, dv, "260517"))

    fig, axes2 = plt.subplots(2, 3, figsize=(183 / 25.4, 112 / 25.4), constrained_layout=True)
    axes = list(axes2.ravel())
    stats, arrays = {}, {}
    ax = axes[0]
    bins = np.arange(180, 401, 5)
    for ds in ("260517", "261024"):
        mc, p, n_inc, n_div = out[ds]
        for who, x, ls in (("mother cycles", mc["rho"].values, "-"), ("daughter cycles", p["d_rho"].values, ":")):
            x = x[np.isfinite(x)]
            ax.hist(x, bins=bins, density=True, histtype="step", lw=0.9, ls=ls, color=base.COLORS[ds],
                    label=f"{ds} {who} (median {np.median(x):.0f}, n={len(x)})")
            arrays[f"{ds}_{who.split()[0]}_cycle_rho"] = x
        stats[f"{ds}_mother_divisions"] = int(n_div)
        stats[f"{ds}_daughters_complete"] = int(len(p))
        stats[f"{ds}_daughters_incomplete"] = int(n_inc)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.set_xlabel("cycle-mean concentration [mg/mL]")
    ax.set_ylabel("probability density [1/(mg/mL)]")
    ax.legend(fontsize=5, loc="upper left", borderaxespad=0.1)
    ax.text(-0.3, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=8)

    specs = [("s_rho", "d_rho", "mother, same generation", "daughter", (180, 380), "b",
              "sisters: cycle-mean conc. [mg/mL]"),
             ("s_rhob", "d_rhob", "mother, same generation", "daughter", (180, 380), "c",
              "sisters: birth conc. [mg/mL]"),
             ("p_rho", "d_rho", "mother, previous generation", "daughter", (180, 380), "d",
              "mother → daughter: mean conc. [mg/mL]"),
             ("s_Mb", "d_Mb", "mother, same generation", "daughter", (0, 40), "e",
              "sisters: birth mass [pg]"),
             ("s_T_h", "d_T_h", "mother, same generation", "daughter", (0, 10), "f",
              "sisters: interdivision time [h]")]
    for ax, (kx, ky, xl, yl, lim, letter, title) in zip(axes[1:], specs):
        for ds in ("260517", "261024"):
            p = out[ds][1]
            x, y = p[kx].astype(float).values, p[ky].astype(float).values
            ok = np.isfinite(x) & np.isfinite(y)
            r = np.corrcoef(x[ok], y[ok])[0, 1]
            stats[f"{ds}_{letter}_r"] = round(float(r), 3)
            stats[f"{ds}_{letter}_n"] = int(ok.sum())
            ax.scatter(x[ok], y[ok], s=2, alpha=0.25, lw=0, color=base.COLORS[ds], rasterized=True,
                       label=f"{ds}: r={r:.2f}, n={ok.sum()}")
            arrays[f"{ds}_{kx}"] = x
            arrays[f"{ds}_{ky}"] = y
        ax.plot(lim, lim, color="0.5", lw=0.5, ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=7)
        leg = ax.legend(fontsize=5.5, loc="upper left", markerscale=4, handletextpad=0.2, borderaxespad=0.1)
        for h in leg.legend_handles:
            h.set_alpha(1)
        ax.text(-0.3, 1.02, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)

    caption = (
        "Daughter cells followed over their whole generation vs their mother, 261024 (2% glucose EMM, 55 h) vs "
        "260517 (2% glucose, 7 d). For every QC-validated mother division, the daughter is followed from birth to its "
        "own first QC-validated division; daughters that leave the channel or the window before dividing are dropped "
        f"(261024: {stats['261024_daughters_complete']} complete of {stats['261024_mother_divisions']} divisions; "
        f"260517: {stats['260517_daughters_complete']} of {stats['260517_mother_divisions']}). Sister = the mother's "
        "cycle starting at the same division; previous = the mother's cycle that ended at it. Per cycle (valid "
        f"frames only, >= {MIN_CYCLE_VALID}): cycle-mean concentration, birth concentration and birth mass (median of "
        "the first 3 valid frames), interdivision time. a, distribution of cycle-mean concentration over mother cycles "
        "(solid) and complete daughter cycles (dotted). b-f, one point per division; r = Pearson; dashed y = x. "
        "Completed daughter cycles are a selected subset (daughters that stayed in the channel until they divided). "
        "Concentration = efd density x 1000; 261024 n_medium 1.335029 carried over from 260517. "
        f"260517 master {tag}, img 2-2017, analysis_recommended channels; 261024 img 2-666, channels whose mother has "
        f">= {base.MIN_MOTHER_FRAMES} valid frames, edge traps excluded. Obvious outliers removed: frames outside "
        f"the mother median +- {base.OUTLIER_Z} x 1.4826 MAD of each dataset ({base.OUTLIER_LOG}). Descriptive only."
    )
    path = save_figure(
        fig, fmt="png", dpi=300,
        params={"outliers": base.OUTLIER_LOG, "min_cycle_valid": MIN_CYCLE_VALID, "master_260517": tag, "stats": stats},
        description="Daughter cells over their whole generation vs mother (sister and mother-daughter "
                    "correlations of concentration, birth mass, interdivision time): 261024 vs 260517",
        data=arrays, caption=caption,
        related=[{"note": "replaces the 1-h-window daughter comparison of 20261029T025402Z_543aee (user: not acceptable)"}],
    )
    print(stats)
    print("saved", path)


if __name__ == "__main__":
    main()
