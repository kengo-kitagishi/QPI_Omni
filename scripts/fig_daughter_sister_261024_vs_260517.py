"""Daughter cells vs their mother at each mother division: 261024 vs 260517 (2% EMM).

Same channel selection, frame windows and concentration definition as
fig_density_returnmap_261024_vs_260517.py (imported from there).

For every QC-validated mother division at frame f (daughter = divisions.daughter_id):
  mother_pre   = mother (cell 0), mean over valid frames in [f - W, f - 1]
  mother_post  = mother, mean over valid frames in [f, f + W - 1]
  daughter     = daughter, mean over valid frames in [f, f + W - 1]
  W = WINDOW_FRAMES (1 h). Each mean needs >= MIN_VALID valid frames.
Valid frame = not is_outlier, not touches_border, finite density.

a  concentration of all valid daughter frames (any in-tree cell except the mother) vs mother frames
b  sister pair: mother_post vs daughter concentration
c  mother_pre vs daughter concentration
d  sister pair: mother_post vs daughter dry mass
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

WINDOW_FRAMES = 12      # 1 h at 5 min/frame
MIN_VALID = 3


def valid_frames(f, ds, chans):
    lo, hi = base.FRAME_WINDOWS[ds]
    v = f[f["frame"].between(lo, hi)].copy()
    v = v[[(p, c) in chans for p, c in zip(v["pos"], v["ch"])]]
    v = v[(~v["is_outlier"].astype(bool)) & (~v["touches_border"].astype(bool)) & np.isfinite(v["dens"])]
    v["conc"] = v["dens"] * 1000.0
    return v


def pairs(v, dv, ds):
    lo, hi = base.FRAME_WINDOWS[ds]
    d = dv[dv["is_mother_division"].astype(bool) & dv["validated"].astype(bool)
           & dv["frame"].between(lo, hi)]
    by = {k: g.set_index("frame") for k, g in v.groupby(["pos", "ch", "cell_id"])}
    rows = []
    for r in d.itertuples(index=False):
        mo = by.get((r.pos, r.ch, 0))
        da = by.get((r.pos, r.ch, int(r.daughter_id)))
        if mo is None or da is None:
            continue
        f = int(r.frame)
        pre = mo.loc[(mo.index >= f - WINDOW_FRAMES) & (mo.index < f)]
        post = mo.loc[(mo.index >= f) & (mo.index < f + WINDOW_FRAMES)]
        dd = da.loc[(da.index >= f) & (da.index < f + WINDOW_FRAMES)]
        if min(len(pre), len(post), len(dd)) < MIN_VALID:
            continue
        rows.append({"pos": r.pos, "ch": r.ch, "frame": f, "daughter_id": int(r.daughter_id),
                     "mother_pre_conc": pre["conc"].mean(), "mother_post_conc": post["conc"].mean(),
                     "daughter_conc": dd["conc"].mean(), "mother_post_mass": post["mass_pg"].mean(),
                     "daughter_mass": dd["mass_pg"].mean(),
                     "daughter_span_h": (da.index.max() - da.index.min() + 1) * 5 / 60})
    return pd.DataFrame(rows)


def main():
    out = {}
    f, dv, _ = base.load_261024()
    m = base.mothers(f, None, "261024")
    chans = set(zip(m["pos"], m["ch"]))
    v = valid_frames(f, "261024", chans)
    out["261024"] = (v, pairs(v, dv, "261024"))
    f, dv, (keep, tag) = base.load_260517()
    m = base.mothers(f, keep, "260517")
    chans = set(zip(m["pos"], m["ch"]))
    v = valid_frames(f, "260517", chans)
    out["260517"] = (v, pairs(v, dv, "260517"))

    fig, axes = plt.subplots(1, 4, figsize=(183 / 25.4, 50 / 25.4), constrained_layout=True)
    bins = np.arange(150, 451, 5)
    ax = axes[0]
    stats, arrays = {}, {}
    for ds in ("260517", "261024"):
        v, p = out[ds]
        for who, sel, ls in (("mother", v["cell_id"] == 0, "-"), ("daughters", v["cell_id"] != 0, ":")):
            x = v.loc[sel, "conc"].values
            ax.hist(x, bins=bins, density=True, histtype="step", lw=0.9, ls=ls, color=base.COLORS[ds],
                    label=f"{ds} {who} (median {np.median(x):.0f})")
            arrays[f"{ds}_{who}_conc"] = x
            stats[f"{ds}_{who}_median"] = round(float(np.median(x)), 1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.set_xlabel("dry-mass concentration [mg/mL]")
    ax.set_ylabel("probability density [1/(mg/mL)]")
    ax.legend(fontsize=5, loc="upper left", borderaxespad=0.1)
    ax.text(-0.3, 1.02, "a", transform=ax.transAxes, fontweight="bold", fontsize=8)

    specs = [("mother_post_conc", "daughter_conc", "mother after division [mg/mL]",
              "daughter [mg/mL]", (180, 380), "b", "sisters: conc."),
             ("mother_pre_conc", "daughter_conc", "mother before division [mg/mL]",
              "daughter [mg/mL]", (180, 380), "c", "mother → daughter: conc."),
             ("mother_post_mass", "daughter_mass", "mother after division [pg]",
              "daughter [pg]", (0, 40), "d", "sisters: dry mass")]
    for ax, (kx, ky, xl, yl, lim, letter, title) in zip(axes[1:], specs):
        for ds in ("260517", "261024"):
            p = out[ds][1]
            x, y = p[kx].values, p[ky].values
            ok = np.isfinite(x) & np.isfinite(y)
            r = np.corrcoef(x[ok], y[ok])[0, 1]
            stats[f"{ds}_{letter}_r"] = round(float(r), 3)
            stats[f"{ds}_{letter}_n"] = int(ok.sum())
            ax.scatter(x[ok], y[ok], s=1.5, alpha=0.15, lw=0, color=base.COLORS[ds], rasterized=True,
                       label=f"{ds}: r={r:.2f}, n={ok.sum()}")
            arrays[f"{ds}_{kx}"] = x
            arrays[f"{ds}_{ky}"] = y
        ax.plot(lim, lim, color="0.5", lw=0.5, ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=7)
        leg = ax.legend(fontsize=5.5, loc="upper left", markerscale=4, handletextpad=0.2, borderaxespad=0.1)
        for h in leg.legend_handles:
            h.set_alpha(1)
        ax.text(-0.36, 1.02, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)
    for ds in ("260517", "261024"):
        sp = out[ds][1]["daughter_span_h"]
        stats[f"{ds}_daughter_span_h_median"] = round(float(sp.median()), 2)
        arrays[f"{ds}_daughter_span_h"] = sp.values

    caption = (
        "Daughter cells vs their mother at each mother division, 261024 (2% EMM, 55 h) vs 260517 (2% EMM, 7 d). "
        "a, dry-mass concentration of valid mother frames (solid) and of valid frames of every other tracked cell in "
        "the same channels (daughters and their progeny, dotted). b-d, one point per QC-validated mother division "
        f"at frame f: mother after division = mean over valid mother frames in [f, f+{WINDOW_FRAMES - 1}], "
        f"mother before = [f-{WINDOW_FRAMES}, f-1], daughter = mean over valid frames of the newborn daughter in "
        f"[f, f+{WINDOW_FRAMES - 1}] (5 min/frame, >= {MIN_VALID} valid frames each). b and d compare the two sister "
        "cells of the division (concentration, dry mass); c compares the mother before division with the daughter. "
        "r = Pearson correlation; n = divisions; dashed y = x. Concentration = efd density x 1000; 261024 n_medium "
        f"1.335029 carried over from 260517. 260517 master {tag}, img 2-2017, analysis_recommended channels; 261024 "
        f"img 2-666, channels whose mother has >= {base.MIN_MOTHER_FRAMES} valid frames, edge traps excluded. "
        "Descriptive only, no statistical test."
    )
    path = save_figure(
        fig, fmt="png", dpi=300,
        params={"window_frames": WINDOW_FRAMES, "min_valid": MIN_VALID, "master_260517": tag, "stats": stats},
        description="Daughter vs mother at each mother division: concentration distribution, sister and "
                    "mother-daughter correlations (261024 vs 260517, 2% glucose)",
        data=arrays, caption=caption,
        related=[{"note": "same selection as fig_density_returnmap_261024_vs_260517 / fig_autocorr_261024_vs_260517"}],
    )
    print(stats)
    print("saved", path)


if __name__ == "__main__":
    main()
