"""Birth to next birth: x = X_b(n), y = X_b(n+1) - X_b(n), 261024 (2% glucose EMM).

Data, mother selection, cycles, obvious-outlier rule and birth values are those of
fig_density_returnmap_261024_vs_260517.py (imported, nothing added):
  X_b = median of the first 3 valid frames after the division that starts the cycle
  pairs = consecutive cycles n, n+1 of the same mother
Quantities: birth concentration [mg/mL], birth volume [um3, efd], birth dry mass [pg].
Figure 1: all pairs; OLS line y = a x + b with slope a, intercept b and n pairs on the panel
(a = 0: birth value carried over, a = -1: no memory of the previous birth value).
Figure 2: the N_EXAMPLES lineages that kept dividing to the end of the run (not dead), one colour each, with
their own OLS slope. x and y share the same scale in every panel. 260517 can be added back through SHOW.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import fig_density_returnmap_261024_vs_260517 as base  # noqa: E402  (also sets rcParams)
from figure_logger import save_figure  # noqa: E402

PANELS = [("rhob", "birth concentration", "mg/mL", (150, 400), (-125, 125)),   # x and y spans equal:
          ("Vb", "birth volume", "µm³", (30, 100), (-35, 35)),                 # same scale on both axes
          ("Mb", "birth dry mass", "pg", (8, 26), (-9, 9))]
SHOW = ("261024",)            # user 2026-11-02: 261024 only
N_EXAMPLES = 6                # lineages that kept dividing to the end, most consecutive pairs first
ALIVE_GAP_H = 6.0             # last validated division within this many hours of the last frame = not dead
EX_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00", "#0072B2"]   # Okabe-Ito
SELECTION = ("Data and selection exactly as fig_density_returnmap_261024_vs_260517.py: 261024 img 2-666, mothers "
             f"with >= {base.MIN_MOTHER_FRAMES} valid frames, edge traps excluded; obvious outliers "
             f"|conc - median| > {base.OUTLIER_Z} x 1.4826 MAD. Birth value = median of the first 3 valid frames after "
             "the division (cycle = between two QC-validated mother divisions); a concentration (efd density x 1000), "
             "b volume (yellow-contour efd), c phase-integral dry mass. x and y on the same scale (equal aspect).")


def xy(p, key):
    x = p[key].astype(float).to_numpy()
    y = p[key + "_next"].astype(float).to_numpy() - x
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def style(ax, title, unit, xlim, ylim, letter):
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{title}, cycle n [{unit}]")
    ax.set_ylabel(f"{title}: n+1 − n [{unit}]")
    ax.legend(fontsize=5, loc="upper right", handlelength=1.2)
    ax.text(-0.22, 1.02, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)


def main():
    loaders = {"261024": lambda: base.load_261024()[:2] + (None,),
               "260517": lambda: (lambda f, dv, kt: (f, dv, kt[0]))(*base.load_260517())}
    data = {}
    for ds in SHOW:
        f, dv, keep = loaders[ds]()
        m = base.mothers(f, keep, ds)
        data[ds] = base.cycles(m, dv, ds)

    # figure 1: all pairs
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 62 / 25.4), constrained_layout=True)
    arrays, fits = {}, {}
    for ax, (key, title, unit, xlim, ylim), letter in zip(axes, PANELS, "abc"):
        for ds in SHOW:
            x, y = xy(data[ds][1], key)
            a, b = np.polyfit(x, y, 1)
            out = int(((x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1])).sum())
            fits[f"{ds}_{key}"] = {"slope": round(float(a), 3), "intercept": round(float(b), 3), "n": int(x.size),
                                   "n_outside_axes": out}     # drawn range only; the fit uses every pair
            arrays.update({f"{ds}_{key}_x": x, f"{ds}_{key}_dy": y})
            ax.scatter(x, y, s=1.5, alpha=0.15, lw=0, color=base.COLORS[ds], rasterized=True)
            xs = np.array(xlim, dtype=float)
            ax.plot(xs, a * xs + b, color=base.COLORS[ds], lw=1.0,
                    label=f"{ds}: slope {a:.2f}, intercept {b:.1f}, n={x.size}")
        style(ax, title, unit, xlim, ylim, letter)
    for k, v in fits.items():
        print(k, v)
    save_figure(
        fig, fmt="png", dpi=300,
        params={"outliers": base.OUTLIER_LOG, "min_mother_frames": base.MIN_MOTHER_FRAMES, "fits": fits,
                "show": list(SHOW)},
        description="Birth value of cycle n vs change to the next birth (n+1 - n): concentration, volume, dry mass; 261024.",
        data=arrays,
        caption=("x = birth value of mother cycle n, y = birth value of cycle n+1 minus that of cycle n (consecutive "
                 "cycles of the same mother). Lines: OLS fit over all pairs; slope, intercept and n (pairs) in the "
                 "legend; slope 0 = birth value carried over, -1 = no memory. Points outside the drawn range are "
                 "counted in the JSON 'fits'; the fits use every pair. " + SELECTION + " No statistical test."),
        related=[{"note": "same data/selection as fig_density_returnmap_261024_vs_260517 (birth return maps c-e)"}],
    )

    # figure 2: a few lineages that did not die (kept dividing to the end)
    ds = "261024"
    c, p = data[ds]
    last = base.FRAME_WINDOWS[ds][1]
    per = c.groupby(["pos", "ch"])["end"].max()
    alive = per[(last - per) * 5.0 / 60.0 <= ALIVE_GAP_H].index
    npairs = p.groupby(["pos", "ch"]).size()
    picks = npairs.reindex(alive).dropna().sort_values(ascending=False).index[:N_EXAMPLES]
    fig2, axes2 = plt.subplots(1, 3, figsize=(183 / 25.4, 62 / 25.4), constrained_layout=True)
    ex_arrays, ex_fits = {}, {}
    for ax, (key, title, unit, xlim, ylim), letter in zip(axes2, PANELS, "abc"):
        for (pos, ch), col in zip(picks, EX_COLORS):
            x, y = xy(p[(p["pos"] == pos) & (p["ch"] == ch)], key)
            a, b = np.polyfit(x, y, 1)
            ex_fits[f"{pos}_{ch}_{key}"] = {"slope": round(float(a), 3), "intercept": round(float(b), 3), "n": int(x.size)}
            ex_arrays.update({f"{pos}_{ch}_{key}_x": x, f"{pos}_{ch}_{key}_dy": y})
            ax.plot(x, y, "o", ms=2.2, color=col, alpha=0.8, mew=0)
            xs = np.array(xlim, dtype=float)
            ax.plot(xs, a * xs + b, color=col, lw=0.8, label=f"{pos} {ch}: slope {a:.2f}, n={x.size}")
        style(ax, title, unit, xlim, ylim, letter)
    for k, v in ex_fits.items():
        print(k, v)
    save_figure(
        fig2, fmt="png", dpi=300,
        params={"lineages": [f"{a} {b}" for a, b in picks], "alive_gap_h": ALIVE_GAP_H, "fits": ex_fits},
        description="Birth to next birth for individual 261024 lineages that kept dividing to the end of the run.",
        data=ex_arrays,
        caption=(f"Same axes as the all-pairs figure for the {len(picks)} 261024 mothers with the most consecutive cycle "
                 f"pairs among those whose last validated division is within {ALIVE_GAP_H:.0f} h of the last frame "
                 "(kept dividing, not dead). One colour per mother; lines: OLS fit per mother (slope, n pairs in the "
                 "legend). " + SELECTION + " No statistical test."),
        related=[{"note": "subset of the all-pairs birth-to-birth figure (same script run)"}],
    )


if __name__ == "__main__":
    main()
