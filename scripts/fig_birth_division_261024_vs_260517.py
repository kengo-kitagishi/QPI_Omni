"""Birth -> division-1 and birth -> next birth, same scale on both axes: 261024 vs 260517.

Mother cycles as in fig_density_returnmap_261024_vs_260517.py (imported, incl. the obvious-outlier rule).
For dry mass, volume and concentration of each cycle:
  X_b = median of the first 3 valid frames after the division that starts the cycle (birth)
  X_d = median of the last 3 valid frames up to division-1 (just before the cycle's own division)
  dX  = X_d - X_b (increment over the cycle)
Rows: mass, volume, concentration. Columns:
  1  X_b vs X_d                  (same scale, dashed y = x)
  2  X_b vs dX                   (same scale; OLS slope: 0 adder, -1 sizer, +1 timer)
  3  X_b(n) vs X_b(n+1)          (birth to birth, consecutive cycles of the same mother)
Also writes an interactive HTML of the same panels next to the 261024 QC pages.
"""
import json
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

HTML_OUT = Path(r"D:\261024_seg\_qc\birth_division_261024_vs_260517.html")
ROWS = [("M", "dry mass", "[pg]", (0, 45), (-10, 35)),
        ("V", "volume", "[µm³]", (0, 200), (-40, 160)),
        ("rho", "concentration", "[mg/mL]", (150, 400), (-125, 125))]


def quantities(c, p, key):
    if key == "rho":
        xb, xd = c["rhob"].values, c["rhod"].values
        pb, pn = p["rhob"].astype(float).values, p["rhob_next"].astype(float).values
    else:
        xb, xd = c[key + "b"].values, c[key + "d"].values
        pb, pn = p[key + "b"].astype(float).values, p[key + "b_next"].astype(float).values
    return xb.astype(float), xd.astype(float), pb, pn


def stats(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    slope = np.polyfit(x[ok], y[ok], 1)[0]
    return ok, float(r), float(slope)


def main():
    data = {}
    f, dv, _ = base.load_261024()
    m = base.mothers(f, None, "261024")
    data["261024"] = base.cycles(m, dv, "261024")
    f, dv, (keep, tag) = base.load_260517()
    m = base.mothers(f, keep, "260517")
    data["260517"] = base.cycles(m, dv, "260517")

    fig, axes = plt.subplots(3, 3, figsize=(183 / 25.4, 170 / 25.4), constrained_layout=True)
    arrays, summ, html_panels = {}, {}, []
    letters = iter("abcdefghi")
    for row, (key, name, unit, lim, dlim) in enumerate(ROWS):
        col_specs = [("birth vs division−1", f"birth {unit}", f"division−1 {unit}", lim, lim, "xd"),
                     ("birth vs increment", f"birth {unit}", f"increment (division−1 − birth) {unit}",
                      lim, dlim, "dx"),
                     ("birth n vs birth n+1", f"birth, cycle n {unit}", f"birth, cycle n+1 {unit}", lim, lim, "bb")]
        for col, (title, xl, yl, xlim, ylim, kind) in enumerate(col_specs):
            ax = axes[row, col]
            traces = []
            for ds in ("260517", "261024"):
                c, p = data[ds]
                xb, xd, pb, pn = quantities(c, p, key)
                x, y = {"xd": (xb, xd), "dx": (xb, xd - xb), "bb": (pb, pn)}[kind]
                ok, r, s = stats(x, y)
                summ[f"{ds}_{key}_{kind}"] = {"r": round(r, 3), "slope": round(s, 3), "n": int(ok.sum()),
                                              "median_x": round(float(np.median(x[ok])), 2),
                                              "median_y": round(float(np.median(y[ok])), 2)}
                lab = f"{ds}: r={r:.2f}, slope={s:.2f}, n={ok.sum()}"
                ax.scatter(x[ok], y[ok], s=1.5, alpha=0.15, lw=0, color=base.COLORS[ds], rasterized=True, label=lab)
                arrays[f"{ds}_{key}_{kind}_x"] = x
                arrays[f"{ds}_{key}_{kind}_y"] = y
                traces.append({"x": np.round(x[ok], 3).tolist(), "y": np.round(y[ok], 3).tolist(),
                               "type": "scattergl", "mode": "markers", "name": lab,
                               "marker": {"size": 3, "color": base.COLORS[ds], "opacity": 0.35}})
            ref_y = [0, 0] if kind == "dx" else list(xlim)
            ax.plot(xlim, ref_y, color="0.5", lw=0.5, ls="--")
            traces.append({"x": list(xlim), "y": ref_y, "type": "scatter", "mode": "lines", "showlegend": False,
                           "line": {"color": "#888", "dash": "dash", "width": 1}, "hoverinfo": "skip"})
            # same scale: equal data units per mm on both axes
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_title(f"{name}: {title}", fontsize=7)
            leg = ax.legend(fontsize=4.8, loc="upper left", markerscale=4, handletextpad=0.2, borderaxespad=0.1)
            for h in leg.legend_handles:
                h.set_alpha(1)
            letter = next(letters)
            ax.text(-0.3, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)
            html_panels.append({"id": f"p{letter}", "data": traces,
                                "layout": {"title": f"{letter}  {name}: {title}",
                                           "xaxis": {"title": xl, "range": list(xlim)},
                                           "yaxis": {"title": yl, "range": list(ylim), "scaleanchor": "x"}}})

    caption = (
        "Mother cycles: birth vs just-before-division values, increments and birth-to-birth, same scale on both "
        "axes; 261024 (2% glucose EMM, 55 h) vs 260517 (2% glucose, 7 d). Cycle = between two QC-validated mother "
        "divisions. Birth = median of the first 3 valid frames after the division that starts the cycle; "
        "division-1 = median of the last 3 valid frames up to the frame before the cycle's own division; increment "
        "= division-1 minus birth. Rows: dry mass (pg), volume (yellow-contour efd, um3), dry-mass concentration "
        "(mg/mL = efd density x 1000). Columns: birth vs division-1 (dashed y = x), birth vs increment (dashed y = 0; "
        "OLS slope 0 = adder, -1 = sizer, +1 = timer), birth of cycle n vs n+1. r = Pearson, slope = OLS of y on x, "
        "n = cycles (or cycle pairs). Valid frame = not tracker outlier, not touching the border, within median +- "
        f"{base.OUTLIER_Z} x 1.4826 MAD of the dataset ({base.OUTLIER_LOG}). 261024 n_medium 1.335029 carried over "
        f"from 260517; mothers with >= {base.MIN_MOTHER_FRAMES} valid frames, edge traps excluded. 260517 master "
        f"{tag}, img 2-2017, analysis_recommended channels. Descriptive only."
    )
    path = save_figure(
        fig, fmt="png", dpi=300,
        params={"outliers": base.OUTLIER_LOG, "master_260517": tag, "summary": summ},
        description="Birth vs division-1, birth vs increment, birth vs next birth for mass, volume and "
                    "concentration (same-scale axes): 261024 vs 260517",
        data=arrays, caption=caption,
        related=[{"note": "replaces the cycle-n vs n+1 return maps of fig_density_returnmap as the user meant them"}],
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Birth to division</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>body{{font-family:Arial,sans-serif;margin:16px;background:#fff;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(380px,1fr));gap:12px}}
@media (max-width:1200px){{.grid{{grid-template-columns:1fr}}}}
.p{{height:440px}} p{{font-size:13px;max-width:1200px}}</style></head><body>
<h2>261024 vs 260517: birth → division−1, increment, birth → next birth (same scale)</h2>
<p>{caption}</p><p>Drag to zoom, double-click to reset, click a legend entry to hide it.</p>
<div class="grid">{''.join(f'<div class="p" id="{p["id"]}"></div>' for p in html_panels)}</div>
<script>
const P = {json.dumps(html_panels)};
for (const p of P) {{
  p.layout.margin = {{l:60,r:10,t:40,b:50}}; p.layout.legend = {{x:0.02,y:0.98,font:{{size:10}}}};
  Plotly.newPlot(p.id, p.data, p.layout, {{responsive:true}});
}}
</script></body></html>"""
    HTML_OUT.write_text(html, encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))
    print("saved", path)
    print("html", HTML_OUT)


if __name__ == "__main__":
    main()
