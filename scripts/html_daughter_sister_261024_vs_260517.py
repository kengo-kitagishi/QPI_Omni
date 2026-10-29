"""Interactive HTML of fig_daughter_sister_261024_vs_260517 (zoom / hover), built from its _data.npz.

    python scripts/html_daughter_sister_261024_vs_260517.py <path to ..._data.npz> [--out page.html]
"""
import argparse
import json
from pathlib import Path

import numpy as np

COLORS = {"261024": "#0072B2", "260517": "#D55E00"}
PANELS = [("s_rho", "d_rho", "b  sisters: cycle-mean concentration [mg/mL]", "mother, same generation", [180, 380]),
          ("s_rhob", "d_rhob", "c  sisters: birth concentration [mg/mL]", "mother, same generation", [180, 380]),
          ("p_rho", "d_rho", "d  mother → daughter: cycle-mean concentration [mg/mL]", "mother, previous generation", [180, 380]),
          ("s_Mb", "d_Mb", "e  sisters: birth mass [pg]", "mother, same generation", [0, 40]),
          ("s_T_h", "d_T_h", "f  sisters: interdivision time [h]", "mother, same generation", [0, 10])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=r"D:\261024_seg\_qc\daughter_sister_261024_vs_260517.html")
    a = ap.parse_args()
    z = np.load(a.npz)
    bins = np.arange(180, 401, 5)
    hist = []
    for ds in ("260517", "261024"):
        for who, dash in (("mother", "solid"), ("daughter", "dot")):
            v = z[f"{ds}_{who}_cycle_rho"]
            h, _ = np.histogram(v, bins=bins, density=True)
            hist.append({"x": ((bins[:-1] + bins[1:]) / 2).tolist(), "y": h.round(6).tolist(),
                         "type": "scatter", "mode": "lines",
                         "line": {"shape": "hvh", "color": COLORS[ds], "dash": dash},
                         "name": f"{ds} {who} cycles (median {np.median(v):.0f}, n={len(v)})"})
    panels = [{"id": "hist", "data": hist,
               "layout": {"title": "a  cycle-mean concentration: mother vs complete daughter cycles",
                          "xaxis": {"title": "cycle-mean concentration [mg/mL]"},
                          "yaxis": {"title": "probability density [1/(mg/mL)]"}}}]
    for kx, ky, title, xl, lim in PANELS:
        traces = []
        for ds in ("260517", "261024"):
            x, y = z[f"{ds}_{kx}"], z[f"{ds}_{ky}"]
            ok = np.isfinite(x) & np.isfinite(y)
            r = np.corrcoef(x[ok], y[ok])[0, 1]
            traces.append({"x": x[ok].round(3).tolist(), "y": y[ok].round(3).tolist(), "type": "scattergl",
                           "mode": "markers", "marker": {"size": 3, "color": COLORS[ds], "opacity": 0.35},
                           "name": f"{ds}: r={r:.2f}, n={ok.sum()}"})
        traces.append({"x": lim, "y": lim, "type": "scatter", "mode": "lines", "showlegend": False,
                       "line": {"color": "#888", "dash": "dash", "width": 1}, "hoverinfo": "skip"})
        panels.append({"id": kx + ky, "data": traces,
                       "layout": {"title": title, "xaxis": {"title": xl, "range": lim},
                                  "yaxis": {"title": "daughter", "range": lim, "scaleanchor": "x"}}})
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Daughter vs mother</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>body{{font-family:Arial,sans-serif;margin:16px;background:#fff;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}}
.p{{height:440px}} p{{font-size:13px;max-width:1100px}}</style></head><body>
<h2>261024 vs 260517: daughters followed over their whole generation vs their mother</h2>
<p>Source: <code>{Path(a.npz).name}</code>. Daughter cycle = birth at a QC-validated mother division to the
daughter's own first QC-validated division (daughters that leave before dividing are dropped). Sister = the mother's
cycle starting at the same division; previous = the mother's cycle that ended at it. Drag to zoom, double-click to
reset, click a legend entry to hide it.</p>
<div class="grid">{''.join(f'<div class="p" id="{p["id"]}"></div>' for p in panels)}</div>
<script>
const P = {json.dumps(panels)};
for (const p of P) {{
  p.layout.margin = {{l:60,r:10,t:40,b:50}}; p.layout.legend = {{x:0.02,y:0.98}};
  Plotly.newPlot(p.id, p.data, p.layout, {{responsive:true}});
}}
</script></body></html>"""
    Path(a.out).write_text(html, encoding="utf-8")
    print(a.out, f"{Path(a.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
