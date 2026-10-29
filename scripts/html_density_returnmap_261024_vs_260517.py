"""Interactive HTML of fig_density_returnmap_261024_vs_260517 (zoom / hover), built from its _data.npz.

    python scripts/html_density_returnmap_261024_vs_260517.py <path to ..._data.npz> [--out page.html]

Histogram counts are binned here (5 mg/mL); return-map points are embedded as-is and drawn
with plotly.js (WebGL) from a CDN.
"""
import argparse
import json
from pathlib import Path

import numpy as np

COLORS = {"261024": "#0072B2", "260517": "#D55E00"}
PANELS = [("rho", "cycle-mean concentration ρ [mg/mL]", [200, 380]),
          ("Mb", "birth mass M_b [pg]", [0, 45]),
          ("T_h", "interdivision time T [h]", [0, 8])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=r"D:\261024_seg\_qc\density_returnmap_261024_vs_260517.html")
    a = ap.parse_args()
    z = np.load(a.npz)
    bins = np.arange(150, 451, 5)
    hist, maps = [], {k: [] for k, _, _ in PANELS}
    for ds in ("260517", "261024"):
        v = z[f"{ds}_conc_valid"]
        h, _ = np.histogram(v, bins=bins, density=True)
        hist.append({"x": ((bins[:-1] + bins[1:]) / 2).tolist(), "y": h.round(6).tolist(),
                     "type": "scatter", "mode": "lines", "line": {"shape": "hvh", "color": COLORS[ds]},
                     "name": f"{ds} (median {np.median(v):.0f} mg/mL, {len(v)} frames)"})
        for key, _, _ in PANELS:
            x, y = z[f"{ds}_{key}_n"], z[f"{ds}_{key}_n1"]
            ok = np.isfinite(x) & np.isfinite(y)
            r = np.corrcoef(x[ok], y[ok])[0, 1]
            maps[key].append({"x": x[ok].round(3).tolist(), "y": y[ok].round(3).tolist(),
                              "type": "scattergl", "mode": "markers",
                              "marker": {"size": 3, "color": COLORS[ds], "opacity": 0.35},
                              "name": f"{ds}: r={r:.2f}, n={ok.sum()}"})
    panels = [{"id": "hist", "data": hist,
               "layout": {"title": "a  mother dry-mass concentration",
                          "xaxis": {"title": "dry-mass concentration [mg/mL]"},
                          "yaxis": {"title": "probability density [1/(mg/mL)]"}}}]
    for (key, lab, lim), letter in zip(PANELS, "bcd"):
        diag = {"x": lim, "y": lim, "type": "scatter", "mode": "lines", "showlegend": False,
                "line": {"color": "#888", "dash": "dash", "width": 1}, "hoverinfo": "skip"}
        panels.append({"id": key, "data": [*maps[key], diag],
                       "layout": {"title": f"{letter}  return map: {lab.split(' [')[0]}",
                                  "xaxis": {"title": f"{lab} (cycle n)", "range": lim},
                                  "yaxis": {"title": f"{lab} (cycle n+1)", "range": lim,
                                            "scaleanchor": "x"}}})
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Density return maps</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>body{{font-family:Arial,sans-serif;margin:16px;background:#fff;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}}
.p{{height:440px}} p{{font-size:13px;max-width:1100px}}</style></head><body>
<h2>261024 vs 260517 (2% EMM): mother dry-mass concentration and return maps</h2>
<p>Source: <code>{Path(a.npz).name}</code>. Mother = cell 0; cycle = between two QC-validated mother divisions;
concentration = efd density x 1000; birth mass = median of the first 3 valid frames after division.
261024 n_medium 1.335029 is carried over from 260517. Drag to zoom, double-click to reset, click a legend entry to hide it.</p>
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
