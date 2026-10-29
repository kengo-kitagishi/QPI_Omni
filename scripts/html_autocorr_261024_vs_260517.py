"""Interactive HTML of fig_autocorr_261024_vs_260517 (zoom / hover), built from its _data.npz.

    python scripts/html_autocorr_261024_vs_260517.py <path to ..._data.npz> [--out page.html]
"""
import argparse
import json
from pathlib import Path

import numpy as np

COLORS = {"261024": "#0072B2", "260517": "#D55E00"}
PANELS = [("rho", "a  cycle-mean concentration", "generation lag k"),
          ("rhob", "b  birth concentration", "generation lag k"),
          ("Mb", "c  birth mass", "generation lag k"),
          ("Vb", "d  birth volume", "generation lag k"),
          ("Vmean", "e  cycle-mean volume", "generation lag k"),
          ("dMdt", "f  mass production rate (pg/h)", "generation lag k"),
          ("T_h", "g  interdivision time", "generation lag k"),
          ("conc_time", "h  mother concentration (time)", "time lag τ [h]")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=r"D:\261024_seg\_qc\autocorr_261024_vs_260517.html")
    a = ap.parse_args()
    z = np.load(a.npz)
    panels = []
    for key, title, xlab in PANELS:
        traces = []
        for ds in ("260517", "261024"):
            c = z[f"{ds}_{key}_acf"]
            x = z["tau_h"] if key == "conc_time" else np.arange(len(c))
            n = z[f"{ds}_{key}_npairs"]
            r3 = lambda v: np.round(np.asarray(v, float), 4).tolist()  # noqa: E731
            lo, hi = r3(z[f"{ds}_{key}_lo"]), r3(z[f"{ds}_{key}_hi"])
            traces.append({"x": r3(x), "y": r3(c), "type": "scatter",
                           "mode": "lines" if key == "conc_time" else "lines+markers",
                           "line": {"color": COLORS[ds], "width": 2}, "marker": {"size": 5},
                           "name": ds,
                           "customdata": [[int(a), b, c_] for a, b, c_ in zip(n, lo, hi)],
                           "hovertemplate": "lag %{x}<br>C = %{y:.3f}<br>95% CI %{customdata[1]:.3f} – "
                                            "%{customdata[2]:.3f}<br>pairs %{customdata[0]}<extra>" + ds + "</extra>"})
        panels.append({"id": key, "data": traces,
                       "layout": {"title": title, "xaxis": {"title": xlab},
                                  "yaxis": {"title": "autocorrelation", "range": [-0.3, 1.02], "zeroline": True}}})
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Autocorrelation 261024</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>body{{font-family:Arial,sans-serif;margin:16px;background:#fff;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}}
.p{{height:400px}} p{{font-size:13px;max-width:1100px}}</style></head><body>
<h2>261024 vs 260517 (2% EMM): autocorrelation of mother-cell quantities</h2>
<p>Source: <code>{Path(a.npz).name}</code>. C = pooled covariance of deviations from the dataset mean at
the lag / pooled variance. Hover shows the 95% bootstrap CI over mothers and the number of pairs at each lag.
Drag to zoom, double-click to reset.</p>
<div class="grid">{''.join(f'<div class="p" id="{p["id"]}"></div>' for p in panels)}</div>
<script>
const P = {json.dumps(panels)};
for (const p of P) {{
  p.layout.margin = {{l:60,r:10,t:40,b:50}}; p.layout.legend = {{x:0.75,y:0.98}};
  Plotly.newPlot(p.id, p.data, p.layout, {{responsive:true}});
}}
</script></body></html>"""
    Path(a.out).write_text(html, encoding="utf-8")
    print(a.out, f"{Path(a.out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
