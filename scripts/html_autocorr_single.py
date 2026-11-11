"""Interactive HTML (zoom / hover) of a single-dataset autocorrelation figure, built from its _data.npz.

For fig_autocorr_261030.py / fig_autocorr_261108.py, whose npz keys are <key>_acf / <key>_sem / <key>_n_mothers.
Points = mean over mothers of the per-mother C_i(k), error bars = SEM over mothers (no bootstrap).

    python scripts/html_autocorr_single.py <..._data.npz> --label "261108 YE run 3" --out page.html
"""
import argparse
import json
from pathlib import Path

import numpy as np

PANELS = [("rho", "a  cycle-mean concentration", "generation lag k"),
          ("rhob", "b  birth concentration", "generation lag k"),
          ("Mb", "c  birth mass", "generation lag k"),
          ("Vb", "d  birth volume", "generation lag k"),
          ("Vmean", "e  cycle-mean volume", "generation lag k"),
          ("dMdt", "f  mass production rate (pg/h)", "generation lag k"),
          ("T_h", "g  interdivision time", "generation lag k"),
          ("conc_time", "h  mother concentration (time, living)", "time lag τ [h]")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--label", required=True)
    ap.add_argument("--color", default="#D55E00")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.npz)
    r3 = lambda v: np.round(np.asarray(v, float), 4).tolist()  # noqa: E731
    panels = []
    for key, title, xlab in PANELS:
        c, sem, n = z[f"{key}_acf"], z[f"{key}_sem"], z[f"{key}_n_mothers"]
        x = z["tau_h"] if key == "conc_time" else np.arange(len(c))
        tr = {"x": r3(x), "y": r3(c), "type": "scatter", "name": a.label,
              "mode": "lines" if key == "conc_time" else "lines+markers",
              "line": {"color": a.color, "width": 2}, "marker": {"size": 5},
              "customdata": [[int(m), s] for m, s in zip(n, r3(sem))],
              "hovertemplate": "lag %{x}<br>C = %{y:.3f} ± %{customdata[1]:.3f} (SEM)<br>mothers %{customdata[0]}"
                               "<extra></extra>"}
        data = [tr]
        if key == "conc_time":   # band instead of a bar at every 5-min lag
            lo, hi = r3(c - sem), r3(c + sem)
            data = [{"x": r3(x) + r3(x)[::-1], "y": hi + lo[::-1], "fill": "toself", "type": "scatter",
                     "mode": "none", "fillcolor": "rgba(213,94,0,0.25)", "hoverinfo": "skip", "showlegend": False}, tr]
        else:
            tr["error_y"] = {"type": "data", "array": r3(sem), "visible": True, "thickness": 1}
        lo_y = min(-0.3, float(np.nanmin(c - sem)) - 0.05)
        panels.append({"id": key, "data": data,
                       "layout": {"title": title, "xaxis": {"title": xlab}, "showlegend": False,
                                  "yaxis": {"title": "autocorrelation", "range": [lo_y, 1.02], "zeroline": True}}})
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Autocorrelation {a.label}</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>body{{font-family:Arial,sans-serif;margin:16px;background:#fff;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}}
.p{{height:400px}} p{{font-size:13px;max-width:1100px}}</style></head><body>
<h2>{a.label}: autocorrelation of mother-cell quantities</h2>
<p>Source: <code>{Path(a.npz).name}</code>. For each mother, C_i(k) = mean product of deviations from the dataset
mean at the lag / that mother's mean squared deviation; points = mean over mothers, error bars (h: band) = SEM over
mothers. Hover shows the number of mothers at each lag. Drag to zoom, double-click to reset.</p>
<div class="grid">{''.join(f'<div class="p" id="{p["id"]}"></div>' for p in panels)}</div>
<script>
const P = {json.dumps(panels)};
for (const p of P) {{ p.layout.margin = {{l:60,r:10,t:40,b:50}}; Plotly.newPlot(p.id, p.data, p.layout, {{responsive:true}}); }}
</script></body></html>"""
    Path(a.out).write_text(html, encoding="utf-8")
    print(a.out, f"{Path(a.out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
