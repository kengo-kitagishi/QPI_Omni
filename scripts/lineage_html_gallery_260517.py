"""lineage_html_gallery_260517.py - HTML gallery: one 3-row plot per mother lineage.

Rows (shared time axis, img_0002 = 0 h, 5 min/frame):
  1. mean RI
  2. phase-integral dry mass  mass_pg = total_phase * lambda_um * px_um^2 / (2 pi alpha) * 1e-3
  3. volume (volume_um3_efd when present, else volume_um3_profile)

Markers: valid frames dark blue (line + dots); tracker outlier red x; border orange triangle;
drift-excluded rank-1 measurements purple diamond; retained (QC-validated) mother divisions
grey vertical lines (#7F8C8D, lw 0.45, alpha 0.55). No 20 h grid lines. The y ranges of each
row are shared by every lineage. In every complete cycle between two retained divisions,
ln(mass) is fitted linearly against time on valid frames only and the back-transformed fit
is drawn in red.

Data: a published master (LATEST by default) -> per_channel/PosN/chNN/. Division validation
uses division_qc_260517.qc_channel on the fly (yellow columns when present, otherwise
mass_pg / volume_um3_profile).

Usage:
    python scripts/lineage_html_gallery_260517.py                 # LATEST master, Pos<=52, 30 lineages
    python scripts/lineage_html_gallery_260517.py --max-lineages 60 --pos-max 104
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import qpi_paths as qp  # noqa: E402
import division_qc_260517 as dqc  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WAVELENGTH_UM = 0.658
PIXEL_UM = 0.34567514677103717
ALPHA_ML_PER_MG = 0.00018
DT_MIN = 5.0
T0_FRAME = 2
END_FRAME = 2017
MIN_FIT_POINTS = 5
EDGE_CHANNELS = ("ch00", "ch11")   # edge traps excluded from the analysis (2026-09-14)

C_VALID = "#1B3A6B"
C_OUT = "#D62728"
C_BORDER = "#E69F00"
C_DRIFT = "#7B3FA0"
C_DIV = "#7F8C8D"
C_FIT = "#E41A1C"


def t_h(frame) -> np.ndarray:
    return (np.asarray(frame, dtype=float) - T0_FRAME) * DT_MIN / 60.0


def phase_mass_pg(total_phase) -> np.ndarray:
    tp = np.asarray(total_phase, dtype=float)
    return tp * WAVELENGTH_UM * PIXEL_UM ** 2 / (2.0 * np.pi * ALPHA_ML_PER_MG) * 1e-3


def load_lineage_dir(d: Path):
    df = pd.read_csv(d / "lineage_data3D.csv")
    bad = pd.read_csv(d / "lineage_bad_frames.csv") if (d / "lineage_bad_frames.csv").exists() else pd.DataFrame()
    return df, bad


def load_lineage(master: Path, pos: str, ch: str):
    return load_lineage_dir(master / "per_channel" / pos / ch)


def working_tree_lineages(pos_max: int, flags: pd.DataFrame | None):
    """(pos, ch, lineage_out) for every yellow-tracker production channel in D:\\260517_seg."""
    import _retrack_260517_newmodel as chain
    out = []
    for pos_dir in sorted(chain.MASK_ROOT.glob("Pos*"), key=lambda p: int(p.name[3:])):
        n = int(pos_dir.name[3:])
        if n > pos_max:
            continue
        z = pos_dir / chain.REL
        if not z.is_dir():
            continue
        for ch in sorted(p for p in z.iterdir() if p.is_dir() and p.name.startswith("ch")):
            if ch.name in EDGE_CHANNELS:
                continue
            lo = ch / "inference_out" / "lineage_out"
            if not chain.is_production(lo):
                continue
            if flags is not None:
                f = flags[(flags["pos"] == pos_dir.name) & (flags["ch"] == ch.name)]
                if len(f) and (f.iloc[0]["classification_status"] not in ("cells", None, np.nan)
                               or bool(f.iloc[0]["qc_oob_excluded"])):
                    continue
            out.append((pos_dir.name, ch.name, lo))
    return out


def prepare(df: pd.DataFrame, bad: pd.DataFrame) -> dict:
    vol_col = "volume_um3_efd" if "volume_um3_efd" in df.columns else "volume_um3_profile"
    mass_col = "mass_pg_efd" if "mass_pg_efd" in df.columns else "mass_pg"
    m = df[df["cell_id"] == 0].sort_values("frame")
    m = m[(m["frame"] >= T0_FRAME) & (m["frame"] <= END_FRAME)]
    t = t_h(m["frame"])
    outl = m["is_outlier"].astype(bool).to_numpy()
    bord = m["touches_border"].astype(bool).to_numpy()
    valid = ~outl & ~bord
    ri = m["mean_ri"].to_numpy(dtype=float)
    mass = phase_mass_pg(m["total_phase"])
    vol = m[vol_col].to_numpy(dtype=float)
    # hidden rows carry NaN physics in the table; for the markers we still need x positions
    d = dict(t=t, ri=ri, mass=mass, vol=vol, valid=valid, outl=outl, bord=bord, frame=m["frame"].to_numpy())
    # drift-excluded rank-1 measurements
    if len(bad) and "rank_in_frame" in bad.columns:
        b = bad[(bad["rank_in_frame"] == 1) & (bad["frame"] >= T0_FRAME) & (bad["frame"] <= END_FRAME)]
        d["bad_t"] = t_h(b["frame"])
        d["bad_mass"] = phase_mass_pg(b["total_phase"])
        d["bad_vol"] = b["volume_um3_rod"].to_numpy(dtype=float)   # bad table has the rod volume only
        d["bad_ri"] = b["mean_ri"].to_numpy(dtype=float)           # NaN by design (drift-uncorrected)
    else:
        d["bad_t"] = np.array([]); d["bad_mass"] = np.array([]); d["bad_vol"] = np.array([]); d["bad_ri"] = np.array([])
    # QC-validated mother divisions (computed on the full channel so +-8 frame windows are complete)
    qc = dqc.qc_channel(df, dt_min=DT_MIN, frame_min=T0_FRAME, mass_col=mass_col, vol_col=vol_col)
    md = qc[qc["is_mother_division"]]
    d["div_all"] = np.sort(md["frame"].to_numpy(dtype=int))
    d["div_ok"] = np.sort(md[md["validated"]]["frame"].to_numpy(dtype=int))
    d["div_ok"] = d["div_ok"][(d["div_ok"] >= T0_FRAME) & (d["div_ok"] <= END_FRAME)]
    # cycle fits: ln(mass) ~ t on valid frames of each complete cycle
    fits = []
    dv = d["div_ok"]
    for a, b_ in zip(dv[:-1], dv[1:]):
        sel = valid & (d["frame"] >= a) & (d["frame"] < b_) & np.isfinite(mass) & (mass > 0)
        if sel.sum() < MIN_FIT_POINTS:
            continue
        x, y = t[sel], np.log(mass[sel])
        slope, icpt = np.polyfit(x, y, 1)
        yhat = slope * x + icpt
        ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
        fits.append(dict(start_frame=int(a), end_frame=int(b_), t_start=float(t_h(a)), t_end=float(t_h(b_)),
                         n_points=int(sel.sum()), slope_per_h=float(slope), intercept=float(icpt),
                         doubling_time_h=float(np.log(2) / slope) if slope > 0 else np.nan,
                         r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan))
    d["fits"] = fits
    d["vol_col"] = vol_col
    d["mass_col_qc"] = mass_col
    return d


def render(pos: str, ch: str, d: dict, ylims: dict, dpi: int = 110) -> bytes:
    fig, axes = plt.subplots(3, 1, figsize=(15, 6.2), sharex=True, dpi=dpi)
    series = [("ri", "mean RI", d["ri"], d["bad_ri"]), ("mass", "phase-integral mass [pg]", d["mass"], d["bad_mass"]),
              ("vol", f"volume [um$^3$] ({d['vol_col']})", d["vol"], d["bad_vol"])]
    t, valid, outl, bord = d["t"], d["valid"], d["outl"], d["bord"]
    for ax, (key, label, y, ybad) in zip(axes, series):
        for f in d["div_ok"]:
            ax.axvline(t_h(f), color=C_DIV, lw=0.45, alpha=0.55, zorder=1)
        yv = np.where(valid, y, np.nan)
        ax.plot(t, yv, "-", color=C_VALID, lw=0.7, alpha=0.9, zorder=3)
        ax.plot(t[valid], y[valid], ".", color=C_VALID, ms=2.2, zorder=4)
        lo, hi = ylims[key]
        # hidden rows have NaN values: draw their markers at the bottom edge of the panel
        yo = np.where(np.isfinite(y), y, lo + 0.03 * (hi - lo))
        if outl.any():
            ax.plot(t[outl], yo[outl], "x", color=C_OUT, ms=4, mew=0.8, zorder=5, label="tracker outlier")
        if bord.any():
            ax.plot(t[bord], yo[bord], "^", color=C_BORDER, ms=4, mew=0, zorder=5, label="border")
        if len(d["bad_t"]):
            yb = np.where(np.isfinite(ybad), ybad, lo + 0.03 * (hi - lo))
            ax.plot(d["bad_t"], yb, "D", color=C_DRIFT, ms=3.2, mew=0, zorder=5, label="drift-excluded rank-1")
        if key == "mass":
            for fz in d["fits"]:
                xx = np.linspace(fz["t_start"], fz["t_end"], 30)
                ax.plot(xx, np.exp(fz["intercept"] + fz["slope_per_h"] * xx), "-", color=C_FIT, lw=1.1, zorder=6)
        ax.set_ylim(lo, hi)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(False)
    axes[-1].set_xlim(0, t_h(END_FRAME))
    axes[-1].set_xlabel("time [h]  (img_0002 = 0 h, 5 min/frame)", fontsize=9)
    nfit = len(d["fits"])
    axes[0].set_title(f"{pos} {ch}  |  retained mother divisions: {len(d['div_ok'])} of {len(d['div_all'])} candidates  |  "
                      f"fitted cycles: {nfit}  |  median doubling time "
                      f"{np.nanmedian([f['doubling_time_h'] for f in d['fits']]) if nfit else float('nan'):.2f} h",
                      fontsize=10, loc="left")
    fig.tight_layout(h_pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master-tag", default=None)
    ap.add_argument("--pos-max", type=int, default=52, help="skip positions above this (tilt-biased in v20260911)")
    ap.add_argument("--max-lineages", type=int, default=30)
    ap.add_argument("--min-coverage", type=float, default=0.98)
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", choices=["master", "working-tree"], default="master",
                    help="working-tree = yellow-tracker lineages in D:\\260517_seg before the master is published")
    ap.add_argument("--ylim-ri", default="1.37,1.40", help="mean RI axis, lo,hi (or auto)")
    ap.add_argument("--ylim-mass", default="0,50", help="mass axis [pg], lo,hi (or auto)")
    ap.add_argument("--ylim-vol", default="0,120", help="volume axis [um3], lo,hi (or auto)")
    ap.add_argument("--min-mother-frames", type=int, default=1000,
                    help="working-tree mode: skip channels whose mother has fewer rows in the window")
    args = ap.parse_args()
    if args.master_tag:
        os.environ["QPI_LINEAGE_MASTER"] = args.master_tag
    master = qp.master_dir()
    if master is None:
        raise SystemExit("no master published")
    chans = pd.read_csv(master / "derived" / "phase1_img0002-2017" / "channels.csv")
    chans["pos_num"] = chans["pos"].str[3:].astype(int)

    data = {}
    if args.source == "master":
        sel = chans[(chans["classification_status"] == "cells") & chans["mother_present"]
                    & (~chans["qc_oob_excluded"].astype(bool)) & (~chans["ch"].isin(EDGE_CHANNELS))
                    & (chans["mother_coverage"] >= args.min_coverage)
                    & (chans["pos_num"] <= args.pos_max)].sort_values(["pos_num", "ch"])
        sel = sel.head(args.max_lineages)
        print(f"master {master.name}: {len(sel)} lineages selected")
        for r in sel.itertuples(index=False):
            df, bad = load_lineage(master, r.pos, r.ch)
            data[(r.pos, r.ch)] = prepare(df, bad)
        source_name = master.name
    else:
        cands = working_tree_lineages(args.pos_max, chans)
        print(f"working tree: {len(cands)} yellow-tracker channels found (classification/OOB flags from {master.name})")
        for pos, ch, lo in cands:
            if len(data) >= args.max_lineages:
                break
            df, bad = load_lineage_dir(lo)
            m = df[(df["cell_id"] == 0) & (df["frame"] >= T0_FRAME) & (df["frame"] <= END_FRAME)]
            if len(m) < args.min_mother_frames:
                continue
            data[(pos, ch)] = prepare(df, bad)
        source_name = "working-tree_yellow"
        if not data:
            raise SystemExit("no yellow-tracker lineages with a mother yet")
        print(f"{len(data)} lineages selected")
    # shared y ranges (1st..99th percentile of valid values, padded)
    def _lim(key, pad=0.06):
        vals = np.concatenate([d[key][d["valid"]] for d in data.values()])
        vals = vals[np.isfinite(vals)]
        lo, hi = np.percentile(vals, [0.5, 99.5])
        span = hi - lo
        return (lo - pad * span, hi + pad * span)
    def _parse(spec, auto):
        if str(spec).lower() == "auto":
            return auto()
        lo, hi = (float(x) for x in str(spec).split(","))
        return (lo, hi)
    ylims = {"ri": _parse(args.ylim_ri, lambda: _lim("ri")),
             "mass": _parse(args.ylim_mass, lambda: (0, _lim("mass")[1])),
             "vol": _parse(args.ylim_vol, lambda: (0, _lim("vol")[1]))}

    out_dir = Path(args.out) if args.out else (SCRIPTS.parent / "results" / "figures" / "lineage_html")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    html_path = out_dir / f"lineage_gallery_{source_name}_{stamp}.html"
    fit_rows, parts = [], []
    for (pos, ch), d in data.items():
        png = render(pos, ch, d, ylims)
        (out_dir / f"{pos}_{ch}_{stamp}.png").write_bytes(png)
        b64 = base64.b64encode(png).decode("ascii")
        for fz in d["fits"]:
            fit_rows.append(dict(pos=pos, ch=ch, **fz))
        parts.append(f"<section id='{pos}_{ch}'><h2>{pos} {ch}</h2>"
                     f"<img src='data:image/png;base64,{b64}' alt='{pos} {ch}'></section>")
    pd.DataFrame(fit_rows).to_csv(out_dir / f"cycle_fits_{source_name}_{stamp}.csv", index=False)

    toc = " | ".join(f"<a href='#{p}_{c}'>{p} {c}</a>" for (p, c) in data)
    style = ("body{font-family:Arial,Helvetica,sans-serif;margin:16px;background:#fff;color:#222}"
             "img{width:100%;max-width:1500px;display:block;border:1px solid #ddd}"
             "h2{font-size:15px;margin:22px 0 4px}section{margin-bottom:6px}"
             ".note{font-size:13px;line-height:1.5;max-width:1100px}.toc{font-size:12px;line-height:1.8}"
             ".sw{display:inline-block;width:10px;height:10px;margin-right:4px;vertical-align:middle}")
    legend = (f"<span class='sw' style='background:{C_VALID}'></span>valid frame &nbsp; "
              f"<span style='color:{C_OUT};font-weight:bold'>&times;</span> tracker outlier &nbsp; "
              f"<span style='color:{C_BORDER}'>&#9650;</span> border &nbsp; "
              f"<span style='color:{C_DRIFT}'>&#9670;</span> drift-excluded rank-1 &nbsp; "
              f"<span style='color:{C_DIV}'>|</span> retained mother division &nbsp; "
              f"<span style='color:{C_FIT}'>&mdash;</span> ln(mass) linear fit per complete cycle (back-transformed)")
    note = (f"<p class='note'><b>Data:</b> <code>{source_name}</code> (yellow-contour tracker where the volume column is volume_um3_efd), mother cell "
            f"(cell_id 0), window img_{T0_FRAME:04d}-img_{END_FRAME:04d} (0-{t_h(END_FRAME):.1f} h), 5 min/frame.<br>"
            f"<b>Mass:</b> phase-integral, mass_pg = total_phase &times; {WAVELENGTH_UM} &micro;m &times; ({PIXEL_UM:.5f} &micro;m)&sup2; "
            f"/ (2&pi; &times; {ALPHA_ML_PER_MG} mL/mg) &times; 1e-3. <b>Volume:</b> {next(iter(data.values()))['vol_col']}.<br>"
            f"<b>Retained divisions:</b> division_qc_260517 rules (direct if no tracker outlier within &plusmn;1 frame; else medians of up to 3 valid "
            f"points within &plusmn;8 frames, post/pre mass 0.25-0.78, volume 0.25-0.85, |diff| &le; 0.25; duplicates within 1 h collapsed), "
            f"QC columns {next(iter(data.values()))['mass_col_qc']} / {next(iter(data.values()))['vol_col']}.<br>"
            f"<b>Fits:</b> complete cycles between two retained divisions, valid frames only (no outlier / border), &ge; {MIN_FIT_POINTS} points. "
            f"Shared y ranges (fixed): RI {ylims['ri'][0]:.3f}-{ylims['ri'][1]:.3f}, mass {ylims['mass'][0]:.0f}-{ylims['mass'][1]:.0f} pg, volume {ylims['vol'][0]:.0f}-{ylims['vol'][1]:.0f} &micro;m&sup3;; values outside are clipped.<br>"
            f"Selection: classification cells, mother present, not OOB, not an edge trap (ch00/ch11), coverage &ge; {args.min_coverage}, Pos &le; {args.pos_max}, first {args.max_lineages}.</p>")
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>260517 lineage gallery {master.name}</title>"
            f"<style>{style}</style></head><body><h1 style='font-size:18px'>260517 mother lineages: RI / phase-integral mass / volume</h1>"
            f"{note}<p class='note'>{legend}</p><p class='toc'>{toc}</p>{''.join(parts)}</body></html>")
    html_path.write_text(html, encoding="utf-8")
    print("html:", html_path)
    print("fits csv:", out_dir / f"cycle_fits_{source_name}_{stamp}.csv", "| cycles fitted:", len(fit_rows))
    (out_dir / f"params_{stamp}.json").write_text(json.dumps(dict(
        source=source_name, wavelength_um=WAVELENGTH_UM, pixel_um=PIXEL_UM, alpha_ml_per_mg=ALPHA_ML_PER_MG,
        dt_min=DT_MIN, t0_frame=T0_FRAME, end_frame=END_FRAME, ylims=ylims, n_lineages=len(data)), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
