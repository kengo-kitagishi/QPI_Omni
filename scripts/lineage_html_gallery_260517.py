"""lineage_html_gallery_260517.py - HTML gallery: one 3-row plot per mother lineage.

Rows (shared time axis, img_0002 = 0 h, 5 min/frame):
  1. mean RI or dry-mass concentration [mg/mL]
  2. tracker dry mass (mass_pg, computed using EFD volume and recorded RI basis)
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
    python scripts/lineage_html_gallery_260517.py                 # LATEST master, through img_3747
    python scripts/lineage_html_gallery_260517.py --max-lineages 1200 --pos-max 104 --panel1 conc
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
END_FRAME = 3747
MIN_FIT_POINTS = 5
# fixed axes horizontal extent in figure fraction, so a click x on the PNG maps to a time
# (data x-range 0..t_h(END_FRAME) spans figure fraction PLOT_LEFT..PLOT_RIGHT).
PLOT_LEFT = 0.060
PLOT_RIGHT = 0.996

C_VALID = "#1B3A6B"
C_OUT = "#D62728"
C_BORDER = "#E69F00"
C_DRIFT = "#7B3FA0"
C_DIV = "#7F8C8D"
C_FIT = "#E41A1C"


def t_h(frame) -> np.ndarray:
    return (np.asarray(frame, dtype=float) - T0_FRAME) * DT_MIN / 60.0


def load_lineage_dir(d: Path):
    df = pd.read_csv(d / "lineage_data3D.csv")
    bad = pd.read_csv(d / "lineage_bad_frames.csv") if (d / "lineage_bad_frames.csv").exists() else pd.DataFrame()
    return df, bad


def load_lineage(master: Path, pos: str, ch: str):
    return load_lineage_dir(master / "per_channel" / pos / ch)


def flag_is_true(value) -> bool:
    return False if pd.isna(value) else str(value).strip().lower() in {"true", "1", "yes"}


def working_tree_lineages(pos_max: int, flags: pd.DataFrame | None, pos_min: int = 1):
    """(pos, ch, lineage_out) for every yellow-tracker production channel in D:\\260517_seg."""
    import _retrack_260517_newmodel as chain
    out = []
    for pos_dir in sorted(chain.MASK_ROOT.glob("Pos*"), key=lambda p: int(p.name[3:])):
        n = int(pos_dir.name[3:])
        if n < pos_min or n > pos_max:
            continue
        z = pos_dir / chain.REL
        if not z.is_dir():
            continue
        for ch in sorted(p for p in z.iterdir() if p.is_dir() and p.name.startswith("ch")):
            lo = ch / "inference_out" / "lineage_out"
            if not chain.is_production(lo):
                continue
            out.append((pos_dir.name, ch.name, lo))
    return out


def prepare(df: pd.DataFrame, bad: pd.DataFrame) -> dict:
    vol_col = "volume_um3_efd" if "volume_um3_efd" in df.columns else "volume_um3_profile"
    mass_col = qp.efd_col(df.columns, "mass_pg")
    ri_col = qp.efd_col(df.columns, "mean_ri")
    dens_col = qp.efd_col(df.columns, "density_pg_um3")
    m = df[df["cell_id"] == 0].sort_values("frame")
    m = m[(m["frame"] >= T0_FRAME) & (m["frame"] <= END_FRAME)]
    t = t_h(m["frame"])
    outl = m["is_outlier"].astype(bool).to_numpy()
    bord = m["touches_border"].astype(bool).to_numpy()
    valid = ~outl & ~bord
    ri = m[ri_col].to_numpy(dtype=float)
    mass = m[mass_col].to_numpy(dtype=float)
    vol = m[vol_col].to_numpy(dtype=float)
    # hidden rows carry NaN physics in the table; for the markers we still need x positions
    d = dict(t=t, ri=ri, mass=mass, vol=vol, valid=valid, outl=outl, bord=bord, frame=m["frame"].to_numpy())
    # dry-mass concentration [mg/mL] = density_pg_um3 (efd) * 1000 (pg/um^3 = g/mL); = (n_cell - n_medium)/alpha,
    # i.e. medium-RI removed, so it is comparable across the media switches.
    d["conc"] = (m[dens_col].to_numpy(dtype=float) * 1000.0
                 if dens_col in m.columns else np.full(len(m), np.nan))
    # drift-excluded rank-1 measurements
    if len(bad) and "rank_in_frame" in bad.columns:
        b = bad[(bad["rank_in_frame"] == 1) & (bad["frame"] >= T0_FRAME) & (bad["frame"] <= END_FRAME)]
        d["bad_t"] = t_h(b["frame"])
        d["bad_mass"] = np.full(len(b), np.nan)
        d["bad_vol"] = (b["volume_um3_efd"].to_numpy(dtype=float) if "volume_um3_efd" in b.columns
                        else np.full(len(b), np.nan))
        d["bad_ri"] = b["mean_ri"].to_numpy(dtype=float)           # NaN by design (drift-uncorrected)
        d["bad_conc"] = np.full(len(b), np.nan)                     # drift rows carry no valid concentration
    else:
        d["bad_t"] = np.array([]); d["bad_mass"] = np.array([]); d["bad_vol"] = np.array([])
        d["bad_ri"] = np.array([]); d["bad_conc"] = np.array([])
    # QC-validated mother divisions (computed on the full channel so +-8 frame windows are complete)
    qc = dqc.qc_channel(df, dt_min=DT_MIN, frame_min=T0_FRAME, mass_col=mass_col, vol_col=vol_col)
    md = qc[qc["is_mother_division"]] if len(qc) and "is_mother_division" in qc.columns else qc.iloc[0:0]
    if len(md) and "frame" in md.columns:  # a mother with zero detected divisions yields an empty qc table
        d["div_all"] = np.sort(md["frame"].to_numpy(dtype=int))
        dvok = md[md["validated"]]["frame"].to_numpy(dtype=int)
    else:
        d["div_all"] = np.array([], dtype=int)
        dvok = np.array([], dtype=int)
    d["div_ok"] = np.sort(dvok)
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


def render(pos: str, ch: str, d: dict, ylims: dict, dpi: int = 110, panel1: str = "ri") -> bytes:
    fig, axes = plt.subplots(3, 1, figsize=(15, 6.2), sharex=True, dpi=dpi)
    s0 = (("conc", "dry-mass conc. [mg/mL]", d["conc"], d.get("bad_conc", np.array([])))
          if panel1 == "conc" else ("ri", "mean RI", d["ri"], d["bad_ri"]))
    series = [s0, ("mass", "dry mass [pg]", d["mass"], d["bad_mass"]),
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
    # fixed geometry (not tight_layout) so the plot area maps deterministically to click coordinates
    fig.subplots_adjust(left=PLOT_LEFT, right=PLOT_RIGHT, top=0.93, bottom=0.085, hspace=0.13)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def main() -> None:
    global END_FRAME
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master-tag", default=None)
    ap.add_argument("--pos-max", type=int, default=104)
    ap.add_argument("--pos-min", type=int, default=1, help="skip positions below this")
    ap.add_argument("--max-lineages", type=int, default=30)
    ap.add_argument("--end-frame", type=int, default=END_FRAME,
                    help=f"last frame to include, inclusive (default: {END_FRAME})")
    ap.add_argument("--out", default=None)
    ap.add_argument("--source", choices=["master", "working-tree", "csv"], default="master",
                    help="master / working-tree (D:\\260517_seg) / csv (a consolidated all-cells CSV you already have)")
    ap.add_argument("--csv", default=None,
                    help="--source csv: path to a consolidated all_cells_lineage_data3D.csv(.gz) with pos,ch,cell_id,frame,... columns")
    ap.add_argument("--bad-csv", default=None,
                    help="--source csv: optional all_cells_lineage_bad_frames.csv(.gz) for the drift-excluded markers")
    ap.add_argument("--channels-csv", default=None,
                    help="--source csv: optional channels.csv (classification_status/qc_oob_excluded flags)")
    ap.add_argument("--panel1", choices=["ri", "conc"], default="ri",
                    help="top panel: ri = mean RI, conc = dry-mass concentration [mg/mL] (efd density_pg_um3*1000)")
    ap.add_argument("--ylim-ri", default="auto", help="mean RI axis, lo,hi (or auto)")
    ap.add_argument("--ylim-conc", default="auto", help="concentration axis [mg/mL], lo,hi (or auto)")
    ap.add_argument("--ylim-mass", default="auto", help="mass axis [pg], lo,hi (or auto)")
    ap.add_argument("--ylim-vol", default="auto", help="volume axis [um3], lo,hi (or auto)")
    ap.add_argument("--min-mother-frames", type=int, default=1,
                    help="minimum tracked mother frames in the selected window (default: 1; keeps partial/death tracks)")
    args = ap.parse_args()
    if args.end_frame < T0_FRAME:
        raise SystemExit(f"--end-frame must be >= {T0_FRAME}")
    END_FRAME = args.end_frame

    data = {}
    selected_status = {}
    selected_details = {}
    if args.source == "csv":
        # Portable mode: render straight from a consolidated all-cells CSV, no published master needed.
        if not args.csv:
            raise SystemExit("--source csv requires --csv <consolidated all_cells_lineage_data3D.csv(.gz)>")
        big = pd.read_csv(args.csv)
        for col in ("pos", "ch", "cell_id", "frame"):
            if col not in big.columns:
                raise SystemExit(f"--csv file lacks the required column '{col}'")
        badbig = pd.read_csv(args.bad_csv) if args.bad_csv else None
        flags = pd.read_csv(args.channels_csv) if args.channels_csv else None
        def _posnum(p):
            return int(str(p)[3:]) if str(p).startswith("Pos") else 0
        keys = sorted(set(map(tuple, big[["pos", "ch"]].drop_duplicates().to_numpy())),
                      key=lambda k: (_posnum(k[0]), str(k[1])))
        for pos, ch in keys:
            if len(data) >= args.max_lineages:
                break
            pn = _posnum(pos)
            if pn < args.pos_min or pn > args.pos_max:
                continue
            f = (flags[(flags["pos"] == pos) & (flags["ch"] == ch)]
                 if flags is not None else pd.DataFrame())
            try:
                df = big[(big["pos"] == pos) & (big["ch"] == ch)].copy()
                bad = (badbig[(badbig["pos"] == pos) & (badbig["ch"] == ch)].copy()
                       if badbig is not None else pd.DataFrame())
                m = df[(df["cell_id"] == 0) & (df["frame"] >= T0_FRAME) & (df["frame"] <= END_FRAME)]
                if len(m) < args.min_mother_frames:
                    continue
                data[(pos, ch)] = prepare(df, bad)
                status = f.iloc[0].get("classification_status") if len(f) else np.nan
                oob = flag_is_true(f.iloc[0].get("qc_oob_excluded")) if len(f) else False
                selected_status[(pos, ch)] = status
                selected_details[(pos, ch)] = {
                    "classification": status, "oob": oob,
                    "mother_frames": len(m), "ri_coverage": float(m[qp.efd_col(m.columns, "mean_ri")].notna().mean()),
                }
            except Exception as e:  # a malformed channel must not abort the gallery
                print(f"  skip {pos}/{ch}: {type(e).__name__}: {e}")
        source_name = "csv"
        if not data:
            raise SystemExit("no lineages selected from --csv (check columns and --min-mother-frames)")
        print(f"csv {Path(args.csv).name}: {len(data)} lineages selected")
        master = None
        parts_hdr = f"csv {Path(args.csv).name}"
    elif args.source == "master":
        if args.master_tag:
            os.environ["QPI_LINEAGE_MASTER"] = args.master_tag
        master = qp.master_dir()
        if master is None:
            raise SystemExit("no master published")
        track_path = master / "consolidated" / "channel_index.csv"
        if not track_path.is_file():
            raise SystemExit(f"master is missing full-run tracking index: {track_path}")
        tracked = pd.read_csv(track_path)
        flags = pd.read_csv(master / "derived" / "phase1_img0002-2017" / "channels.csv")
        manifest = json.loads((master / "MANIFEST.json").read_text(encoding="utf-8"))
        frame_start = int(manifest["consolidated_manifest"].get("frame_min", T0_FRAME))
        frame_end = int(pd.to_numeric(tracked["last_frame"], errors="coerce").max())
        full_frame_count = frame_end - frame_start + 1
        if full_frame_count <= 0:
            raise SystemExit(f"invalid tracked frame range: {frame_start}..{frame_end}")
        chans = tracked.merge(
            flags[["pos", "ch", "classification_status", "qc_oob_excluded"]],
            on=["pos", "ch"], how="left",
        )
        chans["pos_num"] = chans["pos"].str[3:].astype(int)
        chans["mother_frames"] = pd.to_numeric(chans["mother_frames"], errors="coerce").fillna(0)
        chans["mother_frames_with_ri"] = pd.to_numeric(chans["mother_frames_with_ri"], errors="coerce").fillna(0)
        chans["full_mother_coverage"] = chans["mother_frames"] / full_frame_count
        chans["ri_coverage"] = chans["mother_frames_with_ri"] / chans["mother_frames"].replace(0, np.nan)
        in_positions = chans["pos_num"].between(args.pos_min, args.pos_max)
        no_mother_count = int((in_positions & chans["mother_frames"].eq(0)).sum())
        sel = chans[chans["mother_frames"].gt(0)
                    & chans["pos_num"].between(args.pos_min, args.pos_max)].copy()
        sel = sel.sort_values(["pos_num", "ch"])
        sel = sel.head(args.max_lineages)
        selected_status = {(r.pos, r.ch): r.classification_status for r in sel.itertuples(index=False)}
        selected_details = {
            (r.pos, r.ch): {
                "classification": r.classification_status,
                "oob": flag_is_true(r.qc_oob_excluded),
                "mother_frames": int(r.mother_frames),
                "ri_coverage": float(r.ri_coverage) if pd.notna(r.ri_coverage) else np.nan,
                "full_coverage": float(r.full_mother_coverage),
                "n_cells": int(r.n_cells) if pd.notna(r.n_cells) else 0,
            }
            for r in sel.itertuples(index=False)
        }
        n_unclassified = int(sel["classification_status"].isna().sum())
        n_empty_conflict = int(sel["classification_status"].eq("empty").sum())
        n_oob = int(sel["qc_oob_excluded"].map(flag_is_true).sum())
        print(f"master {master.name}: {len(sel)} of {len(chans)} channels selected across {sel['pos'].nunique()} positions "
              f"(mother_frames > 0; {no_mother_count} channels with no mother rows omitted; "
              f"{n_unclassified} unclassified, {n_empty_conflict} marked empty despite tracking, {n_oob} OOB flagged; "
              f"no coverage threshold or edge-channel exclusion)")
        for r in sel.itertuples(index=False):
            df, bad = load_lineage(master, r.pos, r.ch)
            data[(r.pos, r.ch)] = prepare(df, bad)
        source_name = master.name
    else:  # working-tree: still needs the published master's channels.csv for the classification/OOB flags
        if args.master_tag:
            os.environ["QPI_LINEAGE_MASTER"] = args.master_tag
        master = qp.master_dir()
        if master is None:
            raise SystemExit("no master published")
        chans = pd.read_csv(master / "derived" / "phase1_img0002-2017" / "channels.csv")
        chans["pos_num"] = chans["pos"].str[3:].astype(int)
        cands = working_tree_lineages(args.pos_max, chans, pos_min=args.pos_min)
        print(f"working tree: {len(cands)} yellow-tracker channels found (classification/OOB flags from {master.name})")
        for pos, ch, lo in cands:
            if len(data) >= args.max_lineages:
                break
            try:
                df, bad = load_lineage_dir(lo)
                m = df[(df["cell_id"] == 0) & (df["frame"] >= T0_FRAME) & (df["frame"] <= END_FRAME)]
                if len(m) < args.min_mother_frames:
                    continue
                data[(pos, ch)] = prepare(df, bad)
                f = chans[(chans["pos"] == pos) & (chans["ch"] == ch)]
                status = f.iloc[0].get("classification_status") if len(f) else np.nan
                oob = flag_is_true(f.iloc[0].get("qc_oob_excluded")) if len(f) else False
                selected_status[(pos, ch)] = status
                selected_details[(pos, ch)] = {
                    "classification": status, "oob": oob, "mother_frames": len(m),
                    "ri_coverage": float(m[qp.efd_col(m.columns, "mean_ri")].notna().mean()),
                }
            except Exception as e:  # a malformed / half-written channel must not abort the gallery
                print(f"  skip {pos}/{ch}: {type(e).__name__}: {e}")
        source_name = "working-tree_yellow"
        if not data:
            raise SystemExit("no yellow-tracker lineages with a mother yet")
        print(f"{len(data)} lineages selected")
    data_origin = ""
    if args.source == "master":
        source_manifest = json.loads((master / "MANIFEST.json").read_text(encoding="utf-8"))
        source_cons = source_manifest["consolidated_manifest"]
        data_origin = (
            f"; masks <code>{source_manifest['source_mask_root']}</code>; "
            f"phase <code>{source_manifest['source_raw_root']}</code>; "
            f"RI calibration <code>{source_cons.get('ri_calibration')}</code>"
        )
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
             "conc": _parse(args.ylim_conc, lambda: _lim("conc")),
             "mass": _parse(args.ylim_mass, lambda: (0, _lim("mass")[1])),
             "vol": _parse(args.ylim_vol, lambda: (0, _lim("vol")[1]))}
    top_panel_label = "dry-mass concentration [mg/mL]" if args.panel1 == "conc" else "mean RI"
    top_panel_limits = ylims[args.panel1]

    out_dir = Path(args.out) if args.out else (SCRIPTS.parent / "results" / "figures" / "lineage_html")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    html_path = out_dir / f"lineage_gallery_{source_name}_{stamp}.html"
    fit_rows, parts = [], []
    for (pos, ch), d in data.items():
        png = render(pos, ch, d, ylims, panel1=args.panel1)
        (out_dir / f"{pos}_{ch}_{stamp}.png").write_bytes(png)
        b64 = base64.b64encode(png).decode("ascii")
        for fz in d["fits"]:
            fit_rows.append(dict(pos=pos, ch=ch, **fz))
        key = f"{pos} {ch}"
        divs = ",".join(f"{t_h(int(f)):.2f}" for f in d["div_ok"])  # cycle boundaries [h] for click->cycle mapping
        labels = []
        details = selected_details.get((pos, ch), {})
        status = details.get("classification", selected_status.get((pos, ch)))
        if pd.isna(status):
            labels.append("unclassified; review")
        elif status == "empty":
            labels.append("classified empty but tracked; review")
        elif status == "dead_at_start":
            labels.append("dead-at-start flag; review")
        elif status not in ("cells", None):
            labels.append(f"classification={status}; review")
        if details.get("oob"):
            labels.append("OOB flagged; review")
        if details.get("mother_frames") is not None:
            labels.append(f"mother rows {details['mother_frames']}")
        ri_coverage = details.get("ri_coverage")
        if ri_coverage is not None and np.isfinite(ri_coverage):
            labels.append(f"RI-valid {ri_coverage:.0%}")
        status_label = f" <small>({' | '.join(labels)})</small>" if labels else ""
        parts.append(f"<section id='{pos}_{ch}'><h2>{pos} {ch}{status_label} <span class='done' data-for='{key}'></span></h2>"
                     f"<div class='imgwrap' data-key='{key}' data-divs='{divs}'>"
                     f"<img src='data:image/png;base64,{b64}' alt='{pos} {ch}'></div>"
                     f"<div class='cmtwrap'><div class='marks' data-key='{key}'></div>"
                     f"<textarea class='cmt' data-key='{key}' placeholder='{key} 全体メモ (任意)'></textarea>"
                     f"</div></section>")
    pd.DataFrame(fit_rows).to_csv(out_dir / f"cycle_fits_{source_name}_{stamp}.csv", index=False)

    toc = " | ".join(f"<a href='#{p}_{c}'>{p} {c}</a>" for (p, c) in data)
    style = ("body{font-family:Arial,Helvetica,sans-serif;margin:16px;background:#fff;color:#222}"
             "img{width:100%;max-width:1500px;display:block;border:1px solid #ddd}"
             "h2{font-size:15px;margin:22px 0 4px}section{margin-bottom:6px}"
             ".note{font-size:13px;line-height:1.5;max-width:1100px}.toc{font-size:12px;line-height:1.8}"
             ".sw{display:inline-block;width:10px;height:10px;margin-right:4px;vertical-align:middle}"
             "textarea.cmt{width:100%;max-width:1100px;min-height:42px;font-family:inherit;font-size:13px;"
             "padding:6px;border:1px solid #bbb;border-radius:4px;box-sizing:border-box;resize:vertical}"
             "textarea.cmt.has{border-color:#0a8a0a;background:#f2fbf2}.cmtwrap{margin:4px 0 14px}"
             ".imgwrap{position:relative;display:inline-block;width:100%;max-width:1500px;cursor:crosshair}"
             ".imgwrap img{display:block;width:100%;border:1px solid #ddd}"
             ".dot{position:absolute;width:12px;height:12px;border-radius:50%;transform:translate(-50%,-50%);"
             "border:2px solid #fff;box-shadow:0 0 2px #000;pointer-events:none}"
             ".dot.miss{background:#c0392b}.dot.outdiv{background:#e67e22}.dot.outl{background:#16a085}"
             ".dot.weird{background:#8e44ad}.dot.note{background:#2980b9}"
             ".marks{font-size:12px;margin:3px 0;line-height:1.9}.marks .chip{display:inline-block;margin:2px 5px 2px 0;"
             "padding:2px 7px;border-radius:11px;background:#eee}.chip.miss{background:#fdecea}.chip.outdiv{background:#fbeee0}"
             ".chip.outl{background:#e8f6f3}.chip.weird{background:#f4ecf7}"
             ".chip.note{background:#eaf2f8}.chip .x{color:#900;cursor:pointer;margin-left:6px;font-weight:bold}"
             "#bar .mode button{opacity:.5;border:none;background:#555;color:#fff;border-radius:4px}"
             "#bar .mode button.active{opacity:1;background:#0a7a0a}"
             "#bar{position:fixed;top:0;right:0;background:#222;color:#fff;padding:8px 12px;font-size:13px;"
             "z-index:99;border-bottom-left-radius:6px;box-shadow:0 1px 6px rgba(0,0,0,.3)}"
             "#bar button{font-size:13px;margin-left:6px;cursor:pointer;padding:3px 8px}"
             "#bar b{color:#7fd97f}.done:after{content:attr(data-c);color:#0a8a0a;font-size:12px;font-weight:normal}")
    legend = (f"<span class='sw' style='background:{C_VALID}'></span>valid frame &nbsp; "
              f"<span style='color:{C_OUT};font-weight:bold'>&times;</span> tracker outlier &nbsp; "
              f"<span style='color:{C_BORDER}'>&#9650;</span> border &nbsp; "
              f"<span style='color:{C_DRIFT}'>&#9670;</span> drift-excluded rank-1 &nbsp; "
              f"<span style='color:{C_DIV}'>|</span> retained mother division &nbsp; "
              f"<span style='color:{C_FIT}'>&mdash;</span> ln(mass) linear fit per complete cycle (back-transformed)")
    note = (f"<p class='note'><b>Data:</b> <code>{source_name}</code>{data_origin} "
            f"(yellow-contour tracker where the volume column is volume_um3_efd), mother cell "
            f"(cell_id 0), window img_{T0_FRAME:04d}-img_{END_FRAME:04d} (0-{t_h(END_FRAME):.1f} h), 5 min/frame.<br>"
            f"<b>Mass:</b> tracker <code>{next(iter(data.values()))['mass_col_qc']}</code> using EFD volume and the recorded refractive-index basis. "
            f"Drift-excluded frame markers have no valid RI/mass value. <b>Volume:</b> {next(iter(data.values()))['vol_col']}.<br>"
            f"<b>Retained divisions:</b> division_qc_260517 rules (direct if no tracker outlier within &plusmn;1 frame; else medians of up to 3 valid "
            f"points within &plusmn;8 frames, post/pre mass 0.25-0.78, volume 0.25-0.85, |diff| &le; 0.25; duplicates within 1 h collapsed), "
            f"QC columns {next(iter(data.values()))['mass_col_qc']} / {next(iter(data.values()))['vol_col']}.<br>"
            f"<b>Fits:</b> complete cycles between two retained divisions, valid frames only (no outlier / border), &ge; {MIN_FIT_POINTS} points. "
            f"Shared y ranges (fixed): {top_panel_label} {top_panel_limits[0]:.3g}-{top_panel_limits[1]:.3g}, mass {ylims['mass'][0]:.0f}-{ylims['mass'][1]:.0f} pg, volume {ylims['vol'][0]:.0f}-{ylims['vol'][1]:.0f} &micro;m&sup3;; values outside are clipped.<br>"
            f"Selection: every channel with at least {args.min_mother_frames} tracked mother frame(s) in this window is included, "
            f"regardless of track coverage, RI validity, OOB flags, or ch number. Channels with no tracked mother rows are omitted. "
            f"Short/death-tail tracks and classification conflicts are retained and marked for review. "
            f"Pos {args.pos_min}-{args.pos_max}, first {args.max_lineages}.</p>")
    toolbar = ("<div id='bar'>マーク: <span class='mode'>"
               "<button data-m='miss' class='active' onclick='setMode(this)'>&#128308; 分裂見逃し</button>"
               "<button data-m='outdiv' onclick='setMode(this)'>&#128992; outlier誤分裂</button>"
               "<button data-m='outl' onclick='setMode(this)'>&#128994; outlier未除外</button>"
               "<button data-m='weird' onclick='setMode(this)'>&#128995; 変なcycle</button>"
               "<button data-m='note' onclick='setMode(this)'>&#128309; メモ点</button></span>"
               " &nbsp;<b><span id='cnt'>0</span></b>系列 "
               "<button onclick='copyComments()'>コピー</button>"
               "<button onclick='downloadComments()'>.md</button>"
               "<button onclick='clearComments()'>消去</button></div>")
    # plain string (literal braces): click a lineage plot to drop a time-stamped mark of the active
    # category (miss / weird cycle / note); stored per lineage in localStorage; one-click clipboard copy.
    script = ("<script>\n"
              "const KEY='lineage_cmt_260517';let MODE='miss';\n"
              f"const ENDH={t_h(END_FRAME):.4f},L={PLOT_LEFT},R={PLOT_RIGHT};\n"
              "const CAT={miss:'\\u5206\\u88c2\\u898b\\u9003\\u3057',outdiv:'outlier\\u8aa4\\u5206\\u88c2',"
              "outl:'outlier\\u672a\\u9664\\u5916',weird:'\\u5909\\u306acycle',note:'\\u30e1\\u30e2\\u70b9'};\n"
              "function setMode(b){MODE=b.dataset.m;document.querySelectorAll('#bar .mode button').forEach(x=>x.classList.remove('active'));b.classList.add('active');}\n"
              "function store(){let o={};try{o=JSON.parse(localStorage.getItem(KEY)||'{}');}catch(e){}return o;}\n"
              "function save(o){try{localStorage.setItem(KEY,JSON.stringify(o));}catch(e){}}\n"
              "function kd(o,k){if(!o[k])o[k]={marks:[],t:''};if(!o[k].marks)o[k].marks=[];return o[k];}\n"
              "function divsOf(w){const s=w.dataset.divs;return s?s.split(',').map(Number):[];}\n"
              "function cycleOf(dv,t){let a=0,b=ENDH;for(const d of dv){if(d<=t&&d>a)a=d;if(d>t&&d<b){b=d;break;}}return [a,b];}\n"
              "function count(){const o=store();let n=0;Object.keys(o).forEach(k=>{if((o[k].marks&&o[k].marks.length)||(o[k].t&&o[k].t.trim()))n++;});document.getElementById('cnt').textContent=n;}\n"
              "function chipTxt(m){return CAT[m.c]+' @'+m.t.toFixed(1)+'h'+(m.c==='weird'&&m.cyc?' (cycle '+m.cyc[0].toFixed(0)+'-'+m.cyc[1].toFixed(0)+'h)':'');}\n"
              "function draw(){const o=store();document.querySelectorAll('.imgwrap').forEach(w=>{w.querySelectorAll('.dot').forEach(d=>d.remove());"
              "const d=o[w.dataset.key];if(d&&d.marks)d.marks.forEach(m=>{const el=document.createElement('div');el.className='dot '+m.c;"
              "el.style.left=(m.x*100)+'%';el.style.top=(m.y*100)+'%';w.appendChild(el);});});"
              "document.querySelectorAll('.marks').forEach(box=>{const d=o[box.dataset.key];box.innerHTML='';"
              "if(d&&d.marks)d.marks.forEach((m,i)=>{const c=document.createElement('span');c.className='chip '+m.c;"
              "c.innerHTML=chipTxt(m)+\" <span class='x' data-k='\"+box.dataset.key+\"' data-i='\"+i+\"'>\\u00d7</span>\";box.appendChild(c);});});count();}\n"
              "function asText(){const o=store();const out=[];document.querySelectorAll('.imgwrap').forEach(w=>{const k=w.dataset.key;const d=o[k];if(!d)return;"
              "const p=[];if(d.marks)d.marks.forEach(m=>{p.push(CAT[m.c]+'@'+m.t.toFixed(1)+'h'+(m.c==='weird'&&m.cyc?'('+m.cyc[0].toFixed(0)+'-'+m.cyc[1].toFixed(0)+'h)':''));});"
              "let s=p.join(', ');if(d.t&&d.t.trim())s+=(s?'; ':'')+d.t.trim().replace(/\\s*\\n\\s*/g,' ');if(s)out.push(k+': '+s);});return out.join('\\n');}\n"
              "document.addEventListener('click',e=>{const img=e.target.closest('.imgwrap img');if(!img)return;"
              "const w=img.closest('.imgwrap');const r=img.getBoundingClientRect();const x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;"
              "let t=(x-L)/(R-L)*ENDH;t=Math.max(0,Math.min(ENDH,t));const o=store();const d=kd(o,w.dataset.key);"
              "const m={c:MODE,x:x,y:y,t:t};if(MODE==='weird')m.cyc=cycleOf(divsOf(w),t);d.marks.push(m);save(o);draw();});\n"
              "document.addEventListener('click',e=>{if(e.target.classList&&e.target.classList.contains('x')){const o=store();const k=e.target.dataset.k,i=+e.target.dataset.i;"
              "if(o[k]&&o[k].marks){o[k].marks.splice(i,1);save(o);draw();}}});\n"
              "document.addEventListener('input',e=>{if(e.target.classList&&e.target.classList.contains('cmt')){const o=store();kd(o,e.target.dataset.key).t=e.target.value;save(o);"
              "e.target.classList.toggle('has',!!e.target.value.trim());count();}});\n"
              "function copyComments(){const s=asText();if(!s){alert('\\u30de\\u30fc\\u30af\\u3082\\u30e1\\u30e2\\u3082\\u3042\\u308a\\u307e\\u305b\\u3093');return;}const n=s.split('\\n').length;"
              "if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(s).then(()=>alert('\\u30b3\\u30d4\\u30fc\\u3057\\u307e\\u3057\\u305f ('+n+')'),()=>window.prompt('copy:',s));}else{window.prompt('copy:',s);}}\n"
              "function downloadComments(){const s=asText();if(!s){alert('empty');return;}const b=new Blob([s],{type:'text/markdown'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='lineage_marks_'+Date.now()+'.md';a.click();}\n"
              "function clearComments(){if(confirm('clear all?')){try{localStorage.removeItem(KEY);}catch(e){}document.querySelectorAll('textarea.cmt').forEach(t=>{t.value='';t.classList.remove('has');});draw();}}\n"
              "window.addEventListener('DOMContentLoaded',()=>{const o=store();document.querySelectorAll('textarea.cmt').forEach(t=>{const d=o[t.dataset.key];if(d&&d.t){t.value=d.t;t.classList.add('has');}});draw();});\n"
              "</script>")
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>260517 lineage gallery {source_name}: {top_panel_label}</title>"
            f"<style>{style}</style></head><body>{toolbar}<h1 style='font-size:18px'>260517 mother lineages: {top_panel_label} / dry mass / EFD volume</h1>"
            f"{note}<p class='note'>{legend}</p>"
            f"<p class='note'><b>該当箇所を指定する:</b> 右上でマーク種類を選び、その系列の図の該当箇所をクリックすると、その時刻に印が付く。"
            f"<b style='color:#c0392b'>分裂見逃し</b>=分裂すべきなのに灰色の縦線が無い / "
            f"<b style='color:#e67e22'>outlier誤分裂</b>=outlier由来で誤って分裂と判定 / "
            f"<b style='color:#16a085'>outlier未除外</b>=outlierなのに除外されず有効点のまま / "
            f"<b style='color:#8e44ad'>変なcycle</b>=挙動がおかしい cycle（分裂線から範囲も自動付与）/ "
            f"<b style='color:#2980b9'>メモ点</b>。右上「コピー」で <code>Pos5 ch04: 分裂見逃し@42.5h, outlier誤分裂@88.0h</code> "
            f"の形でまとめてコピー -> チャットに1回貼り付け。印は&times;で1つずつ消せる。マークとメモはこのブラウザに自動保存される（再読込で残る）。</p>"
            f"<p class='toc'>{toc}</p>{''.join(parts)}{script}</body></html>")
    html_path.write_text(html, encoding="utf-8")
    print("html:", html_path)
    print("fits csv:", out_dir / f"cycle_fits_{source_name}_{stamp}.csv", "| cycles fitted:", len(fit_rows))
    (out_dir / f"params_{stamp}.json").write_text(json.dumps(dict(
        source=source_name, wavelength_um=WAVELENGTH_UM, pixel_um=PIXEL_UM, alpha_ml_per_mg=ALPHA_ML_PER_MG,
        dt_min=DT_MIN, t0_frame=T0_FRAME, end_frame=END_FRAME, ylims=ylims, n_lineages=len(data)), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
