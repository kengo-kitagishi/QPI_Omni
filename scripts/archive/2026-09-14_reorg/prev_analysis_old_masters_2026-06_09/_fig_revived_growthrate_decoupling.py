"""_fig_revived_growthrate_decoupling.py — A4: revival growth rate + decoupling.

For the 260517 starvation experiment (2% -> 0.0055% -> 0% -> 2%), the revived
mothers (survived 0% starvation, resumed growth after recovery to 2% at frame
2885). Per cell cycle we fit BOTH d ln(dry mass)/dt and d ln(volume)/dt (OLS slope
of log vs time_h, the same method as the canonical mass rate), and ask:
  (1) is the post-recovery rate higher than the same lineages' phase1 steady state?
  (2) do mass growth and volume growth DECOUPLE right after recovery (rates differ)?
plotted against time from recovery. Survivor (revived) aggregates are mean +/- SD.

Two figures:
  A. fit-check grid: per lineage, mass(t) and volume(t) on log-y with the per-cycle
     exponential fits overlaid, windowed around recovery — so the fits can be
     verified visually (on log-y an exponential fit is a straight line).
  B. decoupling: d ln(M)/dt and d ln(V)/dt vs time-from-recovery (per-cycle points +
     binned mean +/- SD), phase1 steady-state mean +/- SD bands; and their
     difference (M rate - V rate) vs time.

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_revived_growthrate_decoupling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overlay_mean_sd_band_full_timecourse import list_revived_mothers  # noqa: E402
from qpi_paths import resolve_lineage_csv, find_corrected_lineage_csv  # noqa: E402
from _fig_panelA_cellcycle import enumerate_all_cycles  # noqa: E402
from figure_logger import save_figure  # noqa: E402

PHASE1_END = 2018
RECOVERY_FRAME = 2885
FPH = 12.0                 # frames per hour (time_h = frame / 12)
MASS_C = "#0072B2"         # blue — dry mass
VOL_C = "#009E73"          # green — volume
WIN = (-12.0, 30.0)        # fit-check window, h from recovery
TMAX = 36.0                # decoupling time axis, h from recovery
BINW = 3.0

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_mother(pos, ch):
    df = pd.read_csv(resolve_lineage_csv(pos, ch))
    return df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)


def _fit_log(t, y):
    sl, ic = np.polyfit(t, np.log(y), 1)
    return float(sl), float(ic)


def cycle_rates_mass_vol(m):
    """Per division-bounded cycle: d ln(M)/dt and d ln(V)/dt (same cycle, both fit).
    Starvation arrest is dropped automatically (too-long cycle)."""
    cyc, _ = enumerate_all_cycles(m, set(), max_frame=3747, min_kept=6)
    out = []
    for c in cyc:
        v = c[~(c["is_outlier"] | c["touches_border"])]
        rec = {}
        vm = v[v["mass_pg"] >= 10.0]
        if len(vm) >= 4 and not np.any(vm["mass_pg"] <= 0):
            sl, ic = _fit_log(vm["time_h"].to_numpy(float), vm["mass_pg"].to_numpy(float))
            rec.update(m_rate=sl, m_ic=ic,
                       m_t0=float(vm["time_h"].min()), m_t1=float(vm["time_h"].max()))
        vv = v[v["volume_um3_rod"] > 0]
        if len(vv) >= 4 and not np.any(vv["volume_um3_rod"] <= 0):
            sl, ic = _fit_log(vv["time_h"].to_numpy(float),
                              vv["volume_um3_rod"].to_numpy(float))
            rec.update(v_rate=sl, v_ic=ic,
                       v_t0=float(vv["time_h"].min()), v_t1=float(vv["time_h"].max()))
        if "m_rate" in rec and "v_rate" in rec:
            rec["f0"] = int(c["frame"].min()); rec["f1"] = int(c["frame"].max())
            rec["fmid"] = 0.5 * (rec["f0"] + rec["f1"])
            rec["tmid"] = (rec["fmid"] - RECOVERY_FRAME) / FPH
            out.append(rec)
    return out


def collect():
    revived = [pc for pc in list_revived_mothers()
               if find_corrected_lineage_csv(*pc) is not None]
    data = {}
    for pos, ch in revived:
        m = load_mother(pos, ch)
        rates = cycle_rates_mass_vol(m)
        p1 = [r for r in rates if r["f1"] <= PHASE1_END]
        rec = sorted([r for r in rates if r["f0"] >= RECOVERY_FRAME],
                     key=lambda r: r["fmid"])
        for g, r in enumerate(rec, 1):
            r["gen"] = g
        if not p1 or not rec:
            continue
        data[f"{pos}_{ch}"] = dict(
            m=m, rec=rec,
            p1_m=float(np.median([r["m_rate"] for r in p1])),
            p1_v=float(np.median([r["v_rate"] for r in p1])))
    return data


def _binned(x, y, edges):
    mu, sd, ct, ctr = [], [], [], []
    for i in range(len(edges) - 1):
        sel = (x >= edges[i]) & (x < edges[i + 1])
        if sel.sum() >= 3:
            mu.append(np.mean(y[sel])); sd.append(np.std(y[sel], ddof=1))
            ct.append(int(sel.sum())); ctr.append(0.5 * (edges[i] + edges[i + 1]))
    return np.array(ctr), np.array(mu), np.array(sd), np.array(ct)


# --------------------------------------------------------------------------- B
def figure_decoupling(data):
    p1m = np.array([d["p1_m"] for d in data.values()])
    p1v = np.array([d["p1_v"] for d in data.values()])
    pts_t, pts_m, pts_v = [], [], []
    for d in data.values():
        for r in d["rec"]:
            if 0 <= r["tmid"] <= TMAX:
                pts_t.append(r["tmid"]); pts_m.append(r["m_rate"]); pts_v.append(r["v_rate"])
    pts_t = np.array(pts_t); pts_m = np.array(pts_m); pts_v = np.array(pts_v)
    edges = np.arange(0, TMAX + BINW, BINW)
    bt_m, bm_m, bs_m, bn = _binned(pts_t, pts_m, edges)
    bt_v, bm_v, bs_v, _ = _binned(pts_t, pts_v, edges)
    bt_d, bm_d, bs_d, _ = _binned(pts_t, pts_m - pts_v, edges)

    fig, axes = plt.subplots(2, 1, figsize=(120 / 25.4, 130 / 25.4),
                             sharex=True, constrained_layout=True)
    ax = axes[0]
    # phase1 steady-state mean +/- SD bands
    ax.axhspan(p1m.mean() - p1m.std(ddof=1), p1m.mean() + p1m.std(ddof=1),
               color=MASS_C, alpha=0.12, lw=0)
    ax.axhspan(p1v.mean() - p1v.std(ddof=1), p1v.mean() + p1v.std(ddof=1),
               color=VOL_C, alpha=0.12, lw=0)
    ax.axhline(p1m.mean(), color=MASS_C, lw=0.7, ls=":")
    ax.axhline(p1v.mean(), color=VOL_C, lw=0.7, ls=":")
    ax.scatter(pts_t, pts_m, s=4, color=MASS_C, alpha=0.18, lw=0)
    ax.scatter(pts_t, pts_v, s=4, color=VOL_C, alpha=0.18, lw=0)
    ax.errorbar(bt_m, bm_m, yerr=bs_m, color=MASS_C, lw=1.8, marker="o", ms=3,
                capsize=2, label="d ln M/dt (mass)")
    ax.errorbar(bt_v, bm_v, yerr=bs_v, color=VOL_C, lw=1.8, marker="s", ms=3,
                capsize=2, label="d ln V/dt (volume)")
    ax.axvline(0, color="#888", lw=0.8, ls="--")
    ax.set_ylabel(r"specific growth rate [h$^{-1}$]")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("revived: post-recovery growth rate vs phase1 steady state "
                 "(bands = phase1 mean±SD)", fontsize=7)
    ax = axes[1]
    p1d = p1m - p1v
    ax.axhspan(p1d.mean() - p1d.std(ddof=1), p1d.mean() + p1d.std(ddof=1),
               color="#888", alpha=0.15, lw=0)
    ax.axhline(0, color="#bbb", lw=0.7)
    ax.scatter(pts_t, pts_m - pts_v, s=4, color="#444", alpha=0.18, lw=0)
    ax.errorbar(bt_d, bm_d, yerr=bs_d, color="#222", lw=1.8, marker="D", ms=3,
                capsize=2)
    ax.axvline(0, color="#888", lw=0.8, ls="--")
    ax.set_xlabel("time from recovery (frame 2885) [h]")
    ax.set_ylabel(r"$d\ln M/dt - d\ln V/dt$ [h$^{-1}$]")
    ax.set_title("decoupling: mass-rate minus volume-rate (>0 = densifying; "
                 "grey band = phase1 mean±SD)", fontsize=7)
    fig.suptitle("A4 — revived growth rate & mass/volume decoupling after recovery "
                 f"(EFD, n={len(data)})", fontsize=8)

    caption = (
        "After the medium returns to 2% glucose, revived mother lineages resume growth; "
        "the figure compares their post-recovery dry-mass and volume specific growth "
        "rates to the same lineages' phase1 steady state and tests whether mass and "
        "volume growth decouple. x = time from recovery (frame 2885; time_h = "
        "frame/12) [h]. Operational definitions: per division-bounded cell cycle, "
        "d ln M/dt and d ln V/dt = OLS slope of ln(mass_pg) resp. ln(volume_um3_rod) "
        "vs time_h (>=4 valid frames; not is_outlier/touches_border; mass_pg >= 10, "
        "volume > 0); the long starvation-arrest 'cycle' is dropped automatically (too "
        "long). phase1 steady state = each lineage's median per-cycle rate over cycles "
        "ending at frame <= 2018, pooled across lineages. Top panel: faint points = "
        "individual post-recovery cycle rates (blue = mass, green = volume); thick "
        "lines = binned (3 h) mean +/- SD across cycles; shaded horizontal bands = "
        "phase1 mean +/- SD (blue mass, green volume); dotted lines = phase1 means; "
        "dashed vertical line = recovery. Bottom panel: per-cycle (mass rate - volume "
        "rate) vs time (>0 means mass outgrows volume = densifying), binned mean +/- "
        "SD, grey band = phase1 mean +/- SD of the difference. Survivor (revived) "
        f"aggregates are mean +/- SD (not median/IQR). n = {len(data)} revived "
        "lineages with an EFD corrected lineage. Statistics: descriptive; error bars = "
        "SD across cycles (per bin, n varies). Conditions: Schizosaccharomyces pombe, "
        "260517 Mother-Machine starvation dataset (2% -> 0.0055% -> 0% -> 2%), QPI EFD "
        "volume variant [strain/genotype and temperature: confirm from acquisition "
        "metadata]. Abbreviations: SD, standard deviation; EFD, elliptic Fourier "
        "descriptor; OLS, ordinary least squares. Data availability: per-cycle rates "
        "and phase1 references in this run's *_data.npz."
    )
    npz = {"p1_mass_rate": p1m, "p1_vol_rate": p1v,
           "pts_t": pts_t, "pts_mass_rate": pts_m, "pts_vol_rate": pts_v,
           "bin_t_mass": bt_m, "bin_mean_mass": bm_m, "bin_sd_mass": bs_m, "bin_n": bn,
           "bin_t_vol": bt_v, "bin_mean_vol": bm_v, "bin_sd_vol": bs_v,
           "bin_t_diff": bt_d, "bin_mean_diff": bm_d, "bin_sd_diff": bs_d}
    return fig, caption, npz, (p1m, p1v, bt_m, bm_m, bm_v)


# --------------------------------------------------------------------------- A
def figure_fitcheck(data):
    keys = list(data.keys())
    n = len(keys)
    ncols = 6
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(183 / 25.4, (30 * nrows + 8) / 25.4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    fdata = {}
    for ax, k in zip(axes, keys):
        m = data[k]["m"]
        v = m[~(m["is_outlier"] | m["touches_border"])].copy()
        v["tr"] = (v["frame"].to_numpy(float) - RECOVERY_FRAME) / FPH
        v = v[(v["tr"] >= WIN[0]) & (v["tr"] <= WIN[1])]
        vm = v[v["mass_pg"] >= 10.0]
        vv = v[v["volume_um3_rod"] > 0]
        ax.set_yscale("log")
        ax.axvspan(0, WIN[1], color="#999", alpha=0.06, lw=0)
        ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.7)
        ax.plot(vm["tr"], vm["mass_pg"], "o", color=MASS_C, ms=1.4, mew=0, alpha=0.6)
        ax.plot(vv["tr"], vv["volume_um3_rod"], "s", color=VOL_C, ms=1.4, mew=0,
                alpha=0.6)
        # per-cycle exp fits (lines on log-y) for all cycles overlapping the window
        cyc, _ = enumerate_all_cycles(m, set(), max_frame=3747, min_kept=6)
        for c in cyc:
            cc = c[~(c["is_outlier"] | c["touches_border"])]
            cm = cc[cc["mass_pg"] >= 10.0]
            cv = cc[cc["volume_um3_rod"] > 0]
            if len(cm) >= 4:
                t = cm["time_h"].to_numpy(float)
                tr = (cm["frame"].to_numpy(float) - RECOVERY_FRAME) / FPH
                if tr.max() < WIN[0] or tr.min() > WIN[1]:
                    pass
                else:
                    sl, ic = _fit_log(t, cm["mass_pg"].to_numpy(float))
                    ax.plot(tr, np.exp(ic + sl * t), color=MASS_C, lw=0.8, ls="-",
                            alpha=0.9)
            if len(cv) >= 4:
                t = cv["time_h"].to_numpy(float)
                tr = (cv["frame"].to_numpy(float) - RECOVERY_FRAME) / FPH
                if not (tr.max() < WIN[0] or tr.min() > WIN[1]):
                    sl, ic = _fit_log(t, cv["volume_um3_rod"].to_numpy(float))
                    ax.plot(tr, np.exp(ic + sl * t), color=VOL_C, lw=0.8, ls="-",
                            alpha=0.9)
        ax.set_xlim(*WIN)
        ax.set_title(k, fontsize=5.5)
        ax.tick_params(length=2, labelsize=5)
        fdata[f"{k}_tr"] = v["tr"].to_numpy(float)
        fdata[f"{k}_mass_pg"] = v["mass_pg"].to_numpy(float)
        fdata[f"{k}_volume_um3"] = v["volume_um3_rod"].to_numpy(float)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("A4 fit-check — revived lineages: dry mass (blue) and volume (green) "
                 "on log-y with per-cycle exponential fits; x = h from recovery, "
                 f"shaded = post-recovery (EFD, n={n}). On log-y a correct exp fit is a "
                 "straight line through the points.", fontsize=6.5)
    caption = (
        "Per revived lineage, dry mass and volume around the 2%-glucose recovery on a "
        "logarithmic y-axis with the per-cycle exponential fits overlaid, so each fit "
        "behind the d ln M/dt and d ln V/dt rates can be checked visually (on log-y an "
        "exponential M0*exp(k t) is a straight line, so a good fit lies on the points). "
        "Each panel is one revived lineage. x = time from recovery (frame 2885) [h]; "
        "y (log) = dry mass [pg] (blue points) and volume [um^3] (green points). Lines "
        "= per-cycle exponential fits (blue mass, green volume) of the cycles "
        "overlapping the window. valid frames only (not is_outlier/touches_border; "
        "mass_pg >= 10; volume > 0). dashed vertical line = recovery; shaded = "
        f"post-recovery. window {WIN[0]:g}..{WIN[1]:g} h. n = {n} revived lineages with "
        "an EFD corrected lineage. Statistics: none (visual QC). Conditions: "
        "Schizosaccharomyces pombe, 260517 Mother-Machine starvation dataset, QPI EFD "
        "volume variant. Abbreviations: EFD, elliptic Fourier descriptor. Data "
        "availability: per-lineage mass and volume traces in this run's *_data.npz."
    )
    return fig, caption, fdata


def main():
    data = collect()
    print(f"revived lineages used: {len(data)}")
    fig_b, cap_b, npz_b, summ = figure_decoupling(data)
    p1m, p1v, bt_m, bm_m, bm_v = summ
    print(f"phase1 steady: mass {p1m.mean():.3f}±{p1m.std(ddof=1):.3f}  "
          f"vol {p1v.mean():.3f}±{p1v.std(ddof=1):.3f}  (mean±SD, n={len(p1m)})")
    if len(bm_m):
        print(f"first post-recovery bin (~{bt_m[0]:.1f}h): mass={bm_m[0]:.3f}  "
              f"vol={bm_v[0]:.3f}")
    save_figure(fig_b,
                params={"volume_variant": "efd", "recovery_frame": RECOVERY_FRAME,
                        "phase1_end": PHASE1_END, "n_revived": len(data),
                        "bin_width_h": BINW, "tmax_h": TMAX},
                description="A4 revived growth rate & mass/volume decoupling vs time "
                            "from recovery, mean±SD, phase1 steady-state bands (EFD)",
                caption=cap_b, data=npz_b)
    plt.close(fig_b)

    fig_a, cap_a, npz_a = figure_fitcheck(data)
    save_figure(fig_a,
                params={"volume_variant": "efd", "recovery_frame": RECOVERY_FRAME,
                        "n_revived": len(data), "window_h": list(WIN)},
                description="A4 fit-check: per-lineage mass(t) & volume(t) log-y with "
                            "per-cycle exponential fits around recovery (EFD)",
                caption=cap_a, data=npz_a)
    plt.close(fig_a)


if __name__ == "__main__":
    main()
