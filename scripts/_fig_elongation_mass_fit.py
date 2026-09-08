"""_fig_elongation_mass_fit.py — mother mass(t) + per-cycle exp fit for the two
elongation-cascade phase1 deaths (Pos20_ch06, Pos30_ch04), EFD.

Zooms into the terminal part of each lineage so the per-cycle exponential fits are
legible and the terminal filament elongation can be compared to the preceding
normal cycles. Everything is DATA-DRIVEN — no hand-picked YAML frames:

  peak  = the highest-mass valid frame within phase1 (<=2018). The terminal
          filament is far heavier than any normal cycle, so this is its peak.
  onset = walk back from the peak along the monotonic rise to its valley = the
          post-division mass minimum, i.e. the frame just after the LAST normal
          division, where filament growth begins.
  elongation fit = exponential over [onset, peak] (the clean monotonic rise).

The plot extends a short margin past the peak to show the lysis drop; the post-
lysis repopulation (contaminated) is not used.

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_elongation_mass_fit.py
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
from qpi_paths import resolve_lineage_csv  # noqa: E402
from _fig_panelA_cellcycle import enumerate_all_cycles  # noqa: E402
from figure_logger import save_figure  # noqa: E402

CHANNELS = ["Pos20_ch06", "Pos30_ch04"]
PHASE1_END = 2018       # EFD / phase1 limit for the peak search (not YAML)
PEAK_MARGIN = 30        # frames shown past the peak, to display the lysis drop
N_TAIL_CYC = 12
ELONG_C = "#D55E00"     # elongation fit (OI vermilion)

# Per-channel override: fit the exponential over only the first N hours of the
# elongation (from the post-division onset) and EXTRAPOLATE it to the peak.
# Channels not listed use the full onset->peak fit.
FIT_HOURS = {"Pos20_ch06": 3.0, "Pos30_ch04": 3.0}


def _fit(t, mass):
    sl, ic = np.polyfit(t, np.log(mass), 1)
    return float(sl), float(ic)


def _valid(m, f0, f1):
    return m[(m["frame"] >= f0) & (m["frame"] <= f1)
             & ~(m["is_outlier"] | m["touches_border"]) & (m["mass_pg"] >= 10.0)]


def find_onset_peak(m: pd.DataFrame) -> tuple[int, int, int]:
    """(onset_frame, peak_frame, last_div_frame).

    peak     = heaviest valid phase1 frame (smoothed) = the filament peak.
    last_div = the last real division (cycle end) before the peak; the filament
               grows after it without dividing.
    onset    = the post-division mass valley just after last_div (start of the
               monotonic filament rise). Anchored to a division, robust to the
               noise/plateau near the peak.
    """
    v = _valid(m, 0, PHASE1_END).sort_values("frame")
    frames = v["frame"].to_numpy(int)
    sm = pd.Series(v["mass_pg"].to_numpy(float)).rolling(
        9, center=True, min_periods=1).mean().to_numpy()
    peak_frame = int(frames[int(np.argmax(sm))])

    cyc, _ = enumerate_all_cycles(m, set(), max_frame=PHASE1_END, min_kept=6)
    divs = [int(c["frame"].max()) for c in cyc if int(c["frame"].max()) < peak_frame]
    last_div = max(divs) if divs else int(frames[0])
    # valley = lightest valid frame within ~20 frames after that division
    w = v[(v["frame"] >= last_div) & (v["frame"] <= min(last_div + 20, peak_frame))]
    onset = int(w.loc[w["mass_pg"].idxmin(), "frame"]) if len(w) else last_div
    return onset, peak_frame, last_div


# metric column, axis label, log-symbol used in the fit label
METRICS = [("mass_pg", "dry mass [pg]", "M"),
           ("volume_um3_rod", r"volume [µm$^3$]", "V")]


def draw_panel(ax, m, cyc, onset, peak, fit_h, metric, ylabel, sym):
    """Plot metric(t) over the terminal window with per-cycle exp fits + the
    elongation fit (first fit_h h + extrapolation, or onset->peak). Returns
    (last_normal_rate, elongation_rate, seg)."""
    fits = []
    for c in cyc:
        v = _valid(c, int(c["frame"].min()), int(c["frame"].max()))
        if len(v) < 4:
            continue
        sl, ic = _fit(v["time_h"].to_numpy(float), v[metric].to_numpy(float))
        fits.append({"rate": sl, "ic": ic, "t0": float(v["time_h"].min()),
                     "t1": float(v["time_h"].max()), "f0": int(c["frame"].min())})
    tail = fits[-N_TAIL_CYC:]
    win_f0 = tail[0]["f0"] if tail else onset - 100

    seg = _valid(m, win_f0, peak + PEAK_MARGIN)
    ax.plot(seg["time_h"], seg[metric], color="#555", lw=0.7, alpha=0.85,
            marker="o", ms=1.5, mfc="#555", mec="none")
    for f in tail:
        tt = np.linspace(f["t0"], f["t1"], 20)
        ax.plot(tt, np.exp(f["ic"] + f["rate"] * tt), color="#222", lw=1.0,
                ls="--", zorder=5)
    ax.plot([], [], color="#222", lw=1.0, ls="--", label="per-cycle exp fit")

    el = _valid(m, onset, peak)
    el_rate = np.nan
    if len(el) >= 4:
        t = el["time_h"].to_numpy(float)
        y = el[metric].to_numpy(float)
        t0, t_peak = float(t.min()), float(t.max())
        if fit_h is not None:
            sel = t <= t0 + fit_h
            if sel.sum() >= 4:
                sl, ic = _fit(t[sel], y[sel])
                el_rate = sl
                tt = np.linspace(t0, t_peak, 60)
                ax.plot(tt, np.exp(ic + sl * tt), color=ELONG_C, lw=2.0, zorder=6,
                        label=f"first {fit_h:.0f} h fit + extrap  "
                              f"$d\\ln {sym}/dt$={sl:.3f} h$^{{-1}}$")
                ax.axvspan(t0, t0 + fit_h, color=ELONG_C, alpha=0.22, zorder=0)
        else:
            sl, ic = _fit(t, y)
            el_rate = sl
            tt = np.linspace(t0, t_peak, 40)
            ax.plot(tt, np.exp(ic + sl * tt), color=ELONG_C, lw=2.0, zorder=6,
                    label=f"onset→peak fit  $d\\ln {sym}/dt$={sl:.3f} h$^{{-1}}$")
            ax.axvspan(t0, t_peak, color=ELONG_C, alpha=0.10, zorder=0)
        ax.axvline(t0, color=ELONG_C, lw=0.6, ls=":")
        if fit_h is not None:
            ax.set_ylim(0, 1.18 * float(seg[metric].max()))
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=False, fontsize=5.5)
    ax.spines[["top", "right"]].set_visible(False)
    return (tail[-1]["rate"] if tail else np.nan), el_rate, seg


def main():
    fig, axes = plt.subplots(2, 2, figsize=(190 / 25.4, 110 / 25.4),
                             constrained_layout=True)
    out = {}
    for row, ch_key in enumerate(CHANNELS):
        pos, ch = ch_key.split("_", 1)
        df = pd.read_csv(resolve_lineage_csv(pos, ch))
        m = df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)
        onset, peak, last_div = find_onset_peak(m)
        cyc, _ = enumerate_all_cycles(m, set(), max_frame=onset - 1, min_kept=6)
        fit_h = FIT_HOURS.get(ch_key)

        rates = {}
        for col, (metric, ylabel, sym) in enumerate(METRICS):
            ax = axes[row, col]
            last_norm, el_rate, seg = draw_panel(
                ax, m, cyc, onset, peak, fit_h, metric, ylabel, sym)
            ax.set_title(f"{ch_key}  {sym}: normal {last_norm:.3f}, "
                         f"elong {el_rate:.3f} h$^{{-1}}$", fontsize=6.5)
            rates[sym] = (last_norm, el_rate)
            out[f"{ch_key}_{metric}_time_h"] = seg["time_h"].to_numpy(float)
            out[f"{ch_key}_{metric}_y"] = seg[metric].to_numpy(float)
            out[f"{ch_key}_{metric}_normal_rate"] = np.array(last_norm)
            out[f"{ch_key}_{metric}_elong_rate"] = np.array(el_rate)
        out[f"{ch_key}_onset_frame"] = np.array(onset)
        out[f"{ch_key}_peak_frame"] = np.array(peak)
        # density implication: d ln(conc)/dt = d ln M/dt - d ln V/dt
        dM, dV = rates["M"][1], rates["V"][1]
        print(f"{ch_key}: onset={onset} peak={peak}  elong d lnM/dt={dM:.4f}  "
              f"d lnV/dt={dV:.4f}  -> d ln(conc)/dt={dM - dV:+.4f} h^-1")

    for col in range(2):
        axes[1, col].set_xlabel("time [h]")
    save_figure(
        fig,
        params={"volume_variant": "efd", "channels": CHANNELS,
                "metrics": ["mass_pg", "volume_um3_rod"],
                "fit_hours": FIT_HOURS,
                "onset_rule": "post-division valley before the global phase1 mass "
                              "peak (data-driven, no YAML frames)",
                "peak_search_max_frame": PHASE1_END, "n_tail_cycles": N_TAIL_CYC},
        description="elongation-cascade mothers (Pos20_ch06, Pos30_ch04): terminal "
                    "dry mass(t) AND volume(t) with per-cycle exp fits + first-3h "
                    "elongation fit & extrapolation; compares d ln M/dt vs d ln V/dt "
                    "(density trend) over the elongation period, EFD",
        data=out,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
