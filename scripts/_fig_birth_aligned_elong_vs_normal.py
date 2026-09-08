"""_fig_birth_aligned_elong_vs_normal.py — volume / density / mass vs TIME FROM
BIRTH, elongation-cascade lineages vs normally-dividing survivors (EFD).

Reference-style (cdc25 vs wild-type) comparison, but time-aligned to birth rather
than to relative cycle progression (the elongation lineages cannot be cycle-
aligned). Question: does density (mass/volume) fall at the same rate in the
elongation cascades as in normal dividing cells?

  - normal  (cyan, wild-type analog): gold-standard survivor cell cycles, pooled
            by time from birth (mean +/- SD); cycles end at division (~2.5-3.3 h).
  - elong   (red, cdc25 analog): the two elongation-cascade lineages, from their
            post-division onset (= birth of the terminal non-dividing growth).

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_birth_aligned_elong_vs_normal.py
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
from overlay_gold_standard_and_phase1_dead import select_gold_standard  # noqa: E402
from gold_standard_phase1_homeostasis import (  # noqa: E402
    load_mother_cycles_csv, PHASE1_END_FRAME,
)
from qpi_paths import resolve_lineage_csv  # noqa: E402
from _fig_elongation_mass_fit import find_onset_peak  # noqa: E402
from figure_logger import save_figure  # noqa: E402

NORMAL_C = "#2c9fb3"        # cyan (wild-type analog)
ELONG_C = ["#c0392b", "#e07b39"]   # reds (elongation lineages)
ELONG_CH = ["Pos20_ch06", "Pos30_ch04"]
TMAX = 12.0          # elongation traces + x-axis
NORMAL_TMAX = 3.0    # normal band only plotted to 3 h
BINW = 0.1
MIN_N = 5

METRICS = [("volume", r"volume [µm$^3$]"),
           ("density", "density [mg/mL]"),
           ("mass", "mass [pg]")]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _metrics(sub: pd.DataFrame) -> dict:
    vol = sub["volume_um3_rod"].to_numpy(float)
    mass = sub["mass_pg"].to_numpy(float)
    return {"volume": vol, "mass": mass, "density": 1000.0 * mass / vol}


def pool_normal():
    """Pool gold survivor cycles by time from birth -> mean/SD per metric."""
    bins = np.arange(0, NORMAL_TMAX + BINW, BINW)
    nb = len(bins)
    acc = {k: [[] for _ in range(nb)] for k, _ in METRICS}
    n_cyc = 0
    for pos, ch in select_gold_standard():
        res = load_mother_cycles_csv(pos, ch, max_frame=PHASE1_END_FRAME)
        if res is None:
            continue
        m_df, cycles = res
        for c in cycles:
            bf, dfr = c["birth_frame"], c["div_frame"]
            sub = m_df[(m_df["frame"] >= bf) & (m_df["frame"] <= dfr - 1)
                       & ~(m_df["is_outlier"] | m_df["touches_border"])
                       & (m_df["mass_pg"] >= 10.0)].sort_values("frame")
            if len(sub) < 4:
                continue
            tb = sub["time_h"].to_numpy(float) - float(sub["time_h"].iloc[0])
            met = _metrics(sub)
            idx = np.round(tb / BINW).astype(int)
            for j, t in enumerate(tb):
                if 0 <= idx[j] < nb and t <= NORMAL_TMAX:
                    for k, _ in METRICS:
                        acc[k][idx[j]].append(met[k][j])
            n_cyc += 1
    out = {}
    for k, _ in METRICS:
        mean = np.array([np.mean(b) if len(b) >= MIN_N else np.nan for b in acc[k]])
        sd = np.array([np.std(b, ddof=1) if len(b) >= MIN_N else np.nan for b in acc[k]])
        out[k] = (bins, mean, sd)
    return out, n_cyc


def elong_traces():
    """Per elongation lineage: time from onset + metrics over [onset, peak]."""
    traces = []
    for ch_key in ELONG_CH:
        pos, ch = ch_key.split("_", 1)
        df = pd.read_csv(resolve_lineage_csv(pos, ch))
        m = df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)
        onset, peak, _ = find_onset_peak(m)
        sub = m[(m["frame"] >= onset) & (m["frame"] <= peak)
                & ~(m["is_outlier"] | m["touches_border"])
                & (m["mass_pg"] >= 10.0)].sort_values("frame")
        t0 = float(sub["time_h"].iloc[0])
        traces.append((ch_key, sub["time_h"].to_numpy(float) - t0, _metrics(sub)))
    return traces


def main():
    normal, n_cyc = pool_normal()
    traces = elong_traces()
    print(f"normal gold cycles pooled={n_cyc}; elong lineages={len(traces)}")

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 62 / 25.4),
                             constrained_layout=True)
    data = {}
    for ax, (key, ylabel) in zip(axes, METRICS):
        bins, mean, sd = normal[key]
        ok = ~np.isnan(mean)
        ax.fill_between(bins[ok], (mean - sd)[ok], (mean + sd)[ok],
                        color=NORMAL_C, alpha=0.30, linewidth=0)
        ax.plot(bins[ok], mean[ok], color=NORMAL_C, lw=1.8,
                label=f"normal (n_cyc={n_cyc})")
        data[f"normal_{key}_t"] = bins[ok]
        data[f"normal_{key}_mean"] = mean[ok]
        data[f"normal_{key}_sd"] = sd[ok]
        for (ch_key, tb, met), col in zip(traces, ELONG_C):
            sel = tb <= TMAX
            ax.plot(tb[sel], met[key][sel], color=col, lw=0.9,
                    label=ch_key + " (elong)")            # raw, no smoothing
            data[f"{ch_key}_{key}_t"] = tb[sel]
            data[f"{ch_key}_{key}"] = met[key][sel]
        ax.set_xlabel("time from birth [h]")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, TMAX)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="upper right", frameon=False, fontsize=5.5)

    # density slope comparison over the first 2 h
    def _slope(t, y, tmax=2.0):
        m = (t <= tmax) & np.isfinite(y)
        return float(np.polyfit(t[m], y[m], 1)[0]) if m.sum() >= 3 else np.nan
    bN, mN, _ = normal["density"]
    sN = _slope(bN, mN)
    print(f"density slope first 2h [mg/mL/h]: normal={sN:.2f}")
    for ch_key, tb, met in traces:
        print(f"  {ch_key} (raw): {_slope(tb, met['density']):.2f}")

    caption = (
        "Intracellular density declines during terminal elongation in dying "
        "lineages, mirroring the normal cell cycle but without the division-coupled "
        "recovery. (A) volume, (B) dry-mass concentration (density), (C) dry mass, "
        "vs time from birth. Operational definitions: time from birth = time_h − "
        "time_h at the cycle's first (birth) frame; for normal cells birth = "
        "post-division cycle start, for elongation lineages birth = the data-driven "
        "post-division valley (last division before the non-dividing filament). "
        "volume = EFD rotational estimate (outline → elliptic-Fourier smoothing → "
        "solid-of-revolution integral of medial-axis perpendicular sections; column "
        "volume_um3_rod of the EFD corrected lineage). dry mass = (λ/2πα)·∬φ dA "
        "(phase area integral; column mass_pg, EFD). density = 1000·mass_pg/"
        "volume_um3_rod (pg/µm³ = g/mL → mg/mL). normal curve = gold-standard "
        f"survivor cycle frames binned by time from birth ({BINW} h bins), mean per "
        f"bin with ≥{MIN_N} cycles, plotted 0–{NORMAL_TMAX:.0f} h. Visual elements: "
        "cyan line = normal mean, cyan band = ±1 SD across cycles; red = Pos20_ch06, "
        "orange = Pos30_ch04 (elongation cascades, raw per-frame, no smoothing), "
        f"plotted 0–{TMAX:.0f} h. Error bars: cyan band = ±1 SD across cycles; "
        f"elongation = individual lineages (no band). n: normal = {n_cyc} cycles "
        "from 26 mother cells; elongation = 2 lineages. Statistics: descriptive, no "
        "test (n=2 elongation). Conditions: Schizosaccharomyces pombe, 260517 "
        "Mother-Machine dataset, EMM + 2% glucose, phase1 (frame ≤ 2018), QPI EFD "
        "volume variant [strain/genotype and temperature: confirm from acquisition "
        "metadata]. Abbreviations: SD, standard deviation; EFD, elliptic Fourier "
        "descriptor. Data availability: source data in this run's *_data.npz / "
        "*_data.csv. See also: cycle-aligned panels (fig_panelG_survivors)."
    )

    save_figure(
        fig,
        params={"volume_variant": "efd", "t_max_h": TMAX, "bin_h": BINW,
                "normal_cohort": "gold-standard survivors, cycles pooled by time-from-birth",
                "elong_channels": ELONG_CH, "n_normal_cycles": int(n_cyc),
                "density_slope_first2h_normal": sN},
        description="volume / density / mass vs time from birth: elongation-cascade "
                    "lineages (red) vs normally-dividing gold survivors (cyan band), "
                    "EFD — compares the density decline rate (cdc25-vs-wildtype style)",
        caption=caption,
        data=data,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
