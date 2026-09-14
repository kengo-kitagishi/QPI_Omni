"""Individual mother trajectories (one line per cell) over the FULL timecourse
for the phase2 survivors (revived mothers), using the adopted EFD volume.

Volume and mean RI are read per mother frame from the full-timecourse EFD
recompute (results/260517/recomputed_axes_efd_full/<pos>_<ch>.csv, medial_axis
rows: volume_efd_um3, mean_ri_efd, mass_pg_efd, time_h). Bad frames
(is_outlier / touches_border from the original lineage, or EFD mass < 10 pg)
are masked to NaN so each line breaks at bad frames.

Two stacked panels (volume top, mean RI bottom), individual thin lines, with the
media-schedule phase boundaries marked.

Usage:
    python scripts/phase2_survivor_efd_traces.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from qpi_paths import results_dir, find_lineage_csv  # noqa: E402
from figure_logger import save_figure  # noqa: E402
import overlay_mean_sd_band_full_timecourse as ov2  # noqa: E402

REC = results_dir() / "recomputed_axes_efd_full"
TIME_INTERVAL_MIN = 5.0
# media-schedule boundaries (frame -> time_h)
PHASE_MARKS = [
    (2019 * TIME_INTERVAL_MIN / 60.0, "0.0055%"),
    (2307 * TIME_INTERVAL_MIN / 60.0, "0%"),
    (2885 * TIME_INTERVAL_MIN / 60.0, "2% recovery"),
]
LOW_MASS_PG = 10.0


def load_efd_full(pos: str, ch: str):
    """(time_h, volume_efd, mean_ri_efd) for the mother, bad frames -> NaN."""
    rp = REC / f"{pos}_{ch}.csv"
    if not rp.exists():
        return None
    rec = pd.read_csv(rp)
    rec = rec[rec["mode"] == "medial_axis"].sort_values("frame")
    if rec.empty:
        return None
    vol = rec["volume_efd_um3"].to_numpy(dtype=float)
    ri = rec["mean_ri_efd"].to_numpy(dtype=float)
    mass = rec["mass_pg_efd"].to_numpy(dtype=float)
    bad = ~np.isfinite(vol) | (mass < LOW_MASS_PG)
    # original-lineage bad-frame flags (outlier / border) mapped onto frames
    lin_path = find_lineage_csv(pos, ch)
    if lin_path is not None:
        lin = pd.read_csv(lin_path)
        lm = lin[lin["rank"] == 1][["frame", "is_outlier", "touches_border"]]
        flag = rec.merge(lm, on="frame", how="left")
        bad = bad | flag["is_outlier"].fillna(False).to_numpy(bool) \
                  | flag["touches_border"].fillna(False).to_numpy(bool)
    vol = vol.copy(); ri = ri.copy()
    vol[bad] = np.nan
    ri[bad] = np.nan
    return rec["time_h"].to_numpy(dtype=float), vol, ri


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", default=None,
                    help="explicit Pos_ch list (default: all revived survivors)")
    args = ap.parse_args()
    if args.channels:
        cohort = [tuple(t.split("_", 1)) for t in args.channels]
    else:
        cohort = ov2.list_revived_mothers()
    import matplotlib.pyplot as plt
    fig, (ax_v, ax_r) = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True)

    n = 0
    t_max = 0.0
    cmap = plt.cm.viridis(np.linspace(0, 1, len(cohort)))
    for (pos, ch), col in zip(cohort, cmap):
        tr = load_efd_full(pos, ch)
        if tr is None:
            continue
        t, vol, ri = tr
        ax_v.plot(t, vol, color=col, lw=0.5, alpha=0.55, solid_capstyle="butt")
        ax_r.plot(t, ri, color=col, lw=0.5, alpha=0.55, solid_capstyle="butt")
        n += 1
        t_max = max(t_max, np.nanmax(t))

    for ax, ylab, ylim in [(ax_v, r"mother volume [$\mu m^3$]  (EFD)", (0, 300)),
                           (ax_r, "mother mean RI  (EFD)", (1.345, 1.41))]:
        for x, lab in PHASE_MARKS:
            ax.axvline(x, color="k", ls="--", lw=0.5, alpha=0.5)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, t_max)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(labelsize=7)
    # phase labels on top panel
    ymax = ax_v.get_ylim()[1]
    bounds = [0] + [x for x, _ in PHASE_MARKS] + [t_max]
    labels = ["2% (phase1)", "0.0055%", "0%", "2% recovery"]
    for x0, x1, lab in zip(bounds[:-1], bounds[1:], labels):
        ax_v.text((x0 + x1) / 2, ymax * 0.97, lab, ha="center", va="top",
                  fontsize=6, color="k", alpha=0.6)
    ax_r.set_xlabel("time [h]", fontsize=8)
    ax_v.set_title(f"Phase2 survivors (revived mothers, n={n}) — individual "
                   f"EFD volume & mean RI over full timecourse", fontsize=8.5)
    fig.tight_layout()
    save_figure(
        fig,
        params={"cohort": "revived_phase2_survivors", "n_cells": n,
                "volume_method": "efd", "efd_k": 6,
                "rec_dir": "recomputed_axes_efd_full"},
        description=("Individual mother trajectories (revived phase2 survivors) "
                     "of EFD volume and mean RI over the full media-switch "
                     "timecourse (2% -> 0.0055% -> 0% -> 2% recovery)."),
    )
    plt.close(fig)
    print(f"plotted {n}/{len(cohort)} revived survivors")


if __name__ == "__main__":
    main()
