"""Preview how much the EFD volume changes the overlay figure vs current rod.

Reads results/260517/recomputed_axes_efd/<pos>_<ch>.csv (medial_axis rows carry
volume_um3_rod = current method, volume_profile_um3 = cos-theta, and
volume_efd_um3 = adopted EFD method, all per mother frame with time_h). Plots
volume vs time for the preview channels: rod (left) vs EFD (right), gold gray +
dead colored, and prints the per-channel EFD/rod median ratio.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from qpi_paths import results_dir  # noqa: E402
import overlay_gold_standard_and_phase1_dead as ov  # noqa: E402

REC = results_dir() / "recomputed_axes_efd"
PHASE1_T_MAX = 168.25


def preview_channels():
    gold = ov.select_gold_standard()[:3]
    dead = [(p, c) for p, c, _ in ov.select_phase1_dead_sorted_by_death()]
    pick_dead = [dead[0], dead[len(dead) // 2], dead[-1]] if len(dead) >= 3 else dead
    return list(gold), pick_dead


def load(pos, ch):
    p = REC / f"{pos}_{ch}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df["mode"] == "medial_axis"].sort_values("frame")
    df = df[df["time_h"] <= PHASE1_T_MAX]
    if df.empty:
        return None
    return df


def main():
    gold, dead = preview_channels()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True, sharey=True)
    palette = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00"]

    print(f"{'channel':14} {'group':5} {'n':>5} {'rod_med':>8} {'efd_med':>8} {'efd/rod':>8}")
    for col, vcol, name in [(0, "volume_um3_rod", "current (rod)"),
                            (1, "volume_efd_um3", "EFD adopted")]:
        ax = axes[col]
        for pos, ch in gold:
            df = load(pos, ch)
            if df is None:
                continue
            ax.plot(df["time_h"], df[vcol], color="#4a4a4a", lw=0.4, alpha=0.5, zorder=1)
        for i, (pos, ch) in enumerate(dead):
            df = load(pos, ch)
            if df is None:
                continue
            ax.plot(df["time_h"], df[vcol], color=palette[i % len(palette)],
                    lw=1.0, alpha=0.9, zorder=2, label=f"{pos}_{ch}")
        ax.set_xlim(0, PHASE1_T_MAX); ax.set_ylim(0, 300)
        ax.set_xlabel("time [h]", fontsize=9)
        ax.set_title(name, fontsize=10)
        if col == 0:
            ax.set_ylabel(r"mother volume [$\mu m^3$]", fontsize=9)
        ax.legend(loc="upper right", fontsize=6, frameon=False)

    # ratio report
    for grp, chans in [("gold", gold), ("dead", dead)]:
        for pos, ch in chans:
            df = load(pos, ch)
            if df is None:
                print(f"{pos+'_'+ch:14} {grp:5} {'--':>5}")
                continue
            rod = df["volume_um3_rod"].to_numpy()
            efd = df["volume_efd_um3"].to_numpy()
            ok = np.isfinite(rod) & np.isfinite(efd) & (rod > 0)
            ratio = np.median(efd[ok] / rod[ok]) if ok.any() else np.nan
            print(f"{pos+'_'+ch:14} {grp:5} {ok.sum():>5} "
                  f"{np.nanmedian(rod):>8.1f} {np.nanmedian(efd):>8.1f} {ratio:>8.3f}")

    fig.suptitle("Preview: mother volume vs time — current rod (left) vs adopted EFD (right)",
                 fontsize=11)
    fig.tight_layout()
    out = Path("results/260517/_preview_efd_vs_rod.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
