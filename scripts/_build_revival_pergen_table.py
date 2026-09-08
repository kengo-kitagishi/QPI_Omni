"""_build_revival_pergen_table.py — shared per-(lineage, cycle) table for revival.

Builds one tidy table over the 260517 revived mothers: one row per division-bounded
cell cycle (phase1 and post-recovery), with birth/division mass, volume, RI, cycle
duration, per-cycle d ln M/dt and d ln V/dt, fold changes, density, plus
generation-after-recovery and per-lineage attributes (arrested size, phase1 medians,
starvation cycle-phase). Downstream hypothesis tests all read THIS table so every
analysis uses identical definitions.

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_build_revival_pergen_table.py
Output: results/260517/_revival_pergen.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overlay_mean_sd_band_full_timecourse import list_revived_mothers
from qpi_paths import resolve_lineage_csv, find_corrected_lineage_csv
from _fig_panelA_cellcycle import enumerate_all_cycles

PHASE1_END = 2018
STARV_FRAME = 2019
RECOVERY_FRAME = 2885
FPH = 12.0


def load_mother(pos, ch):
    df = pd.read_csv(resolve_lineage_csv(pos, ch))
    return df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)


def main():
    revived = [pc for pc in list_revived_mothers()
               if find_corrected_lineage_csv(*pc) is not None]
    rows = []
    for pos, ch in revived:
        ch_key = f"{pos}_{ch}"
        m = load_mother(pos, ch)
        cyc, _ = enumerate_all_cycles(m, set(), max_frame=3747, min_kept=6)
        recs = []
        for c in cyc:
            v = c[~(c["is_outlier"] | c["touches_border"]) & (c["mass_pg"] >= 10.0)
                  & (c["volume_um3_rod"] > 0)]
            if len(v) < 4:
                continue
            f0 = int(c["frame"].min()); f1 = int(c["frame"].max())
            t = v["time_h"].to_numpy(float)
            mass = v["mass_pg"].to_numpy(float)
            vol = v["volume_um3_rod"].to_numpy(float)
            ri = v["mean_ri"].to_numpy(float)
            km = float(np.polyfit(t, np.log(mass), 1)[0])
            kv = float(np.polyfit(t, np.log(vol), 1)[0])
            epoch = ("phase1" if f1 <= PHASE1_END
                     else "post" if f0 >= RECOVERY_FRAME else "transition")
            recs.append(dict(
                ch=ch_key, epoch=epoch, birth_frame=f0, div_frame=f1,
                interval_h=(f1 - f0) / FPH,
                birth_mass=float(mass[0]), div_mass=float(mass[-1]),
                birth_vol=float(vol[0]), div_vol=float(vol[-1]),
                birth_ri=float(ri[0]), div_ri=float(ri[-1]),
                added_mass=float(mass[-1] - mass[0]),
                added_vol=float(vol[-1] - vol[0]),
                mass_fold=float(mass[-1] / mass[0]),
                vol_fold=float(vol[-1] / vol[0]),
                k_mass=km, k_vol=kv, decouple_k=km - kv,
                ri_peak=float(ri.max()), ri_rise=float(ri.max() - ri[0]),
                birth_dens=float(1000.0 * mass[0] / vol[0]),
                n_frames=len(v)))
        if not recs:
            continue
        rdf = pd.DataFrame(recs).sort_values("birth_frame").reset_index(drop=True)
        # generation after recovery (1-based) for post cycles
        post = rdf[rdf["epoch"] == "post"].index.tolist()
        for g, idx in enumerate(post, 1):
            rdf.loc[idx, "gen_rec"] = g
        p1 = rdf[rdf["epoch"] == "phase1"]
        if p1.empty or not post:
            continue
        # lineage-level attributes
        last_p1_div = int(p1["div_frame"].max())
        attrs = dict(
            arrested_mass=float(rdf.loc[post[0], "birth_mass"]),
            arrested_vol=float(rdf.loc[post[0], "birth_vol"]),
            p1_birth_mass=float(p1["birth_mass"].median()),
            p1_birth_vol=float(p1["birth_vol"].median()),
            p1_birth_ri=float(p1["birth_ri"].median()),
            p1_interval_h=float(p1["interval_h"].median()),
            p1_k_mass=float(p1["k_mass"].median()),
            starv_phase_h=(STARV_FRAME - last_p1_div) / FPH)
        for k, val in attrs.items():
            rdf[k] = val
        rows.append(rdf)
    out = pd.concat(rows, ignore_index=True)
    out_path = Path("results/260517/_revival_pergen.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    n_lin = out["ch"].nunique()
    post = out[out["epoch"] == "post"]
    print(f"saved {out_path}  rows={len(out)}  lineages={n_lin}")
    print(f"  phase1 cycles={int((out['epoch']=='phase1').sum())}  "
          f"post cycles={len(post)}  max gen_rec={int(post['gen_rec'].max())}")
    print(f"  columns: {list(out.columns)}")
    # quick per-generation medians (sanity)
    print("  gen_rec medians (interval_h, birth_mass, birth_ri, decouple_k):")
    for g in range(1, 9):
        sub = post[post["gen_rec"] == g]
        if len(sub) >= 5:
            print(f"    gen{g}: n={len(sub):2d}  int={sub['interval_h'].median():.2f}  "
                  f"bmass={sub['birth_mass'].median():.1f}  "
                  f"bri={sub['birth_ri'].median():.4f}  "
                  f"dk={sub['decouple_k'].median():+.3f}")


if __name__ == "__main__":
    main()
