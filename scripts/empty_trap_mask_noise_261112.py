"""Density (concentration) fluctuation floor: 261108 mother-cell masks applied to the cell-free 261112 traps.

The 261112 timelapse (2% EMM, no cells yet, 5 min/frame) and 261108 run 3 (YE, cells) use the same trap crop
geometry (40 x 240 px, crop_h 270, POS_SPLIT 60), so an Omnipose mask of a 261108 mother can be placed at the same
pixel coordinates in a 261112 trap. Masks from Pos < 60 lineages go on Pos < 60 traps and vice versa (the trap
opening is on the other side in the second row).

For every empty trap x template: mass_t = sum of the crop_sub phase inside the mask x MASS_PER_RAD (the tracker's
formula: lambda * px^2 / (2 pi alpha) * 1e-3 pg per rad), conc_t = mass_t / volume_um3_efd of that template x 1000
mg/mL. Because the template is fixed, the temporal SD of conc_t is the measurement floor of the per-frame density,
and the temporal mean is the systematic offset of that trap (what would show up as a between-lineage difference).
The temporal autocorrelation of mass_t (own mean subtracted) tells whether the floor is white or slowly drifting.

    C:/Users/QPI/anaconda3/envs/omnipose/python.exe cache_empty_traps_261112.py <scratch>   # once
    C:/Users/QPI/anaconda3/envs/omnipose/python.exe empty_trap_mask_noise_261112.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from figure_logger import save_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

EMPTY_RAW = Path("D:/AquisitionData/Kitagishi/261112/online_crop_sub_2per_5min_31d_1")
MASK_ROOT = Path("C:/QPI_analysis/261108/seg")
CONS = Path(r"C:\QPI_analysis\261108\seg\_lineage_consolidated\all_cells_lineage_data3D.csv.gz")
EXCLUDED = Path(r"C:\QPI_analysis\261108\inputs\excluded_traps.csv")
REL = Path("output_phase/channels/crop_sub_rawraw/z000")
POS_SPLIT = 60                      # both sessions
SKIP_POS = {"Pos124"}               # 261112: only 5 traps, field edge
TEMPLATE_STEP = 50                  # frames between templates taken from one lineage
EMPTY_P999_MAX_RAD = 0.5            # a trap whose central region (99.9th pct, median over frames) exceeds this is not empty
PX_UM = 0.34567514677103717
WL_UM = 0.658
ALPHA_RI = 0.00018
MASS_PER_RAD = WL_UM * PX_UM ** 2 / (2 * np.pi) / ALPHA_RI * 1e-3   # pg per rad (summed over pixels)
MAX_LAG = 10
DT_H = 5.0 / 60.0

# 261108 run 3 mothers, same selection as fig_autocorr_261108 (reference values, for the vertical lines)
REF_WITHIN_LINEAGE_SD_CONC = None   # computed below from the consolidated table
REF_BETWEEN_LINEAGE_SD_CONC = None


def templates():
    f = pd.read_csv(CONS, usecols=["pos", "ch", "cell_id", "frame", "mask_label", "volume_um3_efd", "mass_pg",
                                   "density_pg_um3", "is_outlier", "touches_border"])
    f = f[(f.cell_id == 0) & f.frame.between(2, 769) & ~f.is_outlier.astype(bool) & ~f.touches_border.astype(bool)]
    ex = set(map(tuple, pd.read_csv(EXCLUDED)[["pos", "ch"]].values))
    f = f[[(p, c) not in ex for p, c in zip(f.pos, f.ch)]]
    f = f[f.groupby(["pos", "ch"]).frame.transform("size") >= 100]
    g = f.groupby(["pos", "ch"])
    ref = {
        "n_lineages": int(g.ngroups),
        "mean_mass_pg": float(f.mass_pg.mean()),
        "mean_conc_mg_ml": float(f.density_pg_um3.median() * 1000),
        "within_lineage_sd_conc_mg_ml": float((f.density_pg_um3 - g.density_pg_um3.transform("median")).std() * 1000),
        "frame_diff_sd_conc_mg_ml": float(g.density_pg_um3.diff().std() / np.sqrt(2) * 1000),
        "between_lineage_sd_conc_mg_ml": float(g.density_pg_um3.median().std() * 1000),
        "within_lineage_sd_mass_pg": float((f.mass_pg - g.mass_pg.transform("mean")).std()),
        "frame_diff_sd_mass_pg": float(g.mass_pg.diff().std() / np.sqrt(2)),
    }
    out = []
    for (pos, ch), gg in g:
        d = MASK_ROOT / pos / REL / ch / "inference_out"
        for _, r in gg.iloc[::TEMPLATE_STEP].iterrows():
            mp = d / f"img_{int(r.frame):09d}_ph_000_masks.tif"
            if not mp.exists():
                continue
            cell = tifffile.imread(str(mp)) == int(r.mask_label)
            if cell.sum() < 40:
                continue
            out.append({"pos": pos, "ch": ch, "frame": int(r.frame), "row": 0 if int(pos[3:]) < POS_SPLIT else 1,
                        "mask": cell, "area_px": int(cell.sum()), "vol": float(r.volume_um3_efd),
                        "mass_pg_261108": float(r.mass_pg)})
    return out, ref


CACHE = Path(r"C:/Users/QPI/AppData/Local/Temp/claude/c--Users-QPI-Documents-QPI-Omni/743a73a1-719e-44ea-ab78-692d0a0fb5bd/scratchpad/empty261112")


def empty_traps():
    """Yield (pos, ch, stack[frames, 40, 240], frame_ids) from the memmap cache written by cache_empty_traps_261112.py
    (one read of the 33k tifs on D:)."""
    idx = pd.read_csv(CACHE / "index.csv")
    mm = np.load(CACHE / "stacks.npy", mmap_mode="r")
    for _, r in idx.iterrows():
        if r.pos in SKIP_POS or r.pos == "Pos0":
            continue
        st = np.asarray(mm[int(r.i)])
        yield r.pos, r.ch, st, np.arange(st.shape[0])


def own_mean_acf(x, max_lag):
    d = x - x.mean()
    v = (d * d).mean()
    out = np.full(max_lag + 1, np.nan)
    for k in range(min(max_lag, len(x) - 1) + 1):
        out[k] = (d[: len(x) - k] * d[k:]).mean() / v if v > 0 else np.nan
    return out


def main():
    tpl, ref = templates()
    print(f"{len(tpl)} templates from {ref['n_lineages']} lineages; MASS_PER_RAD={MASS_PER_RAD:.5f} pg/rad")
    rows, acfs, traces, first_tpl = [], [], {}, {}
    n_traps = n_not_empty = 0
    for pos, ch, stack, frames in empty_traps():
        n_traps += 1
        centre = stack[:, 8:32, 20:220]
        p999 = np.percentile(centre.reshape(len(stack), -1), 99.9, axis=1)
        if np.median(p999) > EMPTY_P999_MAX_RAD:
            n_not_empty += 1
            continue
        row = 0 if int(pos[3:]) < POS_SPLIT else 1
        same = [t for t in tpl if t["row"] == row]
        flat = stack.reshape(len(stack), -1)
        acf_trap = []
        for t in same:
            mass = flat[:, t["mask"].ravel()].sum(axis=1) * MASS_PER_RAD          # pg per frame
            conc = mass / t["vol"] * 1000                                          # mg/mL per frame
            rows.append({"pos": pos, "ch": ch, "row": row, "tpl": f"{t['pos']}_{t['ch']}_f{t['frame']}",
                         "area_px": t["area_px"], "vol_um3": t["vol"],
                         "mass_mean_pg": mass.mean(), "mass_sd_pg": mass.std(ddof=1),
                         "mass_diff_sd_pg": np.diff(mass).std(ddof=1) / np.sqrt(2),
                         "conc_mean": conc.mean(), "conc_sd": conc.std(ddof=1),
                         "conc_diff_sd": np.diff(conc).std(ddof=1) / np.sqrt(2), "n_frames": len(mass)})
            acf_trap.append(own_mean_acf(mass, MAX_LAG))
            if t is same[0]:
                first_tpl[(pos, ch)] = conc
                if len(traces) < 6:
                    traces[f"{pos} {ch}"] = conc
        acfs.append(np.nanmean(acf_trap, axis=0))
    r = pd.DataFrame(rows)
    acfs = np.array(acfs)
    print(f"traps: {n_traps}, flagged not empty: {n_not_empty}, used: {r.groupby(['pos', 'ch']).ngroups}")
    trap = r.groupby(["pos", "ch", "row"]).agg(conc_sd=("conc_sd", "median"), conc_mean=("conc_mean", "median"),
                                              mass_sd=("mass_sd_pg", "median"), mass_mean=("mass_mean_pg", "median"),
                                              conc_diff_sd=("conc_diff_sd", "median")).reset_index()
    res = {"n_templates": len(tpl), "n_traps_total": n_traps, "n_traps_not_empty": n_not_empty,
           "n_traps_used": int(len(trap)), "n_frames": int(r.n_frames.min()), "ref_261108": ref}
    for rw, lab in ((None, "all"), (0, "Pos<60"), (1, "Pos>=60")):
        tt = trap if rw is None else trap[trap.row == rw]
        res[lab] = {
            "n_traps": int(len(tt)),
            "conc_temporal_sd_median_mg_ml": float(tt.conc_sd.median()),
            "conc_temporal_sd_iqr_mg_ml": [float(tt.conc_sd.quantile(.25)), float(tt.conc_sd.quantile(.75))],
            "conc_frame_diff_sd_median_mg_ml": float(tt.conc_diff_sd.median()),
            "conc_offset_between_trap_sd_mg_ml": float(tt.conc_mean.std()),
            "conc_offset_median_mg_ml": float(tt.conc_mean.median()),
            "conc_offset_iqr_mg_ml": [float(tt.conc_mean.quantile(.25)), float(tt.conc_mean.quantile(.75))],
            "mass_temporal_sd_median_pg": float(tt.mass_sd.median()),
            "mass_offset_between_trap_sd_pg": float(tt.mass_mean.std()),
        }
    # common mode within a Pos: subtract, frame by frame, the mean of the other traps of the same Pos
    by_pos = {}
    for (pos, ch), c in first_tpl.items():
        by_pos.setdefault(pos, []).append(c)
    resid_sd, raw_sd = [], []
    for pos, lst in by_pos.items():
        if len(lst) < 3:
            continue
        arr = np.array(lst)
        for i in range(len(arr)):
            others = np.delete(arr, i, axis=0).mean(axis=0)
            raw_sd.append(arr[i].std(ddof=1)); resid_sd.append((arr[i] - others).std(ddof=1))
    res["first_template_conc_sd_median_mg_ml"] = float(np.median(raw_sd))
    res["first_template_conc_sd_after_pos_common_mode_removed_median_mg_ml"] = float(np.median(resid_sd))
    acf_mean, acf_sem = np.nanmean(acfs, axis=0), np.nanstd(acfs, axis=0, ddof=1) / np.sqrt(len(acfs))
    res["acf_lag1"] = float(acf_mean[1])
    res["acf_bias_own_mean"] = -1.0 / res["n_frames"]
    # noise scaling with mask area (white noise: SD ~ sqrt(area); fully correlated background: SD ~ area)
    ar = r.groupby("tpl").agg(area=("area_px", "first"), sd=("mass_sd_pg", "median")).sort_values("area")
    slope = np.polyfit(np.log(ar.area), np.log(ar.sd), 1)[0]
    res["log_slope_mass_sd_vs_area"] = float(slope)
    print(json.dumps(res, indent=1, default=float))

    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 110 / 25.4), constrained_layout=True)
    ax = axes[0, 0]
    t_h = np.arange(res["n_frames"]) * DT_H
    for i, (k, v) in enumerate(traces.items()):
        ax.plot(t_h[: len(v)], v, lw=0.7, label=k)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlabel("time [h]"); ax.set_ylabel("apparent conc. [mg/mL]")
    ax.set_title("empty traps, first template", fontsize=7)
    ax.legend(fontsize=5, ncol=2, frameon=False)
    ax = axes[0, 1]
    ax.hist(trap.conc_sd, bins=40, color="#0072B2")
    ax.axvline(ref["within_lineage_sd_conc_mg_ml"], color="#D55E00", lw=0.8, label="261108 within-lineage SD")
    ax.axvline(ref["frame_diff_sd_conc_mg_ml"], color="#D55E00", lw=0.8, ls="--", label="261108 frame-diff SD/\u221a2")
    ax.set_xlabel("temporal SD [mg/mL]"); ax.set_ylabel("traps"); ax.set_xlim(0, None)
    ax.legend(fontsize=5, frameon=False)
    ax = axes[0, 2]
    ax.hist(trap.conc_mean, bins=40, color="#0072B2")
    ax.axvline(0, color="0.5", lw=0.5)
    ax.set_xlabel("temporal mean (offset) [mg/mL]"); ax.set_ylabel("traps")
    ax.set_title(f"between-trap SD {res['all']['conc_offset_between_trap_sd_mg_ml']:.1f} mg/mL\n"
                 f"(261108 between-lineage SD {ref['between_lineage_sd_conc_mg_ml']:.1f})", fontsize=6.5)
    ax = axes[1, 0]
    lags = np.arange(MAX_LAG + 1) * DT_H
    ax.errorbar(lags, acf_mean, yerr=acf_sem, fmt="o-", ms=2, lw=0.7, color="#0072B2")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.axhline(res["acf_bias_own_mean"], color="0.5", lw=0.5, ls=":")
    ax.set_xlabel("lag [h]"); ax.set_ylabel("autocorr. of masked mass")
    ax.set_title("own mean subtracted; dotted: -1/L", fontsize=7)
    ax = axes[1, 1]
    ax.plot(ar.area, ar.sd, "o", ms=2.5, color="#0072B2", alpha=0.6, rasterized=True)
    xx = np.linspace(ar.area.min(), ar.area.max(), 50)
    ax.plot(xx, ar.sd.median() * np.sqrt(xx / ar.area.median()), "--", lw=0.6, color="0.4", label="\u221aarea")
    ax.plot(xx, ar.sd.median() * (xx / ar.area.median()), ":", lw=0.6, color="0.4", label="area")
    ax.set_xlabel("mask area [px]"); ax.set_ylabel("SD of masked mass [pg]"); ax.set_ylim(0, None)
    ax.set_title(f"log slope {slope:.2f}", fontsize=7); ax.legend(fontsize=5, frameon=False)
    ax = axes[1, 2]
    pidx = trap.pos.str[3:].astype(int); cidx = trap.ch.str[2:].astype(int)
    sc = ax.scatter(pidx, cidx, c=trap.conc_mean, cmap="cividis", s=6, vmin=np.percentile(trap.conc_mean, 2),
                    vmax=np.percentile(trap.conc_mean, 98), rasterized=True)
    ax.axvline(POS_SPLIT - 0.5, color="0.5", lw=0.5)
    ax.set_xlabel("Pos"); ax.set_ylabel("trap (ch)")
    fig.colorbar(sc, ax=ax, label="offset [mg/mL]", shrink=0.8)
    for ax_, letter in zip(axes.ravel(), "abcdef"):
        ax_.text(-0.25, 1.04, letter, transform=ax_.transAxes, fontweight="bold", fontsize=8)
    out_csv = SCRIPTS.parent / "results" / "empty_trap_mask_noise_261112_per_trap.csv"
    out_csv.parent.mkdir(exist_ok=True)
    trap.to_csv(out_csv, index=False)
    save_figure(
        fig, fmt="png", dpi=300,
        params={"empty_raw": str(EMPTY_RAW), "mask_root": str(MASK_ROOT), "template_step": TEMPLATE_STEP,
                "empty_p999_max_rad": EMPTY_P999_MAX_RAD, "mass_per_rad_pg": MASS_PER_RAD, "max_lag": MAX_LAG},
        extra_meta=res,
        description="Density fluctuation floor: 261108 mother masks applied to the cell-free 261112 traps.",
        data={"trap_conc_sd": trap.conc_sd.to_numpy(), "trap_conc_mean": trap.conc_mean.to_numpy(),
              "trap_mass_sd": trap.mass_sd.to_numpy(), "trap_mass_mean": trap.mass_mean.to_numpy(),
              "trap_pos": pidx.to_numpy(), "trap_ch": cidx.to_numpy(), "trap_row": trap.row.to_numpy(),
              "acf_per_trap": acfs, "acf_mean": acf_mean, "acf_sem": acf_sem, "lag_h": lags,
              "tpl_area_px": ar.area.to_numpy(), "tpl_mass_sd_pg": ar.sd.to_numpy(),
              **{f"trace_{k.replace(' ', '_')}": v for k, v in traces.items()}},
        caption=(f"Cell-free 261112 timelapse (2% EMM, 5 min/frame, {res['n_frames']} frames, {res['n_traps_used']} traps; "
                 f"{n_not_empty} traps whose trap-centre 99.9th percentile phase (median over frames) exceeds {EMPTY_P999_MAX_RAD} rad dropped, Pos124 "
                 f"dropped). {len(tpl)} Omnipose mother masks of the {ref['n_lineages']} analysed 261108 run 3 lineages "
                 f"(every {TEMPLATE_STEP}th frame) placed at their own pixel coordinates in traps of the same row "
                 f"(Pos<60 / >=60). Apparent mass = sum of crop_sub phase in the mask x {MASS_PER_RAD:.4f} pg/rad "
                 "(tracker formula, alpha 0.00018, 658 nm); apparent concentration = that / the template's "
                 "volume_um3_efd x 1000. (a) time course for the first template in six traps. (b) per trap: median over "
                 "templates of the temporal SD of apparent concentration; vermillion lines = 261108 mothers' within-"
                 "lineage SD and frame-difference SD/sqrt2 of real concentration. (c) per trap temporal mean (offset). "
                 "(d) autocorrelation of masked mass, each trap centred on its own mean (bias -1/L dotted), mean +- SEM "
                 "over traps. (e) temporal SD of masked mass vs mask area, one point per template (median over traps); "
                 "dashed sqrt(area) = white noise, dotted area = fully correlated background. (f) offset per trap over "
                 "the chip. No statistical test. Per-trap table: results/empty_trap_mask_noise_261112_per_trap.csv."),
    )


if __name__ == "__main__":
    main()
