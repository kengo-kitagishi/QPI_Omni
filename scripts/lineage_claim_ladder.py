"""Downstream lineage analyses for the claim ladder (paper/results_claim_map.md).

Consumes the tracker output pair per channel:
  <Pos>_<ch>_clist.csv          per-cell summary (mother_id / daughterN_id / generation)
  <Pos>_<ch>_lineage_data3D.csv per-cell x per-frame long format

and builds a per-cycle table for ALL in-tree cells (not just the mother):
a cell's divisions are the birth_frames of its children in clist, and its
trace is segmented at those frames. From that table it runs the analyses
of the claim ladder:

  media     cycle-mean density across media (box + lineage means)      [claim 2]
  phase     cycle-phase-normalised V / M / rho, media overlaid          [claim 3]
  memory    autocorrelation vs generation lag + exp fit tau + shuffle   [claim 4]
  control   return maps of V / M / rho + VAR(1) null band for rho       [claim 5]
  division  partition ratios r_V vs r_M, sibling density, variance split[claim 6]
  range     density vs birth volume over the full natural range         [claim 7]
  death     death-aligned trajectories + deviation-stratified fates     [claim 8]

Media-dependent thresholds (division-interval gate etc.) come from a config
dict; YE (~120 min doubling) and EMM (~195 min) are predefined and every
number can be overridden from a JSON file (--config).

Usage (smoke test on an existing EMM run):
  python3 scripts/lineage_claim_ladder.py run \
      --dataset EMM:results/mid_term_0405/data \
      --max-frame 2018 --out results/claim_ladder_smoke

Two media side by side (once the YE run is tracked):
  python3 scripts/lineage_claim_ladder.py run \
      --dataset EMM:results/<emm_run>/data --dataset YE:results/<ye_run>/data \
      --out results/claim_ladder

Every figure writes a companion CSV with the plotted numbers next to it
(provenance rule: figure -> CSV -> this script).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config

# Okabe-Ito; colour = medium, fixed across every figure of the paper.
MEDIUM_COLORS = {"EMM": "#0072B2", "YE": "#D55E00"}
FALLBACK_COLORS = ["#009E73", "#CC79A7", "#56B4E9", "#E69F00"]

DEFAULT_MEDIA = {
    "EMM": {
        "time_interval_min": 5.0,
        "expected_doubling_min": 195.0,
        # accepted cycle duration as fraction of expected doubling
        "cycle_min_frac": 0.5,
        "cycle_max_frac": 1.75,
    },
    "YE": {
        "time_interval_min": 5.0,
        "expected_doubling_min": 120.0,
        # YE divides faster; a segmentation blip costs a larger fraction of
        # the cycle, so the lower gate is slightly tighter.
        "cycle_min_frac": 0.55,
        "cycle_max_frac": 1.75,
    },
}

QC_DEFAULTS = {
    # cycle-level gates
    "max_outlier_frac": 0.25,      # frames flagged is_outlier within the cycle
    "max_border_frac": 0.25,       # frames touching the image border
    "min_valid_frames": 8,         # valid frames needed inside a cycle
    "boundary_search_frames": 3,   # nearest-valid fallback around birth/division
    # death call: no children AND trace ends this many frames before the
    # recording ends (otherwise it is just the end of the movie)
    "death_margin_frames": 24,
    "death_min_frames": 12,
}


@dataclass
class MediumConfig:
    name: str
    time_interval_min: float
    expected_doubling_min: float
    cycle_min_frac: float
    cycle_max_frac: float
    qc: dict = field(default_factory=lambda: dict(QC_DEFAULTS))

    @property
    def frame_h(self) -> float:
        return self.time_interval_min / 60.0

    @property
    def min_cycle_h(self) -> float:
        return self.expected_doubling_min / 60.0 * self.cycle_min_frac

    @property
    def max_cycle_h(self) -> float:
        return self.expected_doubling_min / 60.0 * self.cycle_max_frac


def load_medium_config(name: str, override_path: str | None) -> MediumConfig:
    base = dict(DEFAULT_MEDIA.get(name, DEFAULT_MEDIA["EMM"]))
    qc = dict(QC_DEFAULTS)
    if override_path:
        user = json.loads(Path(override_path).read_text())
        med = user.get("media", {}).get(name, {})
        qc.update(user.get("qc", {}))
        qc.update(med.pop("qc", {}))
        base.update(med)
    return MediumConfig(name=name, qc=qc, **base)


def medium_color(name: str) -> str:
    if name in MEDIUM_COLORS:
        return MEDIUM_COLORS[name]
    return FALLBACK_COLORS[hash(name) % len(FALLBACK_COLORS)]


# ---------------------------------------------------------------- loading

def discover_pairs(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """Find (<tag>, clist, data3D) triples in a flat results/data directory."""
    pairs = []
    for clist in sorted(data_dir.glob("*clist.csv")):
        stem = clist.name.replace("_clist.csv", "").replace("__clist.csv", "")
        for cand in (data_dir / f"{stem}_lineage_data3D.csv",
                     data_dir / f"{stem}__lineage_data3D.csv"):
            if cand.exists():
                pairs.append((stem, clist, cand))
                break
    return pairs


def find_false_births(clist: pd.DataFrame, data3D: pd.DataFrame) -> set[int]:
    """Same rule as lineage_survival_analysis.py: a daughter whose birth lands
    on, or right after, a parent outlier frame is a segmentation artefact."""
    outliers_by_cell = {
        int(cid): set(grp.loc[grp["is_outlier"], "frame"].astype(int))
        for cid, grp in data3D.groupby("cell_id")
    }
    false_ids: set[int] = set()
    for _, row in clist.iterrows():
        parent, birth = int(row["mother_id"]), int(row["birth_frame"])
        if parent < 0:
            continue
        out = outliers_by_cell.get(parent, set())
        if birth in out or (birth - 1) in out:
            false_ids.add(int(row["cell_id"]))
    return false_ids


def _valid_trace(grp: pd.DataFrame) -> pd.DataFrame:
    ok = (~grp["is_outlier"]) & (~grp["touches_border"])
    return grp.loc[ok]


def _value_near(trace: pd.DataFrame, frame: int, col: str, search: int) -> float:
    """Value at `frame`, falling back to the nearest valid frame within
    +/- search (the boundary frame itself is often an outlier)."""
    win = trace[(trace["frame"] >= frame - search) & (trace["frame"] <= frame + search)]
    if win.empty:
        return np.nan
    i = (win["frame"] - frame).abs().idxmin()
    return float(win.loc[i, col])


def build_cycles(tag: str, clist: pd.DataFrame, data3D: pd.DataFrame,
                 cfg: MediumConfig, max_frame: int | None) -> pd.DataFrame:
    """Per-cycle table for every in-tree cell.

    A cell's division events are the birth_frames of its children (clist
    mother_id relation; the daughterN_id columns hold only the first two).
    Cycle i of a cell spans [event_i, event_{i+1} - 1]; for a daughter the
    first event is its own birth. Only cycles with both boundaries observed
    enter the table (the trailing open interval is dropped).
    """
    qc = cfg.qc
    false_ids = find_false_births(clist, data3D)
    clist = clist[~clist["cell_id"].isin(false_ids)].copy()
    data3D = data3D[~data3D["cell_id"].isin(false_ids)].copy()
    if max_frame is not None:
        data3D = data3D[data3D["frame"] <= max_frame]

    children = {}
    for _, row in clist.iterrows():
        if int(row["mother_id"]) >= 0:
            children.setdefault(int(row["mother_id"]), []).append(
                (int(row["birth_frame"]), int(row["cell_id"])))
    for v in children.values():
        v.sort()

    intree = set(clist.loc[clist["in_tree"], "cell_id"].astype(int))
    traces = {int(cid): grp.sort_values("frame")
              for cid, grp in data3D.groupby("cell_id") if int(cid) in intree}

    rows = []
    for cid, grp in traces.items():
        info = clist[clist["cell_id"] == cid]
        if info.empty:
            continue
        info = info.iloc[0]
        own_birth = int(info["birth_frame"])
        events = [own_birth] if own_birth >= 0 else []
        events += [f for f, _ in children.get(cid, [])]
        events = sorted(set(e for e in events if e >= 0))
        if len(events) < 2:
            continue
        valid = _valid_trace(grp)
        for i in range(len(events) - 1):
            f0, f1 = events[i], events[i + 1]
            if max_frame is not None and f1 > max_frame:
                continue
            dur_h = (f1 - f0) * cfg.frame_h
            cyc = grp[(grp["frame"] >= f0) & (grp["frame"] < f1)]
            vcyc = valid[(valid["frame"] >= f0) & (valid["frame"] < f1)]
            if len(cyc) == 0:
                continue
            out_frac = float(cyc["is_outlier"].mean())
            bor_frac = float(cyc["touches_border"].mean())
            child_at_end = next((c for f, c in children.get(cid, []) if f == f1), -1)
            s = qc["boundary_search_frames"]
            row = {
                "source": tag, "cell_id": cid, "cycle_index": i,
                "birth_frame": f0, "div_frame": f1, "duration_h": dur_h,
                "generation": int(info["generation"]) + i,
                "parent_cell": int(info["mother_id"]) if i == 0 else cid,
                "child_at_end": child_at_end,
                "n_frames": len(cyc), "n_valid": len(vcyc),
                "outlier_frac": out_frac, "border_frac": bor_frac,
                "V_birth": _value_near(valid, f0, "volume_um3_rod", s),
                "V_div": _value_near(valid, f1 - 1, "volume_um3_rod", s),
                "M_birth": _value_near(valid, f0, "mass_pg", s),
                "M_div": _value_near(valid, f1 - 1, "mass_pg", s),
                "ri_mean": float(vcyc["mean_ri"].mean()) if len(vcyc) else np.nan,
                "V_mean": float(vcyc["volume_um3_rod"].mean()) if len(vcyc) else np.nan,
                "M_mean": float(vcyc["mass_pg"].mean()) if len(vcyc) else np.nan,
            }
            # dry-mass density in mg/mL: 1 pg/um^3 = 1000 mg/mL
            with np.errstate(invalid="ignore", divide="ignore"):
                row["rho_birth"] = 1000.0 * row["M_birth"] / row["V_birth"]
                row["rho_div"] = 1000.0 * row["M_div"] / row["V_div"]
                m, v = vcyc["mass_pg"].to_numpy(), vcyc["volume_um3_rod"].to_numpy()
                row["rho_mean"] = float(np.nanmean(1000.0 * m / v)) if len(vcyc) else np.nan
            # QC verdict
            row["pass_qc"] = (
                cfg.min_cycle_h <= dur_h <= cfg.max_cycle_h
                and out_frac <= qc["max_outlier_frac"]
                and bor_frac <= qc["max_border_frac"]
                and row["n_valid"] >= qc["min_valid_frames"]
                and np.isfinite(row["rho_birth"]) and np.isfinite(row["rho_div"])
            )
            rows.append(row)
    return pd.DataFrame(rows)


def load_dataset(medium: str, data_dir: Path, cfg: MediumConfig,
                 max_frame: int | None) -> tuple[pd.DataFrame, dict]:
    """Build the pooled cycle table for one medium; keep raw traces for the
    phase / death analyses."""
    pairs = discover_pairs(data_dir)
    if not pairs:
        sys.exit(f"[error] no clist/lineage_data3D pairs under {data_dir}")
    tables, raw = [], {}
    for tag, cpath, dpath in pairs:
        clist = pd.read_csv(cpath)
        data3D = pd.read_csv(dpath)
        cyc = build_cycles(tag, clist, data3D, cfg, max_frame)
        if len(cyc):
            cyc.insert(0, "medium", medium)
            tables.append(cyc)
        raw[tag] = (clist, data3D)
    if not tables:
        sys.exit(f"[error] no cycles built for {medium} from {data_dir}")
    df = pd.concat(tables, ignore_index=True)
    n_all, n_ok = len(df), int(df["pass_qc"].sum())
    print(f"[{medium}] channels={len(pairs)} cycles={n_all} pass_qc={n_ok} "
          f"({100*n_ok/max(n_all,1):.0f}%)  gate={cfg.min_cycle_h:.2f}-"
          f"{cfg.max_cycle_h:.2f} h")
    return df, raw


# ---------------------------------------------------------------- helpers

def save_fig_csv(fig: plt.Figure, df: pd.DataFrame, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
    df.to_csv(out / f"{stem}.csv", index=False)
    plt.close(fig)
    print(f"  wrote {stem}.png / .csv")


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")


def dev_within(series: pd.Series, group: pd.Series) -> pd.Series:
    """Deviation from the group (channel) mean - removes per-channel offsets
    before pooling."""
    return series - series.groupby(group).transform("mean")


# ---------------------------------------------------------------- claim 2

def analysis_media(cycles: pd.DataFrame, out: Path) -> None:
    """Cycle-mean density per medium: box + per-lineage means as points."""
    df = cycles[cycles["pass_qc"]].copy()
    media = list(df["medium"].unique())
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    rows = []
    for i, med in enumerate(media):
        sub = df[df["medium"] == med]
        lin = sub.groupby(["source", "cell_id"])["rho_mean"].mean()
        ax.boxplot(sub["rho_mean"].dropna(), positions=[i], widths=0.5,
                   showfliers=False,
                   medianprops=dict(color="black"), boxprops=dict(color="black"))
        jitter = (np.random.default_rng(0).random(len(lin)) - 0.5) * 0.25
        ax.plot(i + jitter, lin.to_numpy(), "o", ms=3.5, alpha=0.7,
                color=medium_color(med), label=f"{med} (lineages n={len(lin)})")
        rows.append(pd.DataFrame({
            "medium": med, "level": "cycle", "value": sub["rho_mean"]}))
        rows.append(pd.DataFrame({
            "medium": med, "level": "lineage_mean", "value": lin.to_numpy()}))
        print(f"  [{med}] cycles={len(sub)} rho_mean={sub['rho_mean'].mean():.1f} "
              f"mg/mL CV={sub['rho_mean'].std()/sub['rho_mean'].mean():.3f}")
    if len(media) == 2:
        from scipy import stats
        a = df[df["medium"] == media[0]]["rho_mean"].dropna()
        b = df[df["medium"] == media[1]]["rho_mean"].dropna()
        u = stats.mannwhitneyu(a, b)
        print(f"  Mann-Whitney {media[0]} vs {media[1]}: p={u.pvalue:.2e}")
    ax.set_xticks(range(len(media)), media)
    ax.set_ylabel("cycle-mean density [mg/mL]")
    ax.legend(frameon=False, fontsize=7)
    style_ax(ax)
    save_fig_csv(fig, pd.concat(rows, ignore_index=True), out, "claim2_media_density")


# ---------------------------------------------------------------- claim 3

def analysis_phase(cycles: pd.DataFrame, raw_by_medium: dict, out: Path,
                   cfgs: dict, n_bins: int = 24) -> None:
    """V / M / rho against normalised cycle phase, media overlaid."""
    grid = np.linspace(0.0, 1.0, n_bins)
    curves = []
    for med, raws in raw_by_medium.items():
        cfg = cfgs[med]
        ok = cycles[(cycles["medium"] == med) & cycles["pass_qc"]]
        per_cycle = {q: [] for q in ("volume_um3_rod", "mass_pg", "rho")}
        for (src, cid), grp in ok.groupby(["source", "cell_id"]):
            clist, data3D = raws[src]
            tr = data3D[data3D["cell_id"] == cid].sort_values("frame")
            tr = tr[(~tr["is_outlier"]) & (~tr["touches_border"])]
            for _, cyc in grp.iterrows():
                w = tr[(tr["frame"] >= cyc["birth_frame"]) & (tr["frame"] < cyc["div_frame"])]
                if len(w) < 6:
                    continue
                phase = (w["frame"] - cyc["birth_frame"]) / (cyc["div_frame"] - cyc["birth_frame"])
                for q in ("volume_um3_rod", "mass_pg"):
                    y = w[q].to_numpy() / np.nanmean(w[q])   # normalise per cycle
                    per_cycle[q].append(np.interp(grid, phase, y))
                rho = 1000.0 * w["mass_pg"] / w["volume_um3_rod"]
                per_cycle["rho"].append(np.interp(grid, phase, rho))
        for q, arrs in per_cycle.items():
            if not arrs:
                continue
            a = np.vstack(arrs)
            curves.append(pd.DataFrame({
                "medium": med, "quantity": q, "phase": grid,
                "mean": a.mean(0), "sd": a.std(0), "n_cycles": len(arrs)}))
    if not curves:
        print("  [phase] no usable cycles")
        return
    cdf = pd.concat(curves, ignore_index=True)
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 6.4), sharex=True)
    labels = {"volume_um3_rod": "V / <V>", "mass_pg": "M / <M>", "rho": "density [mg/mL]"}
    for ax, q in zip(axes, ("volume_um3_rod", "mass_pg", "rho")):
        for med in cdf["medium"].unique():
            s = cdf[(cdf["medium"] == med) & (cdf["quantity"] == q)]
            if s.empty:
                continue
            c = medium_color(med)
            ax.plot(s["phase"], s["mean"], color=c, label=med)
            ax.fill_between(s["phase"], s["mean"] - s["sd"], s["mean"] + s["sd"],
                            color=c, alpha=0.2, lw=0)
        ax.set_ylabel(labels[q])
        style_ax(ax)
    axes[0].legend(frameon=False, fontsize=7)
    axes[-1].set_xlabel("cycle phase")
    save_fig_csv(fig, cdf, out, "claim3_phase_locked")


# ---------------------------------------------------------------- claim 4

QUANT_COLS = {"V": "V_birth", "M": "M_birth", "rho": "rho_mean"}


def _consecutive_series(cycles: pd.DataFrame) -> dict:
    """{(medium, source, cell_id): DataFrame sorted by cycle_index} with only
    QC-passing consecutive runs of the same physical cell (mother lines carry
    most of the lags)."""
    out = {}
    ok = cycles[cycles["pass_qc"]]
    for key, grp in ok.groupby(["medium", "source", "cell_id"]):
        g = grp.sort_values("cycle_index")
        if len(g) >= 3:
            out[key] = g
    return out


def analysis_memory(cycles: pd.DataFrame, out: Path, max_lag: int = 6,
                    n_shuffle: int = 200) -> None:
    """Autocorrelation of per-cycle deviations vs generation lag with an
    exponential fit (wang2010 Fig.3A idiom), one line per quantity, colour
    per medium, shuffle control as a grey band."""
    series = _consecutive_series(cycles)
    rng = np.random.default_rng(1)
    rows = []
    for med in cycles["medium"].unique():
        for qname, col in QUANT_COLS.items():
            runs = [g[col].to_numpy() for k, g in series.items() if k[0] == med]
            runs = [r - np.nanmean(r) for r in runs if np.isfinite(r).sum() >= 3]
            if not runs:
                continue
            for lag in range(0, max_lag + 1):
                xs, ys = [], []
                for r in runs:
                    if len(r) > lag:
                        xs.append(r[:len(r) - lag] if lag else r)
                        ys.append(r[lag:] if lag else r)
                x = np.concatenate(xs); y = np.concatenate(ys)
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 8:
                    continue
                r_obs = float(np.corrcoef(x[m], y[m])[0, 1])
                # shuffle control: permute cycle order within each run
                r_sh = []
                for _ in range(n_shuffle):
                    xs2, ys2 = [], []
                    for r in runs:
                        p = rng.permutation(r)
                        if len(p) > lag:
                            xs2.append(p[:len(p) - lag] if lag else p)
                            ys2.append(p[lag:] if lag else p)
                    x2 = np.concatenate(xs2); y2 = np.concatenate(ys2)
                    m2 = np.isfinite(x2) & np.isfinite(y2)
                    if m2.sum() >= 8:
                        r_sh.append(np.corrcoef(x2[m2], y2[m2])[0, 1])
                rows.append({"medium": med, "quantity": qname, "lag": lag,
                             "r": r_obs, "n_pairs": int(m.sum()),
                             "shuffle_lo": float(np.percentile(r_sh, 2.5)) if r_sh else np.nan,
                             "shuffle_hi": float(np.percentile(r_sh, 97.5)) if r_sh else np.nan})
    if not rows:
        print("  [memory] not enough consecutive cycles")
        return
    df = pd.DataFrame(rows)

    # exponential fit r(k) = exp(-k/tau) over the positive-r prefix
    taus = []
    for (med, q), grp in df.groupby(["medium", "quantity"]):
        g = grp[grp["lag"] >= 1].sort_values("lag")
        pos = g[g["r"] > 0]
        pos = pos[pos["lag"] <= (pos["lag"].diff().fillna(1) == 1).cumsum().max()]
        tau = np.nan
        if len(pos) >= 2:
            slope = np.polyfit(pos["lag"], np.log(pos["r"]), 1)[0]
            tau = -1.0 / slope if slope < 0 else np.inf
        taus.append({"medium": med, "quantity": q, "tau_generations": tau})
        print(f"  [{med}] {q}: tau = {tau:.2f} gen" if np.isfinite(tau)
              else f"  [{med}] {q}: tau not defined")
    tdf = pd.DataFrame(taus)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ls = {"V": ":", "M": "--", "rho": "-"}
    for (med, q), grp in df.groupby(["medium", "quantity"]):
        g = grp.sort_values("lag")
        ax.plot(g["lag"], g["r"], ls[q], marker="o", ms=3,
                color=medium_color(med), label=f"{med} {q}")
    sh = df.groupby("lag")[["shuffle_lo", "shuffle_hi"]].mean().reset_index()
    ax.fill_between(sh["lag"], sh["shuffle_lo"], sh["shuffle_hi"],
                    color="0.7", alpha=0.4, lw=0, label="shuffle 95%")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlabel("generation lag")
    ax.set_ylabel("autocorrelation")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    style_ax(ax)
    save_fig_csv(fig, df.merge(tdf, on=["medium", "quantity"], how="left"),
                 out, "claim4_memory")


# ---------------------------------------------------------------- claim 5

def _slope_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 500,
              rng=None) -> tuple[float, float, float]:
    rng = rng or np.random.default_rng(2)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    b = np.polyfit(x, y, 1)[0]
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        bs.append(np.polyfit(x[i], y[i], 1)[0])
    return float(b), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analysis_control(cycles: pd.DataFrame, out: Path, n_sim: int = 500) -> None:
    """Return maps of birth V, M, rho (x = value at birth n, y = at birth n+1,
    both as deviations from the channel mean). The rho panel carries a null
    band: rho slope implied by a joint VAR(1) of (lnV, lnM) with no direct
    control of rho itself."""
    series = _consecutive_series(cycles)
    rng = np.random.default_rng(3)
    panels, rows = {}, []
    for med in cycles["medium"].unique():
        runs = [g for k, g in series.items() if k[0] == med]
        if not runs:
            continue
        for qname, col in (("V", "V_birth"), ("M", "M_birth"), ("rho", "rho_birth")):
            xs, ys = [], []
            for g in runs:
                v = g[col].to_numpy()
                v = v - np.nanmean(v)
                xs.append(v[:-1]); ys.append(v[1:])
            x, y = np.concatenate(xs), np.concatenate(ys)
            b, lo, hi = _slope_ci(x, y, rng=rng)
            panels[(med, qname)] = (x, y, b, lo, hi)
            rows.append({"medium": med, "quantity": qname, "slope": b,
                         "ci_lo": lo, "ci_hi": hi,
                         "n_pairs": int(np.isfinite(x + y).sum())})
            print(f"  [{med}] return-map slope {qname}: {b:+.2f} [{lo:+.2f},{hi:+.2f}]")

        # --- null for rho from joint AR(1) of lnV, lnM birth values
        lnv = [np.log(g["V_birth"].to_numpy()) for g in runs]
        lnm = [np.log(g["M_birth"].to_numpy()) for g in runs]
        pairs = []
        for v, m in zip(lnv, lnm):
            v, m = v - np.nanmean(v), m - np.nanmean(m)
            for i in range(len(v) - 1):
                pairs.append((v[i], m[i], v[i + 1], m[i + 1]))
        P = np.array([p for p in pairs if np.all(np.isfinite(p))])
        if len(P) >= 20:
            X, Y = P[:, :2], P[:, 2:]
            A, *_ = np.linalg.lstsq(X, Y, rcond=None)          # VAR(1) matrix
            R = Y - X @ A
            C = np.cov(R.T)
            null_slopes = []
            for _ in range(n_sim):
                n = 400
                z = np.zeros((n, 2))
                noise = rng.multivariate_normal([0, 0], C, size=n)
                for t in range(1, n):
                    z[t] = z[t - 1] @ A + noise[t]
                rho_dev = z[:, 1] - z[:, 0]                    # ln(M/V) deviation
                null_slopes.append(np.polyfit(rho_dev[:-1], rho_dev[1:], 1)[0])
            nlo, nhi = np.percentile(null_slopes, [2.5, 97.5])
            rows.append({"medium": med, "quantity": "rho_null_from_VM",
                         "slope": float(np.mean(null_slopes)),
                         "ci_lo": float(nlo), "ci_hi": float(nhi),
                         "n_pairs": len(P)})
            print(f"  [{med}] rho null (VAR1 of lnV,lnM): "
                  f"{np.mean(null_slopes):+.2f} [{nlo:+.2f},{nhi:+.2f}]")

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
    for ax, qname in zip(axes, ("V", "M", "rho")):
        for med in cycles["medium"].unique():
            if (med, qname) not in panels:
                continue
            x, y, b, lo, hi = panels[(med, qname)]
            c = medium_color(med)
            ax.plot(x, y, "o", ms=2, alpha=0.35, color=c)
            xr = np.linspace(np.nanmin(x), np.nanmax(x), 2)
            ax.plot(xr, b * xr, color=c, lw=1.4,
                    label=f"{med} {b:+.2f} [{lo:+.2f},{hi:+.2f}]")
        ax.axline((0, 0), slope=0, color="0.6", lw=0.6)
        ax.set_title(qname, fontsize=9)
        ax.set_xlabel("deviation at birth n")
        ax.legend(frameon=False, fontsize=6)
        style_ax(ax)
    axes[0].set_ylabel("deviation at birth n+1")
    save_fig_csv(fig, pd.DataFrame(rows), out, "claim5_return_maps")


# ---------------------------------------------------------------- claim 6

def analysis_division(cycles: pd.DataFrame, raw_by_medium: dict, out: Path,
                      cfgs: dict) -> None:
    """Partition at division: r_V vs r_M (child vs parent-after-division at
    the same frame), sibling density difference, and a variance split of the
    daughter birth density into maternal / asymmetry / stochastic
    (proenca Fig.3b idiom)."""
    qc_search = QC_DEFAULTS["boundary_search_frames"]
    rows = []
    ok = cycles[cycles["pass_qc"]]
    for med, raws in raw_by_medium.items():
        for _, cyc in ok[(ok["medium"] == med) & (ok["child_at_end"] >= 0)].iterrows():
            clist, data3D = raws[cyc["source"]]
            f = int(cyc["div_frame"])
            par = data3D[data3D["cell_id"] == cyc["cell_id"]]
            par = par[(~par["is_outlier"]) & (~par["touches_border"])]
            chi = data3D[data3D["cell_id"] == cyc["child_at_end"]]
            chi = chi[(~chi["is_outlier"]) & (~chi["touches_border"])]
            Vp = _value_near(par, f, "volume_um3_rod", qc_search)
            Vc = _value_near(chi, f, "volume_um3_rod", qc_search)
            Mp = _value_near(par, f, "mass_pg", qc_search)
            Mc = _value_near(chi, f, "mass_pg", qc_search)
            if not all(np.isfinite(v) and v > 0 for v in (Vp, Vc, Mp, Mc)):
                continue
            rows.append({
                "medium": med, "source": cyc["source"],
                "parent_cell": cyc["cell_id"], "child_cell": cyc["child_at_end"],
                "div_frame": f,
                "r_V": Vc / (Vc + Vp), "r_M": Mc / (Mc + Mp),
                "rho_child": 1000.0 * Mc / Vc, "rho_parent": 1000.0 * Mp / Vp,
                "rho_mother_cycle": cyc["rho_mean"],
            })
    if not rows:
        print("  [division] no usable division events")
        return
    df = pd.DataFrame(rows)
    df["d_rho_sibling"] = df["rho_child"] - df["rho_parent"]

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    # A: schematic placeholder (the real schematic is drawn in the figure
    # tool; here we keep the axis to fix the panel budget)
    axes[0].axis("off")
    axes[0].text(0.5, 0.5, "schematic:\nr_V, r_M, $\\Delta\\rho$",
                 ha="center", va="center", fontsize=9)
    # B: r_M vs r_V with identity
    ax = axes[1]
    for med, g in df.groupby("medium"):
        ax.plot(g["r_V"], g["r_M"], "o", ms=3, alpha=0.5,
                color=medium_color(med), label=f"{med} (n={len(g)})")
    ax.axline((0.5, 0.5), slope=1, color="0.4", lw=0.8)
    ax.set_xlabel("volume share of new sibling  $r_V$")
    ax.set_ylabel("mass share  $r_M$")
    ax.legend(frameon=False, fontsize=7)
    style_ax(ax)
    # C: variance split of daughter birth density
    ax = axes[2]
    labels, fracs, meds = [], [], []
    for med, g in df.groupby("medium"):
        y = g["rho_child"].to_numpy()
        y = y - np.nanmean(y)
        var_tot = np.nanvar(y)
        # maternal: predict from mother-cycle mean density
        x1 = g["rho_mother_cycle"].to_numpy()
        x1 = x1 - np.nanmean(x1)
        m1 = np.isfinite(x1 + y)
        b1 = np.polyfit(x1[m1], y[m1], 1)[0] if m1.sum() > 4 else 0.0
        res1 = y - b1 * np.where(np.isfinite(x1), x1, 0.0)
        # asymmetry: r_V deviation from 1/2 on top of maternal
        x2 = g["r_V"].to_numpy() - 0.5
        m2 = np.isfinite(x2 + res1)
        b2 = np.polyfit(x2[m2], res1[m2], 1)[0] if m2.sum() > 4 else 0.0
        res2 = res1 - b2 * np.where(np.isfinite(x2), x2, 0.0)
        v1 = var_tot - np.nanvar(res1)
        v2 = np.nanvar(res1) - np.nanvar(res2)
        v3 = np.nanvar(res2)
        for lab, v in (("maternal", v1), ("asymmetry", v2), ("stochastic", v3)):
            labels.append(lab); fracs.append(v / var_tot); meds.append(med)
        print(f"  [{med}] var split maternal/asym/stoch = "
              f"{v1/var_tot:.2f}/{v2/var_tot:.2f}/{v3/var_tot:.2f} (n={len(g)})")
    vs = pd.DataFrame({"medium": meds, "component": labels, "fraction": fracs})
    x = np.arange(3)
    w = 0.35
    for i, med in enumerate(vs["medium"].unique()):
        s = vs[vs["medium"] == med]
        ax.bar(x + (i - 0.5) * w * (len(vs["medium"].unique()) > 1),
               s["fraction"], width=w, color=medium_color(med), label=med)
    ax.set_xticks(x, ["maternal", "asymmetry", "stochastic"], fontsize=8)
    ax.set_ylabel("fraction of daughter\nbirth-density variance")
    ax.legend(frameon=False, fontsize=7)
    style_ax(ax)
    save_fig_csv(fig, df, out, "claim6_division_partition")
    vs.to_csv(out / "claim6_variance_split.csv", index=False)


# ---------------------------------------------------------------- claim 7

def analysis_range(cycles: pd.DataFrame, out: Path, top_pct: float = 5.0) -> None:
    """Density vs birth volume across the full natural range, colour = medium,
    the largest cells highlighted as open circles (knapp Fig.1D idiom)."""
    df = cycles[cycles["pass_qc"]].copy()
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for med, g in df.groupby("medium"):
        cut = np.nanpercentile(g["V_birth"], 100 - top_pct)
        big = g["V_birth"] >= cut
        c = medium_color(med)
        ax.plot(g.loc[~big, "V_birth"], g.loc[~big, "rho_mean"], "o", ms=2.5,
                alpha=0.4, color=c, label=f"{med} (n={len(g)})")
        ax.plot(g.loc[big, "V_birth"], g.loc[big, "rho_mean"], "o", ms=4,
                mfc="none", mec=c, label=f"{med} top {top_pct:.0f}% (n={int(big.sum())})")
        r = np.corrcoef(g["V_birth"].dropna(),
                        g.loc[g["V_birth"].notna(), "rho_mean"])[0, 1]
        print(f"  [{med}] corr(V_birth, rho_mean) = {r:+.2f}")
    ax.set_xlabel("birth volume [$\\mu m^3$]")
    ax.set_ylabel("cycle-mean density [mg/mL]")
    ax.legend(frameon=False, fontsize=6)
    style_ax(ax)
    save_fig_csv(fig, df[["medium", "source", "cell_id", "cycle_index",
                          "V_birth", "M_birth", "rho_mean"]], out,
                 "claim7_range")


# ---------------------------------------------------------------- claim 8

def analysis_death(cycles: pd.DataFrame, raw_by_medium: dict, out: Path,
                   cfgs: dict, window_h: float = 10.0) -> None:
    """Death-aligned density / volume / mass trajectories. A death is a cell
    with no children whose trace ends well before the recording does."""
    rows, traj = [], []
    for med, raws in raw_by_medium.items():
        cfg = cfgs[med]
        qc = cfg.qc
        for src, (clist, data3D) in raws.items():
            last = int(data3D["frame"].max())
            has_child = set(clist.loc[clist["mother_id"] >= 0, "mother_id"].astype(int))
            cand = clist[(clist["in_tree"])
                         & (~clist["cell_id"].isin(has_child))
                         & (clist["death_frame"] < last - qc["death_margin_frames"])
                         & (clist["n_frames"] >= qc["death_min_frames"])]
            for _, row in cand.iterrows():
                cid = int(row["cell_id"])
                tr = data3D[data3D["cell_id"] == cid].sort_values("frame")
                tr = tr[(~tr["is_outlier"]) & (~tr["touches_border"])]
                if len(tr) < 6:
                    continue
                f_death = int(row["death_frame"])
                t = (tr["frame"] - f_death) * cfg.frame_h
                keep = t >= -window_h
                rho = 1000.0 * tr["mass_pg"] / tr["volume_um3_rod"]
                traj.append(pd.DataFrame({
                    "medium": med, "source": src, "cell_id": cid,
                    "t_to_death_h": t[keep],
                    "rho": rho[keep],
                    "V": tr.loc[keep, "volume_um3_rod"],
                    "M": tr.loc[keep, "mass_pg"]}))
                rows.append({"medium": med, "source": src, "cell_id": cid,
                             "death_frame": f_death,
                             "life_h": float(row["age_h"])})
    if not traj:
        print("  [death] no death candidates under the current margins")
        return
    tdf = pd.concat(traj, ignore_index=True)
    print(f"  death candidates: {len(rows)}")

    fig, axes = plt.subplots(3, 1, figsize=(3.6, 6.4), sharex=True)
    for ax, q, lab in zip(axes, ("rho", "V", "M"),
                          ("density [mg/mL]", "V [$\\mu m^3$]", "M [pg]")):
        for med, g in tdf.groupby("medium"):
            c = medium_color(med)
            for _, cell in g.groupby(["source", "cell_id"]):
                ax.plot(cell["t_to_death_h"], cell[q], color=c, alpha=0.25, lw=0.7)
            # binned mean
            bins = np.arange(-window_h, 0.5, 1.0)
            bi = np.digitize(g["t_to_death_h"], bins)
            mu = g.groupby(bi)[q].mean()
            ax.plot(bins[np.clip(mu.index - 1, 0, len(bins) - 1)] + 0.5,
                    mu.to_numpy(), color=c, lw=2, label=med)
        ax.set_ylabel(lab)
        style_ax(ax)
    axes[0].legend(frameon=False, fontsize=7)
    axes[-1].set_xlabel("time to death [h]")
    save_fig_csv(fig, tdf, out, "claim8_death_aligned")
    pd.DataFrame(rows).to_csv(out / "claim8_death_candidates.csv", index=False)


# ---------------------------------------------------------------- main

STEPS = ("media", "phase", "memory", "control", "division", "range", "death")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="build cycle tables and run analyses")
    run.add_argument("--dataset", action="append", required=True,
                     metavar="MEDIUM:DATA_DIR",
                     help="e.g. EMM:results/mid_term_0405/data (repeatable)")
    run.add_argument("--config", default=None,
                     help="JSON overriding media params / qc gates")
    run.add_argument("--max-frame", type=int, default=None,
                     help="ignore frames beyond this (e.g. 2018 = phase1)")
    run.add_argument("--steps", default=",".join(STEPS),
                     help=f"comma list from {STEPS}")
    run.add_argument("--out", default="results/claim_ladder")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_cycles, raw_by_medium, cfgs = [], {}, {}
    for spec in args.dataset:
        med, _, ddir = spec.partition(":")
        cfg = load_medium_config(med, args.config)
        cfgs[med] = cfg
        cyc, raw = load_dataset(med, Path(ddir), cfg, args.max_frame)
        all_cycles.append(cyc)
        raw_by_medium[med] = raw
    cycles = pd.concat(all_cycles, ignore_index=True)
    cycles.to_csv(out / "cycle_table.csv", index=False)
    print(f"cycle_table.csv: {len(cycles)} rows "
          f"({int(cycles['pass_qc'].sum())} pass_qc)")

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for s in steps:
        print(f"-- {s}")
        if s == "media":
            analysis_media(cycles, out)
        elif s == "phase":
            analysis_phase(cycles, raw_by_medium, out, cfgs)
        elif s == "memory":
            analysis_memory(cycles, out)
        elif s == "control":
            analysis_control(cycles, out)
        elif s == "division":
            analysis_division(cycles, raw_by_medium, out, cfgs)
        elif s == "range":
            analysis_range(cycles, out)
        elif s == "death":
            analysis_death(cycles, raw_by_medium, out, cfgs)
        else:
            print(f"  unknown step: {s}")


if __name__ == "__main__":
    main()
