"""
figreq_260908_numbers.py — docs/FIGURE_REQUIREMENTS_260908.md の数値を出し直す。

入力は 260908_outside_quad の確認パッケージ（gallery_100h_v3、feat/ecc-float-input b4096f5）と
docs/260908_death_marks.yaml。図は作らない。要件文書に書いた数を同じ定義で再計算して標準出力に出す。

定義（要件文書 §1 と同じ）:
  - サイクル = cycle_fits_review.csv の1行（accepted な分裂から次の accepted な分裂の直前まで）。
  - death cycle = mark_h を含むサイクル。mark_h 以前で最後の accepted 分裂から始まる。
  - 規則 B<k>: 死亡マークのある母細胞は death cycle とその前の k 個の完結サイクルを捨てる
    （k=4 で「最後のサイクルを含めて 5 サイクル」）。マークの無い母細胞は全サイクルを使う。
  - birth / division の値 = サイクル先頭 / 末尾 3 有効フレームの中央値（review_valid のみ）。

Example:
    python scripts/figreq_260908_numbers.py \\
        --pkg qc_review/260908_outside_quad/gallery_100h_v3 --deaths docs/260908_death_marks.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

METRICS = ["interval_h", "doubling_time_h", "m_birth", "v_birth", "L_birth", "w_mean", "rho_mean", "fold_m"]


def load(pkg: Path, deaths_yaml: Path):
    ch = pd.read_csv(pkg / "channel_review.csv", encoding="utf-8-sig")
    cyc = pd.read_csv(pkg / "cycle_fits_review.csv", encoding="utf-8-sig")
    ev = pd.read_csv(pkg / "division_events_review.csv", encoding="utf-8-sig")
    m = pd.read_csv(pkg / "mother_measurements_review.csv.gz")
    deaths = pd.DataFrame(yaml.safe_load(deaths_yaml.read_text(encoding="utf-8"))["deaths"])
    inc = ch[ch.included].copy()
    keys = set(zip(inc.pos, inc.ch))
    missing = [k for k in zip(deaths.pos, deaths.ch) if k not in keys]
    if missing:
        raise SystemExit(f"death marks on channels that are not included: {missing}")
    acc = ev[ev.accepted]
    starts = []
    for r in deaths.itertuples():
        f = acc[(acc.pos == r.pos) & (acc.ch == r.ch)].frame
        f = f[(f - 2) / 12 <= r.mark_h]
        starts.append(int(f.max()) if len(f) else 2)
    deaths["death_cycle_start_frame"] = starts
    cyc = cyc[[k in keys for k in zip(cyc.pos, cyc.ch)]].reset_index(drop=True)
    return ch, inc, cyc, ev, m, deaths


def enrich(cyc: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    v = m[m.review_valid].sort_values("frame")
    grp = {k: g for k, g in v.groupby(["pos", "ch"])}
    rows = []
    for r in cyc.itertuples():
        g = grp[(r.pos, r.ch)]
        seg = g[(g.frame >= r.start_frame) & (g.frame < r.end_frame)]
        if len(seg) < 5:
            rows.append({})
            continue
        h, t = seg.iloc[:3], seg.iloc[-3:]
        rows.append(dict(m_birth=h.phase_mass.median(), m_div=t.phase_mass.median(),
                         v_birth=h.volume_um3_efd.median(), v_div=t.volume_um3_efd.median(),
                         L_birth=h.long_axis_um.median(), L_div=t.long_axis_um.median(),
                         w_mean=seg.short_axis_um.median(), rho_mean=seg.density_pg_um3_review_efd.mean()))
    out = pd.concat([cyc, pd.DataFrame(rows)], axis=1)
    out["interval_h"] = out.t_end - out.t_start
    out["fold_m"] = out.m_div / out.m_birth
    return out


def rel_index(cyc: pd.DataFrame, deaths: pd.DataFrame) -> pd.Series:
    """-1 = 最後の完結サイクル（death cycle の直前）。death cycle 以降は 0（捨てる）。死亡なしは NaN。"""
    rel = pd.Series(np.nan, index=cyc.index)
    for r in deaths.itertuples():
        sel = (cyc.pos == r.pos) & (cyc.ch == r.ch)
        before = cyc[sel & (cyc.end_frame <= r.death_cycle_start_frame)].sort_values("start_frame")
        rel[sel] = 0
        rel[before.index] = np.arange(-len(before), 0)
    return rel


def keep_rule(rel: pd.Series, k: int) -> pd.Series:
    return rel.isna() | (rel < -k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", type=Path, default=Path("qc_review/260908_outside_quad/gallery_100h_v3"))
    ap.add_argument("--deaths", type=Path, default=Path("docs/260908_death_marks.yaml"))
    ap.add_argument("--drop", type=int, default=4, help="death cycle の前に捨てる完結サイクル数（既定 4 = 計 5 サイクル）")
    args = ap.parse_args()

    ch, inc, cyc, ev, m, deaths = load(args.pkg, args.deaths)
    cyc = enrich(cyc, m)
    cyc["rel"] = rel_index(cyc, deaths)
    dead_keys = set(zip(deaths.pos, deaths.ch))
    cyc["dead"] = [k in dead_keys for k in zip(cyc.pos, cyc.ch)]

    print("## 0. 列の確認")
    eq = {a: float(np.nanmax(np.abs(m[a] - m[b]))) for a, b in
          [("mass_pg_efd", "phase_mass"), ("density_pg_um3_efd", "density_pg_um3_review_efd"),
           ("mean_ri_efd", "mean_ri_review_efd"), ("mass_pg", "mass_pg_efd")]}
    print("max |差|:", {k: f"{v:.2e}" for k, v in eq.items()})
    print(f"rod 列 density_pg_um3 vs efd 中央値: {m.density_pg_um3.median():.4f} vs {m.density_pg_um3_efd.median():.4f}")
    print(f"n_medium_used: {m.n_medium_used.unique()}  medium_name: {m.medium_name.unique()}")

    print("\n## 1. 対象")
    print(f"channel_review: {len(ch)} 行, included {len(inc)}, 死亡マーク {len(deaths)} "
          f"(swelling {sum(deaths.kind == 'swelling')}, elongation {sum(deaths.kind == 'elongation')}, "
          f"unspecified {sum(deaths.kind == 'unspecified')})")
    last = m[m.review_valid].groupby(["pos", "ch"]).frame.max()
    surv = [k for k in zip(inc.pos, inc.ch) if k not in dead_keys]
    to100 = [k for k in surv if last.get(k, 0) >= 1190]
    after = sum(((cyc.pos == r.pos) & (cyc.ch == r.ch) & (cyc.start_frame > r.death_cycle_start_frame)).sum()
                for r in deaths.itertuples())
    print(f"死亡マークなし {len(surv)} 母細胞 / {int((~cyc.dead).sum())} サイクル。"
          f"うち 99 h 以上まで追えた {len(to100)} 母細胞 / {int(cyc[[k in set(to100) for k in zip(cyc.pos, cyc.ch)]].shape[0])} サイクル")
    print(f"death cycle より後に fit されたサイクル（偽の分裂）: {after}")
    for k in [0, 1, 2, 3, 4, 5, 6, 8]:
        kk = keep_rule(cyc.rel, k)
        sub = cyc[kk]
        nd = sub[sub.dead].groupby(["pos", "ch"]).ngroups
        print(f"  B{k}: {len(sub)} サイクル, {sub.groupby(['pos', 'ch']).ngroups} 母細胞（死亡系列から {nd} 母細胞 / {int(sub.dead.sum())} サイクル）")

    use = cyc[keep_rule(cyc.rel, args.drop)].copy()
    good = use[(use.r2 > 0.9) & (use.interval_h < 6)].copy()
    print(f"\n規則 B{args.drop}: {len(use)} サイクル / {use.groupby(['pos', 'ch']).ngroups} 母細胞。"
          f"r2>0.9 かつ interval<6 h: {len(good)}")

    print("\n## 2. 10 h ごと（規則 B / 死亡なしのみ）")
    good["bin"] = (good.t_start // 10 * 10).astype(int)
    tt = good.groupby("bin").agg(n=("pos", "size"), mothers=("ch", lambda s: good.loc[s.index].groupby(["pos", "ch"]).ngroups),
                                 Td=("doubling_time_h", "median"), interval=("interval_h", "median"),
                                 m_birth=("m_birth", "median"), rho=("rho_mean", "median"), w=("w_mean", "median"))
    a = good[~good.dead].groupby("bin").apply(lambda g: g.groupby(["pos", "ch"]).ngroups)
    tt["mothers_survivors_only"] = a
    print(tt.round(3).to_string())

    print("\n## 3. 分布（規則 B）")
    for c in ["doubling_time_h", "interval_h", "m_birth", "v_birth", "L_birth", "L_div", "w_mean", "rho_mean", "fold_m"]:
        s = good[c].dropna()
        print(f"{c}: median {s.median():.3f}  IQR {s.quantile(.25):.3f}-{s.quantile(.75):.3f}  CV {s.std() / s.mean():.3f}")
    print(f"r2 中央値 {use.r2.median():.4f}, r2<0.9: {int((use.r2 < 0.9).sum())}")
    print(f"m_div 中央値 {good.m_div.median():.2f} pg。間隔 < 1.5 h: {int((good.interval_h < 1.5).sum())}, > 3 h: {int((good.interval_h > 3).sum())}"
          f"（robust CV {(good.interval_h.quantile(.75) - good.interval_h.quantile(.25)) / 1.349 / good.interval_h.median():.3f}）")

    print("\n## 4. 死ぬ前のサイクル（時間を合わせた死亡なし系列との robust z の中央値）")
    ref = cyc[~cyc.dead].assign(b=lambda d: (d.t_start // 10).astype(int)).groupby("b")[METRICS]
    med, iqr = ref.median(), ref.quantile(.75) - ref.quantile(.25)
    zz = []
    for r in cyc[cyc.rel < 0].itertuples():
        b = min(int(r.t_start // 10), med.index.max())
        zz.append(dict(rel=r.rel, **{c: (getattr(r, c) - med.loc[b, c]) / (iqr.loc[b, c] / 1.349) for c in METRICS}))
    z = pd.DataFrame(zz)
    tab = z[z.rel >= -10].groupby("rel")[METRICS].median().round(2)
    tab["n"] = z[z.rel >= -10].groupby("rel").size()
    print(tab.to_string())
    flag = (z[["interval_h", "doubling_time_h", "m_birth", "v_birth", "rho_mean"]].abs() > 2).any(axis=1)
    print("|z|>2 が1つでもある割合:", flag[z.rel >= -10].groupby(z.rel).mean().round(2).to_dict())
    for r in [-1, -2, -3]:
        print(f"rel {r} の生の中央値:", cyc[cyc.rel == r][["interval_h", "doubling_time_h", "m_div"]].median().round(3).to_dict())

    print("\n## 5. Pos の偏り（規則 B）")
    good["posn"] = good.pos.str[3:].astype(int)
    good["side"] = np.where(good.posn <= 52, "Pos<=52", "Pos>=53")
    print(good.groupby("side").agg(mothers=("ch", lambda s: good.loc[s.index].groupby(["pos", "ch"]).ngroups),
                                   Td=("doubling_time_h", "median"), m_birth=("m_birth", "median"),
                                   v_birth=("v_birth", "median"), L_birth=("L_birth", "median"), rho=("rho_mean", "median"),
                                   w=("w_mean", "median")).round(3).to_string())
    print("Pos ごとの密度の中央値:", good.groupby("posn").rho_mean.median().round(3).to_dict())
    p = stats.mannwhitneyu(*[g.rho_mean.dropna() for _, g in good.groupby("side")]).pvalue
    print(f"rho の Mann-Whitney U（サイクル単位、母細胞の入れ子は無視）p = {p:.1e}")
    pm = good.groupby(["side", "pos", "ch"]).rho_mean.median().reset_index()
    p2 = stats.mannwhitneyu(*[g.rho_mean for _, g in pm.groupby("side")]).pvalue
    print(f"rho の Mann-Whitney U（母細胞の中央値単位）p = {p2:.1e}, n = {pm.groupby('side').size().to_dict()}")

    print("\n## 6. 母細胞間のばらつき（規則 B）")
    key = good.pos + "_" + good.ch
    for c in ["doubling_time_h", "m_birth", "v_birth", "rho_mean"]:
        share = good.groupby(key)[c].transform("mean").var() / good[c].var()
        cvm = good.groupby(key)[c].median()
        print(f"{c}: 母細胞間の分散の割合 {share:.2f}, 母細胞の中央値の CV {cvm.std() / cvm.mean():.3f}")
    print("死亡なし母細胞あたりのサイクル数:", cyc[~cyc.dead].groupby(["pos", "ch"]).size().describe().round(1).to_dict())

    print("\n## 7. サイズ制御と return map（規則 B）")
    good = good.sort_values(["pos", "ch", "start_frame"])
    good["added_m"], good["added_v"], good["added_L"] = good.m_div - good.m_birth, good.v_div - good.v_birth, good.L_div - good.L_birth
    for x, y in [("m_birth", "added_m"), ("v_birth", "added_v"), ("L_birth", "added_L"), ("L_birth", "L_div"), ("m_birth", "m_div")]:
        s = good[[x, y]].dropna()
        print(f"{y} vs {x}: Pearson r = {stats.pearsonr(s[x], s[y])[0]:.2f}, OLS slope = {np.polyfit(s[x], s[y], 1)[0]:.2f}, n = {len(s)}")
    g = good.groupby(["pos", "ch"])
    for c in ["m_birth", "v_birth", "interval_h", "rho_mean", "doubling_time_h"]:
        s = pd.DataFrame(dict(a=good[c], b=g[c].shift(-1))).dropna()
        print(f"return map {c}(N) vs (N+1): r = {stats.pearsonr(s.a, s.b)[0]:.2f}, slope = {np.polyfit(s.a, s.b, 1)[0]:.2f}, n = {len(s)}")
    frac = (g.m_birth.shift(-1) / good.m_div).dropna()
    print(f"m_birth(N+1) / m_div(N) 中央値 {frac.median():.3f}  IQR {frac.quantile(.25):.3f}-{frac.quantile(.75):.3f}")
    good["gen"] = g.cumcount()
    print("母細胞のサイクル番号ごとの Td 中央値:", good.groupby(good.gen // 10 * 10).doubling_time_h.median().round(3).to_dict())

    print("\n## 8. 分裂イベントの検証（division_events_review.csv）")
    e = ev[[k in set(zip(inc.pos, inc.ch)) for k in zip(ev.pos, ev.ch)]]
    print(f"候補 {len(e)}, accepted {int(e.accepted.sum())}, origin: {e[e.accepted].origin.value_counts().to_dict()}")
    print("reason:", e.reason.value_counts().to_dict())
    print(f"accepted の mass_ratio 中央値 {e[e.accepted].mass_ratio.median():.3f} IQR "
          f"{e[e.accepted].mass_ratio.quantile(.25):.3f}-{e[e.accepted].mass_ratio.quantile(.75):.3f}; "
          f"volume_ratio 中央値 {e[e.accepted].volume_ratio.median():.3f}")


if __name__ == "__main__":
    main()
