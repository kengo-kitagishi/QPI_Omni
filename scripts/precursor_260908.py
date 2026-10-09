"""
precursor_260908.py — 260908 の死亡前兆: 死ぬ系列は、死ぬ何サイクル・何時間前から生存系列と区別できるか。

入力は 260908_outside_quad の確認パッケージ（gallery_100h_v3、feat/ecc-float-input b4096f5）と
docs/260908_death_marks.yaml。結果の読み方は docs/PRECURSOR_260908.md。

基準時刻（anchor）= 死亡系列の最後の分裂（mark_h 以前で最後の accepted 分裂。figreq_260908_numbers.load の
death_cycle_start_frame と同じ）。mark の位置は death cycle の中でばらつく（anchor から 0.4–8.5 h 後）ので、
客観的に決まる最後の分裂にそろえる。lag −1 = 最後の完結サイクル（anchor で終わるサイクル）。

比較の相手（matched control）: 死亡系列ごとに、同じ Pos 群（Pos1–17 / Pos21 以降。密度・体積に Pos の偏りがある）で、
死亡マークが無く anchor の 2 h 後まで追えた系列を集め、anchor に一番近い分裂（±1.5 h）を仮の anchor にする。
時刻と Pos をそろえるので、90–100 h の遅れや Pos の偏りは差に入らない。

指標: AUC = P(死亡系列の値 > control の値)。死亡系列ごとに control の中の順位（mid-rank）を出し、死亡系列で平均する。
0.5 = 区別できない。95% CI は死亡系列と control 系列の両方を復元抽出した bootstrap。
null: 生存系列に死亡系列の anchor をランダムに割り当て、残りの生存系列を control にして同じ AUC を出す（--null 回）。
その 95% 帯の外を「区別できる」とし、lag −1 から途切れずに外にある一番古い lag を onset とする。

モード: abs = 値そのもの（他の細胞との違い）。self = 各系列の lag −20..−11 の中央値を引いた変化（その系列の過去との違い。
系列ごとの癖と Pos の偏りが消える。基準のサイクルが 4 つ以上ある系列だけ）。

Example:
    python scripts/precursor_260908.py \\
        --pkg qc_review/260908_outside_quad/gallery_100h_v3 --deaths docs/260908_death_marks.yaml \\
        --outdir results/precursor_260908
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from figreq_260908_numbers import load, rel_index  # noqa: E402

FRAME_H = 1 / 12
K = 20                      # 遡るサイクル数
HOURS = 24                  # 時間で遡る範囲（1 h ビン）
BASE = range(-20, -10)      # self モードの基準 lag
POS_SPLIT = 17              # Pos1–17 と Pos21 以降（fig260908:position_bias）

# (列名, 図の名前, 単位)。サイズ（出生時・分裂時の大きさ）は screen には入れるが、図の主軸にはしない。
FEATURES = [
    ("interval_h", "generation time", "h"),
    ("mu_M", "d ln M/dt", "1/h"),
    ("dMdt", "dM/dt", "pg/h"),
    ("mu_V", "d ln V/dt", "1/h"),
    ("dLdt", "elongation rate dL/dt", "µm/h"),
    ("w_mean", "width", "µm"),
    ("w_div", "width at division", "µm"),
    ("dwdt", "width change in cycle", "µm/h"),
    ("rho_mean", "mass density", "pg/µm³"),
    ("drho_dt", "density change in cycle", "pg/µm³/h"),
    ("lnM_resid", "deviation from exp growth (SD of ln M residual)", ""),
    ("rho_resid", "density fluctuation (SD of residual)", "pg/µm³"),
    ("w_resid", "width fluctuation (SD of residual)", "µm"),
    ("div_mass_ratio", "division mass ratio (mother kept)", ""),
    ("div_vol_ratio", "division volume ratio (mother kept)", ""),
    ("invalid_frac", "fraction of invalid frames", ""),
    ("M_birth", "mass at birth", "pg"),
    ("M_div", "mass at division", "pg"),
    ("L_birth", "length at birth", "µm"),
    ("L_div", "length at division", "µm"),
    ("V_birth", "volume at birth", "µm³"),
    ("V_div", "volume at division", "µm³"),
    ("added_M", "added mass", "pg"),
    ("added_L", "added length", "µm"),
    ("aspect_div", "length/width at division", ""),
    ("MV_div", "density at division", "pg/µm³"),
]
FCOLS = [f for f, _, _ in FEATURES]
LABEL = {f: lab for f, lab, _ in FEATURES}
UNIT = {f: u for f, _, u in FEATURES}
# 世代ごとの図（Wakamoto 2017 Fig 2B/C の形）の縦軸
GEN_AXES = ["interval_h", "mu_M", "rho_mean", "w_mean", "dLdt", "lnM_resid", "div_mass_ratio"]
# 時間軸の比較に使う frame ごとの量（サイクルの中の位置で割ってから比べる）
HOURLY = [("w", "width"), ("rho", "mass density"), ("dlnM", "d ln M/dt"), ("dL", "dL/dt")]
# 量ごとの線の色（Okabe-Ito から、生死の意味を持つ青・朱を除いたもの。図をまたいで同じ量は同じ色）
QCOLOR = {"mu_M": "#000000", "dlnM": "#000000", "dLdt": "#E69F00", "dL": "#E69F00", "w_div": "#009E73", "w": "#009E73",
          "rho_mean": "#CC79A7", "rho": "#CC79A7", "interval_h": "#56B4E9", "div_mass_ratio": "0.55"}


# ---------------------------------------------------------------- features

def _lfit(t, y):
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    return coef[0], (r.std(ddof=2) if len(y) > 2 else np.nan)


def cycle_features(cyc: pd.DataFrame, m: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """1 サイクル 1 行。出生時 / 分裂時 = 先頭 / 末尾の有効 3 frame の中央値。速さは有効 frame の直線 fit の傾き。"""
    acc = ev[ev.accepted]
    ratio = {(r.pos, r.ch, r.frame): (r.mass_ratio, r.volume_ratio) for r in acc.itertuples()}
    grp = {k: g for k, g in m.sort_values("frame").groupby(["pos", "ch"])}
    rows = []
    for r in cyc.itertuples():
        g = grp[(r.pos, r.ch)]
        seg = g[(g.frame >= r.start_frame) & (g.frame < r.end_frame)]
        s = seg[seg.review_valid]
        o = dict(pos=r.pos, ch=r.ch, start_frame=r.start_frame, end_frame=r.end_frame, t_start=r.t_start, t_end=r.t_end,
                 interval_h=r.t_end - r.t_start, invalid_frac=1 - len(s) / max(r.end_frame - r.start_frame, 1))
        o["div_mass_ratio"], o["div_vol_ratio"] = ratio.get((r.pos, r.ch, r.start_frame), (np.nan, np.nan))
        if len(s) >= 5:
            t = s.t.to_numpy()
            M, V, L = s.phase_mass.to_numpy(), s.volume_um3_efd.to_numpy(), s.long_axis_um.to_numpy()
            w, rho = s.short_axis_um.to_numpy(), s.density_pg_um3_review_efd.to_numpy()
            h, e = s.iloc[:3], s.iloc[-3:]
            o.update(M_birth=h.phase_mass.median(), M_div=e.phase_mass.median(),
                     V_birth=h.volume_um3_efd.median(), V_div=e.volume_um3_efd.median(),
                     L_birth=h.long_axis_um.median(), L_div=e.long_axis_um.median(),
                     w_div=e.short_axis_um.median(), w_mean=np.median(w), rho_mean=rho.mean())
            o["mu_M"], o["lnM_resid"] = _lfit(t, np.log(M))
            o["mu_V"], _ = _lfit(t, np.log(V))
            o["dMdt"], _ = _lfit(t, M)
            o["dLdt"], _ = _lfit(t, L)
            o["dwdt"], o["w_resid"] = _lfit(t, w)
            o["drho_dt"], o["rho_resid"] = _lfit(t, rho)
        rows.append(o)
    c = pd.DataFrame(rows)
    c["added_M"] = c.M_div - c.M_birth
    c["added_L"] = c.L_div - c.L_birth
    c["aspect_div"] = c.L_div / c.w_div
    c["MV_div"] = c.M_div / c.V_div
    c["mid"] = c.pos + "_" + c.ch
    return c


def frame_features(m: pd.DataFrame, acc: pd.DataFrame, surv_mids: set) -> pd.DataFrame:
    """frame ごとの width・密度・d ln M/dt・dL/dt。速さはサイクルの中だけで ±6 frame（1 h 幅）の直線 fit の傾き。
    完結サイクルの frame は、生存系列のサイクル内の位置（20 区間）の中央値で割った値（*_n）も持つ。"""
    out = []
    for (pos, ch), g in m[m.review_valid].sort_values("frame").groupby(["pos", "ch"]):
        f = g.frame.to_numpy()
        divs = np.sort(acc[(acc.pos == pos) & (acc.ch == ch)].frame.to_numpy())
        cid = np.searchsorted(divs, f, side="right")          # 0 = 最初の分裂より前
        nxt = np.r_[divs, np.inf][cid]
        prv = np.r_[-np.inf, divs][cid]
        ph = np.where(np.isfinite(nxt) & np.isfinite(prv), (f - prv) / (nxt - prv), np.nan)
        t = g.t.to_numpy()
        lnM, L = np.log(g.phase_mass.to_numpy()), g.long_axis_um.to_numpy()
        dlnM, dL = np.full(len(f), np.nan), np.full(len(f), np.nan)
        for i in range(len(f)):
            sel = (cid == cid[i]) & (np.abs(f - f[i]) <= 6)
            if sel.sum() >= 7:
                dlnM[i] = _lfit(t[sel], lnM[sel])[0]
                dL[i] = _lfit(t[sel], L[sel])[0]
        out.append(pd.DataFrame(dict(mid=f"{pos}_{ch}", frame=f, t=t, phase=ph, cycle=cid, w=g.short_axis_um.to_numpy(),
                                     rho=g.density_pg_um3_review_efd.to_numpy(), dlnM=dlnM, dL=dL, M=g.phase_mass.to_numpy(),
                                     L=L, V=g.volume_um3_efd.to_numpy())))
    fr = pd.concat(out, ignore_index=True)
    pb = np.minimum((fr.phase * 20).fillna(-1).astype(int), 19)
    ref = fr[fr.mid.isin(surv_mids) & (pb >= 0)]
    for col, _ in HOURLY:
        prof = ref.groupby(pb[ref.index])[col].median()
        fr[col + "_n"] = np.where(pb >= 0, fr[col] / pb.map(prof), np.nan)
    return fr


def channel_adjust(c: pd.DataFrame, fr: pd.DataFrame, surv_mids: set):
    """channel 番号（ch00–ch11 = 視野の中の位置）による偏りを除く。生存系列で見ると幅は ch00–02 で 3.88 µm、
    ch10–11 で 3.68–3.70 µm と単調に違い、死亡は ch00–03 に多い。サイクルの量は生存系列の channel ごとの中央値を引いて
    全体の中央値を足す。frame の *_n は生存系列の channel ごとの中央値で割る。Pos 群の偏りは control の選び方で除く。"""
    c, fr = c.copy(), fr.copy()
    c["chn"] = c.ch.str[2:].astype(int)
    sv = c[c.mid.isin(surv_mids)]
    for f in FCOLS:
        c[f] = c[f] - c.chn.map(sv.groupby("chn")[f].median()) + sv[f].median()
    fr["chn"] = fr.mid.str.split("_ch").str[1].astype(int)
    fs = fr[fr.mid.isin(surv_mids)]
    for col, _ in HOURLY:
        fr[col + "_n"] = fr[col + "_n"] / fr.chn.map(fs.groupby("chn")[col + "_n"].median())
    return c, fr


# ---------------------------------------------------------------- matched comparison

class Matcher:
    """死亡系列ごとの matched control と、anchor から遡った値の配列（lag × 量）を作る。"""

    def __init__(self, c, fr, moth, acc):
        self.cg = {k: g.sort_values("start_frame") for k, g in c.groupby("mid")}
        self.fg = {k: g for k, g in fr.groupby("mid")}
        self.moth = moth
        self.accf = {f"{p}_{h}": np.sort(g.frame.to_numpy()) for (p, h), g in acc.groupby(["pos", "ch"])}
        self.cache = {}

    def controls(self, anchor: int, grp: str, exclude=()):
        s = self.moth[(~self.moth.dead) & (self.moth.posgrp == grp)
                      & (self.moth.last_valid_frame >= min(anchor + 24, 1202)) & ~self.moth.mid.isin(exclude)]
        out = []
        for mid in s.mid:
            f = self.accf.get(mid, np.array([]))
            if len(f) == 0:
                continue
            j = np.argmin(np.abs(f - anchor))
            if abs(f[j] - anchor) <= 18:
                out.append((mid, int(f[j])))
        return out

    def cyc_lags(self, mid, anchor):
        key = ("c", mid, anchor)
        if key not in self.cache:
            g = self.cg[mid]
            b = g[g.end_frame <= anchor].iloc[::-1].iloc[:K]
            a = np.full((K, len(FCOLS)), np.nan)
            a[:len(b)] = b[FCOLS].to_numpy(float)
            self.cache[key] = a[::-1]                      # 行 0 = lag −K、行 K−1 = lag −1
        return self.cache[key]

    def hour_lags(self, mid, anchor):
        key = ("h", mid, anchor)
        if key not in self.cache:
            g = self.fg[mid]
            lag = (g.frame.to_numpy() - anchor) * FRAME_H
            b = np.floor(lag).astype(int)                 # −1 = anchor の前 1 h
            a = np.full((HOURS, len(HOURLY)), np.nan)
            for j, (col, _) in enumerate(HOURLY):
                v = g[col + "_n"].to_numpy()
                for k in range(-HOURS, 0):
                    x = v[(b == k) & np.isfinite(v)]
                    if len(x) >= 4:
                        a[k + HOURS, j] = x.mean()
            self.cache[key] = a
        return self.cache[key]


def self_baseline(a):
    """lag −20..−11 の中央値を引く（4 サイクル以上なければ全部 NaN）。"""
    base = a[[k + K for k in BASE]]
    ok = np.isfinite(base).sum(0) >= 4
    med = np.where(ok, np.nanmedian(np.where(np.isfinite(base), base, np.nan), axis=0), np.nan)
    return a - med


def auc_core(cases, ctrls, ctrl_ids, boot_w=None, case_w=None):
    """cases: list of (L × F)。ctrls: list of (J_i × L × F)。ctrl_ids: list of 長さ J_i の id。
    戻り値: auc (L × F), n (L × F), bootstrap (B × L × F) または None。"""
    U, OK = [], []
    Ind, Msk, Ids = [], [], []
    for x, Y, ids in zip(cases, ctrls, ctrl_ids):
        valid = np.isfinite(Y) & np.isfinite(x)[None]
        ind = np.where(valid, (Y < x[None]) + 0.5 * (Y == x[None]), 0.0)
        nn = valid.sum(0)
        ok = nn >= 5
        U.append(np.where(ok, ind.sum(0) / np.maximum(nn, 1), np.nan))
        OK.append(ok)
        Ind.append(ind), Msk.append(valid.astype(float)), Ids.append(np.asarray(ids))
    U, OK = np.array(U), np.array(OK)
    auc = np.nanmean(U, 0)
    n = OK.sum(0)
    auc[n < 6] = np.nan
    if boot_w is None:
        return auc, n, None, U
    B = boot_w.shape[0]
    Ub = np.full((B, len(cases)) + auc.shape, np.nan)
    for i, (ind, msk, ids) in enumerate(zip(Ind, Msk, Ids)):
        w = boot_w[:, ids]                                   # B × J_i
        num = np.tensordot(w, ind, axes=(1, 0))
        den = np.tensordot(w, msk, axes=(1, 0))
        Ub[:, i] = np.where(OK[i][None] & (den > 0), num / np.where(den > 0, den, 1), np.nan)
    cw = case_w[:, :, None, None] * np.isfinite(Ub)
    ab = np.nansum(Ub * case_w[:, :, None, None], 1) / np.maximum(cw.sum(1), 1e-9)
    ab[:, n < 6] = np.nan
    return auc, n, ab, U


def run_screen(match, deaths, kind, mode, B, n_null, rng):
    """kind: 'cyc' or 'hour'。mode: 'abs' or 'self'（self は cyc のみ）。"""
    lagf = match.cyc_lags if kind == "cyc" else match.hour_lags
    tf = self_baseline if mode == "self" else (lambda a: a)
    surv_ids = {mid: i for i, mid in enumerate(match.moth[~match.moth.dead].mid)}
    cases, ctrls, ids, meta = [], [], [], []
    for r in deaths.itertuples():
        cl = match.controls(r.anchor, r.posgrp)
        if not cl:
            continue
        cases.append(tf(lagf(r.mid, r.anchor)))
        ctrls.append(np.array([tf(lagf(cm, a)) for cm, a in cl]))
        ids.append([surv_ids[cm] for cm, _ in cl])
        meta.append((r.mid, r.anchor, r.posgrp))
    NI, NJ = len(cases), len(surv_ids)
    bw = np.stack([np.bincount(rng.integers(0, NJ, NJ), minlength=NJ) for _ in range(B)]).astype(float)
    cw = np.stack([np.bincount(rng.integers(0, NI, NI), minlength=NI) for _ in range(B)]).astype(float)
    auc, n, ab, U = auc_core(cases, ctrls, ids, bw, cw)
    # null: 生存系列に同じ anchor を割り当てる
    nulls = []
    for _ in range(n_null):
        nc, nC, nI = [], [], []
        for (mid, anchor, grp) in meta:
            cl = match.controls(anchor, grp)
            pick = cl[rng.integers(len(cl))]
            rest = [x for x in cl if x[0] != pick[0]]
            nc.append(tf(lagf(*pick)))
            nC.append(np.array([tf(lagf(cm, a)) for cm, a in rest]))
            nI.append([surv_ids[cm] for cm, _ in rest])
        nulls.append(auc_core(nc, nC, nI)[0])
    nulls = np.array(nulls)
    lo, hi = np.nanpercentile(ab, [2.5, 97.5], axis=0)
    nlo, nhi = np.nanpercentile(nulls, [2.5, 97.5], axis=0)
    return dict(auc=auc, n=n, lo=lo, hi=hi, nlo=nlo, nhi=nhi, nulls=nulls, boot=ab, U=U, meta=meta)


def onset(res, j):
    """lag −1 から遡って、AUC が null の 95% 帯の外に途切れずにある一番古い lag（無ければ None）。"""
    a, lo, hi = res["auc"][:, j], res["nlo"][:, j], res["nhi"][:, j]
    k0 = None
    for i in range(len(a) - 1, -1, -1):
        if np.isfinite(a[i]) and (a[i] > hi[i] or a[i] < lo[i]):
            k0 = i - len(a)
        else:
            break
    return k0


def screen_table(res, names):
    rows = []
    L = res["auc"].shape[0]
    for j, f in enumerate(names):
        for i in range(L):
            rows.append(dict(feature=f, lag=i - L, auc=res["auc"][i, j], ci_lo=res["lo"][i, j],
                             ci_hi=res["hi"][i, j], null_lo=res["nlo"][i, j], null_hi=res["nhi"][i, j], n_dying=int(res["n"][i, j])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- survival

def km(time, event):
    order = np.argsort(time, kind="stable")
    t, e = np.asarray(time)[order], np.asarray(event)[order]
    at_risk, s, ts, ss = len(t), 1.0, [0.0], [1.0]
    for u in np.unique(t):
        d = e[t == u].sum()
        if d:
            s *= 1 - d / at_risk
            ts.append(u), ss.append(s)
        at_risk -= (t == u).sum()
    return np.array(ts), np.array(ss)


def exp_rate(time, event):
    """打ち切りありの指数分布の最尤推定 λ = 死亡数 / 総観察量。CI は Poisson の正確な区間。"""
    D, T = int(np.sum(event)), float(np.sum(time))
    lo = stats.chi2.ppf(0.025, 2 * D) / 2 / T if D else 0.0
    hi = stats.chi2.ppf(0.975, 2 * D + 2) / 2 / T
    return D / T, lo, hi, D, T


# ---------------------------------------------------------------- figures

def _style():
    import matplotlib.pyplot as plt
    try:
        plt.style.use("paper")
    except OSError:
        plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6,
                             "ytick.labelsize": 6, "legend.fontsize": 6, "axes.spines.top": False,
                             "axes.spines.right": False, "xtick.direction": "in", "ytick.direction": "in",
                             "pdf.fonttype": 42, "lines.linewidth": 0.8, "legend.frameon": False,
                             "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                             "axes.prop_cycle": plt.cycler(color=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
                                                                  "#56B4E9", "#000000", "#F0E442"])})  # Okabe-Ito
    try:
        from qpi_colors import fate_color
        return fate_color("survivor"), fate_color("non_survivor")
    except ImportError:
        return "#0072B2", "#D55E00"   # qpi_colors の survivor / non_survivor と同じ Okabe-Ito の青・朱


def _save(fig, name, caption, data, params, outdir):
    try:
        from figure_logger import save_figure
        for fmt in ("pdf", "png"):
            save_figure(fig, params=params, description=name, caption=caption, data=data, fmt=fmt, dpi=300)
    except Exception as e:  # noqa: BLE001  logger が使えない環境でも図は残す
        print(f"[precursor] save_figure failed ({e}); writing to {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{fmt}", dpi=300, bbox_inches="tight")
    (outdir / f"{name}.caption.txt").write_text(caption, encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", type=Path, default=Path("qc_review/260908_outside_quad/gallery_100h_v3"))
    ap.add_argument("--deaths", type=Path, default=Path("docs/260908_death_marks.yaml"))
    ap.add_argument("--outdir", type=Path, default=Path("results/precursor_260908"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--no-channel-adjust", action="store_true", help="channel 番号の偏りを除かない（比較用）")
    ap.add_argument("--replot", action="store_true", help="計算せず <outdir>/state.pkl から図だけ描き直す")
    args = ap.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)  # 全部 NaN の列の nanmean / nanmedian
    rng = np.random.default_rng(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        with open(args.outdir / "state.pkl", "rb") as fh:
            make_figures(args, **pickle.load(fh))
        return

    ch, inc, cyc, ev, m, deaths = load(args.pkg, args.deaths)
    keys = set(zip(inc.pos, inc.ch))
    m = m[[k in keys for k in zip(m.pos, m.ch)]]
    ev = ev[[k in keys for k in zip(ev.pos, ev.ch)]]
    acc = ev[ev.accepted]
    deaths = deaths.rename(columns={"death_cycle_start_frame": "anchor"})
    deaths["mid"] = deaths.pos + "_" + deaths.ch
    deaths["posgrp"] = np.where(deaths.pos.str[3:].astype(int) <= POS_SPLIT, "P1-17", "P21+")
    deaths["anchor_h"] = (deaths.anchor - 2) * FRAME_H

    moth = inc[["pos", "ch"]].copy()
    moth["mid"] = moth.pos + "_" + moth.ch
    moth["dead"] = moth.mid.isin(deaths.mid)
    moth["posgrp"] = np.where(moth.pos.str[3:].astype(int) <= POS_SPLIT, "P1-17", "P21+")
    moth["last_valid_frame"] = moth.mid.map(m[m.review_valid].assign(mid=lambda x: x.pos + "_" + x.ch).groupby("mid").frame.max())

    c = cycle_features(cyc, m, ev)
    c["rel"] = rel_index(c, deaths.rename(columns={"anchor": "death_cycle_start_frame"}))
    c["dead"] = c.mid.isin(deaths.mid)
    c.to_csv(args.outdir / "cycle_features.csv", index=False)
    fr = frame_features(m, acc, set(moth.mid[~moth.dead]))
    sv = c[~c.dead].assign(chn=lambda x: x.ch.str[2:].astype(int)).groupby(["chn", "mid"])[["w_mean", "rho_mean", "interval_h"]]
    print("生存系列の channel ごとの値（母細胞の中央値の中央値）:")
    print(sv.median().groupby("chn").median().round(3).T.to_string())
    print("死亡 / 系列数:", moth.assign(chn=moth.ch.str[2:].astype(int)).groupby("chn").dead.agg(["sum", "size"]).T.to_string())
    if not args.no_channel_adjust:
        c, fr = channel_adjust(c, fr, set(moth.mid[~moth.dead]))
        c.to_csv(args.outdir / "cycle_features_channel_adjusted.csv", index=False)
    match = Matcher(c, fr, moth, acc)

    print("## 0. 対象")
    print(f"母細胞 {len(moth)}（死亡 {int(moth.dead.sum())}、生存 {int((~moth.dead).sum())}）。"
          f"anchor から mark までの時間: 中央値 {np.median(deaths.mark_h - deaths.anchor_h):.1f} h "
          f"（{(deaths.mark_h - deaths.anchor_h).min():.1f}–{(deaths.mark_h - deaths.anchor_h).max():.1f} h）")

    # ---- 1. サイクル単位の screen
    res = {}
    for mode in ("abs", "self"):
        r = run_screen(match, deaths, "cyc", mode, args.boot, args.null, rng)
        res[mode] = r
        tab = screen_table(r, FCOLS)
        tab.to_csv(args.outdir / f"screen_cycles_{mode}.csv", index=False)
        print(f"\n## 1. サイクル単位（mode={mode}）。AUC、* = null の 95% 帯の外。列 = 最後の分裂から数えたサイクル")
        show = tab[tab.lag >= -12].copy()
        show["s"] = show.auc.round(2).astype(str) + np.where((show.auc > show.null_hi) | (show.auc < show.null_lo), "*", " ")
        print(show.pivot(index="feature", columns="lag", values="s").reindex(FCOLS).to_string())
        print("onset:", {f: onset(r, j) for j, f in enumerate(FCOLS) if onset(r, j) is not None})
        print("n（死亡系列）lag −12..−1:", r["n"][-12:, 0].tolist())

    print("\n## 1b. 系列ごと: matched control の 90 パーセンタイルを超える（AUC < 0.5 の量は 10 パーセンタイルを下回る）死亡系列の割合。"
          "null では 10%")
    r = res["abs"]
    for f in ["mu_M", "dLdt", "w_div", "w_mean", "rho_mean", "interval_h", "lnM_resid"]:
        j = FCOLS.index(f)
        up = np.nanmean(r["auc"][-3:, j]) > 0.5
        fr_ = []
        for i in range(K - 6, K):
            u = r["U"][:, i, j]
            u = u[np.isfinite(u)]
            fr_.append(f"{np.mean(u >= 0.9 if up else u <= 0.1):.0%}")
        print(f"  {f:12s} lag −6..−1:", " ".join(fr_))
    sw = deaths[deaths.kind == "swelling"]
    rs = run_screen(match, sw, "cyc", "abs", 200, max(args.null // 2, 20), rng)
    print(f"\n## 1c. 膨潤死だけ（{len(sw)} 系列）の onset:", {f: onset(rs, j) for j, f in enumerate(FCOLS) if onset(rs, j) is not None})
    late = deaths[deaths.anchor_h >= 20]
    rl = run_screen(match, late, "cyc", "abs", 200, max(args.null // 2, 20), rng)
    print(f"## 1d. 最後の分裂が 20 h 以降（{len(late)} 系列）の onset:", {f: onset(rl, j) for j, f in enumerate(FCOLS) if onset(rl, j) is not None})

    # ---- 2. 時間単位
    rh = run_screen(match, deaths, "hour", "abs", args.boot, args.null, rng)
    tabh = screen_table(rh, [h for h, _ in HOURLY])
    tabh.to_csv(args.outdir / "screen_hours.csv", index=False)
    print("\n## 2. 時間単位（最後の分裂の何時間前か。1 h ビン、サイクル内の位置で割った値）")
    sh = tabh[tabh.lag >= -16].copy()
    sh["s"] = sh.auc.round(2).astype(str) + np.where((sh.auc > sh.null_hi) | (sh.auc < sh.null_lo), "*", " ")
    print(sh.pivot(index="feature", columns="lag", values="s").to_string())
    print("onset (h):", {h: onset(rh, j) for j, (h, _) in enumerate(HOURLY)})

    # ---- 3. 生存曲線（Nakaoka & Wakamoto 2017 Fig 2D/E の形）
    ndiv = acc.assign(mid=acc.pos + "_" + acc.ch).groupby("mid").frame.apply(np.array)
    T, E, G = [], [], []
    for mo in moth.itertuples():
        if mo.dead:
            r = deaths[deaths.mid == mo.mid].iloc[0]
            T.append(r.mark_h), E.append(1), G.append(int((ndiv[mo.mid] <= r.anchor).sum()))
        else:
            T.append((mo.last_valid_frame - 2) * FRAME_H), E.append(0), G.append(int((ndiv[mo.mid] <= mo.last_valid_frame).sum()))
    moth["T"], moth["E"], moth["G"] = T, E, G
    moth.to_csv(args.outdir / "mothers_survival.csv", index=False)
    lam = exp_rate(moth["T"], moth["E"])
    lamg = exp_rate(moth["G"], moth["E"])
    print("\n## 3. 生存")
    print(f"死亡率（時間）{lam[0]:.4f} /h（95% CI {lam[1]:.4f}–{lam[2]:.4f}）、半減 {np.log(2) / lam[0]:.0f} h、"
          f"死亡 {lam[3]} / 観察 {lam[4]:.0f} 母細胞・時間")
    print(f"死亡率（世代）{lamg[0]:.4f} /世代（{lamg[1]:.4f}–{lamg[2]:.4f}）、半減 {np.log(2) / lamg[0]:.0f} 世代")
    for a, b in [(0, 25), (25, 50), (50, 75), (75, 100)]:
        ex = np.clip(moth["T"], a, b) - a
        dd = ((moth.E == 1) & (moth["T"] >= a) & (moth["T"] < b)).sum()
        print(f"  {a}–{b} h: 死亡 {dd}、観察 {ex.sum():.0f} 母細胞・時間、率 {dd / ex.sum():.4f} /h")

    # ---- 4. 世代ごと（Fig 2C の形）: 開始から数える / 最後の分裂から逆に数える
    c["gen"] = c.sort_values("start_frame").groupby("mid").cumcount()
    fwd = c[(~c.dead) | (c.rel < 0)].assign(group=lambda x: np.where(x.dead, "extinct", "survived"))
    fwd_t = fwd.groupby(["group", "gen"])[GEN_AXES].agg(["mean", "std", "count"])
    fwd_t.to_csv(args.outdir / "per_generation_forward.csv")
    back = []
    for r in deaths.itertuples():
        a = match.cyc_lags(r.mid, r.anchor)
        for i in range(K):
            back.append(dict(group="extinct", lag=i - K, mid=r.mid, **dict(zip(FCOLS, a[i]))))
        for cm, an in match.controls(r.anchor, r.posgrp):
            a = match.cyc_lags(cm, an)
            for i in range(K):
                back.append(dict(group="matched survived", lag=i - K, mid=cm, of=r.mid, **dict(zip(FCOLS, a[i]))))
    back = pd.DataFrame(back)
    back.to_csv(args.outdir / "per_generation_backward.csv", index=False)
    print("\n## 4. 最後の分裂から数えた世代（死亡系列 / matched 生存系列）の平均")
    bt = back.groupby(["group", "lag"])[GEN_AXES].mean().unstack(0)
    print(bt.loc[-8:].round(3).to_string())

    state = dict(c=c, fr=fr, deaths=deaths, moth=moth, acc=acc, res=res, rh=rh, fwd=fwd, back=back, lam=lam, lamg=lamg)
    with open(args.outdir / "state.pkl", "wb") as fh:
        pickle.dump(state, fh)
    if args.no_fig:
        return
    make_figures(args, **state)


def make_figures(args, c, fr, deaths, moth, acc, res, rh, fwd, back, lam, lamg):
    import matplotlib.pyplot as plt
    c_s, c_d = _style()
    MM = 1 / 25.4
    cond = ("S. pombe mother cells in YE, mother machine, 5 min/frame, 0–100 h (260908_outside_quad, gallery_100h_v3; "
            "background outside_quad, EFD volume, RI not calibrated, n_medium 1.333). Strain and temperature: see "
            "docs/FIGURE_REQUIREMENTS_260908.md §6.")
    params = dict(pkg=str(args.pkg), deaths=str(args.deaths), boot=args.boot, null=args.null, seed=args.seed,
                  pos_split=POS_SPLIT, K=K)
    nd, ns = int(moth.dead.sum()), int((~moth.dead).sum())
    adj = ("" if args.no_channel_adjust else
           " Cycle values are adjusted for channel index (ch00–ch11; surviving lineages are 0.2 µm wider in ch00–02 than in ch10–11): "
           "the surviving lineages' median for that channel index is subtracted and their overall median added back; frame values are "
           "divided by the surviving lineages' median for that channel index.")

    # ---- Fig A: survival
    fig, ax = plt.subplots(1, 2, figsize=(183 * MM, 60 * MM))
    for i, (col, rate, xl) in enumerate([("T", lam, "time [h]"), ("G", lamg, "generation (divisions of the mother)")]):
        t, s = km(moth[col].to_numpy(float), moth.E.to_numpy())
        ax[i].step(t, s, where="post", color="k")
        cens = moth[moth.E == 0][col]
        ax[i].plot(cens, [s[np.searchsorted(t, x, side="right") - 1] for x in cens], "|", color="0.5", ms=4)
        xx = np.linspace(0, moth[col].max(), 100)
        ax[i].plot(xx, np.exp(-rate[0] * xx), "--", color="0.3")
        ax[i].set_yscale("log")
        ax[i].set_ylim(0.5, 1.02)
        ax[i].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax[i].set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1"])
        ax[i].set_xlabel(xl)
        ax[i].set_ylabel("surviving fraction")
        u = "/h" if col == "T" else "/generation"
        ax[i].text(0.03, 0.05, f"λ = {rate[0]:.4f} {u} (95% CI {rate[1]:.4f}–{rate[2]:.4f})", transform=ax[i].transAxes)
        ax[i].text(-0.12, 1.02, "ab"[i], transform=ax[i].transAxes, fontweight="bold", fontsize=8)
    fig.tight_layout()
    cap = (f"Mother-cell deaths in YE occur at a roughly constant rate. (a) Kaplan–Meier surviving fraction of mother-cell lineages "
           f"against time; (b) against generation. ★ Death time = the time of the death mark placed on visual review "
           f"(docs/260908_death_marks.yaml); generation at death = number of accepted divisions of the mother up to its last division. "
           f"Lineages without a mark are censored at their last valid frame (ticks). Dashed line, exponential fit S = exp(−λx) with "
           f"λ = deaths / total observed time (or generations), the maximum-likelihood estimate under censoring; CI, exact Poisson. "
           f"n = {nd + ns} mother cells ({nd} deaths). Log y-axis. {cond}")
    _save(fig, "fig260908_survival", cap, dict(T=moth["T"].to_numpy(), G=moth.G.to_numpy(), E=moth.E.to_numpy()), params, args.outdir)
    plt.close(fig)

    # ---- Fig B: per generation (Wakamoto Fig 2B/C の形 + 死から逆に数えた版)
    axes_ = [a for a in ["interval_h", "mu_M", "rho_mean", "w_mean", "dLdt", "lnM_resid", "div_mass_ratio"]]
    ylab = {"interval_h": "generation time [h]", "mu_M": "d ln M/dt [1/h]", "rho_mean": "mass density [pg/µm³]",
            "w_mean": "width [µm]", "dLdt": "dL/dt [µm/h]", "lnM_resid": "ln M residual SD", "div_mass_ratio": "mother's mass share"}
    ex_d = deaths.sort_values("anchor", ascending=False)
    ex_d = ex_d[ex_d.kind == "swelling"].iloc[0]
    sv = c[~c.dead].groupby("mid").agg(n=("gen", "size"), Td=("interval_h", "median"), grp=("pos", "first"))
    sv = sv[(sv.n >= 45)]
    ex_s = (sv.Td - c[~c.dead].interval_h.median()).abs().idxmin()
    fig, ax = plt.subplots(len(axes_), 3, figsize=(183 * MM, 30 * MM * len(axes_)), sharey="row")
    for i, f in enumerate(axes_):
        # (B) representative
        for mid, col, lab in [(ex_s, c_s, f"survived ({ex_s})"), (ex_d.mid, c_d, f"extinct ({ex_d.mid})")]:
            g = c[c.mid == mid].sort_values("start_frame")
            if mid == ex_d.mid:
                g = g[g.rel < 0]
            ax[i, 0].plot(g.gen, g[f], color=col, label=lab)
        # (C) forward
        for grp, col in [("survived", c_s), ("extinct", c_d)]:
            q = fwd[fwd.group == grp].groupby("gen")[f].agg(["mean", "std", "count"])
            q = q[q["count"] >= 5]
            ax[i, 1].errorbar(q.index, q["mean"], q["std"], fmt="o", ms=1.5, lw=0.5, color=col, capsize=0,
                              label=f"{grp} (n ≥ 5 per generation)")
        # (C') backward
        for grp, col in [("matched survived", c_s), ("extinct", c_d)]:
            q = back[back.group == grp].groupby("lag")[f].agg(["mean", "std", "count"])
            q = q[q["count"] >= 5]
            ax[i, 2].errorbar(q.index, q["mean"], q["std"], fmt="o", ms=2, lw=0.6, color=col, capsize=0, label=grp)
        ax[i, 0].set_ylabel(ylab[f])
        lo, hi = np.nanpercentile(c[f], [0.5, 99.5])
        pad = (hi - lo) * 0.3
        ax[i, 0].set_ylim(lo - pad, hi + pad)
    for j, xl in enumerate(["generation", "generation (from start)", "generations before the last division"]):
        ax[-1, j].set_xlabel(xl)
    for i in range(len(axes_)):
        ax[i, 2].set_xticks([-20, -15, -10, -5, -1])
    ax[0, 0].legend(loc="upper left")
    ax[0, 1].legend(loc="upper left")
    ax[0, 2].legend(loc="upper left")
    for j, t in enumerate(["representative lineages", "aligned at the start", "aligned at death"]):
        ax[0, j].set_title(t)
    fig.tight_layout()
    cap = ("Cycle-level quantities do not separate dying from surviving lineages when aligned at the start, but do in the last "
           "cycles when aligned at death (cf. Nakaoka & Wakamoto 2017, PLoS Biol, Fig 2B,C). Rows: generation time, specific mass "
           "production rate d ln M/dt, cycle-mean mass density, cycle-median width, elongation rate dL/dt, SD of the residual of the "
           "ln M fit, and the mother's share of mass at the division that starts the cycle. Left: one surviving and one extinct lineage. "
           "Middle: mean ± SD per generation counted from the first complete cycle (survived: all cycles of unmarked lineages; extinct: "
           "complete cycles before the last division); points with ≥ 5 cycles. Right: generations counted back from the last division "
           "(−1 = last complete cycle); matched survived = unmarked lineages in the same Pos group (Pos1–17 / Pos21+) at the division "
           "nearest to each dying lineage's last division (±1.5 h), pooled over all matches; mean ± SD. ★ Generation time = time between "
           "consecutive accepted divisions. d ln M/dt = slope of ln(dry mass) vs time over the valid frames of the cycle. Mass density = "
           "dry mass / EFD volume, averaged over the cycle. Width = median over the cycle of the short axis (mean length of the chords "
           "normal to the centre line of the yellow contour, EFD K=6, over the body without the caps). dL/dt = slope of "
           "the long axis vs time. Mother's mass share = mass after / before the division (3 valid frames each side)." + adj +
           f" n = {ns} surviving and {nd} dying mother cells. {cond}")
    _save(fig, "fig260908_generation_traces", cap,
          {f"back_{g}_{f}": back[back.group == g].groupby("lag")[f].mean().to_numpy() for g in ["extinct", "matched survived"] for f in axes_},
          params, args.outdir)
    plt.close(fig)

    # ---- Fig C: AUC screen（位置は固定。(b) は (a) と同じ行に並べる）
    main_feats = ["mu_M", "dLdt", "w_div", "rho_mean", "interval_h", "div_mass_ratio"]
    nonsize = FCOLS[:FCOLS.index("M_birth")]
    fig = plt.figure(figsize=(183 * MM, 175 * MM))
    axh = fig.add_axes([0.30, 0.45, 0.40, 0.52])
    axc = fig.add_axes([0.715, 0.62, 0.012, 0.2])
    axo = fig.add_axes([0.80, 0.45, 0.17, 0.52])
    axl = fig.add_axes([0.08, 0.06, 0.50, 0.30])
    axt = fig.add_axes([0.68, 0.06, 0.29, 0.30])
    r = res["abs"]
    L = 12
    A = r["auc"][-L:].T
    im = axh.imshow(A, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
    for j in range(A.shape[0]):
        for i in range(L):
            if np.isfinite(A[j, i]) and (A[j, i] > r["nhi"][-L + i, j] or A[j, i] < r["nlo"][-L + i, j]):
                axh.text(i, j, "•", ha="center", va="center", fontsize=5)
    axh.axhline(len(nonsize) - 0.5, color="k", lw=0.6)
    axh.set_yticks(range(len(FCOLS)))
    axh.set_yticklabels([LABEL[f] for f in FCOLS], fontsize=5)
    axh.set_xticks(range(L))
    axh.set_xticklabels([str(k) for k in range(-L, 0)])
    axh.set_xlabel("cycles before the last division")
    fig.colorbar(im, cax=axc, label="AUC")
    fig.text(0.01, 0.975, "a", fontweight="bold", fontsize=8)
    # (b) onset（(a) と同じ行）
    for j, f in enumerate(FCOLS):
        a, s_ = onset(res["abs"], j), onset(res["self"], j)
        for k, yy, col in [(a, j - 0.18, "k"), (s_, j + 0.18, "0.6")]:
            if k is not None:
                axo.plot([k - 0.45, -0.55], [yy, yy], color=col, lw=1.8, solid_capstyle="butt")
        if a is None and s_ is None:
            axo.text(-1, j, "n.d.", va="center", ha="center", fontsize=5, color="0.4")
    axo.axhline(len(nonsize) - 0.5, color="k", lw=0.6)
    axo.set_ylim(len(FCOLS) - 0.5, -0.5)
    axo.set_yticks(range(len(FCOLS)))
    axo.set_yticklabels([])
    axo.set_xlim(-6.5, -0.5)
    axo.set_xticks(range(-6, 0))
    axo.set_xlabel("onset [cycles before\nthe last division]")
    axo.plot([], [], color="k", lw=1.8, label="vs other cells")
    axo.plot([], [], color="0.6", lw=1.8, label="vs own past")
    axo.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, borderaxespad=0.2)
    fig.text(0.77, 0.975, "b", fontweight="bold", fontsize=8)
    # (c) AUC vs lag for main features
    lags = np.arange(-L, 0)
    axl.fill_between(lags, np.nanmin(r["nlo"][-L:], 1), np.nanmax(r["nhi"][-L:], 1), color="0.88", lw=0, zorder=0,
                     label="null 95% range")
    for f in main_feats:
        j = FCOLS.index(f)
        axl.plot(lags, r["auc"][-L:, j], marker="o", ms=2, color=QCOLOR[f], label=LABEL[f],
                 ls="--" if f == "div_mass_ratio" else "-")
        axl.fill_between(lags, r["lo"][-L:, j], r["hi"][-L:, j], color=QCOLOR[f], alpha=0.12, lw=0)
    axl.axhline(0.5, color="0.5", lw=0.5)
    axl.set_ylim(0, 1)
    axl.set_xticks(lags)
    axl.set_xlabel("cycles before the last division")
    axl.set_ylabel("AUC (dying > matched survivors)")
    axl.legend(ncol=2, fontsize=5, loc="upper left")
    fig.text(0.01, 0.38, "c", fontweight="bold", fontsize=8)
    # (d) hourly
    hl = np.arange(-HOURS, 0) + 0.5
    axt.fill_between(hl, np.nanmin(rh["nlo"], 1), np.nanmax(rh["nhi"], 1), color="0.88", lw=0, zorder=0)
    for j, (h, lab) in enumerate(HOURLY):
        axt.plot(hl, rh["auc"][:, j], "o-", ms=1.5, color=QCOLOR[h], label=lab)
        axt.fill_between(hl, rh["lo"][:, j], rh["hi"][:, j], color=QCOLOR[h], alpha=0.12, lw=0)
    axt.axhline(0.5, color="0.5", lw=0.5)
    axt.set_xlim(-16, 0)
    axt.set_ylim(0, 1)
    axt.set_xlabel("hours before the last division")
    axt.set_ylabel("AUC")
    axt.legend(fontsize=5, loc="upper left")
    fig.text(0.62, 0.38, "d", fontweight="bold", fontsize=8)
    cap = ("Mass production and elongation slow and the cell widens about three cycles before the last division; density and "
           "generation time change only in the last cycle. (a) AUC for separating dying lineages from matched surviving lineages, "
           "per quantity (rows; below the line, sizes) and per complete cycle counted back from the last division (columns; −1 = last "
           "complete cycle). Red, larger in dying lineages; blue, smaller. Dots, outside the 95% range of a label-shuffled null "
           f"(surviving lineages given the dying lineages' anchors, {args.null} repeats). (b) Onset = the earliest cycle from which the "
           "AUC stays outside the null range through cycle −1; black, raw values (vs other cells); grey, change from the lineage's own "
           "median over cycles −20..−11 (vs own past; lineages with ≥ 4 such cycles); n.d., not detected; rows as in (a). (c) AUC against cycle for the main "
           f"quantities; coloured bands, 95% bootstrap CI ({args.boot} resamples of dying and of surviving lineages); grey band, null 95% "
           "range (envelope over quantities). (d) Same in 1-h bins before the last division, using frame values divided by the "
           "survivors' median at the same cell-cycle phase (20 bins). ★ AUC = mean over dying lineages of the mid-rank of its value "
           "among its matched controls; matched controls = unmarked lineages in the same Pos group (Pos1–17 / Pos21+) tracked ≥ 2 h past "
           "the dying lineage's last division, aligned at their division nearest to it (±1.5 h). Last division = last accepted division "
           "before the death mark. Cycle quantities as in fig260908_generation_traces; frame-level rates = slope over ±6 frames within "
           "one cycle (≥ 7 valid frames)." + adj +
           f" n = {nd} dying lineages (fewer at earlier cycles, see screen_cycles_abs.csv), {ns} surviving lineages. {cond}")
    _save(fig, "fig260908_precursor_screen", cap, dict(auc_abs=res["abs"]["auc"], auc_self=res["self"]["auc"], auc_hours=rh["auc"],
                                                       null_lo=res["abs"]["nlo"], null_hi=res["abs"]["nhi"]), params, args.outdir)
    plt.close(fig)

    # ---- Fig D: hours around the last division (raw traces)
    rows_ = [("w", "width [µm]"), ("rho", "mass density [pg/µm³]"), ("dlnM", "d ln M/dt [1/h]"), ("dL", "dL/dt [µm/h]"),
             ("M", "dry mass [pg]"), ("L", "length [µm]")]
    fig, ax = plt.subplots(len(rows_), 1, figsize=(89 * MM, 32 * MM * len(rows_)), sharex=True)
    match = Matcher(c, fr, moth, acc)
    dk = deaths.set_index("mid")
    fg = {k: g for k, g in fr.groupby("mid")}
    ctl_tr = []
    for r in deaths.itertuples():
        for cm, a in match.controls(r.anchor, r.posgrp):
            g = fg[cm]
            ctl_tr.append(g.assign(lag=(g.frame - a) * FRAME_H))
    ctl = pd.concat(ctl_tr)
    ctl = ctl[(ctl.lag >= -16) & (ctl.lag < 0)]
    ctl["b"] = np.floor(ctl.lag * 2) / 2
    for i, (col, yl) in enumerate(rows_):
        shown = []
        for mid in dk.index:
            g = fg[mid]
            lag = (g.frame - dk.loc[mid, "anchor"]) * FRAME_H
            sel = (lag >= -16) & (lag <= 10)
            y = g[col][sel].rolling(7, center=True, min_periods=3).median() if col in ("w", "rho") else g[col][sel]
            ax[i].plot(lag[sel], y, color=c_d, lw=0.3, alpha=0.35)
            shown.append(y.to_numpy())
        q = ctl.groupby("b")[col].quantile([0.1, 0.5, 0.9]).unstack()
        ax[i].fill_between(q.index + 0.25, q[0.1], q[0.9], color=c_s, alpha=0.25, lw=0)
        ax[i].plot(q.index + 0.25, q[0.5], color=c_s)
        ax[i].axvline(0, color="k", lw=0.5, ls=":")
        ax[i].set_ylabel(yl)
        lo, hi = np.nanpercentile(np.concatenate(shown), [0.5, 99.5])
        ax[i].set_ylim(lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.08)
        ax[i].set_title("abcdef"[i], loc="left", fontweight="bold", fontsize=8)
    ax[-1].set_xlabel("time from the last division [h]")
    ax[-1].set_xticks(range(-15, 11, 5))
    fig.tight_layout()
    cap = ("Individual dying lineages around their last division. Vermilion, each dying lineage (width and density: 7-frame running "
           "median); blue line and band, median and 10–90th percentile of the matched surviving lineages aligned at their division "
           "nearest to the dying lineage's last division (0.5-h bins, before 0 only; after 0 survivors divide again and are not shown). "
           "★ Width = short axis of the yellow contour; mass density = dry mass / EFD volume per frame; d ln M/dt and dL/dt = slope "
           "over ±6 frames within one cycle (≥ 7 valid frames); dry mass = (λ/2πα)·Σφ·A_px; length = long axis (centre-line arc length). "
           "Raw values (not adjusted for channel index); y-axes clipped to the 0.5–99.5th percentile of the dying lineages' values. "
           f"n = {nd} dying lineages; controls as in fig260908_precursor_screen. {cond}")
    _save(fig, "fig260908_around_last_division", cap, {}, params, args.outdir)
    plt.close(fig)


if __name__ == "__main__":
    main()
