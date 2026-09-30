"""How stable is the per-frame cell-free channel selection that drives drift control?

Production (compute_drift_online.py) averages the per-channel shifts of every
channel whose alignment score passes ``ecc_min_corr`` (MAD outlier rejection on
top), and falls back to all channels plus a constant ``cell_bias_nm`` when every
channel fails. This script reads a per-pos drift log and asks:

  1. is the selected channel set stable in time, or does it flicker?
  2. would a fixed "stably cell-free" channel set give a different shift?
  3. when the threshold drops a channel, was that channel actually off-consensus?
  4. how do those terms compare with the closed-loop residual the loop is fighting?

Usage:
    python scripts/analyze_channel_selection_stability.py <drift_log.json> [--config <drift_config.json>]

Read-only: it never touches the acquisition disk, only the log on C:.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_logger import save_figure  # noqa: E402

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# ---- publication defaults (paper.mplstyle is not present on this branch) ----
mpl.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
# Okabe-Ito
C_FREE = "#0072B2"    # cell-free / accepted
C_CELL = "#D55E00"    # cell-bearing / rejected
C_MIX = "#009E73"     # fixed-set reference
C_GREY = "#999999"

MM = 1.0 / 25.4


# ------------------------------------------------------------------ parsing
def stream_positions(log_path: Path):
    """Yield position records from the big pretty-printed JSON array.

    The log can be hundreds of MB and is still being appended to while a
    timelapse runs, so it is read line by line: position records sit at
    indent 6 and are decoded one at a time. A truncated tail is dropped.
    """
    START, END1, END2 = "      {", "      }", "      },"
    buf = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\r\n")
            if buf is None:
                if s == START:
                    buf = [s]
                continue
            buf.append(s)
            if s in (END1, END2):
                txt = "\n".join(buf)
                if s == END2:
                    txt = txt[:-1]
                buf = None
                try:
                    rec = json.loads(txt)
                except Exception:
                    continue
                if "pos_label" in rec:
                    yield rec


def build_tables(log_path: Path):
    """Per-Pos arrays [n_tp, n_ch] of final corr / tx / ty, plus logged n_used."""
    per_pos: dict = {}
    max_tp = -1
    for rec in stream_positions(log_path):
        tp = int(rec["timepoint"])
        max_tp = max(max_tp, tp)
        p = rec["pos_label"]
        d = per_pos.setdefault(p, {"tp": [], "ch": [], "corr": [], "tx": [], "ty": [],
                                   "nu": {}, "txavg": {}})
        d["nu"][tp] = rec.get("n_channels_used")
        d["txavg"][tp] = rec.get("tx_avg_px")
        for cd in rec.get("channel_details", []):
            corr = cd.get("corr3", cd.get("corr2", cd.get("corr1")))
            if corr is None:
                continue
            d["tp"].append(tp)
            d["ch"].append(int(cd["ch"]))
            d["corr"].append(corr)
            d["tx"].append(cd.get("tx3", cd.get("tx2", cd.get("tx1"))))
            d["ty"].append(cd.get("ty3", cd.get("ty2", cd.get("ty1"))))
    ntp = max_tp + 1
    out = {}
    for p, d in per_pos.items():
        tp = np.asarray(d["tp"])
        ch = np.asarray(d["ch"])
        nch = int(ch.max()) + 1
        C = np.full((ntp, nch), np.nan)
        X = np.full((ntp, nch), np.nan)
        Y = np.full((ntp, nch), np.nan)
        C[tp, ch] = d["corr"]
        X[tp, ch] = d["tx"]
        Y[tp, ch] = d["ty"]
        out[p] = dict(C=C, X=X, Y=Y, nu=d["nu"], txavg=d["txavg"])
    return out, ntp


# ------------------------------------------------- production selection rule
def _mad(a):
    return np.median(np.abs(a - np.median(a)))


def _mad_out(v, thresh=5.0):
    v = np.asarray(v, float)
    m = _mad(v)
    if m == 0:
        return np.zeros(len(v), bool)
    return np.abs(v - np.median(v)) > thresh * m


def reproduce_selection(tab, ntp, thr):
    """Rebuild compute_drift_online's used-channel mask (verified against the log)."""
    used = {}
    ok_n = tot_n = 0
    for p, d in tab.items():
        C, X, Y = d["C"], d["X"], d["Y"]
        U = np.zeros(C.shape, bool)
        fallback = np.zeros(ntp, bool)
        for t in range(ntp):
            fin = np.isfinite(C[t])
            if not fin.any():
                continue
            idx = np.where(fin)[0]
            low = C[t][idx] < thr
            if len(idx) >= 3:
                is_out = _mad_out(X[t][idx]) | _mad_out(Y[t][idx]) | low
            else:
                is_out = low
            keep = ~is_out
            if not keep.any():
                # Production's fallback: every channel was flagged, so all are used
                # again. It fires whenever nothing survives -- not only when every
                # score is below threshold, but also when the one channel that
                # passed is a MAD outlier.
                keep = np.ones(len(idx), bool)
                fallback[t] = True
            U[t, idx[keep]] = True
            logged = d["nu"].get(t)
            if logged is not None and np.isfinite(logged):
                tot_n += 1
                ok_n += int(int(logged) == int(keep.sum()))
        used[p] = dict(U=U, fallback=fallback)
    return used, (ok_n, tot_n)


def stable_sets(tab, thr, min_obs=50, pass_frac_min=0.9, cell_frac_max=0.05):
    """Per-Pos channel labels from the whole run: stably cell-free / stably cell-bearing."""
    lab = {}
    for p, d in tab.items():
        C = d["C"]
        fin = np.isfinite(C)
        nobs = fin.sum(0)
        ok = nobs > min_obs
        pf = np.where(ok, np.where(fin, C >= thr, False).sum(0) / np.maximum(nobs, 1), np.nan)
        free = (pf >= pass_frac_min) & ok
        idx = np.where(ok)[0]
        inner = free.copy()
        if len(idx):                     # the two end traps are excluded from analysis anyway
            inner[idx[0]] = False
            inner[idx[-1]] = False
        lab[p] = dict(ok=ok, pass_frac=pf, free=free, inner=inner,
                      cell=(pf <= cell_frac_max) & ok)
    return lab


# ------------------------------------------------------------------ analysis
def analyse(tab, used, lab, ntp, thr, px_nm, cell_bias_nm, pos_split):
    R = {}
    sel_frac = []
    for p, d in tab.items():
        fin = np.isfinite(d["C"])
        nobs = fin.sum(0)
        U = used[p]["U"]
        for c in np.where(lab[p]["ok"])[0]:
            sel_frac.append(U[:, c].sum() / nobs[c])
    R["sel_frac"] = np.asarray(sel_frac)

    jac = []
    for p, d in tab.items():
        ok = lab[p]["ok"]
        fin = np.isfinite(d["C"]).any(1)
        Uv = used[p]["U"][:, ok][fin]
        inter = (Uv[1:] & Uv[:-1]).sum(1)
        uni = (Uv[1:] | Uv[:-1]).sum(1)
        jac.append(np.where(uni > 0, inter / np.maximum(uni, 1), 1.0))
    R["jaccard"] = np.concatenate(jac)

    diff, diff_fb, diff_cont, n_used_all = [], [], [], []
    n_frames_scored = 0
    scatter, resid_sigma, resid_std = [], [], []
    off_pass, off_fail, n_fixed = [], [], []
    for p, d in tab.items():
        C, X = d["C"], d["X"]
        U, fb = used[p]["U"], used[p]["fallback"]
        L = lab[p]
        S = L["inner"] if L["inner"].sum() >= 2 else L["free"]
        n_fixed.append(int(S.sum()))
        fin = np.isfinite(C)
        n_used_all.append(U.sum(1)[fin.any(1)])
        if S.sum() < 2:
            continue
        flick_ch = L["ok"] & (L["pass_frac"] > 0.05) & (L["pass_frac"] < 0.95)
        prod = np.full(ntp, np.nan)
        fx = np.full(ntp, np.nan)
        for t in range(ntp):
            if not fin[t].any():
                continue
            if U[t].any():
                prod[t] = X[t][U[t]].mean()
                if fb[t]:
                    prod[t] += (-1.0 if int(p[3:]) >= pos_split else 1.0) * cell_bias_nm / px_nm
            s = S & fin[t]
            if s.sum() < 2:
                continue
            ref = X[t][s].mean()
            fx[t] = ref
            n_frames_scored += 1
            if s.sum() >= 3:
                scatter.append(np.std(X[t][s], ddof=1) * px_nm)
            for c in np.where(flick_ch & fin[t])[0]:
                (off_pass if C[t][c] >= thr else off_fail).append((X[t][c] - ref) * px_nm)
            if not fb[t] and (U[t] & L["cell"]).any():
                # a stably cell-bearing channel passed the gate on its own merits;
                # fallback frames are counted separately, they are not leaks
                diff_cont.append((X[t][U[t]].mean() - ref) * px_nm)
        m = np.isfinite(prod) & np.isfinite(fx)
        diff.append((prod - fx)[m] * px_nm)
        diff_fb.append((prod - fx)[m & fb] * px_nm)
        v = fx[m][5:] * px_nm
        if len(v) > 50:
            resid_sigma.append(np.std(np.diff(v)) / np.sqrt(2))
            resid_std.append(np.std(v))
    R.update(diff=np.concatenate(diff),
             diff_fb=np.concatenate([a for a in diff_fb if len(a)]),
             diff_cont=np.asarray(diff_cont),
             n_used=np.concatenate(n_used_all),
             scatter=np.asarray(scatter),
             resid_sigma=np.asarray(resid_sigma),
             resid_std=np.asarray(resid_std),
             n_frames=np.asarray([n_frames_scored]),
             off_pass=np.asarray(off_pass),
             off_fail=np.asarray(off_fail),
             n_fixed=np.asarray(n_fixed))
    return R


# -------------------------------------------------------------------- figure
def make_figure(R):
    fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 50 * MM))

    ax = axes[0]
    ax.hist(R["sel_frac"], bins=np.arange(0, 1.0001, 0.05), color=C_GREY,
            edgecolor="white", linewidth=0.3)
    ax.axvspan(0.05, 0.95, color=C_CELL, alpha=0.10, lw=0)
    ax.set_xlabel("selected fraction of time points")
    ax.set_ylabel("channels")
    ax.set_title("a  bimodal, but 30% flicker", loc="left")
    ax.text(0.5, 0.9, "flickering %.0f%%"
            % (100 * ((R["sel_frac"] > .05) & (R["sel_frac"] < .95)).mean()),
            transform=ax.transAxes, ha="center", va="top", color=C_CELL, fontsize=6)

    ax = axes[1]
    nu = R["n_used"].astype(int)
    b = np.bincount(nu, minlength=13)[:13]
    ax.bar(np.arange(len(b)), b, color=C_FREE, width=0.8)
    ax.set_xlabel("channels averaged per time point")
    ax.set_ylabel("Pos $\\times$ time points")
    ax.set_title("b  %.0f%% of frames use $\\leq$2 channels" % (100 * (nu <= 2).mean()),
                 loc="left")

    ax = axes[2]
    bins = np.arange(0, 301, 10)
    ax.hist(np.abs(R["off_pass"]), bins=bins, color=C_FREE, alpha=0.8, density=True, lw=0,
            label="passes ($n$=%d)" % len(R["off_pass"]))
    ax.hist(np.abs(R["off_fail"]), bins=bins, color=C_CELL, alpha=0.8, density=True, lw=0,
            label="fails ($n$=%d)" % len(R["off_fail"]))
    ax.set_xlabel("|shift $-$ cell-free consensus| [nm]")
    ax.set_ylabel("density [nm$^{-1}$]")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(loc="upper right", handlelength=1.0, borderaxespad=0.2)
    ax.set_title("c  the gate fires on the bad frames", loc="left")

    ax = axes[3]
    terms = [
        ("closed-loop residual", np.median(R["resid_std"]), C_GREY),
        ("cell ch leaks in (%.0f%%)" % (100 * len(R["diff_cont"]) / R["n_frames"][0]),
         abs(np.median(R["diff_cont"])), C_CELL),
        ("fallback +181 nm (%.0f%%)" % (100 * len(R["diff_fb"]) / R["n_frames"][0]),
         abs(np.median(R["diff_fb"])), C_CELL),
        ("cell-free ch spread", np.median(R["scatter"]), C_FREE),
        ("production $-$ fixed set", np.median(np.abs(R["diff"])), C_MIX),
    ]
    y = np.arange(len(terms))
    ax.barh(y, [t[1] for t in terms], color=[t[2] for t in terms], height=0.7)
    for i, t in enumerate(terms):
        ax.text(t[1] + 4, i, ("%.0f" % t[1]) if t[1] >= 10 else ("%.1f" % t[1]),
                va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in terms])
    ax.set_xlabel("median magnitude [nm]")
    ax.set_title("d  error budget of the drift estimate", loc="left")
    ax.set_xlim(0, max(t[1] for t in terms) * 1.3)
    fig.tight_layout(pad=0.4, w_pad=1.6)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    log_path = Path(args.log)
    cfg_path = Path(args.config) if args.config else log_path.parent / "drift_config_zstack.json"
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    thr = cfg.get("ecc_min_corr", 0.9925)
    px_nm = cfg.get("pixel_scale_um", 0.34567514677103717) * 1000.0
    cell_bias_nm = cfg.get("cell_bias_nm", 0.0)
    pos_split = cfg.get("pos_split", 57)

    print("reading %s ..." % log_path)
    tab, ntp = build_tables(log_path)
    print("  %d Pos x %d time points" % (len(tab), ntp))
    used, (ok_n, tot_n) = reproduce_selection(tab, ntp, thr)
    print("  selection reproduced from the log: %.2f%% of %d records"
          % (100 * ok_n / max(tot_n, 1), tot_n))
    lab = stable_sets(tab, thr)
    R = analyse(tab, used, lab, ntp, thr, px_nm, cell_bias_nm, pos_split)

    sf = R["sel_frac"]
    print("\n-- selection stability --")
    print("   channels decisive (<=5%% or >=95%% of frames): %.1f%%   flickering: %.1f%%"
          % (100 * ((sf <= .05) | (sf >= .95)).mean(), 100 * ((sf > .05) & (sf < .95)).mean()))
    print("   selected set identical at t and t+1: %.1f%%   mean Jaccard %.3f"
          % (100 * (R["jaccard"] == 1).mean(), R["jaccard"].mean()))
    print("   stably cell-free channels per Pos (end traps excluded): mean %.2f, zero in %d Pos"
          % (R["n_fixed"].mean(), int((R["n_fixed"] == 0).sum())))
    print("\n-- production vs a fixed stably-cell-free set (tx) --")
    for nm, a in [("typical frame", R["diff"]),
                  ("all-ch fallback frames", R["diff_fb"]),
                  ("genuine leak frames (cell ch passed the gate)", R["diff_cont"])]:
        print("   %-42s median|d| %6.1f nm  p95 %7.1f  n=%d"
              % (nm, np.median(np.abs(a)), np.percentile(np.abs(a), 95), len(a)))
    print("   (fallback = %.2f%% of frames, genuine leaks = %.2f%% of frames)"
          % (100 * len(R["diff_fb"]) / R["n_frames"][0],
             100 * len(R["diff_cont"]) / R["n_frames"][0]))
    print("\n-- error budget --")
    print("   cell-free channels disagree within a frame: median std %.1f nm"
          % np.median(R["scatter"]))
    print("   flickering ch offset from consensus: passes %.1f nm | fails %.1f nm (median |.|)"
          % (np.median(np.abs(R["off_pass"])), np.median(np.abs(R["off_fail"]))))
    print("   closed-loop residual: std %.1f nm, per-step sigma %.1f nm (median over Pos)"
          % (np.median(R["resid_std"]), np.median(R["resid_sigma"])))

    fig = make_figure(R)
    caption = (
        "The per-frame alignment-score gate, not the channel identity, is what keeps drift "
        "control honest. Online drift log of the 260928 0%%-glucose z-stack timelapse "
        "(%d Pos x %d time points, 5 min/frame, S. pombe in EMM at 30 C, gaussian2d estimator, "
        "ecc_min_corr = %.4f, cell_bias_nm = %.0f nm, pos_split = %d); one trap channel per crop "
        "(40 x 180 px, tilt-corrected) matched against the cell-free grid(0,0) reference. "
        "(a) Per-channel selected fraction = (time points the channel entered the drift average) / "
        "(time points it was reconstructed); selected = alignment score >= ecc_min_corr AND not a "
        "5 x MAD outlier in tx or ty, reproduced from the log (matches the logged n_channels_used "
        "in 100%% of records). Orange band 0.05-0.95 = flickering channels. (b) Channels averaged "
        "per time point = the same count per (Pos, time point). (c) Flickering channels only: "
        "|shift - cell-free consensus|, where shift = that channel's final-pass tx and consensus = "
        "mean tx over the Pos's stably cell-free channels (score >= threshold in >= 90%% of frames, "
        "two end traps excluded) in the same frame, split by whether the channel passed the gate in "
        "that frame. Bars, densities of the full distributions; no error bars. (d) Median magnitude "
        "of each term of the tx error budget: closed-loop residual = std over time of the consensus "
        "tx per Pos, median over Pos; cell-bearing leak = production mean - consensus on frames "
        "where a stably cell-bearing channel passed the gate; fallback = the same on frames where "
        "every channel failed and the 181 nm constant was applied; cell-free disagreement = "
        "within-frame std across >= 3 stably cell-free channels; production - fixed set = production "
        "mean minus the fixed stably-cell-free mean, all frames. tx is the image-column shift and "
        "drives the stage Y correction; 1 px = %.1f nm. n = %d channels, %d Pos x time points, "
        "%d flickering-channel frames. Descriptive census of one run; no hypothesis test applied. "
        "MAD, median absolute deviation. Source data: the npz saved next to this figure."
        % (len(tab), ntp, thr, cell_bias_nm, pos_split, px_nm,
           len(sf), len(R["n_used"]), len(R["off_pass"]) + len(R["off_fail"]))
    )
    save_figure(
        fig,
        params={"log": str(log_path), "ecc_min_corr": thr, "estimator": cfg.get("estimator"),
                "cell_bias_nm": cell_bias_nm, "pos_split": pos_split, "n_tp": ntp,
                "n_pos": len(tab), "pass_frac_min": 0.9, "mad_thresh": 5.0},
        description=("260928 online drift log: stability of the corr-threshold cell-free channel "
                     "selection vs a fixed stably-cell-free set, and the error budget it sits in"),
        data={k: R[k] for k in ("sel_frac", "jaccard", "diff", "diff_fb", "diff_cont", "n_used",
                                "n_frames",
                                "scatter", "resid_sigma", "resid_std", "off_pass", "off_fail",
                                "n_fixed")},
        caption=caption,
        dpi=300,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
