"""Replay a drift log through the current and the whitelist-anchored selection.

Nothing here runs inside the acquisition loop -- it reads a finished or running
log and re-decides each frame offline, so the change can be judged before it is
wired into compute_drift_online.py.

Two checks, in order:

  1. with the whitelist disabled, the replay must reproduce the logged
     ``tx_avg_px`` exactly. If it does not, the replay is not modelling
     production and nothing below means anything.
  2. with the whitelist enabled (causal: at frame t it is built only from
     frames < t), report every frame whose estimate changes and by how much.

Usage:
    python scripts/replay_drift_selection.py <drift_log.json> [--config <cfg.json>]
                                             [--window 144] [--warmup 36]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_channel_selection_stability import build_tables  # noqa: E402
import drift_channel_whitelist as wl  # noqa: E402


def replay(tab, ntp, thr, px_nm, cell_bias_nm, pos_split, use_whitelist,
           window, warmup, static=None):
    """Re-decide every frame. Returns per-(pos, tp) records.

    ``static`` is {pos_label: bool mask} picked by hand (channel_contact_sheet).
    When given it replaces the rolling tracker: the mask is fixed, so there is
    no warmup and frame 0 is already whitelisted. A Pos missing from it gets
    whitelist=None, i.e. production's rule unchanged.
    """
    out = []
    for p, d in tab.items():
        C, X, Y = d["C"], d["X"], d["Y"]
        nch = C.shape[1]
        tracker = wl.WhitelistTracker(nch, thr, window=window, warmup=warmup)
        mask = None
        if static is not None:
            mask = static.get(p)
        sign = -1.0 if int(p[3:]) >= pos_split else 1.0
        for t in range(ntp):
            fin = np.isfinite(C[t])
            if not fin.any():
                continue
            idx = np.where(fin)[0]
            sub = None
            if use_whitelist and mask is not None:
                sub = mask[idx]
                if not sub.any():
                    sub = None
            keep, fallback, anchored = wl.select_channels(
                C[t][idx], X[t][idx], Y[t][idx], thr, whitelist=sub)
            tx = X[t][idx][keep].mean()
            # Production applies the constant only when it averaged every channel
            # AND every score was below threshold -- a fallback triggered by the
            # MAD rule alone does not get it.
            all_low = bool(np.all(C[t][idx] < thr))
            if fallback and not anchored and all_low and cell_bias_nm:
                tx += sign * cell_bias_nm / px_nm
            spread = (np.std(X[t][idx][keep], ddof=1) * px_nm
                      if keep.sum() >= 3 else np.nan)
            out.append((p, t, tx, int(keep.sum()), fallback, anchored, spread))
            if use_whitelist and static is None:
                row = np.full(nch, np.nan)
                row[idx] = C[t][idx]
                mask = tracker.update(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--config", default=None)
    ap.add_argument("--window", type=int, default=wl.WINDOW)
    ap.add_argument("--warmup", type=int, default=wl.WARMUP)
    ap.add_argument("--whitelist", default=None,
                    help="channel_whitelist.json picked by hand; replaces the rolling "
                         "tracker with that fixed mask (no warmup)")
    args = ap.parse_args()

    log_path = Path(args.log)
    cfg_path = Path(args.config) if args.config else log_path.parent / "drift_config_zstack.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    thr = cfg.get("ecc_min_corr", 0.9925)
    px_nm = cfg.get("pixel_scale_um", 0.34567514677103717) * 1000.0
    cell_bias_nm = cfg.get("cell_bias_nm", 0.0)
    pos_split = cfg.get("pos_split", 57)

    print("reading %s ..." % log_path)
    tab, ntp = build_tables(log_path)
    print("  %d Pos x %d time points, threshold %.4f, cell_bias %.0f nm"
          % (len(tab), ntp, thr, cell_bias_nm))

    # ---- check 1: whitelist off must reproduce the log exactly ----
    static = None
    if args.whitelist:
        payload = json.loads(Path(args.whitelist).read_text(encoding="utf-8"))
        static = {}
        for pos, idx in payload.get("whitelist", {}).items():
            if pos not in tab:
                continue
            m = np.zeros(tab[pos]["C"].shape[1], dtype=bool)
            for i in idx:
                if 0 <= i < len(m):
                    m[i] = True
            static[pos] = m
        absent = sorted(set(tab) - set(static), key=lambda s: int(s[3:]))
        print("  hand-picked whitelist: %d Pos, %d channels; %d Pos without one "
              "(production rule unchanged): %s"
              % (len(static), sum(int(m.sum()) for m in static.values()),
                 len(absent), absent))

    base = replay(tab, ntp, thr, px_nm, cell_bias_nm, pos_split, False,
                  args.window, args.warmup)
    err = []
    for p, t, tx, n, fb, anc, sp in base:
        logged = tab[p]["txavg"].get(t)
        if logged is not None and np.isfinite(logged):
            err.append((tx - logged) * px_nm)
    err = np.asarray(err)
    print("\n[1] whitelist OFF vs the log: max |error| %.6f nm over %d frames -- %s"
          % (np.abs(err).max(), len(err),
             "exact" if np.abs(err).max() < 1e-6 else "MISMATCH, stop here"))
    if np.abs(err).max() >= 1e-6:
        return

    # ---- check 2: whitelist on ----
    new = replay(tab, ntp, thr, px_nm, cell_bias_nm, pos_split, True,
                 args.window, args.warmup, static=static)
    assert len(new) == len(base)
    d = np.array([(b[2] - a[2]) * px_nm for a, b in zip(base, new)])
    fb_old = np.array([a[4] for a in base])
    anc = np.array([b[5] for b in new])
    narrowed = np.array([b[3] < a[3] and not a[4] for a, b in zip(base, new)])
    sp_anch = np.array([b[6] for b in new])[anc]

    n = len(d)
    if static is not None:
        print("\n[2] whitelist ON (hand-picked, fixed mask, no warmup)")
    else:
        print("\n[2] whitelist ON (window %d frames, warmup %d, enter %.2f / leave %.2f)"
              % (args.window, args.warmup, wl.ENTER, wl.LEAVE))
    print("    frames replayed                     : %d" % n)
    print("    frames whose estimate changes       : %d (%.2f%%)"
          % ((np.abs(d) > 1e-9).sum(), 100 * (np.abs(d) > 1e-9).mean()))
    print("    fallback frames in production       : %d (%.2f%%)" % (fb_old.sum(), 100 * fb_old.mean()))
    print("    of those, now anchored on whitelist : %d (%.1f%% of fallbacks)"
          % (anc.sum(), 100 * anc.sum() / max(fb_old.sum(), 1)))
    print("    frames where the mask dropped a ch  : %d (%.2f%%)" % (narrowed.sum(), 100 * narrowed.mean()))
    for name, m in [("anchored fallback frames", anc),
                    ("mask-narrowed frames", narrowed)]:
        if m.sum():
            print("    change on %-26s: median %+.1f nm  |med| %.1f  p95|.| %.1f"
                  % (name, np.median(d[m]), abs(np.median(d[m])),
                     np.percentile(np.abs(d[m]), 95)))
    good = np.isfinite(sp_anch)
    if good.any():
        print("    spread of the channels averaged on anchored frames: median %.1f nm"
              % np.median(sp_anch[good]))
    print("    the 181 nm constant is applied on %d frames instead of %d"
          % (int((fb_old & ~anc).sum()), int(fb_old.sum())))


if __name__ == "__main__":
    main()
