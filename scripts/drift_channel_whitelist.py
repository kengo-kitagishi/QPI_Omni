"""Cell-free channel whitelist for the online drift average.

Why this exists
---------------
``compute_drift_online`` averages every channel whose alignment score passes
``ecc_min_corr`` (5 x MAD rejection on top), and when nothing survives it falls
back to *all* channels plus a constant ``cell_bias_nm``. Measured on the 260928
run (111 Pos x 578 time points, scripts/analyze_channel_selection_stability.py):

* the per-frame score gate itself is sound -- a channel that passes sits 9 nm
  from the cell-free consensus, one that fails sits 111 nm from it;
* channels that genuinely leak through the gate cost 3.5 nm on 0.85% of frames,
  i.e. nothing;
* but the **fallback fires on 3.57% of frames and costs 85 nm**, and on those
  frames the cell-free channels still agree with each other to 12.4 nm (12 nm on
  a normal frame). The best score in a fallback frame is 0.99195 -- 0.0006 below
  the threshold. A usable measurement is being thrown away and replaced by a
  constant.

So the fix is not to replace the gate with a channel label, it is to keep a
whitelist of cell-free channels and use it as the anchor the gate can fall back
onto (and, optionally, as a mask the gate can never widen past).

Causality
---------
The whitelist has to be built from the past only, because it runs inside the
acquisition loop. ``WhitelistTracker`` keeps a rolling pass-rate per channel
with hysteresis, so a trap that gains a cell mid-run leaves the whitelist and a
trap that was briefly noisy comes back. Until ``warmup`` frames have been seen
it returns ``None``, which makes every caller behave exactly as production does
today.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

# Defaults measured on 260928; see the module docstring.
WINDOW = 144           # frames of history (12 h at 5 min/frame)
WARMUP = 36            # frames before the whitelist is allowed to act (3 h)
ENTER = 0.80           # pass rate at which a channel joins the whitelist
LEAVE = 0.50           # pass rate at which it drops out (hysteresis)


def mad(a):
    return np.median(np.abs(a - np.median(a)))


def mad_outliers(v, thresh=5.0):
    """True where *v* is a MAD outlier. Same rule as ecc_utils.remove_outliers_mad."""
    v = np.asarray(v, dtype=float)
    m = mad(v)
    if m == 0:
        return np.zeros(len(v), dtype=bool)
    return np.abs(v - np.median(v)) > thresh * m


def select_channels(scores, tx, ty, thr, whitelist=None, mad_thresh=5.0):
    """Channels to average, and whether the fallback had to be taken.

    With ``whitelist=None`` this is production's rule, bit for bit: score gate,
    5 x MAD rejection on tx and ty, and every channel again if nothing survives.

    With a whitelist:
      * the surviving set is intersected with it, so a known cell-bearing
        channel can never enter the average (skipped if that would empty it);
      * when nothing survives the gate, the whitelist itself becomes the
        average instead of all channels -- those channels still agree to ~12 nm
        on exactly these frames, which the constant bias correction does not.

    Returns ``(keep, fallback, anchored)``. ``anchored`` is True when the
    whitelist supplied the fallback, i.e. when ``cell_bias_nm`` must NOT be
    applied by the caller.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    low = scores < thr
    if n >= 3:
        is_out = mad_outliers(tx, mad_thresh) | mad_outliers(ty, mad_thresh) | low
    else:
        is_out = low
    keep = ~is_out

    wl = None if whitelist is None else np.asarray(whitelist, dtype=bool)
    if wl is not None and len(wl) != n:
        raise ValueError("whitelist has %d entries, expected %d" % (len(wl), n))

    if keep.any():
        if wl is not None and (keep & wl).any():
            keep = keep & wl
        return keep, False, False

    if wl is not None and wl.any():
        return wl.copy(), True, True
    return np.ones(n, dtype=bool), True, False


class WhitelistTracker:
    """Rolling, causal per-Pos cell-free whitelist with hysteresis.

    ``update`` is called once per frame with that frame's per-channel scores and
    returns the whitelist to use for the NEXT selection, or None during warmup.
    """

    def __init__(self, n_channels, thr, window=WINDOW, warmup=WARMUP,
                 enter=ENTER, leave=LEAVE, exclude_end_traps=True):
        self.n = n_channels
        self.thr = thr
        self.window = window
        self.warmup = warmup
        self.enter = enter
        self.leave = leave
        self.exclude_end_traps = exclude_end_traps
        self.hist = deque(maxlen=window)
        self.state = np.zeros(n_channels, dtype=bool)

    def pass_rate(self):
        if not self.hist:
            return np.zeros(self.n)
        h = np.asarray(self.hist, dtype=float)      # [frames, n] of 1 / 0 / nan
        with np.errstate(invalid="ignore"):
            n_obs = np.sum(~np.isnan(h), axis=0)
            n_pass = np.nansum(h, axis=0)
        return np.where(n_obs > 0, n_pass / np.maximum(n_obs, 1), 0.0)

    def update(self, scores):
        """Feed one frame; scores may contain NaN for channels not measured."""
        s = np.asarray(scores, dtype=float)
        row = np.where(np.isnan(s), np.nan, (s >= self.thr).astype(float))
        self.hist.append(row)
        r = self.pass_rate()
        self.state = np.where(self.state, r > self.leave, r >= self.enter)
        if self.exclude_end_traps:
            seen = np.any(~np.isnan(np.asarray(self.hist, dtype=float)), axis=0)
            idx = np.where(seen)[0]
            if len(idx) >= 3:
                self.state[idx[0]] = False
                self.state[idx[-1]] = False
        if len(self.hist) < self.warmup or not self.state.any():
            return None
        return self.state.copy()


def save(path, per_pos_state, meta=None):
    """Persist {pos_label: [channel indices]} so a restart keeps its whitelist."""
    payload = {"meta": meta or {},
               "whitelist": {p: [int(i) for i in np.where(np.asarray(v, bool))[0]]
                             for p, v in per_pos_state.items()}}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load(path, n_channels):
    """Read back what save() wrote, as {pos_label: bool mask}."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for p, idx in payload.get("whitelist", {}).items():
        m = np.zeros(n_channels, dtype=bool)
        for i in idx:
            if 0 <= i < n_channels:
                m[i] = True
        out[p] = m
    return out
