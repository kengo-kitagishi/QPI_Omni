"""Animate a swelling-death mother lineage as a time-synced QPI movie.

Three stacked, time-synchronised panels for one mother-trap channel of the
260517 type-B experiment (phase1 = 2% EMM, frames 0-2018, 5 min/frame =
12 frame/h):

  1. phase image (landscape) in inferno on a fixed 0-1.8 rad scale, with the
     segmentation mask contours overlaid (all cells red, the tracked mother
     emphasised in cyan), a 10 um scale bar and a "t = +X.X h / frame N" stamp;
  2. mean RI of the tracked mother vs. time, revealed progressively;
  3. dry mass [pg] of the tracked mother vs. time, revealed progressively.

The x-axis of panels 2/3 is "time from the last division [h]" with t = 0 at the
curated last-division frame (Panel A anchor). One animation frame == one data
frame.

Mother source -- IMPORTANT
--------------------------
The mother is the lineage tracker's ``rank == 1`` track from
``inference_out/lineage_out/lineage_data3D.csv`` -- the genealogical mother at
the trench's closed end, with real frame-to-frame identity. This is the SAME
source used by per_channel_figures, _fig_panelA_cellcycle and the curated
DEATH_WINDOW analysis, and its mean_ri / mass_pg / volume_um3_rod columns are
the pipeline calibration (660 nm, n_medium, alpha_ri, milliq protein basis)
baked in at tracking time -- nothing is recomputed or hardcoded here.

(The earlier center-nearest selection of build_summary_table is NOT used: in a
full-trench crop it hops between cells when the mother touches the image border,
which breaks the single-lineage requirement.)

Usage (example: swelling-death lineage Pos18/ch02, curated last division 1855 ->
DEATH_WINDOW["Pos18_ch02"] = (1823, 1855) in _fig_predeath_growthrate.py):

  python scripts/_animate_vol_ri_f_ch.py \
      --channel-dir "F:/260517/2per_0055per_0per_2per_crop_sub/Pos18/output_phase/channels/crop_sub_rawraw/z000/ch02" \
      --last-div-frame 1855

--last-div-frame MUST match the value used for this lineage in Panel A; it is
never auto-detected here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as manimation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
from skimage import measure  # noqa: E402

import imageio_ffmpeg  # noqa: E402

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

# ---- styling -------------------------------------------------------------
CMAP = "inferno"
MOTHER_COLOR = "#00E5FF"   # tracked mother contour (bright cyan, pops on inferno)
DAUGHTER_COLOR = "#E69F00"  # n-2 non-mother daughter (Okabe-Ito orange)
OTHER_COLOR = "#FF3B30"    # all other cell contours (red)
MOTHER_TRACE = "#0072B2"   # Okabe-Ito blue (mother lines)
PX_DEFAULT = 0.34567514677103717  # 260517 run param
PHASE1_MIN_DEFAULT = 0
PHASE1_MAX_DEFAULT = 2018  # phase1 (2% EMM) ends here for 260517 type-B


# ---- IO helpers (mirror _fig_panelA_cellcycle paths) ---------------------
def phase_path(ch_dir: Path, frame: int) -> Path:
    return ch_dir / f"img_{frame:09d}_ph_000_phase.tif"


def mask_path(ch_dir: Path, frame: int) -> Path:
    return ch_dir / "inference_out" / f"img_{frame:09d}_ph_000_phase_masks.tif"


def lineage_csv(ch_dir: Path) -> Path:
    return ch_dir / "inference_out" / "lineage_out" / "lineage_data3D.csv"


def load_run_params(ch_dir: Path) -> dict:
    p = ch_dir / "inference_out" / "lineage_out" / "lineage_run_params.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[anim] WARNING: could not parse {p}: {e}", file=sys.stderr)
    else:
        print(f"[anim] WARNING: no run-params JSON at {p}; using defaults",
              file=sys.stderr)
    return {}


def derive_pos_ch(channel_dir: Path) -> tuple[str, str]:
    """Pull 'Pos18' / 'ch02' out of the channel path (fallbacks if absent)."""
    parts = channel_dir.resolve().parts
    pos = next((p for p in parts if re.fullmatch(r"Pos\d+", p)), "Pos")
    ch = next((p for p in reversed(parts) if re.fullmatch(r"ch\d+", p)), "ch")
    return pos, ch


# ---- mask / contour helpers ----------------------------------------------
def label_at(mask: np.ndarray, cy: float, cx: float) -> int:
    """Label under the mother centroid, with a small neighbourhood fallback.

    Same logic as _fig_panelA_cellcycle.label_at so the emphasised cell matches
    the lineage tracker's mother."""
    h, w = mask.shape
    yi, xi = int(round(cy)), int(round(cx))
    if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
        return int(mask[yi, xi])
    y0, y1 = max(yi - 2, 0), min(yi + 3, h)
    x0, x1 = max(xi - 2, 0), min(xi + 3, w)
    nz = mask[y0:y1, x0:x1]
    nz = nz[nz != 0]
    if nz.size:
        v, c = np.unique(nz, return_counts=True)
        return int(v[np.argmax(c)])
    return 0


# ---- EFD (elliptic Fourier descriptor) contour smoothing -----------------
# Kuhl & Giardina (1982): treat the closed boundary as a periodic signal and
# keep the first K harmonics -> a low-pass filter that removes the 1-px
# staircase while preserving area. K=6 is the value adopted by the corrected-
# volume pipeline (mask_volume_schematic.efd_section_geometry / _smooth_efd.py);
# the boundary it draws is exactly this reconstruction.
def _efd_coeffs(xy: np.ndarray, k: int):
    dxy = np.diff(xy, axis=0)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    dt[dt == 0] = 1e-9
    t = np.concatenate([[0.0], np.cumsum(dt)])
    T = t[-1]
    phi = 2.0 * np.pi * t / T
    a = np.zeros(k); b = np.zeros(k); c = np.zeros(k); d = np.zeros(k)
    dx, dy = dxy[:, 0], dxy[:, 1]
    for n in range(1, k + 1):
        const = T / (2.0 * n * n * np.pi * np.pi)
        cos_p = np.cos(n * phi[1:]) - np.cos(n * phi[:-1])
        sin_p = np.sin(n * phi[1:]) - np.sin(n * phi[:-1])
        a[n - 1] = const * np.sum(dx / dt * cos_p)
        b[n - 1] = const * np.sum(dx / dt * sin_p)
        c[n - 1] = const * np.sum(dy / dt * cos_p)
        d[n - 1] = const * np.sum(dy / dt * sin_p)
    xi = np.cumsum(dx) - (dx / dt) * t[1:]
    A0 = (1.0 / T) * np.sum((dx / (2 * dt)) * (t[1:] ** 2 - t[:-1] ** 2)
                            + xi * (t[1:] - t[:-1])) + xy[0, 0]
    delta = np.cumsum(dy) - (dy / dt) * t[1:]
    C0 = (1.0 / T) * np.sum((dy / (2 * dt)) * (t[1:] ** 2 - t[:-1] ** 2)
                            + delta * (t[1:] - t[:-1])) + xy[0, 1]
    return a, b, c, d, A0, C0


def _efd_smooth(contour_rc: np.ndarray, k: int = 6,
                n_points: int = 180) -> np.ndarray | None:
    """EFD K-harmonic reconstruction of a (row,col) contour, returned as
    (row,col). None if the contour is too small for a stable fit."""
    if contour_rc.shape[0] < max(8, 2 * k):
        return None
    xy = contour_rc[:, ::-1].astype(float)                # (x=col, y=row)
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])
    a, b, c, d, A0, C0 = _efd_coeffs(xy, k)
    phi = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=True)
    x = np.full(n_points, A0); y = np.full(n_points, C0)
    for n in range(1, k + 1):
        x += a[n - 1] * np.cos(n * phi) + b[n - 1] * np.sin(n * phi)
        y += c[n - 1] * np.cos(n * phi) + d[n - 1] * np.sin(n * phi)
    return np.column_stack([y, x])                        # back to (row, col)


def label_contours(mask: np.ndarray, efd: bool = True,
                   k: int = 6) -> dict[int, list[np.ndarray]]:
    """Per-label contours (row,col), so touching cells stay separate.

    efd=True draws the EFD K-harmonic smoothed boundary (the corrected-volume
    pipeline's adopted contour); falls back to the raw marching-squares contour
    for cells too small to fit."""
    out: dict[int, list[np.ndarray]] = {}
    for lbl in np.unique(mask):
        if lbl == 0:
            continue
        cs = measure.find_contours((mask == lbl).astype(float), 0.5)
        if not cs:
            continue
        if efd:
            sm = _efd_smooth(max(cs, key=len), k)
            out[int(lbl)] = [sm] if sm is not None else cs
        else:
            out[int(lbl)] = cs
    return out


# ---- daughter (sister) detection -----------------------------------------
def mother_cell_ids(df: pd.DataFrame) -> set[int]:
    """All cell_ids that ever held rank==1 (the mother, across any relabels).

    Using ALL of them (not a single root) is essential: the tracker can relabel
    the mother's cell_id near death (gold_standard notes this), so a single root
    would miss divisions after a relabel (e.g. Pos5_ch08)."""
    return set(int(c) for c in df.loc[df["rank"] == 1, "cell_id"].unique())


def _track_cell(df: pd.DataFrame, cid: int) -> pd.DataFrame:
    """Rows for one cell_id, sorted by frame, truncated at the first border
    touch (beyond it the cell is pushed out of the trench and untrackable)."""
    t = df[df["cell_id"] == cid].sort_values("frame").reset_index(drop=True)
    if "touches_border" in t and t["touches_border"].any():
        first_b = int(t.index[t["touches_border"].to_numpy(bool)][0])
        t = t.iloc[:first_b]
    return t.reset_index(drop=True)


def daughter_candidates(df: pd.DataFrame, anchor: int) -> list[dict]:
    """Every mother division at/below the anchor and its daughter, NEWEST first.

    Entry: {div_frame, cell_id, n_pts, f0, f1} where [f0, f1] is the daughter's
    trackable span (after border truncation). cands[0] is the last division,
    cands[1] the one before, etc."""
    mom = mother_cell_ids(df)
    dau_all = df[df["parent_id"].isin(mom)]
    out: list[dict] = []
    if dau_all.empty:
        return out
    for divf in sorted(int(b) for b in dau_all["birth_frame"].dropna().unique()):
        if divf > anchor + 1:                 # ignore post-death births
            continue
        born = dau_all[dau_all["birth_frame"] == divf]
        cid = int(born["cell_id"].value_counts().idxmax())
        t = _track_cell(df, cid)
        if not len(t):
            continue
        out.append({"div_frame": divf, "cell_id": cid, "n_pts": int(len(t)),
                    "f0": int(t["frame"].min()), "f1": int(t["frame"].max())})
    out.sort(key=lambda d: d["div_frame"], reverse=True)   # newest first
    return out


def find_daughter_track(df: pd.DataFrame, anchor: int, div_back: int = 2,
                        cell_id: int | None = None,
                        div_frame: int | None = None) -> pd.DataFrame | None:
    """Resolve which daughter to track. Selection priority:

      1. explicit ``cell_id``  -> track exactly that lineage cell;
      2. explicit ``div_frame`` -> the daughter born at the nearest division;
      3. otherwise the daughter born at the (last - ``div_back``) division.

    Returns the track (frame-sorted, border-truncated) or None.
    """
    if cell_id is not None:
        t = _track_cell(df, int(cell_id))
        if not len(t):
            return None
        t.attrs["cell_id"] = int(cell_id)
        t.attrs["div_frame"] = (int(t["birth_frame"].iloc[0])
                                if "birth_frame" in t else -1)
        return t
    cands = daughter_candidates(df, anchor)        # newest first
    if not cands:
        return None
    if div_frame is not None:
        chosen = min(cands, key=lambda d: abs(d["div_frame"] - int(div_frame)))
    elif len(cands) >= div_back + 1:
        chosen = cands[div_back]                    # [0]=last, [2]=n-2
    else:
        return None
    t = _track_cell(df, chosen["cell_id"])
    if not len(t):
        return None
    t.attrs["cell_id"] = chosen["cell_id"]
    t.attrs["div_frame"] = chosen["div_frame"]
    return t


# ---- main ----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel-dir", required=True, type=Path,
                    help="ch directory holding img_*_phase.tif + inference_out/")
    ap.add_argument("--last-div-frame", required=True, type=int,
                    help="curated last-division frame (Panel A anchor; t=0). "
                         "MUST match the value used in _fig_panelA_cellcycle.py.")
    ap.add_argument("--before-h", type=float, default=30.0,
                    help="hours before the last division to show (default 30)")
    ap.add_argument("--after-h", type=float, default=15.0,
                    help="hours after the last division to show (default 15)")
    ap.add_argument("--vmin", type=float, default=0.0, help="phase vmin [rad]")
    ap.add_argument("--vmax", type=float, default=1.8, help="phase vmax [rad]")
    ap.add_argument("--fps", type=int, default=12, help="frames/sec (12 or 24)")
    ap.add_argument("--phase-min", type=int, default=PHASE1_MIN_DEFAULT,
                    help="lower frame clip (phase1 start; default 0)")
    ap.add_argument("--phase-max", type=int, default=PHASE1_MAX_DEFAULT,
                    help="upper frame clip (phase1 end; default 2018)")
    ap.add_argument("--no-daughter", action="store_true",
                    help="do not track/overlay the n-2 non-mother daughter "
                         "(use for elongation-cascade lineages)")
    ap.add_argument("--mode-label", default="swelling death",
                    help="death-mode text shown in the phase-panel title")
    ap.add_argument("--daughter-div-back", type=int, default=2,
                    help="daughter born at the (last - N) division (default 2)")
    ap.add_argument("--daughter-cell-id", type=int, default=None,
                    help="track exactly this lineage cell_id (overrides div-back)")
    ap.add_argument("--daughter-div-frame", type=int, default=None,
                    help="track the daughter born at the nearest division to "
                         "this frame (overrides div-back)")
    ap.add_argument("--list-daughters", action="store_true",
                    help="print the candidate daughters for this lineage and "
                         "exit (use to choose --daughter-cell-id/-div-frame)")
    ap.add_argument("--raw-contours", action="store_true",
                    help="draw raw (jagged) mask contours instead of the "
                         "EFD-smoothed boundary (default: EFD K=6)")
    ap.add_argument("--efd-k", type=int, default=6,
                    help="EFD harmonics for contour smoothing (default 6)")
    ap.add_argument("--ri-min", type=float, default=None,
                    help="fixed mean-RI y-axis min (shared scale; default auto)")
    ap.add_argument("--ri-max", type=float, default=None,
                    help="fixed mean-RI y-axis max (shared scale; default auto)")
    ap.add_argument("--mass-min", type=float, default=None,
                    help="fixed dry-mass y-axis min (shared scale; default auto)")
    ap.add_argument("--mass-max", type=float, default=None,
                    help="fixed dry-mass y-axis max (shared scale; default auto)")
    ap.add_argument("--scalebar-um", type=float, default=10.0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", type=Path, default=None,
                    help="output MP4 (auto-named under F:/ if omitted)")
    args = ap.parse_args()

    channel_dir = args.channel_dir
    meta = load_run_params(channel_dir)
    px = float(meta.get("pixel_size_um", PX_DEFAULT))
    time_interval_min = float(meta.get("time_interval_min", 5.0))
    fph = 60.0 / time_interval_min            # frames per hour (=12)
    pos, ch = derive_pos_ch(channel_dir)

    # ---- window (clipped to phase1) -------------------------------------
    ld = int(args.last_div_frame)
    lo_req = ld - int(round(args.before_h * fph))
    hi_req = ld + int(round(args.after_h * fph))
    lo = max(lo_req, args.phase_min)
    hi = min(hi_req, args.phase_max)
    if lo != lo_req:
        print(f"[anim] window clipped at phase1 start: {lo_req} -> {lo} "
              f"(= {(lo - ld) / fph:+.1f} h instead of {-args.before_h:+.1f} h)",
              flush=True)
    if hi != hi_req:
        print(f"[anim] window clipped at phase1 end: {hi_req} -> {hi} "
              f"(= {(hi - ld) / fph:+.1f} h instead of {args.after_h:+.1f} h)",
              flush=True)
    print(f"[anim] {pos} {ch}: last_div={ld}, window frames [{lo}, {hi}] "
          f"({(lo - ld) / fph:+.1f}..{(hi - ld) / fph:+.1f} h from last division)",
          flush=True)

    # ---- mother track (lineage rank==1) over the window -----------------
    csv = lineage_csv(channel_dir)
    if not csv.exists():
        raise FileNotFoundError(f"lineage CSV not found: {csv}")
    full = pd.read_csv(csv)

    if args.list_daughters:
        cands = daughter_candidates(full, ld)
        print(f"[anim] daughter candidates for {pos} {ch} (anchor last_div={ld}), "
              f"newest first:", flush=True)
        print(f"  {'idx':>3} {'tag':>4} {'div_frame':>9} {'t_birth_h':>9} "
              f"{'cell_id':>7} {'n_pts':>5}  span", flush=True)
        for i, c in enumerate(cands):
            tag = {0: "last", 1: "n-1", 2: "n-2", 3: "n-3"}.get(i, f"n-{i}")
            print(f"  {i:>3} {tag:>4} {c['div_frame']:>9} "
                  f"{(c['div_frame'] - ld) / fph:>+9.1f} {c['cell_id']:>7} "
                  f"{c['n_pts']:>5}  [{c['f0']},{c['f1']}]", flush=True)
        return 0

    m = full[full["rank"] == 1].sort_values("frame").reset_index(drop=True)
    win = m[(m["frame"] >= lo) & (m["frame"] <= hi)].copy()
    if win.empty:
        raise RuntimeError(f"no rank==1 mother rows in window [{lo}, {hi}]")
    win["t_h"] = (win["frame"].to_numpy(float) - ld) / fph

    # flag unreliable frames so the trace skips them (point not drawn) but the
    # phase/contour panel still shows the raw image -> continuity is preserved
    bad = win["is_outlier"].to_numpy(bool) if "is_outlier" in win else \
        np.zeros(len(win), bool)
    if "touches_border" in win:
        bad = bad | win["touches_border"].to_numpy(bool)
    ri = win["mean_ri"].to_numpy(float).copy()
    mass = win["mass_pg"].to_numpy(float).copy()
    ri[bad] = np.nan
    mass[bad] = np.nan

    frames = win["frame"].astype(int).to_numpy()
    t_all = win["t_h"].to_numpy(float)
    print(f"[anim] {len(win)} mother frames in window; "
          f"{int(bad.sum())} flagged unreliable (trace point skipped)", flush=True)

    # ---- non-mother daughter (sister) born at the (last - N) division ----
    d_frames = np.array([], int)
    d_t = d_ri = d_mass = np.array([], float)
    d_centroid: dict[int, tuple[float, float]] = {}
    dau_cid = dau_div = None
    if not args.no_daughter:
        dt = find_daughter_track(full, ld, args.daughter_div_back,
                                 cell_id=args.daughter_cell_id,
                                 div_frame=args.daughter_div_frame)
        if dt is None:
            print(f"[anim] no n-{args.daughter_div_back} daughter found "
                  f"(too few prior divisions?) -> mother-only movie", flush=True)
        else:
            dau_cid = dt.attrs.get("cell_id")
            dau_div = dt.attrs.get("div_frame")
            dw = dt[(dt["frame"] >= lo) & (dt["frame"] <= hi)].copy()
            d_frames = dw["frame"].astype(int).to_numpy()
            d_t = (d_frames.astype(float) - ld) / fph
            d_bad = dw["is_outlier"].to_numpy(bool) if "is_outlier" in dw else \
                np.zeros(len(dw), bool)
            d_ri = dw["mean_ri"].to_numpy(float).copy()
            d_mass = dw["mass_pg"].to_numpy(float).copy()
            d_ri[d_bad] = np.nan
            d_mass[d_bad] = np.nan
            for _, r in dw.iterrows():
                d_centroid[int(r["frame"])] = (float(r["centroid_y_px"]),
                                               float(r["centroid_x_px"]))
            print(f"[anim] daughter cell_id={dau_cid} born at div {dau_div} "
                  f"(= {(dau_div - ld) / fph:+.1f} h); tracked over frames "
                  f"[{int(d_frames.min())}, {int(d_frames.max())}] "
                  f"({len(d_frames)} pts) until border/track-end", flush=True)
    track_daughter = len(d_frames) > 0
    d_idx = {int(f): i for i, f in enumerate(d_frames)}

    # ---- pre-cache phase + per-label contours + mother label per frame ---
    phase_cache: dict[int, np.ndarray | None] = {}
    contour_cache: dict[int, dict[int, list[np.ndarray]]] = {}
    mother_label: dict[int, int] = {}
    daughter_label: dict[int, int] = {}
    img_shape = None
    n_missing_mask = 0
    for _, r in win.iterrows():
        fi = int(r["frame"])
        pp, mp = phase_path(channel_dir, fi), mask_path(channel_dir, fi)
        ph = None
        if pp.exists():
            ph = np.squeeze(tifffile.imread(pp)).astype(np.float32)
            img_shape = ph.shape
        phase_cache[fi] = ph
        if mp.exists():
            mk = np.squeeze(tifffile.imread(mp))
            if img_shape is None:
                img_shape = mk.shape
            contour_cache[fi] = label_contours(mk, efd=not args.raw_contours,
                                               k=args.efd_k)
            mother_label[fi] = label_at(mk, float(r["centroid_y_px"]),
                                        float(r["centroid_x_px"]))
            if fi in d_centroid:
                dcy, dcx = d_centroid[fi]
                daughter_label[fi] = label_at(mk, dcy, dcx)
            else:
                daughter_label[fi] = 0
        else:
            contour_cache[fi] = {}
            mother_label[fi] = 0
            daughter_label[fi] = 0
            n_missing_mask += 1
    if img_shape is None:
        raise RuntimeError("no readable phase/mask image found in the window")
    if n_missing_mask:
        print(f"[anim] {n_missing_mask} frames missing a mask "
              f"(contours skipped on those)", flush=True)
    H, W = img_shape

    # ---- trace y-limits from the window (NaN excluded, padded) ----------
    def padded_lim(vals: np.ndarray, frac: float = 0.08,
                   default: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            return default
        lo_v, hi_v = float(v.min()), float(v.max())
        if hi_v <= lo_v:
            hi_v = lo_v + 1e-6
        pad = (hi_v - lo_v) * frac
        return lo_v - pad, hi_v + pad

    if args.ri_min is not None and args.ri_max is not None:
        ri_lim = (args.ri_min, args.ri_max)          # shared fixed scale
    else:
        ri_lim = padded_lim(np.concatenate([ri, d_ri]), default=(1.34, 1.40))
    if args.mass_min is not None and args.mass_max is not None:
        mass_lim = (args.mass_min, args.mass_max)    # shared fixed scale (clips)
    else:
        mass_lim = padded_lim(np.concatenate([mass, d_mass]), default=(0.0, 1.0))
    x_lim = (-args.before_h, args.after_h)   # fixed [-30, +15] h regardless of clip

    # ---- figure: phase (row0) + colorbar, mean RI (row1), mass (row2) ---
    fig = plt.figure(figsize=(8.2, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.022],
                          height_ratios=[0.78, 1.0, 1.0],
                          hspace=0.32, wspace=0.025,
                          left=0.10, right=0.92, top=0.93, bottom=0.08)
    ax_img = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_ri = fig.add_subplot(gs[1, 0])
    ax_mass = fig.add_subplot(gs[2, 0], sharex=ax_ri)

    # phase panel
    im = ax_img.imshow(np.zeros((H, W), np.float32), cmap=CMAP,
                       vmin=args.vmin, vmax=args.vmax, interpolation="nearest",
                       aspect="auto", animated=True)
    ax_img.set_xlim(-0.5, W - 0.5)
    ax_img.set_ylim(H - 0.5, -0.5)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(f"{pos} {ch}  {args.mode_label}  (phase, inferno "
                     f"{args.vmin:g}-{args.vmax:g} rad; mother emphasised)",
                     fontsize=9, loc="left")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("phase [rad]", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # static scale bar (10 um), bottom-left of the phase panel
    bar_px = args.scalebar_um / px
    y_bar = H * 0.86
    x0 = W * 0.04
    ax_img.plot([x0, x0 + bar_px], [y_bar, y_bar], color="white", lw=3,
                solid_capstyle="butt", zorder=6)
    ax_img.text(x0 + bar_px / 2, y_bar - H * 0.07, f"{args.scalebar_um:g} µm",
                color="white", fontsize=8, ha="center", va="bottom", zorder=6)

    time_text = ax_img.text(0.985, 0.92, "", transform=ax_img.transAxes,
                            ha="right", va="top", fontsize=9, color="white",
                            bbox=dict(boxstyle="round,pad=0.3", fc=(0, 0, 0, 0.55),
                                      ec="white", lw=0.5), zorder=7)

    # trace panels
    for ax in (ax_ri, ax_mass):
        ax.axvline(0.0, ls=":", color="0.35", lw=1.2, zorder=1)  # division t=0
        ax.set_xlim(*x_lim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25, linestyle="--")

    ax_ri.set_ylim(*ri_lim)
    ax_ri.set_ylabel("mean RI")
    ax_ri.tick_params(labelbottom=False)
    ax_ri.set_title("mean RI" + ("  (mother vs daughter)" if track_daughter
                    else "  (mother)"), fontsize=8, loc="left")

    ax_mass.set_ylim(*mass_lim)
    ax_mass.set_ylabel("dry mass [pg]")
    ax_mass.set_xlabel("time from last division [h]")
    ax_mass.set_title("dry mass" + ("  (mother vs daughter)" if track_daughter
                      else "  (mother)"), fontsize=8, loc="left")

    (line_ri,) = ax_ri.plot([], [], color=MOTHER_TRACE, lw=1.4, label="mother")
    (mk_ri,) = ax_ri.plot([], [], "o", color=MOTHER_TRACE, ms=5)
    (line_mass,) = ax_mass.plot([], [], color=MOTHER_TRACE, lw=1.4, label="mother")
    (mk_mass,) = ax_mass.plot([], [], "o", color=MOTHER_TRACE, ms=5)

    # daughter lines (only meaningful when track_daughter)
    dlabel = (f"daughter (born n-{args.daughter_div_back} div)"
              if track_daughter else "daughter")
    (dline_ri,) = ax_ri.plot([], [], color=DAUGHTER_COLOR, lw=1.4, label=dlabel)
    (dmk_ri,) = ax_ri.plot([], [], "o", color=DAUGHTER_COLOR, ms=5)
    (dline_mass,) = ax_mass.plot([], [], color=DAUGHTER_COLOR, lw=1.4, label=dlabel)
    (dmk_mass,) = ax_mass.plot([], [], "o", color=DAUGHTER_COLOR, ms=5)
    if track_daughter:
        ax_ri.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=2)

    # index lookup so update() is O(1) per frame
    idx_of = {int(f): i for i, f in enumerate(frames)}
    contour_artists: list = []

    def clear_contours() -> None:
        for art in contour_artists:
            art.remove()
        contour_artists.clear()

    def update(fi: int):
        # ---- phase image + contours ----
        ph = phase_cache.get(fi)
        im.set_data(ph if ph is not None else np.zeros((H, W), np.float32))
        clear_contours()
        ml = mother_label.get(fi, 0)
        dl = daughter_label.get(fi, 0)
        for lbl, conts in contour_cache.get(fi, {}).items():
            if ml != 0 and lbl == ml:
                col, lw = MOTHER_COLOR, 1.8
            elif dl != 0 and lbl == dl:
                col, lw = DAUGHTER_COLOR, 1.8
            else:
                col, lw = OTHER_COLOR, 0.8
            for c in conts:
                (art,) = ax_img.plot(c[:, 1], c[:, 0], color=col, lw=lw, zorder=5)
                contour_artists.append(art)

        # ---- time stamp ----
        t_now = (fi - ld) / fph
        time_text.set_text(f"t = {t_now:+.1f} h\nframe {fi}")

        # ---- progressive traces up to current frame (NaN points skipped) ---
        i = idx_of[fi]
        xr = t_all[:i + 1]
        yri = ri[:i + 1]
        ymass = mass[:i + 1]
        fin_ri = np.isfinite(yri)
        fin_mass = np.isfinite(ymass)
        line_ri.set_data(xr[fin_ri], yri[fin_ri])
        line_mass.set_data(xr[fin_mass], ymass[fin_mass])

        if np.isfinite(ri[i]):
            mk_ri.set_data([t_now], [ri[i]])
        else:
            mk_ri.set_data([], [])
        if np.isfinite(mass[i]):
            mk_mass.set_data([t_now], [mass[i]])
        else:
            mk_mass.set_data([], [])

        # ---- daughter progressive traces (only where tracked) ----
        if track_daughter:
            dsel = d_frames <= fi
            dxr = d_t[dsel]
            dyri = d_ri[dsel]
            dymass = d_mass[dsel]
            fdri = np.isfinite(dyri)
            fdmass = np.isfinite(dymass)
            dline_ri.set_data(dxr[fdri], dyri[fdri])
            dline_mass.set_data(dxr[fdmass], dymass[fdmass])
            j = d_idx.get(fi)
            if j is not None and np.isfinite(d_ri[j]):
                dmk_ri.set_data([t_now], [d_ri[j]])
            else:
                dmk_ri.set_data([], [])
            if j is not None and np.isfinite(d_mass[j]):
                dmk_mass.set_data([t_now], [d_mass[j]])
            else:
                dmk_mass.set_data([], [])
        return ()

    # ---- output path -----------------------------------------------------
    if args.out is not None:
        out_path = args.out
    else:
        out_path = Path("F:/") / (
            f"{pos}_{ch}_swelldeath_lastdiv_-{args.before_h:g}h+{args.after_h:g}h_"
            f"inferno_{args.fps}fps.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = manimation.FFMpegWriter(
        fps=args.fps, codec="libx264", bitrate=-1,
        extra_args=[
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-crf", "20",
            "-preset", "medium",
        ],
    )
    print(f"[anim] writing {out_path} at {args.fps} fps ({len(frames)} frames)",
          flush=True)
    anim = manimation.FuncAnimation(fig, update, frames=frames,
                                    interval=1000.0 / args.fps, blit=False)
    anim.save(str(out_path), writer=writer, dpi=args.dpi,
              progress_callback=lambda i, n: (
                  None if i % 50 else print(f"  [anim] {i}/{n}", flush=True)))
    plt.close(fig)

    # ---- sidecar: trace arrays for later restyle/repro -------------------
    npz_path = out_path.with_suffix(".data.npz")
    np.savez(
        npz_path,
        frame_index=frames,
        t_h=t_all,
        mean_ri=ri,
        mass_pg=mass,
        volume_um3_rod=win["volume_um3_rod"].to_numpy(float),
        unreliable=bad,
        mother_label=np.array([mother_label[int(f)] for f in frames]),
        last_div_frame=ld,
        pixel_size_um=px,
        frames_per_hour=fph,
        window_frames=np.array([lo, hi]),
        vmin=args.vmin, vmax=args.vmax, fps=args.fps,
        pos=pos, ch=ch,
        daughter_frame=d_frames,
        daughter_t_h=d_t,
        daughter_mean_ri=d_ri,
        daughter_mass_pg=d_mass,
        daughter_cell_id=(-1 if dau_cid is None else int(dau_cid)),
        daughter_div_frame=(-1 if dau_div is None else int(dau_div)),
    )
    print(f"[anim] done. {out_path}", flush=True)
    print(f"[anim] trace data -> {npz_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
