"""Animate phase2 (starvation -> 2% refeed) revival/death of a mother trap.

Time-synchronised QPI movie anchored at the 2% refeed (t = 0 = refeed frame,
default 2885 for 260517), window -10 h .. +30 h:

  1. phase image (landscape) in inferno on a fixed 0-1.8 rad scale, EFD-smoothed
     mask contours, 10 um scale bar, "t = +X.X h / frame N" stamp;
  2. mean RI vs time;
  3. dry mass [pg] vs time.

Two cell-tracking modes:
  * default (multi-cell): the mother PLUS every other cell in the trap is tracked
    by lineage cell_id and drawn, each in its own colour, until it touches the
    image border ("oob") and can no longer be followed -- so you see whether the
    non-mother cells recover their mean RI after refeed;
  * --mother-only: just the rank==1 mother (for never-revived / elongation-tip
    lineages where only the mother matters).

Reuses the EFD smoothing, contour and IO helpers from _animate_vol_ri_f_ch so the
look matches the phase1 swelling/elongation movies. mean_ri / mass_pg come from
the lineage CSV (pipeline calibration); nothing is recomputed.

Usage:
  python scripts/_animate_phase2_cells.py \
      --channel-dir "F:/260517/.../PosN/.../z000/chXX" \
      --refeed-frame 2885 --ri-min 1.36 --ri-max 1.42 --mass-min 0 --mass-max 60
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

import imageio_ffmpeg  # noqa: E402

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

sys.path.insert(0, str(Path(__file__).parent))
from _animate_vol_ri_f_ch import (  # noqa: E402
    CMAP, MOTHER_COLOR, OTHER_COLOR, PX_DEFAULT,
    phase_path, mask_path, lineage_csv, label_at, label_contours,
    mother_cell_ids, _track_cell,
)

# distinct colours for non-mother tracked cells (Okabe-Ito minus cyan/blue,
# which read as "mother"); cycled if there are more cells than colours.
OTHER_CELL_COLORS = ["#E69F00", "#009E73", "#CC79A7", "#F0E442",
                     "#D55E00", "#56B4E9", "#999999"]
UNTRACKED = "#7a1f1a"   # dim red: cells present but not in the tracked set


def derive_pos_ch(channel_dir: Path) -> tuple[str, str]:
    parts = channel_dir.resolve().parts
    pos = next((p for p in parts if re.fullmatch(r"Pos\d+", p)), "Pos")
    ch = next((p for p in reversed(parts) if re.fullmatch(r"ch\d+", p)), "ch")
    return pos, ch


def load_run_params(ch_dir: Path) -> dict:
    p = ch_dir / "inference_out" / "lineage_out" / "lineage_run_params.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


def build_cell_tracks(full: pd.DataFrame, lo: int, hi: int, mother_only: bool,
                      min_track: int, max_cells: int) -> list[dict]:
    """Per-cell tracks overlapping [lo, hi].

    Each entry: {cell_id, is_mother, frames, t(set later), ri, mass, centroid:
    {frame:(cy,cx)}, color}. Tracks are border-truncated (via _track_cell) then
    clipped to the window; only cells with >= min_track in-window points survive.
    Mother (rank==1) is always first; others ordered by first appearance and
    capped at max_cells."""
    mom = mother_cell_ids(full)
    win_ids = full[(full["frame"] >= lo) & (full["frame"] <= hi)]["cell_id"].unique()
    entries = []
    for cid in win_ids:
        cid = int(cid)
        is_mom = cid in mom
        if mother_only and not is_mom:
            continue
        t = _track_cell(full, cid)                      # border-truncated
        t = t[(t["frame"] >= lo) & (t["frame"] <= hi)]
        if len(t) < min_track:
            continue
        entries.append({"cell_id": cid, "is_mother": is_mom, "track": t,
                        "first": int(t["frame"].min())})
    # order: mothers first, then by first appearance
    entries.sort(key=lambda e: (not e["is_mother"], e["first"]))
    mothers = [e for e in entries if e["is_mother"]]
    others = [e for e in entries if not e["is_mother"]]
    others = others[:max(0, max_cells - len(mothers))]
    entries = mothers + others
    ci = 0
    for e in entries:
        if e["is_mother"]:
            e["color"] = MOTHER_COLOR
        else:
            e["color"] = OTHER_CELL_COLORS[ci % len(OTHER_CELL_COLORS)]
            ci += 1
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel-dir", required=True, type=Path)
    ap.add_argument("--refeed-frame", type=int, default=2885,
                    help="2%% refeed frame = t0 (default 2885 for 260517)")
    ap.add_argument("--before-h", type=float, default=10.0)
    ap.add_argument("--after-h", type=float, default=30.0)
    ap.add_argument("--mother-only", action="store_true",
                    help="track only the rank==1 mother (dead / elongation-tip)")
    ap.add_argument("--min-track", type=int, default=8,
                    help="min in-window frames for a cell to be drawn")
    ap.add_argument("--max-cells", type=int, default=7,
                    help="max cells drawn (mother always kept)")
    ap.add_argument("--vmin", type=float, default=0.0)
    ap.add_argument("--vmax", type=float, default=1.8)
    ap.add_argument("--ri-min", type=float, default=None)
    ap.add_argument("--ri-max", type=float, default=None)
    ap.add_argument("--mass-min", type=float, default=None)
    ap.add_argument("--mass-max", type=float, default=None)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--phase-min", type=int, default=0)
    ap.add_argument("--phase-max", type=int, default=3747)
    ap.add_argument("--efd-k", type=int, default=6)
    ap.add_argument("--raw-contours", action="store_true")
    ap.add_argument("--scalebar-um", type=float, default=10.0)
    ap.add_argument("--mode-label", default="phase2 revival")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cd = args.channel_dir
    meta = load_run_params(cd)
    px = float(meta.get("pixel_size_um", PX_DEFAULT))
    fph = 60.0 / float(meta.get("time_interval_min", 5.0))
    pos, ch = derive_pos_ch(cd)

    r0 = int(args.refeed_frame)
    lo = max(r0 - int(round(args.before_h * fph)), args.phase_min)
    hi = min(r0 + int(round(args.after_h * fph)), args.phase_max)
    if hi != r0 + int(round(args.after_h * fph)):
        print(f"[p2] window clipped at top: -> {hi} "
              f"({(hi - r0) / fph:+.1f} h)", flush=True)
    print(f"[p2] {pos} {ch}: refeed={r0}, window [{lo}, {hi}] "
          f"({(lo - r0) / fph:+.1f}..{(hi - r0) / fph:+.1f} h from refeed)",
          flush=True)

    csv = lineage_csv(cd)
    if not csv.exists():
        raise FileNotFoundError(f"lineage CSV not found: {csv}")
    full = pd.read_csv(csv)

    cells = build_cell_tracks(full, lo, hi, args.mother_only,
                              args.min_track, args.max_cells)
    if not cells:
        raise RuntimeError("no cells to track in the window")
    # per-cell display arrays (NaN where unreliable)
    for e in cells:
        t = e["track"]
        bad = (t["is_outlier"].to_numpy(bool) | t["touches_border"].to_numpy(bool)
               | (t["mass_pg"].to_numpy(float) < 10.0))
        e["frames"] = t["frame"].astype(int).to_numpy()
        e["t"] = (e["frames"].astype(float) - r0) / fph
        e["ri"] = np.where(~bad, t["mean_ri"].to_numpy(float), np.nan)
        e["mass"] = np.where(~bad, t["mass_pg"].to_numpy(float), np.nan)
        e["centroid"] = {int(rr["frame"]): (float(rr["centroid_y_px"]),
                                            float(rr["centroid_x_px"]))
                         for _, rr in t.iterrows()}
        e["idx"] = {int(f): i for i, f in enumerate(e["frames"])}
    n_mom = sum(e["is_mother"] for e in cells)
    print(f"[p2] tracking {len(cells)} cells ({n_mom} mother, "
          f"{len(cells) - n_mom} other) until border/track-end", flush=True)

    # ---- frames to animate = union of all cell frames in window ----------
    frames = np.array(sorted(set(int(f) for e in cells for f in e["frames"])))

    # ---- pre-cache phase, contours, and per-frame {mask_label: color} ----
    phase_cache: dict[int, np.ndarray | None] = {}
    contour_cache: dict[int, dict[int, list[np.ndarray]]] = {}
    color_cache: dict[int, dict[int, str]] = {}
    img_shape = None
    for fi in frames:
        fi = int(fi)
        pp, mp = phase_path(cd, fi), mask_path(cd, fi)
        ph = None
        if pp.exists():
            ph = np.squeeze(tifffile.imread(pp)).astype(np.float32)
            img_shape = ph.shape
        phase_cache[fi] = ph
        lblcol: dict[int, str] = {}
        if mp.exists():
            mk = np.squeeze(tifffile.imread(mp))
            if img_shape is None:
                img_shape = mk.shape
            contour_cache[fi] = label_contours(mk, efd=not args.raw_contours,
                                               k=args.efd_k)
            for e in cells:
                c = e["centroid"].get(fi)
                if c is None:
                    continue
                lab = label_at(mk, c[0], c[1])
                if lab:
                    lblcol[lab] = e["color"]
        else:
            contour_cache[fi] = {}
        color_cache[fi] = lblcol
    if img_shape is None:
        raise RuntimeError("no readable phase/mask image found in the window")
    H, W = img_shape

    # ---- y-limits ----
    def padded(vals, frac=0.08, default=(0.0, 1.0)):
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            return default
        a, b = float(v.min()), float(v.max())
        if b <= a:
            b = a + 1e-6
        return a - (b - a) * frac, b + (b - a) * frac

    all_ri = np.concatenate([e["ri"] for e in cells])
    all_mass = np.concatenate([e["mass"] for e in cells])
    ri_lim = (args.ri_min, args.ri_max) if None not in (args.ri_min, args.ri_max) \
        else padded(all_ri, default=(1.34, 1.42))
    mass_lim = (args.mass_min, args.mass_max) if None not in (args.mass_min, args.mass_max) \
        else padded(all_mass, default=(0.0, 60.0))
    x_lim = (-args.before_h, args.after_h)

    # ---- figure ----
    fig = plt.figure(figsize=(8.2, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.022],
                          height_ratios=[0.78, 1.0, 1.0],
                          hspace=0.32, wspace=0.025,
                          left=0.10, right=0.92, top=0.93, bottom=0.08)
    ax_img = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1])
    ax_ri = fig.add_subplot(gs[1, 0])
    ax_mass = fig.add_subplot(gs[2, 0], sharex=ax_ri)

    im = ax_img.imshow(np.zeros((H, W), np.float32), cmap=CMAP, vmin=args.vmin,
                       vmax=args.vmax, interpolation="nearest", aspect="auto",
                       animated=True)
    ax_img.set_xlim(-0.5, W - 0.5); ax_img.set_ylim(H - 0.5, -0.5)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title(f"{pos} {ch}  {args.mode_label}  (phase, inferno "
                     f"{args.vmin:g}-{args.vmax:g} rad; mother = cyan)",
                     fontsize=9, loc="left")
    cb = fig.colorbar(im, cax=cax); cb.set_label("phase [rad]", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    bar_px = args.scalebar_um / px
    ax_img.plot([W * 0.04, W * 0.04 + bar_px], [H * 0.86, H * 0.86],
                color="white", lw=3, solid_capstyle="butt", zorder=6)
    ax_img.text(W * 0.04 + bar_px / 2, H * 0.86 - H * 0.07,
                f"{args.scalebar_um:g} µm", color="white", fontsize=8,
                ha="center", va="bottom", zorder=6)
    time_text = ax_img.text(0.985, 0.92, "", transform=ax_img.transAxes,
                            ha="right", va="top", fontsize=9, color="white",
                            bbox=dict(boxstyle="round,pad=0.3", fc=(0, 0, 0, 0.55),
                                      ec="white", lw=0.5), zorder=7)

    for ax in (ax_ri, ax_mass):
        ax.axvline(0.0, ls=":", color="0.35", lw=1.2)   # refeed t=0
        ax.set_xlim(*x_lim)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25, linestyle="--")
    ax_ri.set_ylim(*ri_lim); ax_ri.set_ylabel("mean RI")
    ax_ri.tick_params(labelbottom=False)
    ax_ri.set_title("mean RI", fontsize=8, loc="left")
    ax_mass.set_ylim(*mass_lim); ax_mass.set_ylabel("dry mass [pg]")
    ax_mass.set_xlabel("time from 2% refeed [h]")
    ax_mass.set_title("dry mass", fontsize=8, loc="left")

    # one line + marker per cell, per panel
    for e in cells:
        lw = 1.8 if e["is_mother"] else 1.1
        lbl = "mother" if e["is_mother"] else f"cell {e['cell_id']}"
        e["lri"], = ax_ri.plot([], [], color=e["color"], lw=lw, label=lbl)
        e["mri"], = ax_ri.plot([], [], "o", color=e["color"], ms=4)
        e["lmass"], = ax_mass.plot([], [], color=e["color"], lw=lw)
        e["mmass"], = ax_mass.plot([], [], "o", color=e["color"], ms=4)
    ax_ri.legend(loc="upper left", fontsize=6, ncol=2, framealpha=0.9)

    contour_artists: list = []

    def update(fi: int):
        ph = phase_cache.get(fi)
        im.set_data(ph if ph is not None else np.zeros((H, W), np.float32))
        for art in contour_artists:
            art.remove()
        contour_artists.clear()
        cmap_lab = color_cache.get(fi, {})
        for lab, conts in contour_cache.get(fi, {}).items():
            col = cmap_lab.get(lab, UNTRACKED)
            lw = 1.8 if col == MOTHER_COLOR else (1.3 if col != UNTRACKED else 0.7)
            for c in conts:
                (a,) = ax_img.plot(c[:, 1], c[:, 0], color=col, lw=lw, zorder=5)
                contour_artists.append(a)
        time_text.set_text(f"t = {(fi - r0) / fph:+.1f} h\nframe {fi}")

        for e in cells:
            sel = e["frames"] <= fi
            xr = e["t"][sel]
            yri = e["ri"][sel]; ymass = e["mass"][sel]
            e["lri"].set_data(xr[np.isfinite(yri)], yri[np.isfinite(yri)])
            e["lmass"].set_data(xr[np.isfinite(ymass)], ymass[np.isfinite(ymass)])
            j = e["idx"].get(fi)
            tn = (fi - r0) / fph
            if j is not None and np.isfinite(e["ri"][j]):
                e["mri"].set_data([tn], [e["ri"][j]])
            else:
                e["mri"].set_data([], [])
            if j is not None and np.isfinite(e["mass"][j]):
                e["mmass"].set_data([tn], [e["mass"][j]])
            else:
                e["mmass"].set_data([], [])
        return ()

    if args.out is not None:
        out_path = args.out
    else:
        tag = "p2dead" if args.mother_only else "p2revival"
        out_path = Path("D:/") / (
            f"{pos}_{ch}_{tag}_refeed_-{args.before_h:g}h+{args.after_h:g}h_"
            f"inferno_{args.fps}fps.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = manimation.FFMpegWriter(
        fps=args.fps, codec="libx264", bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-crf", "20", "-preset", "medium"])
    print(f"[p2] writing {out_path} ({len(frames)} frames, {args.fps} fps)",
          flush=True)
    anim = manimation.FuncAnimation(fig, update, frames=frames,
                                    interval=1000.0 / args.fps, blit=False)
    anim.save(str(out_path), writer=writer, dpi=args.dpi,
              progress_callback=lambda i, n: (
                  None if i % 50 else print(f"  [p2] {i}/{n}", flush=True)))
    plt.close(fig)

    npz = out_path.with_suffix(".data.npz")
    np.savez(npz, refeed_frame=r0, frames_per_hour=fph, window=np.array([lo, hi]),
             pos=pos, ch=ch, n_cells=len(cells),
             **{f"cell{e['cell_id']}_frame": e["frames"] for e in cells},
             **{f"cell{e['cell_id']}_t_h": e["t"] for e in cells},
             **{f"cell{e['cell_id']}_mean_ri": e["ri"] for e in cells},
             **{f"cell{e['cell_id']}_mass_pg": e["mass"] for e in cells},
             **{f"cell{e['cell_id']}_is_mother": e["is_mother"] for e in cells})
    print(f"[p2] done. {out_path}\n[p2] data -> {npz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
