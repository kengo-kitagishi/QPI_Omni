"""ICBBS poster Methods strip: one mother cell cycle with its mask geometry.

Default cycle: Pos18 ch02 633-670. ``--pos/--ch/--f0/--f1`` pick another one;
``--random N`` draws N gap-free 2% cycles from distinct positions (edge traps excluded).

Handoff: docs/issues/poster_pos18_ch02_mask_geometry.md (PR #32).

Every overlay is computed from the master label masks with the tracker's own call
(``regionprops(...).image`` padded by 6 px -> ``efd_section_geometry`` with the
dataset's ``contour_offset_px``), so the drawn contour, long axis and chords are the
geometry behind ``long_axis_um`` / ``volume_um3_efd`` in the master. The geometry is
computed in trap-crop pixel coordinates and moved into the tile by the same exact
90-degree rotation as the phase pixels (no interpolation of phase or mask).

The v001 reference (2026-06-23, ``_overlay_efd_on_panelA_strip.py``) used the
June-2026 masks under f:/260517 and PCA rotation with ``ndimage.rotate`` of the
phase and of the binary mask (order=1, >0.5). Its contour sits 0.1-1.2 px below the
bright phase centroid, most on the short cells; the masks it used no longer exist.

Figures (inbox via figure_logger):
  f001  contour-only strip
  f002  contour + long axis + every short-axis chord
  f003  QC: first / middle / shortest / longest tiles, mask pixels vs contour
Also writes a per-tile manifest CSV (copied into the inbox run).

Usage:
    python scripts/fig_poster_mask_geometry_strip.py [--pos Pos18 --ch ch02 --f0 633 --f1 670]
    python scripts/fig_poster_mask_geometry_strip.py --random 4 --seed 20260930
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent))
import figure_logger  # noqa: E402,F401  (applies paper.mplstyle)
from figure_logger import save_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from mask_volume_schematic import efd_section_geometry  # noqa: E402
import qpi_paths  # noqa: E402

POS, CH = "Pos18", "ch02"
F0, F1 = 633, 670
N_TILES = 13
CROP_MARGIN = 2          # px of phase kept around the mask bbox
GAP = 2                  # px of background between tiles
VMIN, VMAX, CMAP = 0.0, 1.8, "inferno"
PAD = 6                  # same pad as central_cell_lineage_tracker.extract_cells_from_frame

CONTOUR_C = "#56B4E9"    # Okabe-Ito sky blue
LONG_C = "#FF2A2A"       # red, no outline
CHORD_C = "#FFFFFF"


# --------------------------------------------------------------------------- io
def master_paths() -> dict:
    mdir = qpi_paths.master_dir()
    if mdir is None:
        raise SystemExit("no master selected (qpi_paths.master_dir() is None)")
    ch_master = mdir / "per_channel" / POS / CH
    params = json.loads((ch_master / "lineage_run_params.json").read_text())
    mask_dir = Path(params["channel_dir"]) / "inference_out"
    cfg = next((mdir / "inputs").glob("*.yaml"))
    ds = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    raw_root = Path(ds["paths"]["raw_root"])
    pos_n = int(POS[3:])
    for ov in ds["paths"].get("raw_root_overrides") or []:
        if ov["pos_min"] <= pos_n <= ov["pos_max"]:
            raw_root = Path(ov["root"])
    phase_dir = raw_root / POS / ds["paths"]["channel_rel"] / CH
    return {"master": mdir, "lineage": ch_master / "lineage_data3D.csv",
            "params": params, "mask_dir": mask_dir, "phase_dir": phase_dir,
            "dataset_yaml": cfg, "offset": float(ds["tracking"]["contour_offset_px"])}


def select_rows(lin: pd.DataFrame) -> pd.DataFrame:
    """Mother rows 633-670, then the same deterministic sub-sampling as v001
    (first and last kept, the rest evenly spaced)."""
    seg = (lin[(lin["rank"] == 1) & lin.frame.between(F0, F1)]
           .sort_values("frame").reset_index(drop=True))
    if seg.cell_id.nunique() != 1:
        raise SystemExit(f"more than one mother cell_id in {F0}-{F1}: {seg.cell_id.unique()}")
    idx = np.unique(np.r_[0, np.linspace(0, len(seg) - 1, N_TILES).round(),
                          len(seg) - 1]).astype(int)
    return seg.iloc[idx].reset_index(drop=True)


# --------------------------------------------------------------------- geometry
def cell_geometry(mask: np.ndarray, label: int, offset: float):
    """Tracker-identical geometry for one label, returned in trap (x=col, y=row)."""
    props = [p for p in measure.regionprops(mask) if p.label == label]
    if not props:
        return None
    p = props[0]
    geo = efd_section_geometry(np.pad(p.image, PAD), pixel_size_um=1.0,
                               contour_offset_px=offset)
    if geo is None:
        return None
    minr, minc = p.bbox[0], p.bbox[1]
    shift = np.array([minc - PAD, minr - PAD], float)
    w = np.asarray(geo.w_perp_px, float)
    return {"prop": p, "contour": np.asarray(geo.contour_xy[0], float) + shift,
            "medial": np.asarray(geo.medial_xy, float) + shift,
            "p0": np.asarray(geo.slice_p0_xy, float) + shift,
            "p1": np.asarray(geo.slice_p1_xy, float) + shift,
            "w": w, "arc": np.asarray(geo.arc_step_px, float),
            "volume_px3": float(geo.volume_px3)}


def short_axis_chords(w: np.ndarray) -> np.ndarray:
    """Indices of the chords averaged into short_axis_um (tracker _yellow_axes rule)."""
    n = len(w)
    ok = w > 0
    trim0 = int(0.10 * n)
    core = w[trim0:n - trim0] if n > 2 * trim0 + 1 else w
    core = core[core > 0]
    rough = float(np.mean(core)) if core.size else float(np.mean(w[ok]))
    cap = int(np.clip(rough / 2.0, 1, max(n // 3, 1)))
    idx = np.arange(cap, n - cap) if n > 2 * cap + 1 else np.arange(n)
    idx = idx[w[idx] > 0]
    return idx[w[idx] >= 0.5 * w[idx].max()]


def chord_check(g) -> dict:
    """Endpoint distance to the contour and whether the long axis stays inside."""
    C = g["contour"]
    seg_a, seg_b = C[:-1], C[1:]

    def dist(P):
        e = seg_b - seg_a
        t = np.clip(np.einsum("ij,ij->i", P - seg_a, e) / np.einsum("ij,ij->i", e, e), 0, 1)
        return float(np.min(np.linalg.norm(seg_a + t[:, None] * e - P, axis=1)))

    valid = np.flatnonzero(g["w"] > 0)
    d_end = max(max(dist(g["p0"][i]), dist(g["p1"][i])) for i in valid)
    inside = MplPath(C).contains_points(g["medial"], radius=1e-6)
    return {"n_valid_chords": int(valid.size), "n_chords": int(len(g["w"])),
            "max_endpoint_dist_px": d_end, "long_axis_inside_frac": float(inside.mean())}


# ------------------------------------------------------------------------ tiles
def to_tile(xy: np.ndarray, r0: int, c0: int, width: int, top_high_x: bool) -> np.ndarray:
    """Trap (x=col, y=row) -> tile (x, y) for the exact 90-degree rotation in make_tile."""
    col, row = xy[..., 0] - c0, xy[..., 1] - r0
    if top_high_x:            # np.rot90(crop, 1): out[i, j] = in[j, W-1-i]
        return np.stack([row, (width - 1) - col], axis=-1)
    return np.stack([row, col], axis=-1)   # crop.T


def make_tile(img: np.ndarray, top_high_x: bool) -> np.ndarray:
    return np.rot90(img, 1) if top_high_x else img.T


def build(paths: dict, rows: pd.DataFrame, px: float, top_high_x: bool):
    tiles, recs = [], []
    for _, r in rows.iterrows():
        f = int(r.frame)
        pp = paths["phase_dir"] / f"img_{f:09d}_ph_000_phase.tif"
        mp = paths["mask_dir"] / f"img_{f:09d}_ph_000_phase_masks.tif"
        phase = np.squeeze(tifffile.imread(pp)).astype(np.float32)
        mask = np.squeeze(tifffile.imread(mp))
        if phase.shape != mask.shape:
            raise SystemExit(f"frame {f}: phase {phase.shape} != mask {mask.shape}")
        lbl = int(r.mask_label)
        g = cell_geometry(mask, lbl, paths["offset"])
        if g is None:
            print(f"frame {f}: geometry failed -> tile rejected")
            continue
        p = g["prop"]
        # the label must be the lineage row's cell (area and centroid identical)
        d_cen = float(np.hypot(p.centroid[0] - r.centroid_y_px, p.centroid[1] - r.centroid_x_px))
        if p.area != int(r.area_px) or d_cen > 0.05:
            raise SystemExit(f"frame {f}: label {lbl} area {p.area} vs {r.area_px}, "
                             f"centroid off {d_cen:.3f} px")
        long_um = float(g["arc"].sum() * px)
        vol_um3 = g["volume_px3"] * px ** 3
        H, W = mask.shape
        minr, minc, maxr, maxc = p.bbox
        r0, r1 = max(minr - CROP_MARGIN, 0), min(maxr + CROP_MARGIN, H)
        c0, c1 = max(minc - CROP_MARGIN, 0), min(maxc + CROP_MARGIN, W)
        crop_ph, crop_bw = phase[r0:r1, c0:c1], (mask[r0:r1, c0:c1] == lbl)
        width = c1 - c0
        T = lambda xy: to_tile(xy, r0, c0, width, top_high_x)  # noqa: E731
        sa = short_axis_chords(g["w"])
        shown = np.flatnonzero(g["w"] > 0)   # every chord used for volume_um3_efd
        chk = chord_check(g)
        tiles.append({"phase": make_tile(crop_ph, top_high_x),
                      "bw": make_tile(crop_bw, top_high_x),
                      "contour": T(g["contour"]), "medial": T(g["medial"]),
                      "p0": T(g["p0"]), "p1": T(g["p1"]), "shown": shown,
                      "frame": f, "time_h": float(r.time_h)})
        recs.append({
            "frame": f, "time_h": float(r.time_h),
            "time_from_birth_min": round((float(r.time_h) - float(rows.time_h.iloc[0])) * 60),
            "cell_id": int(r.cell_id), "mask_label": lbl,
            "phase_path": str(pp), "mask_path": str(mp),
            "crop_r0": r0, "crop_r1": r1, "crop_c0": c0, "crop_c1": c1,
            "long_axis_um_master": float(r.long_axis_um), "long_axis_um_redrawn": long_um,
            "volume_um3_efd_master": float(r.volume_um3_efd), "volume_um3_efd_redrawn": vol_um3,
            "short_axis_um_master": float(r.short_axis_um),
            "short_axis_um_redrawn": float(g["w"][sa].mean() * px),
            "n_short_axis_chords": int(sa.size), "shown_chord_idx": " ".join(map(str, shown)),
            **chk,
        })
    return tiles, pd.DataFrame(recs)


def layout(tiles: list[dict]):
    Hs = max(t["phase"].shape[0] for t in tiles)
    Wtot = sum(t["phase"].shape[1] for t in tiles) + GAP * (len(tiles) - 1)
    canvas = np.full((Hs, Wtot), np.nan, np.float32)
    x = 0
    for t in tiles:
        h, w = t["phase"].shape
        y = (Hs - h) // 2
        canvas[y:y + h, x:x + w] = t["phase"]
        t["off"] = np.array([x, y], float)
        x += w + GAP
    return canvas


# ----------------------------------------------------------------------- render
def draw_geometry(ax, t, with_axes: bool, lw_scale: float = 1.0):
    o = t["off"]
    if with_axes:
        for i in t["shown"]:
            a, b = t["p0"][i] + o, t["p1"][i] + o
            ax.plot([a[0], b[0]], [a[1], b[1]], color=CHORD_C, lw=0.35 * lw_scale,
                    solid_capstyle="butt", zorder=3)
        m = t["medial"] + o
        ax.plot(m[:, 0], m[:, 1], color=LONG_C, lw=1.0 * lw_scale, zorder=4)
    c = t["contour"] + o
    ax.plot(c[:, 0], c[:, 1], color=CONTOUR_C, lw=0.9 * lw_scale, zorder=5)


def render_strip(canvas, tiles, px, with_axes: bool):
    cH, cW = canvas.shape
    fig_w = 183 / 25.4
    top_mm = 3.2
    fig_h = fig_w * cH / cW + top_mm / 25.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, (fig_h - top_mm / 25.4) / fig_h])
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("black")
    ax.imshow(np.ma.masked_invalid(canvas), cmap=cmap, vmin=VMIN, vmax=VMAX,
              interpolation="nearest")
    ax.set_xlim(-0.5, cW - 0.5)
    ax.set_ylim(cH - 0.5, -0.5)
    ax.set_axis_off()
    t0 = tiles[0]["time_h"]
    for t in tiles:
        draw_geometry(ax, t, with_axes)
        xc = t["off"][0] + t["phase"].shape[1] / 2 - 0.5
        dt_min = round((t["time_h"] - t0) * 60)
        ax.text(xc, -1.0, f"{dt_min} min" if t is tiles[0] else f"{dt_min}", ha="center",
                va="bottom", fontsize=7, clip_on=False)
    # 5 um scale bar inside the canvas, bottom-left
    sb = 5.0 / px
    x0, y0 = 1.5, cH - 2.5
    ax.plot([x0, x0 + sb], [y0, y0], color="white", lw=2.0, solid_capstyle="butt", zorder=6)
    ax.text(x0 + sb / 2, y0 - 1.0, "5 µm", color="white", fontsize=7, ha="center",
            va="bottom", zorder=6)
    return fig


def render_qc(tiles, manifest):
    L = manifest.long_axis_um_redrawn.to_numpy()
    picks = {"first": 0, "middle": len(tiles) // 2,
             "shortest": int(np.argmin(L)), "longest": int(np.argmax(L))}
    fig, axes = plt.subplots(2, len(picks), figsize=(183 / 25.4, 110 / 25.4))
    for k, (name, j) in enumerate(picks.items()):
        t = dict(tiles[j], off=np.zeros(2))
        for row, with_axes in ((0, False), (1, True)):
            ax = axes[row, k]
            ax.imshow(t["phase"], cmap=CMAP, vmin=VMIN, vmax=VMAX, interpolation="nearest")
            if row == 0:   # mask pixel edges in thin white
                edge = t["bw"].astype(float)
                ax.contour(edge, levels=[0.5], colors="white", linewidths=0.4,
                           corner_mask=False)
            draw_geometry(ax, t, with_axes, lw_scale=1.4)
            ax.set_axis_off()
        m = manifest.iloc[j]
        axes[0, k].set_title(f"{name}: img {t['frame']}\n"
                             f"L {m.long_axis_um_redrawn:.2f} µm (master {m.long_axis_um_master:.2f})",
                             fontsize=6)
        axes[1, k].set_title(f"chord end–contour ≤ {m.max_endpoint_dist_px:.1e} px\n"
                             f"axis inside {m.long_axis_inside_frac:.0%}", fontsize=6)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------- main
EDGE_CHANNELS = {"ch00", "ch11"}
PHASE1_END = 2017        # last clean 2% frame of 260517


def random_cycles(n: int, seed: int) -> list[tuple[str, str, int, int]]:
    """N mother cycles, one per position, drawn at random from the gap-free 2% cycles.

    Cycle = validated mother division d_k to the frame before the next validated one.
    Kept only if every frame f0..f1 has exactly one rank-1 row of one cell_id, none is an
    outlier or touches the border, and no single-frame long-axis dip (< 0.7 x both
    neighbours). No quality ranking.
    """
    mdir = qpi_paths.master_dir()
    con = mdir / "consolidated"
    dv = pd.read_csv(con / "all_cells_divisions_qc.csv.gz")
    dv = dv[dv.is_mother_division & dv.validated & ~dv.ch.isin(EDGE_CHANNELS)]
    cand = []
    for (pos, ch), g in dv.groupby(["pos", "ch"]):
        fr = np.sort(g.frame.unique())
        cand += [(pos, ch, int(a), int(b) - 1) for a, b in zip(fr[:-1], fr[1:]) if b - 1 <= PHASE1_END]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cand))
    lin_cache, picked, used_pos = {}, [], set()
    for k in order:
        pos, ch, f0, f1 = cand[k]
        if pos in used_pos or (pos, ch, f0) == ("Pos18", "ch02", 633):
            continue
        if (pos, ch) not in lin_cache:
            lin_cache[(pos, ch)] = pd.read_csv(mdir / "per_channel" / pos / ch / "lineage_data3D.csv")
        lin = lin_cache[(pos, ch)]
        seg = lin[(lin["rank"] == 1) & lin.frame.between(f0, f1)].sort_values("frame")
        if (len(seg) != f1 - f0 + 1 or seg.frame.nunique() != len(seg) or seg.cell_id.nunique() != 1
                or seg.is_outlier.any() or seg.touches_border.any()):
            continue
        L = seg.long_axis_um.to_numpy()
        if np.any(L[1:-1] < 0.7 * np.minimum(L[:-2], L[2:])):
            continue
        picked.append((pos, ch, f0, f1))
        used_pos.add(pos)
        if len(picked) == n:
            break
    print(f"{len(cand)} candidate cycles; picked {picked}")
    return picked


def main():
    global POS, CH, F0, F1
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default=POS)
    ap.add_argument("--ch", default=CH)
    ap.add_argument("--f0", type=int, default=F0)
    ap.add_argument("--f1", type=int, default=F1)
    ap.add_argument("--random", type=int, default=0, help="draw N gap-free cycles instead")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--axes-only", action="store_true",
                    help="save only the contour + axes strip (PNG) for browsing")
    args = ap.parse_args()
    cycles = (random_cycles(args.random, args.seed) if args.random
              else [(args.pos, args.ch, args.f0, args.f1)])
    for POS, CH, F0, F1 in cycles:
        print(f"== {POS} {CH} {F0}-{F1}")
        run_one(args.axes_only)


def run_one(axes_only: bool):
    paths = master_paths()
    px = float(paths["params"]["pixel_size_um"])
    lin = pd.read_csv(paths["lineage"])
    rows = select_rows(lin)
    # the mother's end of the trap goes to the top (Pos18 ch02: high x)
    trap_w = np.squeeze(tifffile.imread(next(paths["mask_dir"].glob("*_masks.tif")))).shape[1]
    top_high_x = float(rows.centroid_x_px.mean()) > trap_w / 2
    top_end = "high_x" if top_high_x else "low_x"
    tiles, man = build(paths, rows, px, top_high_x)
    canvas = layout(tiles)

    for c in ("long_axis_um", "volume_um3_efd", "short_axis_um"):
        rel = (man[f"{c}_redrawn"] / man[f"{c}_master"] - 1).abs().max()
        print(f"max |redrawn/master - 1| {c}: {rel:.2e}")
    print(man[["frame", "time_from_birth_min", "mask_label", "long_axis_um_redrawn",
               "n_valid_chords", "n_chords", "max_endpoint_dist_px",
               "long_axis_inside_frac"]].to_string(index=False))

    out_dir = Path(__file__).resolve().parents[1] / "results" / "poster_mask_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{POS}_{CH}_{F0}-{F1}"
    man_path = out_dir / f"manifest_{stem}.csv"
    man.to_csv(man_path, index=False)

    data = {"canvas": canvas, "frames": man.frame.to_numpy(),
            "time_from_birth_min": man.time_from_birth_min.to_numpy(),
            "pixel_size_um": px, "vmin": VMIN, "vmax": VMAX}
    for j, t in enumerate(tiles):
        data[f"tile{j:02d}_offset_xy"] = t["off"]
        data[f"tile{j:02d}_contour_xy"] = t["contour"] + t["off"]
        data[f"tile{j:02d}_long_axis_xy"] = t["medial"] + t["off"]
        data[f"tile{j:02d}_chord_p0_xy"] = t["p0"] + t["off"]
        data[f"tile{j:02d}_chord_p1_xy"] = t["p1"] + t["off"]
        data[f"tile{j:02d}_shown_chord_idx"] = t["shown"]

    params = {"dataset": "260517", "master": paths["master"].name, "pos": POS, "ch": CH,
              "frame_min": F0, "frame_max": F1, "n_tiles": len(tiles),
              "frames": man.frame.tolist(), "cmap": CMAP, "vmin": VMIN, "vmax": VMAX,
              "pixel_size_um": px, "efd_k": 6, "contour_offset_px": paths["offset"],
              "rotation": f"exact 90 deg, trap {top_end} end up",
              "crop_margin_px": CROP_MARGIN, "gap_px": GAP,
              "chords_shown": "all valid chords"}
    src = {"phase_dir": str(paths["phase_dir"]), "mask_dir": str(paths["mask_dir"]),
           "lineage_csv": str(paths["lineage"]), "dataset_yaml": str(paths["dataset_yaml"])}
    cond = (f"S. pombe, 260517 mother machine, {POS} trap {CH}, 2% glucose (growth phase), "
            f"5 min/frame; one mother cell (cell_id {int(rows.cell_id.iloc[0])}) from birth "
            f"(img {F0}) to the frame before division (img {F1}); {len(tiles)} of "
            f"{F1 - F0 + 1} frames shown, evenly spaced with first and last kept. "
            "Numbers above tiles: time since birth [min]. Phase: reconstructed phase "
            f"[rad], inferno, fixed {VMIN}-{VMAX} rad for all tiles; tiles rotated by "
            "exactly 90 deg (no interpolation). Scale bar 5 µm "
            f"(pixel {px:.4f} µm).")
    contour_def = ("Cyan: boundary of the Omnipose label mask of this cell, resampled to "
                   "uniform arc length and low-pass filtered to the lowest 6 Fourier "
                   "harmonics (EFD K=6), no inward offset.")
    axes_def = (" Red: long axis = cell centerline after one midpoint update (midpoints "
                "of the perpendicular chords to the smoothed boundary); long_axis_um is its "
                "arc length. White: chords perpendicular to the centerline, ending on the "
                "smoothed boundary; short_axis_um is the mean length of the body chords "
                "(end caps excluded, chords ≥50% of the maximum). Every chord is drawn "
                "(one per centerline sample, spacing = centerline sampling step); "
                "volume_um3_efd sums pi (w/2)^2 ds over all of them. All lines are computed from the masks, "
                "not drawn by hand.")

    if axes_only:
        f2 = render_strip(canvas, tiles, px, with_axes=True)
        f2.savefig(out_dir / f"strip_axes_{stem}.png", dpi=400)
        save_figure(f2, params={**params, "overlay": "contour + long axis + chords"},
                    description=f"Candidate strip {POS} {CH} {F0}-{F1}: contour + long axis + short-axis chords",
                    caption=contour_def + axes_def + " " + cond, data=data, data_source=src,
                    copy_files=[str(man_path)], dpi=400, fmt="png")
        plt.close(f2)
        return

    f1 = render_strip(canvas, tiles, px, with_axes=False)
    save_figure(f1, params={**params, "overlay": "contour"},
                description=f"ICBBS poster strip {POS} {CH} {F0}-{F1}: phase + EFD K=6 contour from master masks",
                caption=contour_def + " " + cond, data=data, data_source=src,
                copy_files=[str(man_path)], dpi=600, fmt="pdf")
    save_figure(f1, params={**params, "overlay": "contour"},
                description=f"ICBBS poster strip {POS} {CH} {F0}-{F1}: phase + EFD K=6 contour (PNG)",
                caption=contour_def + " " + cond, data=data, dpi=600, fmt="png")
    plt.close(f1)

    f2 = render_strip(canvas, tiles, px, with_axes=True)
    save_figure(f2, params={**params, "overlay": "contour + long axis + chords"},
                description=f"ICBBS poster strip {POS} {CH} {F0}-{F1}: contour + long axis + short-axis chords",
                caption=contour_def + axes_def + " " + cond, data=data, data_source=src,
                copy_files=[str(man_path)], dpi=600, fmt="pdf")
    save_figure(f2, params={**params, "overlay": "contour + long axis + chords"},
                description=f"ICBBS poster strip {POS} {CH} {F0}-{F1}: contour + axes (PNG)",
                caption=contour_def + axes_def + " " + cond, data=data, dpi=600, fmt="png")
    plt.close(f2)

    f3 = render_qc(tiles, man)
    save_figure(f3, params={**params, "overlay": "QC"},
                description=f"QC for the {POS} {CH} poster strip: mask pixel edge (white) vs EFD contour, axes",
                caption=("QC of the poster strip. Top: thin white = label-mask pixel edge "
                         "(0.5 iso-line of the binary mask); cyan = EFD K=6 contour. Bottom: "
                         "contour, long axis and displayed chords. Titles: redrawn long axis "
                         "vs master long_axis_um; maximum distance from any chord endpoint to "
                         "the contour; fraction of centerline points inside the contour. "
                         + cond),
                data=data, dpi=400, fmt="png")
    plt.close(f3)
    print(f"manifest: {man_path}")


if __name__ == "__main__":
    main()
