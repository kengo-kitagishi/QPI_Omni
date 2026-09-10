"""validate_minor_axis_volume_smoothing.py — answer three validity questions.

On the pilot mother (default Pos27_ch06) this quantifies, per phase1 frame:

  Q1  Is the corrected minor (short) axis physically reasonable?
      -> compare medial-axis short axis to the S. pombe literature width band
         (~3.5-4.0 um) and to the biased skimage value, cycle-aligned. A
         constant width across the cycle is the expected biology (tip growth at
         fixed diameter), so within-cycle flatness is the test.

  Q2  Is the rod-shape volume from 2 axes trustworthy?
      -> compare the 2-axis rod formula (cyl + 2 hemispherical caps) with the
         solid-of-revolution volume integrated from the full width profile.
         Agreement means the rod approximation is safe; a gap means the profile
         volume should be preferred.

  Q3  Is mask smoothing (Morphometrics-style) needed?
      -> repeat the short-axis / volume measurement on raw vs smoothed masks
         and report the level shift and the frame-to-frame noise reduction.

Usage:
    python scripts/validate_minor_axis_volume_smoothing.py --pos Pos27 --ch ch06
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from qpi_paths import find_lineage_csv, find_clist_csv, mask_dir_for_channel  # noqa: E402
from recompute_axes_from_masks import (  # noqa: E402
    build_mask_index, _load_label_image, _label_at_centroid,
)
from mask_morphology import (  # noqa: E402
    measure_single_cell_medial, smooth_mask,
)
from central_cell_lineage_tracker import calc_rod_volume_um3  # noqa: E402
from gold_standard import rank1_division_frames  # noqa: E402
from figure_logger import save_figure  # noqa: E402

PHASE1_END_FRAME = 2018
N_REL = 100
LIT_WIDTH_MIN, LIT_WIDTH_MAX = 3.5, 4.0  # S. pombe diameter literature band [um]


def measure_frame(binary, px):
    m = measure_single_cell_medial(binary, pixel_size_um=px)
    rod_vol = calc_rod_volume_um3(m.long_axis_px, m.short_axis_px, px)
    prof_vol = m.volume_profile_px3 * px ** 3
    return m.short_axis_px * px, m.long_axis_px * px, rod_vol, prof_vol


def collect(pos, ch):
    csv_path = find_lineage_csv(pos, ch)
    inf_dir = mask_dir_for_channel(csv_path)
    df = pd.read_csv(csv_path)
    clist_path = find_clist_csv(csv_path)
    clist = pd.read_csv(clist_path) if clist_path else None
    divs = rank1_division_frames(df, clist)
    m = df[(df["rank"] == 1) & (df["frame"] <= PHASE1_END_FRAME)].sort_values("frame")
    bad = (m["is_outlier"].astype(bool) | m["touches_border"].astype(bool)
           | (m["mass_pg"] < 10.0))
    bad_frames = set(m["frame"][bad].astype(int))
    px = 0.34567514677103717
    idx = build_mask_index(inf_dir)

    rows = []
    for _, r in m.iterrows():
        frame = int(r["frame"])
        if frame in bad_frames:
            continue
        mp = idx.get(frame)
        if mp is None:
            continue
        mask = _load_label_image(mp)
        lbl = _label_at_centroid(mask, r["centroid_y_px"], r["centroid_x_px"])
        if lbl == 0:
            continue
        binary = mask == lbl
        sm = smooth_mask(binary, closing_radius=1, opening_radius=1, gaussian_sigma=1.0)
        s_raw, L_raw, rodv_raw, profv_raw = measure_frame(binary, px)
        s_sm, L_sm, rodv_sm, profv_sm = measure_frame(sm, px)
        rows.append({
            "frame": frame,
            "short_raw": s_raw, "short_sm": s_sm,
            "short_skimage": float(r["short_axis_um"]),
            "rodv_raw": rodv_raw, "profv_raw": profv_raw,
            "rodv_sm": rodv_sm, "profv_sm": profv_sm,
            "vol_skimage": float(r["volume_um3_rod"]),
        })
    return pd.DataFrame(rows), divs, px


def cycle_align(df, divs, col):
    rel = np.linspace(0, 1, N_REL)
    stacks = []
    f = df["frame"].to_numpy()
    y = df[col].to_numpy()
    for i in range(len(divs) - 1):
        f0, f1 = divs[i], divs[i + 1]
        if f1 > PHASE1_END_FRAME:
            continue
        sel = (f >= f0) & (f < f1)
        if sel.sum() < 4:
            continue
        t_rel = (f[sel] - f0) / (f1 - f0)
        stacks.append(np.interp(rel, t_rel, y[sel]))
    if not stacks:
        return rel, None, None
    M = np.vstack(stacks)
    return rel, np.nanmean(M, axis=0), np.nanstd(M, axis=0)


def pct_var(mean):
    return 100 * (mean.max() - mean.min()) / mean.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="Pos27")
    ap.add_argument("--ch", default="ch06")
    args = ap.parse_args()
    df, divs, px = collect(args.pos, args.ch)
    print(f"{args.pos}_{args.ch}: {len(df)} valid phase1 frames")

    fig, axes = plt.subplots(3, 1, figsize=(130 / 25.4, 180 / 25.4),
                             constrained_layout=True)

    # --- Q1: short axis cycle-aligned vs literature band ---
    ax = axes[0]
    ax.axhspan(LIT_WIDTH_MIN, LIT_WIDTH_MAX, color="#999999", alpha=0.18,
               label=f"S. pombe lit. width {LIT_WIDTH_MIN}-{LIT_WIDTH_MAX} μm")
    for col, color, lbl in [("short_skimage", "#0072B2", "skimage (biased)"),
                            ("short_raw", "#D55E00", "medial raw"),
                            ("short_sm", "#009E73", "medial smoothed")]:
        rel, mean, sd = cycle_align(df, divs, col)
        if mean is None:
            continue
        ax.plot(rel, mean, color=color, lw=1.3,
                label=f"{lbl}  μ={mean.mean():.2f}μm Δ={pct_var(mean):.1f}%")
    ax.set_ylabel("short axis [μm]", fontsize=8)
    ax.set_ylim(3.0, 4.8)
    ax.legend(loc="upper left", frameon=False, fontsize=5.6)
    ax.set_title("Q1: minor-axis validity (flat ≈ tip growth at constant width)",
                 fontsize=7.5)
    ax.tick_params(labelsize=7)

    # --- Q2: rod-formula vs profile-integrated volume (raw) ---
    ax = axes[1]
    for col, color, lbl in [("vol_skimage", "#0072B2", "skimage rod vol (biased)"),
                            ("rodv_raw", "#D55E00", "rod formula (medial)"),
                            ("profv_raw", "#1a1a1a", "profile-integrated vol")]:
        rel, mean, sd = cycle_align(df, divs, col)
        if mean is None:
            continue
        ax.plot(rel, mean, color=color, lw=1.3,
                label=f"{lbl}  μ={mean.mean():.1f}μm³")
    ax.set_ylabel(r"volume [μm$^3$]", fontsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=5.6)
    ax.set_title("Q2: rod-formula vs solid-of-revolution volume (raw mask)",
                 fontsize=7.5)
    ax.tick_params(labelsize=7)

    # --- Q3: raw vs smoothed level + noise ---
    ax = axes[2]
    d_short = (df["short_sm"] - df["short_raw"])
    d_vol = (df["profv_sm"] - df["profv_raw"])
    # frame-to-frame noise (std of successive differences) as a smoothing metric
    noise_raw = np.nanstd(np.diff(df["short_raw"].to_numpy()))
    noise_sm = np.nanstd(np.diff(df["short_sm"].to_numpy()))
    ax.scatter(df["short_raw"], df["short_sm"], s=3, color="#009E73", alpha=0.4)
    lo, hi = 3.2, 4.4
    ax.plot([lo, hi], [lo, hi], color="gray", lw=0.6, ls="--")
    ax.set_xlabel("medial short axis raw [μm]", fontsize=8)
    ax.set_ylabel("medial short axis smoothed [μm]", fontsize=8)
    ax.set_title(f"Q3: smoothing shifts width {d_short.mean():+.3f}μm; "
                 f"frame-noise {noise_raw:.3f}→{noise_sm:.3f}μm", fontsize=7.5)
    ax.tick_params(labelsize=7)

    fig.suptitle(f"{args.pos}_{args.ch} minor-axis / volume / smoothing validity",
                 fontsize=8.5)
    save_figure(
        fig,
        params={"pos": args.pos, "ch": args.ch, "n_frames": len(df),
                "lit_width_band_um": f"{LIT_WIDTH_MIN}-{LIT_WIDTH_MAX}",
                "smoothing": "closing1+opening1+gaussian1"},
        description=f"{args.pos}_{args.ch} minor-axis/volume/smoothing validity",
        data={
            "frame": df["frame"].to_numpy(),
            "short_raw": df["short_raw"].to_numpy(),
            "short_sm": df["short_sm"].to_numpy(),
            "rodv_raw": df["rodv_raw"].to_numpy(),
            "profv_raw": df["profv_raw"].to_numpy(),
        },
    )
    plt.close(fig)

    # console summary
    print("\n--- Q1 short axis (cycle mean) ---")
    for col, lbl in [("short_skimage", "skimage"), ("short_raw", "medial raw"),
                     ("short_sm", "medial smoothed")]:
        _, mean, _ = cycle_align(df, divs, col)
        print(f"  {lbl:16s} mean={mean.mean():.3f}μm  within-cycle Δ={pct_var(mean):.2f}%")
    print("\n--- Q2 volume (frame mean) ---")
    for col, lbl in [("vol_skimage", "skimage rod"), ("rodv_raw", "rod formula medial"),
                     ("profv_raw", "profile integrated")]:
        print(f"  {lbl:20s} mean={df[col].mean():.2f}μm³")
    rel_gap = 100 * (df["profv_raw"] - df["rodv_raw"]).abs().mean() / df["rodv_raw"].mean()
    print(f"  rod-formula vs profile: mean |gap| = {rel_gap:.2f}%")
    print("\n--- Q3 smoothing ---")
    print(f"  short axis shift raw->smooth: {(df['short_sm']-df['short_raw']).mean():+.4f}μm")
    print(f"  frame-to-frame noise raw={np.nanstd(np.diff(df['short_raw'])):.4f} "
          f"smooth={np.nanstd(np.diff(df['short_sm'])):.4f}μm")


if __name__ == "__main__":
    main()
