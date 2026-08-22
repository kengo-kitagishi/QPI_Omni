"""bench_cellbias_estimators.py -- does the cell-content bias depend on the estimator?

The production drift loop aligns a timelapse crop (cells present) against the
cell-free grid(0,0) reference. Cells are real content that is absent from the
reference, so the best-match position is pulled away from the true stage offset.
`bench_subpix_methods.py` cannot see this: it shifts the SAME image, so it only
measures estimator noise (4-6 nm). The pull measured on real data is ~0.4 px
(~140 nm) -- two orders larger -- which is why production drops cell-bearing
channels via `ecc_min_corr` instead of relying on a better estimator.

This script measures the pull itself, per estimator.

Design
------
ref  = grid PosN_x+0_y+0 at the ECC z plane   (cell-free)
mov  = timelapse PosN at the same z plane     (cells present)

For every (frame, channel) a known shift s is applied to the *cell-bearing*
crop and each estimator is asked to recover it against the *cell-free*
reference:

    residual(s) = estimate - s

The mean of residual over s is the constant content offset for that channel
(true stage offset + cell pull); its spread is the estimator's precision under
content mismatch. The true stage offset is common to every channel of a Pos, so

    cell_bias(channel) = offset(channel) - median(offset over cell-free channels)

isolates the pull. Channels are classified cell-bearing exactly as production
does: ECC-float correlation against the grid reference < ecc_min_corr.

Everything is read-only and reuses the production blocks (`tilt_fit_crop`,
`ecc_align`) and the estimator/shift-generator implementations from
`bench_subpix_methods.py`. No production file is written.

Usage
-----
    python scripts/bench_cellbias_estimators.py \
        --grid-dir "E:\\260819\\grid_ye_1" --pos 3 \
        --tl-phase-dir "<dir with img_*_ph_006_phase.tif of the timelapse>"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))

# The console here is cp932; a non-ASCII character in a print must not abort a run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from ecc_utils import ECC_MIN_CORR, tilt_fit_crop, to_ecc_input, ecc_align
from bench_subpix_methods import (
    ECC_CROP_H, TILT_CROP_H, SHIFT_GENERATORS,
    build_estimators, calibrate_sign, center_crop_cols,
)
from figure_logger import save_figure

ECC_VMIN, ECC_VMAX = -5.0, 2.0
PIXEL_SCALE_UM = 0.34567514677103717


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid-dir", required=True)
    p.add_argument("--pos", type=int, required=True)
    p.add_argument("--tl-phase-dir", required=True,
                   help="directory holding the timelapse img_*_phase.tif at the ECC z plane")
    p.add_argument("--grid-z", type=int, default=6)
    p.add_argument("--pos-split", type=int, default=51)
    p.add_argument("--n-shifts", type=int, default=10)
    p.add_argument("--max-shift", type=float, default=1.5)
    p.add_argument("--methods", default="ECC-uint8,ECC-float,SG-2D,Gaussian-2D",
                   help="comma-separated subset of build_estimators() keys; empty = all")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def load_rois(grid_dir: Path, pos: int):
    import json
    p = (grid_dir / f"Pos{pos}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json")
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    rng = np.random.default_rng(0)
    grid_dir = Path(args.grid_dir)
    fit_right = args.pos >= args.pos_split

    rois = load_rois(grid_dir, args.pos)
    ref_paths = sorted((grid_dir / f"Pos{args.pos}_x+0_y+0" / "output_phase")
                       .glob(f"img_*_ph_{args.grid_z:03d}_phase.tif"))
    if not ref_paths:
        raise FileNotFoundError(f"no grid reference at z={args.grid_z}")
    mov_paths = sorted(Path(args.tl_phase_dir).glob("img_*_phase.tif"))
    if not mov_paths:
        raise FileNotFoundError(f"no timelapse frames in {args.tl_phase_dir}")

    print(f"Pos{args.pos}  channels={len(rois)}  grid z={args.grid_z}  "
          f"fit_right={fit_right}")
    print(f"reference : {ref_paths[0]}")
    print(f"timelapse : {len(mov_paths)} frames from {args.tl_phase_dir}")

    ref_img = tifffile.imread(str(ref_paths[0])).astype(np.float64)

    estimators = build_estimators()
    if args.methods.strip():
        want = [m.strip() for m in args.methods.split(",")]
        missing = [m for m in want if m not in estimators]
        if missing:
            raise KeyError(f"unknown methods {missing}; have {list(estimators)}")
        estimators = {m: estimators[m] for m in want}
    print(f"methods   : {list(estimators)}")

    # Wide cell-free reference crops, one per channel (None where OOB).
    wide_refs = []
    for roi in rois:
        wide_refs.append(tilt_fit_crop(ref_img, roi["cy"], roi["cx"], roi["crop_w"],
                                       ecc_crop_h=TILT_CROP_H, tilt_crop_h=TILT_CROP_H,
                                       fit_right=fit_right))
    usable = [i for i, w in enumerate(wide_refs) if w is not None]
    if not usable:
        raise RuntimeError("all channels OOB for the wide bench window "
                           f"(needs cx +-{TILT_CROP_H // 2} inside the image)")
    print(f"usable channels for the wide window: {usable}")

    print("\nSign calibration:")
    signs = {name: calibrate_sign(fn, wide_refs[usable[0]].astype(np.float32), name)
             for name, fn in estimators.items()}

    # residuals[method][channel] -> list of [res_y, res_x]
    residuals = {m: {c: [] for c in usable} for m in estimators}
    corrs = {c: [] for c in usable}

    for fpath in mov_paths:
        mov_img = tifffile.imread(str(fpath)).astype(np.float64)
        for c in usable:
            roi = rois[c]
            wide_mov = tilt_fit_crop(mov_img, roi["cy"], roi["cx"], roi["crop_w"],
                                     ecc_crop_h=TILT_CROP_H, tilt_crop_h=TILT_CROP_H,
                                     fit_right=fit_right)
            if wide_mov is None:
                continue
            ref80 = center_crop_cols(wide_refs[c], ECC_CROP_H).astype(np.float32)
            mov80 = center_crop_cols(wide_mov, ECC_CROP_H).astype(np.float32)

            # Production cell/cell-free score: float ECC correlation vs the grid.
            res = ecc_align(to_ecc_input(ref80, ECC_VMIN, ECC_VMAX),
                            to_ecc_input(mov80, ECC_VMIN, ECC_VMAX))
            if res is not None:
                corrs[c].append(res[2])

            shifts = rng.uniform(-args.max_shift, args.max_shift, size=(args.n_shifts, 2))
            for dy, dx in shifts:
                for gfn in SHIFT_GENERATORS.values():
                    m80 = center_crop_cols(gfn(wide_mov, dy, dx), ECC_CROP_H).astype(np.float32)
                    for mname, mfn in estimators.items():
                        est = mfn(ref80, m80)
                        if est is None:
                            continue
                        est = est * signs[mname]
                        residuals[mname][c].append([est[0] - dy, est[1] - dx])

    # ---- Per-channel constant offset, then cell bias vs the cell-free consensus ----
    mean_corr = {c: (float(np.mean(v)) if v else np.nan) for c, v in corrs.items()}
    cellfree = [c for c in usable if mean_corr[c] >= ECC_MIN_CORR]
    cellful = [c for c in usable if c not in cellfree]
    print(f"\nECC-float corr per channel (threshold {ECC_MIN_CORR}):")
    for c in usable:
        tag = "cell-free" if c in cellfree else "CELLS"
        print(f"  ch{c:02d}  corr={mean_corr[c]:.4f}  {tag}")
    print(f"cell-free channels: {cellfree}")
    print(f"cell-bearing      : {cellful}")
    if not cellfree:
        raise RuntimeError("no cell-free channel -> no consensus reference available")

    print(f"\n=== Cell bias (px; positive = pulled +) -- 1 px = {PIXEL_SCALE_UM*1000:.1f} nm ===")
    summary = {}
    for mname in estimators:
        off = {}
        sd = {}
        for c in usable:
            e = np.array(residuals[mname][c])
            if len(e) == 0:
                off[c] = np.array([np.nan, np.nan]); sd[c] = np.array([np.nan, np.nan]); continue
            off[c] = e.mean(axis=0)
            sd[c] = e.std(axis=0)
        base = np.nanmedian(np.array([off[c] for c in cellfree]), axis=0)
        bias = {c: off[c] - base for c in usable}
        bc = np.array([bias[c] for c in cellful]) if cellful else np.zeros((0, 2))
        bf = np.array([bias[c] for c in cellfree])
        summary[mname] = dict(
            bias=bias, offset=off, std=sd, base=base,
            cell_absmean_x=float(np.abs(bc[:, 1]).mean()) if len(bc) else np.nan,
            cell_absmax_x=float(np.abs(bc[:, 1]).max()) if len(bc) else np.nan,
            cell_absmean_y=float(np.abs(bc[:, 0]).mean()) if len(bc) else np.nan,
            free_absmean_x=float(np.abs(bf[:, 1]).mean()),
            prec_x=float(np.nanmean([sd[c][1] for c in usable])),
            prec_y=float(np.nanmean([sd[c][0] for c in usable])),
        )
        s = summary[mname]
        print(f"\n{mname}")
        print(f"  cell-bearing |bias_X| mean={s['cell_absmean_x']:.4f} px "
              f"({s['cell_absmean_x']*PIXEL_SCALE_UM*1000:.0f} nm)  "
              f"max={s['cell_absmax_x']:.4f} px "
              f"({s['cell_absmax_x']*PIXEL_SCALE_UM*1000:.0f} nm)")
        print(f"  cell-free    |bias_X| mean={s['free_absmean_x']:.4f} px "
              f"({s['free_absmean_x']*PIXEL_SCALE_UM*1000:.0f} nm)   <- consensus scatter")
        print(f"  precision (std over shifts): X={s['prec_x']:.4f} px "
              f"({s['prec_x']*PIXEL_SCALE_UM*1000:.1f} nm)  "
              f"Y={s['prec_y']:.4f} px ({s['prec_y']*PIXEL_SCALE_UM*1000:.1f} nm)")
        for c in usable:
            tag = "CELLS" if c in cellful else "free "
            print(f"    ch{c:02d} {tag} bias_X={bias[c][1]:+.4f} px "
                  f"({bias[c][1]*PIXEL_SCALE_UM*1000:+7.1f} nm)  "
                  f"bias_Y={bias[c][0]:+.4f} px")

    # ---- Figure: per-channel bias_X by method, cell vs cell-free ----
    names = list(estimators)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    width = 0.8 / len(names)
    xs = np.arange(len(usable))
    for i, mname in enumerate(names):
        vals = [summary[mname]["bias"][c][1] * PIXEL_SCALE_UM * 1000 for c in usable]
        ax.bar(xs + i * width - 0.4 + width / 2, vals, width, label=mname)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"ch{c}\n{'cell' if c in cellful else 'free'}" for c in usable],
                       fontsize=7)
    ax.set_ylabel("Cell bias, X (nm)")
    ax.set_title("Per-channel bias vs cell-free consensus")
    ax.legend(fontsize=7)

    ax = axes[1]
    m_abs = [summary[m]["cell_absmean_x"] * PIXEL_SCALE_UM * 1000 for m in names]
    m_prec = [summary[m]["prec_x"] * PIXEL_SCALE_UM * 1000 for m in names]
    xs2 = np.arange(len(names))
    ax.bar(xs2 - 0.2, m_abs, 0.4, label="cell bias |X| (mean)")
    ax.bar(xs2 + 0.2, m_prec, 0.4, label="precision std X")
    ax.set_xticks(xs2)
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("nm")
    ax.set_title("Cell bias vs estimator noise")
    ax.legend(fontsize=7)
    fig.suptitle(f"Cell-content bias by estimator -- Pos{args.pos}, "
                 f"grid z={args.grid_z}, {len(mov_paths)} frames", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_figure(
        fig,
        params={"pos": args.pos, "grid_z": args.grid_z, "n_frames": len(mov_paths),
                "n_shifts": args.n_shifts, "max_shift": args.max_shift,
                "methods": names, "ecc_min_corr": ECC_MIN_CORR,
                "cellfree_channels": cellfree, "cell_channels": cellful,
                "grid_dir": str(grid_dir), "tl_phase_dir": args.tl_phase_dir},
        description=("Cell-content bias per estimator: known shifts applied to a "
                     "cell-bearing timelapse crop, aligned to the cell-free grid "
                     "reference; bias = per-channel constant offset minus the "
                     "cell-free consensus."),
        data={"channels": np.array(usable),
              "mean_corr": np.array([mean_corr[c] for c in usable]),
              **{f"bias_x_{m}": np.array([summary[m]["bias"][c][1] for c in usable])
                 for m in names},
              **{f"bias_y_{m}": np.array([summary[m]["bias"][c][0] for c in usable])
                 for m in names},
              **{f"prec_x_{m}": np.array([summary[m]["std"][c][1] for c in usable])
                 for m in names}},
    )
    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
