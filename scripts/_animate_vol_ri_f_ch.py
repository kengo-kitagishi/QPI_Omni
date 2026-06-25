"""Animate volume / mean-RI traces for f:/ch02 and f:/ch07 (frames 288-1152) at 24 fps.

Builds per-frame summary from <ch>/inference_out/*_masks.tif + phase tifs using
the existing pipeline helpers, then writes a 2-panel MP4 where each curve is
revealed progressively (one animation frame per data frame).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import imageio_ffmpeg
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

from central_cell_track_figures import build_frame_pairs, build_summary_table  # noqa: E402

FRAME_MIN, FRAME_MAX = 288, 1152
FPS = 24

JOBS = [
    {"label": "ch02", "dir": Path(r"f:/ch02"), "color": "#0072B2"},  # Okabe-Ito blue
]
OUT_PATH = Path(r"f:/ch02_vol_ri_animation_288-1152_24fps.mp4")

SUMMARY_KW = dict(
    min_area=20,
    exclude_border=True,
    pixel_size_um=0.348,
    wavelength_nm=663.0,
    n_medium=1.333,
    alpha_ri=0.00018,
)


def load_series(d: Path) -> pd.DataFrame:
    pairs = build_frame_pairs(d)
    if not pairs:
        raise RuntimeError(f"no frame pairs at {d}")
    df = build_summary_table(pairs, **SUMMARY_KW)
    df = df[(df["frame_index"] >= FRAME_MIN) & (df["frame_index"] <= FRAME_MAX)].copy()
    df = df.sort_values("frame_index").reset_index(drop=True)
    return df


def main() -> int:
    series = []
    for job in JOBS:
        print(f"[anim] loading {job['label']} from {job['dir']}", flush=True)
        df = load_series(job["dir"])
        print(f"  -> {len(df)} rows; tracked={int(df['tracked'].sum())}", flush=True)
        series.append({**job, "df": df})

    fig, (ax_v, ax_ri) = plt.subplots(2, 1, figsize=(7.5, 5.4), constrained_layout=True, sharex=True)
    all_frames = sorted({fi for s in series for fi in s["df"]["frame_index"].astype(int).tolist()})
    frames_used = [fi for fi in all_frames if FRAME_MIN <= fi <= FRAME_MAX]
    print(f"[anim] animation will span {len(frames_used)} frames "
          f"[{frames_used[0]}, {frames_used[-1]}]", flush=True)

    vol_lim_hi = float(np.nanmax([s["df"]["volume_um3_rod"].max() for s in series]))
    ri_vals = np.concatenate([s["df"]["mean_ri"].to_numpy(float) for s in series])
    ri_lo = float(np.nanmin(ri_vals)) if np.isfinite(ri_vals).any() else 1.34
    ri_hi = float(np.nanmax(ri_vals)) if np.isfinite(ri_vals).any() else 1.40

    ax_v.set_xlim(FRAME_MIN, FRAME_MAX)
    ax_v.set_ylim(0.0, vol_lim_hi * 1.05 if np.isfinite(vol_lim_hi) and vol_lim_hi > 0 else 100)
    ax_v.set_ylabel("Volume [um^3]")
    ax_v.set_title("A  Rod volume", loc="left")
    ax_v.grid(True, alpha=0.3, linestyle="--")
    ax_v.spines["top"].set_visible(False)
    ax_v.spines["right"].set_visible(False)

    ax_ri.set_xlim(FRAME_MIN, FRAME_MAX)
    pad = max(0.001, (ri_hi - ri_lo) * 0.1) if ri_hi > ri_lo else 0.01
    ax_ri.set_ylim(ri_lo - pad, ri_hi + pad)
    ax_ri.set_ylabel("Mean RI")
    ax_ri.set_xlabel("Frame")
    ax_ri.set_title("B  Mean RI", loc="left")
    ax_ri.grid(True, alpha=0.3, linestyle="--")
    ax_ri.spines["top"].set_visible(False)
    ax_ri.spines["right"].set_visible(False)

    lines_v = []
    lines_ri = []
    markers_v = []
    markers_ri = []
    for s in series:
        lv, = ax_v.plot([], [], color=s["color"], lw=1.2, label=s["label"])
        lri, = ax_ri.plot([], [], color=s["color"], lw=1.2, label=s["label"])
        mv, = ax_v.plot([], [], "o", color=s["color"], ms=4)
        mri, = ax_ri.plot([], [], "o", color=s["color"], ms=4)
        lines_v.append(lv)
        lines_ri.append(lri)
        markers_v.append(mv)
        markers_ri.append(mri)
    ax_v.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.92)
    frame_text = ax_v.text(0.99, 0.95, "", transform=ax_v.transAxes,
                           ha="right", va="top", fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))

    def update(fi: int):
        for s, lv, lri, mv, mri in zip(series, lines_v, lines_ri, markers_v, markers_ri):
            df = s["df"]
            sub = df[df["frame_index"] <= fi]
            x = sub["frame_index"].to_numpy(float)
            yv = sub["volume_um3_rod"].to_numpy(float)
            yri = sub["mean_ri"].to_numpy(float)
            lv.set_data(x, yv)
            lri.set_data(x, yri)
            cur = df[df["frame_index"] == fi]
            if not cur.empty and np.isfinite(cur["volume_um3_rod"].iloc[0]):
                mv.set_data([fi], [cur["volume_um3_rod"].iloc[0]])
            else:
                mv.set_data([], [])
            if not cur.empty and np.isfinite(cur["mean_ri"].iloc[0]):
                mri.set_data([fi], [cur["mean_ri"].iloc[0]])
            else:
                mri.set_data([], [])
        frame_text.set_text(f"frame {fi}")
        return (*lines_v, *lines_ri, *markers_v, *markers_ri, frame_text)

    writer = manimation.FFMpegWriter(
        fps=FPS, codec="libx264", bitrate=-1,
        extra_args=[
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-crf", "20",
            "-preset", "medium",
        ],
    )
    print(f"[anim] writing {OUT_PATH} at {FPS} fps ({len(frames_used)} frames)", flush=True)

    anim = manimation.FuncAnimation(fig, update, frames=frames_used,
                                    interval=1000.0 / FPS, blit=False)
    anim.save(str(OUT_PATH), writer=writer, dpi=150,
              progress_callback=lambda i, n: None if i % 100 else
              print(f"  [anim] {i}/{n}", flush=True))
    plt.close(fig)
    print(f"[anim] done. {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
