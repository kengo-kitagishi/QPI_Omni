"""
plot_drift_summary.py
---------------------
1コマンドで drift 解析に必要な図を全て生成する。

生成される図 (各々 figure_logger 経由で保存):
  1. drift_overview        : 累積ドリフト / ECC相関 / ステップ補正 (3行)
  2. drift_trajectory_2d   : 2D軌跡 (色 = 時刻)
  3. drift_grid_proximity  : nearest grid距離 + grid index選択 (2行)
  4. drift_kf_analysis     : raw vs KF / innovation / gain (2×2)
  5. drift_center_profiles : centerプロファイル重ね (--profiles-dir 指定時のみ)

Usage:
    python scripts/plot_drift_summary.py
    python scripts/plot_drift_summary.py --profiles-dir C:\\ph_260327\\Pos1\\output_phase
    python scripts/plot_drift_summary.py --log drift_session/drift_log_20260330T184413.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

# ── Publication style ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "Arial",
    "font.size":        10,
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# ── Defaults ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG    = REPO_ROOT / "drift_session" / "drift_log.json"
DEFAULT_CONFIG = REPO_ROOT / "drift_session" / "drift_config.json"
INTERVAL_SEC_FALLBACK = 300   # 1 frame = 5 min

# ── Data loading ──────────────────────────────────────────────────────────────

def load_log(path: Path):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected list in {path}")
    if len(records) == 0:
        raise ValueError(f"drift_log is empty: {path}\n"
                         "Use --log to specify a log file with data, e.g.:\n"
                         "  python scripts/plot_drift_summary.py --log drift_session/drift_log_YYYYMMDD.json")
    print(f"[load_log] {len(records)} timepoints from {path.name}")
    return records


def load_config(path: Path) -> dict:
    if not path.exists():
        print(f"[load_config] {path.name} not found, using defaults")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _arr(records, key, default=np.nan) -> np.ndarray:
    """Extract a scalar field from every record as float array."""
    return np.array([r.get(key, default) for r in records], dtype=float)


def time_hours(records, interval_sec: float) -> np.ndarray:
    tp = np.array([r["timepoint"] for r in records], dtype=float)
    return tp * interval_sec / 3600.0


# ── Figure 1: Overview ────────────────────────────────────────────────────────

def plot_overview(records, time_h):
    cum_x = _arr(records, "cumulative_dx_um")
    cum_y = _arr(records, "cumulative_dy_um")
    ecc   = _arr(records, "ecc_correlation")
    cor_x = _arr(records, "correction_stage_x_um")
    cor_y = _arr(records, "correction_stage_y_um")
    jumps = np.array([r.get("jump_detected", False)   for r in records], dtype=bool)
    inv   = np.array([not r.get("correction_valid", True) for r in records], dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True,
                              gridspec_kw={"hspace": 0.08})
    fig.suptitle("Drift Correction Overview", fontsize=11, fontweight="bold")

    # [A] Cumulative drift
    ax = axes[0]
    ax.plot(time_h, cum_x, color="#2196F3", lw=1.2, label="X (stage)")
    ax.plot(time_h, cum_y, color="#F44336", lw=1.2, label="Y (stage)")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Cumulative correction (μm)")
    ax.legend(loc="best")
    ax.set_title(f"Total: ΔX={cum_x[-1]:.2f} μm, ΔY={cum_y[-1]:.2f} μm",
                 fontsize=9, loc="right", pad=2)

    # [B] ECC correlation
    ax = axes[1]
    ax.plot(time_h, ecc, color="#555", lw=1.0)
    ax.axhline(0.95, color="orange", lw=0.9, ls="--", label="0.95 threshold")
    ecc_min = float(np.nanmin(ecc))
    ax.set_ylim(max(0.80, ecc_min - 0.01), 1.005)
    ax.set_ylabel("ECC correlation")
    ax.legend(loc="best")
    ax.set_title(f"mean={np.nanmean(ecc):.4f}  min={ecc_min:.4f}",
                 fontsize=9, loc="right", pad=2)

    # [C] Step corrections
    ax = axes[2]
    ax.plot(time_h, cor_x, color="#2196F3", lw=0.8, alpha=0.75, label="X step")
    ax.plot(time_h, cor_y, color="#F44336", lw=0.8, alpha=0.75, label="Y step")
    if jumps.any():
        ax.scatter(time_h[jumps], cor_x[jumps],
                   marker="x", color="orange", s=40, zorder=5, label=f"jump (n={jumps.sum()})")
    if inv.any():
        ax.scatter(time_h[inv], cor_x[inv],
                   marker="o", color="black", s=25, facecolors="none", zorder=5,
                   label=f"invalid (n={inv.sum()})")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Step correction (μm)")
    ax.set_xlabel("Time (h)")
    ax.legend(loc="best")

    return fig, {
        "time_h":           time_h,
        "cumulative_dx_um": cum_x,
        "cumulative_dy_um": cum_y,
        "ecc_correlation":  ecc,
        "correction_x_um":  cor_x,
        "correction_y_um":  cor_y,
    }


# ── Figure 2: 2D Trajectory ───────────────────────────────────────────────────

def plot_trajectory_2d(records, time_h):
    cum_x = _arr(records, "cumulative_dx_um")
    cum_y = _arr(records, "cumulative_dy_um")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.suptitle("2D Drift Trajectory", fontsize=11, fontweight="bold")

    ax.plot(cum_x, cum_y, color="gray", lw=0.5, alpha=0.35, zorder=1)
    sc = ax.scatter(cum_x, cum_y, c=time_h, cmap="plasma", s=10, zorder=2,
                    linewidths=0, vmin=0, vmax=time_h.max())
    fig.colorbar(sc, ax=ax, label="Time (h)", shrink=0.85)
    ax.scatter(cum_x[0], cum_y[0],
               marker="o", s=70, color="lime", edgecolors="k", lw=0.8, zorder=3, label="Start")
    ax.scatter(cum_x[-1], cum_y[-1],
               marker="*", s=120, color="red", edgecolors="k", lw=0.8, zorder=3, label="End")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.axvline(0, color="k", lw=0.4, ls="--")
    ax.set_xlabel("Stage X correction (μm)")
    ax.set_ylabel("Stage Y correction (μm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    fig.tight_layout()

    return fig, {
        "time_h":           time_h,
        "cumulative_dx_um": cum_x,
        "cumulative_dy_um": cum_y,
    }


# ── Figure 3: Grid Proximity ──────────────────────────────────────────────────

def _extract_grid_proximity(records, px_scale_nm):
    dists, xis, yis = [], [], []
    for r in records:
        cds = r.get("channel_details", [])
        valid = [c for c in cds
                 if not c.get("outlier", False) and "tx2" in c and "ty2" in c]
        if valid:
            d_px = float(np.median([np.sqrt(c["tx2"]**2 + c["ty2"]**2) for c in valid]))
            dists.append(d_px * px_scale_nm)
            xis.append(float(np.median([c.get("xi", np.nan) for c in valid])))
            yis.append(float(np.median([c.get("yi", np.nan) for c in valid])))
        else:
            dists.append(np.nan)
            xis.append(np.nan)
            yis.append(np.nan)
    return np.array(dists), np.array(xis), np.array(yis)


def plot_grid_proximity(records, time_h, px_scale_nm):
    has_details = any(
        "channel_details" in r and r["channel_details"]
        and "tx2" in r["channel_details"][0]
        for r in records
    )
    if not has_details:
        print("[skip] drift_grid_proximity: no channel_details with tx2/ty2 in log")
        return None, {}

    dists, xis, yis = _extract_grid_proximity(records, px_scale_nm)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                              gridspec_kw={"hspace": 0.08})
    fig.suptitle("Grid Position Proximity", fontsize=11, fontweight="bold")

    ax = axes[0]
    ax.plot(time_h, dists, color="#4CAF50", lw=1.0)
    ax.set_ylabel("Distance to nearest grid (nm)")
    ax.set_ylim(bottom=0)
    d_med = float(np.nanmedian(dists))
    ax.set_title(f"median = {d_med:.1f} nm", fontsize=9, loc="right", pad=2)

    ax = axes[1]
    ax.step(time_h, xis, where="mid", color="#2196F3", lw=1.0, label="xi")
    ax.step(time_h, yis, where="mid", color="#F44336", lw=1.0, label="yi")
    ax.set_ylabel("Grid index")
    ax.set_xlabel("Time (h)")
    ax.legend(loc="best")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[-1].set_xlabel("Time (h)")

    return fig, {
        "time_h":      time_h,
        "grid_dist_nm": dists,
        "grid_xi":     xis,
        "grid_yi":     yis,
    }


# ── Figure 4: Kalman Analysis ─────────────────────────────────────────────────

def plot_kf_analysis(records, time_h, px_scale_nm):
    raw_tx_nm = _arr(records, "tx_avg_px") * px_scale_nm
    raw_ty_nm = _arr(records, "ty_avg_px") * px_scale_nm

    has_filt  = any("tx_filt_px" in r for r in records)
    has_innov = any("kf_innovation_tx_nm" in r for r in records)
    has_gain  = any("kf_K_tx" in r for r in records)

    if not has_filt and not has_innov and not has_gain:
        print("[skip] drift_kf_analysis: no KF fields in log")
        return None, {}

    filt_tx_nm = _arr(records, "tx_filt_px") * px_scale_nm if has_filt else None
    filt_ty_nm = _arr(records, "ty_filt_px") * px_scale_nm if has_filt else None
    innov_tx   = _arr(records, "kf_innovation_tx_nm") if has_innov else None
    innov_ty   = _arr(records, "kf_innovation_ty_nm") if has_innov else None
    kf_K       = _arr(records, "kf_K_tx") if has_gain else None
    kf_P_sigma = np.sqrt(np.abs(_arr(records, "kf_P_tx_nm2"))) if has_gain else None

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True,
                              gridspec_kw={"hspace": 0.1, "wspace": 0.3})
    fig.suptitle("Kalman Filter Analysis", fontsize=11, fontweight="bold")

    # [A] X: raw vs KF
    ax = axes[0, 0]
    ax.plot(time_h, raw_tx_nm, color="gray", lw=0.7, alpha=0.65, label="Raw ECC")
    if filt_tx_nm is not None:
        ax.plot(time_h, filt_tx_nm, color="#2196F3", lw=1.2, label="KF filtered")
    ax.set_ylabel("Shift X (nm)")
    ax.legend(loc="best")
    ax.set_title("X: raw vs KF", fontsize=9)
    if filt_tx_nm is not None:
        noise_x = float(np.nanstd(raw_tx_nm - filt_tx_nm))
        ax.set_title(f"X  |  residual std = {noise_x:.1f} nm", fontsize=9, loc="right", pad=2)

    # [B] Y: raw vs KF
    ax = axes[0, 1]
    ax.plot(time_h, raw_ty_nm, color="gray", lw=0.7, alpha=0.65, label="Raw ECC")
    if filt_ty_nm is not None:
        ax.plot(time_h, filt_ty_nm, color="#F44336", lw=1.2, label="KF filtered")
    ax.set_ylabel("Shift Y (nm)")
    ax.legend(loc="best")
    ax.set_title("Y: raw vs KF", fontsize=9)
    if filt_ty_nm is not None:
        noise_y = float(np.nanstd(raw_ty_nm - filt_ty_nm))
        ax.set_title(f"Y  |  residual std = {noise_y:.1f} nm", fontsize=9, loc="right", pad=2)

    # [C] Innovation
    ax = axes[1, 0]
    if innov_tx is not None:
        ax.plot(time_h, innov_tx, color="#2196F3", lw=0.8, alpha=0.8, label="X")
    if innov_ty is not None:
        ax.plot(time_h, innov_ty, color="#F44336", lw=0.8, alpha=0.8, label="Y")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Innovation (nm)")
    ax.set_xlabel("Time (h)")
    ax.legend(loc="best")
    ax.set_title("KF Innovation", fontsize=9)
    if innov_tx is not None:
        ax.set_title(
            f"Innovation  |  X std = {float(np.nanstd(innov_tx)):.1f} nm",
            fontsize=9, loc="right", pad=2)

    # [D] Gain + uncertainty
    ax = axes[1, 1]
    if kf_K is not None:
        ax.plot(time_h, kf_K, color="#9C27B0", lw=1.0, label="Gain K")
        ax.set_ylabel("Kalman gain K", color="#9C27B0")
    if kf_P_sigma is not None:
        ax2 = ax.twinx()
        ax2.plot(time_h, kf_P_sigma, color="#FF9800", lw=0.9, ls="--", label="σ_P (nm)")
        ax2.set_ylabel("σ_P (nm)", color="#FF9800", fontsize=9)
        ax2.spines["right"].set_visible(True)
        # Combine legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best")
    else:
        ax.legend(loc="best")
    ax.set_xlabel("Time (h)")
    ax.set_title("KF Gain & Uncertainty (X)", fontsize=9)

    data = {
        "time_h":     time_h,
        "raw_tx_nm":  raw_tx_nm,
        "raw_ty_nm":  raw_ty_nm,
    }
    if filt_tx_nm is not None:
        data["filt_tx_nm"] = filt_tx_nm
        data["filt_ty_nm"] = filt_ty_nm
    if innov_tx is not None:
        data["innov_tx_nm"] = innov_tx
        data["innov_ty_nm"] = innov_ty
    if kf_K is not None:
        data["kf_gain"]    = kf_K
        data["kf_sigma_P"] = kf_P_sigma
    return fig, data


# ── Figure 5: Center Profiles ─────────────────────────────────────────────────

def _load_center_profiles(profiles_dir: Path, n_profiles: int, ch):
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile not installed. Run: pip install tifffile")

    tifs = sorted(profiles_dir.glob("img_*_phase.tif"))
    if not tifs:
        tifs = sorted(profiles_dir.glob("img_*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No TIF files found in {profiles_dir}")

    total = len(tifs)
    indices = np.round(np.linspace(0, total - 1, min(n_profiles, total))).astype(int)

    profiles = []
    frame_indices = []
    for idx in indices:
        img = tifffile.imread(str(tifs[idx])).astype(float)
        if img.ndim == 3:
            pick = ch if ch is not None else img.shape[0] // 2
            img = img[pick]
        cy = img.shape[0] // 2
        profiles.append(img[cy, :])
        frame_indices.append(int(idx))

    return np.array(profiles), np.array(frame_indices), total


def plot_center_profiles(profiles_dir: Path, n_profiles: int, ch, interval_sec: float):
    try:
        profiles, frame_indices, total = _load_center_profiles(profiles_dir, n_profiles, ch)
    except (FileNotFoundError, ImportError) as e:
        print(f"[skip] drift_center_profiles: {e}")
        return None, {}

    W       = profiles.shape[1]
    x_px    = np.arange(W)
    time_h  = frame_indices * interval_sec / 3600.0
    t_max   = float(time_h.max()) if time_h.max() > 0 else 1.0
    cmap    = cm.viridis
    norm    = plt.Normalize(vmin=0, vmax=t_max)

    fig, ax = plt.subplots(figsize=(9, 4))
    ch_label = str(ch) if ch is not None else "mid"
    fig.suptitle(
        f"Center-line Phase Profiles  (ch={ch_label}, {len(profiles)}/{total} frames)",
        fontsize=11, fontweight="bold")

    for prof, t in zip(profiles, time_h):
        ax.plot(x_px, prof, color=cmap(norm(t)), lw=0.8, alpha=0.75)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Time (h)")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("Phase (rad)")
    fig.tight_layout()

    return fig, {
        "x_px":          x_px,
        "profiles":      profiles,
        "frame_indices": frame_indices,
        "time_h":        time_h,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Unified drift analysis figure generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--log",          default=str(DEFAULT_LOG),
                    help="drift_log.json path")
    ap.add_argument("--config",       default=str(DEFAULT_CONFIG),
                    help="drift_config.json path")
    ap.add_argument("--profiles-dir", default=None,
                    help="Directory with phase TIF files for center-line profiles")
    ap.add_argument("--ch",           type=int, default=None,
                    help="Channel index for center profiles (default: mid)")
    ap.add_argument("--n-profiles",   type=int, default=20,
                    help="Number of profiles to sample")
    return ap.parse_args()


def main():
    args    = parse_args()
    log_path = Path(args.log)
    cfg_path = Path(args.config)

    records      = load_log(log_path)
    config       = load_config(cfg_path)
    px_scale_um  = float(config.get("pixel_scale_um", 0.34567514677103717))
    px_scale_nm  = px_scale_um * 1000.0
    interval_sec = float(config.get("interval_sec", INTERVAL_SEC_FALLBACK))

    time_h = time_hours(records, interval_sec)
    N = len(records)
    print(f"  {N} timepoints | {time_h[-1]:.2f} h | {px_scale_um:.5f} μm/px")

    # ── Fig 1 ──
    print("\n[1/5] Overview...")
    fig, data = plot_overview(records, time_h)
    save_figure(fig,
                params={"n_timepoints": N, "log": log_path.name},
                description="Drift correction overview: cumulative drift, ECC correlation, step corrections",
                data=data)
    plt.close(fig)

    # ── Fig 2 ──
    print("[2/5] 2D trajectory...")
    fig, data = plot_trajectory_2d(records, time_h)
    save_figure(fig,
                params={"n_timepoints": N},
                description="2D stage drift trajectory colored by time",
                data=data)
    plt.close(fig)

    # ── Fig 3 ──
    print("[3/5] Grid proximity...")
    fig, data = plot_grid_proximity(records, time_h, px_scale_nm)
    if fig is not None:
        save_figure(fig,
                    params={"px_scale_nm": round(px_scale_nm, 2)},
                    description="Nearest-grid distance and grid index selection over time",
                    data=data)
        plt.close(fig)

    # ── Fig 4 ──
    print("[4/5] Kalman filter analysis...")
    fig, data = plot_kf_analysis(records, time_h, px_scale_nm)
    if fig is not None:
        save_figure(fig,
                    params={"px_scale_nm": round(px_scale_nm, 2)},
                    description="Kalman filter: raw ECC vs filtered shift, innovation, gain, uncertainty",
                    data=data)
        plt.close(fig)

    # ── Fig 5 ──
    if args.profiles_dir:
        print("[5/5] Center profiles...")
        fig, data = plot_center_profiles(
            Path(args.profiles_dir), args.n_profiles, args.ch, interval_sec)
        if fig is not None:
            save_figure(fig,
                        params={"profiles_dir": args.profiles_dir,
                                "n_profiles": args.n_profiles,
                                "ch": args.ch},
                        description="Center-line phase profiles sampled over time, colored by timepoint",
                        data=data)
            plt.close(fig)
    else:
        print("[5/5] Center profiles skipped: --profiles-dir not provided")

    print("\nDone.")


if __name__ == "__main__":
    main()
