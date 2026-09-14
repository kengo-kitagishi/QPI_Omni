"""
qpi_fig_04_growth_oscillation.py - Growth-rate oscillation analysis on QPI lineage CSVs.

Reproduction of:
    Liu X, Oh S, Peshkin L, Kirschner MW (2020)
    "Computationally enhanced quantitative phase microscopy reveals
     autonomous oscillations in mammalian cell growth."
    PNAS 117(44):27388-27399. DOI: 10.1073/pnas.2002152117

Headline claim of the paper: single-cell dry-mass trajectories are not cleanly
exponential. The specific growth rate d(ln M)/dt shows autonomous oscillations
with period ~4 h that survive population averaging when traces are aligned to
birth or division.

Eight figures (see plan):
    osc_01_mass_traces            log M(t) per cell with exponential fit + divisions
    osc_02_residuals              residual (log M - linear fit), per-generation
    osc_03_growth_rate_timeseries Savitzky-Golay d(ln M)/dt per cell
    osc_04_autocorrelation        mean autocorrelation of growth-rate residuals
    osc_05_power_spectrum         Welch power spectrum of growth-rate residuals
    osc_06_birth_aligned          mean growth rate vs (t - t_birth)
    osc_07_division_aligned       mean growth rate vs (t - t_div)
    osc_08_growth_rate_carryover  first-half vs second-half mean growth rate

Reuses:
    qpi_fig_03_lineage_analysis.load_cell_data / detect_divisions
    figure_logger.save_figure

Input:
    Same lineage CSVs consumed by qpi_fig_03_lineage_analysis.py.
    Default: 260405 run at
      scripts/results/lineage_csv/test_eroded_volume_overlay__...__f002/*.csv

Usage:
    python qpi_fig_04_growth_oscillation.py
    python qpi_fig_04_growth_oscillation.py --base-dir <dir>
    python qpi_fig_04_growth_oscillation.py --sg-window 11 --sg-poly 3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch
from scipy.stats import pearsonr

# Local imports -------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_logger import save_figure  # noqa: E402
import qpi_fig_03_lineage_analysis as fig03  # noqa: E402
from qpi_fig_03_lineage_analysis import (  # noqa: E402
    OI,
    detect_divisions,
    label_from_path,
    load_cell_data,
    load_filepaths,
)


# ==========================================================================
# Settings
# ==========================================================================

# Default CSV directory: same as qpi_fig_03 default (260405 eroded-mask run).
BASE_DIR = fig03.BASE_DIR

FRAMES_PER_HOUR = fig03.FRAMES_PER_HOUR          # 12 frames/h = 5 min interval
TIME_INTERVAL_H = 1.0 / FRAMES_PER_HOUR          # frame spacing in hours
ANALYSIS_MAX_FRAME = 575                         # pre-starvation window only
DIV_AREA_DROP = fig03.DIV_AREA_DROP              # 0.6 size-drop threshold

# Savitzky-Golay smoothing for d(ln M)/dt.
# Window 9 frames = 45 min; polyorder 2 preserves local curvature.
SG_WINDOW_FRAMES = 9
SG_POLYORDER = 2

# How many frames around a detected division to mask out (division artefact).
DIV_GUARD_FRAMES = 1

# Autocorrelation / spectrum control.
MAX_LAG_H = 6.0                                  # plot autocorr out to 6 h
WELCH_NPERSEG_FRAMES = 64                        # Welch segment length (~5.3 h)

# Alignment window around birth / division events (hours).
ALIGN_BEFORE_H = 0.5
ALIGN_AFTER_H = 1.0


# ==========================================================================
# Data structures
# ==========================================================================

@dataclass
class CellTrace:
    """Per-cell time series plus growth-rate outputs."""
    label: str
    source: str
    time_h: np.ndarray          # shape (T,)
    mass_pg: np.ndarray         # shape (T,)
    log_mass: np.ndarray        # shape (T,)
    div_frames: np.ndarray      # integer frame indices of division events
    div_times: np.ndarray       # wall-clock times at division events
    segment_bounds: list[tuple[int, int]]  # [(start_idx, end_idx_exclusive), ...]
    growth_rate: np.ndarray     # shape (T,), NaN near divisions and edges
    residual_log_mass: np.ndarray  # shape (T,), per-segment linear residual
    residual_growth_rate: np.ndarray  # shape (T,), growth_rate - per-cell mean


# ==========================================================================
# Core computations
# ==========================================================================

def segment_indices(n_frames: int, div_frames: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive), ...] splitting 0..n_frames at divisions.

    A division frame is treated as the first frame of the NEW generation
    (same convention as qpi_fig_03.detect_divisions).
    """
    boundaries = [0] + [int(f) for f in div_frames if 0 < f < n_frames] + [n_frames]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def savgol_growth_rate(log_mass: np.ndarray,
                       dt_h: float,
                       window: int,
                       polyorder: int) -> np.ndarray:
    """Savitzky-Golay first derivative of log M -> d(ln M)/dt [1/h].

    Returns NaN where the window would extend past the input.
    """
    n = len(log_mass)
    out = np.full(n, np.nan, dtype=float)
    if n < window or window < polyorder + 2:
        return out
    if window % 2 == 0:
        window += 1
    # savgol_filter pads internally; we trust values only where the window fits.
    deriv = savgol_filter(log_mass, window_length=window,
                          polyorder=polyorder, deriv=1, delta=dt_h)
    half = window // 2
    out[half:n - half] = deriv[half:n - half]
    return out


def savgol_growth_rate_per_segment(log_mass: np.ndarray,
                                   segments: list[tuple[int, int]],
                                   dt_h: float,
                                   window: int,
                                   polyorder: int) -> np.ndarray:
    """Apply Savitzky-Golay within each (start, end_exclusive) segment only.

    Savitzky-Golay smooths across its window. Applied to the whole trajectory
    it mixes pre- and post-division frames at every division boundary, biasing
    the derivative sharply negative for ~window/2 frames on each side. Running
    SG segment-by-segment keeps the derivative estimate local to one
    generation.
    """
    out = np.full_like(log_mass, np.nan, dtype=float)
    for s, e in segments:
        if e - s < window:
            continue
        seg = log_mass[s:e]
        valid = np.isfinite(seg)
        if valid.sum() < window:
            continue
        seg_growth = savgol_growth_rate(seg, dt_h, window, polyorder)
        out[s:e] = seg_growth
    return out


def per_segment_linear_residual(time_h: np.ndarray,
                                log_mass: np.ndarray,
                                segments: list[tuple[int, int]],
                                min_len: int = 4) -> np.ndarray:
    """For each (start, end) segment, fit log M = a + lambda * t and return residual.

    Segments shorter than min_len are filled with NaN.
    """
    out = np.full_like(log_mass, np.nan, dtype=float)
    for s, e in segments:
        if e - s < min_len:
            continue
        t = time_h[s:e]
        y = log_mass[s:e]
        valid = np.isfinite(y)
        if valid.sum() < min_len:
            continue
        a, b = np.polyfit(t[valid], y[valid], 1)  # a = slope, b = intercept
        out[s:e] = y - (a * t + b)
    return out


def guard_divisions(arr: np.ndarray,
                    div_frames: np.ndarray,
                    guard: int) -> np.ndarray:
    """Set values within +/-guard frames of any division to NaN.

    Division frames are mass-artefact-prone (abrupt drop between mother and
    daughter). Smoothing across them contaminates the growth rate estimate.
    """
    out = arr.copy()
    n = len(arr)
    for f in div_frames:
        lo = max(0, int(f) - guard)
        hi = min(n, int(f) + guard + 1)
        out[lo:hi] = np.nan
    return out


def build_cell_trace(df: pd.DataFrame, source: str,
                     sg_window: int, sg_polyorder: int,
                     div_guard: int) -> CellTrace | None:
    """Convert one loaded CSV dataframe into a CellTrace with growth rate etc."""
    label = label_from_path(source)
    time_h = df["Time"].to_numpy()
    mass = df["DryMass"].to_numpy().astype(float)
    if len(mass) < sg_window + 2:
        return None

    # log(M); guard against nonpositive mass (should not happen, but safe).
    with np.errstate(invalid="ignore", divide="ignore"):
        log_mass = np.log(mass)
    log_mass[~np.isfinite(log_mass)] = np.nan

    divs_df = detect_divisions(df)
    div_frames = divs_df.index.to_numpy()
    div_times = time_h[div_frames] if len(div_frames) else np.array([])

    segments = segment_indices(len(mass), div_frames)

    dt_h = float(np.median(np.diff(time_h))) if len(time_h) > 1 else TIME_INTERVAL_H
    # Apply Savitzky-Golay per generation to avoid smoothing across divisions.
    growth = savgol_growth_rate_per_segment(log_mass, segments, dt_h,
                                            sg_window, sg_polyorder)
    # Belt-and-braces: also blank ±guard frames in case of near-boundary noise.
    growth = guard_divisions(growth, div_frames, div_guard)

    residual = per_segment_linear_residual(time_h, log_mass, segments)

    # Per-cell mean of valid growth-rate samples.
    finite_gr = growth[np.isfinite(growth)]
    mean_gr = float(np.mean(finite_gr)) if finite_gr.size else 0.0
    residual_gr = growth - mean_gr

    return CellTrace(
        label=label,
        source=source,
        time_h=time_h,
        mass_pg=mass,
        log_mass=log_mass,
        div_frames=div_frames,
        div_times=div_times,
        segment_bounds=segments,
        growth_rate=growth,
        residual_log_mass=residual,
        residual_growth_rate=residual_gr,
    )


def pooled_autocorrelation(residual_series: list[np.ndarray],
                           dt_h: float,
                           max_lag_h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean biased autocorrelation across cells.

    Returns (lag_h, mean_acf, sem_acf).
    """
    max_lag = int(round(max_lag_h / dt_h))
    acf_stack = []
    for r in residual_series:
        x = r.copy()
        x = x[np.isfinite(x)]
        if x.size < max_lag + 4:
            continue
        x = x - np.mean(x)
        denom = np.dot(x, x)
        if denom <= 0:
            continue
        acf = np.correlate(x, x, mode="full")[len(x) - 1:] / denom
        acf = acf[:max_lag + 1]
        acf_stack.append(acf)
    if not acf_stack:
        return np.array([]), np.array([]), np.array([])
    m = np.stack(acf_stack)
    lag_h = np.arange(max_lag + 1) * dt_h
    mean = m.mean(axis=0)
    sem = m.std(axis=0, ddof=1) / np.sqrt(m.shape[0]) if m.shape[0] > 1 else np.zeros_like(mean)
    return lag_h, mean, sem


def pooled_welch(residual_series: list[np.ndarray],
                 dt_h: float,
                 nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean Welch power spectrum across cells. Frequencies in 1/h."""
    fs = 1.0 / dt_h  # samples per hour
    specs = []
    freq_ref = None
    for r in residual_series:
        x = r[np.isfinite(r)]
        if x.size < nperseg + 4:
            continue
        f, p = welch(x, fs=fs, nperseg=min(nperseg, x.size), detrend="constant")
        if freq_ref is None:
            freq_ref = f
        # Interpolate onto the first cell's freq grid if segment lengths differ.
        if f.shape != freq_ref.shape:
            p = np.interp(freq_ref, f, p)
        specs.append(p)
    if not specs:
        return np.array([]), np.array([])
    return freq_ref, np.stack(specs).mean(axis=0)


def stack_event_windows(traces: list[CellTrace],
                        event_attr: str,
                        before_h: float,
                        after_h: float,
                        dt_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Stack growth-rate windows around birth or division events.

    event_attr must be 'div_times' (existing events). For birth alignment we
    reuse the division time but interpret window [0, after_h]; for division
    alignment we use window [-before_h, 0].

    Returns (lag_h, matrix) where matrix shape = (n_events, n_lags).
    """
    before_n = int(round(before_h / dt_h))
    after_n = int(round(after_h / dt_h))
    lag_h = np.arange(-before_n, after_n + 1) * dt_h
    rows = []
    for tr in traces:
        for ev_time in getattr(tr, event_attr):
            # Find the nearest frame index in tr.time_h.
            idx = int(np.argmin(np.abs(tr.time_h - ev_time)))
            lo = idx - before_n
            hi = idx + after_n + 1
            if lo < 0 or hi > len(tr.growth_rate):
                continue
            rows.append(tr.growth_rate[lo:hi])
    if not rows:
        return lag_h, np.zeros((0, len(lag_h)))
    return lag_h, np.stack(rows)


def mean_growth_rate_by_half(trace: CellTrace) -> list[tuple[float, float]]:
    """For each complete segment, return (mean first half, mean second half) of d(ln M)/dt."""
    out = []
    for s, e in trace.segment_bounds:
        if e - s < 6:
            continue
        mid = s + (e - s) // 2
        first = trace.growth_rate[s:mid]
        second = trace.growth_rate[mid:e]
        f_valid = first[np.isfinite(first)]
        s_valid = second[np.isfinite(second)]
        if f_valid.size < 3 or s_valid.size < 3:
            continue
        out.append((float(np.mean(f_valid)), float(np.mean(s_valid))))
    return out


# ==========================================================================
# Figures
# ==========================================================================

def fig_osc_01_mass_traces(traces: list[CellTrace]) -> None:
    """log(M) vs time per cell, with per-segment exponential fit."""
    fig, ax = plt.subplots(figsize=(183 / 25.4, 90 / 25.4))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(traces), 1)))
    for tr, color in zip(traces, colors):
        ax.plot(tr.time_h, tr.log_mass, lw=0.6, color=color, alpha=0.8,
                label=tr.label)
        # Per-segment linear fit overlay
        for s, e in tr.segment_bounds:
            if e - s < 4:
                continue
            y = tr.log_mass[s:e]
            t = tr.time_h[s:e]
            valid = np.isfinite(y)
            if valid.sum() < 4:
                continue
            a, b = np.polyfit(t[valid], y[valid], 1)
            ax.plot(t, a * t + b, lw=0.6, color=color, ls="--", alpha=0.6)
        ax.scatter(tr.div_times, tr.log_mass[tr.div_frames],
                   s=10, color=OI["vermilion"], zorder=5)
    ax.set_xlabel("Time [h]", fontsize=8)
    ax.set_ylabel("log(dry mass / pg)", fontsize=8)
    ax.set_title("Single-cell log mass traces with per-generation exponential fit",
                 fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    fig.tight_layout()
    save_figure(
        fig,
        params={"n_cells": len(traces), "max_frame": ANALYSIS_MAX_FRAME},
        description="osc_01 log mass vs time with exponential fits (Liu Fig 3A analog)",
    )


def fig_osc_02_residuals(traces: list[CellTrace]) -> None:
    """Residuals from per-generation linear fit, stacked + histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(183 / 25.4, 80 / 25.4),
                                   gridspec_kw={"width_ratios": [2.0, 1.0]})
    all_resid = []
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(traces), 1)))
    for tr, color in zip(traces, colors):
        r = tr.residual_log_mass
        ax1.plot(tr.time_h, r, lw=0.5, color=color, alpha=0.8, label=tr.label)
        all_resid.append(r[np.isfinite(r)])
    ax1.axhline(0, color="0.6", ls=":", lw=0.5)
    ax1.set_xlabel("Time [h]", fontsize=8)
    ax1.set_ylabel("log M - linear fit", fontsize=8)
    ax1.set_title("Per-generation exponential-fit residuals", fontsize=8)
    ax1.legend(fontsize=6, loc="upper right", ncol=2)
    ax1.tick_params(labelsize=7)

    if all_resid:
        pooled = np.concatenate(all_resid)
        ax2.hist(pooled, bins=40, color=OI["skyblue"], edgecolor="k", linewidth=0.3)
        ax2.set_xlabel("residual", fontsize=8)
        ax2.set_ylabel("frequency", fontsize=8)
        ax2.set_title(f"Residual distribution (n={pooled.size})", fontsize=8)
        ax2.tick_params(labelsize=7)

    fig.tight_layout()
    save_figure(
        fig,
        params={"n_cells": len(traces)},
        description="osc_02 per-generation fit residuals, trace + histogram (Liu Fig 3B/C analog)",
    )


def fig_osc_03_growth_rate_timeseries(traces: list[CellTrace]) -> None:
    """Savitzky-Golay d(ln M)/dt vs time per cell."""
    fig, ax = plt.subplots(figsize=(183 / 25.4, 90 / 25.4))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(traces), 1)))
    all_valid = []
    for tr, color in zip(traces, colors):
        ax.plot(tr.time_h, tr.growth_rate, lw=0.6, color=color, alpha=0.8,
                label=tr.label)
        all_valid.append(tr.growth_rate[np.isfinite(tr.growth_rate)])
        for t_div in tr.div_times:
            ax.axvline(t_div, color=color, ls=":", lw=0.3, alpha=0.4)
    if all_valid:
        pooled = np.concatenate(all_valid)
        mu = float(np.mean(pooled))
        ax.axhline(mu, color="0.4", ls="--", lw=0.6,
                   label=f"pooled mean = {mu:.3f} 1/h")
    ax.set_xlabel("Time [h]", fontsize=8)
    ax.set_ylabel(r"$d(\ln M)/dt$ [1/h]", fontsize=8)
    ax.set_title("Specific growth rate (Savitzky-Golay) vs time", fontsize=8)
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_figure(
        fig,
        params={"n_cells": len(traces),
                "sg_window": SG_WINDOW_FRAMES,
                "sg_polyorder": SG_POLYORDER,
                "window_minutes": SG_WINDOW_FRAMES * 60 / FRAMES_PER_HOUR},
        description="osc_03 Savitzky-Golay specific growth rate per cell (Liu Fig 4 analog)",
    )


def fig_osc_04_autocorrelation(traces: list[CellTrace]) -> None:
    """Mean autocorrelation of growth-rate residuals across cells."""
    residuals = [tr.residual_growth_rate for tr in traces]
    dt_h = TIME_INTERVAL_H
    lag_h, mean_acf, sem_acf = pooled_autocorrelation(residuals, dt_h, MAX_LAG_H)
    if lag_h.size == 0:
        print("  [osc_04] Not enough data for autocorrelation. Skipping.")
        return
    fig, ax = plt.subplots(figsize=(120 / 25.4, 80 / 25.4))
    ax.plot(lag_h, mean_acf, color=OI["blue"], lw=1.2, label="mean ACF")
    if sem_acf.size:
        ax.fill_between(lag_h, mean_acf - sem_acf, mean_acf + sem_acf,
                        color=OI["blue"], alpha=0.25, label="+/- SEM")
    ax.axhline(0, color="0.6", ls=":", lw=0.5)
    ax.set_xlabel("Lag [h]", fontsize=8)
    ax.set_ylabel("Autocorrelation of d(ln M)/dt residual", fontsize=8)
    ax.set_title(f"Pooled autocorrelation (n={len(residuals)} cells)", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    first_peak_lag = None
    if mean_acf.size > 3:
        # Find first local maximum after lag 0.
        for i in range(2, mean_acf.size - 1):
            if mean_acf[i] > mean_acf[i - 1] and mean_acf[i] >= mean_acf[i + 1] \
               and mean_acf[i] > 0:
                first_peak_lag = float(lag_h[i])
                break

    save_figure(
        fig,
        params={"n_cells": len(residuals),
                "max_lag_h": MAX_LAG_H,
                "first_peak_lag_h": first_peak_lag},
        data={"lag_h": lag_h, "mean_acf": mean_acf, "sem_acf": sem_acf},
        description="osc_04 pooled autocorrelation of growth-rate residuals (Liu Fig 5 analog)",
    )


def fig_osc_05_power_spectrum(traces: list[CellTrace]) -> None:
    """Welch power spectrum of growth-rate residuals."""
    residuals = [tr.residual_growth_rate for tr in traces]
    freq, power = pooled_welch(residuals, TIME_INTERVAL_H, WELCH_NPERSEG_FRAMES)
    if freq.size == 0:
        print("  [osc_05] Not enough data for power spectrum. Skipping.")
        return
    fig, ax = plt.subplots(figsize=(120 / 25.4, 80 / 25.4))
    ax.semilogy(freq, power, color=OI["vermilion"], lw=1.2)
    peak_idx = int(np.argmax(power[1:])) + 1 if power.size > 1 else 0
    peak_freq = float(freq[peak_idx])
    peak_period = 1.0 / peak_freq if peak_freq > 0 else float("nan")
    ax.axvline(peak_freq, color="0.4", ls="--", lw=0.5,
               label=f"peak: f={peak_freq:.3f} 1/h (T={peak_period:.2f} h)")
    ax.set_xlabel("Frequency [1/h]", fontsize=8)
    ax.set_ylabel("Power (log scale)", fontsize=8)
    ax.set_title(f"Welch power spectrum of d(ln M)/dt residual (n={len(residuals)} cells)",
                 fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_figure(
        fig,
        params={"n_cells": len(residuals),
                "nperseg_frames": WELCH_NPERSEG_FRAMES,
                "peak_freq_per_h": peak_freq,
                "peak_period_h": peak_period},
        data={"freq_per_h": freq, "power": power},
        description="osc_05 Welch power spectrum of growth-rate residuals (Liu Fig 5 analog)",
    )


def _plot_aligned(ax, lag_h: np.ndarray, matrix: np.ndarray,
                  color: str, label: str) -> None:
    if matrix.shape[0] == 0:
        ax.text(0.5, 0.5, "no events", transform=ax.transAxes, ha="center")
        return
    # Guard against all-NaN columns (no events provide a sample at that lag).
    with np.errstate(all="ignore"):
        mean = np.nanmean(matrix, axis=0)
        n_valid = np.sum(np.isfinite(matrix), axis=0)
        std = np.nanstd(matrix, axis=0, ddof=1)
    sem = np.where(n_valid > 1, std / np.sqrt(np.maximum(n_valid, 1)), 0.0)
    ax.plot(lag_h, mean, color=color, lw=1.2, label=f"{label} (n={matrix.shape[0]})")
    ax.fill_between(lag_h, mean - sem, mean + sem, color=color, alpha=0.25,
                    label="+/- SEM")


def fig_osc_06_birth_aligned(traces: list[CellTrace]) -> None:
    """Mean d(ln M)/dt aligned to birth (= each detected division = start of new cell)."""
    lag_h, matrix = stack_event_windows(traces, "div_times",
                                        before_h=ALIGN_BEFORE_H,
                                        after_h=ALIGN_AFTER_H,
                                        dt_h=TIME_INTERVAL_H)
    fig, ax = plt.subplots(figsize=(120 / 25.4, 80 / 25.4))
    _plot_aligned(ax, lag_h, matrix, OI["green"], "birth-aligned")
    ax.axvline(0, color=OI["vermilion"], ls="--", lw=0.7, label="birth event")
    ax.set_xlabel("Time relative to birth [h]", fontsize=8)
    ax.set_ylabel(r"$d(\ln M)/dt$ [1/h]", fontsize=8)
    ax.set_title("Birth-aligned mean growth rate", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_figure(
        fig,
        params={"n_events": int(matrix.shape[0]),
                "before_h": ALIGN_BEFORE_H, "after_h": ALIGN_AFTER_H},
        description="osc_06 birth-aligned mean growth rate (Liu Fig 6A analog)",
    )


def fig_osc_07_division_aligned(traces: list[CellTrace]) -> None:
    """Mean d(ln M)/dt aligned to next division event (end of cycle)."""
    # Build per-cycle end events: for each segment (s, e), the end event is frame e-1.
    class _Tr:  # lightweight wrapper to reuse stack_event_windows
        pass
    synth = []
    for tr in traces:
        s2 = _Tr()
        s2.time_h = tr.time_h
        s2.growth_rate = tr.growth_rate
        end_times = []
        for s, e in tr.segment_bounds:
            if e - s >= 4 and e - 1 < len(tr.time_h):
                end_times.append(tr.time_h[e - 1])
        s2.div_times = np.array(end_times)
        synth.append(s2)
    lag_h, matrix = stack_event_windows(synth, "div_times",
                                        before_h=ALIGN_AFTER_H,
                                        after_h=ALIGN_BEFORE_H,
                                        dt_h=TIME_INTERVAL_H)
    fig, ax = plt.subplots(figsize=(120 / 25.4, 80 / 25.4))
    _plot_aligned(ax, lag_h, matrix, OI["purple"], "division-aligned")
    ax.axvline(0, color=OI["vermilion"], ls="--", lw=0.7, label="division event")
    ax.set_xlabel("Time relative to division [h]", fontsize=8)
    ax.set_ylabel(r"$d(\ln M)/dt$ [1/h]", fontsize=8)
    ax.set_title("Division-aligned mean growth rate", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_figure(
        fig,
        params={"n_events": int(matrix.shape[0]),
                "before_h": ALIGN_AFTER_H, "after_h": ALIGN_BEFORE_H},
        description="osc_07 division-aligned mean growth rate (Liu Fig 6B analog)",
    )


def fig_osc_08_carryover(traces: list[CellTrace]) -> None:
    """Per-cycle first-half vs second-half mean growth rate."""
    pairs: list[tuple[float, float]] = []
    for tr in traces:
        pairs.extend(mean_growth_rate_by_half(tr))
    if len(pairs) < 4:
        print("  [osc_08] Not enough complete cycles for carryover. Skipping.")
        return
    arr = np.array(pairs)
    first = arr[:, 0]
    second = arr[:, 1]
    r, p = pearsonr(first, second)

    fig, ax = plt.subplots(figsize=(100 / 25.4, 90 / 25.4))
    ax.scatter(first, second, s=18, alpha=0.7, color=OI["orange"], edgecolors="none")
    lim = [min(first.min(), second.min()), max(first.max(), second.max())]
    ax.plot(lim, lim, color="0.6", ls=":", lw=0.5, label="y = x")
    z = np.polyfit(first, second, 1)
    xs = np.linspace(lim[0], lim[1], 50)
    ax.plot(xs, np.polyval(z, xs), color=OI["vermilion"], lw=1, ls="--",
            label=f"slope={z[0]:.2f}")
    ax.set_xlabel("Mean d(ln M)/dt, first half [1/h]", fontsize=8)
    ax.set_ylabel("Mean d(ln M)/dt, second half [1/h]", fontsize=8)
    ax.set_title(f"Within-cycle carryover (n={len(pairs)}, r={r:.2f}, p={p:.2e})",
                 fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_figure(
        fig,
        params={"n_cycles": len(pairs),
                "r": round(float(r), 3),
                "p": float(f"{p:.3e}"),
                "slope": round(float(z[0]), 3)},
        data={"first_half": first, "second_half": second},
        description="osc_08 within-cycle first-half vs second-half growth rate",
    )


# ==========================================================================
# Entry point
# ==========================================================================

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", type=str, default=None,
                    help="CSV directory (default: same as qpi_fig_03 BASE_DIR).")
    ap.add_argument("--sg-window", type=int, default=None,
                    help="Savitzky-Golay window length in frames (odd int).")
    ap.add_argument("--sg-poly", type=int, default=None,
                    help="Savitzky-Golay polynomial order.")
    ap.add_argument("--max-frame", type=int, default=None,
                    help="Drop frames > this (default 575, pre-starvation).")
    ap.add_argument("--div-drop", type=float, default=None,
                    help="Division detection mass-drop ratio (default 0.6).")
    ap.add_argument("--min-valid-frames", type=int, default=100,
                    help="Skip cells with fewer valid frames than this.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    global BASE_DIR, SG_WINDOW_FRAMES, SG_POLYORDER, ANALYSIS_MAX_FRAME, DIV_AREA_DROP
    if args.base_dir:
        BASE_DIR = args.base_dir
    if args.sg_window is not None:
        SG_WINDOW_FRAMES = args.sg_window
    if args.sg_poly is not None:
        SG_POLYORDER = args.sg_poly
    if args.max_frame is not None:
        ANALYSIS_MAX_FRAME = args.max_frame
    if args.div_drop is not None:
        DIV_AREA_DROP = args.div_drop

    # Configure the shared loader from qpi_fig_03 so it produces physical-unit
    # data clipped to the pre-starvation window.
    fig03.DATA_MODE = "physical"
    fig03.ANALYSIS_MAX_FRAME = ANALYSIS_MAX_FRAME
    fig03.DIV_AREA_DROP = DIV_AREA_DROP
    fig03._apply_mode_labels("physical")

    filepaths = load_filepaths(BASE_DIR, [])
    if not filepaths:
        print(f"ERROR: no CSV files in {BASE_DIR}", file=sys.stderr)
        sys.exit(1)

    print("=== Growth-rate oscillation analysis (Liu 2020 reproduction) ===")
    print(f"  Data: {BASE_DIR}")
    print(f"  Files: {len(filepaths)}")
    print(f"  SG window: {SG_WINDOW_FRAMES} frames "
          f"({SG_WINDOW_FRAMES * 60 / FRAMES_PER_HOUR:.1f} min), polyorder={SG_POLYORDER}")
    print(f"  Max frame: {ANALYSIS_MAX_FRAME} "
          f"(= {ANALYSIS_MAX_FRAME / FRAMES_PER_HOUR:.1f} h)")

    traces: list[CellTrace] = []
    for fp in filepaths:
        df = load_cell_data(fp)
        if len(df) < args.min_valid_frames:
            print(f"  {label_from_path(fp)}: SKIP ({len(df)} valid frames)")
            continue
        tr = build_cell_trace(df, fp,
                              sg_window=SG_WINDOW_FRAMES,
                              sg_polyorder=SG_POLYORDER,
                              div_guard=DIV_GUARD_FRAMES)
        if tr is None:
            print(f"  {label_from_path(fp)}: SKIP (trace too short)")
            continue
        traces.append(tr)
        print(f"  {tr.label}: {len(tr.time_h)} frames, "
              f"{len(tr.div_frames)} divisions, "
              f"mean d(ln M)/dt = "
              f"{np.nanmean(tr.growth_rate):.3f} 1/h")

    if not traces:
        print("ERROR: no usable traces.", file=sys.stderr)
        sys.exit(1)

    print("\n--- Generating figures ---")
    print("  [1/8] osc_01 log-mass traces + exponential fits ...")
    fig_osc_01_mass_traces(traces)
    print("  [2/8] osc_02 residuals from per-generation fit ...")
    fig_osc_02_residuals(traces)
    print("  [3/8] osc_03 growth-rate time series ...")
    fig_osc_03_growth_rate_timeseries(traces)
    print("  [4/8] osc_04 autocorrelation ...")
    fig_osc_04_autocorrelation(traces)
    print("  [5/8] osc_05 Welch power spectrum ...")
    fig_osc_05_power_spectrum(traces)
    print("  [6/8] osc_06 birth-aligned growth rate ...")
    fig_osc_06_birth_aligned(traces)
    print("  [7/8] osc_07 division-aligned growth rate ...")
    fig_osc_07_division_aligned(traces)
    print("  [8/8] osc_08 within-cycle carryover ...")
    fig_osc_08_carryover(traces)

    print("\nDone.")


if __name__ == "__main__":
    main()
