"""
compute_drift_online.py
-----------------------
Per-position online drift correction for QPI time-lapse microscopy.

Each position computes its stage-drift correction independently against its own
grid(0,0) reference. Positions are processed in parallel with
ProcessPoolExecutor. When ``drift_sample_interval > 1``, positions are grouped
and only the leader's correction is computed; members share the leader's value.

Config keys (drift_config.json):
  drift_sample_interval: 1     -- 1 = every pos, N = every Nth pos (grouping)
  max_drift_workers: 0         -- 0 = auto (cpu_count - 4)

Usage:
    python compute_drift_online.py --timepoint 5 --config drift_config.json
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import re
import json
import csv
import time
import shutil as _shutil
import argparse
import numpy as np
import tifffile
import cv2
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, ALL_COMPLETED
import threading
import queue as _queue_mod


# ================================================================
# Configuration
# ================================================================

def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


# ================================================================
# Kalman filter (position-only random walk, scalar, nm units)
# ================================================================

def kf_step_posonly_nm(z_nm: float, pos_nm: float, P: float,
                       Q: float, R: float):
    """One Kalman step for a 1D position-only random-walk model.

    Model:  x_k = x_{k-1} + w_k,  w ~ N(0, Q)
            z_k = x_k + v_k,      v ~ N(0, R)
    Ref:    Kalman (1960); Labbe, "Kalman and Bayesian Filters in Python", ch.4.
    Returns (pos_new_nm, P_new, K).
    """
    P_pred = P + Q
    K = P_pred / (P_pred + R)
    pos_new = pos_nm + K * (z_nm - pos_nm)
    P_new = (1.0 - K) * P_pred
    return float(pos_new), float(P_new), float(K)


from ecc_utils import (
    tilt_fit_crop, extract_rect_roi, ecc_align,
    mad, remove_outliers_mad,
    # Float ECC input (clipped float32, no 8-bit quantisation) aliased to the
    # to_uint8 name so every call site below feeds float32 to ecc_align. The
    # *_u8 variable names are kept. ecc_min_corr comes from drift_config (0.99).
    to_ecc_input as to_uint8,
)


def _tilt_correct(img_f64, cy, cx, crop_w, tilt_crop_h, ecc_crop_h,
                  fit_right: bool = False):
    """Thin wrapper binding tilt_crop_h/ecc_crop_h to tilt_fit_crop."""
    return tilt_fit_crop(img_f64, cy, cx, crop_w, ecc_crop_h, tilt_crop_h,
                         fit_right=fit_right)


# ================================================================
# Grid reference management (2-pass / 3-pass ECC)
# ================================================================

def scan_grid_positions(grid_dir, base_label):
    """Scan grid_dir for ``{base_label}_x<i>_y<j>`` folders and return
    a {(xi, yi): folder_path} map."""
    grid_dir = Path(grid_dir)
    pattern = re.compile(rf"^{re.escape(base_label)}_x([+-]?\d+)_y([+-]?\d+)$")
    pos_map = {}
    for d in grid_dir.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            pos_map[(int(m.group(1)), int(m.group(2)))] = d
    return pos_map


def _select_nearest_grid(shift_x, shift_y, pos_map, grid_cal):
    """Pick the grid (xi, yi) nearest to an image-space shift [px].

    Uses measured per-grid pixel offsets from ``grid_cal``. Only entries that
    are also present in ``pos_map`` (i.e. an actual grid TIFF exists) are
    considered.
    """
    best_key, best_dist = None, float('inf')
    for key, (adx, ady) in grid_cal.items():
        if key not in pos_map:
            continue
        dist = ((adx - shift_x) ** 2 + (ady - shift_y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_key = key
    return best_key


def _get_grid_offset_px(xi, yi, grid_cal):
    """Return the grid(xi, yi) reference offset [image px] relative to grid(0, 0).

    ``(xi, yi)`` must be present in ``grid_cal`` (guaranteed by
    ``_select_nearest_grid``, which only picks calibrated keys).
    """
    return grid_cal[(xi, yi)]


def _load_grid_ref_full(pos_map, xi, yi, rois, n_channels, z_index, cfg,
                        tilt_crop_h=0, ecc_crop_h=0, fit_right=False):
    """Load per-channel grid(xi, yi) reference crops for pass 2 / pass 3."""
    pos_dir = pos_map[(xi, yi)]
    fname = f"img_000000000_ph_{z_index:03d}_phase.tif"
    path = pos_dir / "output_phase" / fname
    if not path.exists():
        raise FileNotFoundError(f"Grid reference image not found: {path}")
    grid_img = tifffile.imread(str(path)).astype(np.float64)
    refs_out = []
    for ch in range(n_channels):
        roi = rois[ch] if ch < len(rois) else rois[-1]
        cropped = _tilt_correct(grid_img, roi["cy"], roi["cx"], roi["crop_w"],
                                tilt_crop_h, ecc_crop_h, fit_right=fit_right)
        refs_out.append(cropped)
    return refs_out


# ================================================================
# ProcessPoolExecutor worker state + initializer
# ================================================================

_wk = {}  # worker-process shared data (set by _init_drift_worker)


def _init_drift_worker(cfg, per_pos_rois, leader_data_dict, bg_phases, t,
                       pre_phases=None):
    """Initialize worker process with shared data.

    ``bg_phases`` is a dict ``{"before": ndarray|None, "after": ndarray|None}``
    of pre-reconstructed BG phases (one per crop variant). Workers pick the
    matching one by ``pos_index`` and subtract directly — no redundant BG
    reconstruction per position.

    ``pre_phases`` is a dict ``{pos_idx: ndarray}`` of raw phases
    pre-reconstructed on GPU in the main process. When available, workers skip
    the FFT+unwrap step entirely.
    """
    global _wk
    script_dir = cfg.get("script_dir", "")
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    _wk = {
        'cfg': cfg, 'per_pos_rois': per_pos_rois,
        'leader_data': leader_data_dict,
        'bg_phases': bg_phases, 't': t,
        'pre_phases': pre_phases or {},
    }


def _process_leader_task(args):
    """Top-level worker function for ProcessPoolExecutor.

    Closures are not picklable, so this must be a module-level function.
    Reads shared data from _wk (set by _init_drift_worker).
    """
    idx, label, state_path, kf_path, kf_R = args
    if idx not in _wk['leader_data']:
        return None
    ecc_z = _wk['cfg'].get('raw_tl_z_index', 0)
    n_z = _wk['cfg'].get('n_z_slices', 1)
    raw_path = get_raw_path(_wk['cfg']['save_dir'], label, _wk['t'],
                            z_index=ecc_z, n_z_slices=n_z)
    prev = read_per_pos_state(state_path, idx)
    kf_st = load_per_pos_kf_state(kf_path, label, kf_R)
    ld = _wk['leader_data'][idx]
    pos_split = _wk['cfg'].get("pos_split", 3)
    bg_phase = _wk['bg_phases']["after" if idx >= pos_split else "before"]
    pre_phase = _wk.get('pre_phases', {}).get(idx)
    pos_rois = _wk['per_pos_rois'][idx]
    return _process_one_position(
        idx, label, str(raw_path), bg_phase,
        _wk['cfg'], pos_rois,
        ld['u8_crops'], ld['pos_map'], ld['grid_cal'],
        prev, kf_st, pre_phase_raw=pre_phase)


# ================================================================
# Argument parsing
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Per-position online drift correction")
    p.add_argument("--timepoint", type=int, required=True)
    p.add_argument("--config", required=True)
    return p.parse_args()


# ================================================================
# Position / path helpers
# ================================================================

def load_positions_csv(csv_path):
    """Load positions.csv -> list of dicts with index, label, x, y."""
    positions = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            positions.append({
                "index": int(row[0]),
                "label": row[1].strip(),
                "x": float(row[2]),
                "y": float(row[3]),
            })
    return positions


def get_raw_path(save_dir, label, timepoint, z_index=0, n_z_slices=1):
    base = Path(save_dir) / label / f"z{z_index:03d}"
    return base / f"img_{timepoint:09d}_ph_{z_index:03d}.tif"


def _pos_index_from_label(label):
    """Extract numeric index from label, e.g. 'Pos5' -> 5."""
    m = re.search(r"\d+", label)
    if not m:
        raise ValueError(f"Cannot extract index from position label: {label!r}")
    return int(m.group())


# ================================================================
# Phase reconstruction (per-pos crop selection)
# ================================================================

def _reconstruct_phase_raw(raw_path, cfg, pos_index=0):
    """Reconstruct phase from a raw TIF, no BG subtraction.

    Crop is selected by ``pos_index`` vs ``cfg['pos_split']``. Used for both
    sample and BG reconstruction; callers subtract pre-computed BG themselves.
    """
    script_dir = Path(cfg["script_dir"])
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from PIL import Image
    from qpi import QPIParameters, get_field
    from skimage.restoration import unwrap_phase
    from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

    pos_split = cfg.get("pos_split", 3)
    crop = tuple(cfg["crop_before"]) if pos_index < pos_split else tuple(cfg["crop_after"])
    rs, re_, cs, ce = crop

    img = np.array(Image.open(str(raw_path)))
    img_crop = img[rs:re_, cs:ce]
    qpi_params = QPIParameters(
        wavelength=WAVELENGTH, NA=NA,
        img_shape=img_crop.shape, pixelsize=PIXELSIZE,
        offaxis_center=OFFAXIS_CENTER,
    )
    field = get_field(img_crop, qpi_params)
    if hasattr(field, "get"):  # CuPy array -> numpy
        angle = np.angle(field.get())
    else:
        angle = np.angle(field)
    return unwrap_phase(angle)


def reconstruct_bg_phase_variants(bg_path, cfg, pos_indices):
    """Reconstruct BG phase once per crop variant needed by the given positions.

    Returns ``{"before": ndarray|None, "after": ndarray|None}``. Entries are
    only populated for crop variants actually used by ``pos_indices``, so the
    other entry stays ``None`` and is not sent to workers.
    """
    out = {"before": None, "after": None}
    if bg_path is None or not Path(bg_path).exists():
        return out
    pos_split = cfg.get("pos_split", 3)
    needs_before = any(i < pos_split for i in pos_indices)
    needs_after  = any(i >= pos_split for i in pos_indices)
    if needs_before:
        # pos_index=0 forces crop_before
        out["before"] = _reconstruct_phase_raw(bg_path, cfg, pos_index=0).astype(np.float32)
    if needs_after:
        # pos_index=pos_split forces crop_after
        out["after"]  = _reconstruct_phase_raw(bg_path, cfg, pos_index=pos_split).astype(np.float32)
    return out


def reconstruct_phase_for_pos(raw_path, cfg, bg_phase=None, pos_index=0):
    """QPI phase reconstruction for one sample image.

    ``bg_phase`` is the pre-reconstructed BG phase (ndarray) for the same crop
    variant as this pos; callers are responsible for picking the matching one.
    Subtraction is skipped when ``bg_phase`` is None.
    """
    phase = _reconstruct_phase_raw(raw_path, cfg, pos_index=pos_index)
    if bg_phase is not None:
        phase = phase - bg_phase
    return phase


# ================================================================
# GPU+CPU pipeline: 1 GPU producer (FFT) + N CPU consumers (unwrap)
# ================================================================

_N_CPU_CONSUMERS = 12


def _gpu_cpu_recon_pipeline(frame_specs, cfg, n_consumers=_N_CPU_CONSUMERS):
    """Reconstruct raw holograms using a producer/consumer pipeline.

    ``frame_specs``: list of dicts with keys:
        raw_path (str), pos_index (int), key (any hashable identifier)
        Optional: save_path (str|None) — if set, write phase TIF here
    Returns dict  {key: ndarray(float64)} of reconstructed phases.

    GPU producer runs get_field (CuPy FFT) sequentially, pushes angle arrays
    to a bounded queue; CPU consumer threads run unwrap_phase in parallel.
    """
    if not frame_specs:
        return {}

    script_dir = Path(cfg["script_dir"])
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from PIL import Image
    from qpi import QPIParameters, get_field
    from skimage.restoration import unwrap_phase
    from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

    pos_split = cfg.get("pos_split", 3)
    crop_before = tuple(cfg["crop_before"])
    crop_after = tuple(cfg["crop_after"])

    q = _queue_mod.Queue(maxsize=48)
    results = {}
    results_lock = threading.Lock()
    qpi_cache = {}

    def _producer():
        for spec in frame_specs:
            raw_path = Path(spec["raw_path"])
            if not raw_path.exists():
                continue
            pidx = spec["pos_index"]
            crop = crop_before if pidx < pos_split else crop_after
            try:
                img = np.array(Image.open(str(raw_path)))
                rs, re_, cs, ce = crop
                img_crop = img[rs:re_, cs:ce]
                cache_key = (img_crop.shape, crop)
                if cache_key not in qpi_cache:
                    qpi_cache[cache_key] = QPIParameters(
                        wavelength=WAVELENGTH, NA=NA,
                        img_shape=img_crop.shape, pixelsize=PIXELSIZE,
                        offaxis_center=OFFAXIS_CENTER,
                    )
                field = get_field(img_crop, qpi_cache[cache_key])
                if hasattr(field, "get"):
                    angle = np.angle(field.get())
                else:
                    angle = np.angle(field)
                q.put((spec, angle))
            except Exception as e:
                print(f"  [pipeline] producer error {raw_path.name}: {e}")
        q.put(None)

    def _consumer():
        while True:
            item = q.get()
            if item is None:
                q.put(None)
                break
            spec, angle = item
            try:
                phase = unwrap_phase(angle)
                save_path = spec.get("save_path")
                if save_path:
                    p = Path(save_path)
                    p.parent.mkdir(exist_ok=True)
                    tifffile.imwrite(str(p), phase.astype(np.float32))
                with results_lock:
                    results[spec["key"]] = phase
            except Exception as e:
                print(f"  [pipeline] consumer error: {e}")

    with ThreadPoolExecutor(max_workers=1 + n_consumers) as pool:
        pool.submit(_producer)
        futs = [pool.submit(_consumer) for _ in range(n_consumers)]
        for f in futs:
            f.result()

    return results


# ================================================================
# Grid reference management
# ================================================================

def load_grid_ref_crops_for_pos(pos_label, cfg, rois):
    """Load grid(0,0) reference crops for a position.
    Returns (float64_crops, uint8_crops) lists.
    """
    grid_dir = Path(cfg["grid_dir"])
    z_index = cfg.get("grid_z_index", 0)
    tilt_crop_h = cfg.get("tilt_crop_h", 0)
    ecc_crop_h = cfg.get("ecc_crop_h", 0)
    vmin = cfg.get("ecc_vmin", -5.0)
    vmax = cfg.get("ecc_vmax", 2.0)
    pos_split = cfg.get("pos_split", 3)
    pos_index = _pos_index_from_label(pos_label)
    fit_right = pos_index >= pos_split

    grid_ref_path = (grid_dir / f"{pos_label}_x+0_y+0" / "output_phase"
                     / f"img_000000000_ph_{z_index:03d}_phase.tif")
    if not grid_ref_path.exists():
        raise FileNotFoundError(f"Grid ref not found: {grid_ref_path}")

    grid_img = tifffile.imread(str(grid_ref_path)).astype(np.float64)

    gradient_sigma = cfg.get("gradient_sigma", 0)
    if gradient_sigma > 0:
        from scipy.ndimage import gaussian_filter
        grid_img = grid_img - gaussian_filter(grid_img, sigma=gradient_sigma, mode="nearest")

    f64_crops, u8_crops = [], []
    for roi in rois:
        crop = _tilt_correct(grid_img, roi["cy"], roi["cx"], roi["crop_w"],
                             tilt_crop_h, ecc_crop_h, fit_right=fit_right)
        if crop is None:
            f64_crops.append(None)
            u8_crops.append(None)
        else:
            f64_crops.append(crop.astype(np.float64))
            u8_crops.append(to_uint8(crop, vmin, vmax))
    return f64_crops, u8_crops


def load_grid_cal_for_pos(pos_label, cfg):
    """Load measured grid calibration for a position.

    Returns a dict ``{(xi, yi): (dx_px, dy_px)}`` of measured pixel offsets
    relative to grid(0, 0). The calibration JSON
    (``grid_dir/grid_calibration_{pos_label}.json``) is mandatory; a missing
    file is a configuration error and raises ``FileNotFoundError`` so the
    nominal-step fallback path can no longer be silently entered.
    """
    grid_dir = Path(cfg["grid_dir"])
    cal_path = grid_dir / f"grid_calibration_{pos_label}.json"
    if not cal_path.exists():
        raise FileNotFoundError(
            f"Grid calibration file not found for {pos_label}: {cal_path}. "
            f"Run calibrate_grid_pos_per_pos.py first."
        )
    with open(cal_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        (e["xi"], e["yi"]): (-e["actual_dx_px"], -e["actual_dy_px"])
        for e in data.get("positions", [])
    }


# ================================================================
# Per-pos state management
# ================================================================

def read_per_pos_state(state_path, pos_idx):
    """Read previous per-pos state from drift_state.txt.

    Returns zeros / ``None`` defaults when the file does not exist yet
    (first timepoint). Malformed lines propagate as exceptions.
    """
    result = {"cumulative_dx_um": 0.0, "cumulative_dy_um": 0.0,
              "ema_tx_px": None, "ema_ty_px": None}
    keys = {
        f"CUMULATIVE_DX_UM_{pos_idx}": "cumulative_dx_um",
        f"CUMULATIVE_DY_UM_{pos_idx}": "cumulative_dy_um",
        f"EMA_TX_PX_{pos_idx}": "ema_tx_px",
        f"EMA_TY_PX_{pos_idx}": "ema_ty_px",
    }
    try:
        f = open(state_path, encoding="utf-8")
    except FileNotFoundError:
        return result
    with f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k in keys:
                result[keys[k]] = float(v)
    return result


def write_per_pos_state(state_path, t, pos_results, bg_pos_index):
    """Write drift_state.txt with per-position entries."""
    valid_results = [r for r in pos_results if r.get("valid") and not r.get("jump")]
    any_jump = any(r.get("jump", False) for r in pos_results)

    if valid_results:
        avg_dx = float(np.mean([r["cumulative_dx_um"] for r in valid_results]))
        avg_dy = float(np.mean([r["cumulative_dy_um"] for r in valid_results]))
        avg_step_dx = float(np.mean([r["dx_um"] for r in valid_results]))
        avg_step_dy = float(np.mean([r["dy_um"] for r in valid_results]))
        avg_corr = float(np.mean([r["corr"] for r in valid_results]))
    else:
        avg_dx = avg_dy = 0.0
        avg_step_dx = avg_step_dy = 0.0
        avg_corr = 0.0

    lines = [
        "# drift_state.txt - written by compute_drift_online.py",
        f"STATUS={'correction_ready' if valid_results else 'correction_skipped'}",
        f"TIMEPOINT={t}",
        f"PER_POS=true",
        f"CUMULATIVE_DX_UM={avg_dx:.6f}",
        f"CUMULATIVE_DY_UM={avg_dy:.6f}",
        f"DX_UM={avg_step_dx:.6f}",
        f"DY_UM={avg_step_dy:.6f}",
        f"ECC_CORRELATION={avg_corr:.6f}",
        f"CORRECTION_VALID={'true' if valid_results and not any_jump else 'false'}",
        f"JUMP_DETECTED={'true' if any_jump else 'false'}",
    ]
    for r in pos_results:
        i = r["pos_idx"]
        lines.extend([
            f"CUMULATIVE_DX_UM_{i}={r['cumulative_dx_um']:.6f}",
            f"CUMULATIVE_DY_UM_{i}={r['cumulative_dy_um']:.6f}",
            f"DX_UM_{i}={r['dx_um']:.6f}",
            f"DY_UM_{i}={r['dy_um']:.6f}",
            f"EMA_TX_PX_{i}={r['ema_tx']:.6f}",
            f"EMA_TY_PX_{i}={r['ema_ty']:.6f}",
            f"CORRECTION_VALID_{i}={'true' if r['valid'] and not r['jump'] else 'false'}",
            f"ECC_CORRELATION_{i}={r['corr']:.6f}",
        ])
    lines.append(f"TIMESTAMP={datetime.now().isoformat()}")
    with open(state_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def load_per_pos_kf_state(kf_path, pos_label, R_nm2):
    """Load Kalman-filter state for a single position.

    The KF-state file stores one nested dict per position label. Returns a
    default state (zero position, P = R) when the file does not exist or
    the label has no entry yet.
    """
    default = {"kf_pos_tx_nm": 0.0, "kf_P_tx": R_nm2,
               "kf_pos_ty_nm": 0.0, "kf_P_ty": R_nm2}
    try:
        f = open(kf_path, encoding="utf-8")
    except FileNotFoundError:
        return default
    with f:
        data = json.load(f)
    entry = data.get(pos_label)
    if not isinstance(entry, dict):
        return default
    return {k: entry.get(k, default[k]) for k in default}


def save_all_kf_states(kf_path, kf_updates):
    """Merge per-pos KF-state updates into the on-disk JSON file.

    ``kf_updates`` maps pos_label -> state dict. Existing entries for other
    positions are preserved.
    """
    try:
        f = open(kf_path, encoding="utf-8")
    except FileNotFoundError:
        existing = {}
    else:
        with f:
            existing = json.load(f)
    existing.update(kf_updates)
    with open(kf_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


# ================================================================
# Thread-safe grid cache for per-channel ECC threading
# ================================================================

class _GridCache:
    """Thread-safe lazy cache for grid reference crops (Pass 2 / Pass 3).

    Multiple ECC threads may request the same ``(xi, yi)`` grid
    simultaneously.  Double-checked locking with per-key ``Event``
    ensures only one thread performs the TIF I/O for a given key while
    others that need a *different* key proceed in parallel.
    """

    def __init__(self, pos_map, rois, n_channels, grid_z_index, cfg,
                 tilt_crop_h, ecc_crop_h, fit_right, vmin, vmax):
        self._pos_map = pos_map
        self._rois = rois
        self._n_channels = n_channels
        self._grid_z_index = grid_z_index
        self._cfg = cfg
        self._tilt_crop_h = tilt_crop_h
        self._ecc_crop_h = ecc_crop_h
        self._fit_right = fit_right
        self._vmin = vmin
        self._vmax = vmax
        self._lock = threading.Lock()
        self._halves = {}
        self._halves_u8 = {}
        self._loading = {}

    def get(self, xi, yi):
        """Return ``(halves_f64, halves_u8)`` for grid ``(xi, yi)``.

        Thread-safe: fast path returns cached data without locking;
        slow path uses a per-key ``Event`` so at most one thread loads
        any given key.
        """
        key = (xi, yi)

        # Fast path -- already cached
        if key in self._halves_u8:
            return self._halves[key], self._halves_u8[key]

        with self._lock:
            # Double-check after acquiring lock
            if key in self._halves_u8:
                return self._halves[key], self._halves_u8[key]
            if key in self._loading:
                event = self._loading[key]
                is_loader = False
            else:
                event = threading.Event()
                self._loading[key] = event
                is_loader = True

        if not is_loader:
            event.wait()
            return self._halves[key], self._halves_u8[key]

        # We are the loader -- I/O outside the lock
        try:
            halves = _load_grid_ref_full(
                self._pos_map, xi, yi, self._rois, self._n_channels,
                self._grid_z_index, self._cfg,
                tilt_crop_h=self._tilt_crop_h,
                ecc_crop_h=self._ecc_crop_h,
                fit_right=self._fit_right)
            halves_u8 = [
                None if h is None else to_uint8(h, self._vmin, self._vmax)
                for h in halves
            ]
        except Exception:
            with self._lock:
                self._loading.pop(key, None)
            event.set()
            raise

        with self._lock:
            self._halves[key] = halves
            self._halves_u8[key] = halves_u8
            self._loading.pop(key, None)
        event.set()
        return halves, halves_u8


# ================================================================
# Core: process one position
# ================================================================

def _process_one_position(pos_idx, pos_label, raw_path, bg_phase,
                          cfg, rois, grid_ref_u8, pos_map, grid_cal,
                          prev_state, kf_state, pre_phase_raw=None):
    """Full drift pipeline for one position.

    When ``pre_phase_raw`` is provided (GPU pre-reconstructed in main process),
    the FFT+unwrap step is skipped entirely.

    Returns dict with:
        pos_idx, pos_label, dx_um, dy_um, cumulative_dx_um, cumulative_dy_um,
        valid, jump, corr, ema_tx, ema_ty, kf_update, channel_details
    """
    t_start = datetime.now()
    n_channels = len(rois)
    vmin = cfg.get("ecc_vmin", -5.0)
    vmax = cfg.get("ecc_vmax", 2.0)
    ecc_min_corr = cfg.get("ecc_min_corr", 0.0)
    jump_thresh = cfg.get("jump_thresh_um", 1.0)
    max_total = cfg.get("max_total_corr_um", 15.0)
    pixel_scale_um = cfg.get("pixel_scale_um", 0.3462)
    sx_sign = cfg.get("shift_sign_x", 1)
    sy_sign = cfg.get("shift_sign_y", 1)
    grid_z_index = cfg.get("grid_z_index", 0)
    enable_third_pass = cfg.get("enable_third_pass", True)
    tilt_crop_h = cfg.get("tilt_crop_h", 0)
    ecc_crop_h = cfg.get("ecc_crop_h", 0)
    pos_split = cfg.get("pos_split", 3)
    fit_right = pos_idx >= pos_split
    ema_alpha = cfg.get("correction_ema_alpha", 1.0)
    use_kalman = cfg.get("use_kalman_filter", False)

    fail_result = {
        "pos_idx": pos_idx, "pos_label": pos_label,
        "dx_um": 0.0, "dy_um": 0.0,
        "cumulative_dx_um": prev_state["cumulative_dx_um"],
        "cumulative_dy_um": prev_state["cumulative_dy_um"],
        "valid": False, "jump": False, "corr": 0.0,
        "ema_tx": prev_state.get("ema_tx_px", 0.0) or 0.0,
        "ema_ty": prev_state.get("ema_ty_px", 0.0) or 0.0,
        "kf_update": None, "channel_details": [],
    }

    # ---- Phase reconstruction ----
    if pre_phase_raw is not None:
        phase_raw = pre_phase_raw
    elif not Path(raw_path).exists():
        print(f"  [{pos_label}] ERROR: raw image not found: {raw_path}")
        return fail_result
    else:
        try:
            phase_raw = _reconstruct_phase_raw(raw_path, cfg, pos_idx)
        except Exception as ex:
            print(f"  [{pos_label}] ERROR: phase reconstruction failed: {ex}")
            return fail_result

    # Save raw phase (no BG subtraction) - matches batch step0's output_phase_raw/
    raw_out_dir = Path(raw_path).parent / "output_phase_raw"
    raw_out_dir.mkdir(exist_ok=True)
    raw_out_path = raw_out_dir / (Path(raw_path).stem + "_phase.tif")
    tifffile.imwrite(str(raw_out_path), phase_raw.astype(np.float32))

    phase = phase_raw - bg_phase if bg_phase is not None else phase_raw.copy()

    # Mean removal
    h_p, w_p = phase.shape
    region = phase[1:h_p - 1, 1:w_p // 2]
    if region.size > 0:
        phase -= np.mean(region)

    # Gradient removal
    gradient_sigma = cfg.get("gradient_sigma", 0)
    if gradient_sigma > 0:
        from scipy.ndimage import gaussian_filter
        phase = phase - gaussian_filter(phase, sigma=gradient_sigma, mode="nearest")

    # Save BG-subtracted phase
    out_dir = Path(raw_path).parent / "output_phase"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (Path(raw_path).stem + "_phase.tif")
    tifffile.imwrite(str(out_path), phase.astype(np.float32))

    # ---- Channel crops (tilt correction) ----
    # None entries mark channels whose tilt wide-crop went OOB; downstream
    # ECC skips them rather than falling back to a narrower crop.
    current_crops = []
    for roi in rois:
        crop = _tilt_correct(phase, roi["cy"], roi["cx"], roi["crop_w"],
                             tilt_crop_h, ecc_crop_h, fit_right=fit_right)
        current_crops.append(crop)

    # ---- Multi-pass ECC per channel (threaded) ----
    ecc_threads = cfg.get("ecc_threads_per_pos", 1)
    grid_cache = _GridCache(pos_map, rois, n_channels, grid_z_index, cfg,
                            tilt_crop_h, ecc_crop_h, fit_right, vmin, vmax)

    def _ecc_one_channel(ch_idx):
        """Multi-pass ECC for one channel (closure over read-only locals)."""
        ref_u8_p1 = grid_ref_u8[ch_idx] if ch_idx < len(grid_ref_u8) else grid_ref_u8[-1]
        cur_crop = current_crops[ch_idx]

        if cur_crop is None or ref_u8_p1 is None:
            return (ch_idx, None, None, None,
                    {"ch": ch_idx, "outlier": True, "status": "tilt_bounds_ng"})

        cur_u8 = to_uint8(cur_crop, vmin, vmax)

        # Pass 1: grid(0,0)
        result1 = ecc_align(ref_u8_p1, cur_u8)
        if result1 is None:
            return (ch_idx, None, None, None,
                    {"ch": ch_idx, "outlier": True, "status": "pass1_failed"})

        fine1_x, fine1_y, corr1 = result1
        shift1_x, shift1_y = fine1_x, fine1_y
        detail = {"tx1": shift1_x, "ty1": shift1_y, "corr1": corr1,
                  "xi": 0, "yi": 0}

        # Pass 2: nearest grid
        xi2, yi2 = _select_nearest_grid(shift1_x, shift1_y, pos_map, grid_cal)
        offset_x2, offset_y2 = _get_grid_offset_px(xi2, yi2, grid_cal)

        _, halves_u8_p2 = grid_cache.get(xi2, yi2)
        ref_u8_p2 = halves_u8_p2[ch_idx]
        if ref_u8_p2 is None:
            result2 = None
        else:
            result2 = ecc_align(ref_u8_p2, cur_u8)
        if result2 is None:
            detail.update({"xi": xi2, "yi": yi2, "tx2": shift1_x, "ty2": shift1_y,
                           "corr2": corr1, "status": "pass2_ecc_failed"})
            return (ch_idx, shift1_x, shift1_y, corr1,
                    {"ch": ch_idx, "outlier": False, **detail})

        fine2_x, fine2_y, corr2 = result2
        shift2_x = fine2_x + offset_x2
        shift2_y = fine2_y + offset_y2
        detail.update({"xi": xi2, "yi": yi2, "tx2": shift2_x, "ty2": shift2_y, "corr2": corr2})

        # Pass 3
        final_shift_x, final_shift_y, final_corr = shift2_x, shift2_y, corr2
        final_status = "pass2_ok"

        if enable_third_pass:
            xi3, yi3 = _select_nearest_grid(shift2_x, shift2_y, pos_map, grid_cal)

            if (xi3, yi3) != (xi2, yi2):
                offset_x3, offset_y3 = _get_grid_offset_px(xi3, yi3, grid_cal)

                _, halves_u8_p3 = grid_cache.get(xi3, yi3)
                ref_u8_p3 = halves_u8_p3[ch_idx]
                if ref_u8_p3 is None:
                    result3 = None
                else:
                    result3 = ecc_align(ref_u8_p3, cur_u8)
                if result3 is not None:
                    fine3_x, fine3_y, corr3 = result3
                    final_shift_x = fine3_x + offset_x3
                    final_shift_y = fine3_y + offset_y3
                    final_corr = corr3
                    final_status = "pass3_ok"
                    detail.update({"xi3": xi3, "yi3": yi3,
                                   "tx3": final_shift_x, "ty3": final_shift_y, "corr3": corr3})

        detail["status"] = final_status
        return (ch_idx, final_shift_x, final_shift_y, final_corr,
                {"ch": ch_idx, "outlier": False, **detail})

    # Dispatch channels: threaded when ecc_threads > 1, sequential otherwise
    ch_range = list(range(min(n_channels, len(current_crops))))

    if ecc_threads <= 1 or len(ch_range) <= 1:
        ch_results = [_ecc_one_channel(ch) for ch in ch_range]
    else:
        with ThreadPoolExecutor(max_workers=min(ecc_threads, len(ch_range))) as tex:
            ch_results = list(tex.map(_ecc_one_channel, ch_range))

    # Collect results (deterministic channel order from tex.map)
    tx_list, ty_list, corr_list = [], [], []
    valid_ch_indices = []
    channel_details = []
    for ch_idx, tx, ty, corr, detail in ch_results:
        channel_details.append(detail)
        if tx is not None:
            tx_list.append(tx)
            ty_list.append(ty)
            corr_list.append(corr)
            valid_ch_indices.append(ch_idx)

    # ---- All channels failed ----
    if not tx_list:
        print(f"  [{pos_label}] ERROR: ECC failed on all channels")
        return fail_result

    # ---- Channel averaging (outlier removal) ----
    n_ch_raw = len(tx_list)
    low_corr_mask = np.zeros(n_ch_raw, dtype=bool)
    if ecc_min_corr > 0:
        low_corr_mask = np.array([c < ecc_min_corr for c in corr_list])

    if n_ch_raw >= 3:
        is_out = remove_outliers_mad(tx_list, 5.0) | remove_outliers_mad(ty_list, 5.0) | low_corr_mask
        used_idx = [i for i, o in enumerate(is_out) if not o]
        if not used_idx:
            used_idx = list(range(n_ch_raw))
    else:
        is_out = low_corr_mask
        used_idx = [i for i, o in enumerate(is_out) if not o]
        if not used_idx:
            used_idx = list(range(n_ch_raw))

    # Mark outliers in channel_details
    detail_idx = 0
    for cd in channel_details:
        if cd.get("status") != "pass1_failed":
            if detail_idx < len(is_out):
                cd["outlier"] = bool(is_out[detail_idx])
            detail_idx += 1

    tx_arr = np.array(tx_list)
    ty_arr = np.array(ty_list)
    corr_arr = np.array(corr_list)
    tx_avg = float(np.mean(tx_arr[used_idx]))
    ty_avg = float(np.mean(ty_arr[used_idx]))
    corr_avg = float(np.mean(corr_arr[used_idx]))

    # ---- EMA filter ----
    prev_ema_tx = prev_state["ema_tx_px"]
    prev_ema_ty = prev_state["ema_ty_px"]
    if prev_ema_tx is None:
        tx_filt, ty_filt = tx_avg, ty_avg
    else:
        tx_filt = ema_alpha * tx_avg + (1.0 - ema_alpha) * prev_ema_tx
        ty_filt = ema_alpha * ty_avg + (1.0 - ema_alpha) * prev_ema_ty

    # ---- Sign/scale conversion (pixel -> um, image -> stage) ----
    correction_stage_x_um = sx_sign * ty_filt * pixel_scale_um
    correction_stage_y_um = sy_sign * tx_filt * pixel_scale_um

    # ---- Kalman filter ----
    kf_update = None
    if use_kalman:
        kf_Q_ty = cfg.get("kf_Q_ty_nm2", cfg.get("kf_Q_pos_nm2", 548.0))
        kf_Q_tx = cfg.get("kf_Q_tx_nm2", cfg.get("kf_Q_pos_nm2", 548.0))
        kf_R_ty = cfg.get("kf_R_ty_nm2", cfg.get("kf_R_nm2", 454.0))
        kf_R_tx = cfg.get("kf_R_tx_nm2", cfg.get("kf_R_nm2", 454.0))
        px_scale_nm = pixel_scale_um * 1000.0

        z_ty_nm = tx_avg * px_scale_nm * sy_sign
        z_tx_nm = ty_avg * px_scale_nm * sx_sign
        ol_pos_ty_nm = prev_state["cumulative_dy_um"] * 1000.0 + z_ty_nm
        ol_pos_tx_nm = prev_state["cumulative_dx_um"] * 1000.0 + z_tx_nm

        pos_ty_new, P_ty_new, K_ty = kf_step_posonly_nm(
            ol_pos_ty_nm, kf_state["kf_pos_ty_nm"], kf_state["kf_P_ty"], kf_Q_ty, kf_R_ty)
        pos_tx_new, P_tx_new, K_tx = kf_step_posonly_nm(
            ol_pos_tx_nm, kf_state["kf_pos_tx_nm"], kf_state["kf_P_tx"], kf_Q_tx, kf_R_tx)

        correction_stage_y_um = pos_ty_new / 1000.0 - prev_state["cumulative_dy_um"]
        correction_stage_x_um = pos_tx_new / 1000.0 - prev_state["cumulative_dx_um"]

        kf_update = {
            "kf_pos_tx_nm": float(pos_tx_new), "kf_P_tx": float(P_tx_new),
            "kf_pos_ty_nm": float(pos_ty_new), "kf_P_ty": float(P_ty_new),
        }

    # ---- Cumulative drift ----
    is_first_frame = prev_state["ema_tx_px"] is None
    if is_first_frame:
        cum_dx = cum_dy = 0.0
    else:
        cum_dx = prev_state["cumulative_dx_um"] + correction_stage_x_um
        cum_dy = prev_state["cumulative_dy_um"] + correction_stage_y_um

    # ---- Jump detection ----
    step_um = (correction_stage_x_um**2 + correction_stage_y_um**2) ** 0.5
    total_um = (cum_dx**2 + cum_dy**2) ** 0.5
    if jump_thresh is None:
        jump = total_um > max_total
    else:
        jump = (step_um > jump_thresh) or (total_um > max_total)

    if jump:
        cum_dx = prev_state["cumulative_dx_um"]
        cum_dy = prev_state["cumulative_dy_um"]

    elapsed = (datetime.now() - t_start).total_seconds()
    status_str = "JUMP" if jump else "OK"
    print(f"  [{pos_label}] {status_str}  "
          f"dx={correction_stage_x_um:+.4f}um dy={correction_stage_y_um:+.4f}um  "
          f"cum=({cum_dx:+.3f},{cum_dy:+.3f})um  "
          f"corr={corr_avg:.3f}  {len(used_idx)}/{n_ch_raw}ch  {elapsed:.1f}s")

    return {
        "pos_idx": pos_idx,
        "pos_label": pos_label,
        "dx_um": correction_stage_x_um,
        "dy_um": correction_stage_y_um,
        "cumulative_dx_um": cum_dx,
        "cumulative_dy_um": cum_dy,
        "valid": True,
        "jump": jump,
        "corr": corr_avg,
        "ema_tx": tx_filt,
        "ema_ty": ty_filt,
        "kf_update": kf_update,
        "channel_details": sorted(channel_details, key=lambda x: x["ch"]),
        "n_channels_used": len(used_idx),
        "n_channels_raw": n_ch_raw,
        "tx_avg_px": tx_avg,
        "ty_avg_px": ty_avg,
        "raw_path": str(raw_path),
    }


# ================================================================
# Phase B: crop-subtracted save (grid_subtract on all positions)
# ================================================================
# Phase B runs AFTER Phase A (drift feedback) in the same process. For each
# Pos with a valid Phase A result, it:
#   (a) reconstructs the raw timelapse phase (RAW_CROP from optical_config),
#   (b) selects the nearest grid Pos from the measured calibration,
#   (c) loads the grid reference (raw reconstruction, prefer prerecon TIF),
#   (d) runs grid_subtract.process_single_frame for byte-compatible output,
#   (e) writes per-channel TIFs atomically,
#   (f) appends one entry to pos_shifts_cal_online.json per Pos.
# Budget-limited via crop_sub_max_seconds; leftover Pos fall back to offline.


def _save_crop_sub_one_pos(args):
    """Phase B worker: save crop-subtracted per-channel TIFs for one (t, pos).

    Must be module-level for ProcessPoolExecutor pickling. Returns a dict
    describing the outcome; main collects these to update the per-Pos
    pos_shifts_cal_online.json.
    """
    (t, pos_label, sx, sy, cfg, rois, crop_sub_root_str) = args
    try:
        script_dir = cfg.get("script_dir", "")
        if script_dir and script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        import grid_subtract as gs

        crop_sub_root = Path(crop_sub_root_str)
        save_dir = Path(cfg["save_dir"])

        m = re.match(r"Pos(\d+)", pos_label)
        pos_num = int(m.group(1)) if m else 0
        pos_split = int(cfg.get("pos_split", 0))
        raw_crop = tuple(cfg["crop_before"]) if pos_num < pos_split else tuple(cfg["crop_after"])
        fit_right = pos_num >= pos_split

        grid_dir = Path(cfg["grid_dir"])
        pos_map = gs.scan_grid_positions(grid_dir, pos_label)
        if not pos_map:
            return {"pos_label": pos_label, "status": "no_pos_map"}
        cal_path = grid_dir / f"grid_calibration_{pos_label}.json"
        if not cal_path.exists():
            return {"pos_label": pos_label, "status": "no_grid_cal",
                    "path": str(cal_path)}
        grid_cal = gs.load_grid_calibration(str(cal_path))

        xi, yi, dist_um, dx_um, dy_um, cal_dx, cal_dy, residual_x, residual_y = gs.select_grid(
            sx, sy, pos_map, grid_cal,
            pixel_scale_um=cfg["pixel_scale_um"],
            x_step=cfg.get("crop_sub_x_step_um", 0.1),
            y_step=cfg.get("crop_sub_y_step_um", 0.1),
            shift_sign_x=-1, shift_sign_y=-1,
        )

        grid_pos_dir = pos_map.get((xi, yi))
        tilt_h = cfg.get("tilt_crop_h_raw", 270)
        out_base = (crop_sub_root / pos_label / "output_phase" /
                    "channels" / "crop_sub_rawraw")

        # Determine z-slices to process
        n_z_slices = cfg.get("n_z_slices", 1)
        primary_tl_z = cfg.get("raw_tl_z_index", 0)
        z_indices = list(range(n_z_slices)) if n_z_slices > 1 else [primary_tl_z]

        n_saved = 0
        for z_idx in z_indices:
            raw_holo = save_dir / pos_label / f"z{z_idx:03d}" / f"img_{t:09d}_ph_{z_idx:03d}.tif"

            # Prefer pre-reconstructed phase (saved by _prerecon_save_all_z)
            prerecon_tl = raw_holo.parent / "output_phase_raw" / (raw_holo.stem + "_phase.tif")
            if prerecon_tl.exists():
                tl_img = tifffile.imread(str(prerecon_tl)).astype(np.float64)
            elif raw_holo.exists():
                qpi_params = gs._make_qpi_params_raw(raw_holo, raw_crop)
                tl_img = gs._reconstruct_raw(raw_holo, qpi_params, raw_crop)
            else:
                continue

            # Derive grid z from raw_grid_z_index (single source of truth)
            grid_z = z_idx + cfg["raw_grid_z_index"] - cfg.get("raw_tl_z_index", 0)
            grid_img = None
            if grid_pos_dir is not None:
                prerecon = grid_pos_dir / "output_phase_raw" / f"img_000000000_ph_{grid_z:03d}_phase.tif"
                if prerecon.exists():
                    grid_img = tifffile.imread(str(prerecon)).astype(np.float64)
                else:
                    grid_holo = grid_pos_dir / f"img_000000000_ph_{grid_z:03d}.tif"
                    if grid_holo.exists():
                        g_qpi = gs._make_qpi_params_raw(grid_holo, raw_crop)
                        grid_img = gs._reconstruct_raw(grid_holo, g_qpi, raw_crop)

            per_channel_out, _ = gs.process_single_frame(
                tl_img, sx, sy, rois,
                cal_dx, cal_dy, residual_x, residual_y,
                grid_img,
                output_crop_h_override=tilt_h,
                tilt_crop_h_raw=tilt_h,
                use_raw_phase=True,
                apply_subpixel_correction=True,
                fit_right=fit_right,
                apply_inverse_shift=False,
            )

            tif_name = raw_holo.name
            for ch in range(len(rois)):
                ch_dir = out_base / f"z{z_idx:03d}" / f"ch{ch:02d}"
                ch_dir.mkdir(parents=True, exist_ok=True)
                final = ch_dir / tif_name
                tmp = ch_dir / (tif_name + ".tmp")
                tifffile.imwrite(str(tmp), per_channel_out[ch].astype(np.float32))
                os.replace(str(tmp), str(final))
            n_saved += 1

        if n_saved == 0:
            return {"pos_label": pos_label, "status": "missing_holo",
                    "path": str(save_dir / pos_label / f"img_{t:09d}_ph_*.tif")}

        return {
            "pos_label": pos_label,
            "status": "ok",
            "n_z_saved": n_saved,
            "frame_entry": {
                "frame_index": int(t),
                "shift_x_avg": float(sx),
                "shift_y_avg": float(sy),
                "grid_xi": int(xi),
                "grid_yi": int(yi),
                "residual_x_px": float(residual_x),
                "residual_y_px": float(residual_y),
                "grid_nearest_dist_um": (float(dist_um)
                                          if dist_um is not None else None),
            },
        }
    except Exception as e:
        import traceback
        return {"pos_label": pos_label, "status": "error",
                "error": repr(e), "tb": traceback.format_exc()}


def _append_online_pos_shifts(crop_sub_root, pos_label, frame_entry, cfg):
    """Atomic append of one frame entry to pos_shifts_cal_online.json for a Pos.

    frame_results is a sparse list indexed by frame_index (padded with None).
    Safe for re-run: the same t simply overwrites the entry. On first write,
    populates top-level metadata so correct_0pergluc.py can consume the JSON
    directly without a synthesized grid_subtract_log.json.
    """
    json_path = (Path(crop_sub_root) / pos_label / "output_phase" /
                 "channels" / "pos_shifts_cal_online.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    pos_num = _pos_index_from_label(pos_label)
    pos_split = int(cfg.get("pos_split", 0))
    raw_crop = (cfg["crop_before"] if pos_num < pos_split
                else cfg["crop_after"])

    default = {
        "schema_version": 2,
        "source": "online_crop_sub",
        "base_label": pos_label,
        "grid_dir": cfg.get("grid_dir", ""),
        "grid_z_index": int(cfg.get("raw_grid_z_index", 18)),
        "tl_z_index": int(cfg.get("raw_tl_z_index", 0)),
        "x_step_um": float(cfg.get("crop_sub_x_step_um", 0.1)),
        "y_step_um": float(cfg.get("crop_sub_y_step_um", 0.1)),
        "shift_sign_x": int(cfg.get("shift_sign_x", -1)),
        "shift_sign_y": int(cfg.get("shift_sign_y", -1)),
        "apply_subpixel_correction": True,
        "apply_inverse_shift": False,
        "use_raw_phase": True,
        "raw_crop": list(raw_crop),
        "tilt_crop_h_raw": int(cfg.get("tilt_crop_h_raw", 270)),
        "frame_results": [],
    }

    data = default
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                loaded = json.load(f)
            # Preserve existing metadata; backfill any keys missing in old files.
            for k, v in default.items():
                loaded.setdefault(k, v)
            loaded["schema_version"] = default["schema_version"]
            data = loaded
        except Exception:
            pass

    fr = data.setdefault("frame_results", [])
    t_idx = int(frame_entry["frame_index"])
    while len(fr) <= t_idx:
        fr.append(None)
    fr[t_idx] = frame_entry
    data["n_frames_seen"] = sum(1 for x in fr if x is not None)

    tmp = json_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp), str(json_path))


def _phase_b_save_crop_sub(t, sample_positions, all_results, cfg,
                           per_pos_rois, deadline_monotonic):
    """Orchestrate Phase B: reconstruct + crop-subtract + save for all Pos.

    Budget-limited: submits tasks to a fresh ProcessPoolExecutor and cancels
    anything not finished by ``deadline_monotonic`` (monotonic clock).
    Positions whose Phase A result is invalid/jump are skipped (offline
    fallback will fill them).
    """
    crop_sub_root = Path(cfg["crop_sub_root"])
    min_free_gb = cfg.get("crop_sub_min_free_gb", 2.0)
    crop_sub_root.mkdir(parents=True, exist_ok=True)
    try:
        free_gb = _shutil.disk_usage(str(crop_sub_root)).free / 1e9
        if free_gb < min_free_gb:
            print(f"[phase B] disk free {free_gb:.1f} GB < min {min_free_gb} GB; skipping")
            return
    except Exception as e:
        print(f"[phase B] disk_usage check failed: {e}")

    # Use raw ECC-average shift (tx_avg_px / ty_avg_px). This matches
    # compute_pos_shifts.shift_x_avg used by offline grid_subtract, so online
    # and offline crop-subs share the same grid selection and residuals.
    result_map = {r["pos_idx"]: r for r in all_results
                  if r.get("valid") and not r.get("jump")}

    # Compute fallback shift: mean of all valid positions' ECC shifts.
    # Positions that failed ECC use this average so Phase B still runs.
    valid_sx = [r["tx_avg_px"] for r in result_map.values()
                if r.get("tx_avg_px") is not None]
    valid_sy = [r["ty_avg_px"] for r in result_map.values()
                if r.get("ty_avg_px") is not None]
    fallback_sx = float(np.mean(valid_sx)) if valid_sx else None
    fallback_sy = float(np.mean(valid_sy)) if valid_sy else None

    tasks = []
    skipped = 0
    n_fallback = 0
    for pos in sample_positions:
        pos_rois = per_pos_rois.get(pos["index"])
        if pos_rois is None:
            skipped += 1
            continue
        r = result_map.get(pos["index"])
        if r is not None:
            sx = r.get("tx_avg_px")
            sy = r.get("ty_avg_px")
            if sx is None or sy is None:
                skipped += 1
                continue
        elif fallback_sx is not None:
            sx, sy = fallback_sx, fallback_sy
            n_fallback += 1
        else:
            skipped += 1
            continue
        tasks.append((t, pos["label"], float(sx), float(sy),
                      cfg, pos_rois, str(crop_sub_root)))

    if not tasks:
        print(f"[phase B] no valid positions ({skipped} skipped)")
        return

    max_workers = cfg.get("crop_sub_max_workers", 0)
    if max_workers <= 0:
        max_workers = max(1, os.cpu_count() - 4)
    max_workers = min(max_workers, len(tasks))

    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 1.0:
        print(f"[phase B] no budget left ({remaining:.1f}s); skipping")
        return

    print(f"[phase B] T={t}  {len(tasks)} pos  workers={max_workers}  "
          f"budget={remaining:.0f}s  (skipped={skipped}  fallback={n_fallback})")
    t0 = time.monotonic()

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_save_crop_sub_one_pos, a) for a in tasks]
        done, not_done = wait(futures, timeout=remaining,
                              return_when=ALL_COMPLETED)
        for f in done:
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"status": "exception", "error": repr(e)})
        if not_done:
            print(f"[phase B] budget exceeded; {len(not_done)} tasks canceled")
            for f in not_done:
                f.cancel()

    ok = [r for r in results if r and r.get("status") == "ok"]
    failed = [r for r in results if r and r.get("status") != "ok"]
    elapsed = time.monotonic() - t0
    print(f"[phase B] done  {len(ok)} ok  {len(failed)} failed  {elapsed:.1f}s")
    for r in failed[:5]:
        print(f"  - {r.get('pos_label')}: {r.get('status')} "
              f"{r.get('error', r.get('path', ''))}")

    for r in ok:
        try:
            _append_online_pos_shifts(crop_sub_root, r["pos_label"],
                                      r["frame_entry"], cfg)
        except Exception as e:
            print(f"  [phase B] JSON append failed for {r['pos_label']}: {e}")


# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()
    cfg = load_config(args.config)
    t = args.timepoint

    # ---- Load positions ----
    positions = load_positions_csv(cfg["positions_csv"])
    bg_pos_index = cfg["bg_pos_index"]
    bg_label = positions[bg_pos_index]["label"]
    sample_positions = [p for p in positions if p["index"] != bg_pos_index]

    # ---- Grouping ----
    interval = cfg.get("drift_sample_interval", 1)
    group_leaders = sample_positions[::interval] if interval > 1 else sample_positions
    group_map = {}
    for i, pos in enumerate(sample_positions):
        leader_idx = min((i // interval) * interval, len(sample_positions) - 1) if interval > 1 else i
        group_map[pos["index"]] = sample_positions[leader_idx]["index"]

    t_total_start = datetime.now()
    print(f"[T={t}] compute_drift_online.py  "
          f"{len(group_leaders)} leaders / {len(sample_positions)} positions  "
          f"interval={interval}")

    # ---- Load channel ROIs (per-pos from grid) ----
    grid_dir = Path(cfg["grid_dir"])
    per_pos_rois = {}
    for leader in group_leaders:
        label = leader["label"]
        per_pos_rois_path = (grid_dir / f"{label}_x+0_y+0" / "output_phase"
                             / "channels" / "channel_rois.json")
        if not per_pos_rois_path.exists():
            raise FileNotFoundError(
                f"channel_rois.json not found for {label}: {per_pos_rois_path}")
        with open(per_pos_rois_path, encoding="utf-8") as f:
            per_pos_rois[leader["index"]] = json.load(f)
    n_channels = len(next(iter(per_pos_rois.values())))

    # ---- GPU init (main process only, not in workers) ----
    _use_gpu = False
    try:
        from qpi import set_backend, _HAS_CUPY
        if _HAS_CUPY:
            import cupy as _cp
            _cp.array([1.0]) * 2  # smoke test
            set_backend("cupy")
            _use_gpu = True
            gpu_name = _cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            print(f"  GPU mode: {gpu_name}")
    except Exception as e:
        print(f"  GPU init failed, using CPU: {e}")
        try:
            set_backend("numpy")
        except Exception:
            pass

    # ---- BG image path ----
    ecc_tl_z = cfg.get("raw_tl_z_index", 0)
    n_z_slices = cfg.get("n_z_slices", 1)
    bg_raw = get_raw_path(cfg["save_dir"], bg_label, t, z_index=ecc_tl_z,
                          n_z_slices=n_z_slices)
    if not bg_raw.exists():
        print(f"WARNING: BG image not found: {bg_raw}")
        bg_raw = None

    # ---- Pre-reconstruct BG phase once per crop variant ----
    # Reconstructed here (in main) so every worker does not redo the BG FFT +
    # unwrap_phase on each sample position (saves one BG reconstruction per
    # Pos per timepoint).  When GPU is available, get_field uses CuPy FFT.
    leader_indices = [ld["index"] for ld in group_leaders]
    t_bg_start = datetime.now()
    bg_phases = reconstruct_bg_phase_variants(bg_raw, cfg, leader_indices)
    if bg_raw is not None:
        variants = [k for k, v in bg_phases.items() if v is not None]
        gpu_tag = " (GPU)" if _use_gpu else ""
        print(f"  BG phase reconstructed ({','.join(variants) or 'none'}){gpu_tag} "
              f"in {(datetime.now() - t_bg_start).total_seconds():.2f}s")

    # ---- Pre-reconstruct leader sample phases (GPU+CPU pipeline) ----
    # GPU producer runs FFT sequentially, CPU consumers unwrap in parallel.
    t_pre_start = datetime.now()
    leader_specs = []
    for leader in group_leaders:
        raw_path = get_raw_path(cfg['save_dir'], leader['label'], t,
                                z_index=ecc_tl_z, n_z_slices=n_z_slices)
        if raw_path.exists():
            leader_specs.append({
                "raw_path": str(raw_path),
                "pos_index": leader["index"],
                "key": leader["index"],
            })
    pre_phases = _gpu_cpu_recon_pipeline(leader_specs, cfg)
    n_pre = len(pre_phases)
    if n_pre > 0:
        gpu_tag = " (GPU+CPU pipeline)" if _use_gpu else ""
        print(f"  Pre-reconstructed {n_pre}/{len(group_leaders)} leader phases{gpu_tag} "
              f"in {(datetime.now() - t_pre_start).total_seconds():.2f}s")

    # Reset backend to numpy before thread pool.  CuPy CUDA context is
    # not thread-safe; pre_phases covers all positions anyway.
    if _use_gpu:
        set_backend("numpy")

    # ---- Load grid references and calibration for leaders ----
    # Both grid TIFFs and grid_calibration_{label}.json are mandatory:
    # missing files raise FileNotFoundError and abort the whole session,
    # rather than silently skipping a leader.
    leader_data = {}
    for leader in group_leaders:
        label = leader["label"]
        leader_rois = per_pos_rois[leader["index"]]
        _, u8_crops = load_grid_ref_crops_for_pos(label, cfg, leader_rois)
        pos_map = scan_grid_positions(str(grid_dir), label)
        grid_cal = load_grid_cal_for_pos(label, cfg)
        leader_data[leader["index"]] = {
            "u8_crops": u8_crops,
            "pos_map": pos_map,
            "grid_cal": grid_cal,
        }
        print(f"  {label}: {len(pos_map)} grid pos, {len(grid_cal)} cal")

    # ---- Load previous states and KF states ----
    state_path = cfg["state_file"]
    kf_path = cfg.get("kf_state_file",
                       str(Path(cfg["session_dir"]) / "drift_kf_state.json"))
    kf_R = max(cfg.get("kf_R_ty_nm2", 454.0), cfg.get("kf_R_tx_nm2", 454.0))

    # ---- Process leaders in parallel (ThreadPoolExecutor) ----
    # ThreadPool instead of ProcessPool: OpenCV findTransformECC releases
    # the GIL, so threads achieve true parallelism for ECC.  All threads
    # share pre_phases / leader_data in-process — zero pickling overhead.
    max_workers = cfg.get("max_drift_workers", 0)
    if max_workers <= 0:
        max_workers = max(1, os.cpu_count() - 4)
    max_workers = min(max_workers, len(group_leaders))

    global _wk
    _wk = {
        'cfg': cfg, 'per_pos_rois': per_pos_rois,
        'leader_data': leader_data, 'bg_phases': bg_phases, 't': t,
        'pre_phases': pre_phases,
    }

    task_args = [
        (leader["index"], leader["label"], state_path, kf_path, kf_R)
        for leader in group_leaders
    ]

    t_ecc_start = datetime.now()
    print(f"  ECC processing with {max_workers} workers (ThreadPool)...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        leader_results = list(pool.map(_process_leader_task, task_args))

    # Filter out None results (skipped leaders)
    leader_results = [r for r in leader_results if r is not None]
    print(f"  ECC done in {(datetime.now() - t_ecc_start).total_seconds():.1f}s")

    # ---- Apply group mapping ----
    leader_result_map = {r["pos_idx"]: r for r in leader_results}
    all_results = []
    for pos in sample_positions:
        leader_idx = group_map[pos["index"]]
        if leader_idx in leader_result_map:
            if pos["index"] == leader_idx:
                all_results.append(leader_result_map[leader_idx])
            else:
                # Copy leader's correction for group member
                lr = leader_result_map[leader_idx]
                all_results.append({
                    "pos_idx": pos["index"],
                    "pos_label": pos["label"],
                    "dx_um": lr["dx_um"],
                    "dy_um": lr["dy_um"],
                    "cumulative_dx_um": lr["cumulative_dx_um"],
                    "cumulative_dy_um": lr["cumulative_dy_um"],
                    "valid": lr["valid"],
                    "jump": lr["jump"],
                    "corr": lr["corr"],
                    "ema_tx": lr["ema_tx"],
                    "ema_ty": lr["ema_ty"],
                    "kf_update": None,
                    "channel_details": [],
                })

    # ---- Fallback: replace failed positions' cumulative with valid average ----
    valid_for_fallback = [r for r in all_results
                          if r.get("valid") and not r.get("jump")]
    n_fallback_stage = 0
    if valid_for_fallback:
        avg_cum_dx = float(np.mean([r["cumulative_dx_um"] for r in valid_for_fallback]))
        avg_cum_dy = float(np.mean([r["cumulative_dy_um"] for r in valid_for_fallback]))
        avg_step_dx = float(np.mean([r["dx_um"] for r in valid_for_fallback]))
        avg_step_dy = float(np.mean([r["dy_um"] for r in valid_for_fallback]))
        avg_ema_tx = float(np.mean([r["ema_tx"] for r in valid_for_fallback]))
        avg_ema_ty = float(np.mean([r["ema_ty"] for r in valid_for_fallback]))
        for r in all_results:
            if not r.get("valid") or r.get("jump"):
                r["cumulative_dx_um"] = avg_cum_dx
                r["cumulative_dy_um"] = avg_cum_dy
                r["dx_um"] = avg_step_dx
                r["dy_um"] = avg_step_dy
                r["ema_tx"] = avg_ema_tx
                r["ema_ty"] = avg_ema_ty
                r["valid"] = True
                r["jump"] = False
                n_fallback_stage += 1
    if n_fallback_stage:
        print(f"[T={t}] {n_fallback_stage} failed pos -> stage fallback "
              f"(avg cum=({avg_cum_dx:+.3f},{avg_cum_dy:+.3f})um)")

    # ---- Write per-pos state ----
    write_per_pos_state(state_path, t, all_results, bg_pos_index)

    # ---- Save KF states ----
    kf_updates = {}
    for r in leader_results:
        if r.get("kf_update"):
            kf_updates[r["pos_label"]] = r["kf_update"]
    if kf_updates:
        save_all_kf_states(kf_path, kf_updates)

    # ---- Write log ----
    log_path = cfg["log_file"]
    log_entries = []
    for r in leader_results:
        log_entries.append({
            "timepoint": t,
            "timestamp": datetime.now().isoformat(),
            "pos_idx": r["pos_idx"],
            "pos_label": r["pos_label"],
            "raw_path": r.get("raw_path"),
            "n_channels_used": r.get("n_channels_used", 0),
            "n_channels_raw": r.get("n_channels_raw", 0),
            "tx_avg_px": r.get("tx_avg_px", 0.0),
            "ty_avg_px": r.get("ty_avg_px", 0.0),
            "ecc_correlation": r["corr"],
            "correction_stage_x_um": r["dx_um"],
            "correction_stage_y_um": r["dy_um"],
            "cumulative_dx_um": r["cumulative_dx_um"],
            "cumulative_dy_um": r["cumulative_dy_um"],
            "jump_detected": r["jump"],
            "correction_valid": r["valid"] and not r["jump"],
            "channel_details": r.get("channel_details", []),
        })

    try:
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
    except Exception:
        log = []
    log.append({"timepoint": t, "per_pos": True, "positions": log_entries})
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    n_valid_native = sum(1 for r in all_results if r["valid"] and not r["jump"]) - n_fallback_stage
    n_jump = sum(1 for r in all_results if r["jump"])
    n_failed = len(all_results) - n_valid_native - n_fallback_stage - n_jump
    t_phase_a_elapsed = (datetime.now() - t_total_start).total_seconds()
    print(f"[T={t}] Phase A done in {t_phase_a_elapsed:.1f}s  "
          f"{n_valid_native} valid, {n_fallback_stage} fallback, "
          f"{n_jump} jump, {n_failed} failed")

    # ---- Pre-reconstruct ALL z-slices for all positions (GPU) ----
    # Phase A reconstructed only the ECC z-slice.  Here we reconstruct
    # the remaining z-slices so Phase B can read output_phase_raw TIFs
    # instead of re-running FFT+unwrap on CPU workers.
    n_z_slices = cfg.get("n_z_slices", 1)
    if n_z_slices > 1:
        if _use_gpu:
            try:
                from qpi import set_backend, _HAS_CUPY
                if _HAS_CUPY:
                    set_backend("cupy")
            except Exception:
                pass
        t_zr_start = datetime.now()
        _prerecon_save_all_z(t, positions, cfg, bg_phases, ecc_tl_z)
        if _use_gpu:
            try:
                set_backend("numpy")
            except Exception:
                pass
        print(f"  All-z pre-recon saved in "
              f"{(datetime.now() - t_zr_start).total_seconds():.1f}s")

    # ---- Phase B: crop-subtracted save (optional) ----
    if cfg.get("enable_crop_sub_save", False):
        budget = float(cfg.get("crop_sub_max_seconds", 200.0))
        deadline = time.monotonic() + budget
        try:
            _phase_b_save_crop_sub(t, sample_positions, all_results, cfg,
                                   per_pos_rois, deadline)
        except Exception as e:
            import traceback
            print(f"[phase B] fatal: {e}")
            traceback.print_exc()

    # ---- Cleanup: delete raw holograms for this timepoint ----
    if cfg.get("cleanup_raw_holograms", False):
        _cleanup_raw_holograms(t, sample_positions, cfg)


def _prerecon_save_all_z(t, all_positions, cfg, bg_phases, ecc_z):
    """Reconstruct and save output_phase_raw for ALL z-slices using GPU+CPU pipeline.

    GPU producer runs FFT sequentially, CPU consumers unwrap in parallel.
    Phase A already reconstructed the ECC z-slice; here we reconstruct the
    remaining slices so Phase B (crop_sub) can read the saved TIFs.

    For the BG position (bg_pos_index), both crop_before and crop_after
    variants are saved (output_phase_raw/ and output_phase_raw_after/).
    """
    save_dir = Path(cfg["save_dir"])
    n_z = cfg.get("n_z_slices", 1)
    bg_idx = cfg.get("bg_pos_index", 0)
    pos_split = cfg.get("pos_split", 0)

    specs = []
    for pos in all_positions:
        label = pos["label"]
        idx = pos["index"]
        for z_idx in range(n_z):
            raw_path = get_raw_path(save_dir, label, t,
                                    z_index=z_idx, n_z_slices=n_z)
            out_dir = raw_path.parent / "output_phase_raw"
            out_path = out_dir / (raw_path.stem + "_phase.tif")
            if out_path.exists() or not raw_path.exists():
                continue
            specs.append({
                "raw_path": str(raw_path),
                "pos_index": idx,
                "key": (idx, z_idx, "default"),
                "save_path": str(out_path),
            })

            if idx == bg_idx:
                alt_out_dir = raw_path.parent / "output_phase_raw_after"
                alt_out_path = alt_out_dir / (raw_path.stem + "_phase.tif")
                if not alt_out_path.exists():
                    specs.append({
                        "raw_path": str(raw_path),
                        "pos_index": pos_split,
                        "key": (idx, z_idx, "after"),
                        "save_path": str(alt_out_path),
                    })

    results = _gpu_cpu_recon_pipeline(specs, cfg)
    print(f"  [prerecon] T={t}: saved {len(results)} phase images "
          f"({len(all_positions)} pos x {n_z} z)")


def _cleanup_raw_holograms(t, sample_positions, cfg):
    """Delete raw hologram TIFs for the given timepoint after processing.

    Keeps output_phase/ and output_phase_raw/ intact.
    """
    save_dir = Path(cfg["save_dir"])
    n_z = cfg.get("n_z_slices", 1)
    deleted = 0
    z_range = range(n_z) if n_z > 1 else [cfg.get("raw_tl_z_index", 0)]
    for pos in sample_positions:
        label = pos["label"]
        pos_dir = save_dir / label
        for z_idx in z_range:
            f = pos_dir / f"z{z_idx:03d}" / f"img_{t:09d}_ph_{z_idx:03d}.tif"
            if f.exists():
                f.unlink()
                deleted += 1
    # Also clean BG pos
    bg_label = [p for p in load_positions_csv(cfg["positions_csv"])
                if p["index"] == cfg["bg_pos_index"]]
    if bg_label:
        bg_dir = save_dir / bg_label[0]["label"]
        for z_idx in z_range:
            f = bg_dir / f"z{z_idx:03d}" / f"img_{t:09d}_ph_{z_idx:03d}.tif"
            if f.exists():
                f.unlink()
                deleted += 1
    print(f"  [cleanup] T={t}: deleted {deleted} raw holograms")


if __name__ == "__main__":
    main()
