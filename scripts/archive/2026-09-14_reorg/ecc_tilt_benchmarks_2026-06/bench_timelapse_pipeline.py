"""
Benchmark each phase of the online timelapse pipeline.

Measures wall-clock time for:
  1. Phase reconstruction (GPU get_field + CPU unwrap_phase)
  2. BG subtraction + tilt correction
  3. ECC alignment (single channel, multi-pass)
  4. Full per-position pipeline (varying thread counts)
  5. Phase B: crop-sub save

Usage:
    python bench_timelapse_pipeline.py --config ../drift_session/drift_config.json --timepoint 0
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import json
import time
import argparse
import numpy as np
import tifffile
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, stdev

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(t):
    if t < 0.001:
        return f"{t*1e6:.0f} us"
    if t < 1.0:
        return f"{t*1000:.1f} ms"
    return f"{t:.2f} s"


def _stats(times, label):
    if len(times) == 1:
        print(f"  {label}: {_fmt(times[0])}")
    else:
        print(f"  {label}: {_fmt(mean(times))} avg, "
              f"{_fmt(min(times))} min, {_fmt(max(times))} max  "
              f"(n={len(times)})")


# ================================================================
# Component benchmarks
# ================================================================

def bench_phase_recon(raw_path, cfg, pos_index, n_repeat=5):
    """Benchmark: raw TIF -> FFT -> angle -> unwrap_phase."""
    from PIL import Image
    from qpi import QPIParameters, get_field, set_backend, _HAS_CUPY
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

    results = {"gpu_fft": [], "cpu_fft": [], "unwrap": [], "total_gpu": [], "total_cpu": []}

    # GPU benchmark
    if _HAS_CUPY:
        import cupy as cp
        set_backend("cupy")
        cp.array([1.0]) * 2  # warmup
        get_field(img_crop, qpi_params)  # warmup JIT
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            field = get_field(img_crop, qpi_params)
            cp.cuda.Stream.null.synchronize()
            t_fft = time.perf_counter() - t0

            t0 = time.perf_counter()
            angle = np.angle(field.get())
            t_angle = time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = unwrap_phase(angle)
            t_unwrap = time.perf_counter() - t0

            results["gpu_fft"].append(t_fft)
            results["unwrap"].append(t_unwrap)
            results["total_gpu"].append(t_fft + t_angle + t_unwrap)
        set_backend("numpy")

    # CPU benchmark
    set_backend("numpy")
    get_field(img_crop, qpi_params)  # warmup
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        field = get_field(img_crop, qpi_params)
        t_fft = time.perf_counter() - t0

        t0 = time.perf_counter()
        angle = np.angle(field)
        t_angle = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = unwrap_phase(angle)
        t_unwrap = time.perf_counter() - t0

        results["cpu_fft"].append(t_fft)
        results["unwrap"].append(t_unwrap)
        results["total_cpu"].append(t_fft + t_angle + t_unwrap)

    return results


def bench_tilt_correct(phase, rois, cfg, n_repeat=20):
    """Benchmark: tilt_fit_crop for all channels."""
    from ecc_utils import tilt_fit_crop
    tilt_h = cfg.get("tilt_crop_h", 270)
    ecc_h = cfg.get("ecc_crop_h", 80)
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        for roi in rois:
            tilt_fit_crop(phase, roi["cy"], roi["cx"], roi["crop_w"],
                          ecc_h, tilt_h, fit_right=False)
        times.append(time.perf_counter() - t0)
    return times


def bench_backsub_offset(phase, rois, cfg, n_repeat=20):
    """Benchmark: compute_backsub_offset (Gaussian fit) for all channels."""
    from compute_drift_online import compute_backsub_offset
    from ecc_utils import extract_rect_roi
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        for roi in rois:
            crop = extract_rect_roi(phase, roi["cy"], roi["cx"],
                                    roi["crop_w"], roi.get("crop_h", 80))
            compute_backsub_offset(crop, cfg)
        times.append(time.perf_counter() - t0)
    return times


def bench_to_uint8(phase, rois, cfg, n_repeat=50):
    """Benchmark: to_uint8 conversion."""
    from ecc_utils import tilt_fit_crop, to_uint8
    tilt_h = cfg.get("tilt_crop_h", 270)
    ecc_h = cfg.get("ecc_crop_h", 80)
    vmin, vmax = cfg.get("ecc_vmin", -5.0), cfg.get("ecc_vmax", 2.0)
    crops = []
    for roi in rois:
        c = tilt_fit_crop(phase, roi["cy"], roi["cx"], roi["crop_w"],
                          ecc_h, tilt_h, fit_right=False)
        if c is not None:
            crops.append(c)
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        for c in crops:
            to_uint8(c, vmin, vmax)
        times.append(time.perf_counter() - t0)
    return times


def bench_ecc_single(ref_u8, tl_u8, n_repeat=10):
    """Benchmark: single ECC alignment call."""
    from ecc_utils import ecc_align
    ecc_align(ref_u8, tl_u8)  # warmup
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        result = ecc_align(ref_u8, tl_u8)
        times.append(time.perf_counter() - t0)
    return times, result


def bench_ecc_channels_threading(ref_u8_list, tl_u8_list, thread_counts, n_repeat=3):
    """Benchmark: ECC on all channels with varying thread counts."""
    from ecc_utils import ecc_align
    results = {}
    for n_threads in thread_counts:
        times = []
        for _ in range(n_repeat):
            if n_threads <= 1:
                t0 = time.perf_counter()
                for ref, tl in zip(ref_u8_list, tl_u8_list):
                    if ref is not None and tl is not None:
                        ecc_align(ref, tl)
                times.append(time.perf_counter() - t0)
            else:
                def _do_ecc(pair):
                    ref, tl = pair
                    if ref is None or tl is None:
                        return None
                    return ecc_align(ref, tl)
                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=n_threads) as tex:
                    list(tex.map(_do_ecc, zip(ref_u8_list, tl_u8_list)))
                times.append(time.perf_counter() - t0)
        results[n_threads] = times
    return results


def bench_opencv_thread_settings(ref_u8, tl_u8, thread_settings, n_repeat=10):
    """Benchmark: ECC with different OpenCV thread counts."""
    from ecc_utils import ecc_align
    results = {}
    for n_cv_threads in thread_settings:
        cv2.setNumThreads(n_cv_threads)
        ecc_align(ref_u8, tl_u8)  # warmup
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            ecc_align(ref_u8, tl_u8)
            times.append(time.perf_counter() - t0)
        results[n_cv_threads] = times
    cv2.setNumThreads(0)  # restore default
    return results


def bench_recon_parallel_gpu_vs_cpu(raw_paths, cfg, pos_indices, n_workers_list):
    """Benchmark: parallel reconstruction with different worker counts."""
    from qpi import set_backend, _HAS_CUPY
    from concurrent.futures import ProcessPoolExecutor

    results = {}

    # Sequential GPU
    if _HAS_CUPY:
        set_backend("cupy")
        import cupy as cp
        cp.array([1.0]) * 2
        from compute_drift_online import _reconstruct_phase_raw
        _reconstruct_phase_raw(str(raw_paths[0]), cfg, pos_indices[0])  # warmup
        t0 = time.perf_counter()
        for rp, pi in zip(raw_paths, pos_indices):
            _reconstruct_phase_raw(str(rp), cfg, pi)
        t_gpu_seq = time.perf_counter() - t0
        set_backend("numpy")
        results["gpu_sequential"] = t_gpu_seq
        print(f"  GPU sequential ({len(raw_paths)} images): {_fmt(t_gpu_seq)} "
              f"({_fmt(t_gpu_seq/len(raw_paths))}/img)")

    return results


# ================================================================
# Main
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--timepoint", type=int, default=0)
    args = p.parse_args()
    cfg = load_config(args.config)
    t = args.timepoint

    print("=" * 60)
    print("QPI Timelapse Pipeline Benchmark")
    print("=" * 60)
    print(f"CPU cores: {os.cpu_count()}")
    print(f"OpenCV threads: {cv2.getNumThreads()}")
    print(f"Config: max_drift_workers={cfg.get('max_drift_workers')}, "
          f"ecc_threads_per_pos={cfg.get('ecc_threads_per_pos')}")
    print(f"Channels: {cfg.get('n_channels', 12)}, "
          f"tilt_crop_h={cfg.get('tilt_crop_h')}, ecc_crop_h={cfg.get('ecc_crop_h')}")
    print()

    # Load positions
    import csv
    positions = []
    with open(cfg["positions_csv"], encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            positions.append({"index": int(row[0]), "label": row[1].strip()})

    bg_idx = cfg["bg_pos_index"]
    sample_positions = [p for p in positions if p["index"] != bg_idx]
    bg_label = positions[bg_idx]["label"]

    # Find raw images
    save_dir = Path(cfg["save_dir"])
    ecc_z = cfg.get("raw_tl_z_index", 0)
    n_z = cfg.get("n_z_slices", 1)

    def get_raw(label):
        base = save_dir / label
        if n_z > 1:
            base = base / f"z{ecc_z:03d}"
        return base / f"img_{t:09d}_ph_{ecc_z:03d}.tif"

    bg_raw = get_raw(bg_label)
    sample_raw = get_raw(sample_positions[0]["label"])
    if not sample_raw.exists():
        print(f"ERROR: sample raw not found: {sample_raw}")
        return

    print(f"Test image: {sample_raw}")
    print(f"BG image: {bg_raw} (exists={bg_raw.exists()})")
    print()

    # ---- 1. Phase Reconstruction ----
    print("=" * 60)
    print("1. Phase Reconstruction")
    print("-" * 60)
    recon = bench_phase_recon(sample_raw, cfg, sample_positions[0]["index"])
    if recon["gpu_fft"]:
        _stats(recon["gpu_fft"], "GPU FFT+mask+crop+IFFT")
    _stats(recon["cpu_fft"], "CPU FFT+mask+crop+IFFT")
    _stats(recon["unwrap"], "unwrap_phase (CPU only)")
    if recon["total_gpu"]:
        _stats(recon["total_gpu"], "Total GPU (FFT+angle+unwrap)")
    _stats(recon["total_cpu"], "Total CPU (FFT+angle+unwrap)")
    if recon["total_gpu"] and recon["total_cpu"]:
        speedup = mean(recon["total_cpu"]) / mean(recon["total_gpu"])
        print(f"  GPU speedup: {speedup:.1f}x")
        pct_unwrap = mean(recon["unwrap"]) / mean(recon["total_gpu"]) * 100
        print(f"  unwrap_phase fraction of GPU total: {pct_unwrap:.0f}%")
    print()

    # ---- 2. Tilt Correction vs Gaussian backsub ----
    print("=" * 60)
    print("2. Tilt Correction vs Gaussian Backsub")
    print("-" * 60)
    # Reconstruct a phase image
    from compute_drift_online import _reconstruct_phase_raw
    from qpi import set_backend, _HAS_CUPY
    if _HAS_CUPY:
        import cupy as cp
        set_backend("cupy")
        cp.array([1.0]) * 2
    phase = _reconstruct_phase_raw(str(sample_raw), cfg, sample_positions[0]["index"])
    if _HAS_CUPY:
        set_backend("numpy")

    # Load ROIs
    grid_dir = Path(cfg["grid_dir"])
    rois_path = (grid_dir / f"{sample_positions[0]['label']}_x+0_y+0" /
                 "output_phase" / "channels" / "channel_rois.json")
    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)
    print(f"  Channels: {len(rois)}, crop shape: {rois[0].get('crop_w')}x{cfg.get('ecc_crop_h')}")

    tilt_times = bench_tilt_correct(phase, rois, cfg)
    _stats(tilt_times, f"tilt_fit_crop ({len(rois)} ch)")
    backsub_times = bench_backsub_offset(phase, rois, cfg)
    _stats(backsub_times, f"gaussian backsub ({len(rois)} ch)")
    u8_times = bench_to_uint8(phase, rois, cfg)
    _stats(u8_times, f"to_uint8 ({len(rois)} ch)")
    print()

    # ---- 3. Single ECC alignment ----
    print("=" * 60)
    print("3. Single ECC Alignment")
    print("-" * 60)
    from ecc_utils import tilt_fit_crop, to_uint8
    tilt_h = cfg.get("tilt_crop_h", 270)
    ecc_h = cfg.get("ecc_crop_h", 80)
    vmin, vmax = cfg.get("ecc_vmin", -5.0), cfg.get("ecc_vmax", 2.0)

    # Load grid ref
    from compute_drift_online import load_grid_ref_crops_for_pos
    _, u8_crops = load_grid_ref_crops_for_pos(
        sample_positions[0]["label"], cfg, rois)

    # Make sample crops
    sample_u8 = []
    for roi in rois:
        c = tilt_fit_crop(phase, roi["cy"], roi["cx"], roi["crop_w"],
                          ecc_h, tilt_h, fit_right=False)
        sample_u8.append(None if c is None else to_uint8(c, vmin, vmax))

    # Find a valid channel pair
    valid_ch = None
    for i in range(len(rois)):
        if u8_crops[i] is not None and sample_u8[i] is not None:
            valid_ch = i
            break

    if valid_ch is not None:
        ref = u8_crops[valid_ch]
        tl = sample_u8[valid_ch]
        print(f"  ECC input shape: {ref.shape} (crop_w x ecc_crop_h)")
        ecc_times, result = bench_ecc_single(ref, tl)
        _stats(ecc_times, "Single ECC call")
        if result:
            print(f"  Result: tx={result[0]:.3f}, ty={result[1]:.3f}, corr={result[2]:.4f}")

        # OpenCV internal thread count
        print()
        print("  OpenCV internal thread count sweep (single ECC):")
        cv_thread_settings = [1, 2, 4, 8, 14, 28]
        cv_results = bench_opencv_thread_settings(ref, tl, cv_thread_settings)
        for n_t, times in cv_results.items():
            _stats(times, f"    cv2.setNumThreads({n_t})")
    print()

    # ---- 4. Multi-channel ECC threading ----
    print("=" * 60)
    print("4. Multi-channel ECC Threading (pass 1 only)")
    print("-" * 60)
    valid_refs = [r for r in u8_crops if r is not None]
    valid_samples = [s for s in sample_u8 if s is not None]
    n_valid = min(len(valid_refs), len(valid_samples))
    print(f"  Valid channels: {n_valid}/{len(rois)}")

    thread_counts = [1, 2, 3, 4, 6, 8, 12]
    ecc_thread_results = bench_ecc_channels_threading(
        valid_refs[:n_valid], valid_samples[:n_valid], thread_counts)
    for n_t, times in ecc_thread_results.items():
        _stats(times, f"  {n_valid}ch x {n_t} threads")
    print()

    # ---- 5. Outer worker sweep (positions in parallel) ----
    print("=" * 60)
    print("5. Full Per-Position Pipeline (2 leaders)")
    print("-" * 60)
    from compute_drift_online import (
        _process_leader_task, load_grid_cal_for_pos,
        scan_grid_positions, read_per_pos_state,
        load_per_pos_kf_state, reconstruct_bg_phase_variants,
        _reconstruct_phase_raw as recon_raw,
    )
    import compute_drift_online as cdo

    # Pre-reconstruct for 2 leaders
    leaders = sample_positions[:2]
    leader_indices = [ld["index"] for ld in leaders]

    # BG reconstruction
    t0 = time.perf_counter()
    bg_phases = reconstruct_bg_phase_variants(bg_raw, cfg, leader_indices)
    t_bg = time.perf_counter() - t0
    print(f"  BG phase recon: {_fmt(t_bg)}")

    # Pre-reconstruct leader phases
    if _HAS_CUPY:
        set_backend("cupy")
    pre_phases = {}
    t0 = time.perf_counter()
    for ld in leaders:
        raw = get_raw(ld["label"])
        if raw.exists():
            pre_phases[ld["index"]] = recon_raw(str(raw), cfg, ld["index"])
    t_pre = time.perf_counter() - t0
    if _HAS_CUPY:
        set_backend("numpy")
    print(f"  Pre-recon {len(pre_phases)} leaders: {_fmt(t_pre)}")

    # Load grid data
    per_pos_rois = {}
    leader_data = {}
    for ld in leaders:
        label = ld["label"]
        rois_p = (grid_dir / f"{label}_x+0_y+0" / "output_phase" /
                  "channels" / "channel_rois.json")
        with open(rois_p, encoding="utf-8") as f:
            per_pos_rois[ld["index"]] = json.load(f)
        _, u8c = load_grid_ref_crops_for_pos(label, cfg, per_pos_rois[ld["index"]])
        pm = scan_grid_positions(str(grid_dir), label)
        gc = load_grid_cal_for_pos(label, cfg)
        leader_data[ld["index"]] = {"u8_crops": u8c, "pos_map": pm, "grid_cal": gc}

    state_path = cfg["state_file"]
    kf_path = cfg.get("kf_state_file", "")
    kf_R = max(cfg.get("kf_R_ty_nm2", 454.0), cfg.get("kf_R_tx_nm2", 454.0))

    # Set up worker global
    cdo._wk = {
        'cfg': cfg, 'per_pos_rois': per_pos_rois,
        'leader_data': leader_data, 'bg_phases': bg_phases, 't': t,
        'pre_phases': pre_phases,
    }

    task_args = [
        (ld["index"], ld["label"], state_path, kf_path, kf_R)
        for ld in leaders
    ]

    # Sequential
    t0 = time.perf_counter()
    for ta in task_args:
        _process_leader_task(ta)
    t_seq = time.perf_counter() - t0
    print(f"  Sequential (2 pos): {_fmt(t_seq)} ({_fmt(t_seq/2)}/pos)")

    # Sweep outer worker counts
    outer_counts = [1, 2, 4, 8, 12, 16, 24]
    print(f"\n  Outer ThreadPool sweep (2 leaders, ecc_threads_per_pos={cfg.get('ecc_threads_per_pos', 1)}):")
    for n_w in outer_counts:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_w) as pool:
            list(pool.map(_process_leader_task, task_args))
        t_par = time.perf_counter() - t0
        print(f"    workers={n_w:2d}: {_fmt(t_par)} ({_fmt(t_par/2)}/pos)")
    print()

    # ---- 6. Summary & Recommendations ----
    print("=" * 60)
    print("6. Summary")
    print("=" * 60)
    n_pos = len(sample_positions)
    n_ch = len(rois)
    interval_sec = cfg.get("interval_sec", 300)

    print(f"  Positions: {n_pos}, Channels: {n_ch}")
    print(f"  Interval: {interval_sec}s")
    print(f"  Budget: {interval_sec}s total, "
          f"~{interval_sec - cfg.get('crop_sub_max_seconds', 60):.0f}s for Phase A")
    print()

    if recon["total_gpu"]:
        t_recon_per = mean(recon["total_gpu"])
    else:
        t_recon_per = mean(recon["total_cpu"])
    t_tilt_per_ch = mean(tilt_times) / n_ch
    t_ecc_per_ch = mean(ecc_times) if valid_ch is not None else 0.1
    t_ecc_3pass = t_ecc_per_ch * 2.5  # avg ~2.5 passes

    print(f"  Per-position breakdown (estimated):")
    print(f"    Reconstruction: {_fmt(t_recon_per)}")
    print(f"    Tilt correct:   {_fmt(mean(tilt_times))}")
    print(f"    to_uint8:       {_fmt(mean(u8_times))}")
    print(f"    ECC ({n_ch}ch x ~2.5pass): {_fmt(t_ecc_3pass * n_ch)}")
    total_per_pos = t_recon_per + mean(tilt_times) + mean(u8_times) + t_ecc_3pass * n_ch
    print(f"    Total/pos:      {_fmt(total_per_pos)}")
    print()
    print(f"  Naive sequential ({n_pos} pos): {_fmt(total_per_pos * n_pos)}")
    print(f"  With 8 workers: ~{_fmt(total_per_pos * n_pos / 8)}")
    print(f"  With 12 workers: ~{_fmt(total_per_pos * n_pos / 12)}")


if __name__ == "__main__":
    main()
