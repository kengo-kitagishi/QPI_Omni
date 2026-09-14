"""
Benchmark: CuPy (GPU) vs NumPy (CPU) for QPI reconstruction.
Tests single-point and parallel reconstruction to find optimal config.
"""
import os, sys, time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# CUDA_PATH setup
if not os.environ.get("CUDA_PATH"):
    _nvrtc = Path(sys.prefix) / "Lib/site-packages/nvidia/cuda_nvrtc"
    if _nvrtc.exists():
        os.environ["CUDA_PATH"] = str(_nvrtc)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from PIL import Image
from skimage.restoration import unwrap_phase
from qpi import QPIParameters, get_field, set_backend, _HAS_CUPY
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

# ---------- config ----------
GRID_DIR = Path(r"C:\260416\0per_gridgluc_1")
CROP = (0, 2048, 400, 2448)
N_TEST_POINTS = 8  # grid points to benchmark


def _get_test_dirs():
    """Return (target_dirs, bg_dirs) for first N_TEST_POINTS of Pos1."""
    import re
    target_dirs = []
    bg_dirs = []
    pattern = re.compile(r"^Pos1_x([+-]?\d+)_y([+-]?\d+)$")
    for d in sorted(GRID_DIR.iterdir()):
        m = pattern.match(d.name)
        if not m:
            continue
        xi, yi = int(m.group(1)), int(m.group(2))
        bg_name = f"Pos0_x{xi:+d}_y{yi:+d}"
        bg_d = GRID_DIR / bg_name
        if bg_d.exists():
            target_dirs.append(d)
            bg_dirs.append(bg_d)
        if len(target_dirs) >= N_TEST_POINTS:
            break
    return target_dirs, bg_dirs


def _get_z_files(pos_dir):
    return sorted(pos_dir.glob("img_*_ph_*.tif"))


def _recon_one_point_with_backend(backend, target_dir, bg_dir):
    """Reconstruct all z-slices for one grid point, return elapsed time."""
    set_backend(backend)
    z_tgt = _get_z_files(target_dir)
    z_bg = _get_z_files(bg_dir)

    sample = np.array(Image.open(str(z_tgt[0])))
    rs, re_, cs, ce = CROP
    cropped = sample[rs:re_, cs:ce]
    qpi_params = QPIParameters(
        wavelength=WAVELENGTH, NA=NA,
        img_shape=cropped.shape, pixelsize=PIXELSIZE,
        offaxis_center=OFFAXIS_CENTER,
    )

    t0 = time.perf_counter()
    for tgt_path, bg_path in zip(z_tgt, z_bg):
        img_t = np.array(Image.open(str(tgt_path)))[rs:re_, cs:ce]
        field_t = get_field(img_t, qpi_params)
        if hasattr(field_t, "get"):
            angle_t = np.angle(field_t.get())
        else:
            angle_t = np.angle(field_t)
        phase_t = unwrap_phase(angle_t)

        img_b = np.array(Image.open(str(bg_path)))[rs:re_, cs:ce]
        field_b = get_field(img_b, qpi_params)
        if hasattr(field_b, "get"):
            angle_b = np.angle(field_b.get())
        else:
            angle_b = np.angle(field_b)
        phase_b = unwrap_phase(angle_b)

        _ = phase_t - phase_b
    elapsed = time.perf_counter() - t0
    return elapsed


def _worker_numpy(target_dir_str, bg_dir_str):
    """Worker: reconstruct with numpy backend."""
    set_backend("numpy")
    return _recon_one_point_with_backend("numpy",
                                         Path(target_dir_str), Path(bg_dir_str))


def _worker_cupy(target_dir_str, bg_dir_str):
    """Worker: reconstruct with cupy backend."""
    try:
        set_backend("cupy")
    except Exception:
        set_backend("numpy")
    return _recon_one_point_with_backend("cupy" if _HAS_CUPY else "numpy",
                                         Path(target_dir_str), Path(bg_dir_str))


def main():
    target_dirs, bg_dirs = _get_test_dirs()
    n = len(target_dirs)
    print(f"Test data: {n} grid points from Pos1, {len(_get_z_files(target_dirs[0]))} z-slices each")
    print(f"Crop: {CROP}")
    print()

    # --- Test A: single point, numpy ---
    print("=" * 50)
    print("Test A: single point, numpy")
    set_backend("numpy")
    t = _recon_one_point_with_backend("numpy", target_dirs[0], bg_dirs[0])
    print(f"  {t:.2f} s")

    # --- Test B: single point, cupy ---
    if _HAS_CUPY:
        print("=" * 50)
        print("Test B: single point, cupy (incl. warmup)")
        # warmup
        set_backend("cupy")
        import cupy as cp
        cp.array([1.0]) * 2
        _recon_one_point_with_backend("cupy", target_dirs[0], bg_dirs[0])
        # actual
        t = _recon_one_point_with_backend("cupy", target_dirs[0], bg_dirs[0])
        print(f"  {t:.2f} s (after warmup)")
    else:
        print("Test B: SKIPPED (no CuPy)")

    # --- Test C: parallel, numpy, 16 workers ---
    print("=" * 50)
    print(f"Test C: {n} points parallel, numpy, 16 workers")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(_worker_numpy, str(td), str(bd))
                for td, bd in zip(target_dirs, bg_dirs)]
        times = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    print(f"  Wall: {wall:.2f} s | Per-worker avg: {np.mean(times):.2f} s")

    # --- Test D: parallel, cupy, 4 workers ---
    if _HAS_CUPY:
        print("=" * 50)
        print(f"Test D: {n} points parallel, cupy, 4 workers")
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_worker_cupy, str(td), str(bd))
                    for td, bd in zip(target_dirs, bg_dirs)]
            times = [f.result() for f in futs]
        wall = time.perf_counter() - t0
        print(f"  Wall: {wall:.2f} s | Per-worker avg: {np.mean(times):.2f} s")

        # --- Test E: parallel, cupy, 8 workers ---
        print("=" * 50)
        print(f"Test E: {n} points parallel, cupy, 8 workers")
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(_worker_cupy, str(td), str(bd))
                    for td, bd in zip(target_dirs, bg_dirs)]
            times = [f.result() for f in futs]
        wall = time.perf_counter() - t0
        print(f"  Wall: {wall:.2f} s | Per-worker avg: {np.mean(times):.2f} s")

        # --- Test F: parallel, cupy, 16 workers ---
        print("=" * 50)
        print(f"Test F: {n} points parallel, cupy, 16 workers")
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(_worker_cupy, str(td), str(bd))
                    for td, bd in zip(target_dirs, bg_dirs)]
            times = [f.result() for f in futs]
        wall = time.perf_counter() - t0
        print(f"  Wall: {wall:.2f} s | Per-worker avg: {np.mean(times):.2f} s")
    else:
        print("Tests D/E/F: SKIPPED (no CuPy)")

    print()
    print("=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
