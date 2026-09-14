# -*- coding: utf-8 -*-
"""check_env.py - smoke test of the analysis environment.

    python environment/check_env.py [--model models/<checkpoint>] [--cpu-ok]

Exit 0 when every pinned module imports at the pinned version, torch sees a CUDA device
(unless --cpu-ok), and, if --model is given, the checkpoint matches its sha256 in
models/MODELS.json, loads, and runs one inference on a synthetic trap image.
"""
from __future__ import annotations

import argparse
import base64
import os
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# module -> version prefix pinned in environment/ (2026-09-14 export)
PINS = {
    "torch": "2.5.1", "omnipose": "1.0.6", "cellpose_omni": "1.0.6", "numpy": "1.25.2",
    "scipy": "1.10.1", "skimage": "0.25.2", "pandas": "2.3.2", "tifffile": "2023.4.12",
    "cv2": "4.12.0", "matplotlib": "3.10.6", "yaml": "6.0.2", "numba": "0.61.2",
}
EVAL = dict(channels=None, channel_axis=None, diameter=20, normalize=True, tile=False, omni=True,
            verbose=False, flow_threshold=0.11, mask_threshold=0, min_size=10, net_avg=False)


# cellpose_omni imports its GUI module when the package is imported, and that module downloads GUI
# assets into ~/.omnipose/ whenever they are missing. Two of those URLs answer 404 (seen 2026-09-15 on
# a fresh PC: gui/logo.png and gui/gamma.svg upstream), so `import cellpose_omni` itself raises. Both
# files only decorate the GUI window: 1x1 stand-ins let the import through. The test images it also
# fetches (docs/test_files/) still download. Same guard in environment/bootstrap_windows.ps1,
# environment/check_env.py and scripts/seg_omnipose.py.
_GUI_ASSETS_B64 = {
    "logo.png": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
    "gamma.svg": "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxIiBoZWlnaHQ9IjEiLz4=",
}


def _ensure_gui_icon() -> None:
    d = Path.home() / ".omnipose"
    for name, b64 in _GUI_ASSETS_B64.items():
        f = d / name
        if not f.is_file():
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(base64.b64decode(b64))


def _ensure_cuda_dlls_on_path() -> None:
    """Put the env's bin/ on PATH so NVRTC can load nvrtc-builtins64_*.dll.

    From the second inference on, torch's TorchScript fuser compiles CUDA kernels at run time
    with NVRTC, and NVRTC finds its builtins DLL through PATH. The conda env keeps that DLL in
    <env>/bin, which is on PATH only after `conda activate`; the kit starts <env>/python.exe
    directly, so without this every inference after the first raised "failed to open
    nvrtc-builtins64_118.dll" (seen 2026-09-15 on the microscope PC). Same helper in
    environment/check_env.py and scripts/seg_omnipose.py.
    """
    prefix = Path(sys.prefix)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    for d in (prefix / "Library" / "bin", prefix / "bin"):
        if d.is_dir() and str(d) not in parts:
            parts.insert(0, str(d))
    os.environ["PATH"] = os.pathsep.join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="checkpoint path (relative paths resolve from the repo root)")
    ap.add_argument("--cpu-ok", action="store_true", help="do not fail when no CUDA device is visible")
    a = ap.parse_args()
    ok = True
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    _ensure_gui_icon()
    _ensure_cuda_dlls_on_path()
    for m, want in PINS.items():
        try:
            mod = __import__(m)
            got = str(getattr(mod, "__version__", "?"))
            flag = "ok" if got.startswith(want) else "VERSION MISMATCH"
            ok &= flag == "ok"
            print(f"  {m:14s} {got:12s} pinned {want:10s} {flag}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  {m:14s} MISSING: {e!r}")

    import torch
    cuda = torch.cuda.is_available()
    print(f"torch.cuda.is_available() = {cuda}; torch.version.cuda = {torch.version.cuda}")
    if cuda:
        print(f"  device: {torch.cuda.get_device_name(0)}")
    elif not a.cpu_ok:
        ok = False
        print("  NO CUDA DEVICE: Omnipose segmentation must run on the GPU (CPU masks differ)")

    if a.model:
        p = Path(a.model)
        if not p.is_absolute():
            p = REPO / p
        if not p.exists():
            print(f"model not found: {p}")
            return 1
        reg = REPO / "models" / "MODELS.json"
        entry = None
        if reg.exists():
            entry = next((e for e in json.loads(reg.read_text(encoding="utf-8"))["models"]
                          if e["file"] == p.name), None)
        if entry:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            same = h == entry["sha256"]
            ok &= same
            print(f"model {p.name}: sha256 {'ok' if same else 'MISMATCH'} (trained {entry.get('trained')})")
        else:
            print(f"model {p.name}: not listed in models/MODELS.json (hash not checked)")
        import numpy as np
        from cellpose_omni.models import CellposeModel
        t0 = time.time()
        model = CellposeModel(gpu=cuda, pretrained_model=str(p), omni=True, nchan=1, nclasses=3, dim=2)
        print(f"model loaded in {time.time() - t0:.1f}s")
        # synthetic trap crop (40 x 180 px): two rod-shaped cells of about 1.5 rad on a flat background
        img = np.zeros((40, 180), np.float32)
        yy, xx = np.mgrid[:40, :180]
        for cx in (30, 70):
            img[((xx - cx) / 14.0) ** 2 + ((yy - 20) / 5.0) ** 2 <= 1.0] = 1.5
        img += np.random.default_rng(0).normal(0, 0.05, img.shape).astype(np.float32)
        # Three passes: the TorchScript fuser compiles CUDA kernels with NVRTC only from the second
        # call on, so a single inference passes even when NVRTC cannot load its builtins DLL.
        for k in range(3):
            t0 = time.time()
            m = model.eval([img], **EVAL)[0][0]
            n = int(np.asarray(m).max()) if m is not None else 0
            print(f"inference {k + 1}/3 on a synthetic 40x180 image: {n} mask(s) in {time.time() - t0:.2f}s"
                  + ("" if n else "  (0 masks is not an error on synthetic input; real crops are the test)"))

    print("ENVIRONMENT OK" if ok else "ENVIRONMENT CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
