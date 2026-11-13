"""_qc_stride_test_261112.py

Did the 2D-stride visiting order spread the grid points?  Reconstructs and calibrates the
test acquisition (Pos0 + Pos82, E:\\261112\\grid_2per_stridetest_1) into a SEPARATE output
folder with the production pipeline (scheduled_recon_and_calibrate.py, gaussian2d), then
compares the measured point positions with the sequential (unidirectional) acquisition of
the same Pos (D:\\...\\recon_0p03um_19x19\\grid_calibration_Pos82.json):

  nearest-neighbour distance between measured points (what the residual warp can reach)
  coverage: fraction of a fine mesh over the nominal square within 0.05 / 0.1 px of a point
  per-step gain along each axis and the scatter around an ideal lattice

    python scripts/_qc_stride_test_261112.py
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import figure_logger  # noqa: F401
from figure_logger import save_figure
import matplotlib.pyplot as plt

TEST_RAW = Path(r"E:\261112\grid_2per_stridetest_1")
TEST_OUT = Path(r"D:\AquisitionData\Kitagishi\261112\recon_stridetest_1")
SEQ_OUT = Path(r"D:\AquisitionData\Kitagishi\261112\recon_0p03um_19x19")
POS = 82
POS_SPLIT = 60
PX_UM = 0.34567514677103717
NOMINAL_STEP_PX = 0.03 / PX_UM


def patch_config(path, assignments):
    p = SCRIPT_DIR / path
    text = p.read_text(encoding="utf-8")
    for name, value in assignments.items():
        pat = re.compile(rf"^({re.escape(name)}\s*=\s*)(.*?)(\s*(?:#.*)?)$", re.M)
        m = pat.search(text)
        if not m:
            raise RuntimeError(f"{path}: {name} not found")
        text = pat.sub(lambda mm: f"{mm.group(1)}{value}{mm.group(3)}", text, count=1)
    p.write_text(text, encoding="utf-8")


def load_points(cal_path):
    pts = json.load(open(cal_path, encoding="utf-8"))["positions"]
    xi = np.array([e["xi"] for e in pts], float)
    yi = np.array([e["yi"] for e in pts], float)
    ax = np.array([-e["actual_dx_px"] for e in pts])
    ay = np.array([-e["actual_dy_px"] for e in pts])
    return xi, yi, ax, ay


def describe(tag, xi, yi, ax, ay):
    P = np.column_stack([ax, ay])
    d, _ = cKDTree(P).query(P, k=2)
    nn = d[:, 1]
    M = np.column_stack([xi, yi, np.ones_like(xi)])
    cx = np.linalg.lstsq(M, ax, rcond=None)[0]
    cy = np.linalg.lstsq(M, ay, rcond=None)[0]
    rx = ax - M @ cx
    ry = ay - M @ cy
    half = 9 * NOMINAL_STEP_PX
    g = np.linspace(-half, half, 61)
    gx, gy = np.meshgrid(g, g)
    dm, _ = cKDTree(P).query(np.column_stack([gx.ravel(), gy.ravel()]))
    print(f"{tag}:")
    print(f"  nearest-neighbour distance between measured points: p10 {np.percentile(nn,10):.3f}  p50 {np.median(nn):.3f}  p90 {np.percentile(nn,90):.3f} px")
    print(f"  mesh over the nominal +-{half:.2f} px square: within 0.05 px of a point {np.mean(dm<0.05)*100:4.1f}%, within 0.1 px {np.mean(dm<0.1)*100:4.1f}%, median distance {np.median(dm):.3f} px")
    print(f"  per-step gain: x/yi {cx[1]:+.3f} px/step, y/xi {cy[0]:+.3f} px/step (nominal {NOMINAL_STEP_PX:.3f}); scatter around the lattice x {rx.std():.3f} y {ry.std():.3f} px")
    print(f"  extent: x {ax.min():+.2f}..{ax.max():+.2f}, y {ay.min():+.2f}..{ay.max():+.2f} px")
    return nn, dm


def main():
    t0 = time.perf_counter()
    cal_test = TEST_OUT / f"grid_calibration_Pos{POS}.json"
    if not cal_test.exists():
        n = sum(1 for d in TEST_RAW.iterdir() if d.is_dir())
        print(f"{TEST_RAW}: {n} point folders; running recon + detect + calibration into {TEST_OUT}")
        TEST_OUT.mkdir(parents=True, exist_ok=True)
        patch_config("batch_reconstruction_grid.py", {"GRID_DIR": f'r"{TEST_RAW}"', "POS_SPLIT": str(POS_SPLIT), "Z_INDICES": "None"})
        patch_config("parallel_calibrate.py", {"GRID_DIR": f'Path(r"{TEST_OUT}")', "Z_INDEX": "0", "POS_SPLIT": str(POS_SPLIT)})
        patch_config("scheduled_recon_and_calibrate.py", {"GRID_DIR": f'Path(r"{TEST_RAW}")', "OUTPUT_DIR": f'Path(r"{TEST_OUT}")',
                                                         "Z_INDEX": "0", "POS_SPLIT": str(POS_SPLIT), "SKIP_RECON": "False"})
        rc = subprocess.run([sys.executable, str(SCRIPT_DIR / "scheduled_recon_and_calibrate.py")], cwd=str(SCRIPT_DIR)).returncode
        if rc != 0 or not cal_test.exists():
            raise SystemExit(f"pipeline failed (exit {rc}) or no calibration written")
        print(f"pipeline done in {(time.perf_counter()-t0)/60:.1f} min")

    seq = load_points(SEQ_OUT / f"grid_calibration_Pos{POS}.json")
    tst = load_points(cal_test)
    nn_s, dm_s = describe("sequential (unidirectional rows, 30 nm steps)", *seq)
    nn_t, dm_t = describe("2D stride (every move >= 0.15 um per axis)", *tst)

    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 80 / 25.4))
    for ax, (xi, yi, px, py), tag in ((axes[0], seq, "sequential"), (axes[1], tst, "2D stride")):
        ax.scatter(px, py, s=6, c="0.2", linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlabel("measured x [px]")
        ax.set_ylabel("measured y [px]")
        ax.set_title(f"{tag}: NN distance p50 {np.median(cKDTree(np.column_stack([px, py])).query(np.column_stack([px, py]), k=2)[0][:, 1]):.3f} px", fontsize=7)
    fig.suptitle(f"261112 Pos{POS}: measured positions of the 361 grid points (gaussian2d calibration)", fontsize=7)
    fig.tight_layout()
    save_figure(fig, params={"pos": POS, "test_raw": str(TEST_RAW), "seq_out": str(SEQ_OUT)},
                description="Measured (calibrated) positions of the 361 grid points of one Pos acquired in the unidirectional row order with 30 nm steps versus the 2D stride order where both stage axes move at least 0.15 um at every point.",
                data={"seq_x": seq[2], "seq_y": seq[3], "stride_x": tst[2], "stride_y": tst[3], "nn_seq": nn_s, "nn_stride": nn_t},
                fmt="png")


if __name__ == "__main__":
    main()
