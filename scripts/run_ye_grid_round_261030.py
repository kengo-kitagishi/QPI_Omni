"""One round of the 261030 YE grid: wait for the round's acquisition on D:, recon + calibrate, delete its raw.

    python run_ye_grid_round_261030.py <round>      (rounds as in split_grid_pos_261030_ye.py)

Raw: D:\\AquisitionData\\Kitagishi\\261029\\grid_YE_<round>; output (shared by all rounds):
D:\\AquisitionData\\Kitagishi\\261029\\recon_YE_grid. Rounds 2-4 have no Pos0 raw; reconstruction takes
the BG from round 1's output (watch_grid_then_recon_260922_sz.bg_already_reconstructed).
While the acquisition runs only the top folder is listed, so the watcher does not compete with it.
"""
import importlib.util
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from split_grid_pos_261030_ye import ROUNDS  # noqa: E402

ROUND = int(sys.argv[1])
FIRST, LAST = ROUNDS[ROUND]
SESSION = Path(r"D:\AquisitionData\Kitagishi\261029")
WATCH = SESSION / f"grid_YE_{ROUND}"       # the name MM gave round 1
if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
    WATCH = Path(sys.argv[2])                # round 2 went to C:\261029_YE_grid\grid_YE_1
N_EXPECTED = (LAST - FIRST + 1) * 121
IDLE_MIN = 10

spec = importlib.util.spec_from_file_location(
    "watch_grid_impl", SCRIPT_DIR / "watch_grid_then_recon_260922_sz.py"
)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)

impl.SESSION_DIR = SESSION
impl.WATCH_DIR = WATCH
impl.OUTPUT_DIR = SESSION / "recon_YE_grid"
impl.BATCHES[0]["grid_dir"] = impl.WATCH_DIR
impl.BATCHES[0]["output_dir"] = impl.OUTPUT_DIR
impl.BATCHES[0]["delete_raw"] = True          # user: raw goes once recon + calib are verified
impl.GRID_HALF = 5
impl.N_Z = 1
impl.Z_INDEX = 0
impl.POS_SPLIT = 59
impl.LAST_POS = LAST
impl.RECON_Z_INDICES = None
impl.MIN_FREE_GB_OUT = 10     # output is 0.32 GB/Pos; the round sizes leave room for it on D:
impl.MIN_FREE_GB_RAW = 5
impl.LOG_PATH = SESSION / f"recon_YE_grid_r{ROUND}.log"
impl.ALERT_PATH = SESSION / f"recon_YE_grid_r{ROUND}.ALERT.txt"


def wait_light():
    last_n, last_change = -1, time.time()
    while True:
        n = sum(1 for e in os.scandir(WATCH) if e.is_dir()) if WATCH.is_dir() else 0
        if n != last_n:
            last_n, last_change = n, time.time()
            impl.log(f"round {ROUND}: {n}/{N_EXPECTED} point folders")
        if n >= N_EXPECTED and (time.time() - last_change) / 60 >= IDLE_MIN:
            return
        time.sleep(60)


def add_empty_bg_dirs():
    """batch_reconstruction_grid.py insists on Pos0_x*_y* folders in the raw dir, but reads the BG
    phase from the output tree when it is there (round 1 put it there). Empty folders satisfy the check."""
    for xi in range(-5, 6):
        for yi in range(-5, 6):
            (WATCH / f"Pos0_x{xi:+d}_y{yi:+d}").mkdir(exist_ok=True)


if __name__ == "__main__":
    if "--now" in sys.argv:          # acquisition known to be finished
        impl.log(f"round {ROUND}: Pos{FIRST}-{LAST} in {WATCH}, starting now")
    else:
        impl.log(f"round {ROUND}: Pos{FIRST}-{LAST} -> waiting for {WATCH} "
                 f"({datetime.now():%H:%M}, idle {IDLE_MIN} min ends the wait)")
        wait_light()
    if FIRST > 0:
        add_empty_bg_dirs()
    sys.argv = [sys.argv[0], "--skip-wait"]
    impl.main()
