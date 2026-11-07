"""Wait for the 261107 YE grid (Pos0-117, single z) on E: to finish, then recon + calibrate it. Raw is kept.

While the grid is acquired only the top folder is listed (no per-point folder opens).
Done = all 118 x 121 point folders present and no new one for IDLE_MIN minutes.
POS_SPLIT 58: in timelapse.pos the second row starts at Pos58 (y -6100 -> -6500 um).
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "watch_grid_impl", SCRIPT_DIR / "watch_grid_then_recon_260922_sz.py"
)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)

impl.SESSION_DIR = Path(r"E:\261107")
impl.WATCH_DIR = Path(r"E:\261107\grid_YE_1")
impl.OUTPUT_DIR = Path(r"E:\261107\recon_YE_grid_1")
impl.BATCHES[0]["grid_dir"] = impl.WATCH_DIR
impl.BATCHES[0]["output_dir"] = impl.OUTPUT_DIR
impl.BATCHES[0]["delete_raw"] = False         # raw kept (not asked to delete)
impl.GRID_HALF = 5
impl.N_Z = 1
impl.Z_INDEX = 0
impl.POS_SPLIT = 58
impl.LAST_POS = 117
impl.RECON_Z_INDICES = None
impl.MIN_FREE_GB_OUT = 40
impl.LOG_PATH = impl.SESSION_DIR / "recon_calib_YE_grid.log"
impl.ALERT_PATH = impl.SESSION_DIR / "recon_calib_YE_grid.ALERT.txt"

N_POINTS = 118 * 121
IDLE_MIN = 10


def wait_light():
    last_n, last_change = -1, time.time()
    while True:
        n = sum(1 for e in os.scandir(impl.WATCH_DIR) if e.is_dir()) if impl.WATCH_DIR.is_dir() else 0
        if n != last_n:
            last_n, last_change = n, time.time()
            impl.log(f"{n}/{N_POINTS} point folders")
        if n >= N_POINTS and (time.time() - last_change) / 60 >= IDLE_MIN:
            return
        time.sleep(60)


if __name__ == "__main__":
    if "--now" not in sys.argv:          # --now: acquisition known to be finished
        impl.log(f"waiting for {impl.WATCH_DIR} (top-folder count only, idle {IDLE_MIN} min ends the wait)")
        wait_light()
    sys.argv = [sys.argv[0], "--skip-wait"]
    impl.main()
