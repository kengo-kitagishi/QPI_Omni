"""Wait for the 261029 2% grid (Pos0-118, single z) to finish, recon + calibrate it, then delete the raw."""
import importlib.util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "watch_grid_impl", SCRIPT_DIR / "watch_grid_then_recon_260922_sz.py"
)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)

impl.SESSION_DIR = Path(r"E:\261029")
impl.WATCH_DIR = Path(r"E:\261029\grid_2per_1")
impl.OUTPUT_DIR = Path(r"E:\261029\recon_2per_grid_1")
impl.BATCHES[0]["grid_dir"] = impl.WATCH_DIR
impl.BATCHES[0]["output_dir"] = impl.OUTPUT_DIR
impl.BATCHES[0]["delete_raw"] = True          # user: delete the holograms after recon + calib
impl.GRID_HALF = 5
impl.N_Z = 1
impl.Z_INDEX = 0
impl.POS_SPLIT = 59           # timelapse.pos: second row starts at Pos59
impl.LAST_POS = 118
impl.RECON_Z_INDICES = None   # = every plane; the grid has only one, so the raw may be deleted after verify
impl.MIN_FREE_GB_OUT = 40
impl.LOG_PATH = impl.SESSION_DIR / "recon_calib_2per_grid.log"
impl.ALERT_PATH = impl.SESSION_DIR / "recon_calib_2per_grid.ALERT.txt"

if __name__ == "__main__":
    impl.main()
