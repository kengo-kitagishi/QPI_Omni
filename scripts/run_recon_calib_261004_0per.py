"""Reconstruct and calibrate the 261004 single-z 0% glucose grid.

  raw     E:\261004\0%_grid_hologram_1              Pos0..Pos109, 121 points, 1 z
  output  D:\AquisitionData\Kitagishi\261004\0%_grid_hologram_1

Same geometry as the 2% grid of the same session (run_recon_calib_261004.py):
POS_SPLIT 56 from the timelapse.pos reversal at Pos56, one z at the working
focus. Only the paths differ.

Default run waits for the acquisition to go idle (10 min with no new point)
before touching anything, so it is safe to start while the grid is still being
written. Pass --skip-wait once the grid is known to be finished.

IMPORTANT: reconstruction reads the raw from E:, which is also the timelapse
save disk. Run it BEFORE resuming the timelapse, or not until the timelapse is
stopped -- two acquisitions' worth of I/O on one disk has dropped frames before.
"""
import importlib.util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "watch_grid_impl", SCRIPT_DIR / "watch_grid_then_recon_260922_sz.py"
)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)

impl.SESSION_DIR = Path(r"E:\261004")
impl.WATCH_DIR = Path(r"E:\261004\0%_grid_hologram_1")
impl.OUTPUT_DIR = Path(r"D:\AquisitionData\Kitagishi\261004\0%_grid_hologram_1")
impl.BATCHES[0]["grid_dir"] = impl.WATCH_DIR
impl.BATCHES[0]["output_dir"] = impl.OUTPUT_DIR
impl.BATCHES[0]["delete_raw"] = False
impl.GRID_HALF = 5
impl.N_Z = 1
impl.Z_INDEX = 0
impl.POS_SPLIT = 56
impl.LAST_POS = 109
impl.RECON_Z_INDICES = [0]
impl.LOG_PATH = impl.SESSION_DIR / "recon_calib_grid_0per.log"
impl.ALERT_PATH = impl.SESSION_DIR / "recon_calib_grid_0per.ALERT.txt"

if __name__ == "__main__":
    impl.main()
