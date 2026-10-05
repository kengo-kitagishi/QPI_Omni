"""Reconstruct and calibrate the completed single-z 261004 grid.

  raw     E:\261004\2%_grid_hologram_1              Pos0..Pos109, 121 points, 1 z
  output  D:\AquisitionData\Kitagishi\261004\2%_grid_hologram_1

POS_SPLIT = 56: timelapse.pos runs Pos1..Pos55 with X falling 6535 -> -3186 um,
then reverses at Pos56 (X -3224 -> 6502, Y 518 -> 6), so the traps are mirrored
from Pos56 on. Same rule gave 57 for 260927.
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
impl.WATCH_DIR = Path(r"E:\261004\2%_grid_hologram_1")
impl.OUTPUT_DIR = Path(r"D:\AquisitionData\Kitagishi\261004\2%_grid_hologram_1")
impl.BATCHES[0]["grid_dir"] = impl.WATCH_DIR
impl.BATCHES[0]["output_dir"] = impl.OUTPUT_DIR
impl.BATCHES[0]["delete_raw"] = False
impl.GRID_HALF = 5
impl.N_Z = 1
impl.Z_INDEX = 0
impl.POS_SPLIT = 56
impl.LAST_POS = 109
impl.RECON_Z_INDICES = [0]
impl.LOG_PATH = impl.SESSION_DIR / "recon_calib_grid.log"
impl.ALERT_PATH = impl.SESSION_DIR / "recon_calib_grid.ALERT.txt"

if __name__ == "__main__":
    impl.main()
