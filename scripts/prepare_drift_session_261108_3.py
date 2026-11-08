"""Prepare the 261108 YE timelapse (2nd grid, grid_YE_2; run 3 = with cells) with cells on the 261108 YE grid (single z, 5 min, 31 d cap).

Raw on E: (NVMe), crop_sub on D:. Grid retaken on 261108 (the 261107 grid was taken at too low a pressure). timelapse.pos z offset 303.35.
"""
import runpy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
module = runpy.run_path(
    str(SCRIPT_DIR / "prepare_drift_session.py"),
    run_name="prepare_261108_3",
)
configurator = module["main"].__globals__

configurator["POSITIONS_FILE"] = r"D:\AquisitionData\Kitagishi\261108\timelapse.pos"
configurator["GRID_DIR"] = r"E:\261108\recon_YE_grid_2"
configurator["GRID_Z_INDEX"] = 0
configurator["SESSION_DIR"] = r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261108_3"
configurator["SAVE_DIR"] = r"E:\261108\ph_YE_5min_31d_3"
configurator["N_TIMEPOINTS"] = 8928
configurator["INTERVAL_SEC"] = 300
configurator["POS_SPLIT"] = 60
configurator["N_Z_SLICES"] = 1
configurator["RAW_TL_Z_INDEX"] = 0
configurator["CROP_SUB_ROOT"] = r"D:\AquisitionData\Kitagishi\261108\online_crop_sub_YE_5min_31d_3"

configurator["main"]()

config_path = Path(configurator["SESSION_DIR"]) / "drift_config_zstack.json"
source = SCRIPT_DIR / "realtime_drift_mda_zstack.bsh"
target = Path(configurator["SESSION_DIR"]) / "realtime_drift_mda_zstack_261108_3.bsh"
text = source.read_text(encoding="utf-8")
old = 'String CONFIG_FILE = "C:/Users/QPI/Documents/QPI_Omni/drift_session_261016/drift_config_zstack.json";'
if old not in text:
    raise RuntimeError("CONFIG_FILE line not found in realtime_drift_mda_zstack.bsh")
text = text.replace(old, f'String CONFIG_FILE = "{config_path.as_posix()}";')
target.write_text(text, encoding="utf-8")
print(f"Micro-Manager script: {target}")
