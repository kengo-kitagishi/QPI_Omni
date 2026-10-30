"""Prepare the 261030 YE timelapse on the YE grid (single z, 5 min, 31 d cap).

Grid: D:/.../261029/recon_YE_grid (acquired in 4 rounds on 2026-10-30, Pos0-118).
Raw goes to E: (394 GB free after 260928 went back to F:); crop_sub to D: so the two writers use different disks.
"""
import runpy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
module = runpy.run_path(
    str(SCRIPT_DIR / "prepare_drift_session.py"),
    run_name="prepare_261030_ye",
)
configurator = module["main"].__globals__

configurator["POSITIONS_FILE"] = r"D:\AquisitionData\Kitagishi\261029\timelapse.pos"
configurator["GRID_DIR"] = r"D:\AquisitionData\Kitagishi\261029\recon_YE_grid"
configurator["GRID_Z_INDEX"] = 0
configurator["SESSION_DIR"] = r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261030_YE"
configurator["SAVE_DIR"] = r"E:\261030\ph_YE_5min_31d"
configurator["N_TIMEPOINTS"] = 8928
configurator["INTERVAL_SEC"] = 300
configurator["POS_SPLIT"] = 59
configurator["N_Z_SLICES"] = 1
configurator["RAW_TL_Z_INDEX"] = 0
configurator["CROP_SUB_ROOT"] = r"D:\AquisitionData\Kitagishi\261030\online_crop_sub_YE_5min_31d"

configurator["main"]()

config_path = Path(configurator["SESSION_DIR"]) / "drift_config_zstack.json"
source = SCRIPT_DIR / "realtime_drift_mda_zstack.bsh"
target = Path(configurator["SESSION_DIR"]) / "realtime_drift_mda_zstack_261030_YE.bsh"
text = source.read_text(encoding="utf-8")
old = 'String CONFIG_FILE = "C:/Users/QPI/Documents/QPI_Omni/drift_session_261016/drift_config_zstack.json";'
if old not in text:
    raise RuntimeError("CONFIG_FILE line not found in realtime_drift_mda_zstack.bsh")
text = text.replace(old, f'String CONFIG_FILE = "{config_path.as_posix()}";')
target.write_text(text, encoding="utf-8")
print(f"Micro-Manager script: {target}")
