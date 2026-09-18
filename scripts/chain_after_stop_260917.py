"""Wait for the running timelapse to stop, then set up the next one and build the QC pages.

    you press Stop on the MDA
        |   this script sees compute_drift_online gone and the crop_sub tree quiet
    prepare_drift_session.py   -> drift_config_zstack.json for ph_zstack_test_3
        |                         (interval 60 s so it overruns, raw holograms KEPT)
        |   BEEP + START_TIMELAPSE.txt: you press Run
    wait for FRAMES_WANTED frames
        |
    channel_contact_sheet.py x2  -> inferno 0-1.8 and +-0.2 rad HTML, opened

The config file is only rewritten once the old run is really gone: compute_drift_online reads
it every frame, so rewriting it mid-run would change the settings under a live acquisition.
"""
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
SESSION = Path(r"E:\260917")
OLD_CROP_SUB = SESSION / "online_crop_sub_zstack_test_2"
NEW_CROP_SUB = SESSION / "online_crop_sub_zstack_test_3"
PROBE_CH = NEW_CROP_SUB / "Pos1" / "output_phase" / "channels" / "crop_sub_rawraw" / "z000" / "ch00"
QC_DIR = SESSION / "_qc"
QUIET_MIN = 6.0          # longer than one cycle (~2.5-3 min), so a mid-cycle gap is not "stopped"
FRAMES_WANTED = 15
FRAME_FOR_SHEET = 14
POLL_SEC = 30
LOG = SESSION / "chain_after_stop.log"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def beep(n=6):
    try:
        import winsound
        for _ in range(n):
            winsound.Beep(880, 400)
            time.sleep(0.15)
    except Exception:
        pass


def drift_workers():
    """How many compute_drift_online processes are alive (0 = the MDA is not stepping)."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
         "Where-Object { $_.CommandLine -like '*compute_drift_online*' } | Measure-Object).Count"],
        capture_output=True, text=True).stdout.strip()
    try:
        return int(out)
    except ValueError:
        return -1


def newest_write_age_min(tree):
    newest = 0.0
    for p in tree.glob("Pos*/output_phase/channels/crop_sub_rawraw/z000/ch00/*.tif"):
        m = p.stat().st_mtime
        if m > newest:
            newest = m
    return (time.time() - newest) / 60 if newest else 1e9


def wait_for_stop():
    log(f"waiting for the running timelapse to stop "
        f"(no compute_drift_online and {QUIET_MIN:.0f} min without a new crop_sub file)")
    while True:
        n = drift_workers()
        age = newest_write_age_min(OLD_CROP_SUB)
        if n == 0 and age >= QUIET_MIN:
            log(f"stopped: no worker, last crop_sub write {age:.1f} min ago")
            return
        time.sleep(POLL_SEC)


def run(script, *args):
    log(f"running {script} {' '.join(args)}")
    rc = subprocess.run([PYTHON, str(SCRIPT_DIR / script), *args], cwd=str(SCRIPT_DIR)).returncode
    log(f"{script} exit {rc}")
    return rc == 0


def main():
    log("=" * 70)
    log("chain_after_stop_260917 start")
    wait_for_stop()
    if not run("prepare_drift_session.py"):
        log("prepare_drift_session failed")
        beep()
        sys.exit(3)
    msg = ("ph_zstack_test_3 is ready: interval 60 s (it will overrun and cycle back to back), "
           "raw holograms are KEPT at 1.0 GB per cycle. Press Run on "
           "realtime_drift_mda_zstack.bsh (CONFIG_FILE = "
           r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config_zstack.json).")
    (SESSION / "START_TIMELAPSE.txt").write_text(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n", encoding="utf-8")
    log(msg)
    beep()
    log(f"waiting for {FRAMES_WANTED} frames in {PROBE_CH}")
    while True:
        n = len(list(PROBE_CH.glob("img_*_ph_000.tif"))) if PROBE_CH.is_dir() else 0
        if n >= FRAMES_WANTED:
            log(f"{n} frames present")
            break
        time.sleep(POLL_SEC)
    for vmin, vmax, tag in ((0, 1.8, "p18"), (-0.2, 0.2, "pm02")):
        out = QC_DIR / f"test3_f{FRAME_FOR_SHEET}_{tag}.html"
        if run("channel_contact_sheet.py", "--raw-root", str(NEW_CROP_SUB),
               "--channel-rel", "output_phase/channels/crop_sub_rawraw/z000",
               "--phase-glob", "img_*_ph_000.tif", "--frame", str(FRAME_FOR_SHEET),
               "--vmin", str(vmin), "--vmax", str(vmax), "--out", str(out)) and out.exists():
            os.startfile(str(out))          # noqa: S606  (our own QC page)
            log(f"opened {out.name}")
    beep(3)


if __name__ == "__main__":
    main()
