"""Everything between the single-z grid finishing and the QC pages, unattended.

    [watch_grid_then_recon_260917_sz.py]  grid -> recon -> channel detect -> calibrate
        |   this script waits for that to report OK
    prepare_drift_session.py              -> drift_config_zstack.json for ph_zstack_test_2
        |   BEEP + START_TIMELAPSE.txt: the ONE manual step is pressing Run on
        |   realtime_drift_mda_zstack.bsh in the Micro-Manager script panel -- MM's script
        |   panel cannot be driven from outside, so nothing here can press it.
    wait for FRAMES_WANTED frames of online crop_sub
        |
    channel_contact_sheet.py x2           -> inferno 0-1.8 and +-0.2 rad HTML, opened in the browser

The tilt correction stays exactly as production does it (1D line on the aperture-end third
inside grid_subtract). No plane, no 2D flatten: those were offline re-renders only.
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
WATCH_LOG = SESSION / "watch_grid_then_recon_sz.log"
# the --dry-run of the watcher also prints "batch b1: OK", so key on the line that
# only a real, verified run can write
GRID_OK = "batch b1: verification passed"
CROP_SUB = SESSION / "online_crop_sub_zstack_test_2"
PROBE_CH = CROP_SUB / "Pos1" / "output_phase" / "channels" / "crop_sub_rawraw" / "z000" / "ch00"
QC_DIR = SESSION / "_qc"
FRAMES_WANTED = 15
FRAME_FOR_SHEET = 14
POLL_SEC = 60
LOG = SESSION / "chain_after_sz_grid.log"


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


def wait_for_grid():
    log(f"waiting for {WATCH_LOG.name} to report '{GRID_OK}'")
    while True:
        if WATCH_LOG.exists() and GRID_OK in WATCH_LOG.read_text(encoding="utf-8", errors="replace"):
            log("grid recon + calibration reported OK")
            return True
        if WATCH_LOG.exists():
            txt = WATCH_LOG.read_text(encoding="utf-8", errors="replace")
            if "FAILED" in txt:
                log("the grid pipeline reported FAILED -- stopping here, nothing is prepared")
                beep()
                return False
        time.sleep(POLL_SEC)


def run(script, *args):
    log(f"running {script} {' '.join(args)}")
    rc = subprocess.run([PYTHON, str(SCRIPT_DIR / script), *args], cwd=str(SCRIPT_DIR)).returncode
    log(f"{script} exit {rc}")
    return rc == 0


def wait_for_frames():
    log(f"waiting for {FRAMES_WANTED} frames in {PROBE_CH}")
    while True:
        n = len(list(PROBE_CH.glob("img_*_ph_000.tif"))) if PROBE_CH.is_dir() else 0
        if n >= FRAMES_WANTED:
            log(f"{n} frames present")
            return
        time.sleep(POLL_SEC)


def sheets():
    QC_DIR.mkdir(parents=True, exist_ok=True)
    outs = []
    for vmin, vmax, tag in ((0, 1.8, "p18"), (-0.2, 0.2, "pm02")):
        out = QC_DIR / f"test2_f{FRAME_FOR_SHEET}_{tag}.html"
        ok = run("channel_contact_sheet.py",
                 "--raw-root", str(CROP_SUB),
                 "--channel-rel", "output_phase/channels/crop_sub_rawraw/z000",
                 "--phase-glob", "img_*_ph_000.tif",
                 "--frame", str(FRAME_FOR_SHEET),
                 "--vmin", str(vmin), "--vmax", str(vmax),
                 "--out", str(out))
        if ok and out.exists():
            outs.append(out)
            os.startfile(str(out))       # noqa: S606  (opening our own QC page)
    return outs


def main():
    log("=" * 70)
    log("chain_after_sz_grid_260917 start")
    if not wait_for_grid():
        sys.exit(2)
    if not run("prepare_drift_session.py"):
        log("prepare_drift_session failed -- not waiting for frames")
        beep()
        sys.exit(3)
    msg = ("Grid and drift session are ready. Press Run on realtime_drift_mda_zstack.bsh in the "
           "Micro-Manager script panel (CONFIG_FILE = "
           r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config_zstack.json"
           "). The QC pages build themselves once 15 frames are in.")
    (SESSION / "START_TIMELAPSE.txt").write_text(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n", encoding="utf-8")
    log(msg)
    beep()
    wait_for_frames()
    made = sheets()
    log("sheets: " + ", ".join(p.name for p in made) if made else "no sheet was produced")
    beep(3)


if __name__ == "__main__":
    main()
