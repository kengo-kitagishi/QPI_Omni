"""Regenerate the curated phase1-dead full cell-cycle sheets (260517).

Per-channel curation collected interactively with the user. Each entry maps a
Pos:ch to the --channel-all flags that shape its final figure: which detected
cycle rows to show, and how the cell's death/arrest is rendered as the final row
(an auto death window, an explicit last cycle, or a long arrest "tail").

Run all:        python _regen_phase1dead.py
Run a subset:   python _regen_phase1dead.py Pos6:ch09 Pos9:ch08
List config:    python _regen_phase1dead.py --list
"""
import subprocess
import sys
from pathlib import Path

PY = sys.executable
SCRIPT = str(Path(__file__).with_name("_fig_panelA_cellcycle.py"))
COMMON = ["--vmin", "0", "--vmax", "1.8"]   # fixed phase scale, comparable rows

# Pos:ch -> extra flags. Order is the review order.
CURATION = {
    "Pos2:ch03":  ["--rows", "1-9"],                    # 9 clean cycles + death
    "Pos2:ch08":  ["--no-death"],                        # drop the death row
    "Pos6:ch09":  ["--tail-from", "503"],                # 3h death tail from 503
    "Pos9:ch08":  ["--tail-from", "1202"],               # 3h death tail from 1202
    "Pos10:ch07": ["--tail-from", "797"],                # arrest tail
    "Pos11:ch06": ["--tail-from", "1007"],               # arrest tail
    "Pos17:ch09": ["--tail-from", "587"],                # arrest tail
    "Pos20:ch06": ["--tail-from", "1267", "--tail-to", "1450"],  # elongation death
    "Pos26:ch03": ["--tail-from", "792"],                # arrest tail
    "Pos26:ch08": ["--tail-from", "693"],                # arrest tail
    "Pos31:ch06": ["--tail-from", "1139"],               # arrest tail
    "Pos32:ch05": ["--tail-from", "419"],                # arrest tail
    "Pos35:ch05": ["--tail-from", "343"],                # arrest tail
    "Pos45:ch00": ["--tail-from", "720"],                # arrest tail
    # --- channels with no per-channel trim: regenerated only for the global
    #     frame-label change (all cycles + auto death, frame-numbered rows) ---
    "Pos1:ch09":  [],
    "Pos5:ch08":  [],
    "Pos9:ch04":  ["--max-cycle-len", "110", "--tail-from", "1397"],  # keep long cycles incl. 1296-1396 (divides ~117h), 3h death tail from 1397
    "Pos14:ch08": [],
    "Pos17:ch02": [],
    "Pos18:ch02": [],
    "Pos30:ch04": [],
    "Pos37:ch08": [],
    "Pos37:ch11": [],
    "Pos39:ch05": [],
    "Pos42:ch01": [],
    "Pos45:ch02": [],
}


def run(only=None):
    items = [(c, f) for c, f in CURATION.items() if not only or c in only]
    for chan, flags in items:
        cmd = [PY, SCRIPT, "--channel-all", chan] + flags + COMMON
        print(">>>", chan, " ".join(flags), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--list":
        for c, f in CURATION.items():
            print(f"{c:14s} {' '.join(f)}")
    else:
        run(only=set(argv) or None)
