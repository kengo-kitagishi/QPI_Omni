"""_qc_keep_center_raw_261121.py

Keeps one raw hologram per Pos (the x+0_y+0 point) of the 261121 11x11 grid (chunk 1: Pos0-45,
chunk 2: Pos0 + Pos46-90), for the shot-noise prediction at the channel positions (local mean signal
and fringe visibility need the raw hologram; the chunk watcher deletes the raw after reconstruction).

The acquisition disk (E:) is not touched while the BeanShell is acquiring: for each chunk the script
waits until the watcher has started writing that chunk's reconstruction output to D: (= acquisition
of the chunk is over, the BeanShell is blocked in the watcher call), then copies its ~46 files
(about 0.5 GB) before the watcher deletes the raw.

    python scripts/_qc_keep_center_raw_261121.py
"""
import shutil
import time
from pathlib import Path

RECON = Path(r"D:\AquisitionData\Kitagishi\261121\recon_0p03um_11x11")
OUT = Path(r"D:\AquisitionData\Kitagishi\261121\shotnoise_source\raw_center_11x11")
CHUNKS = [
    (Path(r"E:\261121\grid11_s1"), list(range(1, 46)), "chunk1"),
    (Path(r"E:\261121\grid11_s2"), list(range(46, 91)), "chunk2"),
]
MAX_WAIT_S = 4 * 3600


def recon_started(pos):
    return any(any(RECON.glob(f"Pos{n}_x*")) for n in (pos[0], pos[len(pos) // 2], pos[-1]))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for raw, pos, name in CHUNKS:
        while not recon_started(pos):
            if time.time() - t0 > MAX_WAIT_S:
                print(f"gave up waiting for the {name} reconstruction", flush=True)
                return
            time.sleep(10)
        print(f"{time.strftime('%H:%M:%S')} {name}: reconstruction output seen; copying", flush=True)
        n_ok = 0
        for n in [0] + pos:
            src = raw / f"Pos{n}_x+0_y+0" / "img_000000000_ph_000.tif"
            tag = f"{name}_" if n == 0 else ""
            try:
                shutil.copyfile(src, OUT / f"{tag}Pos{n}_x+0_y+0_img_000000000_ph_000.tif")
                n_ok += 1
            except OSError as e:
                print(f"  Pos{n}: {e}", flush=True)
        print(f"{time.strftime('%H:%M:%S')} {name}: copied {n_ok}/{len(pos) + 1} holograms to {OUT}", flush=True)


if __name__ == "__main__":
    main()
