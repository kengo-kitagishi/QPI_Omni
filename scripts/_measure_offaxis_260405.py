"""Measure the off-axis FFT peak for representative Pos in the
260405_acute_z18_200h dataset, under both candidate crops."""
from __future__ import annotations
import numpy as np
import tifffile
from pathlib import Path


def peak_in_fft(path: str, crop: tuple[int, int, int, int],
                dc_radius: int = 80) -> tuple[int, int]:
    img = tifffile.imread(path)[crop[0]:crop[1], crop[2]:crop[3]].astype(np.float64)
    fft = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img))))
    h, w = fft.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    fft[(yy - cy) ** 2 + (xx - cx) ** 2 < dc_radius ** 2] = 0
    fft[:cy] = 0  # keep LOWER half (matches OFFAXIS_CENTER convention from history)
    return np.unravel_index(int(np.argmax(fft)), fft.shape)


CASES = [
    ("Pos1  crop=(400:2448)", r"F:\260405_acute_z18_200h\ph_260405\Pos1\img_000000000_ph_000.tif", (0, 2048, 400, 2448)),
    ("Pos2  crop=(400:2448)", r"F:\260405_acute_z18_200h\ph_260405\Pos2\img_000000000_ph_000.tif", (0, 2048, 400, 2448)),
    ("Pos10 crop=(400:2448)", r"F:\260405_acute_z18_200h\ph_260405\Pos10\img_000000000_ph_000.tif", (0, 2048, 400, 2448)),
    ("Pos32 crop=(400:2448)", r"F:\260405_acute_z18_200h\ph_260405\Pos32\img_000000000_ph_000.tif", (0, 2048, 400, 2448)),
    ("Pos33 crop=(400:2448)", r"F:\260405_acute_z18_200h\ph_260405\Pos33\img_000000000_ph_000.tif", (0, 2048, 400, 2448)),
    ("Pos33 crop=(0:2048)",   r"F:\260405_acute_z18_200h\ph_260405\Pos33\img_000000000_ph_000.tif", (0, 2048, 0, 2048)),
    ("Pos40 crop=(0:2048)",   r"F:\260405_acute_z18_200h\ph_260405\Pos40\img_000000000_ph_000.tif", (0, 2048, 0, 2048)),
    ("Pos40 crop=(400:2448)", r"F:\260405_acute_z18_200h\ph_260405\Pos40\img_000000000_ph_000.tif", (0, 2048, 400, 2448)),
    ("Pos52 crop=(0:2048)",   r"F:\260405_acute_z18_200h\ph_260405\Pos52\img_000000000_ph_000.tif", (0, 2048, 0, 2048)),
    ("Pos60 crop=(0:2048)",   r"F:\260405_acute_z18_200h\ph_260405\Pos60\img_000000000_ph_000.tif", (0, 2048, 0, 2048)),
]

for name, path, crop in CASES:
    if not Path(path).exists():
        print(f"{name:30s}  MISSING")
        continue
    p = peak_in_fft(path, crop)
    print(f"{name:30s}  fft_peak(row,col)={p}")
