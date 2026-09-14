"""Diagnose why perpendicular slices overshoot the contour at the caps, and
whether that overshoot changes the integrated volume. Caches the frame-1800
crop so repeat runs don't rebuild the inbox index."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mask_volume_schematic import load_mother_crop, medial_axis_geometry

cache = Path("results/260517/_cap_debug_crop.npz")
if cache.exists():
    d = np.load(cache)
    binary, px, frame = d["binary"].astype(bool), float(d["px"]), int(d["frame"])
    print(f"[cache] loaded crop {binary.shape} frame {frame}")
else:
    binary, frame, px = load_mother_crop("Pos27", "ch06", 1800)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, binary=binary, px=px, frame=frame)
    print(f"[fresh] loaded+cached crop {binary.shape} frame {frame}")

geo = medial_axis_geometry(binary, px)
raw_len = np.linalg.norm(geo.slice_p1_xy - geo.slice_p0_xy, axis=1)   # = w_perp
clip_len = np.linalg.norm(geo.clip_p1_xy - geo.clip_p0_xy, axis=1)
overshoot = raw_len - clip_len
theta_deg = np.degrees(geo.theta_rad)
arc = geo.arc_step_px

vol_raw = float(np.sum(np.pi * (raw_len / 2) ** 2 * arc))
vol_clip = float(np.sum(np.pi * (clip_len / 2) ** 2 * arc))

print(f"\nframe {frame}  n_col={len(raw_len)}  px={px:.4f} um")
print(f"theta(deg): min {theta_deg.min():+.1f}  max {theta_deg.max():+.1f}  "
      f"mean|theta| {np.mean(np.abs(theta_deg)):.1f}")
print(f"overshoot(px): max {overshoot.max():.2f}  median {np.median(overshoot):.2f}  "
      f"#cols>0.5px = {(overshoot > 0.5).sum()} / {len(overshoot)}")
print(f"V(raw w_perp) = {vol_raw * px ** 3:7.2f} um3   (geo: {geo.volume_um3:.2f})")
print(f"V(clip chord) = {vol_clip * px ** 3:7.2f} um3   "
      f"clip/raw = {vol_clip / vol_raw:.4f}")

print("\n  i    x    h_vert  theta  w_perp  clip   over   arc")
ends = list(range(0, 7)) + list(range(len(raw_len) - 7, len(raw_len)))
for i in ends:
    print(f"{i:3d} {geo.medial_xy[i,0]:6.1f} {geo.h_vert_px[i]:6.2f} "
          f"{theta_deg[i]:6.1f} {raw_len[i]:6.2f} {clip_len[i]:6.2f} "
          f"{overshoot[i]:6.2f} {arc[i]:5.2f}")

# where is the overshoot concentrated? (cap = outer 1 cap-length of columns)
cap = int(np.clip(np.median(geo.h_vert_px) / 2.0, 1, len(raw_len) // 3))
body = slice(cap, len(raw_len) - cap)
print(f"\ncap_len~{cap}px  body overshoot max = {overshoot[body].max():.2f}px  "
      f"cap overshoot max = {max(overshoot[:cap].max(), overshoot[-cap:].max()):.2f}px")
