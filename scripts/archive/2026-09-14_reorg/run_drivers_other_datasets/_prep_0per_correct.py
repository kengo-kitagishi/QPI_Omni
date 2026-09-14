"""Prepare directories for correct_0pergluc on 260508 online_crop_sub data.

1. Copy channel_rois.json from grid_2per to each Pos in online_crop_sub
2. Create directory junctions in 0per_gluc: Pos{n}_x+0_y+0 -> Pos{n}/z005
"""
import shutil
import subprocess
from pathlib import Path

GRID_2PER = Path(r"E:\260504\grid_2pergluc_1")
CROP_SUB = Path(r"D:\AquisitionData\Kitagishi\260508\online_crop_sub_zstack")
GRID_0PER = Path(r"E:\260504\0per_gluc")
Z_INDEX = 5

copied = 0
skipped = 0
junctions = 0
junction_skipped = 0

for n in range(1, 105):
    label = f"Pos{n}"

    # --- 1. Copy channel_rois.json ---
    src = GRID_2PER / f"{label}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json"
    dst_dir = CROP_SUB / label / "output_phase" / "channels"
    dst = dst_dir / "channel_rois.json"

    if dst.exists():
        skipped += 1
    elif src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        copied += 1
    else:
        print(f"WARNING: source missing: {src}")

    # --- 2. Create junction Pos{n}_x+0_y+0 -> Pos{n}/z{Z_INDEX:03d} ---
    junction = GRID_0PER / f"{label}_x+0_y+0"
    target = GRID_0PER / label / f"z{Z_INDEX:03d}"

    if junction.exists():
        junction_skipped += 1
    elif target.exists():
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
            check=True, capture_output=True,
        )
        junctions += 1
    else:
        print(f"WARNING: 0per target missing: {target}")

print(f"\nchannel_rois.json: copied={copied}, skipped={skipped}")
print(f"0per junctions:    created={junctions}, skipped={junction_skipped}")
