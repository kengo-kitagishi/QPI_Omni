"""
scheduled_backup_copy.py
------------------------
Sequential backup copy to F: drive with initial delay.

Steps:
  1. Wait for scheduled start time
  2. Copy crop_sub T=2..2018 → F:\260517_crop_sub_T0002_2018\
  3. Copy crop_sub T=1442..3746 → F:\260517_crop_sub_T1442_3746\
  4. robocopy E:\260517\grid_2pergluc_2
  5. robocopy E:\260517\2per_0055per_0per_2per
  6. robocopy D:\AquisitionData\Kitagishi\260517\0per_gluc
  7. robocopy D:\AquisitionData\Kitagishi\260517\online_crop_sub_zstack
  8. robocopy E:\260517\focus_check_zstack

Usage:
    python scheduled_backup_copy.py [--delay-hours 2] [--threads 8]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def copy_crop_sub_filtered(src_root, dst_root, t_min, t_max, n_threads=8):
    """Copy crop_sub frames with frame number filtering."""
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    pat = re.compile(r"img_(\d{9})_ph_")

    file_pairs = []
    positions = sorted(d for d in src_root.iterdir()
                       if d.is_dir() and re.match(r"Pos\d+$", d.name))

    for pos_dir in positions:
        ch_base = pos_dir / "output_phase" / "channels" / "crop_sub_rawraw" / "z000"
        if not ch_base.exists():
            continue
        for ch_dir in sorted(ch_base.iterdir()):
            if not ch_dir.is_dir() or not ch_dir.name.startswith("ch"):
                continue
            dst_ch = dst_root / pos_dir.name / "output_phase" / "channels" / "crop_sub_rawraw" / "z000" / ch_dir.name
            for f in ch_dir.iterdir():
                m = pat.match(f.name)
                if m and t_min <= int(m.group(1)) <= t_max:
                    file_pairs.append((str(f), str(dst_ch / f.name), str(dst_ch)))

    if not file_pairs:
        log("  No files to copy")
        return 0

    log(f"  Files to copy: {len(file_pairs)}")
    total_bytes = sum(os.path.getsize(p[0]) for p in file_pairs[:100])
    avg_size = total_bytes / min(len(file_pairs), 100)
    est_gb = avg_size * len(file_pairs) / 1e9
    log(f"  Estimated size: {est_gb:.1f} GB")

    created_dirs = set()
    def ensure_dir(d):
        if d not in created_dirs:
            os.makedirs(d, exist_ok=True)
            created_dirs.add(d)

    copied = 0
    t0 = time.time()
    report_interval = max(len(file_pairs) // 20, 1000)

    def copy_one(src, dst, dst_dir):
        ensure_dir(dst_dir)
        shutil.copy2(src, dst)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(copy_one, s, d, dd) for s, d, dd in file_pairs]
        for f in as_completed(futures):
            f.result()
            copied += 1
            if copied % report_interval == 0:
                elapsed = time.time() - t0
                rate = copied / elapsed
                eta = (len(file_pairs) - copied) / rate / 60
                log(f"  {copied}/{len(file_pairs)}  {rate:.0f} files/sec  ETA {eta:.0f} min")

    elapsed = time.time() - t0
    rate = copied / elapsed
    log(f"  Done: {copied} files in {elapsed/60:.1f} min ({rate:.0f} files/sec)")
    return copied


def robocopy_dir(src, dst_root, n_threads=8):
    """Copy a full directory using robocopy."""
    src = Path(src)
    dst = Path(dst_root) / src.name
    log(f"  robocopy {src} -> {dst}")
    log(f"  MT:{n_threads}")

    cmd = [
        "robocopy", str(src), str(dst),
        "/E",
        f"/MT:{n_threads}",
        "/R:3", "/W:5",
        "/NP",
        "/NDL",
        "/NFL",
        "/NC",
        "/NS",
        "/LOG+:" + str(Path(dst_root) / f"robocopy_{src.name}.log"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # robocopy exit codes: 0-7 = success (various levels), 8+ = error
    if result.returncode >= 8:
        log(f"  ERROR: robocopy exit code {result.returncode}")
        log(f"  {result.stderr[:500]}")
    else:
        log(f"  Done (exit code {result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay-hours", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-delay", action="store_true")
    parser.add_argument("--robocopy-threads", type=int, default=8)
    args = parser.parse_args()

    crop_sub_src = r"E:\260517\2per_0055per_0per_2per_crop_sub"
    dst_drive = r"F:"

    full_dirs = [
        r"E:\260517\grid_2pergluc_2",
        r"E:\260517\2per_0055per_0per_2per",
        r"D:\AquisitionData\Kitagishi\260517\0per_gluc",
        r"D:\AquisitionData\Kitagishi\260517\online_crop_sub_zstack",
        r"E:\260517\focus_check_zstack",
    ]

    log("=== Scheduled Backup Copy ===")
    log(f"Copy threads: {args.threads}  Robocopy threads: {args.robocopy_threads}")

    # Step 0: Wait
    if not args.skip_delay and args.delay_hours > 0:
        start_time = datetime.now() + timedelta(hours=args.delay_hours)
        log(f"Waiting {args.delay_hours} hours (start at {start_time.strftime('%H:%M:%S')})")
        time.sleep(args.delay_hours * 3600)
        log("Wait complete, starting copies")

    overall_t0 = time.time()

    # Step 1: crop_sub T=2..2018
    log("--- Step 1/7: crop_sub T=2..2018 ---")
    dst1 = os.path.join(dst_drive, "260517_crop_sub_T0002_2018")
    copy_crop_sub_filtered(crop_sub_src, dst1, 2, 2018, args.threads)

    # Step 2: crop_sub T=1442..3746
    log("--- Step 2/7: crop_sub T=1442..3746 ---")
    dst2 = os.path.join(dst_drive, "260517_crop_sub_T1442_3746")
    copy_crop_sub_filtered(crop_sub_src, dst2, 1442, 3746, args.threads)

    # Steps 3-7: Full directory copies
    for i, src_dir in enumerate(full_dirs, start=3):
        log(f"--- Step {i}/7: {Path(src_dir).name} ---")
        if not Path(src_dir).exists():
            log(f"  SKIP: not found")
            continue
        robocopy_dir(src_dir, dst_drive, args.robocopy_threads)

    total_elapsed = time.time() - overall_t0
    log(f"=== All done: {total_elapsed/3600:.1f} hours ===")


if __name__ == "__main__":
    main()
