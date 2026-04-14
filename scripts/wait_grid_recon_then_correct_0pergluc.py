# %%
"""
batch_reconstruction_grid.py などで grid_0per を再構成している間、
指定ベースラベル（例: Pos6）の全グリッド点に img_*_ph_ZZZ_phase.tif が揃うまで待機し、
揃ったら correct_0pergluc.py を実行する。

例:
  python scripts/wait_grid_recon_then_correct_0pergluc.py
  python scripts/wait_grid_recon_then_correct_0pergluc.py --interval-sec 120 --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from grid_subtract import scan_grid_positions


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="grid_0per 再構成完了を待ち、correct_0pergluc.py を実行する。",
    )
    p.add_argument(
        "--grid-0per-dir",
        default=r"C:\grid_0pergluc_60ms_1",
        help="GRID_0PER_DIR（correct_0pergluc と同じ）",
    )
    p.add_argument(
        "--recon-base-label",
        default="Pos6",
        help="再構成対象のフォルダ接頭辞（Pos6_x*_y*）",
    )
    p.add_argument(
        "--z-index",
        type=int,
        default=None,
        help="確認する z（未指定時は --grid-sub-log の grid_z_index、なければ 18）",
    )
    p.add_argument(
        "--grid-sub-log",
        default=r"D:\AquisitionData\Kitagishi\260405\ph_260405\Pos6\output_phase\channels\grid_subtract_log.json",
        help="grid_z_index を読むための grid_subtract_log.json",
    )
    p.add_argument(
        "--interval-sec",
        type=float,
        default=60.0,
        help="再チェック間隔（秒）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="待機後に correct_0pergluc.py は起動しない",
    )
    return p.parse_args()


def _resolve_z_index(args: argparse.Namespace) -> int:
    if args.z_index is not None:
        return int(args.z_index)
    log_path = Path(args.grid_sub_log)
    if log_path.exists():
        data = json.loads(log_path.read_text(encoding="utf-8"))
        return int(data.get("grid_z_index", 18))
    return 18


def _missing_phase_tifs(grid_dir: Path, base_label: str, z: int) -> tuple[list[str], int]:
    pos_map = scan_grid_positions(str(grid_dir), base_label)
    if not pos_map:
        return [f"no folders matching {base_label}_x*_y* under {grid_dir}"], 0

    fname = f"img_000000000_ph_{z:03d}_phase.tif"
    missing: list[str] = []
    for _xy, d in sorted(pos_map.items()):
        f = d / "output_phase" / fname
        if not f.exists():
            missing.append(str(f))
    return missing, len(pos_map)


def main() -> None:
    args = _parse_args()
    z = _resolve_z_index(args)
    grid_dir = Path(args.grid_0per_dir)

    print(
        f"[wait] grid_0per={grid_dir}  label={args.recon_base_label}  "
        f"z={z}  interval={args.interval_sec}s"
    )

    while True:
        missing, n_total = _missing_phase_tifs(grid_dir, args.recon_base_label, z)
        if n_total == 0:
            print("[wait] エラー: グリッドフォルダが見つかりません。")
            sys.exit(1)

        n_miss = len(missing)
        if n_miss == 0:
            print(f"[wait] 完了: {n_total} 点すべてに ph_{z:03d}_phase.tif があります。")
            break

        print(f"[wait] 未完了: {n_miss}/{n_total} 不足。{args.interval_sec}s 後に再チェック。")
        show = missing[:8]
        for m in show:
            print(f"  - {m}")
        if len(missing) > len(show):
            print(f"  ... 他 {len(missing) - len(show)} 件")

        time.sleep(args.interval_sec)

    cmd = [sys.executable, str(_SCRIPT_DIR / "correct_0pergluc.py")]
    print(f"[run] {' '.join(cmd)}")
    if args.dry_run:
        print("[run] --dry-run のためスキップしました。")
        return

    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

# %%
