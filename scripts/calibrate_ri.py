# %%
"""
calibrate_ri.py -- RI calibration via MilliQ + EtOH two-point method
=====================================================================

Prerequisite:
  extract_timelapse_delta.py を MilliQ / EtOH それぞれで実行済み。
  各 Pos について delta_z{Z:03d}.tif (511x511, float32) = medium_aligned - grid_2per
  が存在すること。

Procedure:

  PREPARATION
  -----------
  1. offaxis center を確認・更新 (optical_config.py)
  2. 2per grid 撮影済み + 再構成済み (output_phase/, output_phase_raw/)
  3. channel_rois.json を grid (x+0_y+0) から生成済み
  4. EtOH 濃度を記録 -> N_ETOH の値を確認

  MEASUREMENT
  -----------
  5. MilliQ を device に注入 -> 5 min 待機 -> multi-z timelapse 撮影
  6. EtOH を注入 -> 5 min 待機 -> 同条件で撮影

  DELTA EXTRACTION
  ----------------
  7. extract_timelapse_delta.py を MilliQ / EtOH それぞれで実行
     -> 各 Pos に delta_z{Z:03d}.tif が生成される

  RI CALIBRATION
  --------------
  8. 下記パラメータを設定して実行:  python scripts/calibrate_ri.py

Theory:
  S = sum( delta[channel_mask] ) = (n_medium - n_2per) * V_total
  n_2per = (S_miliq * n_etoh - S_etoh * n_miliq) / (S_miliq - S_etoh)
"""
import numpy as np
import tifffile
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from optical_config import WAVELENGTH
from ecc_utils import apply_2pi_tilt_crop, extract_rect_roi


# Optional: free-form note attached to this calibration entry. Edit before running.
CALIBRATION_NOTES = ""

# ============================================================
# Configuration
# ============================================================

# 2% glucose grid directory (for channel_rois.json and output_phase mask)
GRID_2PER_DIR = r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1"

# Session roots for delta TIF directories
MILIQ_SESSION = r"E:\260424\0per_gluc"
ETOH_SESSION = r"E:\260424\0per_gluc"

# Relative path from PosN/ to the delta TIF directory
DELTA_SUBDIR = "output_phase/channels/delta_timelapse"
MILIQ_DELTA_SUBDIR = "output_phase/channels/delta_miliq"
ETOH_DELTA_SUBDIR = "output_phase/channels/delta_etoh"

# Grid z index to load (delta_z{DELTA_Z:03d}.tif)
DELTA_Z = 5

# Pos numbers to process
POS_NUMBERS = list(range(1, 102))

# Reference media refractive indices (at lambda=658nm, ~25C)
# *** VERIFY for your exact concentration and temperature ***
N_MILIQ = 1.3312  # ultrapure water @ 658nm 25C
N_ETOH = 1.3588   # 100% ethanol @ 658nm 25C  -- CHANGE if diluted

# ============================================================
# Crop / mask parameters
# ============================================================
CROP_W = 40             # channel width (Y direction)
TILT_CROP_H = 270       # wide crop for tilt fitting
OUTPUT_CROP_H = 180     # output crop for mask & summation
MASK_THRESHOLD = -1.0   # phase threshold (rad) on BG-subtracted grid

POS_SPLIT = 53          # fit_right boundary
SKIP_EDGE_CHANNELS = True  # exclude first and last channel (Y-direction boundary artifacts)

# 0% glucose grid directory (optional, for n_0per)
GRID_0PER_DIR = None

# ============================================================


def _git_commit_short() -> str | None:
    """Return short git SHA of the current HEAD (with -dirty suffix), or None."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return None


def _load_history(path: Path) -> dict:
    """Load existing append-history JSON, migrating legacy single-entry if needed."""
    if not path.exists():
        return {"schema_version": "1.0", "active": None, "calibrations": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        # Corrupt or empty -- back it up and start fresh.
        backup = path.with_suffix(path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        path.replace(backup)
        print(f"  WARNING: existing JSON unreadable ({e}); backed up to {backup.name}")
        return {"schema_version": "1.0", "active": None, "calibrations": []}

    if "calibrations" in data:
        data.setdefault("schema_version", "1.0")
        data.setdefault("active", None)
        data.setdefault("calibrations", [])
        return data

    # Legacy single-entry format -- wrap it
    legacy_entry = dict(data)
    legacy_entry.setdefault("calibration_id", "legacy_pre_history")
    legacy_entry.setdefault("calibrated_at", "")
    legacy_entry.setdefault("notes", "auto-migrated from single-entry schema")
    print("  Migrating legacy single-entry JSON to append-history schema")
    return {
        "schema_version": "1.0",
        "active": legacy_entry["calibration_id"],
        "calibrations": [legacy_entry],
    }


def _fit_right(pos_num):
    return pos_num >= POS_SPLIT


def load_channel_rois(grid_dir, pos_num):
    base = Path(grid_dir) / f"Pos{pos_num}_x+0_y+0"
    candidates = [
        base / "output_phase" / "channels" / "channel_rois.json",
        base / "channel_rois.json",
        Path(grid_dir) / f"channel_rois_Pos{pos_num}.json",
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")), p
    raise FileNotFoundError(
        f"channel_rois.json not found for Pos{pos_num}:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def load_grid_output_phase(grid_dir, pos_num, z_index):
    """Load grid00 output_phase (BG-subtracted) for mask creation."""
    path = (
        Path(grid_dir)
        / f"Pos{pos_num}_x+0_y+0"
        / "output_phase"
        / f"img_000000000_ph_{z_index:03d}_phase.tif"
    )
    if not path.exists():
        raise FileNotFoundError(f"Grid output_phase not found: {path}")
    return tifffile.imread(str(path)).astype(np.float64)


def create_channel_masks(phase_bg, rois, fit_right):
    """Per-channel masks from BG-subtracted grid00.

    For each channel:
      extract_rect_roi(40, 270) -> apply_2pi_tilt_crop -> (40, 180)
      mask = (phase < MASK_THRESHOLD)
    """
    masks = []
    for roi in rois:
        large = extract_rect_roi(
            phase_bg, roi["cy"], roi["cx"], CROP_W, TILT_CROP_H,
        )
        corrected = apply_2pi_tilt_crop(
            large.copy(), OUTPUT_CROP_H, TILT_CROP_H, fit_right=fit_right,
        )
        masks.append(corrected < MASK_THRESHOLD)
    return masks


def load_delta_tif(session_dir, pos_num, z_index, delta_subdir=None):
    """Load pre-computed delta TIF (511x511) from extract_timelapse_delta.py."""
    subdir = delta_subdir if delta_subdir is not None else DELTA_SUBDIR
    path = (
        Path(session_dir)
        / f"Pos{pos_num}"
        / subdir
        / f"delta_z{z_index:03d}.tif"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Delta TIF not found: {path}\n"
            f"Run extract_timelapse_delta.py first."
        )
    return tifffile.imread(str(path)).astype(np.float64), path


def compute_channel_sums(delta, rois, masks, fit_right):
    """Per-channel: crop(40x270) -> tilt_correct -> center(40x180) -> mask -> sum.

    Returns list of (sum_value, n_mask_pixels).
    """
    results = []
    for roi, mask in zip(rois, masks):
        large = extract_rect_roi(
            delta, roi["cy"], roi["cx"], CROP_W, TILT_CROP_H,
        )
        corrected = apply_2pi_tilt_crop(
            large.copy(), OUTPUT_CROP_H, TILT_CROP_H, fit_right=fit_right,
        )
        results.append((
            float(np.sum(corrected[mask])),
            int(np.sum(mask)),
        ))
    return results


def compute_0per_delta_sums(grid_0per_dir, grid_2per_dir, pos_numbers,
                            rois_dict, masks_dict):
    """Masked sum of (grid_0per - grid_2per) across all Pos.

    Uses per-channel ECC for alignment, then warp-subtract.
    """
    from ecc_utils import tilt_fit_crop, to_uint8, ecc_align
    from compute_pos_shifts import compute_backsub_offset
    from grid_subtract import apply_inverse_shift_warp

    ECC_CROP_H = 80
    total = 0.0
    details = []

    for pos_num in pos_numbers:
        fit_r = _fit_right(pos_num)
        rois = rois_dict[pos_num]
        masks = masks_dict[pos_num]

        g2_raw_path = (
            Path(grid_2per_dir) / f"Pos{pos_num}_x+0_y+0"
            / "output_phase_raw"
            / f"img_000000000_ph_{DELTA_Z:03d}_phase.tif"
        )
        g0_raw_path = (
            Path(grid_0per_dir) / f"Pos{pos_num}_x+0_y+0"
            / "output_phase_raw"
            / f"img_000000000_ph_{DELTA_Z:03d}_phase.tif"
        )
        if not g0_raw_path.exists():
            print(f"  [0per] Pos{pos_num}: not found, skipping")
            continue

        g2_raw = tifffile.imread(str(g2_raw_path)).astype(np.float64)
        g0_raw = tifffile.imread(str(g0_raw_path)).astype(np.float64)

        all_tx, all_ty = [], []
        for roi in rois:
            a = tilt_fit_crop(
                g2_raw, roi["cy"], roi["cx"], CROP_W, ECC_CROP_H,
                TILT_CROP_H, fit_right=fit_r,
            )
            b = tilt_fit_crop(
                g0_raw, roi["cy"], roi["cx"], CROP_W, ECC_CROP_H,
                TILT_CROP_H, fit_right=fit_r,
            )
            if a is None or b is None:
                continue
            a = a + compute_backsub_offset(a)
            b = b + compute_backsub_offset(b)
            res = ecc_align(to_uint8(a, -5.0, 2.0), to_uint8(b, -5.0, 2.0))
            if res:
                all_tx.append(res[0])
                all_ty.append(res[1])

        if not all_tx:
            print(f"  [0per] Pos{pos_num}: ECC failed, skipping")
            continue

        tx, ty = float(np.median(all_tx)), float(np.median(all_ty))
        delta = apply_inverse_shift_warp(g0_raw, tx, ty) - g2_raw

        sums = compute_channel_sums(delta, rois, masks, fit_r)
        pos_sum = sum(s for s, _ in sums)
        total += pos_sum
        print(f"  [0per] Pos{pos_num}: tx={tx:.3f} ty={ty:.3f} sum={pos_sum:.2f}")
        details.append({
            "pos": pos_num, "ecc_tx": tx, "ecc_ty": ty,
            "sum": pos_sum, "per_ch": [(s, n) for s, n in sums],
        })

    return total, details


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figure_logger import save_figure

    print("=" * 60)
    print("  RI Calibration: MilliQ + EtOH two-point method")
    print("=" * 60)
    print(f"Grid (2per):     {GRID_2PER_DIR}")
    print(f"MilliQ session:  {MILIQ_SESSION}")
    print(f"EtOH session:    {ETOH_SESSION}")
    print(f"Delta subdir:    {DELTA_SUBDIR}")
    print(f"Delta z index:   {DELTA_Z}")
    print(f"Pos numbers:     {POS_NUMBERS}")
    print(f"n_miliq (ref):   {N_MILIQ}")
    print(f"n_etoh  (ref):   {N_ETOH}")
    print(f"Mask threshold:  {MASK_THRESHOLD} rad")
    print(f"Output crop:     {CROP_W} x {OUTPUT_CROP_H}")
    print()

    total_miliq = 0.0
    total_etoh = 0.0
    total_mask_pixels = 0
    per_pos_results = []
    rois_dict = {}
    masks_dict = {}
    example_data = None

    for pos_num in POS_NUMBERS:
        print(f"\n--- Pos{pos_num} ---")
        fit_r = _fit_right(pos_num)

        # channel ROIs (from grid, for cy/cx only)
        rois, rois_path = load_channel_rois(GRID_2PER_DIR, pos_num)
        n_ch = len(rois)
        print(f"  Channels: {n_ch}")

        # mask from BG-subtracted grid00
        phase_bg = load_grid_output_phase(GRID_2PER_DIR, pos_num, DELTA_Z)
        masks = create_channel_masks(phase_bg, rois, fit_r)
        mask_pixels = sum(int(np.sum(m)) for m in masks)
        print(f"  Mask: {mask_pixels} pixels (threshold={MASK_THRESHOLD})")

        rois_dict[pos_num] = rois
        masks_dict[pos_num] = masks

        # --- MilliQ delta ---
        delta_m, delta_m_path = load_delta_tif(MILIQ_SESSION, pos_num, DELTA_Z, MILIQ_DELTA_SUBDIR)
        sums_m = compute_channel_sums(delta_m, rois, masks, fit_r)

        # --- EtOH delta ---
        delta_e, delta_e_path = load_delta_tif(ETOH_SESSION, pos_num, DELTA_Z, ETOH_DELTA_SUBDIR)
        sums_e = compute_channel_sums(delta_e, rois, masks, fit_r)

        if SKIP_EDGE_CHANNELS and n_ch > 2:
            used_m = sums_m[1:-1]
            used_e = sums_e[1:-1]
            n_skip = 2
        else:
            used_m = sums_m
            used_e = sums_e
            n_skip = 0

        pos_miliq = sum(s for s, _ in used_m)
        pos_etoh = sum(s for s, _ in used_e)
        used_mask = sum(n for _, n in used_m)
        print(f"  [MilliQ] {delta_m_path.name}  sum={pos_miliq:.2f} rad*px"
              f"  (skip_edge={n_skip})")
        print(f"  [EtOH]  {delta_e_path.name}  sum={pos_etoh:.2f} rad*px")

        total_miliq += pos_miliq
        total_etoh += pos_etoh
        total_mask_pixels += used_mask

        per_pos_results.append({
            "pos": pos_num,
            "n_channels": n_ch,
            "n_channels_used": n_ch - n_skip,
            "mask_pixels": used_mask,
            "miliq_sum": pos_miliq,
            "miliq_per_ch": [(s, n) for s, n in sums_m],
            "etoh_sum": pos_etoh,
            "etoh_per_ch": [(s, n) for s, n in sums_e],
            "skip_edge_channels": n_skip,
        })

        if example_data is None:
            mid_ch = n_ch // 2
            roi_ex = rois[mid_ch]
            bg_large = extract_rect_roi(
                phase_bg, roi_ex["cy"], roi_ex["cx"], CROP_W, TILT_CROP_H,
            )
            bg_crop = apply_2pi_tilt_crop(
                bg_large.copy(), OUTPUT_CROP_H, TILT_CROP_H, fit_right=fit_r,
            )
            dm_large = extract_rect_roi(
                delta_m, roi_ex["cy"], roi_ex["cx"], CROP_W, TILT_CROP_H,
            )
            dm_crop = apply_2pi_tilt_crop(
                dm_large.copy(), OUTPUT_CROP_H, TILT_CROP_H, fit_right=fit_r,
            )
            de_large = extract_rect_roi(
                delta_e, roi_ex["cy"], roi_ex["cx"], CROP_W, TILT_CROP_H,
            )
            de_crop = apply_2pi_tilt_crop(
                de_large.copy(), OUTPUT_CROP_H, TILT_CROP_H, fit_right=fit_r,
            )
            example_data = {
                "pos": pos_num, "ch": mid_ch,
                "bg_crop": bg_crop, "mask": masks[mid_ch],
                "dm_crop": dm_crop, "de_crop": de_crop,
            }

    # ============================================================
    # Compute RI
    # ============================================================
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"Total MilliQ sum:  {total_miliq:.4f} rad*px")
    print(f"Total EtOH sum:    {total_etoh:.4f} rad*px")
    print(f"Total mask pixels: {total_mask_pixels}")

    if abs(total_miliq - total_etoh) < 1e-10:
        print("ERROR: S_miliq ~ S_etoh, cannot solve")
        sys.exit(1)

    n_2per = (total_miliq * N_ETOH - total_etoh * N_MILIQ) / (
        total_miliq - total_etoh
    )
    V_total = total_etoh / (N_ETOH - n_2per)

    print(f"\n  n_2per (2% glucose) = {n_2per:.6f}")
    print(f"  V_total = {V_total:.2f} rad*px")
    print(f"  Cross-check (from MilliQ): {N_MILIQ - total_miliq / V_total:.6f}")

    if not (N_MILIQ - 0.01 < n_2per < N_ETOH + 0.01):
        print("  WARNING: n_2per outside expected range!")

    # --- 0per glucose RI (optional) ---
    n_0per = None
    total_0per = None
    delta_0per_details = None
    if GRID_0PER_DIR and Path(GRID_0PER_DIR).exists():
        print("\n--- Computing n_0per from 0% glucose grid ---")
        total_0per, delta_0per_details = compute_0per_delta_sums(
            GRID_0PER_DIR, GRID_2PER_DIR, POS_NUMBERS, rois_dict, masks_dict,
        )
        n_0per = n_2per + total_0per / V_total
        print(f"  n_0per (0% glucose) = {n_0per:.6f}")
        print(f"  Delta n (2per - 0per) = {n_2per - n_0per:.6f}")

    # ============================================================
    # Save results JSON (append-history schema)
    # ============================================================
    out_json = Path(GRID_2PER_DIR) / "ri_calibration_results.json"

    # Build the new entry
    now_local = datetime.now(timezone.utc).astimezone()
    timestamp_iso = now_local.isoformat(timespec="seconds")
    timestamp_id = now_local.strftime("%Y%m%dT%H%M%S")
    session_label = Path(MILIQ_SESSION).name or "session"
    calibration_id = f"{session_label}_{timestamp_id}"

    media: dict = {
        "wo_milliq": float(N_MILIQ),
        "wo_2": float(n_2per),
        "wo_etoh": float(N_ETOH),
    }
    if n_0per is not None:
        media["wo_0"] = float(n_0per)

    # Channel depth (µm) from V_total: d = V_total * λ / (2π · N_mask_pixels)
    if total_mask_pixels > 0:
        wavelength_um = float(WAVELENGTH) * 1e6  # m → µm
        channel_depth_um = float(V_total) * wavelength_um / (
            2.0 * math.pi * float(total_mask_pixels)
        )
    else:
        channel_depth_um = None

    entry = {
        "calibration_id": calibration_id,
        "calibrated_at": timestamp_iso,
        "session": session_label,
        "method": "two-point (MilliQ + EtOH)",
        "wavelength_nm": float(WAVELENGTH) * 1e9,
        "git_commit": _git_commit_short(),
        "reference": {
            "n_miliq": float(N_MILIQ),
            "n_etoh": float(N_ETOH),
            "source": "literature @658nm 25C (verify)",
        },
        "media": media,
        "raw": {
            "S_miliq_rad_px": float(total_miliq),
            "S_etoh_rad_px": float(total_etoh),
            "S_0per_rad_px": float(total_0per) if total_0per is not None else None,
            "V_total_rad_px": float(V_total),
            "n_mask_pixels": int(total_mask_pixels),
        },
        "exclusions": {
            "skip_edge_channels": bool(SKIP_EDGE_CHANNELS),
            "excluded_pos": [],
        },
        "channel_depth_um": channel_depth_um,
        "config": {
            "grid_2per_dir": str(GRID_2PER_DIR),
            "miliq_session": str(MILIQ_SESSION),
            "etoh_session": str(ETOH_SESSION),
            "miliq_delta_subdir": MILIQ_DELTA_SUBDIR,
            "etoh_delta_subdir": ETOH_DELTA_SUBDIR,
            "delta_subdir": DELTA_SUBDIR,
            "grid_0per_dir": str(GRID_0PER_DIR) if GRID_0PER_DIR else None,
            "delta_z": DELTA_Z,
            "crop_w": CROP_W,
            "tilt_crop_h": TILT_CROP_H,
            "output_crop_h": OUTPUT_CROP_H,
            "mask_threshold": MASK_THRESHOLD,
            "pos_numbers": list(POS_NUMBERS),
            "pos_split": POS_SPLIT,
        },
        "per_pos": per_pos_results,
        "delta_0per": delta_0per_details,
        "notes": CALIBRATION_NOTES,
    }

    history = _load_history(out_json)
    history["calibrations"].append(entry)
    history["active"] = calibration_id

    out_json.write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nResults JSON: {out_json}")
    print(f"  calibration_id: {calibration_id}")
    print(f"  total entries: {len(history['calibrations'])}")
    if channel_depth_um is not None:
        print(f"  channel depth: {channel_depth_um:.3f} µm")

    # ============================================================
    # Diagnostic figure
    # ============================================================
    n_pos = len(per_pos_results)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # (0, 0-1) per-Pos sum comparison
    ax_sum = fig.add_subplot(gs[0, :2])
    x_pos = np.arange(n_pos)
    w = 0.35
    ax_sum.bar(
        x_pos - w / 2,
        [p["miliq_sum"] for p in per_pos_results],
        w, label="MilliQ", color="C0", alpha=0.8,
    )
    ax_sum.bar(
        x_pos + w / 2,
        [p["etoh_sum"] for p in per_pos_results],
        w, label="EtOH", color="C1", alpha=0.8,
    )
    ax_sum.set_xticks(x_pos)
    ax_sum.set_xticklabels([f"P{p['pos']}" for p in per_pos_results], fontsize=7)
    ax_sum.set_ylabel("Masked sum (rad*px)")
    ax_sum.set_title("Per-Pos sums")
    ax_sum.legend(fontsize=8)
    ax_sum.axhline(0, ls="-", lw=0.3, color="gray")

    # (0, 2) per-ch sums for first Pos
    if per_pos_results:
        ax_ch = fig.add_subplot(gs[0, 2])
        pr0 = per_pos_results[0]
        n_ch0 = pr0["n_channels"]
        x_ch = np.arange(n_ch0)
        ax_ch.bar(
            x_ch - w / 2,
            [s for s, _ in pr0["miliq_per_ch"]],
            w, label="MilliQ", color="C0", alpha=0.8,
        )
        ax_ch.bar(
            x_ch + w / 2,
            [s for s, _ in pr0["etoh_per_ch"]],
            w, label="EtOH", color="C1", alpha=0.8,
        )
        ax_ch.set_xlabel("Channel")
        ax_ch.set_ylabel("Sum (rad*px)")
        ax_ch.set_title(f"Per-ch sums (Pos{pr0['pos']})")
        ax_ch.legend(fontsize=7)

    # (1, 0) grid00 BG-sub + mask
    if example_data:
        ax_bg = fig.add_subplot(gs[1, 0])
        im = ax_bg.imshow(
            example_data["bg_crop"], aspect="auto", cmap="RdBu_r",
            vmin=-5, vmax=2,
        )
        ax_bg.contour(
            example_data["mask"].astype(float),
            levels=[0.5], colors="lime", linewidths=0.8,
        )
        ax_bg.set_title(
            f"Grid00 + mask (Pos{example_data['pos']} ch{example_data['ch']})"
        )
        ax_bg.set_xlabel("X (px)")
        ax_bg.set_ylabel("Y (px)")
        plt.colorbar(im, ax=ax_bg, fraction=0.046, pad=0.04, label="rad")

        # (1, 1) MilliQ delta
        ax_dm = fig.add_subplot(gs[1, 1])
        im_m = ax_dm.imshow(
            example_data["dm_crop"], aspect="auto", cmap="RdBu_r",
            vmin=-1, vmax=1,
        )
        ax_dm.contour(
            example_data["mask"].astype(float),
            levels=[0.5], colors="lime", linewidths=0.8,
        )
        ax_dm.set_title("Delta: MilliQ - grid")
        ax_dm.set_xlabel("X (px)")
        plt.colorbar(im_m, ax=ax_dm, fraction=0.046, pad=0.04, label="rad")

        # (1, 2) EtOH delta
        ax_de = fig.add_subplot(gs[1, 2])
        im_e = ax_de.imshow(
            example_data["de_crop"], aspect="auto", cmap="RdBu_r",
            vmin=-1, vmax=1,
        )
        ax_de.contour(
            example_data["mask"].astype(float),
            levels=[0.5], colors="lime", linewidths=0.8,
        )
        ax_de.set_title("Delta: EtOH - grid")
        ax_de.set_xlabel("X (px)")
        plt.colorbar(im_e, ax=ax_de, fraction=0.046, pad=0.04, label="rad")

        # (2, 0) center horizontal profile
        ax_prof = fig.add_subplot(gs[2, 0])
        mid_y = CROP_W // 2
        ax_prof.plot(
            example_data["dm_crop"][mid_y, :], lw=0.8, color="C0", label="MilliQ",
        )
        ax_prof.plot(
            example_data["de_crop"][mid_y, :], lw=0.8, color="C1", label="EtOH",
        )
        ax_prof.fill_between(
            np.arange(OUTPUT_CROP_H), -2, 2,
            where=example_data["mask"][mid_y, :],
            alpha=0.1, color="green", label="Mask",
        )
        ax_prof.axhline(0, ls="--", lw=0.5, color="gray")
        ax_prof.set_xlabel("X (px)")
        ax_prof.set_ylabel("Phase diff (rad)")
        ax_prof.set_title("Center profile")
        ax_prof.legend(fontsize=7)

    # (2, 1-2) result summary
    ax_txt = fig.add_subplot(gs[2, 1:])
    ax_txt.axis("off")
    lines = [
        f"n_2per (2% glucose) = {n_2per:.6f}",
        f"n_miliq (ref)       = {N_MILIQ:.6f}",
        f"n_etoh  (ref)       = {N_ETOH:.6f}",
        f"S_miliq = {total_miliq:.2f} rad*px",
        f"S_etoh  = {total_etoh:.2f} rad*px",
        f"V_total = {V_total:.2f} rad*px",
    ]
    if n_0per is not None:
        lines.append(f"n_0per (0% glucose) = {n_0per:.6f}")
        lines.append(f"Delta n (2per-0per) = {n_2per - n_0per:.6f}")
    lines.extend([
        f"Mask threshold = {MASK_THRESHOLD} rad",
        f"Output crop = {CROP_W} x {OUTPUT_CROP_H}",
        f"Pos: {POS_NUMBERS}",
    ])
    ax_txt.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax_txt.transAxes, fontsize=11,
        va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle("RI Calibration: MilliQ + EtOH Two-Point Method", fontsize=14)

    save_figure(
        fig,
        params={
            "n_2per": n_2per, "n_0per": n_0per,
            "n_miliq": N_MILIQ, "n_etoh": N_ETOH,
            "total_miliq": total_miliq, "total_etoh": total_etoh,
            "V_total": V_total,
            "mask_threshold": MASK_THRESHOLD,
            "output_crop_h": OUTPUT_CROP_H,
            "delta_z": DELTA_Z,
        },
        description="RI calibration two-point method results",
        data={
            "per_pos_miliq_sums": np.array(
                [p["miliq_sum"] for p in per_pos_results],
            ),
            "per_pos_etoh_sums": np.array(
                [p["etoh_sum"] for p in per_pos_results],
            ),
            "pos_numbers": np.array(POS_NUMBERS),
        },
    )
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()

# %%
