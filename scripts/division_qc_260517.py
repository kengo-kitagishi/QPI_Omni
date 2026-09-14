"""division_qc_260517.py - validate the tracker's division calls with a mass / volume test.

The tracker (central_cell_lineage_tracker.py) calls a division from the area of a
single frame: continuation if curr/prev > 0.68, division if |(curr + next) - prev| /
prev < 0.30, otherwise the frame is an outlier and the ID continues. It never checks
whether the split persists, so a transient segmentation split produces a spurious
daughter. This QC re-examines every candidate division (= birth of a daughter) with
the parent's dry mass and volume before and after the event (yellow-contour
``mass_pg_efd`` / ``volume_um3_efd``), using the rules agreed on 2026-09-14:

  1. every daughter birth in lineage_data3D.csv is a candidate;
  2. if the parent has no tracker outlier within +-1 frame of the event, accept it as is
     (method = direct);
  3. otherwise never use the outlier frames themselves but the nearest valid frames
     (not outlier, not border, finite mass) before and after;
  4. within +-8 frames (40 min) take up to 3 valid points on each side and compare
     medians;
  5. rescue the event only if
        0.25 <= post_mass / pre_mass <= 0.78
        0.25 <= post_vol  / pre_vol  <= 0.85
        |mass_ratio - vol_ratio| <= 0.25
     (method = rescued); fewer than 2 valid points on a side = insufficient;
  6. several rescued candidates of the same parent within 1 h (12 frames) are one
     event: the earliest is kept, the others are marked duplicate;
  7. outlier frames stay excluded from mass / growth fits downstream, but a cell cycle
     is not discarded because of them (analysis-side rule, not applied here).

Output per channel: <lineage_out>/divisions_qc.csv, one row per candidate.

Usage:
    python scripts/division_qc_260517.py --lineage-dir <.../inference_out/lineage_out>
    python scripts/division_qc_260517.py --all      # every production channel of the master root
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

# --- rule constants (2026-09-14) ---
OUTLIER_NEAR_FRAMES = 1        # +-1 frame: direct acceptance if no tracker outlier there
WINDOW_FRAMES = 8              # +-8 frames = +-40 min search window
N_POINTS = 3                   # up to 3 valid points per side
MIN_POINTS = 2
MASS_RATIO = (0.25, 0.78)
VOL_RATIO = (0.25, 0.85)
MAX_RATIO_DIFF = 0.25
DUPLICATE_FRAMES = 12          # 1 h at 5 min/frame
MASS_COL = "mass_pg_efd"
VOL_COL = "volume_um3_efd"

OUT_COLS = ["parent_id", "daughter_id", "frame", "time_h", "is_mother_division", "in_tree",
            "outlier_near", "n_pre", "n_post", "pre_mass_pg", "post_mass_pg", "mass_ratio",
            "pre_volume_um3", "post_volume_um3", "volume_ratio", "method", "validated", "reason"]


def _side_values(rows: pd.DataFrame, frames: np.ndarray, before: bool,
                 mass_col: str = MASS_COL, vol_col: str = VOL_COL):
    """Median of up to N_POINTS nearest valid (mass, volume) pairs on one side."""
    sel = rows[rows["frame"].isin(frames)]
    sel = sel.sort_values("frame", ascending=not before)   # nearest first
    sel = sel.head(N_POINTS)
    n = len(sel)
    if n == 0:
        return 0, np.nan, np.nan
    return n, float(sel[mass_col].median()), float(sel[vol_col].median())


def qc_channel(df: pd.DataFrame, dt_min: float = 5.0, frame_min: int | None = None,
               mass_col: str = MASS_COL, vol_col: str = VOL_COL) -> pd.DataFrame:
    """Run the QC on one channel's long table. Returns one row per candidate division.

    ``mass_col`` / ``vol_col`` default to the yellow-contour columns; older lineages
    (pre-2026-09-14 masters) can pass e.g. mass_pg / volume_um3_profile.
    """
    if df.empty or "parent_id" not in df.columns:
        return pd.DataFrame(columns=OUT_COLS)
    df = df.copy()
    MASS_COL_, VOL_COL_ = mass_col, vol_col
    valid = (~df["is_outlier"].astype(bool)) & (~df["touches_border"].astype(bool)) \
        & np.isfinite(df[MASS_COL_]) & np.isfinite(df[VOL_COL_]) & (df[VOL_COL_] > 0)
    df["_valid"] = valid
    t0 = int(frame_min) if frame_min is not None else int(df["frame"].min())

    cells = df.groupby("cell_id").agg(parent_id=("parent_id", "first"), birth=("birth_frame", "first"),
                                      in_tree=("in_tree", "first"))
    cand = cells[(cells["parent_id"] >= 0)].reset_index()
    if cand.empty:
        return pd.DataFrame(columns=OUT_COLS)

    by_cell = {cid: g.sort_values("frame") for cid, g in df.groupby("cell_id")}
    out = []
    for r in cand.itertuples(index=False):
        p, d, f = int(r.parent_id), int(r.cell_id), int(r.birth)
        prow = by_cell.get(p)
        rec = dict(parent_id=p, daughter_id=d, frame=f, time_h=(f - t0) * dt_min / 60.0,
                   is_mother_division=(p == 0), in_tree=bool(r.in_tree),
                   outlier_near=False, n_pre=0, n_post=0, pre_mass_pg=np.nan, post_mass_pg=np.nan,
                   mass_ratio=np.nan, pre_volume_um3=np.nan, post_volume_um3=np.nan, volume_ratio=np.nan,
                   method="insufficient", validated=False, reason="")
        if prow is None or prow.empty:
            rec["reason"] = "parent has no rows"
            out.append(rec)
            continue
        near = prow[prow["frame"].between(f - OUTLIER_NEAR_FRAMES, f + OUTLIER_NEAR_FRAMES)]
        outlier_near = bool(near["is_outlier"].astype(bool).any())
        rec["outlier_near"] = outlier_near
        pv = prow[prow["_valid"]]
        n_pre, pre_m, pre_v = _side_values(pv, np.arange(f - WINDOW_FRAMES, f), before=True, mass_col=MASS_COL_, vol_col=VOL_COL_)
        n_post, post_m, post_v = _side_values(pv, np.arange(f, f + WINDOW_FRAMES + 1), before=False, mass_col=MASS_COL_, vol_col=VOL_COL_)
        rec.update(n_pre=n_pre, n_post=n_post, pre_mass_pg=pre_m, post_mass_pg=post_m,
                   pre_volume_um3=pre_v, post_volume_um3=post_v)
        if n_pre >= 1 and n_post >= 1 and pre_m > 0 and pre_v > 0:
            rec["mass_ratio"] = post_m / pre_m
            rec["volume_ratio"] = post_v / pre_v
        if not outlier_near:
            rec.update(method="direct", validated=True, reason="no tracker outlier within +-1 frame")
            out.append(rec)
            continue
        if n_pre < MIN_POINTS or n_post < MIN_POINTS:
            rec.update(method="insufficient", validated=False,
                       reason=f"outlier near event; only {n_pre} pre / {n_post} post valid points within +-{WINDOW_FRAMES} frames")
            out.append(rec)
            continue
        mr, vr = rec["mass_ratio"], rec["volume_ratio"]
        ok_m = MASS_RATIO[0] <= mr <= MASS_RATIO[1]
        ok_v = VOL_RATIO[0] <= vr <= VOL_RATIO[1]
        ok_d = abs(mr - vr) <= MAX_RATIO_DIFF
        if ok_m and ok_v and ok_d:
            rec.update(method="rescued", validated=True,
                       reason=f"outlier near event; mass ratio {mr:.2f}, volume ratio {vr:.2f} within limits")
        else:
            fails = [n for n, ok in (("mass_ratio", ok_m), ("volume_ratio", ok_v), ("ratio_diff", ok_d)) if not ok]
            rec.update(method="rejected", validated=False,
                       reason=f"outlier near event; failed {','.join(fails)} (mass {mr:.2f}, volume {vr:.2f})")
        out.append(rec)

    res = pd.DataFrame(out, columns=OUT_COLS).sort_values(["parent_id", "frame", "daughter_id"]).reset_index(drop=True)
    # rule 6: rescued candidates within 1 h of an already validated event of the same parent
    for p, g in res.groupby("parent_id"):
        last_valid_frame = None
        for idx in g.index:
            row = res.loc[idx]
            if row["validated"]:
                if row["method"] == "rescued" and last_valid_frame is not None \
                        and row["frame"] - last_valid_frame <= DUPLICATE_FRAMES:
                    res.loc[idx, ["method", "validated", "reason"]] = [
                        "duplicate", False,
                        f"rescued candidate within {DUPLICATE_FRAMES} frames of the validated event at frame {last_valid_frame}"]
                    continue
                last_valid_frame = int(row["frame"])
    return res


def run_lineage_dir(lo: Path, force: bool = False) -> Path | None:
    csv = lo / "lineage_data3D.csv"
    out = lo / "divisions_qc.csv"
    if not csv.exists():
        return None
    if out.exists() and not force and out.stat().st_mtime >= csv.stat().st_mtime:
        return out
    df = pd.read_csv(csv)
    if MASS_COL not in df.columns:
        raise RuntimeError(f"{csv} lacks {MASS_COL}: not a yellow-geometry lineage")
    frame_min = None
    params = lo / "lineage_run_params.json"
    dt_min = 5.0
    if params.exists():
        import json
        j = json.loads(params.read_text(encoding="utf-8"))
        frame_min = j.get("time_zero_frame", j.get("frame_min"))
        dt_min = float(j.get("time_interval_min") or 5.0)
    res = qc_channel(df, dt_min=dt_min, frame_min=frame_min)
    res.to_csv(out, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lineage-dir", default=None, help="one <...>/inference_out/lineage_out directory")
    ap.add_argument("--all", action="store_true", help="every production channel under the mask root")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.lineage_dir:
        out = run_lineage_dir(Path(args.lineage_dir), force=args.force)
        print("wrote", out)
        if out is not None:
            r = pd.read_csv(out)
            print(r["method"].value_counts().to_dict(), "| mother validated:",
                  int(r[r.is_mother_division & r.validated].shape[0]), "/", int(r.is_mother_division.sum()))
        return
    if args.all:
        import _retrack_260517_newmodel as chain
        n = 0
        for pos_dir in sorted(chain.MASK_ROOT.glob("Pos*"), key=lambda p: int(p.name[3:])):
            z = pos_dir / chain.REL
            if not z.is_dir():
                continue
            for ch in sorted(p for p in z.iterdir() if p.is_dir() and p.name.startswith("ch")):
                lo = ch / "inference_out" / "lineage_out"
                if chain.is_production(lo):
                    run_lineage_dir(lo, force=args.force)
                    n += 1
        print(f"division QC written for {n} channels")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
