"""Export every tracked cell (mother, daughters, further descendants and
out-of-tree cells) in the reviewed 260908 channels, for cross-PC analysis.

The review package (gallery_100h_v3) only carries the mother (cell_id == 0).
This writes the same per-frame measurements for all cells in the included
channels, plus the tracker's per-cell summary, division QC and bad-frame
tables, so sisters, position in the channel (rank) and non-mother cycles can
be analysed on a machine that has no access to E:.

How to read the per-frame table (central_cell_lineage_tracker.py):
  - rank 1 is the cell closest to the image x-centre (the mother); rank grows
    outward. centroid_x_px is kept so the order can be recomputed.
  - At every division the inner product keeps the parent's cell_id and the
    outer product gets a new cell_id with parent_id = parent and
    birth_frame = division frame. This holds for all cells, not only the mother.
  - parent_id == -1 with in_tree == 0 is a cell of unknown origin (present at
    the first frame at rank >= 2, or picked up after a tracking break).
  - On is_outlier frames the source has NaN in volume_um3_efd and mass_pg_efd.
    total_phase, area_px and the axes are still measured there, so mass can be
    recomputed as total_phase * mass_factor; volume cannot.

Nothing is filtered by quality, and frames after 100 h or after a channel's
cutoff are kept (channel_review.csv / manifest.json of the review package
define those cuts). Non-mother divisions were not reviewed by eye.

Columns that are exact functions of kept columns are dropped to keep the
file small; the manifest records the largest deviation found in the source:
    time_h             = (frame - 2) / 12
    area_um2           = area_px * pixel_size_um**2
    density_pg_um3_efd = mass_pg_efd / volume_um3_efd
    mean_ri_efd        = n_medium_used + 0.18 * density_pg_um3_efd
    mass_pg            = mass_pg_efd = total_phase * mass_factor (where not NaN)
Rod-era columns (volume_um3_rod, mean_ri, density_pg_um3) are not exported.

Outputs (in --out):
    all_cells_frames_v3.csv.gz            (or ..._partNofK.csv.gz above --max-mb)
    all_cells_clist_v3.csv.gz             (if the source exists)
    all_cells_divisions_qc_v3.csv.gz
    all_cells_bad_frames_v3.csv.gz        (if the source exists)
    all_cells_v3_manifest.json

Example:
    python scripts/extract_all_cells_260908.py
"""
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np
import pandas as pd

PIXEL_UM = 0.34567514677103717
MASS_FACTOR = 0.658 * PIXEL_UM ** 2 / (2 * np.pi * 0.00018) * 1e-3  # pg per (rad * px)
KEEP = ['pos', 'ch', 'cell_id', 'parent_id', 'in_tree', 'birth_frame', 'death_frame', 'frame', 'rank',
        'mask_label', 'area_px', 'long_axis_um', 'short_axis_um', 'centroid_x_px', 'centroid_y_px',
        'total_phase', 'volume_um3_efd', 'mass_pg_efd', 'is_outlier', 'touches_border']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def only_included(df, included):
    return df[pd.MultiIndex.from_frame(df[['pos', 'ch']]).isin(included)].copy()


def max_dev(df, col, expected, relative=True):
    """Largest |source column - recomputed value| over rows where both are finite."""
    if col not in df.columns:
        return 'column absent in source'
    d = np.abs(df[col].to_numpy(float) - np.asarray(expected, float))
    if relative:
        d = d / np.abs(np.asarray(expected, float))
    d = d[np.isfinite(d)]
    return float(d.max()) if len(d) else None


def write_parts(df, out, stem, max_mb):
    """Write one gz file, or several split by Pos when it would exceed max_mb."""
    kw = dict(index=False, float_format='%.6g')
    path = out / f'{stem}.csv.gz'
    df.to_csv(path, **kw)
    size = path.stat().st_size / 1e6
    if size <= max_mb:
        return [path]
    path.unlink()
    k = math.ceil(size / max_mb * 1.15)
    rows = df.groupby('pos', sort=False).size()
    part = np.minimum((rows.cumsum().shift(fill_value=0) / rows.sum() * k).astype(int), k - 1)
    paths = []
    for i in range(k):
        p = out / f'{stem}_part{i + 1}of{k}.csv.gz'
        df[df.pos.map(part) == i].to_csv(p, **kw)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=Path('E:/260908_outside_quad/seg/_lineage_consolidated'))
    ap.add_argument('--review', type=Path,
        default=Path('E:/260908_outside_quad/seg/_qc/review_20260925/gallery_100h_v3'))
    ap.add_argument('--out', type=Path,
        default=Path('C:/Users/QPI/Documents/QPI_Omni/qc_review/260908_outside_quad/all_cells_v3'))
    ap.add_argument('--max-mb', type=float, default=40.0, help='split the frames file by Pos above this size (GitHub warns at 50 MB)')
    args = ap.parse_args()

    full_csv = args.root / 'all_cells_lineage_data3D.csv.gz'
    channels = pd.read_csv(args.review / 'channel_review.csv')
    included = set(zip(channels.loc[channels.included, 'pos'], channels.loc[channels.included, 'ch']))

    full = only_included(pd.read_csv(full_csv), included)
    missing = [c for c in KEEP if c not in full.columns]
    if missing:
        raise SystemExit(f'columns missing from {full_csv}: {missing}\navailable: {list(full.columns)}')

    density = full.mass_pg_efd / full.volume_um3_efd
    n_med = full.n_medium_used if 'n_medium_used' in full.columns else np.nan
    checks = {
        'time_h = (frame - 2) / 12': max_dev(full, 'time_h', (full.frame - 2) / 12, relative=False),
        'area_um2 = area_px * pixel_size_um**2': max_dev(full, 'area_um2', full.area_px * PIXEL_UM ** 2),
        'density_pg_um3_efd = mass_pg_efd / volume_um3_efd': max_dev(full, 'density_pg_um3_efd', density),
        'mean_ri_efd = n_medium_used + 0.18 * density_pg_um3_efd': max_dev(full, 'mean_ri_efd', n_med + 0.18 * density),
        'mass_pg = mass_pg_efd': max_dev(full, 'mass_pg', full.mass_pg_efd),
        'mass_pg_efd = total_phase * mass_factor': max_dev(full, 'mass_pg_efd', full.total_phase * MASS_FACTOR),
    }

    full['posn'] = full.pos.str[3:].astype(int)
    full = full.sort_values(['posn', 'ch', 'cell_id', 'frame'])
    frames = full[KEEP].copy()
    for c in ['in_tree', 'is_outlier', 'touches_border']:
        frames[c] = frames[c].astype(int)

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob('all_cells_frames_v3*.csv.gz'):
        old.unlink()
    paths = write_parts(frames, args.out, 'all_cells_frames_v3', args.max_mb)

    side, side_info = [], {}
    for src_name, dst_name in [('all_cells_clist.csv.gz', 'all_cells_clist_v3.csv.gz'),
                               ('all_cells_divisions_qc.csv.gz', 'all_cells_divisions_qc_v3.csv.gz'),
                               ('all_cells_lineage_bad_frames.csv.gz', 'all_cells_bad_frames_v3.csv.gz')]:
        src = args.root / src_name
        if not src.exists():
            side_info[src_name] = 'absent in source'
            continue
        try:
            t = only_included(pd.read_csv(src), included)
        except pd.errors.EmptyDataError:
            side_info[src_name] = 'empty in source (0 bytes decompressed)'
            continue
        t.to_csv(args.out / dst_name, index=False)
        side.append(args.out / dst_name)
        side_info[src_name] = dict(rows=int(len(t)), columns=list(t.columns), sha256=digest(src))

    per_frame = frames.groupby(['pos', 'ch', 'frame']).size()
    cells = frames.drop_duplicates(['pos', 'ch', 'cell_id'])
    hidden = frames.is_outlier.astype(bool) | frames.touches_border.astype(bool)
    manifest = dict(
        source_full_csv=str(full_csv), source_full_sha256=digest(full_csv),
        review_dir=str(args.review), review_manifest_sha256=digest(args.review / 'manifest.json'),
        n_included_channels=len(included), n_channels_found=int(frames.groupby(['pos', 'ch']).ngroups),
        frame_rows=int(len(frames)), n_cells=int(len(cells)), n_cells_in_tree=int(cells.in_tree.sum()),
        n_cells_unknown_parent=int((cells.parent_id < 0).sum()) - int((cells.cell_id == 0).sum()),
        frame_min=int(frames.frame.min()), frame_max=int(frames.frame.max()),
        cells_per_channel_frame={q: float(per_frame.quantile(v)) for q, v in [('p05', .05), ('median', .5), ('p95', .95), ('max', 1)]},
        rank_counts={int(k): int(v) for k, v in frames['rank'].value_counts().sort_index().items()},
        n_outlier_rows=int(frames.is_outlier.sum()), n_touches_border_rows=int(frames.touches_border.sum()),
        n_rows_mass_nan=int(frames.mass_pg_efd.isna().sum()), n_rows_volume_nan=int(frames.volume_um3_efd.isna().sum()),
        n_rows_mass_nan_not_flagged=int((frames.mass_pg_efd.isna() & ~hidden).sum()),
        n_rows_total_phase_nan=int(frames.total_phase.isna().sum()),
        columns=KEEP, float_format='%.6g', pixel_size_um=PIXEL_UM, mass_factor_pg_per_rad_px=MASS_FACTOR,
        n_medium_used_unique=([float(v) for v in pd.unique(full.n_medium_used.dropna())] if 'n_medium_used' in full.columns else None),
        source_columns=[c for c in full.columns if c != 'posn'],
        dropped_columns_max_deviation_in_source=checks, side_tables=side_info,
        files={p.name: dict(mb=round(p.stat().st_size / 1e6, 2), sha256=digest(p)) for p in paths + side},
        note='no quality filtering; frames beyond the review window (end_frame / cutoff_h) are included; '
             'non-mother divisions are tracker calls checked by the automatic mass/volume QC only')
    (args.out / 'all_cells_v3_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('\nfiles to commit:')
    for p in paths + side + [args.out / 'all_cells_v3_manifest.json']:
        print(f'  {p}  ({p.stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
