"""Extract mother vs. sibling-daughter mass/volume/density around each validated
260908 division, for the sister-density-asymmetry question (fig 7 follow-up).

Mother keeps cell_id=0 for the whole run (central-cell convention); at each
division a direct daughter appears with parent_id==0, in_tree==True, and
birth_frame == the division frame used by the review package. That daughter
is tracked for many frames afterwards (not flushed immediately), so we can
read its mass/volume/density for a few frames right after division the same
way we read the mother's.

Only channels marked included in the round-3 review are used, and only
divisions the review marked accepted (same division set figure 7 was built
from).
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full-csv', type=Path,
        default=Path('E:/260908_outside_quad/seg/_lineage_consolidated/all_cells_lineage_data3D.csv.gz'))
    ap.add_argument('--review', type=Path,
        default=Path('E:/260908_outside_quad/seg/_qc/review_20260925/gallery_100h_v3'))
    ap.add_argument('--n-frames', type=int, default=3, help='frames after division to keep (frame offsets 0..n-1)')
    ap.add_argument('--out', type=Path,
        default=Path('C:/Users/QPI/Documents/QPI_Omni/qc_review/260908_outside_quad/division_sibling_mass_v3.csv'))
    args = ap.parse_args()

    channels = pd.read_csv(args.review / 'channel_review.csv')
    included = set(zip(channels.loc[channels.included, 'pos'], channels.loc[channels.included, 'ch']))
    events = pd.read_csv(args.review / 'division_events_review.csv')
    acc = events[events.accepted & events.apply(lambda r: (r.pos, r.ch) in included, axis=1)]
    print(f'{len(acc)} accepted divisions across {len(included)} included channels')

    full = pd.read_csv(args.full_csv)
    full = full[full[['pos', 'ch']].apply(tuple, axis=1).isin(included)].reset_index(drop=True)
    by_channel = {key: g for key, g in full.groupby(['pos', 'ch'])}

    cols = ['time_h', 'mass_pg_efd', 'volume_um3_efd', 'mean_ri_efd',
            'density_pg_um3_efd', 'is_outlier', 'touches_border']
    rows = []
    n_unmatched = 0
    for r in acc.itertuples():
        pos, ch, F = r.pos, r.ch, int(r.frame)
        g = by_channel.get((pos, ch))
        if g is None:
            n_unmatched += 1
            continue
        daughters = g[(g.parent_id == 0) & (g.in_tree) & (g.birth_frame == F)]
        daughter_ids = daughters.cell_id.unique()
        if len(daughter_ids) != 1:
            n_unmatched += 1
            continue
        daughter_id = int(daughter_ids[0])
        for role, cid in [('mother', 0), ('daughter', daughter_id)]:
            cell_rows = g[(g.cell_id == cid) & (g.frame >= F) & (g.frame < F + args.n_frames)]
            for _, row in cell_rows.iterrows():
                rows.append(dict(pos=pos, ch=ch, division_frame=F, role=role, cell_id=cid,
                                  frame_offset=int(row.frame) - F, frame=int(row.frame),
                                  **{c: row[c] for c in cols}))

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding='utf-8-sig')

    manifest = dict(
        source_full_csv=str(args.full_csv), source_full_sha256=digest(args.full_csv),
        review_dir=str(args.review), review_manifest_sha256=digest(args.review / 'manifest.json'),
        n_divisions_considered=int(len(acc)), n_divisions_matched=int(len(acc) - n_unmatched),
        n_divisions_unmatched=int(n_unmatched), n_frames_per_role=args.n_frames,
        n_included_channels=len(included), output_rows=len(out),
        method='mother keeps cell_id=0 for the whole run; direct daughter identified by '
               'parent_id==0, in_tree==True, birth_frame==division frame F (same frame review '
               'used as the cycle boundary). Frames F..F+n-1 kept for both, subject to availability '
               '(death_frame / dataset end). is_outlier/touches_border kept as columns, not filtered out.')
    (args.out.parent / 'division_sibling_mass_v3_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')

    print(f'matched {len(acc) - n_unmatched}/{len(acc)} divisions, {len(out)} rows -> {args.out}')
    if n_unmatched:
        print(f'WARNING: {n_unmatched} divisions had 0 or >1 daughter candidates and were skipped')

if __name__ == '__main__':
    main()
