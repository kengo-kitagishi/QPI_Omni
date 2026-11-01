# 261030 YE — consolidated lineage tables (frame 2–430)

Produced 2026-11-02 by `scripts/run_dataset_pipeline.py datasets/261030.yaml --stages track,qc,consolidate`
(omnipose env). Copy of `C:/QPI_analysis/261030/seg/_lineage_consolidated/`.

| file | content |
|---|---|
| `all_cells_lineage_data3D.csv.gz` | every cell × frame (pos, ch, cell_id, frame, time_h, area, long/short axis, volume_um3_efd, mass_pg, density_pg_um3, mean_ri, is_outlier, touches_border, …). 742,956 rows. Columns: `docs/LINEAGE_DATAFRAME_SCHEMA.md` |
| `all_cells_divisions_qc.csv.gz` | division candidates with mass/volume-ratio QC (`validated`) |
| `all_cells_clist.csv.gz` | per-cell list (birth/death frame, parent) |
| `all_cells_lineage_bad_frames.csv.gz` | measurements on bad frames (excluded from tracking) |
| `channel_index.csv` | per channel counts; `edge_channel` marks ch00/ch11 |
| `channel_whitelist.json` | empty traps used for drift correction — **not segmented** |
| `manifest.json` | pipeline provenance |

## Conditions
- YE medium, 5 min/frame, single z, YE grid `D:/AquisitionData/Kitagishi/261029/recon_YE_grid`.
- 578 non-whitelist channels segmented; 528 had cells and were tracked. Edge traps ch00/ch11 are in the table but
  are excluded from analysis.
- **Frames after 430 are excluded (contamination).** `death_frame` = 430 means the observation ended, not that the
  cell died.
- Dry mass and concentration come from the phase integral only; they do not depend on the medium RI.
- **`mean_ri` uses a placeholder medium RI 1.333.** The YE value measured from the PDMS-wall/trap phase step is
  n_YE = 1.33446 (95% CI 1.33423–1.33464; `scripts/ri_ye_from_pdms_step_261029.py`), so true mean_ri =
  mean_ri + 0.00146.
- Two death modes are seen in the traces (user, 2026-11-02): swelling death (division stops, mass keeps rising slowly,
  concentration rises) and death while elongating.
