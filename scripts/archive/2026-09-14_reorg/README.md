# archive/2026-09-14_reorg

Scripts moved out of `scripts/` on 2026-09-14 when the 260517 analysis method was fixed
(yellow-contour volumes, division QC, master dataset). Moved with `git mv`; history is intact.
Nothing here is imported by the remaining scripts (checked by `reorg_archive_2026_09_14.py`).

| folder | what | why superseded |
|---|---|---|
| legacy_imagej_ellipse_era | numbered 00..42 scripts, ImageJ ROI / ellipse-volume era, old pipeline runner | replaced by mask-based tracking (central_cell_lineage_tracker) |
| volume_method_exploration_2026-06_09 | contour smoothing / centerline iteration / EFD-vs-rod trials, minor-axis validation | decision 2026-09-14: only the yellow-contour rod + efd volumes are kept; comparison figure `_fig_volume_method_comparison_260517` |
| corrected_lineage_gold_standard_2026-06 | June corrected-volume toolkit (write_corrected_lineage, batch_recompute_axes, regenerate_all_figures) and the gold_standard cohort scripts | tracker now measures the adopted geometry at the source; gold_standard cohorts were a workaround for the noisy old model |
| prev_analysis_old_masters_2026-06_09 | `_fig_*` paper-figure attempts, overlays, grids, movies, one-off patches that read the June / 2026-09-11 masters | to be rebuilt on the yellow master (`v*_yellow`) with `channels.csv: analysis_recommended` |
| run_drivers_other_datasets | chains, watchers and recon drivers for 260405 / 260426 / 260508 / 260617 / 260810 / 260819 and the pre-yellow 260517 chains, plus their logs and pid files (`run_logs/`) | those runs are finished; the 260517 pipeline is `_chain_tiltfix_260517` + `_retrack_260517_newmodel` + `_finalize_yellow_260517` |
| ecc_tilt_benchmarks_2026-06 | ECC / sub-pixel / tilt / 2-pi / cell-free-channel benchmarks and diagnostics, drift figures | method settled (tilt_utils / ecc_utils / grid_subtract as used) |
| _archive_pre2026-03 | the previous `scripts/_archive/` folder | consolidated here |

Kept in `scripts/` on purpose although legacy: `recompute_axes_from_masks.py` (lazily imported by
`mask_volume_schematic.load_mother_crop`), `CursorVisualizer.py` (imported by focus tools), and the June
per-channel figure chain (`batch_all_channels`, `per_channel_figures`, `batch_figures`,
`central_cell_track_figures`, `mother_cell_cycle_stats`, `lineage_survival_analysis`).
