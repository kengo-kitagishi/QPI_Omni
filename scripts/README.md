# scripts/ index (2026-09-14 reorganisation)

Superseded code lives in `archive/2026-09-14_reorg/` (see its README). Everything below is current or
still referenced. Each script's docstring is the primary documentation; `docs/PROTOCOL_TIMELAPSE.md` is
the pipeline protocol. Run Python with the pinned `omnipose` env (`environment/`; on this PC
`C:/Users/QPI/anaconda3/envs/omnipose/python.exe`). Dataset settings live in `datasets/<id>.yaml`,
production checkpoints in `models/`.

## Dataset driver (entry point for any experiment)

- `run_dataset_pipeline.py` - `datasets/<id>.yaml` -> seg, track, qc, consolidate, publish (`--plan` dry run; resumable)
- `seg_omnipose.py` - GPU Omnipose inference for one dataset layout (called per Pos by the driver)

## Tracking, master dataset and the paper package (current)

- `_chain_tiltfix_260517.py`
- `_finalize_yellow_260517.py`
- `_make_ri_calibration_260517.py`
- `_retrack_260517_newmodel.py`
- `build_phase1_dataset_260517.py`
- `calibrate_ri.py`
- `central_cell_lineage_tracker.py`
- `division_qc_260517.py`
- `make_channel_classification_template.py`
- `publish_master_260517.py`
- `qpi_paths.py`
- `refit_tilt_right_260517.py`
- `ri_calibration.py`
- `run_seg_260517_gpu.py`

## QC and figures on the master (current)

- `_fig_mother_lineage_qc_260517.py`
- `_fig_switch_frame_check_260517.py`
- `_fig_volume_method_comparison_260517.py`
- `figure_logger.py`
- `lineage_html_gallery_260517.py`

## Cell geometry modules

- `mask_morphology.py`
- `mask_volume_schematic.py`
- `recompute_axes_from_masks.py`

## Segmentation and Omnipose training

- `06_seg_npy_to_masks.py`
- `07_segmentation.py`
- `08_train.py`
- `26_horizontal_flip.py`
- `_build_trainset_260517.py`
- `_seg_overlay_f_ch.py`
- `checkpoint_eval.py`
- `checkpoint_overlay_runner.py`
- `checkpoint_watcher.py`
- `mask_overlay_check.py`
- `monitor_train.py`
- `run_omnipose_chm_batch.py`
- `sample_train_frames.py`

## Acquisition: drift session, grid calibration, reconstruction

- `01_realtime_visibility_monitor.py`
- `09_single_reconstruction.py`
- `CursorVisualizer.py`
- `align_timelapse_pos.py`
- `analyze_drift_outliers.py`
- `analyze_drift_outliers_v2.py`
- `batch_grid_calibration.py`
- `batch_reconstruction_grid.py`
- `calibrate_grid_pos.py`
- `calibrate_grid_pos_per_pos.py`
- `calibrate_grid_positions.py`
- `compute_drift_online.py`
- `focus_check_subtract.py`
- `generate_grid_pos.py`
- `optical_config.py`
- `parallel_calibrate.py`
- `parse_bsh_log.py`
- `plot_cumdrift_260517.py`
- `plot_drift_summary.py`
- `plot_grid_calibration.py`
- `plot_grid_true_positions.py`
- `prepare_drift_session.py`
- `qpi.py`
- `qpi_01_focus_setup.py`
- `qpi_02_single_image.py`
- `qpi_04_alignment_diff.py`
- `qpi_05_focus_analysis.py`
- `qpi_common.py`
- `reconstruct_grid_corner.py`
- `resume_drift_session.py`
- `rollback_drift_state.py`
- `run_recon_batches_260908.py`
- `run_recon_cycle.py`
- `scheduled_recon_and_calibrate.py` — grid recon → channel detect → calibration（watch_grid_then_recon_* と skill batch-recon-calibrate が呼ぶ）
- `visualize_drift_log.py`
- `visualize_grid_true_positions.py`
- `visualize_timelapse_qc.py`
- `watch_grid_then_recon_260906.py`
- `watch_grid_then_recon_260906_b2.py`
- `watch_grid_then_recon_260908.py`

## Preprocessing: channel crops, ECC shifts, grid subtraction, 0% correction, tilt

- `apply_final_2d_flatten.py`
- `apply_oob_mask.py`
- `batch_compute_pos_shifts_260517.py`
- `batch_grid_subtract_260517.py`
- `batch_pipeline_all_pos.py`
- `batch_pos_shifts_posparallel_260517.py`
- `channel_crop.py`
- `complete_crop_sub.py`
- `compute_pos_shifts.py`
- `correct_0pergluc.py`
- `ecc_utils.py`
- `extract_bad_frames.py`
- `extract_timelapse_delta.py`
- `grid_subtract.py`
- `prep_channel_rois.py`
- `shift_visualize.py`
- `tilt_utils.py`

## Legacy per-channel figure chain (June 2026; review when paper figures are rebuilt)

- `batch_all_channels.py`
- `batch_figures.py`
- `central_cell_lineage_overlay.py`
- `central_cell_track_figures.py`
- `lineage_survival_analysis.py`
- `mother_cell_cycle_stats.py`
- `per_channel_figures.py`
- `qpi_fig_01_generate_panels.py`
- `qpi_fig_01_panel.py`
- `qpi_fig_01_reconstruction_procedure.py`
- `qpi_fig_02_visibility.py`

## Disk and data housekeeping

- `cleanup_verified_raw.py`
- `delete_ph000_frames_over_index.py`
- `free_c_after_recon.py`
- `pos_archive_watcher.py`
- `scheduled_backup_copy.py`
- `verify_backup_then_delete.py`

## Ops: logging, Notion, Obsidian, ClickUp

- `_find_db.py`
- `_notion_archive_today.py`
- `_notion_create_page.py`
- `_notion_page.py`
- `_notion_save_path_memo.py`
- `_notion_test.py`
- `_parse_jsonl.py`
- `chat_logger.py`
- `clickup_helper.py`
- `figure_inbox_to_obsidian.py`
- `generate_daily_log.py`
- `jsonl_to_obsidian.py`
- `notion_plan_save.py`
- `notion_save_session.py`
- `notion_setup_type.py`
- `reorg_archive_2026_09_14.py`
- `session_activity_logger.py`
- `session_db.py`
- `session_to_notion.py`
- `weekly_report_hub.py`

## Subdirectories

- `archive/` superseded scripts (dated subfolders; `_archive_pre2026-03/` is the old `_archive/`)
- `sync/` scheduled log sync, `tests/` unit tests, `run_logs/`, `results/` local outputs
