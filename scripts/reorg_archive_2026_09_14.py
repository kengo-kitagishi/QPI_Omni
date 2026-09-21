"""reorg_archive_2026_09_14.py - move superseded analysis scripts into scripts/archive/2026-09-14_reorg/.

Run with --dry-run (default) to see the plan and the import guard, then --apply to git-mv.
Nothing used by the running jobs (tilt-fix chain, Pos1-52 re-track, finalize), by the
acquisition pipeline (drift session / recon / grid subtract / calibration / Sept watchers)
or by the ops tools (Notion / Obsidian / sync / chat logger) is touched. A staying module
that imports an archive candidate blocks the move (guard).

Kept on purpose although legacy: recompute_axes_from_masks.py (mask_volume_schematic.load_mother_crop
imports it lazily), batch_all_channels / per_channel_figures / batch_figures / mother_cell_cycle_stats /
lineage_survival_analysis / central_cell_track_figures (referenced by docs and the June chain; to be
reviewed when the paper figures are rebuilt on the yellow master).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
ARCH = S / "archive" / "2026-09-14_reorg"

GROUPS = {
    "legacy_imagej_ellipse_era": """
        00_contours 03_sensitivity 05_copy_phase_files 23_plot_summary 24_ellipse_volume 25_roiset_from_zstack
        27_compare_volume_estimation_methods 28_compare_all_conditions 29_reprocess_with_thickness_filter
        30_plot_filtered_conditions 30_simple_mean_ri_analysis 31_create_filtered_visualizations 33_simplest_ellipse_ri
        40_qpi_noise_analysis 41_42_from_phase_planes 41_phase_diff_noise 42_phase_drift_from_ref 42_rolling_ball_backsub
        arrconv qpi_analysis CursorVisualizer run_pipeline""",
    "volume_method_exploration_2026-06_09": """
        _smooth_chaikin_savgol _smooth_cubic_spline_periodic _smooth_efd _smooth_raw_baseline _smooth_sweep_f1800
        _smooth_upsample _iterate_centerline_f1800 _contour_singlepass_f1800 _singlepass_raw_vs_smoothed_f1800
        _compare_volume_methods_f1800 _preview_efd_vs_rod _debug_schematic_caps _plot_minor_axis_pos1_z001
        _plot_minor_axis_pos1_z001_ch09 _overlay_efd_on_panelA_strip validate_minor_axis_volume_smoothing
        rod_axis_correction mask_measurement_qc batch_recompute_efd""",
    "corrected_lineage_gold_standard_2026-06": """
        gold_standard gold_standard_4way_comparison gold_standard_cycle_aligned_vol_mass_ri gold_standard_minor_axis_cycle
        gold_standard_phase1_homeostasis gold_standard_rod_corrected_cycle batch_recompute_axes
        write_corrected_lineage regenerate_all_figures _label_corrected _retrack_corrected_260517
        overlay_gold_standard_and_phase1_dead overlay_elongation_pair_gsd""",
    "prev_analysis_old_masters_2026-06_09": """
        _fig_added_within_cycle _fig_birth_aligned_elong_vs_normal _fig_birth_vol_vs_ri _fig_dead_massvol_decoupling
        _fig_death_aligned_ri_vs_shape _fig_elongation_mass_fit _fig_growthrate_by_generation_3groups
        _fig_growthrate_dead_vs_alive _fig_growthrate_swelling_individual _fig_homeostasis_per_lineage
        _fig_lastdiv_aligned_swelling _fig_lastdiv_ri_shape_individual _fig_mass_fit_all_dead _fig_meanri_divlines_swelling
        _fig_panelA_cellcycle _fig_panelG_survivors _fig_predeath_growthrate _fig_revival_regrowth_rate_lag
        _fig_revival_return_curves _fig_revived_convergence_cv _fig_revived_growthrate_decoupling _fig_ri_vs_period
        overlay_mean_sd_band_full_timecourse overlay_mean_sd_band_phase1_normalized_ri overlay_elongation_death_pair
        overlay_mother_revived_vs_dead grid_revived_mother_individual grid_phase2_dead_mother_individual
        analyze_starvation_entry_cell_cycle analyze_death_cell_cycle ri_drop_starvation lineage_claim_ladder
        phase2_survivor_efd_traces quality_check_all_mother_revived quality_check_division_intervals
        extract_well_tracked_mothers check_phase2_dead_tracking_coverage split_lineage_csvs_by_scope npz_to_lineage_csv
        batch_per_channel_traces batch_volume_trace_overlay mothercell_cross_channel_runner _animate_phase2_cells
        _animate_vol_ri_f_ch _batch_phase2_movies _batch_swelldeath_movies _build_revival_pergen_table
        _label_phase1dead_sheets _montage_phase1dead_sheets _regen_phase1dead _survey_phase1_dead_deathmode
        _run_binwidth_sweep _run_clean_0020 _run_ri_at_switch_efd _run_ri_at_switch_exact_efd _run_ri_drop_matched
        _run_viridis_ch10 _export_viridis_pngs_ch10 _rewrite_viridis_cbar_ch10 _patch_ctx_binsweep _patch_ctx_ridrop_matched
        _patch_ctx_riswitch _patch_ctx_riswitch_exact _inspect_cycles _inspect_longsegs _dump_track _dump_track_ds
        _homeostasis_260426_z001 qpi_fig_03_lineage_analysis qpi_fig_04_growth_oscillation qpi_fig_lineage_pos18_ch09_125_197
        build_inferno_hyperstack render_crop_sub_inferno""",
    "run_drivers_other_datasets": """
        _chain_recon_pipeline_260405_acute _chain_recon_pipeline_260426 _chain_resilient_260517
        _chain_resume_pipeline_260405_acute _chain_resume_step_d_260405_acute _chain_seg_260426_zstack _chain_seg_260517
        _recover_output_phase_260405_acute _run_seg_260514 _wait_retrack_then_publish_260517
        wait_grid_recon_then_correct_0pergluc watch_grid_then_recon_260810 watch_grid_then_recon_260819
        watch_grid_then_recon_260819_b2 watch_grid_then_recon_260819_b3 run_recon_0per2_partial_260617
        run_recon_0per_260617 run_recon_0per_all_260617 run_recon_continuation_260617 run_redetect_0per_260617
        run_260310_grid_and_prepare _cleanup_260405_acute_intermediates _fast_grid_cleanup
        _gen_channel_rois_grid_260405_acute _make_ri_calibration_260405_acute _measure_offaxis_260405 _apply_drift_to_pos
        _compute_0per_delta _prep_0per_correct _reverse_0per_correct _check_grid_actual_dy _analyze_error_dy_pixel_scale
        regenerate_grid_subtract_260508 batch_correct_0pergluc_260508""",
    "ecc_tilt_benchmarks_2026-06": """
        bench_cellbias_estimators bench_cellfree_discrimination bench_cellfree_score_from_delta bench_ecc_iter_figure
        bench_ecc_vs_sgpeak bench_freech_compare bench_perchannel_bias bench_recalibrate bench_reconstruction
        bench_subpix_methods bench_subtract_ab bench_subtract_ab_analyze bench_subtract_ab_visual bench_timelapse_pipeline
        analyze_bg_ytilt_vs_ecc analyze_ch_xtilt_corrected_ecc analyze_ch_xtilt_vs_ecc analyze_ch_ytilt_corrected_ecc
        analyze_tilt_correct_precision analyze_tilt_corrected_ecc diagnose_tilt_correct_ecc diagnostic_ecc_half_compare
        eval_tilt_correct_ecc_precision ecc_channel_inspect test_ecc_precision test_gaussian2d_align test_crop_sweep
        test_iarpls_tilt_subtract test_seg_volume_overlay test_eroded_volume_overlay compare_subtract_crop
        build_collar_compare build_collar_compare_pos0sub drift_cellfree_channel_bias fix_2pi_combo fix_2pi_residue
        fix_2pi_temporal find_permanent_shifts compute_shifts_bgroi compute_center_correction gen_tilt_timelapse_compare
        generate_aligned_raw negate_phase_tree plot_center_line_profiles plot_grid_sub_center_profiles
        plot_shift_invariant_profiles plot_threshold_decision plot_values_comparison quick_compare_v1v2 quick_grid_sub
        quick_preview_grid_sub render_pos0sub_focus show_jump_examples sim_drift_gain analyze_kalman_gain
        analyze_stage_repeatability timelapse_iarpls_bgsub timelapse_plane_bgsub visualize_bg_tilt align_and_subtract_simple
        analyze_drift_control analyze_channel_drift qpi_fig_drift_after_quiet qpi_fig_drift_lines
        qpi_fig_drift_nearest_grid_24h qpi_fig_drift_window""",
}
SIDE_EXTRA = ["scheduled_260504_recon_calib.bat", "run_recon_then_calib_260331.bat", "run_pipeline_260327.bat",
              "pipeline_schedule_260327.json", "_chain_seg_260517_pos1.log",
              "_chain_seg_260426_zstack.log.prev", "_chain_seg_260426_zstack.err.log.prev"]
IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_]\w*)\s+import|import\s+([A-Za-z_]\w*))", re.M)


def main() -> None:
    apply = "--apply" in sys.argv
    plan = {g: v.split() for g, v in GROUPS.items()}
    cand = {n for v in plan.values() for n in v}
    present = {p.stem for p in S.glob("*.py")}
    missing = sorted(cand - present)
    stay = present - cand
    violations = []
    for n in sorted(stay):
        src = (S / f"{n}.py").read_text(encoding="utf-8", errors="replace")
        for m in IMPORT_RE.finditer(src):
            mod = m.group(1) or m.group(2)
            if mod in cand:
                violations.append((n, mod))
    side = []
    for n in plan["run_drivers_other_datasets"] + plan["corrected_lineage_gold_standard_2026-06"]:
        side += [p for p in S.glob(f"{n}.*") if p.suffix != ".py"]
    side += [S / x for x in SIDE_EXTRA if (S / x).exists()]
    print(f"candidates {len(cand)} (present {len(cand & present)}, missing {len(missing)}), staying {len(stay)}")
    for g, v in plan.items():
        print(f"  {g}: {len([n for n in v if n in present])}")
    print("missing:", missing)
    print("side files:", len(side))
    print("IMPORT GUARD violations:", violations or "none")
    if violations:
        print("refusing to move the imported candidates:", sorted({m for _, m in violations}))
        cand -= {m for _, m in violations}
    if not apply:
        print("dry run only; add --apply to move")
        return
    tracked = set(subprocess.run(["git", "ls-files", "--", "."], cwd=S, capture_output=True, text=True).stdout.split())

    def mv(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        rel = src.relative_to(S).as_posix()
        if rel in tracked:
            subprocess.run(["git", "mv", "-k", str(src), str(dst)], cwd=S, check=True)
        if src.exists():
            shutil.move(str(src), str(dst))

    n_moved = 0
    for g, v in plan.items():
        for n in v:
            p = S / f"{n}.py"
            if n in cand and p.exists():
                mv(p, ARCH / g / p.name)
                n_moved += 1
    for p in side:
        if p.exists():
            mv(p, ARCH / "run_logs" / p.name)
            n_moved += 1
    old = S / "_archive"
    if old.exists():
        dst = S / "archive" / "_archive_pre2026-03"
        subprocess.run(["git", "mv", "-k", str(old), str(dst)], cwd=S)
        if old.exists():
            shutil.move(str(old), str(dst))
        print("consolidated _archive/ -> archive/_archive_pre2026-03/")
    print("moved", n_moved, "files into", ARCH)


if __name__ == "__main__":
    main()
