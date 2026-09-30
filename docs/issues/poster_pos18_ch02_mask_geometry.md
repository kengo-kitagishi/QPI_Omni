# ICBBS poster: Pos18 ch02 mask contour, long axis, and slice widths

## Goal

Revise the cell-cycle image strip used in the ICBBS poster Methods section. The current reference image is `docs/issues/assets/pos18_ch02_mask_overlay_reference.png` (`fig_qpi_segmentation_overlay` v001, Pos18 ch02, starting near frame 633). It shows reconstructed phase images with cyan mask outlines and a 5 µm scale bar. The analysis PC has the underlying phase and mask dataset; this checkout does not.

Create two outputs from the **same selected frames and cell masks**:

1. A corrected phase-image strip with the mask contour visibly aligned to each cell.
2. A companion strip that overlays the measured long axis and a legible subset of slice short axes on that same corrected contour. These are the chords used for the section geometry, not a fitted ellipse's major/minor axes.

Do not redraw the contour or axes by eye on the exported PNG. Generate them from the masks, and keep the frame and cell identity aligned with the phase images.

## Starting point on the analysis PC

1. Locate the original Pos18 ch02 figure source and its list of frames/crops. The registered reference is `fig_qpi_segmentation_overlay` v001 in figure-hub; the filename indicates `Panel-A_cell-cycle_strip_Pos18_ch02_633-...`. Confirm the full frame list from its source or figure logger metadata before rendering. Do not infer frame intervals from the montage.
2. Locate the matching reconstructed phase crops, label masks (`inference_out/*_masks.tif`), and lineage rows. For each panel record the experiment, `Pos18`, `ch02`, frame, mask path, cell label, crop box, and phase path in a manifest. The phase and mask images must share pixel coordinates; validate their shapes and alignment before cropping.
3. Determine why the current cyan outline needs correction by comparing it against the label mask on several frames, including the shortest and longest cells. Check label selection, crop origin, coordinate transforms, contour smoothing and offset. Record the cause and fix before replacing the line.

## Geometry and display rules

- Use the **current yellow-contour method** from the analysis pipeline: mask boundary smoothed with EFD `K=6` and `contour_offset_px=0.0` (see `CLAUDE.md`). Do not reuse the older 0.5 px inward-offset contour. If the current tracker has a canonical contour helper, call it so the poster geometry matches `volume_um3_efd`.
- `scripts/mask_volume_schematic.py:efd_section_geometry()` already returns the smoothed contour, midpoint-updated long-axis polyline, and the two endpoints of each perpendicular slice. Reuse or reconcile this with the canonical tracker implementation; inspect numerical agreement on representative frames before treating the overlay as an illustration of the reported measurements.
- Overlay the long axis in a distinct warm color and the slice short axes in a lighter contrasting color. Subsample slices **only for display**; retain the full per-slice geometry for measurement. Keep line widths and labels readable at the poster's printed size.
- The 5 µm scale bar must be derived from `lineage_run_params.json` `pixel_size_um`, remain inside the image, and be the same physical length in every panel. Use one fixed phase colormap and display range across the strip, and preserve the original time labels.
- No schematic values or hand-placed cell outlines. If an endpoint cannot be robustly intersected with the corrected contour, flag that frame instead of silently drawing a plausible chord.

## Deliverables

- Corrected contour-only strip and contour + long-axis + slice-short-axis strip, as poster-ready PNG and vector/PDF overlay where practical.
- Source-data manifest with the selected frame IDs, mask labels, paths, crop boxes, pixel size, contour settings, plotted long-axis coordinates and plotted slice endpoints. Follow `docs/FIGURE_SPEC.md` for caption and provenance; log each figure through `figure_logger` and register the final version in figure-hub.
- Short caption defining the contour, long axis and slice widths operationally. State that the slices are drawn sparsely for legibility while measurement uses all valid sections.
- Side-by-side QC for at least the first, middle, shortest and longest selected cell images. Confirm that the contour follows the cell boundary, long axis stays inside it, and slice endpoints terminate on it. Report any rejected frames.

## Acceptance

- Every displayed frame maps to a verified phase image and cell label; no frame is inferred from the reference PNG.
- The outline correction is explained and visually verified, including the cell tips and any division neck.
- Long-axis and displayed slice endpoints come from the same corrected mask geometry. No chord visibly overshoots the contour.
- The two strips use matching crops, times, color scaling and scale bar, so readers can compare them directly.
- The figure is suitable for the ICBBS poster Methods panel and does not imply that the overlaid axes were manually measured.

The Mac checkout can review the reference image and code, but the dataset-dependent render and QC must run on the analysis PC.
