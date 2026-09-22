"""Replay validation: physical frame IDs, unchanged cell signal, safe resumption."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reprocess_bg_from_shifts as replay


def synthetic_run(tmp_path):
    source, tl_root, grid_root, output = [tmp_path / name for name in ("old", "tl", "grid", "new")]
    metadata = source / "Pos1/output_phase/channels"
    grid_node = grid_root / "Pos1_x+0_y+0"
    y, x = np.mgrid[:320, :320]
    mask = np.zeros((320, 320), dtype=np.float32)
    mask[150:170, 80:240] = -2
    cell = np.zeros_like(mask)
    cell[154:166, 145:175] = 1.2
    surface = 0.2 + 0.001*x + 0.002*y + 0.00001*x*x - 0.000004*x*y
    for part, image in (("output_phase_raw", np.zeros_like(mask)), ("output_phase", mask)):
        folder = grid_node / part
        folder.mkdir(parents=True)
        tifffile.imwrite(folder / "img_000000000_ph_005_phase.tif", image)
    tl_dir = tl_root / "Pos1/z000/output_phase_raw"
    tl_dir.mkdir(parents=True)
    for frame in range(2):
        tifffile.imwrite(tl_dir / f"img_{frame:09d}_ph_000_phase.tif", (surface + cell).astype(np.float32))
    entries = [dict(frame_index=f, shift_x_avg=0, shift_y_avg=0, grid_xi=0, grid_yi=0,
                    residual_x_px=0, residual_y_px=0) for f in (1, 0)]
    replay.write_json(metadata / "pos_shifts_cal_online.json", dict(frame_results=entries,
                      grid_dir=str(grid_root), use_raw_phase=True, apply_inverse_shift=False,
                      apply_subpixel_correction=True, grid_z_index=5, tl_z_index=0))
    replay.write_json(metadata / "channel_rois.json", [dict(cx=160, cy=160, crop_w=40, crop_h=270)])
    replay.write_json(grid_root / "grid_calibration_Pos1.json",
                      {"positions": [dict(xi=0, yi=0, actual_dx_px=0, actual_dy_px=0)]})
    cfg = dict(paths=dict(raw_root=str(output), channel_rel="output_phase/channels/crop_sub_rawraw/z000"),
               background_replay=dict(source_crop_root=str(source), tl_root=str(tl_root),
                    grid_dir=str(grid_root), grid_z=5, frame_min=0, frame_max=1,
                    tilt_h=270, out_h=240, pos_split=52, workers=1))
    return cfg, cell


class ReplayTests(unittest.TestCase):
    def test_frames_use_recorded_ids_not_list_offsets(self):
        entries = [None, {"frame_index": 2}, {"frame_index": 0}, {"frame_index": 1}]
        self.assertEqual([e["frame_index"] for e in replay.frame_entries({"frame_results": entries}, 0, 2)], [0, 1, 2])

    def test_incomplete_or_duplicate_alignment_is_rejected(self):
        for frames, message in [([0, 2], "Missing"), ([0, 0, 1, 2], "Duplicate")]:
            with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, message):
                replay.frame_entries({"frame_results": [{"frame_index": f} for f in frames]}, 0, 2)

    def test_output_cannot_overlap_input(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "raw"
            for output in (source, source / "new", Path(temp)):
                with self.assertRaisesRegex(ValueError, "separate"):
                    replay.safe_output(output, [source])

    def test_replay_removes_quadratic_preserves_cells_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, cell = synthetic_run(Path(temp))
            self.assertTrue(replay.run_pos(cfg, 1)["complete"])
            base = Path(cfg["paths"]["raw_root"]) / "Pos1" / cfg["paths"]["channel_rel"]
            path = base / "ch00/img_000000000_ph_000.tif"
            expected = replay.gs.extract_rect_roi(cell, 160, 160, 40, 240)
            np.testing.assert_allclose(tifffile.imread(path), expected, atol=2e-6)
            stamp = path.stat().st_mtime_ns
            self.assertTrue(replay.run_pos(cfg, 1)["complete"])
            self.assertEqual(path.stat().st_mtime_ns, stamp)
            changed = copy.deepcopy(cfg)
            changed["background_replay"]["out_h"] = 230
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                replay.run_pos(changed, 1)

    def test_missing_raw_phase_stops_before_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, _ = synthetic_run(Path(temp))
            cfg["background_replay"]["tl_root"] += "_missing"
            with self.assertRaises(FileNotFoundError):
                replay.run_pos(cfg, 1)
            self.assertFalse(Path(cfg["paths"]["raw_root"]).exists())


if __name__ == "__main__":
    unittest.main()
