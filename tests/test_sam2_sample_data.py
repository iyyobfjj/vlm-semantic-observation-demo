from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.check_sam2_inputs import main as check_sam2_inputs_main
from src.sam2_description import export_chinese_descriptions
from src.sam2_export import export_sam2_observations
from src.sam2_observation import match_frame_inputs, parse_frame_observation
from src.sam2_visualization import render_visualization_set


ROOT = Path(__file__).resolve().parents[2]
RGB_DIR = ROOT / "rgb" / "rgb"
SAM2_DIR = ROOT / "sam2_video" / "sam2_video"


@unittest.skipUnless(RGB_DIR.is_dir() and SAM2_DIR.is_dir(), "local SAM2 sample data is unavailable")
class Sam2SampleDataTests(unittest.TestCase):
    def test_parses_complete_thirty_frame_sample(self) -> None:
        frame_inputs = match_frame_inputs(RGB_DIR, SAM2_DIR)
        frames = [parse_frame_observation(frame_input) for frame_input in frame_inputs]

        self.assertEqual(len(frames), 30)
        self.assertEqual(sum(frame.total_records for frame in frames), 240)
        self.assertEqual(sum(len(frame.objects) for frame in frames), 28)
        self.assertEqual(
            {item.label for frame in frames for item in frame.objects},
            {"microwave", "refrigerator", "rubbish_bin", "storage_cabinet", "switch"},
        )
        self.assertFalse(
            [
                frame.frame_id
                for frame in frames
                if frame.unmatched_metadata_ids or frame.unmatched_mask_ids
            ]
        )

    def test_input_check_command_reports_sample_baseline(self) -> None:
        output = io.StringIO()
        arguments = [
            "check_sam2_inputs.py",
            "--rgb-dir",
            str(RGB_DIR),
            "--sam2-dir",
            str(SAM2_DIR),
        ]

        with patch.object(sys, "argv", arguments), redirect_stdout(output):
            with self.assertRaises(SystemExit) as exit_result:
                check_sam2_inputs_main()

        self.assertEqual(exit_result.exception.code, 0)
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["matched_frames"], 30)
        self.assertEqual(summary["visible_records"], 28)

    def test_exports_complete_sample_summary(self) -> None:
        frame_inputs = match_frame_inputs(RGB_DIR, SAM2_DIR)
        frames = [parse_frame_observation(frame_input) for frame_input in frame_inputs]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "sam2_description"

            summary = export_sam2_observations(frames, output_dir)

            frame_files = sorted((output_dir / "frames").glob("*.json"))
            summary_payload = json.loads(
                (output_dir / "scene_summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(len(frame_files), 30)
        self.assertEqual(summary.visible_observation_count, 28)
        self.assertEqual(summary.unique_track_count, 8)
        self.assertEqual(summary_payload["label_track_counts"]["storage_cabinet"], 3)
        self.assertEqual(len(summary.rgb_unverified_frame_ids), 30)
        self.assertTrue(summary.mask_metadata_alignment_ok)

    def test_renders_complete_sample_visualizations(self) -> None:
        frame_inputs = match_frame_inputs(RGB_DIR, SAM2_DIR)
        frames = [parse_frame_observation(frame_input) for frame_input in frame_inputs]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "sam2_description"

            manifest = render_visualization_set(frames, output_dir)

            overlay_files = sorted((output_dir / "overlays").glob("*.png"))
        self.assertEqual(len(overlay_files), 30)
        self.assertEqual(manifest.rendered_frame_count, 30)
        self.assertEqual(manifest.object_observation_count, 28)
        self.assertEqual(manifest.base_source_counts, {"sam2_overlay": 30})

    def test_exports_complete_sample_chinese_descriptions(self) -> None:
        frame_inputs = match_frame_inputs(RGB_DIR, SAM2_DIR)
        frames = [parse_frame_observation(frame_input) for frame_input in frame_inputs]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "sam2_description"

            summary = export_chinese_descriptions(frames, output_dir)

            frame_files = sorted((output_dir / "descriptions" / "frames").glob("*.json"))
        self.assertEqual(len(frame_files), 30)
        self.assertEqual(summary.unique_track_count, 8)
        self.assertEqual(
            summary.speech_text,
            "视觉巡检完成，识别到3个储物柜、2个开关、1个冰箱、1个微波炉和1个垃圾桶。",
        )
