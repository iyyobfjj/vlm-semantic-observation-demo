from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.sam2_observation import match_frame_inputs, parse_frame_observation


class Sam2ObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.rgb_dir = self.root / "rgb"
        self.sam2_dir = self.root / "sam2"
        self.rgb_dir.mkdir()
        self.sam2_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_frame(self, frame_id: str, records: list[dict], labels: list[int]) -> None:
        Image.new("RGB", (4, 3), color=(20, 30, 40)).save(self.rgb_dir / f"{frame_id}.png")
        label_image = Image.new("I;16", (4, 3), color=0)
        label_image.putdata(labels)
        label_image.save(self.sam2_dir / f"{frame_id}_labels.png")
        (self.sam2_dir / f"{frame_id}.json").write_text(
            json.dumps(records),
            encoding="utf-8",
        )
        (self.sam2_dir / f"{frame_id}_timing.json").write_text("{}", encoding="utf-8")

    @staticmethod
    def record(instance_id: int, label: str, area: int, bbox: list[float]) -> dict:
        return {
            "id": instance_id,
            "area": area,
            "bbox_xywh": bbox,
            "predicted_iou": 0.75,
            "source": {
                "track_id": instance_id,
                "label": label,
                "prompted_on_this_frame": True,
            },
        }

    def test_matches_rgb_metadata_and_label_images(self) -> None:
        self.write_frame("frame_001", [], [0] * 12)
        self.write_frame("frame_002", [], [0] * 12)

        inputs = match_frame_inputs(self.rgb_dir, self.sam2_dir)

        self.assertEqual([item.frame_id for item in inputs], ["frame_001", "frame_002"])

    def test_rejects_missing_label_image(self) -> None:
        Image.new("RGB", (4, 3)).save(self.rgb_dir / "frame_001.png")
        (self.sam2_dir / "frame_001.json").write_text("[]", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "missing_labels"):
            match_frame_inputs(self.rgb_dir, self.sam2_dir)

    def test_parses_visible_objects_and_two_dimensional_position(self) -> None:
        records = [self.record(1, "Storage Cabinet", 4, [0, 0, 2, 2])]
        self.write_frame("frame_001", records, [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
        frame_input = match_frame_inputs(self.rgb_dir, self.sam2_dir)[0]

        result = parse_frame_observation(frame_input)

        self.assertEqual(result.mask_instance_ids, [1])
        self.assertTrue(result.rgb_image_verified)
        self.assertIsNone(result.rgb_source_reference)
        self.assertEqual(len(result.objects), 1)
        self.assertEqual(result.objects[0].label, "storage_cabinet")
        self.assertEqual(result.objects[0].center_xy, (1.0, 1.0))
        self.assertEqual(result.objects[0].center_normalized, (0.25, 1.0 / 3.0))
        self.assertAlmostEqual(result.objects[0].area_ratio, 1.0 / 3.0)
        self.assertTrue(result.objects[0].present_in_label_image)

    def test_filters_zero_area_and_background_records(self) -> None:
        records = [
            self.record(1, "switch", 0, [0, 0, 0, 0]),
            self.record(2, "carpet", 4, [0, 0, 2, 2]),
            self.record(3, "rubbish-bin", 2, [2, 1, 2, 1]),
        ]
        self.write_frame("frame_001", records, [2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 3, 3])
        frame_input = match_frame_inputs(self.rgb_dir, self.sam2_dir)[0]

        result = parse_frame_observation(frame_input)

        self.assertEqual(result.zero_area_records, 1)
        self.assertEqual(result.background_filtered_records, 1)
        self.assertEqual([item.label for item in result.objects], ["rubbish_bin"])

    def test_reports_mask_and_metadata_id_mismatches(self) -> None:
        records = [self.record(1, "switch", 2, [0, 0, 2, 1])]
        self.write_frame("frame_001", records, [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        frame_input = match_frame_inputs(self.rgb_dir, self.sam2_dir)[0]

        result = parse_frame_observation(frame_input)

        self.assertEqual(result.unmatched_metadata_ids, [1])
        self.assertEqual(result.unmatched_mask_ids, [2])
        self.assertFalse(result.objects[0].present_in_label_image)

    def test_preserves_linux_rgb_symlink_reference(self) -> None:
        frame_id = "frame_001"
        reference = "/home/robot/videos/task/rgb/frame_001.png"
        (self.rgb_dir / f"{frame_id}.png").write_text(reference, encoding="utf-8")
        Image.new("I;16", (4, 3), color=0).save(self.sam2_dir / f"{frame_id}_labels.png")
        (self.sam2_dir / f"{frame_id}.json").write_text("[]", encoding="utf-8")
        frame_input = match_frame_inputs(self.rgb_dir, self.sam2_dir)[0]

        result = parse_frame_observation(frame_input)

        self.assertFalse(result.rgb_image_verified)
        self.assertEqual(result.rgb_source_reference, reference)


if __name__ == "__main__":
    unittest.main()
