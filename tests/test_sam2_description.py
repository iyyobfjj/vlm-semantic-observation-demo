from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.sam2_description import build_frame_description, build_speech_summary, export_chinese_descriptions
from src.sam2_schema import Sam2FrameObservation, Sam2InstanceObservation


def make_object(track_id: int, label: str, center: tuple[float, float]) -> Sam2InstanceObservation:
    return Sam2InstanceObservation(instance_id=track_id, track_id=track_id, label=label, bbox_xywh=(0.0, 0.0, 2.0, 2.0), center_xy=(center[0] * 100, center[1] * 100), center_normalized=center, area_pixels=100, area_ratio=0.01, present_in_label_image=True)


def make_frame(frame_id: str, objects: list[Sam2InstanceObservation]) -> Sam2FrameObservation:
    return Sam2FrameObservation(frame_id=frame_id, rgb_path="rgb.png", rgb_image_verified=True, metadata_path="sam2.json", label_image_path="labels.png", image_width=100, image_height=100, total_records=len(objects), zero_area_records=0, background_filtered_records=0, mask_instance_ids=[item.instance_id for item in objects], objects=objects)


class Sam2DescriptionTests(unittest.TestCase):
    def test_generates_localized_frame_and_unique_track_speech_summaries(self) -> None:
        frames = [
            make_frame("frame_001", [make_object(1, "switch", (0.1, 0.5))]),
            make_frame("frame_002", [make_object(1, "switch", (0.2, 0.5)), make_object(2, "refrigerator", (0.8, 0.5))]),
        ]
        description = build_frame_description(frames[1], Path("missing_overlay.png"))
        summary = build_speech_summary(frames)
        self.assertIn("左侧有1个开关", description.summary_zh)
        self.assertIn("右侧有1个冰箱", description.summary_zh)
        self.assertFalse(description.visualization_available)
        self.assertEqual(summary.speech_text, "视觉巡检完成，识别到1个开关和1个冰箱。")

    def test_exports_json_and_text_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            export_chinese_descriptions([make_frame("frame_001", [make_object(1, "switch", (0.1, 0.5))])], output_dir)
            frame_payload = json.loads((output_dir / "descriptions" / "frames" / "frame_001.json").read_text(encoding="utf-8"))
            speech_payload = json.loads((output_dir / "speech_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(frame_payload["objects"][0]["name_zh"], "开关")
        self.assertEqual(speech_payload["schema_version"], "sam2_speech_summary.v1")


if __name__ == "__main__":
    unittest.main()
