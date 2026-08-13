from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.sam2_pipeline import run_sam2_pipeline


class Sam2PipelineTests(unittest.TestCase):
    def test_runs_complete_offline_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rgb_dir = root / "rgb"
            sam2_dir = root / "sam2"
            output_dir = root / "output"
            rgb_dir.mkdir()
            sam2_dir.mkdir()
            frame_id = "frame_001"
            Image.new("RGB", (20, 12), color=(40, 50, 60)).save(
                rgb_dir / f"{frame_id}.png"
            )
            labels = Image.new("I;16", (20, 12), color=0)
            labels.putdata(
                [
                    1 if 4 <= index % 20 < 10 and 3 <= index // 20 < 9 else 0
                    for index in range(240)
                ]
            )
            labels.save(sam2_dir / f"{frame_id}_labels.png")
            metadata = [
                {
                    "id": 1,
                    "area": 36,
                    "bbox_xywh": [4.0, 3.0, 6.0, 6.0],
                    "predicted_iou": 0.8,
                    "source": {
                        "track_id": 1,
                        "label": "switch",
                        "prompted_on_this_frame": True,
                    },
                }
            ]
            (sam2_dir / f"{frame_id}.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            report = run_sam2_pipeline(rgb_dir, sam2_dir, output_dir)

            acceptance_payload = json.loads(
                (output_dir / "acceptance_report.json").read_text(encoding="utf-8")
            )
        self.assertTrue(report.success)
        self.assertEqual(report.frame_count, 1)
        self.assertEqual(report.rgb_verified_frame_count, 1)
        self.assertEqual(report.rendered_frame_count, 1)
        self.assertEqual(report.description_frame_count, 1)
        self.assertEqual(report.speech_text, "视觉巡检完成，识别到1个开关。")
        self.assertEqual(acceptance_payload["schema_version"], "sam2_pipeline_report.v1")


if __name__ == "__main__":
    unittest.main()
