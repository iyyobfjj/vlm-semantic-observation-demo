from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.sam2_vlm_task import run_sam2_vlm_task


class FakeAnalyzer:
    def __init__(self, response: str | None = None) -> None:
        self.response = response or json.dumps(
            {
                "object_name": "墙面开关",
                "category": "wall_switch",
                "visual_description": "白色矩形面板",
                "visible_state": "无法判断",
                "attributes": ["固定"],
                "interaction_parts": ["按键"],
                "occlusion": "无遮挡",
                "confidence": 0.88,
                "uncertainty": ["开关状态不可见"],
            },
            ensure_ascii=False,
        )
        self.calls: list[tuple[Path, str, str]] = []
        self.last_call_metrics = {"attempt": 1, "elapsed_seconds": 0.01}

    def analyze_image(self, image_path: Path, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((image_path, system_prompt, user_prompt))
        return self.response


class Sam2VlmTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.rgb_dir = self.root / "rgb"
        self.sam2_dir = self.root / "sam2"
        self.rgb_dir.mkdir()
        self.sam2_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_frame(
        self,
        frame_id: str,
        area: int,
        bbox: list[float],
        label: str = "switch",
        track_id: int = 7,
        instance_id: int = 1,
    ) -> None:
        Image.new("RGB", (12, 8), color=(60, 80, 100)).save(
            self.rgb_dir / f"{frame_id}.png"
        )
        labels = [0] * 96
        x, y, width, height = [int(value) for value in bbox]
        for row in range(y, y + height):
            for column in range(x, x + width):
                labels[row * 12 + column] = instance_id
        label_image = Image.new("I;16", (12, 8), color=0)
        label_image.putdata(labels)
        label_image.save(self.sam2_dir / f"{frame_id}_labels.png")
        record = {
            "id": instance_id,
            "area": area,
            "bbox_xywh": bbox,
            "source": {"track_id": track_id, "label": label},
        }
        (self.sam2_dir / f"{frame_id}.json").write_text(
            json.dumps([record]), encoding="utf-8"
        )

    def test_describes_track_once_and_writes_traceable_artifacts(self) -> None:
        self.write_frame("frame_001", 6, [1, 1, 3, 2])
        self.write_frame("frame_002", 20, [6, 3, 5, 4])
        output_dir = self.root / "output"
        pointcloud = self.root / "switch.pcd"
        pointcloud.write_text("VERSION .7\n", encoding="ascii")
        pointcloud_map = self.root / "pointcloud_map.json"
        pointcloud_map.write_text(
            json.dumps({"switch:7": pointcloud.name}), encoding="utf-8"
        )
        analyzer = FakeAnalyzer()

        summary = run_sam2_vlm_task(
            self.rgb_dir,
            self.sam2_dir,
            output_dir,
            "task_001",
            analyzer,
            pointcloud_map_path=pointcloud_map,
        )

        self.assertTrue(summary.success)
        self.assertEqual(summary.unique_track_count, 1)
        self.assertEqual(summary.described_track_count, 1)
        self.assertEqual(len(analyzer.calls), 1)
        self.assertTrue(analyzer.calls[0][0].is_file())
        result = json.loads((output_dir / "objects" / "switch_track_0007.json").read_text(encoding="utf-8"))
        self.assertEqual(result["representative_frame_id"], "frame_002")
        self.assertEqual(result["observation_frame_ids"], ["frame_001", "frame_002"])
        self.assertEqual(result["description"]["category"], "wall_switch")
        self.assertEqual(result["pointcloud_path"], str(pointcloud.resolve()))
        self.assertEqual(result["position_scope"], "image_2d_only")
        self.assertIn("sam2_track_id: 7", analyzer.calls[0][2])
        self.assertTrue((output_dir / "acceptance_report.json").is_file())

    def test_invalid_vlm_response_creates_failure_artifact(self) -> None:
        self.write_frame("frame_001", 6, [1, 1, 3, 2])
        output_dir = self.root / "output"

        summary = run_sam2_vlm_task(
            self.rgb_dir,
            self.sam2_dir,
            output_dir,
            "task_002",
            FakeAnalyzer("not json"),
        )

        self.assertFalse(summary.success)
        self.assertEqual(summary.failed_track_count, 1)
        failure = json.loads(
            (output_dir / "failures" / "switch_track_0007.json").read_text(encoding="utf-8")
        )
        self.assertEqual(failure["raw_response"], "not json")
        self.assertTrue((output_dir / "evidence" / "switch_track_0007.png").is_file())

    def test_background_only_task_does_not_call_vlm(self) -> None:
        self.write_frame("frame_001", 6, [1, 1, 3, 2], label="ceiling")
        analyzer = FakeAnalyzer()

        summary = run_sam2_vlm_task(
            self.rgb_dir,
            self.sam2_dir,
            self.root / "output",
            "task_003",
            analyzer,
        )

        self.assertTrue(summary.success)
        self.assertEqual(summary.unique_track_count, 0)
        self.assertEqual(analyzer.calls, [])

    def test_rejects_missing_explicit_pointcloud_file(self) -> None:
        self.write_frame("frame_001", 6, [1, 1, 3, 2])
        pointcloud_map = self.root / "pointcloud_map.json"
        pointcloud_map.write_text(
            json.dumps({"switch:7": "missing.pcd"}), encoding="utf-8"
        )

        with self.assertRaises(FileNotFoundError):
            run_sam2_vlm_task(
                self.rgb_dir,
                self.sam2_dir,
                self.root / "output",
                "task_004",
                FakeAnalyzer(),
                pointcloud_map_path=pointcloud_map,
            )


if __name__ == "__main__":
    unittest.main()
