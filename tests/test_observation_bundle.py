from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.observation_bundle import build_observation_bundle


class ObservationBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.scene_dir = self.root / "scene"
        self.sam2_dir = self.root / "sam2_vlm"
        self.objects_dir = self.sam2_dir / "objects"
        self.scene_dir.mkdir()
        self.objects_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_inputs(self, task_id: str = "Switch001") -> None:
        (self.scene_dir / "frame_001.json").write_text(
            json.dumps(
                {
                    "image_id": "frame_001",
                    "image_path": "rgb/frame_001.png",
                    "scene_brief": "A wall switch is visible.",
                    "overall_lighting": "bright",
                    "items": [
                        {
                            "item_id": "item1",
                            "possible_name": "wall switch",
                            "shape": "rectangle",
                            "location_in_image": "center",
                            "operable": True,
                            "confidence": 0.8,
                        }
                    ],
                    "uncertainty": [],
                }
            ),
            encoding="utf-8",
        )
        (self.sam2_dir / "task_summary.json").write_text(
            json.dumps({"task_id": task_id}), encoding="utf-8"
        )
        (self.objects_dir / "switch_track_0007.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "object_key": "switch_track_0007",
                    "sam2_label": "switch",
                    "sam2_track_id": 7,
                    "representative_frame_id": "frame_001",
                    "representative_instance_id": 1,
                    "observation_count": 1,
                    "observation_frame_ids": ["frame_001"],
                    "position_2d_zh": "中央",
                    "center_normalized": [0.5, 0.5],
                    "area_ratio": 0.1,
                    "pointcloud_path": "/tmp/switch.pcd",
                    "evidence": {
                        "rgb_path": "rgb/frame_001.png",
                        "sam2_metadata_path": "sam2/frame_001.json",
                        "sam2_label_image_path": "sam2/frame_001_labels.png",
                        "evidence_image_path": "evidence/switch.png",
                    },
                    "description": {
                        "object_name": "墙面开关",
                        "category": "wall_switch",
                        "visual_description": "白色矩形面板",
                        "visible_state": "无法判断",
                        "attributes": [],
                        "interaction_parts": ["按键"],
                        "occlusion": "无遮挡",
                        "confidence": 0.88,
                        "uncertainty": [],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def test_bundles_scene_and_track_results_without_identity_inference(self) -> None:
        self.write_inputs()
        output = self.root / "bundle.json"

        bundle = build_observation_bundle(
            "Switch001", self.scene_dir, self.sam2_dir, output
        )

        self.assertEqual(bundle["merge_policy"], "layered_no_identity_inference")
        self.assertEqual(bundle["scene_observation_count"], 1)
        self.assertEqual(bundle["object_observation_count"], 1)
        self.assertEqual(
            bundle["sam2_object_observations"][0]["observation"]["pointcloud_path"],
            "/tmp/switch.pcd",
        )
        self.assertTrue(output.is_file())

    def test_rejects_task_id_mismatch(self) -> None:
        self.write_inputs(task_id="another_task")

        with self.assertRaisesRegex(ValueError, "task_id mismatch"):
            build_observation_bundle(
                "Switch001", self.scene_dir, self.sam2_dir, self.root / "bundle.json"
            )


if __name__ == "__main__":
    unittest.main()
