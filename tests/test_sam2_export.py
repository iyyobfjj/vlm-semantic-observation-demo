from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.sam2_export import build_scene_summary, export_sam2_observations
from src.sam2_schema import Sam2FrameObservation, Sam2InstanceObservation


def make_object(
    track_id: int,
    label: str,
    area_pixels: int,
    center: tuple[float, float],
) -> Sam2InstanceObservation:
    return Sam2InstanceObservation(
        instance_id=track_id,
        track_id=track_id,
        label=label,
        bbox_xywh=(0.0, 0.0, 2.0, 2.0),
        center_xy=center,
        center_normalized=(center[0] / 10.0, center[1] / 10.0),
        area_pixels=area_pixels,
        area_ratio=area_pixels / 100.0,
        present_in_label_image=True,
    )


def make_frame(frame_id: str, objects: list[Sam2InstanceObservation]) -> Sam2FrameObservation:
    return Sam2FrameObservation(
        frame_id=frame_id,
        rgb_path=f"rgb/{frame_id}.png",
        rgb_image_verified=True,
        metadata_path=f"sam2/{frame_id}.json",
        label_image_path=f"sam2/{frame_id}_labels.png",
        image_width=10,
        image_height=10,
        total_records=len(objects),
        zero_area_records=0,
        background_filtered_records=0,
        mask_instance_ids=[item.instance_id for item in objects],
        objects=objects,
    )


class Sam2ExportTests(unittest.TestCase):
    def test_summarizes_tracks_without_counting_repeated_observations_as_objects(self) -> None:
        frames = [
            make_frame("frame_001", [make_object(1, "switch", 10, (2.0, 3.0))]),
            make_frame(
                "frame_002",
                [
                    make_object(1, "switch", 20, (4.0, 5.0)),
                    make_object(2, "refrigerator", 15, (6.0, 7.0)),
                ],
            ),
        ]

        summary = build_scene_summary(frames)

        self.assertEqual(summary.visible_observation_count, 3)
        self.assertEqual(summary.unique_track_count, 2)
        self.assertEqual(summary.label_observation_counts, {"refrigerator": 1, "switch": 2})
        self.assertEqual(summary.label_track_counts, {"refrigerator": 1, "switch": 1})
        switch = next(item for item in summary.tracks if item.label == "switch")
        self.assertEqual(switch.representative_frame_id, "frame_002")
        self.assertEqual(switch.representative_center_xy, (4.0, 5.0))

    def test_exports_frame_files_and_scene_summary(self) -> None:
        frames = [make_frame("frame_001", [make_object(1, "switch", 10, (2.0, 3.0))])]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "description"

            summary = export_sam2_observations(frames, output_dir)

            frame_payload = json.loads(
                (output_dir / "frames" / "frame_001.json").read_text(encoding="utf-8")
            )
            summary_payload = json.loads(
                (output_dir / "scene_summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(frame_payload["schema_version"], "sam2_frame_observation.v1")
        self.assertEqual(frame_payload["objects"][0]["center_normalized"], [0.2, 0.3])
        self.assertEqual(summary_payload["schema_version"], "sam2_scene_summary.v1")
        self.assertEqual(summary_payload["unique_track_count"], 1)
        self.assertTrue(summary.mask_metadata_alignment_ok)


if __name__ == "__main__":
    unittest.main()
