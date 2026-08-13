from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.sam2_schema import Sam2FrameObservation, Sam2InstanceObservation
from src.sam2_visualization import render_frame_overlay, render_visualization_set


def make_frame(root: Path, rgb_is_image: bool) -> Sam2FrameObservation:
    frame_id = "frame_001"
    rgb_path = root / f"{frame_id}.png"
    label_path = root / f"{frame_id}_labels.png"
    if rgb_is_image:
        Image.new("RGB", (20, 12), color=(40, 50, 60)).save(rgb_path)
    else:
        rgb_path.write_text("/home/robot/videos/frame_001.png", encoding="utf-8")
        Image.new("RGB", (20, 12), color=(70, 80, 90)).save(
            root / f"{frame_id}_overlay.png"
        )
    labels = Image.new("I;16", (20, 12), color=0)
    labels.putdata([1 if 4 <= index % 20 < 10 and 3 <= index // 20 < 9 else 0 for index in range(240)])
    labels.save(label_path)
    item = Sam2InstanceObservation(
        instance_id=1,
        track_id=1,
        label="switch",
        bbox_xywh=(4.0, 3.0, 6.0, 6.0),
        center_xy=(7.0, 6.0),
        center_normalized=(0.35, 0.5),
        area_pixels=36,
        area_ratio=0.15,
        present_in_label_image=True,
    )
    return Sam2FrameObservation(
        frame_id=frame_id,
        rgb_path=str(rgb_path),
        rgb_image_verified=rgb_is_image,
        rgb_source_reference=None if rgb_is_image else "/home/robot/videos/frame_001.png",
        metadata_path=str(root / f"{frame_id}.json"),
        label_image_path=str(label_path),
        image_width=20,
        image_height=12,
        total_records=1,
        zero_area_records=0,
        background_filtered_records=0,
        mask_instance_ids=[1],
        objects=[item],
    )


class Sam2VisualizationTests(unittest.TestCase):
    def test_renders_from_real_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frame = make_frame(root, rgb_is_image=True)
            output_path = root / "output.png"

            artifact = render_frame_overlay(frame, output_path)

            with Image.open(output_path) as image:
                size = image.size
        self.assertEqual(artifact.base_source, "rgb")
        self.assertEqual(artifact.rendered_instance_ids, [1])
        self.assertEqual(size, (20, 12))

    def test_falls_back_to_existing_sam2_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frame = make_frame(root, rgb_is_image=False)
            output_dir = root / "description"

            manifest = render_visualization_set([frame], output_dir)

            output_path = output_dir / "overlays" / "frame_001.png"
            manifest_path = output_dir / "visualization_manifest.json"
            output_exists = output_path.is_file()
            manifest_exists = manifest_path.is_file()
        self.assertTrue(output_exists)
        self.assertTrue(manifest_exists)
        self.assertEqual(manifest.base_source_counts, {"sam2_overlay": 1})
        self.assertEqual(manifest.object_observation_count, 1)


if __name__ == "__main__":
    unittest.main()
