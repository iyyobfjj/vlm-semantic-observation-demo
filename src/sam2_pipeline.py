from __future__ import annotations

import json
from pathlib import Path

from .sam2_description import export_chinese_descriptions
from .sam2_export import export_sam2_observations
from .sam2_observation import match_frame_inputs, parse_frame_observation
from .sam2_schema import Sam2PipelineReport
from .sam2_visualization import render_visualization_set


def run_sam2_pipeline(
    rgb_dir: Path,
    sam2_dir: Path,
    output_dir: Path,
) -> Sam2PipelineReport:
    rgb_dir = Path(rgb_dir)
    sam2_dir = Path(sam2_dir)
    output_dir = Path(output_dir)

    frame_inputs = match_frame_inputs(rgb_dir, sam2_dir)
    frames = [parse_frame_observation(frame_input) for frame_input in frame_inputs]
    scene_summary = export_sam2_observations(frames, output_dir)
    if not scene_summary.mask_metadata_alignment_ok:
        raise ValueError(
            "SAM2 mask and metadata IDs do not align; inspect scene_summary.json before rendering"
        )

    visualization_manifest = render_visualization_set(frames, output_dir)
    speech_summary = export_chinese_descriptions(frames, output_dir)
    description_dir = output_dir / "descriptions" / "frames"
    description_frame_count = len(list(description_dir.glob("*.json")))
    success = (
        visualization_manifest.rendered_frame_count == len(frames)
        and description_frame_count == len(frames)
        and speech_summary.frame_count == len(frames)
    )
    report = Sam2PipelineReport(
        success=success,
        rgb_dir=str(rgb_dir),
        sam2_dir=str(sam2_dir),
        output_dir=str(output_dir),
        frame_count=len(frames),
        rgb_verified_frame_count=sum(frame.rgb_image_verified for frame in frames),
        visible_observation_count=scene_summary.visible_observation_count,
        unique_track_count=scene_summary.unique_track_count,
        mask_metadata_alignment_ok=scene_summary.mask_metadata_alignment_ok,
        rendered_frame_count=visualization_manifest.rendered_frame_count,
        description_frame_count=description_frame_count,
        base_source_counts=visualization_manifest.base_source_counts,
        speech_text=speech_summary.speech_text,
        artifacts={
            "scene_summary": str(output_dir / "scene_summary.json"),
            "visualization_manifest": str(output_dir / "visualization_manifest.json"),
            "speech_summary": str(output_dir / "speech_summary.json"),
            "text_summary": str(output_dir / "summary.txt"),
            "frame_observations": str(output_dir / "frames"),
            "overlays": str(output_dir / "overlays"),
            "frame_descriptions": str(description_dir),
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acceptance_report.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
