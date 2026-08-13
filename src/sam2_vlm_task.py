from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .json_utils import extract_json_object, write_json
from .sam2_export import build_scene_summary
from .sam2_language import image_position_zh
from .sam2_observation import match_frame_inputs, parse_frame_observation
from .sam2_schema import Sam2FrameObservation, Sam2InstanceObservation, Sam2TrackSummary
from .sam2_vlm_schema import (
    Sam2VlmAcceptanceReport,
    Sam2VlmDescription,
    Sam2VlmEvidence,
    Sam2VlmFailure,
    Sam2VlmObjectResult,
    Sam2VlmTaskSummary,
)


SYSTEM_PROMPT = """You describe one SAM2-segmented object from a robot camera image.
The evidence image has three panels: the scene with the target highlighted, the isolated target mask, and a target crop.
Describe only the highlighted target. Do not describe surrounding objects as the target.
Do not infer map coordinates, metric distance, robot reachability, grasp success, or navigation feasibility.
Use concise Chinese text values and the exact English JSON keys requested by the user prompt.
Return one valid JSON object only, without Markdown or additional fields.
"""


class ImageAnalyzer(Protocol):
    last_call_metrics: dict[str, object]

    def analyze_image(self, image_path: Path, system_prompt: str, user_prompt: str) -> str:
        ...


def _object_key(label: str, track_id: int) -> str:
    safe_label = re.sub(r"[^a-z0-9_]+", "_", label.lower()).strip("_") or "object"
    return f"{safe_label}_track_{track_id:04d}"


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, (245, 247, 250))
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    panel.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return panel


def build_instance_evidence(
    frame: Sam2FrameObservation,
    item: Sam2InstanceObservation,
    output_path: Path,
) -> Path:
    if not frame.rgb_image_verified:
        raise ValueError(f"representative RGB is not a decodable image: {frame.rgb_path}")

    with Image.open(frame.rgb_path) as source:
        rgb = source.convert("RGB")
    with Image.open(frame.label_image_path) as source:
        labels = np.asarray(source)

    if labels.shape != (rgb.height, rgb.width):
        raise ValueError(
            f"RGB and label dimensions differ for {frame.frame_id}: "
            f"rgb={rgb.size}, labels={(labels.shape[1], labels.shape[0])}"
        )
    mask = labels == item.instance_id
    if not bool(mask.any()):
        raise ValueError(
            f"instance {item.instance_id} is absent from label image {frame.label_image_path}"
        )

    highlighted = rgb.copy()
    overlay = Image.new("RGB", rgb.size, (16, 185, 129))
    alpha = Image.fromarray(mask.astype(np.uint8) * 105)
    highlighted.paste(overlay, mask=alpha)
    draw = ImageDraw.Draw(highlighted)
    x, y, width, height = item.bbox_xywh
    draw.rectangle((x, y, x + width, y + height), outline=(239, 68, 68), width=4)

    isolated = Image.new("RGB", rgb.size, (232, 236, 241))
    isolated.paste(rgb, mask=Image.fromarray(mask.astype(np.uint8) * 255))

    pad_x = max(8, int(width * 0.15))
    pad_y = max(8, int(height * 0.15))
    left = max(0, int(x) - pad_x)
    top = max(0, int(y) - pad_y)
    right = min(rgb.width, int(x + width) + pad_x)
    bottom = min(rgb.height, int(y + height) + pad_y)
    crop = rgb.crop((left, top, right, bottom))

    panel_size = (480, 360)
    title_height = 36
    canvas = Image.new("RGB", (panel_size[0] * 3, panel_size[1] + title_height), "white")
    panels = (
        ("SCENE + TARGET", highlighted),
        ("MASKED TARGET", isolated),
        ("TARGET CROP", crop),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (title, panel_image) in enumerate(panels):
        left_offset = index * panel_size[0]
        draw.text((left_offset + 12, 12), title, fill=(20, 28, 42), font=font)
        canvas.paste(_fit_panel(panel_image, panel_size), (left_offset, title_height))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _build_user_prompt(track: Sam2TrackSummary, position_2d_zh: str) -> str:
    return f"""Describe the highlighted SAM2 target.

Known segmentation evidence:
- sam2_label: {track.label}
- sam2_track_id: {track.track_id}
- image_2d_position: {position_2d_zh}

The SAM2 label is identity evidence, not permission to invent invisible details. State uncertainty when needed.
Return exactly this JSON structure:
{{
  "object_name": "<concise Chinese object name>",
  "category": "<short machine-readable English category>",
  "visual_description": "<concise Chinese visible appearance>",
  "visible_state": "<concise Chinese visible state, or 无法判断>",
  "attributes": ["<visible Chinese attribute>"],
  "interaction_parts": ["<clearly visible Chinese interaction part>"],
  "occlusion": "<无遮挡, 部分遮挡, 严重遮挡, or 无法判断>",
  "confidence": 0.0,
  "uncertainty": ["<concise Chinese uncertainty>"]
}}

Rules:
- confidence must be a number from 0.0 to 1.0.
- Use empty arrays when no attribute, interaction part, or uncertainty is visible.
- Do not add fields.
"""


def _load_pointcloud_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    map_path = Path(path)
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
    ):
        raise ValueError("pointcloud map must be a JSON object of object keys to file paths")

    resolved: dict[str, str] = {}
    for key, raw_path in payload.items():
        pointcloud_path = Path(raw_path)
        if not pointcloud_path.is_absolute():
            pointcloud_path = map_path.parent / pointcloud_path
        pointcloud_path = pointcloud_path.resolve()
        if not pointcloud_path.is_file():
            raise FileNotFoundError(f"pointcloud file not found for {key}: {pointcloud_path}")
        resolved[key] = str(pointcloud_path)
    return resolved


def _representative_item(
    frame: Sam2FrameObservation,
    track: Sam2TrackSummary,
) -> Sam2InstanceObservation:
    candidates = [
        item
        for item in frame.objects
        if item.label == track.label and item.track_id == track.track_id
    ]
    if not candidates:
        raise ValueError(
            f"representative object {track.label}:{track.track_id} missing from {frame.frame_id}"
        )
    return max(candidates, key=lambda item: item.area_pixels)


def run_sam2_vlm_task(
    rgb_dir: Path,
    sam2_dir: Path,
    output_dir: Path,
    task_id: str,
    analyzer: ImageAnalyzer,
    pointcloud_map_path: Path | None = None,
    overwrite: bool = False,
) -> Sam2VlmTaskSummary:
    if not task_id.strip():
        raise ValueError("task_id must not be empty")

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = output_dir / "evidence"
    object_dir = output_dir / "objects"
    failure_dir = output_dir / "failures"
    evidence_dir.mkdir(exist_ok=True)
    object_dir.mkdir(exist_ok=True)

    frames = [parse_frame_observation(item) for item in match_frame_inputs(rgb_dir, sam2_dir)]
    scene_summary = build_scene_summary(frames)
    frame_by_id = {frame.frame_id: frame for frame in frames}
    pointcloud_map = _load_pointcloud_map(pointcloud_map_path)

    failures: list[Sam2VlmFailure] = []
    object_result_paths: list[str] = []
    for track in scene_summary.tracks:
        object_key = _object_key(track.label, track.track_id)
        frame = frame_by_id[track.representative_frame_id]
        evidence_path = evidence_dir / f"{object_key}.png"
        raw_response: str | None = None
        try:
            if not scene_summary.mask_metadata_alignment_ok:
                raise ValueError("SAM2 metadata and label images are not aligned")
            item = _representative_item(frame, track)
            build_instance_evidence(frame, item, evidence_path)
            position_2d_zh = image_position_zh(item.center_normalized)
            raw_response = analyzer.analyze_image(
                evidence_path,
                SYSTEM_PROMPT,
                _build_user_prompt(track, position_2d_zh),
            )
            description = Sam2VlmDescription.model_validate(extract_json_object(raw_response))
            result = Sam2VlmObjectResult(
                task_id=task_id,
                object_key=object_key,
                sam2_label=track.label,
                sam2_track_id=track.track_id,
                representative_frame_id=frame.frame_id,
                representative_instance_id=item.instance_id,
                observation_count=track.observation_count,
                observation_frame_ids=track.frame_ids,
                position_2d_zh=position_2d_zh,
                center_normalized=item.center_normalized,
                area_ratio=item.area_ratio,
                pointcloud_path=pointcloud_map.get(f"{track.label}:{track.track_id}"),
                evidence=Sam2VlmEvidence(
                    rgb_path=frame.rgb_path,
                    sam2_metadata_path=frame.metadata_path,
                    sam2_label_image_path=frame.label_image_path,
                    evidence_image_path=str(evidence_path),
                ),
                description=description,
                vlm_call_metrics=dict(getattr(analyzer, "last_call_metrics", {}) or {}),
            )
            result_path = object_dir / f"{object_key}.json"
            write_json(result_path, result.model_dump(mode="json"))
            object_result_paths.append(str(result_path))
        except Exception as error:
            failure = Sam2VlmFailure(
                object_key=object_key,
                sam2_label=track.label,
                sam2_track_id=track.track_id,
                representative_frame_id=frame.frame_id,
                evidence_image_path=str(evidence_path) if evidence_path.is_file() else None,
                raw_response=raw_response,
                error_type=type(error).__name__,
                error=str(error),
            )
            failures.append(failure)
            failure_dir.mkdir(exist_ok=True)
            write_json(
                failure_dir / f"{object_key}.json",
                failure.model_dump(mode="json"),
            )

    success = scene_summary.mask_metadata_alignment_ok and not failures
    summary = Sam2VlmTaskSummary(
        success=success,
        task_id=task_id,
        rgb_dir=str(Path(rgb_dir)),
        sam2_dir=str(Path(sam2_dir)),
        output_dir=str(output_dir),
        frame_count=scene_summary.frame_count,
        visible_observation_count=scene_summary.visible_observation_count,
        unique_track_count=scene_summary.unique_track_count,
        described_track_count=len(object_result_paths),
        failed_track_count=len(failures),
        mask_metadata_alignment_ok=scene_summary.mask_metadata_alignment_ok,
        object_result_paths=object_result_paths,
        failures=failures,
    )
    summary_path = output_dir / "task_summary.json"
    acceptance_path = output_dir / "acceptance_report.json"
    write_json(summary_path, summary.model_dump(mode="json"))
    acceptance = Sam2VlmAcceptanceReport(
        success=success,
        task_id=task_id,
        checks={
            "frame_inputs_matched": True,
            "mask_metadata_alignment_ok": scene_summary.mask_metadata_alignment_ok,
            "all_tracks_described": not failures,
            "one_result_per_track": len(object_result_paths) == scene_summary.unique_track_count,
        },
        counts={
            "frames": scene_summary.frame_count,
            "visible_observations": scene_summary.visible_observation_count,
            "unique_tracks": scene_summary.unique_track_count,
            "described_tracks": len(object_result_paths),
            "failed_tracks": len(failures),
        },
        artifacts={
            "task_summary": str(summary_path),
            "objects_dir": str(object_dir),
            "evidence_dir": str(evidence_dir),
        },
    )
    write_json(acceptance_path, acceptance.model_dump(mode="json"))
    return summary
