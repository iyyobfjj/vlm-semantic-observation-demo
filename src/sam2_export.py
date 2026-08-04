from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .sam2_schema import Sam2FrameObservation, Sam2InstanceObservation, Sam2SceneSummary, Sam2TrackSummary


def build_scene_summary(frames: Iterable[Sam2FrameObservation]) -> Sam2SceneSummary:
    ordered_frames = sorted(frames, key=lambda frame: frame.frame_id)
    frame_ids = [frame.frame_id for frame in ordered_frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("frame observations contain duplicate frame_id values")
    grouped_objects: dict[tuple[str, int], list[tuple[str, Sam2InstanceObservation]]] = defaultdict(list)
    label_observation_counts: Counter[str] = Counter()
    for frame in ordered_frames:
        for item in frame.objects:
            grouped_objects[(item.label, item.track_id)].append((frame.frame_id, item))
            label_observation_counts[item.label] += 1
    tracks: list[Sam2TrackSummary] = []
    label_track_counts: Counter[str] = Counter()
    for (label, track_id), observations in sorted(grouped_objects.items()):
        representative_frame_id, representative = max(observations, key=lambda observation: observation[1].area_pixels)
        observation_frame_ids = [frame_id for frame_id, _ in observations]
        tracks.append(Sam2TrackSummary(
            track_id=track_id, label=label, observation_count=len(observations), frame_ids=observation_frame_ids,
            first_frame_id=observation_frame_ids[0], last_frame_id=observation_frame_ids[-1],
            representative_frame_id=representative_frame_id, representative_center_xy=representative.center_xy,
            representative_center_normalized=representative.center_normalized, max_area_pixels=representative.area_pixels,
            max_area_ratio=representative.area_ratio,
        ))
        label_track_counts[label] += 1
    mismatch_frame_ids = [frame.frame_id for frame in ordered_frames if frame.unmatched_metadata_ids or frame.unmatched_mask_ids]
    return Sam2SceneSummary(
        frame_count=len(ordered_frames), frame_ids=frame_ids, frames_with_objects=sum(bool(frame.objects) for frame in ordered_frames),
        visible_observation_count=sum(len(frame.objects) for frame in ordered_frames), unique_track_count=len(tracks),
        label_observation_counts=dict(sorted(label_observation_counts.items())), label_track_counts=dict(sorted(label_track_counts.items())),
        total_records=sum(frame.total_records for frame in ordered_frames), zero_area_records=sum(frame.zero_area_records for frame in ordered_frames),
        background_filtered_records=sum(frame.background_filtered_records for frame in ordered_frames),
        rgb_unverified_frame_ids=[frame.frame_id for frame in ordered_frames if not frame.rgb_image_verified],
        mask_metadata_mismatch_frame_ids=mismatch_frame_ids, mask_metadata_alignment_ok=not mismatch_frame_ids, tracks=tracks,
    )


def export_sam2_observations(frames: Iterable[Sam2FrameObservation], output_dir: Path) -> Sam2SceneSummary:
    ordered_frames = sorted(frames, key=lambda frame: frame.frame_id)
    summary = build_scene_summary(ordered_frames)
    output_dir = Path(output_dir)
    frame_output_dir = output_dir / "frames"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    for frame in ordered_frames:
        (frame_output_dir / f"{frame.frame_id}.json").write_text(
            json.dumps(frame.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (output_dir / "scene_summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
