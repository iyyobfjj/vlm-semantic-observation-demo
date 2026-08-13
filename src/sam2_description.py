from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .sam2_export import build_scene_summary
from .sam2_language import (
    LABEL_NAMES_ZH,
    POSITION_NOTE_ZH,
    POSITION_ORDER,
    image_position_zh,
    join_chinese_phrases,
    label_name_zh,
)
from .sam2_schema import (
    Sam2FrameDescription,
    Sam2FrameObservation,
    Sam2InventoryItem,
    Sam2LocalizedObjectDescription,
    Sam2SpeechSummary,
)


def build_frame_description(
    frame: Sam2FrameObservation,
    visualization_path: Path,
) -> Sam2FrameDescription:
    localized_objects = [
        Sam2LocalizedObjectDescription(
            instance_id=item.instance_id,
            track_id=item.track_id,
            label=item.label,
            name_zh=label_name_zh(item.label),
            position_zh=image_position_zh(item.center_normalized),
            center_normalized=item.center_normalized,
            area_ratio=item.area_ratio,
        )
        for item in frame.objects
    ]
    grouped = Counter((item.position_zh, item.name_zh) for item in localized_objects)
    phrases = [
        f"{position}有{count}个{name}"
        for (position, name), count in sorted(
            grouped.items(), key=lambda item: (POSITION_ORDER[item[0][0]], item[0][1])
        )
    ]
    if phrases:
        details = "，".join(phrases)
        summary = f"当前画面识别到{len(localized_objects)}个目标物品：{details}。"
    else:
        summary = "当前画面未识别到目标物品。"
    return Sam2FrameDescription(
        frame_id=frame.frame_id,
        object_count=len(localized_objects),
        summary_zh=summary,
        speech_text=summary,
        position_note_zh=POSITION_NOTE_ZH,
        visualization_path=str(visualization_path),
        visualization_available=visualization_path.is_file(),
        objects=localized_objects,
    )


def build_speech_summary(frames: Iterable[Sam2FrameObservation]) -> Sam2SpeechSummary:
    ordered_frames = sorted(frames, key=lambda frame: frame.frame_id)
    scene_summary = build_scene_summary(ordered_frames)
    tracks_by_label: dict[str, list] = {}
    for track in scene_summary.tracks:
        tracks_by_label.setdefault(track.label, []).append(track)

    labels_in_order = [label for label in LABEL_NAMES_ZH if label in tracks_by_label]
    labels_in_order.extend(sorted(set(tracks_by_label) - set(labels_in_order)))
    inventory: list[Sam2InventoryItem] = []
    count_phrases: list[str] = []
    for label in labels_in_order:
        tracks = tracks_by_label[label]
        name_zh = label_name_zh(label)
        track_count = len(tracks)
        inventory.append(
            Sam2InventoryItem(
                label=label,
                name_zh=name_zh,
                track_count=track_count,
                observation_count=scene_summary.label_observation_counts[label],
                representative_positions_zh=[
                    image_position_zh(track.representative_center_normalized)
                    for track in tracks
                ],
            )
        )
        count_phrases.append(f"{track_count}个{name_zh}")

    inventory_text = join_chinese_phrases(count_phrases)
    if inventory_text:
        summary_zh = (
            f"本批次共分析{scene_summary.frame_count}帧，其中"
            f"{scene_summary.frames_with_objects}帧识别到目标物品；"
            f"共识别{scene_summary.unique_track_count}个稳定物品实例，包括{inventory_text}。"
        )
        speech_text = f"视觉巡检完成，识别到{inventory_text}。"
    else:
        summary_zh = f"本批次共分析{scene_summary.frame_count}帧，未识别到目标物品。"
        speech_text = "视觉巡检完成，未识别到目标物品。"
    return Sam2SpeechSummary(
        frame_count=scene_summary.frame_count,
        frames_with_objects=scene_summary.frames_with_objects,
        visible_observation_count=scene_summary.visible_observation_count,
        unique_track_count=scene_summary.unique_track_count,
        summary_zh=summary_zh,
        speech_text=speech_text,
        position_note_zh=POSITION_NOTE_ZH,
        inventory=inventory,
    )


def _write_json(path: Path, model: object) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def export_chinese_descriptions(
    frames: Iterable[Sam2FrameObservation],
    output_dir: Path,
) -> Sam2SpeechSummary:
    ordered_frames = sorted(frames, key=lambda frame: frame.frame_id)
    output_dir = Path(output_dir)
    frame_description_dir = output_dir / "descriptions" / "frames"
    frame_description_dir.mkdir(parents=True, exist_ok=True)

    for frame in ordered_frames:
        visualization_path = output_dir / "overlays" / f"{frame.frame_id}.png"
        description = build_frame_description(frame, visualization_path)
        _write_json(frame_description_dir / f"{frame.frame_id}.json", description)

    summary = build_speech_summary(ordered_frames)
    _write_json(output_dir / "speech_summary.json", summary)
    (output_dir / "summary.txt").write_text(
        summary.summary_zh + "\n" + summary.position_note_zh + "\n",
        encoding="utf-8",
    )
    return summary
