from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from .sam2_schema import Sam2FrameInput, Sam2FrameObservation, Sam2InstanceObservation


DEFAULT_BACKGROUND_LABELS = frozenset({
    "air_vent", "carpet", "ceiling", "ceiling_light", "ceiling_pipe", "floor",
    "floor_marking", "wall",
})


def normalize_label(label: str) -> str:
    return "_".join(label.strip().lower().replace("-", " ").split())


def _files_by_frame(directory: Path, suffix: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.glob("*" + suffix)):
        frame_id = path.name[: -len(suffix)]
        if frame_id in result:
            raise ValueError(f"duplicate frame {frame_id!r} for suffix {suffix!r}")
        result[frame_id] = path
    return result


def match_frame_inputs(rgb_dir: Path, sam2_dir: Path) -> list[Sam2FrameInput]:
    rgb_dir, sam2_dir = Path(rgb_dir), Path(sam2_dir)
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not sam2_dir.is_dir():
        raise FileNotFoundError(f"SAM2 directory not found: {sam2_dir}")
    rgb_files = _files_by_frame(rgb_dir, ".png")
    metadata_files = {
        frame_id: path for frame_id, path in _files_by_frame(sam2_dir, ".json").items()
        if not frame_id.endswith("_timing")
    }
    label_files = _files_by_frame(sam2_dir, "_labels.png")
    frame_ids = set(rgb_files)
    missing_metadata = sorted(frame_ids - set(metadata_files))
    missing_labels = sorted(frame_ids - set(label_files))
    extra_metadata = sorted(set(metadata_files) - frame_ids)
    extra_labels = sorted(set(label_files) - frame_ids)
    if missing_metadata or missing_labels or extra_metadata or extra_labels:
        raise ValueError(
            "frame inputs do not match: "
            f"missing_metadata={missing_metadata}, missing_labels={missing_labels}, "
            f"extra_metadata={extra_metadata}, extra_labels={extra_labels}"
        )
    return [Sam2FrameInput(
        frame_id=frame_id, rgb_path=str(rgb_files[frame_id]),
        metadata_path=str(metadata_files[frame_id]), label_image_path=str(label_files[frame_id]),
    ) for frame_id in sorted(frame_ids)]


def _read_label_counts(path: Path) -> tuple[tuple[int, int], dict[int, int]]:
    with Image.open(path) as image:
        if len(image.getbands()) != 1:
            raise ValueError(f"label image must be single-channel: {path}")
        values, frequencies = np.unique(np.asarray(image), return_counts=True)
        return image.size, {int(value): int(count) for value, count in zip(values, frequencies) if int(value)}


def _read_rgb_size(path: Path) -> tuple[tuple[int, int] | None, str | None]:
    try:
        with Image.open(path) as image:
            return image.size, None
    except UnidentifiedImageError as error:
        try:
            reference = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            raise ValueError(f"cannot decode RGB image: {path}") from error
        if reference.startswith("/") and Path(reference).suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return None, reference
        raise ValueError(f"cannot decode RGB image: {path}") from error


def _validate_bbox(bbox: Iterable[object], width: int, height: int) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in bbox)
    if len(values) != 4:
        raise ValueError(f"bbox_xywh must contain four values, got {values}")
    x, y, box_width, box_height = values
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        raise ValueError(f"invalid visible bbox_xywh: {values}")
    if x + box_width > width + 1 or y + box_height > height + 1:
        raise ValueError(f"bbox_xywh exceeds image dimensions {width}x{height}: {values}")
    return values


def parse_frame_observation(frame_input: Sam2FrameInput, background_labels: Iterable[str] = DEFAULT_BACKGROUND_LABELS) -> Sam2FrameObservation:
    rgb_path, metadata_path, label_image_path = map(Path, (frame_input.rgb_path, frame_input.metadata_path, frame_input.label_image_path))
    label_size, label_counts = _read_label_counts(label_image_path)
    rgb_size, rgb_source_reference = _read_rgb_size(rgb_path)
    width, height = label_size
    if rgb_size is not None and rgb_size != label_size:
        raise ValueError(f"RGB and label dimensions do not match for {frame_input.frame_id}: rgb={rgb_size}, labels={label_size}")
    try:
        records = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read SAM2 metadata {metadata_path}: {error}") from error
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"SAM2 metadata must be a JSON array of objects: {metadata_path}")

    ignored_labels = {normalize_label(label) for label in background_labels}
    visible_metadata_ids: set[int] = set()
    objects: list[Sam2InstanceObservation] = []
    zero_area_records = background_filtered_records = 0
    for record in records:
        instance_id = int(record.get("id", 0))
        if instance_id < 1:
            raise ValueError(f"invalid SAM2 instance id in {metadata_path}: {instance_id}")
        area_pixels = int(record.get("area", 0))
        if area_pixels <= 0:
            zero_area_records += 1
            continue
        source = record.get("source") or {}
        label = normalize_label(str(source.get("label", "")))
        if not label:
            raise ValueError(f"visible SAM2 instance {instance_id} has no label in {metadata_path}")
        if label in ignored_labels:
            background_filtered_records += 1
            continue
        x, y, box_width, box_height = _validate_bbox(record.get("bbox_xywh", ()), width, height)
        track_id = int(source.get("track_id", instance_id))
        if track_id < 1:
            raise ValueError(f"invalid SAM2 track id in {metadata_path}: {track_id}")
        visible_metadata_ids.add(instance_id)
        center_x, center_y = x + box_width / 2.0, y + box_height / 2.0
        objects.append(Sam2InstanceObservation(
            instance_id=instance_id, track_id=track_id, label=label,
            bbox_xywh=(x, y, box_width, box_height), center_xy=(center_x, center_y),
            center_normalized=(center_x / width, center_y / height), area_pixels=area_pixels,
            area_ratio=area_pixels / float(width * height),
            prompted_on_this_frame=bool(source.get("prompted_on_this_frame", False)),
            predicted_iou=float(record["predicted_iou"]) if record.get("predicted_iou") is not None else None,
            present_in_label_image=instance_id in label_counts,
        ))
    mask_instance_ids = sorted(label_counts)
    mask_id_set = set(mask_instance_ids)
    return Sam2FrameObservation(
        frame_id=frame_input.frame_id, rgb_path=str(rgb_path), rgb_image_verified=rgb_size is not None,
        rgb_source_reference=rgb_source_reference, metadata_path=str(metadata_path), label_image_path=str(label_image_path),
        image_width=width, image_height=height, total_records=len(records), zero_area_records=zero_area_records,
        background_filtered_records=background_filtered_records, mask_instance_ids=mask_instance_ids,
        unmatched_metadata_ids=sorted(visible_metadata_ids - mask_id_set), unmatched_mask_ids=sorted(mask_id_set - visible_metadata_ids),
        objects=objects,
    )
