from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .json_utils import read_json, write_json
from .sam2_vlm_schema import Sam2VlmObjectResult
from .schema import SceneDescriptionObservation, SemanticObservation


def _load_scene_observation(path: Path) -> tuple[str, dict[str, Any]]:
    payload = read_json(path)
    try:
        if "scene_brief" in payload:
            observation = SceneDescriptionObservation.model_validate(payload)
            return "scene_description", observation.model_dump(mode="json")
        if "scene_summary" in payload:
            observation = SemanticObservation.model_validate(payload)
            return "light_inspection", observation.model_dump(mode="json")
    except ValidationError as error:
        raise ValueError(f"invalid scene VLM JSON {path}: {error}") from error
    raise ValueError(f"unsupported scene VLM JSON {path}")


def build_observation_bundle(
    task_id: str,
    scene_json_dir: Path,
    sam2_vlm_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not task_id.strip():
        raise ValueError("task_id must not be empty")

    scene_json_dir = Path(scene_json_dir)
    sam2_vlm_dir = Path(sam2_vlm_dir)
    output_path = Path(output_path)
    if not scene_json_dir.is_dir():
        raise FileNotFoundError(f"scene JSON directory not found: {scene_json_dir}")

    task_summary_path = sam2_vlm_dir / "task_summary.json"
    objects_dir = sam2_vlm_dir / "objects"
    if not task_summary_path.is_file():
        raise FileNotFoundError(f"SAM2 VLM task summary not found: {task_summary_path}")
    if not objects_dir.is_dir():
        raise FileNotFoundError(f"SAM2 VLM objects directory not found: {objects_dir}")

    task_summary = read_json(task_summary_path)
    if task_summary.get("task_id") != task_id:
        raise ValueError(
            f"task_id mismatch: requested {task_id!r}, "
            f"SAM2 VLM summary has {task_summary.get('task_id')!r}"
        )

    scene_entries: list[dict[str, Any]] = []
    for path in sorted(scene_json_dir.glob("*.json")):
        if path.name.endswith(".failed.json"):
            continue
        kind, observation = _load_scene_observation(path)
        scene_entries.append(
            {
                "source_path": str(path.resolve()),
                "kind": kind,
                "observation": observation,
            }
        )
    if not scene_entries:
        raise ValueError(f"no valid scene VLM JSON found in {scene_json_dir}")

    object_entries: list[dict[str, Any]] = []
    for path in sorted(objects_dir.glob("*.json")):
        try:
            observation = Sam2VlmObjectResult.model_validate(read_json(path))
        except ValidationError as error:
            raise ValueError(f"invalid SAM2 VLM object JSON {path}: {error}") from error
        if observation.task_id != task_id:
            raise ValueError(
                f"task_id mismatch in {path}: expected {task_id!r}, "
                f"found {observation.task_id!r}"
            )
        object_entries.append(
            {
                "source_path": str(path.resolve()),
                "observation": observation.model_dump(mode="json"),
            }
        )

    bundle = {
        "schema_version": "semantic_observation_bundle.v1",
        "task_id": task_id,
        "merge_policy": "layered_no_identity_inference",
        "scene_observation_count": len(scene_entries),
        "object_observation_count": len(object_entries),
        "scene_observations": scene_entries,
        "sam2_object_observations": object_entries,
        "notes": [
            "scene observations describe whole images",
            "SAM2 object observations describe stable segmented tracks",
            "no object identity match is inferred between the two layers",
            "pointcloud_path is evidence linkage, not pointcloud geometry analysis",
        ],
    }
    write_json(output_path, bundle)
    return bundle
