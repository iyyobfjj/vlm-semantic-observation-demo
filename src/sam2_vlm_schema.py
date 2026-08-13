from __future__ import annotations

from pydantic import BaseModel, Field


class Sam2VlmDescription(BaseModel):
    object_name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    visual_description: str = Field(min_length=1)
    visible_state: str = Field(min_length=1)
    attributes: list[str] = Field(default_factory=list)
    interaction_parts: list[str] = Field(default_factory=list)
    occlusion: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: list[str] = Field(default_factory=list)


class Sam2VlmEvidence(BaseModel):
    rgb_path: str
    sam2_metadata_path: str
    sam2_label_image_path: str
    evidence_image_path: str


class Sam2VlmObjectResult(BaseModel):
    schema_version: str = "sam2_vlm_object.v1"
    task_id: str
    object_key: str
    sam2_label: str
    sam2_track_id: int = Field(ge=1)
    representative_frame_id: str
    representative_instance_id: int = Field(ge=1)
    observation_count: int = Field(gt=0)
    observation_frame_ids: list[str]
    position_scope: str = "image_2d_only"
    position_2d_zh: str
    center_normalized: tuple[float, float]
    area_ratio: float = Field(gt=0.0, le=1.0)
    pointcloud_path: str | None = None
    evidence: Sam2VlmEvidence
    description: Sam2VlmDescription
    vlm_call_metrics: dict[str, object] = Field(default_factory=dict)


class Sam2VlmFailure(BaseModel):
    object_key: str
    sam2_label: str
    sam2_track_id: int = Field(ge=1)
    representative_frame_id: str
    evidence_image_path: str | None = None
    raw_response: str | None = None
    error_type: str
    error: str


class Sam2VlmTaskSummary(BaseModel):
    schema_version: str = "sam2_vlm_task_summary.v1"
    success: bool
    task_id: str
    rgb_dir: str
    sam2_dir: str
    output_dir: str
    frame_count: int = Field(ge=0)
    visible_observation_count: int = Field(ge=0)
    unique_track_count: int = Field(ge=0)
    described_track_count: int = Field(ge=0)
    failed_track_count: int = Field(ge=0)
    mask_metadata_alignment_ok: bool
    object_result_paths: list[str]
    failures: list[Sam2VlmFailure]


class Sam2VlmAcceptanceReport(BaseModel):
    schema_version: str = "sam2_vlm_acceptance.v1"
    success: bool
    task_id: str
    checks: dict[str, bool]
    counts: dict[str, int]
    artifacts: dict[str, str]
