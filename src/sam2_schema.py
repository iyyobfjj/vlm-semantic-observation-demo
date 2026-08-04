from __future__ import annotations

from pydantic import BaseModel, Field


class Sam2FrameInput(BaseModel):
    frame_id: str
    rgb_path: str
    metadata_path: str
    label_image_path: str


class Sam2InstanceObservation(BaseModel):
    instance_id: int = Field(ge=1)
    track_id: int = Field(ge=1)
    label: str
    bbox_xywh: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    center_normalized: tuple[float, float]
    area_pixels: int = Field(gt=0)
    area_ratio: float = Field(gt=0.0, le=1.0)
    prompted_on_this_frame: bool = False
    predicted_iou: float | None = None
    present_in_label_image: bool


class Sam2FrameObservation(BaseModel):
    schema_version: str = "sam2_frame_observation.v1"
    frame_id: str
    rgb_path: str
    rgb_image_verified: bool
    rgb_source_reference: str | None = None
    metadata_path: str
    label_image_path: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    total_records: int = Field(ge=0)
    zero_area_records: int = Field(ge=0)
    background_filtered_records: int = Field(ge=0)
    mask_instance_ids: list[int] = Field(default_factory=list)
    unmatched_metadata_ids: list[int] = Field(default_factory=list)
    unmatched_mask_ids: list[int] = Field(default_factory=list)
    objects: list[Sam2InstanceObservation] = Field(default_factory=list)


class Sam2TrackSummary(BaseModel):
    track_id: int = Field(ge=1)
    label: str
    observation_count: int = Field(gt=0)
    frame_ids: list[str]
    first_frame_id: str
    last_frame_id: str
    representative_frame_id: str
    representative_center_xy: tuple[float, float]
    representative_center_normalized: tuple[float, float]
    max_area_pixels: int = Field(gt=0)
    max_area_ratio: float = Field(gt=0.0, le=1.0)


class Sam2SceneSummary(BaseModel):
    schema_version: str = "sam2_scene_summary.v1"
    frame_count: int = Field(ge=0)
    frame_ids: list[str]
    frames_with_objects: int = Field(ge=0)
    visible_observation_count: int = Field(ge=0)
    unique_track_count: int = Field(ge=0)
    label_observation_counts: dict[str, int]
    label_track_counts: dict[str, int]
    total_records: int = Field(ge=0)
    zero_area_records: int = Field(ge=0)
    background_filtered_records: int = Field(ge=0)
    rgb_unverified_frame_ids: list[str]
    mask_metadata_mismatch_frame_ids: list[str]
    mask_metadata_alignment_ok: bool
    tracks: list[Sam2TrackSummary]


class Sam2OverlayArtifact(BaseModel):
    frame_id: str
    visualization_path: str
    base_source: str
    base_path: str
    object_count: int = Field(ge=0)
    rendered_instance_ids: list[int]


class Sam2VisualizationManifest(BaseModel):
    schema_version: str = "sam2_visualization_manifest.v1"
    frame_count: int = Field(ge=0)
    rendered_frame_count: int = Field(ge=0)
    object_observation_count: int = Field(ge=0)
    base_source_counts: dict[str, int]
    artifacts: list[Sam2OverlayArtifact]


class Sam2LocalizedObjectDescription(BaseModel):
    instance_id: int = Field(ge=1)
    track_id: int = Field(ge=1)
    label: str
    name_zh: str
    position_zh: str
    center_normalized: tuple[float, float]
    area_ratio: float = Field(gt=0.0, le=1.0)


class Sam2FrameDescription(BaseModel):
    schema_version: str = "sam2_frame_description.v1"
    frame_id: str
    object_count: int = Field(ge=0)
    summary_zh: str
    speech_text: str
    position_scope: str = "image_2d_only"
    position_note_zh: str
    visualization_path: str
    visualization_available: bool
    objects: list[Sam2LocalizedObjectDescription]


class Sam2InventoryItem(BaseModel):
    label: str
    name_zh: str
    track_count: int = Field(gt=0)
    observation_count: int = Field(gt=0)
    representative_positions_zh: list[str]


class Sam2SpeechSummary(BaseModel):
    schema_version: str = "sam2_speech_summary.v1"
    frame_count: int = Field(ge=0)
    frames_with_objects: int = Field(ge=0)
    visible_observation_count: int = Field(ge=0)
    unique_track_count: int = Field(ge=0)
    summary_zh: str
    speech_text: str
    position_scope: str = "image_2d_only"
    position_note_zh: str
    inventory: list[Sam2InventoryItem]


class Sam2PipelineReport(BaseModel):
    schema_version: str = "sam2_pipeline_report.v1"
    success: bool
    rgb_dir: str
    sam2_dir: str
    output_dir: str
    frame_count: int = Field(ge=0)
    rgb_verified_frame_count: int = Field(ge=0)
    visible_observation_count: int = Field(ge=0)
    unique_track_count: int = Field(ge=0)
    mask_metadata_alignment_ok: bool
    rendered_frame_count: int = Field(ge=0)
    description_frame_count: int = Field(ge=0)
    base_source_counts: dict[str, int]
    speech_text: str
    artifacts: dict[str, str]
