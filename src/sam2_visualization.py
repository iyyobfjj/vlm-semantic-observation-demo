from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from .sam2_language import label_name_zh
from .sam2_schema import Sam2FrameObservation, Sam2OverlayArtifact, Sam2VisualizationManifest


COLORS = ((24, 160, 251), (250, 204, 21), (244, 114, 182), (52, 211, 153), (251, 113, 133), (167, 139, 250), (251, 146, 60), (34, 211, 238))


def _font(image_width: int) -> ImageFont.ImageFont:
    size = max(16, image_width // 80)
    for candidate in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_base_image(frame: Sam2FrameObservation) -> tuple[Image.Image, str, Path]:
    rgb_path = Path(frame.rgb_path)
    try:
        with Image.open(rgb_path) as image:
            return image.convert("RGBA"), "rgb", rgb_path
    except (OSError, UnidentifiedImageError):
        fallback_path = Path(frame.label_image_path).with_name(f"{frame.frame_id}_overlay.png")
        try:
            with Image.open(fallback_path) as image:
                return image.convert("RGBA"), "sam2_overlay", fallback_path
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"no usable RGB or SAM2 overlay image for frame {frame.frame_id}") from error


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    interior = np.zeros_like(mask, dtype=bool)
    interior[1:-1, 1:-1] = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    return mask & ~interior


def _apply_mask(canvas: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha_value: int) -> None:
    alpha = Image.fromarray(np.where(mask, alpha_value, 0).astype(np.uint8), mode="L")
    layer = Image.new("RGBA", canvas.size, color + (0,))
    layer.putalpha(alpha)
    canvas.alpha_composite(layer)


def _draw_label(draw: ImageDraw.ImageDraw, text: str, box: tuple[float, float, float, float], color: tuple[int, int, int], font: ImageFont.ImageFont, image_size: tuple[int, int]) -> None:
    x, y, width, height = box
    line_width = max(2, image_size[0] // 640)
    draw.rectangle((x, y, x + width, y + height), outline=color + (255,), width=line_width)
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
    padding = max(4, line_width * 2)
    label_x = min(max(0, int(x)), max(0, image_size[0] - text_width - padding * 2))
    label_y = max(0, int(y) - text_height - padding * 2)
    draw.rectangle((label_x, label_y, label_x + text_width + padding * 2, label_y + text_height + padding * 2), fill=(15, 23, 42, 220))
    draw.text((label_x + padding, label_y + padding - text_box[1]), text, fill=(255, 255, 255, 255), font=font)


def render_frame_overlay(frame: Sam2FrameObservation, output_path: Path) -> Sam2OverlayArtifact:
    canvas, base_source, base_path = _load_base_image(frame)
    if canvas.size != (frame.image_width, frame.image_height):
        raise ValueError(f"base image dimensions do not match frame {frame.frame_id}: base={canvas.size}, expected={(frame.image_width, frame.image_height)}")
    with Image.open(frame.label_image_path) as label_image:
        labels = np.asarray(label_image)
    if labels.shape != (frame.image_height, frame.image_width):
        raise ValueError(f"label dimensions do not match frame {frame.frame_id}")
    draw, font = ImageDraw.Draw(canvas, "RGBA"), _font(frame.image_width)
    rendered_instance_ids: list[int] = []
    for item in frame.objects:
        color = COLORS[(item.track_id - 1) % len(COLORS)]
        mask = labels == item.instance_id
        if mask.any():
            if base_source == "rgb":
                _apply_mask(canvas, mask, color, 72)
            _apply_mask(canvas, _mask_boundary(mask), color, 255)
        _draw_label(draw, f"{label_name_zh(item.label)} / {item.label}  track:{item.track_id}", item.bbox_xywh, color, font, canvas.size)
        center_x, center_y = item.center_xy
        radius = max(4, frame.image_width // 320)
        draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=color + (255,), outline=(255, 255, 255, 255), width=max(1, radius // 3))
        rendered_instance_ids.append(item.instance_id)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, format="PNG")
    return Sam2OverlayArtifact(frame_id=frame.frame_id, visualization_path=str(output_path), base_source=base_source, base_path=str(base_path), object_count=len(frame.objects), rendered_instance_ids=rendered_instance_ids)


def render_visualization_set(frames: Iterable[Sam2FrameObservation], output_dir: Path) -> Sam2VisualizationManifest:
    ordered_frames = sorted(frames, key=lambda frame: frame.frame_id)
    output_dir = Path(output_dir)
    artifacts = [render_frame_overlay(frame, output_dir / "overlays" / f"{frame.frame_id}.png") for frame in ordered_frames]
    manifest = Sam2VisualizationManifest(frame_count=len(ordered_frames), rendered_frame_count=len(artifacts), object_observation_count=sum(item.object_count for item in artifacts), base_source_counts=dict(sorted(Counter(item.base_source for item in artifacts).items())), artifacts=artifacts)
    (output_dir / "visualization_manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
