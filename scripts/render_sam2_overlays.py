from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sam2_observation import match_frame_inputs, parse_frame_observation
from src.sam2_visualization import render_visualization_set


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render labeled SAM2 instance overlays for dashboard and review."
    )
    parser.add_argument("--rgb-dir", required=True, help="Directory containing source RGB PNG files.")
    parser.add_argument("--sam2-dir", required=True, help="Directory containing SAM2 outputs.")
    parser.add_argument("--output-dir", required=True, help="Existing SAM2 description output directory.")
    args = parser.parse_args()

    frame_inputs = match_frame_inputs(Path(args.rgb_dir), Path(args.sam2_dir))
    frames = [parse_frame_observation(frame_input) for frame_input in frame_inputs]
    output_dir = Path(args.output_dir)
    manifest = render_visualization_set(frames, output_dir)
    result = {
        "success": manifest.rendered_frame_count == manifest.frame_count,
        "frame_count": manifest.frame_count,
        "rendered_frame_count": manifest.rendered_frame_count,
        "object_observation_count": manifest.object_observation_count,
        "base_source_counts": manifest.base_source_counts,
        "overlay_dir": str(output_dir / "overlays"),
        "manifest_path": str(output_dir / "visualization_manifest.json"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
