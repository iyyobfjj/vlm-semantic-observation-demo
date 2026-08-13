from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sam2_export import export_sam2_observations
from src.sam2_observation import match_frame_inputs, parse_frame_observation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-frame SAM2 observations and a track-level scene summary."
    )
    parser.add_argument("--rgb-dir", required=True, help="Directory containing source RGB PNG files.")
    parser.add_argument("--sam2-dir", required=True, help="Directory containing SAM2 JSON and label PNG files.")
    parser.add_argument("--output-dir", required=True, help="Destination for generated JSON files.")
    args = parser.parse_args()

    frame_inputs = match_frame_inputs(Path(args.rgb_dir), Path(args.sam2_dir))
    frames = [parse_frame_observation(frame_input) for frame_input in frame_inputs]
    output_dir = Path(args.output_dir)
    summary = export_sam2_observations(frames, output_dir)
    result = {
        "success": summary.mask_metadata_alignment_ok,
        "frame_count": summary.frame_count,
        "visible_observation_count": summary.visible_observation_count,
        "unique_track_count": summary.unique_track_count,
        "frame_output_dir": str(output_dir / "frames"),
        "scene_summary_path": str(output_dir / "scene_summary.json"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary.mask_metadata_alignment_ok else 1)


if __name__ == "__main__":
    main()
