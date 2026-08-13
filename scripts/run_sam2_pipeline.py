from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.sam2_pipeline import run_sam2_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete offline SAM2 observation and description pipeline."
    )
    parser.add_argument(
        "--rgb-dir",
        default=str(WORKSPACE_ROOT / "rgb" / "rgb"),
        help="Directory containing RGB PNG files.",
    )
    parser.add_argument(
        "--sam2-dir",
        default=str(WORKSPACE_ROOT / "sam2_video" / "sam2_video"),
        help="Directory containing SAM2 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "sam2_description"),
        help="Destination for all generated artifacts.",
    )
    args = parser.parse_args()

    report = run_sam2_pipeline(
        rgb_dir=Path(args.rgb_dir),
        sam2_dir=Path(args.sam2_dir),
        output_dir=Path(args.output_dir),
    )
    result = {
        "success": report.success,
        "frame_count": report.frame_count,
        "rgb_verified_frame_count": report.rgb_verified_frame_count,
        "visible_observation_count": report.visible_observation_count,
        "unique_track_count": report.unique_track_count,
        "rendered_frame_count": report.rendered_frame_count,
        "description_frame_count": report.description_frame_count,
        "base_source_counts": report.base_source_counts,
        "speech_text": report.speech_text,
        "acceptance_report_path": str(Path(args.output_dir) / "acceptance_report.json"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report.success else 1)


if __name__ == "__main__":
    main()
