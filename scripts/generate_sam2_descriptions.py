from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sam2_description import export_chinese_descriptions
from src.sam2_observation import match_frame_inputs, parse_frame_observation


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic Chinese SAM2 object and speech descriptions.")
    parser.add_argument("--rgb-dir", required=True, help="Directory containing source RGB PNG files.")
    parser.add_argument("--sam2-dir", required=True, help="Directory containing SAM2 outputs.")
    parser.add_argument("--output-dir", required=True, help="SAM2 description output directory.")
    args = parser.parse_args()
    frames = [parse_frame_observation(item) for item in match_frame_inputs(Path(args.rgb_dir), Path(args.sam2_dir))]
    output_dir = Path(args.output_dir)
    summary = export_chinese_descriptions(frames, output_dir)
    print(json.dumps({
        "success": True, "frame_count": summary.frame_count, "unique_track_count": summary.unique_track_count,
        "speech_text": summary.speech_text, "description_dir": str(output_dir / "descriptions" / "frames"),
        "speech_summary_path": str(output_dir / "speech_summary.json"), "text_summary_path": str(output_dir / "summary.txt"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
