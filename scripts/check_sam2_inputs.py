from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sam2_observation import match_frame_inputs, parse_frame_observation


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate paired RGB and SAM2 frame outputs.")
    parser.add_argument("--rgb-dir", required=True, help="Directory containing source RGB PNG files.")
    parser.add_argument("--sam2-dir", required=True, help="Directory containing SAM2 JSON and label PNG files.")
    args = parser.parse_args()

    frame_inputs = match_frame_inputs(Path(args.rgb_dir), Path(args.sam2_dir))
    frames = [parse_frame_observation(frame_input) for frame_input in frame_inputs]
    labels = Counter(item.label for frame in frames for item in frame.objects)
    unmatched_frames = [
        {
            "frame_id": frame.frame_id,
            "unmatched_metadata_ids": frame.unmatched_metadata_ids,
            "unmatched_mask_ids": frame.unmatched_mask_ids,
        }
        for frame in frames
        if frame.unmatched_metadata_ids or frame.unmatched_mask_ids
    ]
    summary = {
        "matched_frames": len(frames),
        "total_records": sum(frame.total_records for frame in frames),
        "zero_area_records": sum(frame.zero_area_records for frame in frames),
        "background_filtered_records": sum(frame.background_filtered_records for frame in frames),
        "visible_records": sum(len(frame.objects) for frame in frames),
        "labels": dict(sorted(labels.items())),
        "mask_metadata_mismatch_frames": unmatched_frames,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(1 if unmatched_frames else 0)


if __name__ == "__main__":
    main()
