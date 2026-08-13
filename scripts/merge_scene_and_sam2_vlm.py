from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.observation_bundle import build_observation_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle whole-scene VLM JSON and SAM2 track-level VLM JSON."
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--scene-json-dir", required=True)
    parser.add_argument("--sam2-vlm-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bundle = build_observation_bundle(
        task_id=args.task_id,
        scene_json_dir=Path(args.scene_json_dir),
        sam2_vlm_dir=Path(args.sam2_vlm_dir),
        output_path=Path(args.output),
    )
    print(
        json.dumps(
            {
                "success": True,
                "task_id": bundle["task_id"],
                "scene_observation_count": bundle["scene_observation_count"],
                "object_observation_count": bundle["object_observation_count"],
                "output": str(Path(args.output)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
