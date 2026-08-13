from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.sam2_vlm_task import run_sam2_vlm_task
from src.vlm_client import VLMClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Describe stable SAM2 tracks with an external VLM and save traceable artifacts."
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--rgb-dir", default=str(WORKSPACE_ROOT / "rgb" / "rgb"))
    parser.add_argument("--sam2-dir", default=str(WORKSPACE_ROOT / "sam2_video" / "sam2_video"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pointcloud-map", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = run_sam2_vlm_task(
        rgb_dir=Path(args.rgb_dir),
        sam2_dir=Path(args.sam2_dir),
        output_dir=Path(args.output_dir),
        task_id=args.task_id,
        analyzer=VLMClient(model=args.model),
        pointcloud_map_path=Path(args.pointcloud_map) if args.pointcloud_map else None,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary.success else 1)


if __name__ == "__main__":
    main()
