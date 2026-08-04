from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sam2_pipeline import run_sam2_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete offline SAM2 observation and description pipeline.")
    parser.add_argument("--rgb-dir", required=True, help="Directory containing RGB PNG files.")
    parser.add_argument("--sam2-dir", required=True, help="Directory containing SAM2 outputs.")
    parser.add_argument("--output-dir", default="outputs/sam2_description", help="Destination for generated artifacts.")
    args = parser.parse_args()
    report = run_sam2_pipeline(Path(args.rgb_dir), Path(args.sam2_dir), Path(args.output_dir))
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    raise SystemExit(0 if report.success else 1)


if __name__ == "__main__":
    main()
