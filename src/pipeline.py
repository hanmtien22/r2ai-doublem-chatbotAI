#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import DEFAULT_CONFIG_PATH, load_config
from src.ingestion.pipeline import run_ingestion_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy pipeline ingestion báo cáo tài chính.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Đường dẫn file YAML cấu hình.",
    )
    parser.add_argument(
        "--source-dir",
        help="Ghi đè source_dir trong file cấu hình.",
    )
    parser.add_argument(
        "--output-dir",
        help="Ghi đè output_dir trong file cấu hình.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if args.source_dir:
        config["source_dir"] = args.source_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir

    result = run_ingestion_pipeline(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
