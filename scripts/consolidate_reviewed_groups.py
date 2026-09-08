"""Validate human review decisions and consolidate deterministic group IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_source_tree_to_path

add_source_tree_to_path()

from grape_disease.data.group_review import (  # noqa: E402
    GroupReviewConfig,
    validate_and_consolidate_groups,
)
from grape_disease.utils.config import load_json_config  # noqa: E402
from grape_disease.utils.logging import configure_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--candidates-path", type=Path, required=True)
    parser.add_argument("--decisions-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    config = GroupReviewConfig.from_mapping(load_json_config(args.config))
    report = validate_and_consolidate_groups(
        manifest_path=args.manifest_path,
        candidates_path=args.candidates_path,
        decisions_path=args.decisions_path,
        output_dir=args.output_dir,
        config=config,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
