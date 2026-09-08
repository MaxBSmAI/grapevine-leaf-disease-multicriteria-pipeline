"""Independently audit a generated original-image manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_source_tree_to_path

add_source_tree_to_path()

from grape_disease.data.audit import audit_original_manifest  # noqa: E402
from grape_disease.data.config import DatasetConfig  # noqa: E402
from grape_disease.utils.config import load_json_config  # noqa: E402
from grape_disease.utils.logging import configure_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--verify-content-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    config = DatasetConfig.from_mapping(load_json_config(args.config))
    report = audit_original_manifest(
        dataset_root=args.dataset_root,
        manifest_dir=args.manifest_dir,
        output_dir=args.output_dir,
        config=config,
        verify_content_hashes=args.verify_content_hashes,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("passed", args.dry_run) else 2


if __name__ == "__main__":
    raise SystemExit(main())
