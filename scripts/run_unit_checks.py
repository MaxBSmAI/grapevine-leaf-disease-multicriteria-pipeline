"""Run the Gate 2 unit-test suite without requiring pytest."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

from _bootstrap import add_source_tree_to_path

REPOSITORY_ROOT = add_source_tree_to_path()

from grape_disease.utils.logging import configure_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    if args.dry_run:
        print("Would discover tests under tests/unit")
        return 0
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(REPOSITORY_ROOT / "tests" / "unit"),
        pattern="test_*.py",
        top_level_dir=str(REPOSITORY_ROOT),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
