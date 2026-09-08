"""Write the test unlock only after explicit, attributable Gate-9 approval."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from _bootstrap import add_source_tree_to_path

REPOSITORY_ROOT = add_source_tree_to_path()

from _entrypoints import _paths  # noqa: E402
from grape_disease.training.freeze import build_test_lock_record  # noqa: E402
from grape_disease.utils.config import load_structured_config  # noqa: E402
from grape_disease.utils.gates import require_gate_approval  # noqa: E402
from grape_disease.utils.io import ensure_output_directory, write_json  # noqa: E402
from grape_disease.utils.logging import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorised-by", required=True)
    parser.add_argument("--authorisation-date", required=True)
    parser.add_argument("--confirm-test-access", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = load_structured_config(args.config)
    paths = _paths(REPOSITORY_ROOT, config)
    require_gate_approval(paths["gate_approvals"], 9)
    if not args.confirm_test_access:
        raise PermissionError("--confirm-test-access is mandatory")
    authorisation_date = date.fromisoformat(args.authorisation_date)
    if authorisation_date > date.today():
        raise ValueError("Test authorisation date cannot be in the future")
    record = build_test_lock_record(
        paths["frozen_experiment"],
        args.authorised_by,
        authorisation_date.isoformat(),
        explicit_authorisation=True,
    )
    if not args.dry_run:
        output = ensure_output_directory(args.output_dir)
        if output != paths["test_lock"].parent.resolve():
            raise ValueError("--output-dir must be the configured test-lock directory")
        destination = paths["test_lock"]
        if destination.exists():
            current = json.loads(destination.read_text(encoding="utf-8"))
            if current.get("test_access") is True:
                raise FileExistsError("Test access is already authorised")
        write_json(destination, record)
    print(json.dumps({**record, "dry_run": args.dry_run}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
