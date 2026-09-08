"""Run the gated FP32 batch-size 16 CUDA memory preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_source_tree_to_path

REPOSITORY_ROOT = add_source_tree_to_path()

from _entrypoints import _paths  # noqa: E402
from grape_disease.training.smoke import run_batch_memory_preflight  # noqa: E402
from grape_disease.utils.config import load_structured_config  # noqa: E402
from grape_disease.utils.gates import require_gate_approval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_structured_config(args.config)
    paths = _paths(REPOSITORY_ROOT, config)
    require_gate_approval(paths["gate_approvals"], 5)
    report = run_batch_memory_preflight(
        REPOSITORY_ROOT,
        args.output_dir,
        batch_size=int(config["tuning"]["base_training"]["batch_size"]),
        device_name=str(config["device"]),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
