"""Diagnostic paired statistics runner with configurable bootstrap resamples.

This script is intentionally non-publication. It bypasses the frozen-input hash
check so reduced-bootstrap diagnostics can be executed without changing the
frozen protocol. Use scripts/run_paired_statistics.py for protocol-valid final
statistics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from grape_disease.statistics.paired import PairedBootstrapConfig  # noqa: E402
from grape_disease.statistics.runner import run_predictive_statistics  # noqa: E402
from grape_disease.utils.config import load_structured_config  # noqa: E402
from grape_disease.utils.io import write_json  # noqa: E402
from grape_disease.utils.logging import configure_logging  # noqa: E402


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run non-publication diagnostic paired statistics."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=424242)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = load_structured_config(_path(args.config))
    paths = {name: _path(value) for name, value in config["paths"].items()}
    hypotheses = load_structured_config(paths["hypotheses"])
    bootstrap_config = PairedBootstrapConfig(
        int(args.bootstrap_resamples),
        int(args.bootstrap_seed),
        float(args.confidence_level),
    )
    output_dir = _path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_note = {
        "publication_eligible": False,
        "reason": "Reduced-bootstrap diagnostic run; frozen statistics_config remains unchanged.",
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "confidence_level": float(args.confidence_level),
        "results_root": str(_path(args.results_root)),
    }
    write_json(output_dir / "DIAGNOSTIC_NOT_FOR_PUBLICATION.json", diagnostic_note)
    result = run_predictive_statistics(
        _path(args.results_root),
        output_dir,
        hypotheses,
        [str(name) for name in config["classes"]],
        bootstrap_config,
        args.dry_run,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
