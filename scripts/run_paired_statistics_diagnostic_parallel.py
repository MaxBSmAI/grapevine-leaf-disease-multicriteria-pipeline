"""Parallel diagnostic paired statistics runner.

This script is intentionally non-publication. It bypasses the frozen-input hash
check so reduced-bootstrap diagnostics can be executed without changing the
frozen protocol. It parallelises independent pairwise comparisons across CPU
workers. Use scripts/run_paired_statistics.py for protocol-valid final statistics.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from grape_disease.statistics.paired import (  # noqa: E402
    PairedBootstrapConfig,
    crossed_paired_bootstrap,
    exact_mcnemar,
    holm_adjust,
)
from grape_disease.utils.config import load_structured_config  # noqa: E402
from grape_disease.utils.io import ensure_output_directory, write_csv, write_json  # noqa: E402
from grape_disease.utils.logging import configure_logging  # noqa: E402


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _load_predictions(results_root: Path) -> pd.DataFrame:
    paths = sorted(results_root.rglob("predictions_test.csv"))
    if not paths:
        raise ValueError(f"No test prediction CSV files found below {results_root}")
    frame = pd.concat(
        [pd.read_csv(path, dtype={"image_id": str, "group_id": str}) for path in paths],
        ignore_index=True,
    )
    required = {
        "model_name",
        "regime",
        "seed",
        "group_id",
        "image_id",
        "true_class",
        "predicted_class",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction registry lacks columns: {missing}")
    return frame


def _one_comparison(task: dict[str, Any]) -> dict[str, Any]:
    predictions = task["predictions"]
    class_names = task["class_names"]
    bootstrap_config = task["bootstrap_config"]
    comparison_type = task["comparison_type"]
    family = task["family"]
    left_name = task["left"]
    right_name = task["right"]

    if comparison_type == "architectures_within_regime":
        regime = task["regime"]
        left = predictions.loc[
            (predictions["regime"] == regime)
            & (predictions["model_name"] == left_name)
        ]
        right = predictions.loc[
            (predictions["regime"] == regime)
            & (predictions["model_name"] == right_name)
        ]
    elif comparison_type == "regimes_within_architecture":
        architecture = task["architecture"]
        left = predictions.loc[
            (predictions["model_name"] == architecture)
            & (predictions["regime"] == left_name)
        ]
        right = predictions.loc[
            (predictions["model_name"] == architecture)
            & (predictions["regime"] == right_name)
        ]
    else:
        raise ValueError(f"Unknown comparison_type: {comparison_type}")

    result = crossed_paired_bootstrap(left, right, class_names, bootstrap_config)
    merged = left.merge(
        right,
        on=["seed", "group_id", "image_id", "true_class"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    mcnemar = exact_mcnemar(
        merged["predicted_class_left"] == merged["true_class"],
        merged["predicted_class_right"] == merged["true_class"],
    )
    return {
        "task_index": task["task_index"],
        "family": family,
        "left": left_name,
        "right": right_name,
        **result,
        "mcnemar_raw_p_value": mcnemar["raw_p_value"],
        "mcnemar_discordant_pairs": mcnemar["discordant_pairs"],
    }


def _build_tasks(
    predictions: pd.DataFrame,
    hypotheses: dict[str, Any],
    class_names: list[str],
    bootstrap_config: PairedBootstrapConfig,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    families = hypotheses["families"]
    architecture_family = families["architectures_within_regime"]
    for regime in architecture_family["regimes"]:
        for left_name, right_name in architecture_family["comparisons"]:
            tasks.append(
                {
                    "task_index": len(tasks),
                    "comparison_type": "architectures_within_regime",
                    "family": f"architectures_within_{regime}",
                    "regime": regime,
                    "left": left_name,
                    "right": right_name,
                    "predictions": predictions,
                    "class_names": class_names,
                    "bootstrap_config": bootstrap_config,
                }
            )
    regime_family = families["regimes_within_architecture"]
    for architecture in regime_family["architectures"]:
        for left_name, right_name in regime_family["comparisons"]:
            tasks.append(
                {
                    "task_index": len(tasks),
                    "comparison_type": "regimes_within_architecture",
                    "family": f"regimes_within_{architecture}",
                    "architecture": architecture,
                    "left": left_name,
                    "right": right_name,
                    "predictions": predictions,
                    "class_names": class_names,
                    "bootstrap_config": bootstrap_config,
                }
            )
    return tasks


def _write_outputs(rows: list[dict[str, Any]], predictions: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: int(row["task_index"]))
    for row in rows:
        row.pop("task_index", None)

    adjusted_rows: list[dict[str, Any]] = []
    for _, family_rows in pd.DataFrame(rows).groupby("family", sort=False):
        family_dicts = family_rows.to_dict("records")
        adjusted = holm_adjust(row["raw_p_value"] for row in family_dicts)
        for row, value in zip(family_dicts, adjusted, strict=True):
            row["holm_adjusted_p_value"] = value
            adjusted_rows.append(row)

    output = ensure_output_directory(output_dir)
    csv_rows = [
        {
            **row,
            "confidence_interval_low": row["confidence_interval"][0],
            "confidence_interval_high": row["confidence_interval"][1],
        }
        for row in adjusted_rows
    ]
    for row in csv_rows:
        row.pop("confidence_interval")
    write_csv(output / "paired_predictive_statistics.csv", csv_rows, tuple(csv_rows[0]))
    report = {
        "schema_version": "1.0.0",
        "comparisons": adjusted_rows,
        "multiplicity": "holm_separately_within_each_preregistered_family",
        "prediction_rows": len(predictions),
        "parallel_execution": True,
    }
    write_json(output / "paired_predictive_statistics.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run non-publication parallel diagnostic paired statistics."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=424242)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = load_structured_config(_path(args.config))
    paths = {name: _path(value) for name, value in config["paths"].items()}
    hypotheses = load_structured_config(paths["hypotheses"])
    predictions = _load_predictions(_path(args.results_root))
    bootstrap_config = PairedBootstrapConfig(
        int(args.bootstrap_resamples),
        int(args.bootstrap_seed),
        float(args.confidence_level),
    )
    tasks = _build_tasks(
        predictions,
        hypotheses,
        [str(name) for name in config["classes"]],
        bootstrap_config,
    )
    output_dir = _path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_note = {
        "publication_eligible": False,
        "reason": "Reduced-bootstrap parallel diagnostic run; frozen statistics_config remains unchanged.",
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "confidence_level": float(args.confidence_level),
        "workers": int(args.workers),
        "results_root": str(_path(args.results_root)),
        "comparison_count": len(tasks),
    }
    write_json(output_dir / "DIAGNOSTIC_NOT_FOR_PUBLICATION.json", diagnostic_note)

    if args.dry_run:
        print(
            {
                "dry_run": True,
                "prediction_rows": len(predictions),
                "test_files_opened": len(list(_path(args.results_root).rglob("predictions_test.csv"))),
                "comparisons": len(tasks),
                "workers": int(args.workers),
                "bootstrap_resamples": int(args.bootstrap_resamples),
            }
        )
        return 0

    start_time = time.perf_counter()
    print(
        "[statistics-parallel] Starting paired predictive statistics: "
        f"{len(tasks)} comparisons, {args.bootstrap_resamples} bootstrap resamples each, "
        f"workers={args.workers}.",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_to_task = {executor.submit(_one_comparison, task): task for task in tasks}
        for completed, future in enumerate(as_completed(future_to_task), start=1):
            task = future_to_task[future]
            row = future.result()
            rows.append(row)
            elapsed = time.perf_counter() - start_time
            average = elapsed / completed
            remaining = average * (len(tasks) - completed)
            print(
                "[statistics-parallel] "
                f"{completed}/{len(tasks)} completed: "
                f"{task['family']}: {task['left']} vs {task['right']} | "
                f"elapsed={elapsed / 60:.1f} min | eta={remaining / 60:.1f} min",
                flush=True,
            )

    report = _write_outputs(rows, predictions, output_dir)
    total_elapsed = time.perf_counter() - start_time
    print(
        "[statistics-parallel] Completed paired predictive statistics. "
        f"elapsed={total_elapsed / 60:.1f} min. "
        f"Wrote outputs to {output_dir}.",
        flush=True,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
