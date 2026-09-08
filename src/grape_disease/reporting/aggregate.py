"""Aggregate seed-level results without hiding individual observations."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from grape_disease.utils.io import ensure_output_directory, write_csv, write_json


def _read_run_metrics(run_dir: Path) -> dict[str, Any] | None:
    metadata_path = run_dir / "run_metadata.json"
    metrics_path = run_dir / "metrics_test.json"
    if not metrics_path.is_file():
        metrics_path = run_dir / "validation_metrics.json"
    if not metadata_path.is_file() or not metrics_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "experiment_id": metadata["experiment_id"],
        "model_name": metadata.get("model_metadata", {})
        .get("spec", {})
        .get("name", metadata.get("model_name")),
        "regime": metadata.get("regime"),
        "seed": metadata.get("seed"),
        "evaluation_partition": (
            "test" if (run_dir / "metrics_test.json").is_file() else "validation"
        ),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        **{
            key: value
            for key, value in metrics.items()
            if isinstance(value, int | float)
        },
    }


def aggregate_results(results_root: Path, output_dir: Path) -> dict[str, Any]:
    """Collect runs and report arithmetic seed means, standard deviations and CIs."""

    rows = [
        value
        for path in results_root.rglob("run_metadata.json")
        if (value := _read_run_metrics(path.parent)) is not None
    ]
    if not rows:
        raise ValueError(f"No completed run metrics found below {results_root}")
    frame = pd.DataFrame(rows)
    output = ensure_output_directory(output_dir)
    write_csv(output / "individual_seed_results.csv", rows, tuple(frame.columns))
    metric_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "experiment_id",
            "model_name",
            "regime",
            "seed",
            "evaluation_partition",
            "checkpoint_sha256",
        }
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["model_name", "regime", "evaluation_partition"], dropna=False
    ):
        row: dict[str, Any] = {
            "model_name": keys[0],
            "regime": keys[1],
            "evaluation_partition": keys[2],
            "seed_count": len(group),
        }
        for metric in metric_columns:
            values = [float(value) for value in group[metric].dropna()]
            if not values:
                continue
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            half_width = 1.96 * std / (len(values) ** 0.5) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half_width
            row[f"{metric}_ci95_high"] = mean + half_width
        aggregate_rows.append(row)
    fields = tuple(dict.fromkeys(key for row in aggregate_rows for key in row))
    write_csv(output / "aggregate_results.csv", aggregate_rows, fields)
    summary = {
        "schema_version": "1.0.0",
        "run_count": len(rows),
        "combination_count": len(aggregate_rows),
        "primary_estimator": "arithmetic_mean_of_seed_level_macro_f1",
        "individual_results_retained": True,
        "post_hoc_weighted_ranking_used": False,
    }
    write_json(output / "aggregation_metadata.json", summary)
    return summary
