"""Execute the pre-registered predictive comparison families."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from grape_disease.statistics.paired import (
    PairedBootstrapConfig,
    crossed_paired_bootstrap,
    exact_mcnemar,
    holm_adjust,
)
from grape_disease.utils.io import ensure_output_directory, write_csv, write_json


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


def run_predictive_statistics(
    results_root: Path,
    output_dir: Path,
    hypotheses: dict[str, Any],
    class_names: list[str],
    bootstrap_config: PairedBootstrapConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run each architecture/regime comparison and Holm-adjust within family."""

    predictions = _load_predictions(results_root)
    if dry_run:
        return {
            "dry_run": True,
            "prediction_rows": len(predictions),
            "test_files_opened": len(list(results_root.rglob("predictions_test.csv"))),
        }
    rows: list[dict[str, Any]] = []
    families = hypotheses["families"]
    architecture_family = families["architectures_within_regime"]
    regime_family = families["regimes_within_architecture"]
    total_comparisons = (
        len(architecture_family["regimes"])
        * len(architecture_family["comparisons"])
        + len(regime_family["architectures"])
        * len(regime_family["comparisons"])
    )
    completed_comparisons = 0
    print(
        "[statistics] Starting paired predictive statistics: "
        f"{total_comparisons} comparisons, "
        f"{bootstrap_config.resamples} bootstrap resamples each.",
        flush=True,
    )
    for regime in architecture_family["regimes"]:
        family_rows: list[dict[str, Any]] = []
        for left_name, right_name in architecture_family["comparisons"]:
            completed_comparisons += 1
            print(
                "[statistics] "
                f"{completed_comparisons}/{total_comparisons} "
                f"architectures_within_{regime}: {left_name} vs {right_name}",
                flush=True,
            )
            left = predictions.loc[
                (predictions["regime"] == regime)
                & (predictions["model_name"] == left_name)
            ]
            right = predictions.loc[
                (predictions["regime"] == regime)
                & (predictions["model_name"] == right_name)
            ]
            result = crossed_paired_bootstrap(
                left, right, class_names, bootstrap_config
            )
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
            family_rows.append(
                {
                    "family": f"architectures_within_{regime}",
                    "left": left_name,
                    "right": right_name,
                    **result,
                    "mcnemar_raw_p_value": mcnemar["raw_p_value"],
                    "mcnemar_discordant_pairs": mcnemar["discordant_pairs"],
                }
            )
        adjusted = holm_adjust(row["raw_p_value"] for row in family_rows)
        for row, value in zip(family_rows, adjusted, strict=True):
            row["holm_adjusted_p_value"] = value
        rows.extend(family_rows)
    for architecture in regime_family["architectures"]:
        family_rows = []
        for left_name, right_name in regime_family["comparisons"]:
            completed_comparisons += 1
            print(
                "[statistics] "
                f"{completed_comparisons}/{total_comparisons} "
                f"regimes_within_{architecture}: {left_name} vs {right_name}",
                flush=True,
            )
            left = predictions.loc[
                (predictions["model_name"] == architecture)
                & (predictions["regime"] == left_name)
            ]
            right = predictions.loc[
                (predictions["model_name"] == architecture)
                & (predictions["regime"] == right_name)
            ]
            result = crossed_paired_bootstrap(
                left, right, class_names, bootstrap_config
            )
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
            family_rows.append(
                {
                    "family": f"regimes_within_{architecture}",
                    "left": left_name,
                    "right": right_name,
                    **result,
                    "mcnemar_raw_p_value": mcnemar["raw_p_value"],
                    "mcnemar_discordant_pairs": mcnemar["discordant_pairs"],
                }
            )
        adjusted = holm_adjust(row["raw_p_value"] for row in family_rows)
        for row, value in zip(family_rows, adjusted, strict=True):
            row["holm_adjusted_p_value"] = value
        rows.extend(family_rows)
    output = ensure_output_directory(output_dir)
    csv_rows = [
        {
            **row,
            "confidence_interval_low": row["confidence_interval"][0],
            "confidence_interval_high": row["confidence_interval"][1],
        }
        for row in rows
    ]
    for row in csv_rows:
        row.pop("confidence_interval")
    write_csv(output / "paired_predictive_statistics.csv", csv_rows, tuple(csv_rows[0]))
    report = {
        "schema_version": "1.0.0",
        "comparisons": rows,
        "multiplicity": "holm_separately_within_each_preregistered_family",
        "prediction_rows": len(predictions),
    }
    write_json(output / "paired_predictive_statistics.json", report)
    print(
        "[statistics] Completed paired predictive statistics. "
        f"Wrote outputs to {output}.",
        flush=True,
    )
    return report
