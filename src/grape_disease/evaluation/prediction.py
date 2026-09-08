"""Model inference and complete per-image prediction artefacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from grape_disease.evaluation.metrics import classification_metrics
from grape_disease.utils.dependencies import require_module
from grape_disease.utils.io import ensure_output_directory, write_csv, write_json

torch = require_module("torch")


def predict_loader(
    model: Any,
    loader: Any,
    device: Any,
    experiment_metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], npt.NDArray[Any], npt.NDArray[Any]]:
    """Run logits-only inference and return rows plus arrays used for metrics."""

    model.eval()
    rows: list[dict[str, Any]] = []
    all_labels: list[int] = []
    all_probabilities: list[npt.NDArray[Any]] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["image"].to(device, non_blocking=True)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy().astype(np.int64)
            predicted = probabilities.argmax(axis=1)
            for index in range(len(labels)):
                row = {
                    **experiment_metadata,
                    "image_id": batch["image_id"][index],
                    "original_id": batch["original_id"][index],
                    "group_id": batch["group_id"][index],
                    "true_class": int(labels[index]),
                    "predicted_class": int(predicted[index]),
                    "confidence": float(probabilities[index, predicted[index]]),
                    "correct": bool(predicted[index] == labels[index]),
                }
                for class_index, probability in enumerate(probabilities[index]):
                    row[f"probability_class_{class_index}"] = float(probability)
                rows.append(row)
            all_labels.extend(labels.tolist())
            all_probabilities.extend(probabilities)
    return rows, np.asarray(all_labels), np.asarray(all_probabilities)


def write_prediction_evaluation(
    output_dir: Path,
    prediction_filename: str,
    metrics_filename: str,
    rows: list[dict[str, Any]],
    true_labels: npt.NDArray[Any],
    probabilities: npt.NDArray[Any],
    class_names: list[str],
) -> dict[str, Any]:
    """Derive and persist predictions, metrics, per-class rows, and confusion."""

    if not rows:
        raise ValueError("Cannot write an empty prediction artefact")
    metrics, per_class, matrix = classification_metrics(
        true_labels, probabilities, class_names
    )
    output = ensure_output_directory(output_dir)
    partition_name = "test" if "test" in metrics_filename.lower() else "validation"
    prediction_fields = tuple(rows[0].keys())
    write_csv(output / prediction_filename, rows, prediction_fields)
    write_json(output / metrics_filename, metrics)
    write_csv(
        output / f"metrics_per_class_{partition_name}.csv",
        per_class,
        (
            "class_id",
            "class_name",
            "precision",
            "recall",
            "f1",
            "support",
            "roc_auc_ovr",
        ),
    )
    confusion_rows = [
        {
            "true_class_id": true_index,
            **{
                f"predicted_class_{predicted_index}": int(
                    matrix[true_index, predicted_index]
                )
                for predicted_index in range(len(class_names))
            },
        }
        for true_index in range(len(class_names))
    ]
    confusion_fields = (
        "true_class_id",
        *(f"predicted_class_{index}" for index in range(len(class_names))),
    )
    write_csv(
        output / f"confusion_matrix_{partition_name}.csv",
        confusion_rows,
        confusion_fields,
    )
    return metrics
