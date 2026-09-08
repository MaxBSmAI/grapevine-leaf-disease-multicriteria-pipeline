"""Prediction-derived multiclass metrics with no hidden model selection."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt


def confusion_matrix(
    true_labels: npt.NDArray[Any],
    predicted_labels: npt.NDArray[Any],
    num_classes: int,
) -> npt.NDArray[Any]:
    """Return a fixed-size row-true, column-predicted confusion matrix."""

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, predicted_label in zip(true_labels, predicted_labels, strict=True):
        matrix[int(true_label), int(predicted_label)] += 1
    return matrix


def _average_ranks(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def binary_roc_auc(labels: npt.NDArray[Any], scores: npt.NDArray[Any]) -> float | None:
    """Compute Mann-Whitney ROC-AUC with average ranks for ties."""

    positives = labels.astype(bool)
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = _average_ranks(scores)
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def expected_calibration_error(
    probabilities: npt.NDArray[Any],
    true_labels: npt.NDArray[Any],
    bins: int = 15,
) -> float:
    """Top-label ECE using fixed-width confidence bins including confidence one."""

    if bins <= 0:
        raise ValueError("ECE bins must be positive")
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == true_labels
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= boundaries[index]) & (
                confidence <= boundaries[index + 1]
            )
        else:
            mask = (confidence >= boundaries[index]) & (
                confidence < boundaries[index + 1]
            )
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(
            float(correct[mask].mean()) - float(confidence[mask].mean())
        )
    return ece


def classification_metrics(
    true_labels: npt.NDArray[Any],
    probabilities: npt.NDArray[Any],
    class_names: list[str],
    ece_bins: int = 15,
) -> tuple[dict[str, Any], list[dict[str, Any]], npt.NDArray[Any]]:
    """Calculate all frozen predictive metrics from saved probabilities."""

    true_labels = np.asarray(true_labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    num_classes = len(class_names)
    if probabilities.ndim != 2 or probabilities.shape != (
        len(true_labels),
        num_classes,
    ):
        raise ValueError("Probability matrix shape does not match labels/classes")
    if len(true_labels) == 0:
        raise ValueError("Metrics require at least one prediction")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or Inf")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("Probability rows must sum to one")
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(true_labels, predicted, num_classes)
    per_class: list[dict[str, Any]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    supports: list[int] = []
    auc_values: list[float] = []
    for class_index, class_name in enumerate(class_names):
        true_positive = int(matrix[class_index, class_index])
        false_positive = int(matrix[:, class_index].sum() - true_positive)
        support = int(matrix[class_index, :].sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        auc = binary_roc_auc(
            (true_labels == class_index).astype(np.int64),
            probabilities[:, class_index],
        )
        if auc is not None:
            auc_values.append(auc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
        per_class.append(
            {
                "class_id": class_index,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "roc_auc_ovr": auc,
            }
        )
    clipped = np.clip(probabilities, 1e-12, 1.0)
    log_loss = float(-np.log(clipped[np.arange(len(true_labels)), true_labels]).mean())
    one_hot = np.eye(num_classes, dtype=np.float64)[true_labels]
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    accuracy = float((predicted == true_labels).mean())
    support_total = sum(supports)
    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "weighted_f1": float(
            sum(
                score * support
                for score, support in zip(f1_scores, supports, strict=True)
            )
            / support_total
        ),
        "log_loss": log_loss,
        "multiclass_brier_score": brier,
        "expected_calibration_error": expected_calibration_error(
            probabilities, true_labels, ece_bins
        ),
        "ece_bins": ece_bins,
        "macro_roc_auc_ovr": (
            float(np.mean(auc_values)) if len(auc_values) == num_classes else None
        ),
        "sample_count": len(true_labels),
        "correct_count": int((predicted == true_labels).sum()),
        "error_count": int((predicted != true_labels).sum()),
    }
    if not all(
        math.isfinite(value) for value in metrics.values() if isinstance(value, float)
    ):
        raise ValueError("A computed metric is not finite")
    return metrics, per_class, matrix
