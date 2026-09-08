from __future__ import annotations

import numpy as np

from grape_disease.evaluation.metrics import classification_metrics


def test_perfect_predictions_have_unit_primary_metrics() -> None:
    labels = np.asarray([0, 1, 2, 3, 0, 1, 2, 3])
    probabilities = np.eye(4)[labels]
    metrics, per_class, matrix = classification_metrics(
        labels, probabilities, ["a", "b", "c", "d"]
    )
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert len(per_class) == 4
    assert np.array_equal(matrix, np.diag([2, 2, 2, 2]))
