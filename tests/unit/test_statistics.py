from __future__ import annotations

import pandas as pd

from grape_disease.statistics.paired import (
    PairedBootstrapConfig,
    crossed_paired_bootstrap,
    exact_mcnemar,
    holm_adjust,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed": [1, 1, 1, 1, 2, 2, 2, 2],
            "group_id": ["a", "b", "c", "d"] * 2,
            "image_id": ["a", "b", "c", "d"] * 2,
            "true_class": [0, 1, 2, 3] * 2,
            "predicted_class": [0, 1, 2, 3] * 2,
        }
    )


def test_identical_systems_have_zero_paired_effect() -> None:
    predictions = _predictions()
    result = crossed_paired_bootstrap(
        predictions,
        predictions.copy(),
        ["a", "b", "c", "d"],
        PairedBootstrapConfig(resamples=50, seed=7),
    )
    assert result["difference_left_minus_right"] == 0.0
    assert result["confidence_interval"] == [0.0, 0.0]
    assert result["raw_p_value"] == 1.0


def test_holm_is_monotone_and_mcnemar_exact() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]
    mcnemar = exact_mcnemar([True, True, False], [False, False, False])
    assert mcnemar["discordant_pairs"] == 2
    assert mcnemar["raw_p_value"] == 0.5
