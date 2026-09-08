"""Crossed paired seed-by-group bootstrap and exact McNemar tests."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from grape_disease.evaluation.metrics import classification_metrics


@dataclass(frozen=True)
class PairedBootstrapConfig:
    resamples: int = 10_000
    seed: int = 424_242
    confidence_level: float = 0.95


def _macro_f1(frame: pd.DataFrame, class_names: list[str]) -> float:
    labels = frame["true_class"].to_numpy(dtype=int)
    predictions = frame["predicted_class"].to_numpy(dtype=int)
    probabilities = np.zeros((len(frame), len(class_names)), dtype=np.float64)
    probabilities[np.arange(len(frame)), predictions] = 1.0
    metrics, _, _ = classification_metrics(labels, probabilities, class_names)
    return float(metrics["macro_f1"])


def _validate_pair(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "group_id", "image_id", "true_class"]
    required = set(keys) | {"predicted_class"}
    for name, frame in (("left", left), ("right", right)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions lack columns: {sorted(missing)}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} predictions contain duplicate paired keys")
    merged = left[[*keys, "predicted_class"]].merge(
        right[[*keys, "predicted_class"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_left", "_right"),
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Paired predictions do not contain identical observations")
    return merged.drop(columns="_merge")


def crossed_paired_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    class_names: list[str],
    config: PairedBootstrapConfig | None = None,
) -> dict[str, Any]:
    """Compare systems by paired resampling of seeds and biological groups."""

    config = config or PairedBootstrapConfig()
    merged = _validate_pair(left, right)
    left_view = merged.rename(columns={"predicted_class_left": "predicted_class"})
    right_view = merged.rename(columns={"predicted_class_right": "predicted_class"})
    point = float(
        np.mean(
            [
                _macro_f1(left_view.loc[left_view["seed"] == seed], class_names)
                for seed in sorted(left_view["seed"].unique())
            ]
        )
        - np.mean(
            [
                _macro_f1(right_view.loc[right_view["seed"] == seed], class_names)
                for seed in sorted(right_view["seed"].unique())
            ]
        )
    )
    seeds = np.array(sorted(merged["seed"].unique()))
    groups = np.array(sorted(merged["group_id"].unique()))
    rng = np.random.default_rng(config.seed)
    differences = np.empty(config.resamples, dtype=np.float64)
    for index in range(config.resamples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        left_scores: list[float] = []
        right_scores: list[float] = []
        for sampled_seed in sampled_seeds:
            chunks = [
                merged[
                    (merged["seed"] == sampled_seed)
                    & (merged["group_id"] == sampled_group)
                ]
                for sampled_group in sampled_groups
            ]
            sampled = pd.concat(chunks, ignore_index=True)
            left_scores.append(
                _macro_f1(
                    sampled.rename(columns={"predicted_class_left": "predicted_class"}),
                    class_names,
                )
            )
            right_scores.append(
                _macro_f1(
                    sampled.rename(
                        columns={"predicted_class_right": "predicted_class"}
                    ),
                    class_names,
                )
            )
        differences[index] = np.mean(left_scores) - np.mean(right_scores)
    alpha = 1.0 - config.confidence_level
    p_value = min(
        1.0,
        2.0
        * min(
            (np.count_nonzero(differences <= 0.0) + 1) / (config.resamples + 1),
            (np.count_nonzero(differences >= 0.0) + 1) / (config.resamples + 1),
        ),
    )
    return {
        "difference_left_minus_right": point,
        "confidence_interval": [
            float(np.quantile(differences, alpha / 2.0)),
            float(np.quantile(differences, 1.0 - alpha / 2.0)),
        ],
        "confidence_level": config.confidence_level,
        "raw_p_value": p_value,
        "resamples": config.resamples,
        "bootstrap_seed": config.seed,
        "design": "crossed_paired_seed_by_group",
    }


def _binomial_lower_tail_p_equal_half(trials: int, successes: int) -> float:
    """Return P[X <= successes] for X ~ Binomial(trials, 0.5) stably."""

    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        - trials * math.log(2.0)
        for index in range(successes + 1)
    ]
    maximum = max(log_terms)
    return float(math.exp(maximum) * sum(math.exp(value - maximum) for value in log_terms))


def exact_mcnemar(
    left_correct: Iterable[bool], right_correct: Iterable[bool]
) -> dict[str, Any]:
    """Return a two-sided exact McNemar test for paired correctness.

    The exact binomial tail is evaluated in log space to avoid overflow for
    large numbers of discordant pairs.
    """

    pairs = list(zip(left_correct, right_correct, strict=True))
    left_only = sum(left and not right for left, right in pairs)
    right_only = sum(right and not left for left, right in pairs)
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(left_only, right_only)
        p_value = min(1.0, 2.0 * _binomial_lower_tail_p_equal_half(discordant, lower))
    return {
        "left_correct_right_incorrect": left_only,
        "left_incorrect_right_correct": right_only,
        "discordant_pairs": discordant,
        "raw_p_value": p_value,
        "method": "exact_two_sided_mcnemar",
    }


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""

    values = np.asarray(list(p_values), dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, original_index in enumerate(order):
        running = max(running, (count - rank) * values[original_index])
        adjusted[original_index] = min(1.0, running)
    return [float(value) for value in adjusted]
