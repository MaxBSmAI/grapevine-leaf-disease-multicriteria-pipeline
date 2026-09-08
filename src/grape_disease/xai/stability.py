"""Map stability, ranking overlap, and sanity-check utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def _array_structural_similarity(
    first: npt.NDArray[Any],
    second: npt.NDArray[Any],
    window_size: int,
) -> float:
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("SSIM window must be odd and at least three")
    if min(first.shape) < window_size:
        raise ValueError("SSIM window is larger than an attribution map")
    first_windows = np.lib.stride_tricks.sliding_window_view(
        first, (window_size, window_size)
    )
    second_windows = np.lib.stride_tricks.sliding_window_view(
        second, (window_size, window_size)
    )
    axes = (-2, -1)
    first_mean = first_windows.mean(axis=axes)
    second_mean = second_windows.mean(axis=axes)
    first_variance = first_windows.var(axis=axes, ddof=1)
    second_variance = second_windows.var(axis=axes, ddof=1)
    covariance = (
        (first_windows - first_mean[..., None, None])
        * (second_windows - second_mean[..., None, None])
    ).sum(axis=axes) / (window_size * window_size - 1)
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * first_mean * second_mean + c1) * (2.0 * covariance + c2)
    denominator = (first_mean**2 + second_mean**2 + c1) * (
        first_variance + second_variance + c2
    )
    scores = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator != 0,
    )
    return float(np.clip(scores.mean(), -1.0, 1.0))


def _ranks(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman_ranking_correlation(
    first: npt.NDArray[Any], second: npt.NDArray[Any]
) -> float:
    """Calculate Pearson correlation between deterministic ordinal ranks."""

    first_flat = np.asarray(first, dtype=np.float64).ravel()
    second_flat = np.asarray(second, dtype=np.float64).ravel()
    if first_flat.shape != second_flat.shape:
        raise ValueError("Stability maps must share a shape")
    first_rank = _ranks(first_flat)
    second_rank = _ranks(second_flat)
    if np.std(first_rank) == 0.0 or np.std(second_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def top_k_overlap(
    first: npt.NDArray[Any], second: npt.NDArray[Any], fraction: float
) -> float:
    """Return Jaccard overlap between equal-sized top attribution sets."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("Top-k fraction must be in (0, 1]")
    first_flat = np.asarray(first).ravel()
    second_flat = np.asarray(second).ravel()
    if first_flat.shape != second_flat.shape:
        raise ValueError("Stability maps must share a shape")
    count = max(1, round(fraction * len(first_flat)))
    first_indices = set(np.argsort(-first_flat, kind="mergesort")[:count].tolist())
    second_indices = set(np.argsort(-second_flat, kind="mergesort")[:count].tolist())
    return len(first_indices & second_indices) / len(first_indices | second_indices)


def compare_maps(
    first: npt.NDArray[Any],
    second: npt.NDArray[Any],
    top_k_fraction: float = 0.10,
    ssim_window_size: int = 11,
) -> dict[str, Any]:
    """Return the three frozen complementary stability measures."""

    first_array = np.asarray(first, dtype=np.float64).squeeze()
    second_array = np.asarray(second, dtype=np.float64).squeeze()
    return {
        "spearman": spearman_ranking_correlation(first_array, second_array),
        "ssim": _array_structural_similarity(
            first_array, second_array, ssim_window_size
        ),
        "top_k_fraction": top_k_fraction,
        "top_k_jaccard_overlap": top_k_overlap(
            first_array, second_array, top_k_fraction
        ),
    }
