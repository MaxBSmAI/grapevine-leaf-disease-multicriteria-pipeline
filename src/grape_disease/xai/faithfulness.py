"""Blockwise insertion/deletion faithfulness with a blurred-image baseline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from grape_disease.utils.dependencies import require_module

torch = require_module("torch")
functional = require_module("torchvision.transforms.functional", "torchvision")


@dataclass(frozen=True)
class PerturbationConfig:
    """Frozen perturbation design."""

    kernel_size: int
    sigma: float
    block_size: int
    fraction_start: float
    fraction_stop: float
    fraction_step: float
    random_trials: int

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> PerturbationConfig:
        config = cls(
            kernel_size=int(value["gaussian_kernel_size"]),
            sigma=float(value["gaussian_sigma"]),
            block_size=int(value["block_size"]),
            fraction_start=float(value["fraction_start"]),
            fraction_stop=float(value["fraction_stop"]),
            fraction_step=float(value["fraction_step"]),
            random_trials=int(value["random_trials"]),
        )
        if config.kernel_size % 2 == 0 or config.kernel_size <= 0:
            raise ValueError("Gaussian kernel size must be positive and odd")
        if config.block_size <= 0 or config.random_trials <= 0:
            raise ValueError("Block size and random trials must be positive")
        return config


def _blurred_baseline(
    inputs: Any, mean: list[float], std: list[float], config: PerturbationConfig
) -> Any:
    mean_tensor = torch.tensor(mean, device=inputs.device, dtype=inputs.dtype).view(
        1, 3, 1, 1
    )
    std_tensor = torch.tensor(std, device=inputs.device, dtype=inputs.dtype).view(
        1, 3, 1, 1
    )
    pixel_space = torch.clamp(inputs * std_tensor + mean_tensor, 0.0, 1.0)
    blurred = functional.gaussian_blur(
        pixel_space,
        kernel_size=[config.kernel_size, config.kernel_size],
        sigma=[config.sigma, config.sigma],
    )
    return (blurred - mean_tensor) / std_tensor


def _block_slices(
    height: int, width: int, block_size: int
) -> list[tuple[slice, slice]]:
    if height % block_size or width % block_size:
        raise ValueError("Input dimensions must be divisible by block_size")
    return [
        (slice(row, row + block_size), slice(column, column + block_size))
        for row in range(0, height, block_size)
        for column in range(0, width, block_size)
    ]


def _normalised_partial_auc(
    fractions: npt.NDArray[Any], scores: npt.NDArray[Any]
) -> float:
    interval = float(fractions[-1] - fractions[0])
    if interval <= 0.0:
        raise ValueError("Perturbation fraction interval must be positive")
    area = float(np.asarray(np.trapz(scores, fractions)).item())
    return area / interval


def _score(model: Any, inputs: Any, target: int) -> float:
    with torch.inference_mode():
        probability = torch.softmax(model(inputs), dim=1)[0, target]
    return float(probability.detach().cpu().item())


def perturbation_curves(
    model: Any,
    inputs: Any,
    ranking_map: Any,
    target: int,
    mean: list[float],
    std: list[float],
    config: PerturbationConfig,
    block_order: npt.NDArray[Any] | None = None,
) -> dict[str, Any]:
    """Calculate insertion/deletion curves over identical non-overlapping blocks."""

    if inputs.shape[0] != 1 or ranking_map.shape[0] != 1:
        raise ValueError("Faithfulness evaluation processes one image at a time")
    height, width = int(inputs.shape[-2]), int(inputs.shape[-1])
    blocks = _block_slices(height, width, config.block_size)
    block_scores = np.asarray(
        [
            float(ranking_map[0, rows, columns].mean().detach().cpu().item())
            for rows, columns in blocks
        ]
    )
    order = (
        np.argsort(-block_scores, kind="mergesort")
        if block_order is None
        else np.asarray(block_order, dtype=np.int64)
    )
    fractions = np.arange(
        config.fraction_start,
        config.fraction_stop + config.fraction_step / 2.0,
        config.fraction_step,
    )
    baseline = _blurred_baseline(inputs, mean, std, config)
    insertion_scores: list[float] = []
    deletion_scores: list[float] = []
    for fraction in fractions:
        count = min(len(blocks), round(float(fraction) * len(blocks)))
        insertion = baseline.clone()
        deletion = inputs.clone()
        for block_index in order[:count]:
            rows, columns = blocks[int(block_index)]
            insertion[:, :, rows, columns] = inputs[:, :, rows, columns]
            deletion[:, :, rows, columns] = baseline[:, :, rows, columns]
        insertion_scores.append(_score(model, insertion, target))
        deletion_scores.append(_score(model, deletion, target))
    insertion_array = np.asarray(insertion_scores)
    deletion_array = np.asarray(deletion_scores)
    return {
        "fractions": fractions,
        "insertion_scores": insertion_array,
        "deletion_scores": deletion_array,
        "insertion_partial_auc": _normalised_partial_auc(fractions, insertion_array),
        "deletion_partial_auc": _normalised_partial_auc(fractions, deletion_array),
        "block_order": order,
        "block_count": len(blocks),
        "score_function": "softmax_target_probability",
    }


def deterministic_random_seed(
    image_id: str, checkpoint_sha256: str, method: str, trial_number: int
) -> int:
    """Derive a portable 64-bit seed from the frozen random-baseline identity."""

    payload = f"{image_id}:{checkpoint_sha256}:{method}:{trial_number}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def random_baseline_trials(
    model: Any,
    inputs: Any,
    ranking_map: Any,
    target: int,
    mean: list[float],
    std: list[float],
    config: PerturbationConfig,
    image_id: str,
    checkpoint_sha256: str,
    method: str,
) -> list[dict[str, Any]]:
    """Repeat curves with permutations of the same 196 perturbation blocks."""

    block_count = (int(inputs.shape[-2]) // config.block_size) * (
        int(inputs.shape[-1]) // config.block_size
    )
    results: list[dict[str, Any]] = []
    for trial in range(config.random_trials):
        seed = deterministic_random_seed(image_id, checkpoint_sha256, method, trial)
        order = np.random.default_rng(seed).permutation(block_count)
        curves = perturbation_curves(
            model, inputs, ranking_map, target, mean, std, config, order
        )
        results.append(
            {
                "trial": trial,
                "seed": seed,
                "insertion_partial_auc": curves["insertion_partial_auc"],
                "deletion_partial_auc": curves["deletion_partial_auc"],
            }
        )
    return results
