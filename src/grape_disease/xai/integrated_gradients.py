"""Adaptive Integrated Gradients with relative completeness convergence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from grape_disease.utils.dependencies import require_module

torch = require_module("torch")


@dataclass(frozen=True)
class IntegratedGradientsConfig:
    """Frozen common IG configuration."""

    method: str
    initial_steps: int
    retry_steps: tuple[int, ...]
    internal_batch_size: int
    convergence_relative_error_threshold: float

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> IntegratedGradientsConfig:
        config = cls(
            method=str(value["method"]),
            initial_steps=int(value["initial_steps"]),
            retry_steps=tuple(int(item) for item in value["retry_steps"]),
            internal_batch_size=int(value["internal_batch_size"]),
            convergence_relative_error_threshold=float(
                value["convergence_relative_error_threshold"]
            ),
        )
        steps = (config.initial_steps, *config.retry_steps)
        if any(step <= 0 for step in steps) or tuple(sorted(steps)) != steps:
            raise ValueError("IG steps must be positive and strictly ordered")
        if config.method != "gausslegendre":
            raise ValueError("Frozen IG integration method is gausslegendre")
        return config


def explain_tensor(
    model: Any,
    inputs: Any,
    target: int,
    config: IntegratedGradientsConfig,
) -> dict[str, Any]:
    """Compute IG, retrying fixed step counts until relative completeness passes."""

    captum = require_module("captum.attr", "captum")
    baseline = torch.zeros_like(inputs)
    integrated_gradients = captum.IntegratedGradients(model)
    selected_attributions = None
    selected_delta = None
    relative_error = float("inf")
    selected_steps = config.initial_steps
    model.eval()
    with torch.no_grad():
        input_difference = float(
            abs(model(inputs)[0, target].item() - model(baseline)[0, target].item())
        )
    denominator = max(input_difference, 1e-6)
    for steps in (config.initial_steps, *config.retry_steps):
        attributions, delta = integrated_gradients.attribute(
            inputs,
            baselines=baseline,
            target=target,
            n_steps=steps,
            method=config.method,
            internal_batch_size=config.internal_batch_size,
            return_convergence_delta=True,
        )
        delta_value = float(torch.as_tensor(delta).abs().max().detach().cpu().item())
        relative_error = delta_value / denominator
        selected_attributions = attributions
        selected_delta = delta_value
        selected_steps = steps
        if relative_error <= config.convergence_relative_error_threshold:
            break
    if selected_attributions is None or selected_delta is None:
        raise RuntimeError("Integrated Gradients produced no attribution")
    if not torch.isfinite(selected_attributions).all():
        raise FloatingPointError("Integrated Gradients contains NaN or Inf")
    ranking_map = selected_attributions.abs().sum(dim=1)
    signed_map = selected_attributions.sum(dim=1)
    minimum = ranking_map.amin(dim=(-2, -1), keepdim=True)
    maximum = ranking_map.amax(dim=(-2, -1), keepdim=True)
    normalised_map = (ranking_map - minimum) / (maximum - minimum + 1e-12)
    return {
        "attributions": selected_attributions,
        "ranking_map": ranking_map,
        "signed_map": signed_map,
        "normalised_map": normalised_map,
        "convergence_delta": selected_delta,
        "relative_completeness_error": relative_error,
        "n_steps": selected_steps,
        "converged": relative_error <= config.convergence_relative_error_threshold,
        "target": target,
        "baseline": "zero_tensor_in_normalised_space",
    }


def save_attribution_bundle(path: Path, result: dict[str, Any]) -> None:
    """Store full, signed, ranking, and display maps in one compressed bundle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        attributions=result["attributions"].detach().cpu().numpy(),
        ranking_map=result["ranking_map"].detach().cpu().numpy(),
        signed_map=result["signed_map"].detach().cpu().numpy(),
        normalised_map=result["normalised_map"].detach().cpu().numpy(),
        convergence_delta=np.asarray(result["convergence_delta"]),
        relative_completeness_error=np.asarray(result["relative_completeness_error"]),
        n_steps=np.asarray(result["n_steps"]),
        target=np.asarray(result["target"]),
    )
