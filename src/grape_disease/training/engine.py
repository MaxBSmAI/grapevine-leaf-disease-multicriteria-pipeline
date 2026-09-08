"""Shared auditable train and validation loops."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from grape_disease.evaluation.metrics import classification_metrics
from grape_disease.utils.dependencies import require_module

torch = require_module("torch")


@dataclass(frozen=True)
class EpochResult:
    """One complete epoch record before serialisation."""

    loss: float
    optimizer_updates: int
    batches: int
    samples: int
    elapsed_seconds: float
    mean_grad_norm: float | None
    effective_class_counts: dict[int, int]
    unique_originals: int
    repeated_samples: int


def _grad_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            norm = float(parameter.grad.detach().norm(2).item())
            total += norm * norm
    return math.sqrt(total)


def train_one_epoch(
    model: Any,
    loader: Any,
    loss_function: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    device: Any,
    remaining_updates: int,
    gradient_accumulation_steps: int,
    gradient_clip_norm: float | None,
    amp: bool,
) -> EpochResult:
    """Train until the loader ends or the global update budget is exhausted."""

    if remaining_updates <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("Update budget and accumulation steps must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    loss_sum = 0.0
    samples = 0
    batches = 0
    updates = 0
    grad_norms: list[float] = []
    class_counts: Counter[int] = Counter()
    observed_originals: list[str] = []
    for batch_index, batch in enumerate(loader):
        inputs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda", enabled=amp and device.type == "cuda"
        ):
            logits = model(inputs)
            loss = loss_function(logits, labels) / gradient_accumulation_steps
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is NaN or Inf")
        scaler.scale(loss).backward()
        actual_loss = float(loss.detach().item()) * gradient_accumulation_steps
        batch_size = int(labels.shape[0])
        loss_sum += actual_loss * batch_size
        samples += batch_size
        batches += 1
        class_counts.update(int(value) for value in labels.detach().cpu().tolist())
        observed_originals.extend(str(value) for value in batch["original_id"])
        should_update = (batch_index + 1) % gradient_accumulation_steps == 0
        is_last_batch = batch_index + 1 == len(loader)
        if should_update or is_last_batch:
            scaler.unscale_(optimizer)
            norm = _grad_norm(model.parameters())
            grad_norms.append(norm)
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            updates += 1
            if updates >= remaining_updates:
                break
    elapsed = time.perf_counter() - started
    if samples == 0:
        raise RuntimeError("Training loader yielded no samples")
    unique_originals = len(set(observed_originals))
    return EpochResult(
        loss=loss_sum / samples,
        optimizer_updates=updates,
        batches=batches,
        samples=samples,
        elapsed_seconds=elapsed,
        mean_grad_norm=float(np.mean(grad_norms)) if grad_norms else None,
        effective_class_counts=dict(sorted(class_counts.items())),
        unique_originals=unique_originals,
        repeated_samples=len(observed_originals) - unique_originals,
    )


def validate_one_epoch(
    model: Any,
    loader: Any,
    loss_function: Any,
    device: Any,
    class_names: list[str],
) -> tuple[dict[str, Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Evaluate validation only and return loss plus prediction-derived metrics."""

    model.eval()
    loss_sum = 0.0
    sample_count = 0
    labels_all: list[int] = []
    probabilities_all: list[npt.NDArray[Any]] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(inputs)
            loss = loss_function(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Validation loss is NaN or Inf")
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            label_values = labels.detach().cpu().numpy().astype(np.int64)
            batch_size = len(label_values)
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
            labels_all.extend(label_values.tolist())
            probabilities_all.extend(probabilities)
    if sample_count == 0:
        raise RuntimeError("Validation loader yielded no samples")
    label_array = np.asarray(labels_all, dtype=np.int64)
    probability_array = np.asarray(probabilities_all, dtype=np.float64)
    metrics, _, _ = classification_metrics(label_array, probability_array, class_names)
    metrics["validation_loss"] = loss_sum / sample_count
    return metrics, label_array, probability_array
