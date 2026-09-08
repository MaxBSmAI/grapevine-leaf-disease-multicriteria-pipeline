"""Explicit loss functions for the three frozen training regimes."""

from __future__ import annotations

from typing import Any

from grape_disease.utils.dependencies import require_module

torch = require_module("torch")
nn = torch.nn
functional = torch.nn.functional


class FocalLoss(nn.Module):  # type: ignore[misc, name-defined]
    """Unweighted multiclass focal loss with alpha fixed to None."""

    def __init__(self, gamma: float, reduction: str = "mean") -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError("Focal gamma must be non-negative")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: Any, targets: Any) -> Any:
        log_probabilities = functional.log_softmax(logits, dim=1)
        log_probability_true = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        cross_entropy = -log_probability_true
        probability_true = log_probability_true.exp().clamp(min=0.0, max=1.0)
        modulating_factor = torch.pow((1.0 - probability_true).clamp(min=0.0), self.gamma)
        loss = modulating_factor * cross_entropy
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(regime: str, focal_gamma: float | None = None) -> Any:
    """Return the only loss allowed for one frozen regime."""

    if regime in {"standard_ce", "balanced_ce"}:
        return nn.CrossEntropyLoss()
    if regime == "focal_loss":
        if focal_gamma is None:
            raise ValueError("focal_loss requires a selected gamma")
        return FocalLoss(gamma=focal_gamma)
    raise ValueError(f"Unknown training regime: {regime}")
