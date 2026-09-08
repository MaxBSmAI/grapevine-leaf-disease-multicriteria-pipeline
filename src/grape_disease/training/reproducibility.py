"""Central PyTorch reproducibility controls and RNG checkpoint state."""

from __future__ import annotations

import os
import random
import warnings
from typing import Any

import numpy as np

from grape_disease.utils.dependencies import require_module

torch = require_module("torch")


def set_reproducibility(seed: int, deterministic_level: str) -> dict[str, Any]:
    """Seed all supported RNGs and configure one named determinism level."""

    if deterministic_level not in {"strict", "balanced", "performance"}:
        raise ValueError(f"Unknown deterministic level: {deterministic_level}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_level == "performance":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(
            True, warn_only=deterministic_level == "balanced"
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return {
        "seed": seed,
        "deterministic_level": deterministic_level,
        "dataloader_generator": generator,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU, and optional CUDA RNG state."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    def _as_byte_tensor(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=torch.uint8, device="cpu")
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_byte_tensor(state["torch_cpu"]))
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [_as_byte_tensor(cuda_state) for cuda_state in state["torch_cuda"]]
        )


def determinism_warning_context() -> Any:
    """Capture nondeterminism warnings for run metadata."""

    return warnings.catch_warnings(record=True)
