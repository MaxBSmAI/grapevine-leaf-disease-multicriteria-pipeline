"""Atomic persistence for final and resumable PyTorch checkpoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from grape_disease.utils.dependencies import require_module
from grape_disease.utils.hashing import sha256_file
from grape_disease.utils.io import write_text

torch = require_module("torch")


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_checkpoint_once(path: Path, payload: dict[str, Any]) -> str:
    """Save a checkpoint atomically, refuse overwrite, and write its digest."""

    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    _atomic_torch_save(path, payload)
    digest = sha256_file(path)
    write_text(path.with_name("checkpoint_sha256.txt"), digest)
    return digest


def save_progress_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    """Atomically create or replace one explicitly resumable checkpoint."""

    _atomic_torch_save(path, payload)
    digest = sha256_file(path)
    write_text(path.with_suffix(f"{path.suffix}.sha256"), digest)
    return digest


def load_checkpoint(path: Path, map_location: str | Any = "cpu") -> dict[str, Any]:
    """Load a dictionary checkpoint and reject incompatible payload types."""

    value = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    return value
