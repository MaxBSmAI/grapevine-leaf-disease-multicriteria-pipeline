"""Optional dependency checks and software provenance helpers."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from typing import Any


class MissingDependencyError(ImportError):
    """Raised with an actionable message when an execution extra is absent."""


def require_module(module_name: str, install_name: str | None = None) -> Any:
    """Import one optional module or explain which package must be installed."""

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        package = install_name or module_name
        raise MissingDependencyError(
            f"Missing dependency '{package}'. Run scripts/setup_environment.ps1."
        ) from exc


def package_version(name: str) -> str | None:
    """Return an installed distribution version without importing the package."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_software_versions() -> dict[str, Any]:
    """Collect versions relevant to provenance while tolerating CPU-only hosts."""

    packages = (
        "torch",
        "torchvision",
        "ultralytics",
        "captum",
        "grad-cam",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "optuna",
        "Pillow",
        "PyYAML",
    )
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in packages},
    }
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        result["torch_runtime"] = None
        return result
    result["torch_runtime"] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    return result
