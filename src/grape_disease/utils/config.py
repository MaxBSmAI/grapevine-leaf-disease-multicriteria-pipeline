"""Validated JSON and YAML configuration loading for command-line tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a user-supplied configuration is invalid."""


def load_json_config(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject other top-level types."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot load configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration must be a JSON object: {path}")
    return value


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping with safe parsing and a clear error contract."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration must be a YAML mapping: {path}")
    return value


def load_structured_config(path: Path) -> dict[str, Any]:
    """Load JSON or YAML based on the file suffix."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json_config(path)
    if suffix in {".yaml", ".yml"}:
        return load_yaml_config(path)
    raise ConfigurationError(f"Unsupported configuration format: {path}")


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    """Return a mapping-like object as a dictionary or raise a clear error."""

    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be an object")
    return value
