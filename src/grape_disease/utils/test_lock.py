"""Fail-closed test-partition access guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LockedTestAccessError(PermissionError):
    """Raised when a command attempts to access a locked test partition."""


REQUIRED_UNLOCK_FIELDS = (
    "authorised_by",
    "authorisation_date",
    "frozen_protocol_sha256",
    "code_commit",
    "manifest_sha256",
    "split_sha256",
    "frozen_experiment_sha256",
    "hypotheses_sha256",
    "final_config_sha256",
    "explicit_evaluation_authorisation",
)


def load_test_lock(path: Path) -> dict[str, Any]:
    """Load and validate the basic shape of the test lock."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LockedTestAccessError(f"Cannot validate test lock {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LockedTestAccessError("Test lock must contain a JSON object")
    return value


def require_test_access(path: Path) -> dict[str, Any]:
    """Return a valid unlocked record or fail closed."""

    value = load_test_lock(path)
    if value.get("test_access") is not True:
        raise LockedTestAccessError(str(value.get("reason", "Test access is locked")))
    missing = [field for field in REQUIRED_UNLOCK_FIELDS if not value.get(field)]
    if missing:
        raise LockedTestAccessError(
            f"Test lock is incomplete; missing authorisation fields: {missing}"
        )
    return value
