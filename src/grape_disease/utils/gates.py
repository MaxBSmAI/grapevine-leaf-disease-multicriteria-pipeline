"""Fail-closed protocol-gate validation shared by all mutating stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grape_disease.utils.hashing import sha256_file


class GateApprovalError(PermissionError):
    """Raised when a scientific stage has not received recorded approval."""


def load_gate_approvals(path: Path) -> dict[str, Any]:
    """Load the versioned gate record and validate its basic structure."""

    try:
        # ``utf-8-sig`` accepts both canonical UTF-8 and files exported by
        # Windows tools with a leading BOM.
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateApprovalError(f"Cannot read gate approvals {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("gates"), dict):
        raise GateApprovalError("Gate approval record must contain a gates object")
    return value


def require_gate_approval(path: Path, gate_number: int) -> dict[str, Any]:
    """Return one explicitly approved gate or fail closed."""

    value = load_gate_approvals(path)
    gate_key = f"gate_{gate_number}"
    gate = value["gates"].get(gate_key)
    if not isinstance(gate, dict) or gate.get("approved") is not True:
        reason = gate.get("reason") if isinstance(gate, dict) else "missing record"
        raise GateApprovalError(f"{gate_key} is not approved: {reason}")
    required = ("approved_by", "approval_date", "evidence")
    missing = [field for field in required if not gate.get(field)]
    if missing:
        raise GateApprovalError(f"{gate_key} approval is incomplete: {missing}")
    return gate


def require_artefact_hash(path: Path, expected_sha256: str, name: str) -> None:
    """Fail when a frozen artefact no longer matches its recorded digest."""

    if not path.is_file():
        raise GateApprovalError(f"Required {name} is missing: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise GateApprovalError(
            f"{name} hash mismatch: expected {expected_sha256}, got {actual}"
        )
