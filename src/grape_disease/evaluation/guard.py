"""Complete frozen-experiment and test-access verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grape_disease.utils.hashing import sha256_file
from grape_disease.utils.test_lock import LockedTestAccessError, require_test_access


def require_frozen_test_evaluation(
    test_lock_path: Path,
    frozen_experiment_path: Path,
    manifest_path: Path,
    split_path: Path,
    protocol_path: Path,
    hypotheses_path: Path,
    final_config_path: Path,
) -> dict[str, Any]:
    """Require all nine frozen-test preconditions with matching hashes."""

    lock = require_test_access(test_lock_path)
    try:
        frozen = json.loads(frozen_experiment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LockedTestAccessError(f"Cannot load frozen experiment: {exc}") from exc
    if not isinstance(frozen, dict) or frozen.get("status") != "frozen":
        raise LockedTestAccessError("Experiment is not frozen")
    actual = {
        "manifest_sha256": sha256_file(manifest_path),
        "split_sha256": sha256_file(split_path),
        "frozen_protocol_sha256": sha256_file(protocol_path),
        "hypotheses_sha256": sha256_file(hypotheses_path),
        "final_config_sha256": sha256_file(final_config_path),
        "frozen_experiment_sha256": sha256_file(frozen_experiment_path),
    }
    mismatches = {
        name: {"lock": lock.get(name), "actual": digest}
        for name, digest in actual.items()
        if str(lock.get(name, "")).lower() != digest.lower()
    }
    if mismatches:
        raise LockedTestAccessError(f"Frozen test hash mismatch: {mismatches}")
    if lock.get("explicit_evaluation_authorisation") is not True:
        raise LockedTestAccessError("Explicit test evaluation authorisation is absent")
    if frozen.get("code_commit") != lock.get("code_commit"):
        raise LockedTestAccessError("Code commit differs between lock and experiment")
    return {"lock": lock, "frozen_experiment": frozen, "verified_hashes": actual}
