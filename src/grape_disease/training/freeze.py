"""Immutable experiment manifest and explicit test-lock authorisation records."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grape_disease.utils.hashing import sha256_canonical_json, sha256_file
from grape_disease.utils.io import ensure_output_directory, write_json


def verify_frozen_inputs(
    frozen_experiment_path: Path, inputs: dict[str, Path]
) -> dict[str, Any]:
    """Fail if any current execution input differs from the frozen record."""

    try:
        frozen = json.loads(frozen_experiment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load frozen experiment: {exc}") from exc
    if not isinstance(frozen, dict) or frozen.get("status") != "frozen":
        raise ValueError("Experiment record is not frozen")
    mismatches: dict[str, dict[str, str]] = {}
    for name, path in inputs.items():
        expected = str(frozen.get(f"{name}_sha256", ""))
        actual = sha256_file(path)
        if expected.lower() != actual.lower():
            mismatches[name] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ValueError(f"Frozen execution input mismatch: {mismatches}")
    return frozen


def freeze_experiment(
    output_dir: Path,
    code_commit: str,
    protocol_path: Path,
    manifest_path: Path,
    split_path: Path,
    normalisation_path: Path,
    selected_hyperparameters_path: Path,
    focal_gamma_path: Path,
    hypotheses_path: Path,
    xai_config_path: Path,
    profiling_config_path: Path,
    statistics_config_path: Path,
    confirmatory_seeds: list[int],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Hash every frozen input and write one non-overwriting experiment record."""

    paths = {
        "protocol": protocol_path,
        "manifest": manifest_path,
        "split": split_path,
        "normalisation": normalisation_path,
        "selected_hyperparameters": selected_hyperparameters_path,
        "focal_gamma": focal_gamma_path,
        "hypotheses": hypotheses_path,
        "xai_config": xai_config_path,
        "profiling_config": profiling_config_path,
        "statistics_config": statistics_config_path,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze experiment; missing: {missing}")
    hashes = {f"{name}_sha256": sha256_file(path) for name, path in paths.items()}
    final_config = {
        "schema_version": "1.0.0",
        "selected_hyperparameters_sha256": hashes["selected_hyperparameters_sha256"],
        "focal_gamma_sha256": hashes["focal_gamma_sha256"],
        "normalisation_sha256": hashes["normalisation_sha256"],
        "xai_config_sha256": hashes["xai_config_sha256"],
        "profiling_config_sha256": hashes["profiling_config_sha256"],
        "statistics_config_sha256": hashes["statistics_config_sha256"],
        "confirmatory_seeds": confirmatory_seeds,
    }
    final_config_sha256 = sha256_canonical_json(final_config)
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "frozen",
        "frozen_at": datetime.now(UTC).isoformat(),
        "code_commit": code_commit,
        "confirmatory_seeds": confirmatory_seeds,
        "final_config_sha256": final_config_sha256,
        **hashes,
        "test_results_present": False,
    }
    if dry_run:
        return {**record, "dry_run": True, "written": False}
    output = ensure_output_directory(output_dir)
    destination = output / "FROZEN_EXPERIMENT.json"
    final_config_path = output / "FINAL_CONFIG.json"
    if destination.exists() or final_config_path.exists():
        raise FileExistsError(f"Frozen experiment artefacts already exist in: {output}")
    write_json(final_config_path, final_config)
    final_config_sha256 = sha256_file(final_config_path)
    record["final_config_sha256"] = final_config_sha256
    write_json(destination, record)
    record["frozen_experiment_sha256"] = sha256_file(destination)
    return record


def build_test_lock_record(
    frozen_experiment_path: Path,
    authorised_by: str,
    authorisation_date: str,
    explicit_authorisation: bool,
) -> dict[str, Any]:
    """Construct the only valid unlocked record from a frozen experiment."""

    frozen = json.loads(frozen_experiment_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen":
        raise ValueError("Experiment must be frozen before test authorisation")
    if not authorised_by or not authorisation_date or not explicit_authorisation:
        raise ValueError("Test authorisation must be explicit and attributable")
    return {
        "schema_version": "1.0.0",
        "test_access": True,
        "reason": "Explicit Gate 9 authorisation after experiment freeze",
        "authorised_by": authorised_by,
        "authorisation_date": authorisation_date,
        "explicit_evaluation_authorisation": True,
        "code_commit": frozen["code_commit"],
        "frozen_protocol_sha256": frozen["protocol_sha256"],
        "manifest_sha256": frozen["manifest_sha256"],
        "split_sha256": frozen["split_sha256"],
        "hypotheses_sha256": frozen["hypotheses_sha256"],
        "final_config_sha256": frozen["final_config_sha256"],
        "frozen_experiment_sha256": sha256_file(frozen_experiment_path),
    }
