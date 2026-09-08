"""Independent validation of generated original-image manifest artefacts."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from grape_disease.data.config import DatasetConfig
from grape_disease.data.manifest import MANIFEST_FIELDS, classify_variant
from grape_disease.utils.hashing import sha256_file
from grape_disease.utils.io import ensure_output_directory, write_json

LOGGER = logging.getLogger(__name__)


def _resolve_portable_path(dataset_root: Path, relative_path: str) -> Path:
    candidate = (dataset_root / Path(relative_path)).resolve()
    root = dataset_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Relative path escapes dataset root: {relative_path}")
    return candidate


def audit_original_manifest(
    dataset_root: Path,
    manifest_dir: Path,
    output_dir: Path,
    config: DatasetConfig,
    verify_content_hashes: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Audit portability, uniqueness, counts, and optionally every content hash."""

    root = dataset_root.resolve()
    manifest_csv = manifest_dir.resolve() / "master_originals_manifest.csv"
    manifest_parquet = manifest_dir.resolve() / "master_originals_manifest.parquet"
    if not manifest_csv.is_file() or not manifest_parquet.is_file():
        raise FileNotFoundError("Both CSV and Parquet master manifests are required")
    frame = pd.read_csv(manifest_csv, dtype=str, keep_default_na=False)
    missing_columns = sorted(set(MANIFEST_FIELDS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {missing_columns}")
    if dry_run:
        return {
            "dry_run": True,
            "manifest_rows": len(frame),
            "verify_content_hashes": verify_content_hashes,
        }
    errors: list[dict[str, Any]] = []
    missing_files = 0
    hash_mismatches = 0
    variants_found = 0
    for row_number, row in frame.iterrows():
        relative_path = str(row["relative_path"])
        try:
            path = _resolve_portable_path(root, relative_path)
        except ValueError as exc:
            errors.append({"row": int(row_number), "error": str(exc)})
            continue
        if not path.is_file():
            missing_files += 1
            errors.append(
                {
                    "row": int(row_number),
                    "relative_path": relative_path,
                    "error": "missing_file",
                }
            )
            continue
        if classify_variant(path, config) is not None:
            variants_found += 1
            errors.append(
                {
                    "row": int(row_number),
                    "relative_path": relative_path,
                    "error": "historical_variant_in_original_manifest",
                }
            )
        if verify_content_hashes:
            actual_hash = sha256_file(path)
            if actual_hash != str(row["sha256"]):
                hash_mismatches += 1
                errors.append(
                    {
                        "row": int(row_number),
                        "relative_path": relative_path,
                        "error": "sha256_mismatch",
                    }
                )
    duplicate_counts = {
        "image_id": int(frame["image_id"].duplicated(keep=False).sum()),
        "original_id": int(frame["original_id"].duplicated(keep=False).sum()),
        "sha256": int(frame["sha256"].duplicated(keep=False).sum()),
    }
    class_counts = Counter(frame["class_name"].tolist())
    parquet_frame = pd.read_parquet(manifest_parquet, engine="pyarrow").fillna("")
    cross_format_rows_match = len(parquet_frame) == len(frame)
    cross_format_columns_match = list(parquet_frame.columns) == list(frame.columns)
    expected_class_counts_match = all(
        class_counts[class_name] == expected
        for class_name, expected in config.expected_originals_by_class.items()
    )
    checks = {
        "row_count_matches": len(frame) == config.expected_originals,
        "class_counts_match": expected_class_counts_match,
        "all_rows_marked_original": set(frame["is_original"].str.lower()) == {"true"},
        "no_rows_marked_augmented": set(frame["is_augmented"].str.lower()) == {"false"},
        "split_is_blank": bool((frame["split"] == "").all()),
        "split_seed_is_blank": bool((frame["split_seed"] == "").all()),
        "group_id_is_provisional": set(frame["group_id_status"])
        == {"provisional_original_id"},
        "all_validation_status_valid": set(frame["validation_status"]) == {"valid"},
        "no_missing_files": missing_files == 0,
        "no_hash_mismatches": hash_mismatches == 0,
        "no_variants": variants_found == 0,
        "unique_identifiers_and_content": all(
            value == 0 for value in duplicate_counts.values()
        ),
        "csv_parquet_row_match": cross_format_rows_match,
        "csv_parquet_column_match": cross_format_columns_match,
    }
    report = {
        "schema_version": "1.0.0",
        "manifest_csv_sha256": sha256_file(manifest_csv),
        "manifest_parquet_sha256": sha256_file(manifest_parquet),
        "rows": len(frame),
        "class_counts": dict(sorted(class_counts.items())),
        "duplicate_row_counts": duplicate_counts,
        "verify_content_hashes": verify_content_hashes,
        "checks": checks,
        "errors": errors,
        "passed": all(checks.values()) and not errors,
        "split_created": False,
        "test_accessed": False,
    }
    output = ensure_output_directory(output_dir)
    write_json(output / "originals_audit_report.json", report)
    LOGGER.info("Original manifest audit completed", extra={"passed": report["passed"]})
    return report
