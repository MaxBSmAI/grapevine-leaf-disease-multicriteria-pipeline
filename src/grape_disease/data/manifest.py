"""Build and validate a portable manifest of original DataVID images."""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from grape_disease.data.config import DatasetConfig
from grape_disease.data.fingerprints import dhash, phash
from grape_disease.utils.hashing import sha256_file
from grape_disease.utils.identifiers import extract_original_id, make_image_id
from grape_disease.utils.io import (
    ensure_output_directory,
    write_csv,
    write_json,
    write_text,
)

LOGGER = logging.getLogger(__name__)

MANIFEST_FIELDS = (
    "image_id",
    "original_id",
    "source_dataset",
    "relative_path",
    "absolute_path",
    "filename",
    "extension",
    "class_name",
    "class_id",
    "width",
    "height",
    "colour_mode",
    "image_format",
    "file_size_bytes",
    "sha256",
    "perceptual_hash",
    "dhash",
    "is_original",
    "is_augmented",
    "parent_original_id",
    "augmentation_type",
    "group_id",
    "group_id_status",
    "validation_status",
    "validation_errors",
    "exclusion_reason",
    "split",
    "split_seed",
)

EXCLUDED_FIELDS = (
    "relative_path",
    "filename",
    "class_name",
    "class_id",
    "original_id",
    "parent_original_id",
    "augmentation_type",
    "file_size_bytes",
    "sha256",
    "exclusion_reason",
    "validation_status",
)


@dataclass
class ImageRecord:
    """One original-image row in the portable master manifest."""

    image_id: str = ""
    original_id: str = ""
    source_dataset: str = ""
    relative_path: str = ""
    absolute_path: str = ""
    filename: str = ""
    extension: str = ""
    class_name: str = ""
    class_id: int | str = ""
    width: int | str = ""
    height: int | str = ""
    colour_mode: str = ""
    image_format: str = ""
    file_size_bytes: int | str = ""
    sha256: str = ""
    perceptual_hash: str = ""
    dhash: str = ""
    is_original: bool = True
    is_augmented: bool = False
    parent_original_id: str = ""
    augmentation_type: str = ""
    group_id: str = ""
    group_id_status: str = "provisional_original_id"
    validation_status: str = "valid"
    validation_errors: str = ""
    exclusion_reason: str = ""
    split: str = ""
    split_seed: str = ""


def _variant_pattern(config: DatasetConfig) -> re.Pattern[str]:
    escaped = sorted(
        (re.escape(item) for item in config.variant_suffixes), key=len, reverse=True
    )
    return re.compile(rf"_(?P<variant>{'|'.join(escaped)})$", re.IGNORECASE)


def classify_variant(path: Path, config: DatasetConfig) -> str | None:
    """Return an approved offline augmentation label or ``None`` for originals."""

    match = _variant_pattern(config).search(path.stem)
    return match.group("variant") if match else None


def discover_image_paths(dataset_root: Path, config: DatasetConfig) -> list[Path]:
    """Return deterministic image paths under approved class directories."""

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {dataset_root}")
    paths = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in config.image_extensions
    ]
    return sorted(
        paths, key=lambda item: item.relative_to(dataset_root).as_posix().lower()
    )


def _inspect_original(
    path: Path,
    dataset_root: Path,
    config: DatasetConfig,
) -> ImageRecord:
    relative_path = path.relative_to(dataset_root).as_posix()
    class_name = relative_path.split("/", maxsplit=1)[0]
    errors: list[str] = []
    class_id: int | str = ""
    if class_name not in config.class_to_idx:
        errors.append("unknown_class")
    else:
        class_id = config.class_to_idx[class_name]
    try:
        original_id = extract_original_id(path)
        image_id = make_image_id(config.source_dataset, original_id)
    except ValueError:
        original_id = ""
        image_id = ""
        errors.append("invalid_or_missing_original_uuid")
    try:
        file_size: int | str = path.stat().st_size
        content_hash = sha256_file(path)
    except OSError as exc:
        file_size = ""
        content_hash = ""
        errors.append(f"file_read_error:{type(exc).__name__}")
    width: int | str = ""
    height: int | str = ""
    mode = ""
    image_format = ""
    perceptual = ""
    difference = ""
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            image_format = str(image.format or "")
            perceptual = phash(image)
            difference = dhash(image)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        errors.append(f"decode_error:{type(exc).__name__}")
    if width and width != config.expected_width:
        errors.append(f"unexpected_width:{width}")
    if height and height != config.expected_height:
        errors.append(f"unexpected_height:{height}")
    if mode and mode != config.expected_colour_mode:
        errors.append(f"unexpected_colour_mode:{mode}")
    if image_format and image_format.upper() != config.expected_format.upper():
        errors.append(f"unexpected_format:{image_format}")
    return ImageRecord(
        image_id=image_id,
        original_id=original_id,
        source_dataset=config.source_dataset,
        relative_path=relative_path,
        filename=path.name,
        extension=path.suffix.lower(),
        class_name=class_name,
        class_id=class_id,
        width=width,
        height=height,
        colour_mode=mode,
        image_format=image_format,
        file_size_bytes=file_size,
        sha256=content_hash,
        perceptual_hash=perceptual,
        dhash=difference,
        parent_original_id="",
        group_id=original_id,
        validation_status="valid" if not errors else "invalid",
        validation_errors=";".join(errors),
    )


def _inspect_augmented(
    path: Path,
    dataset_root: Path,
    config: DatasetConfig,
    augmentation_type: str,
) -> dict[str, Any]:
    relative_path = path.relative_to(dataset_root).as_posix()
    class_name = relative_path.split("/", maxsplit=1)[0]
    errors: list[str] = []
    try:
        original_id = extract_original_id(path)
    except ValueError:
        original_id = ""
        errors.append("invalid_or_missing_original_uuid")
    try:
        file_size: int | str = path.stat().st_size
        content_hash = sha256_file(path)
    except OSError as exc:
        file_size = ""
        content_hash = ""
        errors.append(f"file_read_error:{type(exc).__name__}")
    if class_name not in config.class_to_idx:
        errors.append("unknown_class")
    return {
        "relative_path": relative_path,
        "filename": path.name,
        "class_name": class_name,
        "class_id": config.class_to_idx.get(class_name, ""),
        "original_id": original_id,
        "parent_original_id": original_id,
        "augmentation_type": augmentation_type,
        "file_size_bytes": file_size,
        "sha256": content_hash,
        "exclusion_reason": "historical_offline_augmentation",
        "validation_status": (
            "excluded_valid" if not errors else f"excluded_invalid:{';'.join(errors)}"
        ),
    }


def _mark_duplicate_identifiers(records: list[ImageRecord]) -> dict[str, int]:
    by_original_id: dict[str, list[ImageRecord]] = defaultdict(list)
    by_image_id: dict[str, list[ImageRecord]] = defaultdict(list)
    by_sha256: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.original_id:
            by_original_id[record.original_id].append(record)
        if record.image_id:
            by_image_id[record.image_id].append(record)
        if record.sha256:
            by_sha256[record.sha256].append(record)
    categories = {
        "duplicate_original_id_groups": by_original_id,
        "duplicate_image_id_groups": by_image_id,
        "exact_duplicate_sha256_groups": by_sha256,
    }
    counts: dict[str, int] = {}
    for error_name, groups in categories.items():
        duplicate_groups = [group for group in groups.values() if len(group) > 1]
        counts[error_name] = len(duplicate_groups)
        for group in duplicate_groups:
            for record in group:
                errors = [item for item in record.validation_errors.split(";") if item]
                errors.append(error_name)
                record.validation_errors = ";".join(sorted(set(errors)))
                record.validation_status = "invalid"
    return counts


def _write_parquet_atomic(records: list[dict[str, Any]], path: Path) -> None:
    if importlib.util.find_spec("pyarrow") is None:
        raise RuntimeError(
            "pyarrow is required for the mandatory Parquet manifest "
            "but is not installed"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp.parquet", dir=path.parent
        )
        os.close(descriptor)
        temporary_path = Path(raw_path)
        pd.DataFrame.from_records(records, columns=MANIFEST_FIELDS).to_parquet(
            temporary_path, index=False, engine="pyarrow"
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_master_manifest(
    dataset_root: Path,
    output_dir: Path,
    config: DatasetConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build the original-only manifest and mandatory Gate 2 audit artefacts."""

    root = dataset_root.resolve()
    paths = discover_image_paths(root, config)
    LOGGER.info("Discovered source images", extra={"count": len(paths)})
    variants: list[tuple[Path, str]] = []
    originals: list[Path] = []
    for path in paths:
        variant = classify_variant(path, config)
        if variant is None:
            originals.append(path)
        else:
            variants.append((path, variant))
    if dry_run:
        return {
            "dry_run": True,
            "discovered_images": len(paths),
            "candidate_originals": len(originals),
            "candidate_augmented": len(variants),
            "output_dir": str(output_dir.resolve()),
        }
    if config.require_parquet and importlib.util.find_spec("pyarrow") is None:
        raise RuntimeError(
            "Configuration requires Parquet output, but pyarrow is unavailable"
        )
    records = [_inspect_original(path, root, config) for path in originals]
    excluded = [
        _inspect_augmented(path, root, config, augmentation_type)
        for path, augmentation_type in variants
    ]
    duplicate_counts = _mark_duplicate_identifiers(records)
    record_dicts = [asdict(record) for record in records]
    class_original_counts = Counter(record.class_name for record in records)
    class_augmented_counts = Counter(row["class_name"] for row in excluded)
    valid_original_counts = Counter(
        record.class_name for record in records if record.validation_status == "valid"
    )
    discrepancies: list[str] = []
    if len(paths) != config.expected_total_images:
        discrepancies.append(
            f"total_images:{len(paths)}!=expected:{config.expected_total_images}"
        )
    if len(records) != config.expected_originals:
        discrepancies.append(
            f"originals:{len(records)}!=expected:{config.expected_originals}"
        )
    if len(excluded) != config.expected_augmented:
        discrepancies.append(
            f"augmented:{len(excluded)}!=expected:{config.expected_augmented}"
        )
    for class_name, expected in config.expected_originals_by_class.items():
        actual = class_original_counts[class_name]
        if actual != expected:
            discrepancies.append(
                f"class_originals:{class_name}:{actual}!=expected:{expected}"
            )
    invalid_originals = [
        record for record in records if record.validation_status != "valid"
    ]
    invalid_excluded = [
        row for row in excluded if row["validation_status"] != "excluded_valid"
    ]
    if invalid_originals:
        discrepancies.append(f"invalid_originals:{len(invalid_originals)}")
    if invalid_excluded:
        discrepancies.append(f"invalid_augmented_records:{len(invalid_excluded)}")
    for name, count in duplicate_counts.items():
        if count:
            discrepancies.append(f"{name}:{count}")
    output = ensure_output_directory(output_dir)
    manifest_csv = output / "master_originals_manifest.csv"
    manifest_parquet = output / "master_originals_manifest.parquet"
    excluded_csv = output / "excluded_augmented_files.csv"
    class_summary_csv = output / "dataset_class_summary.csv"
    write_csv(manifest_csv, record_dicts, MANIFEST_FIELDS)
    if config.require_parquet:
        _write_parquet_atomic(record_dicts, manifest_parquet)
    write_csv(excluded_csv, excluded, EXCLUDED_FIELDS)
    class_summary_rows = [
        {
            "class_name": class_name,
            "class_id": config.class_to_idx[class_name],
            "originals": class_original_counts[class_name],
            "valid_originals": valid_original_counts[class_name],
            "excluded_augmented": class_augmented_counts[class_name],
        }
        for class_name in sorted(
            config.class_to_idx, key=lambda name: config.class_to_idx[name]
        )
    ]
    write_csv(
        class_summary_csv,
        class_summary_rows,
        (
            "class_name",
            "class_id",
            "originals",
            "valid_originals",
            "excluded_augmented",
        ),
    )
    manifest_hash = sha256_file(manifest_csv)
    parquet_hash = sha256_file(manifest_parquet) if manifest_parquet.exists() else None
    write_text(output / "manifest_sha256.txt", manifest_hash)
    report = {
        "schema_version": "1.0.0",
        "source_dataset": config.source_dataset,
        "dataset_root_local": str(root),
        "discovered_images": len(paths),
        "originals": len(records),
        "valid_originals": len(records) - len(invalid_originals),
        "excluded_augmented": len(excluded),
        "invalid_augmented_records": len(invalid_excluded),
        "class_original_counts": dict(sorted(class_original_counts.items())),
        "class_augmented_counts": dict(sorted(class_augmented_counts.items())),
        "duplicate_checks": duplicate_counts,
        "discrepancies": discrepancies,
        "passed": not discrepancies,
        "manifest_sha256": manifest_hash,
        "manifest_parquet_sha256": parquet_hash,
        "group_id_status": "provisional_original_id_pending_similarity_review",
        "split_created": False,
        "test_accessed": False,
    }
    write_json(output / "dataset_validation_report.json", report)
    LOGGER.info(
        "Master manifest completed",
        extra={"passed": report["passed"], "originals": len(records)},
    )
    return report
