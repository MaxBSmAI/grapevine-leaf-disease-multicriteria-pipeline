"""Validated configuration models for the DataVID manifest stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from grape_disease.utils.config import ConfigurationError, require_mapping


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration required to identify and validate DataVID source files."""

    source_dataset: str
    class_to_idx: dict[str, int]
    expected_total_images: int
    expected_originals: int
    expected_augmented: int
    expected_originals_by_class: dict[str, int]
    expected_width: int
    expected_height: int
    expected_colour_mode: str
    expected_format: str
    image_extensions: tuple[str, ...]
    variant_suffixes: tuple[str, ...]
    require_parquet: bool

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> DatasetConfig:
        """Build a validated immutable configuration from JSON-compatible data."""

        class_to_idx_raw = require_mapping(value.get("class_to_idx"), "class_to_idx")
        originals_by_class_raw = require_mapping(
            value.get("expected_originals_by_class"), "expected_originals_by_class"
        )
        class_to_idx = {str(key): int(item) for key, item in class_to_idx_raw.items()}
        originals_by_class = {
            str(key): int(item) for key, item in originals_by_class_raw.items()
        }
        if set(class_to_idx) != set(originals_by_class):
            raise ConfigurationError(
                "class_to_idx and expected_originals_by_class must contain "
                "the same classes"
            )
        if len(set(class_to_idx.values())) != len(class_to_idx):
            raise ConfigurationError("class_to_idx values must be unique")
        expected_indices = list(range(len(class_to_idx)))
        if sorted(class_to_idx.values()) != expected_indices:
            raise ConfigurationError(
                f"class_to_idx must use contiguous indices {expected_indices}"
            )
        extensions = tuple(str(item).lower() for item in value["image_extensions"])
        if any(not extension.startswith(".") for extension in extensions):
            raise ConfigurationError("Every image extension must start with a dot")
        suffixes = tuple(str(item) for item in value["variant_suffixes"])
        if not suffixes:
            raise ConfigurationError("variant_suffixes must not be empty")
        return cls(
            source_dataset=str(value["source_dataset"]),
            class_to_idx=class_to_idx,
            expected_total_images=int(value["expected_total_images"]),
            expected_originals=int(value["expected_originals"]),
            expected_augmented=int(value["expected_augmented"]),
            expected_originals_by_class=originals_by_class,
            expected_width=int(value["expected_width"]),
            expected_height=int(value["expected_height"]),
            expected_colour_mode=str(value["expected_colour_mode"]),
            expected_format=str(value["expected_format"]),
            image_extensions=extensions,
            variant_suffixes=suffixes,
            require_parquet=bool(value.get("require_parquet", True)),
        )


@dataclass(frozen=True)
class SimilarityConfig:
    """Configuration for conservative near-duplicate candidate generation."""

    dhash_max_distance: int
    phash_max_distance: int
    ssim_window_size: int
    embedding_size: int
    contact_sheet_pairs_per_page: int
    thumbnail_size: int

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> SimilarityConfig:
        """Build and validate similarity configuration."""

        config = cls(
            dhash_max_distance=int(value["dhash_max_distance"]),
            phash_max_distance=int(value["phash_max_distance"]),
            ssim_window_size=int(value.get("ssim_window_size", 11)),
            embedding_size=int(value.get("embedding_size", 32)),
            contact_sheet_pairs_per_page=int(
                value.get("contact_sheet_pairs_per_page", 12)
            ),
            thumbnail_size=int(value.get("thumbnail_size", 224)),
        )
        if config.dhash_max_distance < 0 or config.phash_max_distance < 0:
            raise ConfigurationError("Hash distance thresholds must be non-negative")
        if config.ssim_window_size < 3 or config.ssim_window_size % 2 == 0:
            raise ConfigurationError("ssim_window_size must be odd and at least 3")
        if config.embedding_size < 8:
            raise ConfigurationError("embedding_size must be at least 8")
        if config.contact_sheet_pairs_per_page <= 0 or config.thumbnail_size <= 0:
            raise ConfigurationError("Contact sheet dimensions must be positive")
        return config
