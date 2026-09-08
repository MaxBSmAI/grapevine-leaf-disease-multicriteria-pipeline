"""Reproducible prioritisation of similarity candidates for human review."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from grape_disease.utils.hashing import sha256_file
from grape_disease.utils.io import ensure_output_directory, write_csv, write_json

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "candidate_pair_id",
    "image_id_a",
    "image_id_b",
    "class_name_a",
    "class_name_b",
    "same_class",
    "dhash_distance",
    "phash_distance",
    "triggered_by",
    "ssim_uniform_window",
    "rgb_embedding_cosine_similarity",
)
OUTPUT_FIELDS = (
    "review_rank",
    "contact_sheet_page",
    "candidate_pair_id",
    "image_id_a",
    "image_id_b",
    "class_name_a",
    "class_name_b",
    "same_class",
    "priority",
    "priority_reason_flags",
    "dhash_distance",
    "phash_distance",
    "ssim_uniform_window",
    "rgb_embedding_cosine_similarity",
    "ai_assessment",
    "required_action",
)


@dataclass(frozen=True)
class ReviewPriorityConfig:
    """Conservative thresholds used only to order, never decide, reviews."""

    contact_sheet_pairs_per_page: int
    close_hash_distance: int
    high_ssim_threshold: float
    high_rgb_cosine_threshold: float

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ReviewPriorityConfig:
        config = cls(
            contact_sheet_pairs_per_page=int(value["contact_sheet_pairs_per_page"]),
            close_hash_distance=int(value["close_hash_distance"]),
            high_ssim_threshold=float(value["high_ssim_threshold"]),
            high_rgb_cosine_threshold=float(value["high_rgb_cosine_threshold"]),
        )
        if config.contact_sheet_pairs_per_page <= 0:
            raise ValueError("contact_sheet_pairs_per_page must be positive")
        if config.close_hash_distance < 0:
            raise ValueError("close_hash_distance must be non-negative")
        if not 0.0 <= config.high_ssim_threshold <= 1.0:
            raise ValueError("high_ssim_threshold must be between zero and one")
        if not -1.0 <= config.high_rgb_cosine_threshold <= 1.0:
            raise ValueError(
                "high_rgb_cosine_threshold must be between minus one and one"
            )
        return config


def prioritise_similarity_review(
    metrics_path: Path,
    output_dir: Path,
    config: ReviewPriorityConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create a risk-ordered queue without assigning scientific decisions."""

    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    frame = pd.read_csv(metrics_path, dtype=str, keep_default_na=False)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Similarity metrics missing columns: {missing_columns}")
    if frame["candidate_pair_id"].duplicated().any():
        raise ValueError("candidate_pair_id must be unique")

    work = frame.copy()
    work["_source_order"] = range(len(work))
    work["_dhash"] = pd.to_numeric(work["dhash_distance"], errors="raise")
    work["_phash"] = pd.to_numeric(work["phash_distance"], errors="raise")
    work["_ssim"] = pd.to_numeric(work["ssim_uniform_window"], errors="raise")
    work["_rgb"] = pd.to_numeric(
        work["rgb_embedding_cosine_similarity"], errors="raise"
    )
    work["_same_class"] = work["same_class"].str.lower() == "true"

    priority_values: list[str] = []
    priority_tiers: list[int] = []
    reason_values: list[str] = []
    for row in work.to_dict("records"):
        flags: list[str] = []
        if not row["_same_class"]:
            flags.append("cross_class_label_conflict_risk")
        if (
            row["_dhash"] <= config.close_hash_distance
            and row["_phash"] <= config.close_hash_distance
        ):
            flags.append("close_on_both_perceptual_hashes")
        if row["_ssim"] >= config.high_ssim_threshold:
            flags.append("higher_ssim_within_candidate_set")
        if row["_rgb"] >= config.high_rgb_cosine_threshold:
            flags.append("higher_rgb_cosine_within_candidate_set")
        if not flags:
            flags.append("hash_threshold_candidate")

        if not row["_same_class"]:
            priority, tier = "critical", 1
        elif len(flags) > 1 or flags[0] != "hash_threshold_candidate":
            priority, tier = "high", 2
        else:
            priority, tier = "standard", 3
        priority_values.append(priority)
        priority_tiers.append(tier)
        reason_values.append(";".join(flags))

    work["priority"] = priority_values
    work["_priority_tier"] = priority_tiers
    work["priority_reason_flags"] = reason_values
    work = work.sort_values(
        [
            "_priority_tier",
            "_rgb",
            "_ssim",
            "_dhash",
            "_phash",
            "candidate_pair_id",
        ],
        ascending=[True, False, False, True, True, True],
        kind="mergesort",
    )

    output_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(work.to_dict("records"), start=1):
        source_order = int(row["_source_order"])
        output_rows.append(
            {
                "review_rank": rank,
                "contact_sheet_page": (
                    source_order // config.contact_sheet_pairs_per_page + 1
                ),
                "candidate_pair_id": row["candidate_pair_id"],
                "image_id_a": row["image_id_a"],
                "image_id_b": row["image_id_b"],
                "class_name_a": row["class_name_a"],
                "class_name_b": row["class_name_b"],
                "same_class": row["same_class"],
                "priority": row["priority"],
                "priority_reason_flags": row["priority_reason_flags"],
                "dhash_distance": row["dhash_distance"],
                "phash_distance": row["phash_distance"],
                "ssim_uniform_window": row["ssim_uniform_window"],
                "rgb_embedding_cosine_similarity": row[
                    "rgb_embedding_cosine_similarity"
                ],
                "ai_assessment": "no_automatic_relationship_conclusion",
                "required_action": "independent_human_review",
            }
        )

    priority_counts = {
        name: sum(row["priority"] == name for row in output_rows)
        for name in ("critical", "high", "standard")
    }
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "metrics_sha256": sha256_file(metrics_path),
        "candidate_count": len(output_rows),
        "priority_counts": priority_counts,
        "scientific_decisions_assigned": 0,
        "human_confirmation_required": True,
        "split_created": False,
        "test_accessed": False,
        "dry_run": dry_run,
    }
    if not dry_run:
        output = ensure_output_directory(output_dir)
        write_csv(
            output / "ai_assisted_review_priorities.csv",
            output_rows,
            OUTPUT_FIELDS,
        )
        write_json(output / "ai_assisted_review_report.json", report)
    LOGGER.info(
        "Similarity review queue prioritised",
        extra={"candidates": len(output_rows), "priority_counts": priority_counts},
    )
    return report
