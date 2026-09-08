"""Fail-closed validation and deterministic consolidation of review groups."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from grape_disease.utils.hashing import sha256_file
from grape_disease.utils.identifiers import make_group_id
from grape_disease.utils.io import ensure_output_directory, write_csv, write_json

LOGGER = logging.getLogger(__name__)

MANIFEST_REQUIRED_COLUMNS = (
    "image_id",
    "original_id",
    "class_name",
    "is_original",
    "split",
)
CANDIDATE_REQUIRED_COLUMNS = (
    "candidate_pair_id",
    "image_id_a",
    "image_id_b",
    "class_name_a",
    "class_name_b",
)
DECISION_REQUIRED_COLUMNS = (
    "candidate_pair_id",
    "image_id_a",
    "image_id_b",
    "decision",
    "reason",
    "reviewer",
    "date",
    "evidence",
    "group_id_assigned",
)
GROUP_OUTPUT_FIELDS = (
    "image_id",
    "original_id",
    "class_name",
    "group_id",
    "group_size",
    "group_id_status",
    "review_protocol_version",
)


@dataclass(frozen=True)
class GroupReviewConfig:
    """Policy controlling which human decisions may create a group edge."""

    protocol_version: str
    allowed_decisions: frozenset[str]
    grouping_decisions: frozenset[str]
    blocking_decisions: frozenset[str]
    required_metadata_fields: tuple[str, ...]
    require_iso_date: bool
    block_multiclass_groups: bool

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> GroupReviewConfig:
        """Create a validated immutable review policy."""

        config = cls(
            protocol_version=str(value["protocol_version"]),
            allowed_decisions=frozenset(
                str(item).strip() for item in value["allowed_decisions"]
            ),
            grouping_decisions=frozenset(
                str(item).strip() for item in value["grouping_decisions"]
            ),
            blocking_decisions=frozenset(
                str(item).strip() for item in value["blocking_decisions"]
            ),
            required_metadata_fields=tuple(
                str(item).strip() for item in value["required_metadata_fields"]
            ),
            require_iso_date=bool(value.get("require_iso_date", True)),
            block_multiclass_groups=bool(value.get("block_multiclass_groups", True)),
        )
        if not config.protocol_version:
            raise ValueError("protocol_version must not be blank")
        if not config.allowed_decisions:
            raise ValueError("allowed_decisions must not be empty")
        if not config.grouping_decisions <= config.allowed_decisions:
            raise ValueError("grouping_decisions must be allowed decisions")
        if not config.blocking_decisions <= config.allowed_decisions:
            raise ValueError("blocking_decisions must be allowed decisions")
        unknown_metadata = set(config.required_metadata_fields) - set(
            DECISION_REQUIRED_COLUMNS
        )
        if unknown_metadata:
            raise ValueError(
                f"Unknown required metadata fields: {sorted(unknown_metadata)}"
            )
        return config


class _DisjointSet:
    """Minimal deterministic disjoint-set data structure."""

    def __init__(self, values: list[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self._parent[second] = first

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for value in sorted(self._parent):
            groups.setdefault(self.find(value), []).append(value)
        return groups


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _missing_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    return sorted(set(required) - set(frame.columns))


def _is_iso_date(value: str) -> bool:
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _base_report(
    manifest_path: Path,
    candidates_path: Path,
    decisions_path: Path,
    config: GroupReviewConfig,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "review_protocol_version": config.protocol_version,
        "dry_run": dry_run,
        "input_hashes": {
            "manifest_sha256": sha256_file(manifest_path),
            "candidates_sha256": sha256_file(candidates_path),
            "decisions_sha256": sha256_file(decisions_path),
        },
        "checks": {},
        "blocking_issues": [],
        "counts": {},
        "passed": False,
        "confirmed_groups_written": False,
        "split_created": False,
        "test_accessed": False,
    }


def _write_report_if_requested(
    output_dir: Path, report: dict[str, Any], dry_run: bool
) -> None:
    if not dry_run:
        output = ensure_output_directory(output_dir)
        write_json(output / "group_validation_report.json", report)


def validate_and_consolidate_groups(
    manifest_path: Path,
    candidates_path: Path,
    decisions_path: Path,
    output_dir: Path,
    config: GroupReviewConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate all review decisions and write groups only if every check passes."""

    manifest = _read_csv(manifest_path)
    candidates = _read_csv(candidates_path)
    decisions = _read_csv(decisions_path)
    report = _base_report(
        manifest_path, candidates_path, decisions_path, config, dry_run
    )
    checks: dict[str, bool] = report["checks"]
    issues: list[dict[str, Any]] = report["blocking_issues"]

    missing_schema_columns = {
        "manifest": _missing_columns(manifest, MANIFEST_REQUIRED_COLUMNS),
        "candidates": _missing_columns(candidates, CANDIDATE_REQUIRED_COLUMNS),
        "decisions": _missing_columns(decisions, DECISION_REQUIRED_COLUMNS),
    }
    checks["schemas_valid"] = not any(missing_schema_columns.values())
    if not checks["schemas_valid"]:
        issues.append({"type": "missing_columns", "details": missing_schema_columns})
        report["counts"] = {
            "manifest_rows": len(manifest),
            "candidate_rows": len(candidates),
            "decision_rows": len(decisions),
        }
        _write_report_if_requested(output_dir, report, dry_run)
        return report

    manifest_image_duplicates = int(manifest["image_id"].duplicated().sum())
    manifest_original_duplicates = int(manifest["original_id"].duplicated().sum())
    candidate_duplicates = int(candidates["candidate_pair_id"].duplicated().sum())
    decision_duplicates = int(decisions["candidate_pair_id"].duplicated().sum())
    checks["manifest_and_review_ids_unique"] = (
        manifest_image_duplicates == 0
        and manifest_original_duplicates == 0
        and candidate_duplicates == 0
        and decision_duplicates == 0
    )
    if not checks["manifest_and_review_ids_unique"]:
        issues.append(
            {
                "type": "duplicate_identifiers",
                "manifest_image_duplicates": manifest_image_duplicates,
                "manifest_original_duplicates": manifest_original_duplicates,
                "candidate_duplicates": candidate_duplicates,
                "decision_duplicates": decision_duplicates,
            }
        )

    candidate_ids = set(candidates["candidate_pair_id"])
    decision_ids = set(decisions["candidate_pair_id"])
    missing_decisions = sorted(candidate_ids - decision_ids)
    extra_decisions = sorted(decision_ids - candidate_ids)
    checks["decision_coverage_complete"] = not missing_decisions and not extra_decisions
    if not checks["decision_coverage_complete"]:
        issues.append(
            {
                "type": "decision_coverage_mismatch",
                "missing_candidate_pair_ids": missing_decisions,
                "extra_candidate_pair_ids": extra_decisions,
            }
        )

    decisions = decisions.copy()
    decisions["decision"] = decisions["decision"].str.strip().str.lower()
    invalid_decision_rows = decisions.loc[
        ~decisions["decision"].isin(config.allowed_decisions),
        ["candidate_pair_id", "decision"],
    ].to_dict("records")
    checks["decisions_allowed_and_complete"] = not invalid_decision_rows
    if invalid_decision_rows:
        issues.append(
            {
                "type": "blank_or_invalid_decisions",
                "rows": invalid_decision_rows,
            }
        )

    blocking_rows = decisions.loc[
        decisions["decision"].isin(config.blocking_decisions),
        ["candidate_pair_id", "decision"],
    ].to_dict("records")
    checks["no_blocking_decisions"] = not blocking_rows
    if blocking_rows:
        issues.append({"type": "blocking_decisions", "rows": blocking_rows})

    missing_metadata: dict[str, list[str]] = {}
    for field in config.required_metadata_fields:
        row_ids = decisions.loc[
            decisions[field].str.strip() == "", "candidate_pair_id"
        ].tolist()
        if row_ids:
            missing_metadata[field] = sorted(row_ids)
    checks["review_metadata_complete"] = not missing_metadata
    if missing_metadata:
        issues.append({"type": "missing_review_metadata", "fields": missing_metadata})

    invalid_dates: list[str] = []
    if config.require_iso_date:
        invalid_dates = decisions.loc[
            ~decisions["date"].map(_is_iso_date), "candidate_pair_id"
        ].tolist()
    checks["review_dates_iso_8601"] = not invalid_dates
    if invalid_dates:
        issues.append(
            {"type": "invalid_review_dates", "candidate_pair_ids": invalid_dates}
        )

    candidate_by_id = candidates.drop_duplicates(
        "candidate_pair_id", keep="first"
    ).set_index("candidate_pair_id", drop=False)
    pair_mismatches: list[str] = []
    for row in decisions.to_dict("records"):
        candidate_id = row["candidate_pair_id"]
        if candidate_id not in candidate_by_id.index:
            continue
        expected = candidate_by_id.loc[candidate_id]
        if (
            row["image_id_a"] != expected["image_id_a"]
            or row["image_id_b"] != expected["image_id_b"]
        ):
            pair_mismatches.append(candidate_id)
    checks["decision_pair_identity_matches"] = not pair_mismatches
    if pair_mismatches:
        issues.append(
            {
                "type": "decision_pair_identity_mismatch",
                "candidate_pair_ids": sorted(pair_mismatches),
            }
        )

    manifest_image_ids = set(manifest["image_id"])
    candidate_image_ids = set(candidates["image_id_a"]) | set(candidates["image_id_b"])
    unknown_images = sorted(candidate_image_ids - manifest_image_ids)
    checks["candidate_images_exist_in_manifest"] = not unknown_images
    if unknown_images:
        issues.append({"type": "unknown_candidate_images", "image_ids": unknown_images})

    manifest_by_image = manifest.drop_duplicates("image_id", keep="first").set_index(
        "image_id", drop=False
    )
    label_mismatches: list[str] = []
    for row in candidates.to_dict("records"):
        left = row["image_id_a"]
        right = row["image_id_b"]
        if left not in manifest_by_image.index or right not in manifest_by_image.index:
            continue
        if (
            row["class_name_a"] != manifest_by_image.loc[left]["class_name"]
            or row["class_name_b"] != manifest_by_image.loc[right]["class_name"]
        ):
            label_mismatches.append(row["candidate_pair_id"])
    checks["candidate_labels_match_manifest"] = not label_mismatches
    if label_mismatches:
        issues.append(
            {
                "type": "candidate_label_mismatch",
                "candidate_pair_ids": sorted(label_mismatches),
            }
        )

    originals_only = set(manifest["is_original"].str.lower()) == {"true"}
    split_blank = bool((manifest["split"].str.strip() == "").all())
    checks["manifest_contains_only_originals"] = originals_only
    checks["manifest_split_remains_blank"] = split_blank
    if not originals_only:
        issues.append({"type": "manifest_contains_non_original_rows"})
    if not split_blank:
        issues.append({"type": "manifest_split_already_populated"})

    disjoint_set = _DisjointSet(manifest["image_id"].tolist())
    grouping_rows = decisions.loc[
        decisions["decision"].isin(config.grouping_decisions)
        & decisions["candidate_pair_id"].isin(candidate_ids)
    ]
    for row in grouping_rows.to_dict("records"):
        left = row["image_id_a"]
        right = row["image_id_b"]
        if left in manifest_image_ids and right in manifest_image_ids:
            disjoint_set.union(left, right)

    components = disjoint_set.components()
    image_to_group: dict[str, str] = {}
    multiclass_groups: list[dict[str, Any]] = []
    group_sizes: dict[str, int] = {}
    for image_ids in components.values():
        original_ids = [
            str(manifest_by_image.loc[item]["original_id"]) for item in image_ids
        ]
        group_id = make_group_id(original_ids)
        group_sizes[group_id] = len(image_ids)
        for image_id in image_ids:
            image_to_group[image_id] = group_id
        classes = sorted(
            {str(manifest_by_image.loc[item]["class_name"]) for item in image_ids}
        )
        if len(classes) > 1:
            multiclass_groups.append(
                {"group_id": group_id, "image_ids": image_ids, "classes": classes}
            )
    checks["no_multiclass_groups"] = (
        not config.block_multiclass_groups or not multiclass_groups
    )
    if not checks["no_multiclass_groups"]:
        issues.append({"type": "multiclass_groups", "groups": multiclass_groups})

    non_grouping_with_assignment = decisions.loc[
        ~decisions["decision"].isin(config.grouping_decisions)
        & (decisions["group_id_assigned"].str.strip() != ""),
        "candidate_pair_id",
    ].tolist()
    checks["non_grouping_assignments_blank"] = not non_grouping_with_assignment
    if non_grouping_with_assignment:
        issues.append(
            {
                "type": "group_id_assigned_to_non_grouping_decision",
                "candidate_pair_ids": sorted(non_grouping_with_assignment),
            }
        )

    assigned_id_mismatches: list[str] = []
    for row in grouping_rows.to_dict("records"):
        assigned = row["group_id_assigned"].strip()
        if assigned and assigned != image_to_group.get(row["image_id_a"], ""):
            assigned_id_mismatches.append(row["candidate_pair_id"])
    checks["provided_group_ids_match_deterministic_ids"] = not assigned_id_mismatches
    if assigned_id_mismatches:
        issues.append(
            {
                "type": "provided_group_id_mismatch",
                "candidate_pair_ids": sorted(assigned_id_mismatches),
            }
        )

    checks["all_validation_checks_pass"] = all(
        value for key, value in checks.items() if key != "all_validation_checks_pass"
    )
    report["counts"] = {
        "manifest_rows": len(manifest),
        "candidate_rows": len(candidates),
        "decision_rows": len(decisions),
        "blank_or_invalid_decisions": len(invalid_decision_rows),
        "blocking_decisions": len(blocking_rows),
        "grouping_edges": len(grouping_rows),
        "deterministic_groups": len(components),
        "multi_image_groups": sum(size > 1 for size in group_sizes.values()),
        "multiclass_groups": len(multiclass_groups),
    }
    report["passed"] = bool(checks["all_validation_checks_pass"] and not issues)

    if report["passed"] and not dry_run:
        group_rows: list[dict[str, Any]] = []
        for row in manifest.sort_values(["original_id", "image_id"]).to_dict("records"):
            group_id = image_to_group[row["image_id"]]
            group_size = group_sizes[group_id]
            group_rows.append(
                {
                    "image_id": row["image_id"],
                    "original_id": row["original_id"],
                    "class_name": row["class_name"],
                    "group_id": group_id,
                    "group_size": group_size,
                    "group_id_status": (
                        "review_confirmed_component"
                        if group_size > 1
                        else "singleton_original_id"
                    ),
                    "review_protocol_version": config.protocol_version,
                }
            )
        output = ensure_output_directory(output_dir)
        write_csv(output / "confirmed_groups.csv", group_rows, GROUP_OUTPUT_FIELDS)
        report["confirmed_groups_written"] = True

    _write_report_if_requested(output_dir, report, dry_run)
    LOGGER.info(
        "Group review validation completed",
        extra={"passed": report["passed"], "issues": len(issues)},
    )
    return report
