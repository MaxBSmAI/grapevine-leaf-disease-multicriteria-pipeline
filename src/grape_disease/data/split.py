"""Deterministic grouped-stratified canonical split construction and audit."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from grape_disease.utils.hashing import sha256_file
from grape_disease.utils.io import (
    ensure_output_directory,
    write_csv,
    write_json,
    write_text,
)

LOGGER = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_OUTPUT_FIELDS = (
    "image_id",
    "original_id",
    "group_id",
    "class_name",
    "class_id",
    "relative_path",
    "split",
    "split_seed",
)


@dataclass(frozen=True)
class SplitConfig:
    """Frozen split objective plus pre-approved implementation parameters."""

    seed: int
    proportions: dict[str, float]
    preferred_class_tolerance_pp: float
    maximum_class_tolerance_pp: float
    overall_tolerance_pp: float
    candidate_restarts: int
    total_cost_weight: float
    class_cost_weight: float
    violation_cost_weight: float
    approval_status: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> SplitConfig:
        proportions_raw = value.get("proportions")
        if not isinstance(proportions_raw, dict):
            raise ValueError("proportions must be an object")
        proportions = {name: float(proportions_raw[name]) for name in SPLIT_NAMES}
        config = cls(
            seed=int(value["seed"]),
            proportions=proportions,
            preferred_class_tolerance_pp=float(value["preferred_class_tolerance_pp"]),
            maximum_class_tolerance_pp=float(value["maximum_class_tolerance_pp"]),
            overall_tolerance_pp=float(value["overall_tolerance_pp"]),
            candidate_restarts=int(value["candidate_restarts"]),
            total_cost_weight=float(value.get("total_cost_weight", 1.0)),
            class_cost_weight=float(value.get("class_cost_weight", 2.0)),
            violation_cost_weight=float(value.get("violation_cost_weight", 10.0)),
            approval_status=str(value.get("approval_status", "pending")),
        )
        if abs(sum(proportions.values()) - 1.0) > 1e-12:
            raise ValueError("Split proportions must sum to one")
        if any(fraction <= 0.0 for fraction in proportions.values()):
            raise ValueError("Every split proportion must be positive")
        if config.candidate_restarts <= 0:
            raise ValueError("candidate_restarts must be positive")
        if config.maximum_class_tolerance_pp < config.preferred_class_tolerance_pp:
            raise ValueError("Maximum class tolerance cannot be below preferred")
        return config


def _prepare_group_table(
    manifest: pd.DataFrame, groups: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_manifest = {
        "image_id",
        "original_id",
        "class_name",
        "class_id",
        "relative_path",
        "is_original",
    }
    required_groups = {"image_id", "original_id", "group_id", "class_name"}
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    missing_groups = sorted(required_groups - set(groups.columns))
    if missing_manifest or missing_groups:
        raise ValueError(
            f"Missing columns: manifest={missing_manifest}, groups={missing_groups}"
        )
    if set(manifest["is_original"].str.lower()) != {"true"}:
        raise ValueError("Canonical split input must contain originals only")
    if manifest["image_id"].duplicated().any() or groups["image_id"].duplicated().any():
        raise ValueError("image_id must be unique in manifest and groups")
    if set(manifest["image_id"]) != set(groups["image_id"]):
        raise ValueError("confirmed_groups must cover exactly the master manifest")
    reviewed_groups = groups[["image_id", "group_id", "class_name"]].rename(
        columns={
            "group_id": "confirmed_group_id",
            "class_name": "class_name_review",
        }
    )
    merged = manifest.merge(
        reviewed_groups,
        on="image_id",
        how="inner",
        validate="one_to_one",
    )
    if not (merged["class_name"] == merged["class_name_review"]).all():
        raise ValueError("Group review labels do not match the master manifest")
    if "group_id" in merged.columns:
        merged = merged.drop(columns="group_id")
    merged = merged.rename(columns={"confirmed_group_id": "group_id"})
    group_classes = merged.groupby("group_id")["class_name"].nunique()
    multiclass = group_classes[group_classes > 1]
    if not multiclass.empty:
        raise ValueError(f"Unresolved multiclass groups: {multiclass.index.tolist()}")
    group_table = (
        merged.groupby("group_id", as_index=False)
        .agg(
            class_name=("class_name", "first"),
            class_id=("class_id", "first"),
            size=("image_id", "size"),
        )
        .sort_values("group_id")
        .reset_index(drop=True)
    )
    return merged, group_table


def _deviations(
    assignments: npt.NDArray[Any],
    group_sizes: npt.NDArray[Any],
    class_codes: npt.NDArray[Any],
    proportions: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    split_count = len(proportions)
    total = float(group_sizes.sum())
    totals = np.array(
        [
            group_sizes[assignments == split_index].sum()
            for split_index in range(split_count)
        ],
        dtype=np.float64,
    )
    overall_pp = np.abs(totals / total - proportions) * 100.0
    classes = np.unique(class_codes)
    class_pp = np.zeros((len(classes), split_count), dtype=np.float64)
    for row_index, class_code in enumerate(classes):
        mask = class_codes == class_code
        class_total = float(group_sizes[mask].sum())
        for split_index in range(split_count):
            count = group_sizes[mask & (assignments == split_index)].sum()
            class_pp[row_index, split_index] = (
                abs(count / class_total - proportions[split_index]) * 100.0
            )
    return overall_pp, class_pp


def _candidate_cost(
    assignments: npt.NDArray[Any],
    group_sizes: npt.NDArray[Any],
    class_codes: npt.NDArray[Any],
    config: SplitConfig,
) -> tuple[
    int,
    int,
    float,
    npt.NDArray[Any],
    npt.NDArray[Any],
]:
    proportions = np.array([config.proportions[name] for name in SPLIT_NAMES])
    overall_pp, class_pp = _deviations(
        assignments, group_sizes, class_codes, proportions
    )
    missing = 0
    for class_code in np.unique(class_codes):
        for split_index in range(len(SPLIT_NAMES)):
            missing += int(
                not np.any((class_codes == class_code) & (assignments == split_index))
            )
    violations = (
        int(np.sum(overall_pp > config.overall_tolerance_pp))
        + int(np.sum(class_pp > config.maximum_class_tolerance_pp))
        + missing
    )
    preferred_exceedances = int(np.sum(class_pp > config.preferred_class_tolerance_pp))
    cost = (
        config.total_cost_weight * float(overall_pp.sum())
        + config.class_cost_weight * float(class_pp.sum())
        + config.violation_cost_weight * violations
    )
    return violations, preferred_exceedances, cost, overall_pp, class_pp


def _construct_candidate(
    rng: np.random.Generator,
    group_sizes: npt.NDArray[Any],
    class_codes: npt.NDArray[Any],
    config: SplitConfig,
) -> npt.NDArray[Any]:
    proportions = np.array([config.proportions[name] for name in SPLIT_NAMES])
    targets_total = proportions * group_sizes.sum()
    total_denominators = np.maximum(targets_total, 1.0)
    class_values = np.unique(class_codes)
    targets_class = {
        code: proportions * group_sizes[class_codes == code].sum()
        for code in class_values
    }
    class_denominators = {
        code: np.maximum(targets_class[code], 1.0) for code in class_values
    }
    totals = np.zeros(len(SPLIT_NAMES), dtype=np.float64)
    per_class = {
        code: np.zeros(len(SPLIT_NAMES), dtype=np.float64) for code in class_values
    }
    assignments = np.full(len(group_sizes), -1, dtype=np.int64)
    order = rng.permutation(len(group_sizes))
    for group_index in order:
        size = float(group_sizes[group_index])
        class_code = class_codes[group_index]
        total_residuals = (totals - targets_total) / total_denominators
        class_residuals = (
            per_class[class_code] - targets_class[class_code]
        ) / class_denominators[class_code]
        total_error_base = float(np.square(total_residuals).sum())
        class_error_base = float(np.square(class_residuals).sum())
        candidate_scores: list[tuple[float, float, int]] = []
        for split_index in range(len(SPLIT_NAMES)):
            proposed_total_residual = (
                totals[split_index] + size - targets_total[split_index]
            ) / total_denominators[split_index]
            proposed_class_residual = (
                per_class[class_code][split_index]
                + size
                - targets_class[class_code][split_index]
            ) / class_denominators[class_code][split_index]
            total_error = (
                total_error_base
                - float(total_residuals[split_index] ** 2)
                + float(proposed_total_residual**2)
            )
            class_error = (
                class_error_base
                - float(class_residuals[split_index] ** 2)
                + float(proposed_class_residual**2)
            )
            candidate_scores.append(
                (
                    float(total_error + 2.0 * class_error),
                    float(rng.random()),
                    split_index,
                )
            )
        _, _, chosen = min(candidate_scores)
        assignments[group_index] = chosen
        totals[chosen] += size
        per_class[class_code][chosen] += size
    return assignments


def optimise_grouped_stratified_split(
    group_table: pd.DataFrame, config: SplitConfig
) -> tuple[npt.NDArray[Any], dict[str, Any]]:
    """Search deterministic randomised greedy candidates and return the best."""

    group_sizes = group_table["size"].to_numpy(dtype=np.int64)
    class_codes, class_names = pd.factorize(group_table["class_name"], sort=True)
    rng = np.random.default_rng(config.seed)
    best_assignments: npt.NDArray[Any] | None = None
    best_key: tuple[int, int, float, tuple[int, ...]] | None = None
    best_details: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None
    accepted_candidates = 0
    for _ in range(config.candidate_restarts):
        assignments = _construct_candidate(rng, group_sizes, class_codes, config)
        violations, preferred_exceedances, cost, overall_pp, class_pp = _candidate_cost(
            assignments, group_sizes, class_codes, config
        )
        if violations == 0:
            accepted_candidates += 1
        key = (
            violations,
            preferred_exceedances,
            round(cost, 12),
            tuple(assignments.tolist()),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_assignments = assignments.copy()
            best_details = (overall_pp.copy(), class_pp.copy())
    if best_assignments is None or best_key is None or best_details is None:
        raise RuntimeError("Split candidate optimisation produced no candidate")
    overall_pp, class_pp = best_details
    report = {
        "candidate_restarts": config.candidate_restarts,
        "accepted_candidates": accepted_candidates,
        "best_violations": best_key[0],
        "preferred_class_tolerance_exceedances": best_key[1],
        "best_cost": best_key[2],
        "overall_deviation_pp": {
            name: float(overall_pp[index]) for index, name in enumerate(SPLIT_NAMES)
        },
        "class_deviation_pp": {
            str(class_names[class_index]): {
                name: float(class_pp[class_index, split_index])
                for split_index, name in enumerate(SPLIT_NAMES)
            }
            for class_index in range(len(class_names))
        },
        "tolerances_satisfied": best_key[0] == 0,
    }
    return best_assignments, report


def create_canonical_split(
    manifest_path: Path,
    confirmed_groups_path: Path,
    output_dir: Path,
    config: SplitConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create all canonical split artefacts or report infeasibility without relaxing."""

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    groups = pd.read_csv(confirmed_groups_path, dtype=str, keep_default_na=False)
    merged, group_table = _prepare_group_table(manifest, groups)
    if dry_run:
        return {
            "dry_run": True,
            "manifest_rows": len(merged),
            "groups": len(group_table),
            "candidate_restarts": config.candidate_restarts,
            "split_created": False,
            "test_accessed": False,
        }
    assignments, optimisation = optimise_grouped_stratified_split(group_table, config)
    group_table = group_table.copy()
    group_table["split"] = [SPLIT_NAMES[index] for index in assignments]
    if "split" in merged.columns:
        existing_split = merged["split"].astype(str).str.strip()
        if existing_split.ne("").any():
            raise ValueError(
                "Master manifest split column must remain blank before splitting"
            )
        merged = merged.drop(columns="split")
    merged = merged.merge(
        group_table[["group_id", "split"]], on="group_id", validate="many_to_one"
    )
    rows = [
        {
            "image_id": row["image_id"],
            "original_id": row["original_id"],
            "group_id": row["group_id"],
            "class_name": row["class_name"],
            "class_id": row["class_id"],
            "relative_path": row["relative_path"],
            "split": row["split"],
            "split_seed": config.seed,
        }
        for row in merged.sort_values("image_id").to_dict("records")
    ]
    split_frame = pd.DataFrame(rows)
    group_crossings = int(
        split_frame.groupby("group_id")["split"].nunique().gt(1).sum()
    )
    missing_classes = {
        split_name: sorted(
            set(split_frame["class_name"])
            - set(split_frame.loc[split_frame["split"] == split_name, "class_name"])
        )
        for split_name in SPLIT_NAMES
    }
    leakage_report = {
        "group_crossings": group_crossings,
        "missing_classes": missing_classes,
        "all_images_assigned_once": len(split_frame) == len(manifest)
        and not split_frame["image_id"].duplicated().any(),
        "historical_variants_present": False,
        "passed": group_crossings == 0
        and not any(missing_classes.values())
        and optimisation["tolerances_satisfied"],
        "test_accessed": False,
    }
    report = {
        "schema_version": "1.0.0",
        "algorithm": "grouped_stratified_candidate_optimisation",
        "seed": config.seed,
        "manifest_sha256": sha256_file(manifest_path),
        "confirmed_groups_sha256": sha256_file(confirmed_groups_path),
        "rows": len(split_frame),
        "groups": int(split_frame["group_id"].nunique()),
        "optimisation": optimisation,
        "leakage_check": leakage_report,
        "passed": bool(leakage_report["passed"]),
        "test_accessed": False,
    }
    output = ensure_output_directory(output_dir)
    write_json(output / "split_candidate_optimisation_report.json", report)
    if not report["passed"]:
        LOGGER.error("Best split candidate violates frozen tolerances")
        return report
    write_csv(output / "canonical_split.csv", rows, SPLIT_OUTPUT_FIELDS)
    write_json(output / "canonical_split.json", {"rows": rows})
    class_distribution = (
        split_frame.groupby(["split", "class_name"], as_index=False)
        .size()
        .rename(columns={"size": "image_count"})
        .to_dict("records")
    )
    group_distribution = (
        split_frame.groupby("split", as_index=False)
        .agg(image_count=("image_id", "size"), group_count=("group_id", "nunique"))
        .to_dict("records")
    )
    write_csv(
        output / "split_class_distribution.csv",
        class_distribution,
        ("split", "class_name", "image_count"),
    )
    write_csv(
        output / "split_group_distribution.csv",
        group_distribution,
        ("split", "image_count", "group_count"),
    )
    write_json(output / "split_leakage_check.json", leakage_report)
    split_summary = {
        **report,
        "counts_by_split": split_frame["split"].value_counts().sort_index().to_dict(),
        "class_distribution": class_distribution,
        "group_distribution": group_distribution,
    }
    write_json(output / "canonical_split_summary.json", split_summary)
    digest = sha256_file(output / "canonical_split.csv")
    write_text(output / "split_sha256.txt", digest)
    report["canonical_split_sha256"] = digest
    LOGGER.info("Canonical split created", extra={"sha256": digest})
    return report
