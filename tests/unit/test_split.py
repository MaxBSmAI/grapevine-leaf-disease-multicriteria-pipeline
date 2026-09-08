from __future__ import annotations

from pathlib import Path

import pandas as pd

from grape_disease.data.split import (
    SplitConfig,
    create_canonical_split,
    optimise_grouped_stratified_split,
)


def test_grouped_candidate_split_is_deterministic_and_feasible() -> None:
    groups = pd.DataFrame(
        [
            {"group_id": f"{class_name}_{index}", "class_name": class_name, "size": 1}
            for class_name in ("a", "b", "c", "d")
            for index in range(20)
        ]
    )
    config = SplitConfig(
        seed=1729,
        proportions={"train": 0.70, "validation": 0.15, "test": 0.15},
        preferred_class_tolerance_pp=1.0,
        maximum_class_tolerance_pp=2.0,
        overall_tolerance_pp=0.5,
        candidate_restarts=200,
        total_cost_weight=1.0,
        class_cost_weight=2.0,
        violation_cost_weight=10.0,
        approval_status="approved_gate_3",
    )
    first, first_report = optimise_grouped_stratified_split(groups, config)
    second, second_report = optimise_grouped_stratified_split(groups, config)
    assert first.tolist() == second.tolist()
    assert first_report == second_report
    assert first_report["tolerances_satisfied"] is True


def test_create_canonical_split_replaces_blank_manifest_split(
    tmp_path: Path,
) -> None:
    manifest_rows = []
    group_rows = []
    for class_id, class_name in enumerate(("a", "b", "c", "d")):
        for index in range(40):
            image_id = f"{class_name}_{index}"
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "original_id": image_id,
                    "class_name": class_name,
                    "class_id": str(class_id),
                    "relative_path": f"{class_name}/{image_id}.jpg",
                    "is_original": "true",
                    "group_id": f"provisional_{image_id}",
                    "split": "",
                }
            )
            group_rows.append(
                {
                    "image_id": image_id,
                    "original_id": image_id,
                    "group_id": f"confirmed_{class_name}_{index // 2}",
                    "class_name": class_name,
                }
            )
    manifest_path = tmp_path / "manifest.csv"
    groups_path = tmp_path / "groups.csv"
    output_dir = tmp_path / "split"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    pd.DataFrame(group_rows).to_csv(groups_path, index=False)
    config = SplitConfig(
        seed=1729,
        proportions={"train": 0.70, "validation": 0.15, "test": 0.15},
        preferred_class_tolerance_pp=1.0,
        maximum_class_tolerance_pp=2.0,
        overall_tolerance_pp=0.5,
        candidate_restarts=200,
        total_cost_weight=1.0,
        class_cost_weight=2.0,
        violation_cost_weight=10.0,
        approval_status="approved_gate_3",
    )

    report = create_canonical_split(
        manifest_path, groups_path, output_dir, config
    )

    assert report["passed"] is True
    split = pd.read_csv(output_dir / "canonical_split.csv")
    assert len(split) == 160
    assert set(split["split"]) == {"train", "validation", "test"}
    assert split["group_id"].nunique() == 80
    assert split.groupby("group_id")["split"].nunique().max() == 1
