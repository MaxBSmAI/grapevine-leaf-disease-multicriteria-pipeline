"""Model-independent stratified XAI sample selection from frozen split metadata."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grape_disease.utils.hashing import sha256_file
from grape_disease.utils.io import ensure_output_directory, write_csv, write_text

XAI_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://example.invalid/grape-disease-reproducible/xai-sample/v1",
)


def create_xai_sample_manifest(
    split_path: Path,
    manifest_path: Path,
    output_dir: Path,
    images_per_class: int,
    selection_seed: int | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Select test metadata before predictions, without opening test images."""

    if selection_seed is None:
        raise ValueError("XAI selection_seed must be explicitly approved")
    if images_per_class <= 0:
        raise ValueError("images_per_class must be positive")
    split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    required = {
        "image_id",
        "original_id",
        "group_id",
        "class_name",
        "relative_path",
        "split",
    }
    missing = sorted(required - set(split.columns))
    if missing:
        raise ValueError(f"Canonical split missing XAI columns: {missing}")
    test = split.loc[split["split"] == "test"].copy()
    classes = sorted(test["class_name"].unique())
    counts = test["class_name"].value_counts().to_dict()
    insufficient = {
        name: counts.get(name, 0)
        for name in classes
        if counts.get(name, 0) < images_per_class
    }
    if insufficient:
        raise ValueError(f"Insufficient test images for XAI sample: {insufficient}")
    if dry_run:
        return {
            "dry_run": True,
            "classes": classes,
            "images_per_class": images_per_class,
            "selected": len(classes) * images_per_class,
            "test_pixels_accessed": False,
        }
    rng = np.random.default_rng(selection_seed)
    rows: list[dict[str, Any]] = []
    selection_order = 0
    for class_name in classes:
        class_rows = test.loc[test["class_name"] == class_name].sort_values("image_id")
        selected_indices = rng.choice(
            len(class_rows), size=images_per_class, replace=False
        )
        for row_index in selected_indices:
            source = class_rows.iloc[int(row_index)]
            selection_order += 1
            image_id = str(source["image_id"])
            sample_id = str(uuid.uuid5(XAI_NAMESPACE, f"{selection_seed}:{image_id}"))
            rows.append(
                {
                    "xai_sample_id": sample_id,
                    "image_id": image_id,
                    "original_id": source["original_id"],
                    "group_id": source["group_id"],
                    "true_class": source["class_name"],
                    "relative_path": source["relative_path"],
                    "split": "test",
                    "selection_seed": selection_seed,
                    "selection_order": selection_order,
                    "sha256": "resolved_from_master_manifest",
                }
            )
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    sha_by_image = manifest.set_index("image_id")["sha256"].to_dict()
    for row in rows:
        if row["image_id"] not in sha_by_image:
            raise ValueError(
                f"XAI image absent from master manifest: {row['image_id']}"
            )
        row["sha256"] = sha_by_image[row["image_id"]]
    output = ensure_output_directory(output_dir)
    fields = (
        "xai_sample_id",
        "image_id",
        "original_id",
        "group_id",
        "true_class",
        "relative_path",
        "split",
        "selection_seed",
        "selection_order",
        "sha256",
    )
    destination = output / "xai_sample_manifest.csv"
    write_csv(destination, rows, fields)
    digest = sha256_file(destination)
    write_text(output / "xai_sample_manifest_sha256.txt", digest)
    write_text(
        output / "xai_selection_report.md",
        "# XAI sample selection\n\n"
        f"- Seed: {selection_seed}\n"
        f"- Images per class: {images_per_class}\n"
        f"- Total: {len(rows)}\n"
        "- Model-dependent filtering: false\n"
        "- Test pixels opened during selection: false\n",
    )
    return {
        "selected": len(rows),
        "classes": classes,
        "sha256": digest,
        "test_pixels_accessed": False,
    }
