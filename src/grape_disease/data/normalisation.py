"""Train-only streaming RGB normalisation statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from grape_disease.utils.hashing import sha256_file
from grape_disease.utils.io import ensure_output_directory, write_json, write_text


def calculate_train_normalisation(
    dataset_root: Path,
    split_path: Path,
    manifest_path: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Compute float64 channel moments using only rows labelled train."""

    split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    required = {"image_id", "relative_path", "split"}
    missing = sorted(required - set(split.columns))
    if missing:
        raise ValueError(f"Canonical split missing columns: {missing}")
    train = split.loc[split["split"] == "train"].copy()
    if train.empty:
        raise ValueError("Canonical split has no train images")
    if dry_run:
        return {
            "dry_run": True,
            "train_images": len(train),
            "validation_images_accessed": 0,
            "test_images_accessed": 0,
        }
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    root = dataset_root.resolve()
    for relative_path in train["relative_path"]:
        path = (root / Path(relative_path)).resolve()
        if root not in path.parents:
            raise ValueError(f"Path escapes dataset root: {relative_path}")
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        pixels = rgb.reshape(-1, 3)
        channel_sum += pixels.sum(axis=0, dtype=np.float64)
        channel_squared_sum += np.square(pixels).sum(axis=0, dtype=np.float64)
        pixel_count += len(pixels)
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_squared_sum / pixel_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std <= 0.0):
        raise ValueError("At least one train RGB channel has zero variance")
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "partition": "train",
        "algorithm": "streaming_channel_sum_and_squared_sum",
        "numeric_precision": "float64",
        "mean_rgb": mean.tolist(),
        "std_rgb": std.tolist(),
        "image_count": len(train),
        "pixel_count": pixel_count,
        "manifest_sha256": sha256_file(manifest_path),
        "split_sha256": sha256_file(split_path),
        "validation_images_accessed": 0,
        "test_images_accessed": 0,
    }
    output = ensure_output_directory(output_dir)
    write_json(output / "train_normalisation.json", report)
    markdown = (
        "# Train normalisation report\n\n"
        f"- Images: {len(train)}\n"
        f"- Pixels: {pixel_count}\n"
        f"- Mean RGB: {mean.tolist()}\n"
        f"- Standard deviation RGB: {std.tolist()}\n"
        "- Validation images accessed: 0\n"
        "- Test images accessed: 0\n"
    )
    write_text(output / "train_normalisation_report.md", markdown)
    return report
