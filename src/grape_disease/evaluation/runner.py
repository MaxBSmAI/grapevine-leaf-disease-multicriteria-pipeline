"""Checkpoint evaluation on validation or strictly authorised test data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grape_disease.data.dataset import (
    ManifestImageDataset,
    build_dataloader,
    build_transforms,
    load_canonical_split,
)
from grape_disease.evaluation.guard import require_frozen_test_evaluation
from grape_disease.evaluation.prediction import (
    predict_loader,
    write_prediction_evaluation,
)
from grape_disease.models.factory import ModelSpec, build_model, load_model_state_strict
from grape_disease.training.checkpoint import load_checkpoint
from grape_disease.training.reproducibility import set_reproducibility
from grape_disease.utils.dependencies import require_module
from grape_disease.utils.hashing import sha256_file

torch = require_module("torch")


def evaluate_checkpoint(
    repository_root: Path,
    dataset_root: Path,
    split_path: Path,
    normalisation_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    partition: str,
    batch_size: int,
    device_name: str,
    test_guard_paths: dict[str, Path] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate one checkpoint, with the full guard mandatory for test."""

    if partition not in {"validation", "test"}:
        raise ValueError("Evaluation partition must be validation or test")
    guard: dict[str, Any] | None = None
    if partition == "test":
        if test_guard_paths is None:
            raise PermissionError("Test evaluation requires all guard paths")
        guard = require_frozen_test_evaluation(**test_guard_paths)
    split = load_canonical_split(split_path)
    checkpoint = load_checkpoint(checkpoint_path)
    actual_split_hash = sha256_file(split_path)
    if checkpoint.get("split_sha256") != actual_split_hash:
        raise ValueError("Checkpoint and canonical split hashes differ")
    normalisation = json.loads(normalisation_path.read_text(encoding="utf-8"))
    if checkpoint.get("normalisation_sha256") != sha256_file(normalisation_path):
        raise ValueError("Checkpoint and normalisation hashes differ")
    class_to_idx = checkpoint["class_to_idx"]
    class_names = [
        name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]
    if dry_run:
        return {
            "dry_run": True,
            "partition": partition,
            "rows": int((split["split"] == partition).sum()),
            "guard_verified": guard is not None,
        }
    reproducibility = set_reproducibility(
        int(checkpoint["seed"]),
        str(checkpoint["training_config"]["deterministic_level"]),
    )
    generator = reproducibility["dataloader_generator"]
    transforms_by_partition = build_transforms(
        normalisation["mean_rgb"], normalisation["std_rgb"]
    )
    dataset = ManifestImageDataset(
        dataset_root,
        split,
        partition,
        transforms_by_partition[partition],
        allow_test=partition == "test" and guard is not None,
    )
    loader = build_dataloader(dataset, batch_size, generator)
    spec = ModelSpec(**checkpoint["model_config"])
    model = build_model(spec, repository_root=repository_root)
    load_model_state_strict(model, checkpoint["model_state_dict"])
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device)
    metadata = {
        "experiment_id": checkpoint["experiment_id"],
        "model_name": checkpoint["model_name"],
        "regime": checkpoint["regime"],
        "seed": checkpoint["seed"],
        "split": partition,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "protocol_sha256": checkpoint["protocol_sha256"],
        "split_sha256": actual_split_hash,
    }
    rows, labels, probabilities = predict_loader(model, loader, device, metadata)
    prediction_name = (
        "predictions_test.csv" if partition == "test" else "validation_predictions.csv"
    )
    metrics_name = (
        "metrics_test.json" if partition == "test" else "validation_metrics.json"
    )
    metrics = write_prediction_evaluation(
        output_dir,
        prediction_name,
        metrics_name,
        rows,
        labels,
        probabilities,
        class_names,
    )
    return {
        "partition": partition,
        "metrics": metrics,
        "guard_verified": guard is not None,
        "predictions": len(rows),
    }
