"""Guarded execution of common and architecture-specific XAI methods."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from grape_disease.data.dataset import build_transforms
from grape_disease.evaluation.guard import require_frozen_test_evaluation
from grape_disease.models.factory import ModelSpec, build_model, load_model_state_strict
from grape_disease.training.checkpoint import load_checkpoint
from grape_disease.utils.dependencies import require_module
from grape_disease.utils.hashing import sha256_file
from grape_disease.utils.io import ensure_output_directory, write_csv, write_json
from grape_disease.xai.faithfulness import (
    PerturbationConfig,
    perturbation_curves,
    random_baseline_trials,
)
from grape_disease.xai.integrated_gradients import (
    IntegratedGradientsConfig,
    explain_tensor,
    save_attribution_bundle,
)
from grape_disease.xai.specific import run_specific_method

torch = require_module("torch")


def _load_context(
    repository_root: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    normalisation_path: Path,
    sample_manifest_path: Path,
    test_guard_paths: dict[str, Path],
    device_name: str,
) -> tuple[
    Any,
    Any,
    dict[str, Any],
    pd.DataFrame,
    Any,
    list[str],
    str,
    dict[str, Any],
]:
    require_frozen_test_evaluation(**test_guard_paths)
    checkpoint = load_checkpoint(checkpoint_path)
    normalisation = json.loads(normalisation_path.read_text(encoding="utf-8"))
    if checkpoint["normalisation_sha256"] != sha256_file(normalisation_path):
        raise ValueError("Checkpoint and XAI normalisation hashes differ")
    sample = pd.read_csv(sample_manifest_path, dtype=str, keep_default_na=False)
    if set(sample["split"]) != {"test"}:
        raise ValueError("XAI sample manifest must contain test rows only")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    spec = ModelSpec(**checkpoint["model_config"])
    model = build_model(spec, repository_root).to(device).eval()
    load_model_state_strict(model, checkpoint["model_state_dict"])
    transform = build_transforms(normalisation["mean_rgb"], normalisation["std_rgb"])[
        "test"
    ]
    class_names = [
        name
        for name, _ in sorted(
            checkpoint["class_to_idx"].items(), key=lambda item: item[1]
        )
    ]
    return (
        model,
        device,
        normalisation,
        sample,
        transform,
        class_names,
        sha256_file(checkpoint_path),
        checkpoint,
    )


def _input_tensor(
    dataset_root: Path, row: dict[str, str], transform: Any, device: Any
) -> Any:
    path = dataset_root / Path(row["relative_path"])
    if sha256_file(path) != row["sha256"]:
        raise ValueError(f"XAI input hash mismatch: {path}")
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0).to(device)


def run_integrated_gradients_sample(
    repository_root: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    normalisation_path: Path,
    sample_manifest_path: Path,
    output_dir: Path,
    config: IntegratedGradientsConfig,
    test_guard_paths: dict[str, Path],
    device_name: str,
    include_ground_truth_target: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run predicted-logit IG and optional ground-truth-logit IG."""

    if dry_run:
        sample = pd.read_csv(sample_manifest_path, dtype=str, keep_default_na=False)
        return {"dry_run": True, "planned_images": len(sample), "pixels_opened": 0}
    context = _load_context(
        repository_root,
        dataset_root,
        checkpoint_path,
        normalisation_path,
        sample_manifest_path,
        test_guard_paths,
        device_name,
    )
    model, device, _, sample, transform, class_names, checkpoint_hash, checkpoint = (
        context
    )
    output = ensure_output_directory(output_dir)
    rows: list[dict[str, Any]] = []
    for source in sample.to_dict("records"):
        inputs = _input_tensor(dataset_root, source, transform, device)
        with torch.inference_mode():
            predicted = int(model(inputs).argmax(dim=1).item())
        ground_truth = class_names.index(source["true_class"])
        targets = [("predicted_class_logit", predicted)]
        if include_ground_truth_target:
            targets.append(("ground_truth_class_logit", ground_truth))
        for target_name, target in targets:
            result = explain_tensor(model, inputs, target, config)
            filename = f"{source['xai_sample_id']}__{target_name}.npz"
            save_attribution_bundle(output / "maps" / filename, result)
            rows.append(
                {
                    "model_name": checkpoint["model_name"],
                    "regime": checkpoint["regime"],
                    "seed": checkpoint["seed"],
                    "xai_sample_id": source["xai_sample_id"],
                    "image_id": source["image_id"],
                    "true_class": source["true_class"],
                    "predicted_class": class_names[predicted],
                    "target_type": target_name,
                    "target_class": class_names[target],
                    "converged": result["converged"],
                    "relative_completeness_error": result[
                        "relative_completeness_error"
                    ],
                    "n_steps": result["n_steps"],
                    "map_path": str(Path("maps") / filename),
                }
            )
    write_csv(output / "integrated_gradients_registry.csv", rows, tuple(rows[0]))
    summary = {
        "images": len(sample),
        "maps": len(rows),
        "converged_maps": sum(bool(row["converged"]) for row in rows),
        "checkpoint_sha256": checkpoint_hash,
        "model_name": checkpoint["model_name"],
        "regime": checkpoint["regime"],
        "seed": checkpoint["seed"],
        "config": asdict(config),
        "test_access_authorised": True,
    }
    write_json(output / "integrated_gradients_summary.json", summary)
    return summary


def run_specific_xai_sample(
    repository_root: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    normalisation_path: Path,
    sample_manifest_path: Path,
    output_dir: Path,
    test_guard_paths: dict[str, Path],
    device_name: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run each architecture's predeclared complementary XAI method."""

    if dry_run:
        sample = pd.read_csv(sample_manifest_path, dtype=str, keep_default_na=False)
        return {"dry_run": True, "planned_images": len(sample), "pixels_opened": 0}
    context = _load_context(
        repository_root,
        dataset_root,
        checkpoint_path,
        normalisation_path,
        sample_manifest_path,
        test_guard_paths,
        device_name,
    )
    model, device, _, sample, transform, class_names, checkpoint_hash, checkpoint = (
        context
    )
    model_name = str(checkpoint["model_name"])
    output = ensure_output_directory(output_dir)
    rows: list[dict[str, Any]] = []
    for source in sample.to_dict("records"):
        inputs = _input_tensor(dataset_root, source, transform, device)
        with torch.inference_mode():
            predicted = int(model(inputs).argmax(dim=1).item())
        result = run_specific_method(model, inputs, predicted, model_name)
        filename = f"{source['xai_sample_id']}__predicted_class.npz"
        path = output / "maps" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ranking_map=result["map"].detach().cpu().numpy())
        rows.append(
            {
                "model_name": model_name,
                "regime": checkpoint["regime"],
                "seed": checkpoint["seed"],
                "xai_sample_id": source["xai_sample_id"],
                "image_id": source["image_id"],
                "true_class": source["true_class"],
                "predicted_class": class_names[predicted],
                "method": result["method"],
                "map_path": str(Path("maps") / filename),
            }
        )
    write_csv(output / "specific_xai_registry.csv", rows, tuple(rows[0]))
    summary = {
        "images": len(rows),
        "model_name": model_name,
        "method": rows[0]["method"],
        "checkpoint_sha256": checkpoint_hash,
        "regime": checkpoint["regime"],
        "seed": checkpoint["seed"],
        "test_access_authorised": True,
    }
    write_json(output / "specific_xai_summary.json", summary)
    return summary


def run_faithfulness_sample(
    repository_root: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    normalisation_path: Path,
    sample_manifest_path: Path,
    attribution_registry_path: Path,
    attribution_root: Path,
    output_dir: Path,
    config: PerturbationConfig,
    test_guard_paths: dict[str, Path],
    device_name: str,
    method_name: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate attribution order against deterministic random block orders."""

    registry = pd.read_csv(attribution_registry_path, dtype=str, keep_default_na=False)
    if "target_type" in registry and "predicted_class_logit" in set(
        registry["target_type"]
    ):
        registry = registry.loc[registry["target_type"] == "predicted_class_logit"]
    if dry_run:
        return {"dry_run": True, "planned_maps": len(registry), "pixels_opened": 0}
    context = _load_context(
        repository_root,
        dataset_root,
        checkpoint_path,
        normalisation_path,
        sample_manifest_path,
        test_guard_paths,
        device_name,
    )
    (
        model,
        device,
        normalisation,
        sample,
        transform,
        class_names,
        checkpoint_hash,
        checkpoint,
    ) = context
    sample_by_id = sample.set_index("xai_sample_id").to_dict("index")
    output = ensure_output_directory(output_dir)
    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for attribution in registry.to_dict("records"):
        source = sample_by_id[attribution["xai_sample_id"]]
        inputs = _input_tensor(dataset_root, source, transform, device)
        target = class_names.index(attribution["predicted_class"])
        bundle = np.load(attribution_root / attribution["map_path"])
        ranking = torch.as_tensor(
            bundle["ranking_map"], device=device, dtype=inputs.dtype
        )
        curves = perturbation_curves(
            model,
            inputs,
            ranking,
            target,
            normalisation["mean_rgb"],
            normalisation["std_rgb"],
            config,
        )
        summary_rows.append(
            {
                "model_name": checkpoint["model_name"],
                "regime": checkpoint["regime"],
                "seed": checkpoint["seed"],
                "xai_sample_id": attribution["xai_sample_id"],
                "image_id": source["image_id"],
                "method": method_name,
                "target_class": class_names[target],
                "insertion_partial_auc": curves["insertion_partial_auc"],
                "deletion_partial_auc": curves["deletion_partial_auc"],
                "score_function": curves["score_function"],
            }
        )
        for fraction, insertion, deletion in zip(
            curves["fractions"],
            curves["insertion_scores"],
            curves["deletion_scores"],
            strict=True,
        ):
            curve_rows.append(
                {
                    "xai_sample_id": attribution["xai_sample_id"],
                    "fraction": float(fraction),
                    "insertion_score": float(insertion),
                    "deletion_score": float(deletion),
                }
            )
        for random_result in random_baseline_trials(
            model,
            inputs,
            ranking,
            target,
            normalisation["mean_rgb"],
            normalisation["std_rgb"],
            config,
            source["image_id"],
            checkpoint_hash,
            method_name,
        ):
            random_rows.append(
                {"xai_sample_id": attribution["xai_sample_id"], **random_result}
            )
    write_csv(output / "faithfulness_summary.csv", summary_rows, tuple(summary_rows[0]))
    write_csv(output / "faithfulness_curves.csv", curve_rows, tuple(curve_rows[0]))
    write_csv(output / "random_baselines.csv", random_rows, tuple(random_rows[0]))
    result = {
        "images": len(summary_rows),
        "random_trials": len(random_rows),
        "checkpoint_sha256": checkpoint_hash,
        "config": asdict(config),
        "test_access_authorised": True,
    }
    write_json(output / "faithfulness_metadata.json", result)
    return result
