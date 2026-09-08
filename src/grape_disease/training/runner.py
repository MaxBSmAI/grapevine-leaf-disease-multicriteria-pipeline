"""End-to-end train/validation execution with complete provenance artefacts."""

from __future__ import annotations

import copy
import json
import logging
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from grape_disease.data.dataset import (
    ManifestImageDataset,
    build_dataloader,
    build_transforms,
    load_canonical_split,
)
from grape_disease.evaluation.prediction import (
    predict_loader,
    write_prediction_evaluation,
)
from grape_disease.models.factory import ModelSpec, build_model, model_metadata
from grape_disease.training.checkpoint import (
    load_checkpoint,
    save_checkpoint_once,
    save_progress_checkpoint,
)
from grape_disease.training.engine import train_one_epoch, validate_one_epoch
from grape_disease.training.losses import build_loss
from grape_disease.training.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    set_reproducibility,
)
from grape_disease.utils.dependencies import collect_software_versions, require_module
from grape_disease.utils.hashing import sha256_canonical_json, sha256_file
from grape_disease.utils.io import (
    ensure_output_directory,
    write_csv,
    write_json,
    write_text,
)

torch = require_module("torch")

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingRunConfig:
    """One serialisable training execution."""

    model_name: str
    regime: str
    seed: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    momentum: float
    batch_size: int
    effective_batch_size: int
    gradient_accumulation_steps: int
    max_optimizer_updates: int
    warmup_updates: int
    early_stopping_patience_checks: int
    minimum_improvement: float
    amp: bool
    deterministic_level: str
    device: str
    num_workers: int
    focal_gamma: float | None
    gradient_clip_norm: float | None
    run_kind: str
    code_commit: str
    architecture_config: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> TrainingRunConfig:
        config = cls(
            model_name=str(value["model_name"]),
            regime=str(value["regime"]),
            seed=int(value["seed"]),
            optimizer=str(value["optimizer"]),
            learning_rate=float(value["learning_rate"]),
            weight_decay=float(value["weight_decay"]),
            momentum=float(value.get("momentum", 0.9)),
            batch_size=int(value["batch_size"]),
            effective_batch_size=int(value["effective_batch_size"]),
            gradient_accumulation_steps=int(value["gradient_accumulation_steps"]),
            max_optimizer_updates=int(value["max_optimizer_updates"]),
            warmup_updates=int(value["warmup_updates"]),
            early_stopping_patience_checks=int(value["early_stopping_patience_checks"]),
            minimum_improvement=float(value["minimum_improvement"]),
            amp=bool(value["amp"]),
            deterministic_level=str(value["deterministic_level"]),
            device=str(value.get("device", "cuda")),
            num_workers=int(value.get("num_workers", 0)),
            focal_gamma=(
                None
                if value.get("focal_gamma") is None
                else float(value["focal_gamma"])
            ),
            gradient_clip_norm=(
                None
                if value.get("gradient_clip_norm") is None
                else float(value["gradient_clip_norm"])
            ),
            run_kind=str(value["run_kind"]),
            code_commit=str(value.get("code_commit", "uncommitted")),
            architecture_config=(
                None
                if value.get("architecture_config") is None
                else str(value["architecture_config"])
            ),
        )
        if (
            config.batch_size * config.gradient_accumulation_steps
            != config.effective_batch_size
        ):
            raise ValueError(
                "Batch size times accumulation must equal effective batch size"
            )
        if config.max_optimizer_updates <= 0 or config.warmup_updates < 0:
            raise ValueError("Update budgets are invalid")
        if config.warmup_updates >= config.max_optimizer_updates:
            raise ValueError("Warmup must be shorter than the total update budget")
        if config.optimizer not in {"AdamW", "SGD"}:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        return config


def _build_optimizer(model: Any, config: TrainingRunConfig) -> Any:
    if config.optimizer == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )


def _build_scheduler(optimizer: Any, config: TrainingRunConfig) -> Any:
    def multiplier(update: int) -> float:
        one_based = update + 1
        if one_based <= config.warmup_updates:
            return one_based / max(config.warmup_updates, 1)
        progress = (one_based - config.warmup_updates) / max(
            config.max_optimizer_updates - config.warmup_updates, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _class_order(split: pd.DataFrame) -> list[str]:
    class_rows = (
        split[["class_id", "class_name"]]
        .drop_duplicates()
        .assign(class_id=lambda frame: frame["class_id"].astype(int))
        .sort_values("class_id")
    )
    expected = list(range(len(class_rows)))
    if class_rows["class_id"].tolist() != expected:
        raise ValueError(f"Class IDs must be contiguous: {expected}")
    return [str(name) for name in class_rows["class_name"]]


def run_training(
    repository_root: Path,
    dataset_root: Path,
    split_path: Path,
    normalisation_path: Path,
    protocol_path: Path,
    output_dir: Path,
    config: TrainingRunConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train one model using train/validation only and write all run artefacts."""

    split = load_canonical_split(split_path)
    class_names = _class_order(split)
    normalisation = json.loads(normalisation_path.read_text(encoding="utf-8"))
    config_payload = asdict(config)
    input_hashes = {
        "dataset_manifest_sha256": str(normalisation["manifest_sha256"]),
        "split_sha256": sha256_file(split_path),
        "protocol_sha256": sha256_file(protocol_path),
        "normalisation_sha256": sha256_file(normalisation_path),
        "config_sha256": sha256_canonical_json(config_payload),
    }
    identity = {
        "model_name": config.model_name,
        "regime": config.regime,
        "seed": config.seed,
        "split_sha256": input_hashes["split_sha256"],
        "config_sha256": input_hashes["config_sha256"],
        "run_kind": config.run_kind,
    }
    experiment_id = sha256_canonical_json(identity)[:20]
    if dry_run:
        return {
            "dry_run": True,
            "experiment_id": experiment_id,
            "train_rows": int((split["split"] == "train").sum()),
            "validation_rows": int((split["split"] == "validation").sum()),
            "test_rows_accessed": 0,
            "input_hashes": input_hashes,
        }
    output = ensure_output_directory(output_dir)
    metadata_path = output / "run_metadata.json"
    if metadata_path.is_file():
        completed = cast(
            dict[str, Any],
            json.loads(metadata_path.read_text(encoding="utf-8")),
        )
        if completed.get("experiment_id") != experiment_id:
            raise RuntimeError(f"Completed run identity mismatch: {output}")
        if completed.get("input_hashes") != input_hashes:
            raise RuntimeError(f"Completed run input hash mismatch: {output}")
        if completed.get("status") != "completed":
            raise RuntimeError(f"Run metadata is not completed: {output}")
        return completed
    progress_path = output / "progress_checkpoint.pth"
    config_path = output / "run_config.yaml"
    allowed_existing = {
        config_path.name,
        progress_path.name,
        f"{progress_path.name}.sha256",
    }
    unexpected = [
        path.name for path in output.iterdir() if path.name not in allowed_existing
    ]
    if unexpected:
        raise FileExistsError(
            f"Run output has unrecognised partial artefacts {unexpected}: {output}"
        )
    if config_path.is_file():
        existing_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if existing_config != config_payload:
            raise RuntimeError(f"Resume configuration mismatch: {output}")
    else:
        write_text(config_path, yaml.safe_dump(config_payload, sort_keys=True))

    reproducibility = set_reproducibility(config.seed, config.deterministic_level)
    generator = reproducibility.pop("dataloader_generator")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    transforms_by_partition = build_transforms(
        normalisation["mean_rgb"], normalisation["std_rgb"]
    )
    train_dataset = ManifestImageDataset(
        dataset_root, split, "train", transforms_by_partition["train"]
    )
    validation_dataset = ManifestImageDataset(
        dataset_root,
        split,
        "validation",
        transforms_by_partition["validation"],
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        generator=generator,
        regime=config.regime,
        num_workers=config.num_workers,
    )
    validation_loader = build_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        generator=generator,
        num_workers=config.num_workers,
    )
    spec = ModelSpec(
        name=config.model_name,
        num_classes=len(class_names),
        architecture_config=config.architecture_config,
    )
    model = build_model(spec, repository_root=repository_root).to(device)
    loss_function = build_loss(config.regime, config.focal_gamma)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    history: list[dict[str, Any]] = []
    global_updates = 0
    epoch = 0
    checks_without_improvement = 0
    best_key: tuple[float, float, int] | None = None
    best_snapshot: dict[str, Any] | None = None
    if progress_path.is_file():
        progress = load_checkpoint(progress_path, map_location=device)
        if progress.get("experiment_id") != experiment_id:
            raise RuntimeError(f"Progress checkpoint identity mismatch: {output}")
        if progress.get("input_hashes") != input_hashes:
            raise RuntimeError(f"Progress checkpoint input hash mismatch: {output}")
        model.load_state_dict(progress["model_state_dict"], strict=True)
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        scheduler.load_state_dict(progress["scheduler_state_dict"])
        scaler.load_state_dict(progress["scaler_state_dict"])
        global_updates = int(progress["global_updates"])
        epoch = int(progress["epoch"])
        checks_without_improvement = int(progress["checks_without_improvement"])
        raw_best_key = progress.get("best_key")
        best_key = (
            None
            if raw_best_key is None
            else (
                float(raw_best_key[0]),
                float(raw_best_key[1]),
                int(raw_best_key[2]),
            )
        )
        best_snapshot = progress.get("best_snapshot")
        history = list(progress["history"])
        restore_rng_state(progress["rng_states"])
        dataloader_generator_state = progress["dataloader_generator_state"]
        if not isinstance(dataloader_generator_state, torch.Tensor):
            dataloader_generator_state = torch.as_tensor(
                dataloader_generator_state,
                dtype=torch.uint8,
                device="cpu",
            )
        else:
            dataloader_generator_state = dataloader_generator_state.to(
                dtype=torch.uint8,
                device="cpu",
            )
        generator.set_state(dataloader_generator_state)
    while global_updates < config.max_optimizer_updates:
        epoch += 1
        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            remaining_updates=config.max_optimizer_updates - global_updates,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_clip_norm=config.gradient_clip_norm,
            amp=config.amp,
        )
        global_updates += train_result.optimizer_updates
        validation_metrics, _, _ = validate_one_epoch(
            model, validation_loader, loss_function, device, class_names
        )
        current_key = (
            -float(validation_metrics["macro_f1"]),
            float(validation_metrics["validation_loss"]),
            epoch,
        )
        improved = (
            best_key is None
            or (
                float(validation_metrics["macro_f1"])
                > -best_key[0] + config.minimum_improvement
            )
            or (
                abs(float(validation_metrics["macro_f1"]) + best_key[0])
                <= config.minimum_improvement
                and current_key < best_key
            )
        )
        if improved:
            best_key = current_key
            checks_without_improvement = 0
            best_snapshot = {
                "epoch": epoch,
                "optimizer_updates": global_updates,
                "model_state_dict": copy.deepcopy(
                    {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    }
                ),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
                "scaler_state_dict": copy.deepcopy(scaler.state_dict()),
                "validation_metrics": copy.deepcopy(validation_metrics),
                "rng_states": capture_rng_state(),
            }
        else:
            checks_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "optimizer_updates": global_updates,
                "train_loss": train_result.loss,
                "validation_loss": validation_metrics["validation_loss"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "mean_grad_norm": train_result.mean_grad_norm,
                "elapsed_seconds": train_result.elapsed_seconds,
                "samples": train_result.samples,
                "batches": train_result.batches,
                "unique_originals": train_result.unique_originals,
                "repeated_samples": train_result.repeated_samples,
                "effective_class_counts": json.dumps(
                    train_result.effective_class_counts, sort_keys=True
                ),
                "improved": improved,
            }
        )
        LOGGER.info(
            "Epoch complete",
            extra={
                "experiment_id": experiment_id,
                "epoch": epoch,
                "updates": global_updates,
                "validation_macro_f1": validation_metrics["macro_f1"],
            },
        )
        save_progress_checkpoint(
            progress_path,
            {
                "schema_version": "1.0.0",
                "experiment_id": experiment_id,
                "input_hashes": input_hashes,
                "epoch": epoch,
                "global_updates": global_updates,
                "checks_without_improvement": checks_without_improvement,
                "best_key": best_key,
                "best_snapshot": best_snapshot,
                "history": history,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng_states": capture_rng_state(),
                "dataloader_generator_state": generator.get_state(),
            },
        )
        if checks_without_improvement >= config.early_stopping_patience_checks:
            break
    if best_snapshot is None or best_key is None:
        raise RuntimeError("Training finished without a valid validation checkpoint")
    model.load_state_dict(best_snapshot["model_state_dict"], strict=True)
    checkpoint_name = (
        f"{config.model_name}__{config.regime}__{config.seed}__"
        f"{input_hashes['split_sha256'][:12]}__best_val_macro_f1.pth"
    )
    checkpoint_payload = {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "model_name": config.model_name,
        "regime": config.regime,
        "seed": config.seed,
        "split_id": input_hashes["split_sha256"][:12],
        "epoch": best_snapshot["epoch"],
        "optimizer_updates": best_snapshot["optimizer_updates"],
        "model_state_dict": best_snapshot["model_state_dict"],
        "optimizer_state_dict": best_snapshot["optimizer_state_dict"],
        "scheduler_state_dict": best_snapshot["scheduler_state_dict"],
        "scaler_state_dict": best_snapshot["scaler_state_dict"],
        "validation_macro_f1": best_snapshot["validation_metrics"]["macro_f1"],
        "validation_loss": best_snapshot["validation_metrics"]["validation_loss"],
        "class_to_idx": {name: index for index, name in enumerate(class_names)},
        "model_config": asdict(spec),
        "training_config": config_payload,
        "normalisation": normalisation,
        **input_hashes,
        "code_commit": config.code_commit,
        "software_versions": collect_software_versions(),
        "rng_states": best_snapshot["rng_states"],
    }
    checkpoint_path = output / checkpoint_name
    checkpoint_sha256 = save_checkpoint_once(checkpoint_path, checkpoint_payload)
    prediction_metadata = {
        "experiment_id": experiment_id,
        "model_name": config.model_name,
        "regime": config.regime,
        "seed": config.seed,
        "split": "validation",
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": input_hashes["protocol_sha256"],
        "split_sha256": input_hashes["split_sha256"],
    }
    prediction_rows, labels, probabilities = predict_loader(
        model, validation_loader, device, prediction_metadata
    )
    validation_metrics = write_prediction_evaluation(
        output,
        "validation_predictions.csv",
        "validation_metrics.json",
        prediction_rows,
        labels,
        probabilities,
        class_names,
    )
    history_fields = tuple(history[0].keys())
    write_csv(output / "training_history.csv", history, history_fields)
    metadata = {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "model_name": config.model_name,
        "regime": config.regime,
        "seed": config.seed,
        "status": "completed",
        "run_kind": config.run_kind,
        "sanity_baseline_only": config.model_name == "optional_basic_cnn",
        "best_epoch": best_snapshot["epoch"],
        "optimizer_updates": global_updates,
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_metrics": validation_metrics,
        "model_metadata": model_metadata(model, spec),
        "reproducibility": reproducibility,
        "software_versions": collect_software_versions(),
        "host": platform.node(),
        "test_accessed": False,
        "test_rows_accessed": 0,
        "input_hashes": input_hashes,
    }
    write_json(output / "run_metadata.json", metadata)
    progress_path.unlink(missing_ok=True)
    progress_path.with_suffix(f"{progress_path.suffix}.sha256").unlink(missing_ok=True)
    return metadata
