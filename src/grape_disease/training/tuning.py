"""Validation-only Optuna tuning and architecture-specific focal gamma selection."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from grape_disease.training.runner import TrainingRunConfig, run_training
from grape_disease.utils.dependencies import require_module
from grape_disease.utils.io import (
    ensure_output_directory,
    write_csv,
    write_json,
    write_text,
)


def _suggest_parameters(trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
    optimizers = list(search_space["optimizers"])
    optimizer = trial.suggest_categorical("optimizer", optimizers)
    learning_rate_space = search_space["learning_rate"]
    if isinstance(learning_rate_space, dict):
        learning_rates = learning_rate_space[optimizer]
        learning_rate_name = f"learning_rate__{optimizer.lower()}"
    else:
        learning_rates = learning_rate_space
        learning_rate_name = "learning_rate"
    return {
        "optimizer": optimizer,
        "learning_rate": trial.suggest_categorical(
            learning_rate_name, list(learning_rates)
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", list(search_space["weight_decay"])
        ),
        "momentum": float(search_space.get("momentum", {}).get(optimizer, 0.9)),
    }


def _resolved_trial_parameters(
    trial: Any, search_space: dict[str, Any]
) -> dict[str, Any]:
    """Convert conditional Optuna parameter names into training parameters."""

    optimizer = str(trial.params["optimizer"])
    conditional_name = f"learning_rate__{optimizer.lower()}"
    learning_rate = trial.params.get(
        conditional_name, trial.params.get("learning_rate")
    )
    if learning_rate is None:
        raise RuntimeError(f"Trial {trial.number} has no resolved learning rate")
    return {
        "optimizer": optimizer,
        "learning_rate": float(learning_rate),
        "weight_decay": float(trial.params["weight_decay"]),
        "momentum": float(search_space.get("momentum", {}).get(optimizer, 0.9)),
    }


def _tuning_rows(output: Path, architecture: str) -> list[dict[str, Any]]:
    """Rebuild the durable run journal from completed seed directories."""

    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(output.glob("trials/trial_*/seed_*/run_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config_path = metadata_path.with_name("run_config.yaml")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        trial_number = int(metadata_path.parents[1].name.removeprefix("trial_"))
        rows.append(
            {
                "architecture": architecture,
                "trial": trial_number,
                "seed": int(config["seed"]),
                "optimizer": config["optimizer"],
                "learning_rate": float(config["learning_rate"]),
                "weight_decay": float(config["weight_decay"]),
                "momentum": float(config["momentum"]),
                "validation_macro_f1": metadata["validation_metrics"]["macro_f1"],
                "validation_loss": metadata["validation_metrics"]["log_loss"],
                "experiment_id": metadata["experiment_id"],
                "status": metadata["status"],
                "test_accessed": bool(metadata["test_accessed"]),
            }
        )
    return rows


def _write_tuning_journal(output: Path, architecture: str) -> list[dict[str, Any]]:
    rows = _tuning_rows(output, architecture)
    if rows:
        write_csv(output / "tuning_run_journal.csv", rows, tuple(rows[0].keys()))
    return rows


def run_architecture_tuning(
    repository_root: Path,
    dataset_root: Path,
    split_path: Path,
    normalisation_path: Path,
    protocol_path: Path,
    output_dir: Path,
    architecture: str,
    search_space: dict[str, Any],
    base_training: dict[str, Any],
    tuning_seeds: list[int],
    trials: int,
    sampler_seed: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Tune Standard CE using the mean validation macro-F1 across two seeds."""

    if len(tuning_seeds) != 2:
        raise ValueError("The frozen tuning design requires exactly two seeds")
    if dry_run:
        return {
            "dry_run": True,
            "architecture": architecture,
            "trials": trials,
            "seeds": tuning_seeds,
            "test_accessed": False,
        }
    optuna = require_module("optuna")
    output = ensure_output_directory(output_dir)

    def objective(trial: Any) -> float:
        selected = _suggest_parameters(trial, search_space)
        seed_scores: list[float] = []
        for seed_index, seed in enumerate(tuning_seeds):
            payload = {
                **base_training,
                **selected,
                "model_name": architecture,
                "regime": "standard_ce",
                "seed": seed,
                "focal_gamma": None,
                "run_kind": "tuning",
            }
            run_config = TrainingRunConfig.from_mapping(payload)
            run_output = (
                output / "trials" / f"trial_{trial.number:03d}" / f"seed_{seed}"
            )
            metadata = run_training(
                repository_root,
                dataset_root,
                split_path,
                normalisation_path,
                protocol_path,
                run_output,
                run_config,
            )
            score = float(metadata["validation_metrics"]["macro_f1"])
            seed_scores.append(score)
            _write_tuning_journal(output, architecture)
            trial.report(float(statistics.fmean(seed_scores)), step=seed_index)
        return float(statistics.fmean(seed_scores))

    storage_path = (output / "optuna_study.sqlite3").resolve()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        pruner=optuna.pruners.NopPruner(),
        study_name=f"{architecture}_standard_ce",
        storage=f"sqlite:///{storage_path.as_posix()}",
        load_if_exists=True,
    )
    for stale in study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
    ):
        study._storage.set_trial_state_values(
            stale._trial_id,
            state=optuna.trial.TrialState.WAITING,
        )
    completed_before = len(
        study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    )
    remaining_trials = max(0, trials - completed_before)
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials)
    completed_trials = study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    )
    completed_numbers = {trial.number for trial in completed_trials}
    journal_rows = _write_tuning_journal(output, architecture)
    registry_rows = [row for row in journal_rows if row["trial"] in completed_numbers]
    if not registry_rows:
        raise RuntimeError("Tuning produced no completed seed execution")
    if len(completed_trials) < trials:
        raise RuntimeError(
            f"Tuning completed {len(completed_trials)} of {trials} required trials"
        )
    registry_fields = tuple(registry_rows[0].keys())
    write_csv(output / "tuning_registry.csv", registry_rows, registry_fields)
    write_csv(output / "trial_metrics.csv", registry_rows, registry_fields)
    best_parameters = _resolved_trial_parameters(study.best_trial, search_space)
    selected_hyperparameters = {
        architecture: {
            **best_parameters,
            "mean_validation_macro_f1": study.best_value,
            "trial": study.best_trial.number,
        }
    }
    write_text(
        output / "selected_hyperparameters.yaml",
        yaml.safe_dump(selected_hyperparameters, sort_keys=True),
    )
    write_text(
        output / "search_space.yaml", yaml.safe_dump(search_space, sort_keys=True)
    )
    report = {
        "architecture": architecture,
        "completed_trials": len(completed_trials),
        "failed_or_interrupted_trials": len(study.trials) - len(completed_trials),
        "persistent_storage": storage_path.name,
        "best_trial": study.best_trial.number,
        "best_mean_validation_macro_f1": study.best_value,
        "best_parameters": best_parameters,
        "selection_partition": "validation",
        "test_accessed": False,
    }
    write_json(output / "tuning_report.json", report)
    write_text(
        output / "tuning_report.md",
        "# Tuning report\n\n"
        f"- Architecture: {architecture}\n"
        f"- Best trial: {study.best_trial.number}\n"
        f"- Mean validation macro-F1: {study.best_value}\n"
        "- Test accessed: false\n",
    )
    return report


def tune_focal_gamma(
    repository_root: Path,
    dataset_root: Path,
    split_path: Path,
    normalisation_path: Path,
    protocol_path: Path,
    output_dir: Path,
    architecture: str,
    selected_hyperparameters: dict[str, Any],
    base_training: dict[str, Any],
    tuning_seeds: list[int],
    gamma_candidates: list[float],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Select focal gamma using the frozen four-way candidates and tie breaks."""

    if dry_run:
        return {
            "dry_run": True,
            "architecture": architecture,
            "gamma_candidates": gamma_candidates,
            "seeds": tuning_seeds,
        }
    output = ensure_output_directory(output_dir)
    rows: list[dict[str, Any]] = []
    for gamma in gamma_candidates:
        for seed in tuning_seeds:
            payload = {
                **base_training,
                **selected_hyperparameters,
                "model_name": architecture,
                "regime": "focal_loss",
                "seed": seed,
                "focal_gamma": gamma,
                "run_kind": "focal_gamma_tuning",
            }
            if payload.get("gradient_clip_norm") is None:
                payload["gradient_clip_norm"] = 1.0
            payload["learning_rate"] = float(payload["learning_rate"]) * 0.01
            payload.pop("mean_validation_macro_f1", None)
            payload.pop("trial", None)
            run_config = TrainingRunConfig.from_mapping(payload)
            run_output = output / "gamma_runs" / f"gamma_{gamma}" / f"seed_{seed}"
            metadata = run_training(
                repository_root,
                dataset_root,
                split_path,
                normalisation_path,
                protocol_path,
                run_output,
                run_config,
            )
            rows.append(
                {
                    "architecture": architecture,
                    "gamma": gamma,
                    "seed": seed,
                    "validation_macro_f1": metadata["validation_metrics"]["macro_f1"],
                    "validation_loss": metadata["validation_metrics"]["log_loss"],
                    "experiment_id": metadata["experiment_id"],
                    "test_accessed": False,
                }
            )
            write_csv(
                output / "focal_gamma_run_journal.csv",
                rows,
                tuple(rows[0].keys()),
            )
    aggregates: list[dict[str, Any]] = []
    for gamma in gamma_candidates:
        gamma_rows = [row for row in rows if row["gamma"] == gamma]
        scores = [float(row["validation_macro_f1"]) for row in gamma_rows]
        losses = [float(row["validation_loss"]) for row in gamma_rows]
        aggregates.append(
            {
                "gamma": gamma,
                "mean_validation_macro_f1": statistics.fmean(scores),
                "std_validation_macro_f1": statistics.stdev(scores),
                "mean_validation_loss": statistics.fmean(losses),
            }
        )
    selected = min(
        aggregates,
        key=lambda row: (
            -row["mean_validation_macro_f1"],
            row["std_validation_macro_f1"],
            row["mean_validation_loss"],
            row["gamma"],
        ),
    )
    write_csv(
        output / "focal_gamma_selection.csv",
        rows,
        tuple(rows[0].keys()),
    )
    result = {
        "architecture": architecture,
        "selected_gamma": selected["gamma"],
        "tie_breaks": [
            "highest_mean_validation_macro_f1",
            "lowest_between_seed_standard_deviation",
            "lowest_validation_loss",
            "lowest_gamma",
        ],
        "aggregates": aggregates,
        "test_accessed": False,
    }
    write_json(output / "selected_focal_gamma.json", result)
    return result


def load_selected_hyperparameters(path: Path, architecture: str) -> dict[str, Any]:
    """Load one architecture mapping from the selected YAML artefact."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get(architecture), dict):
        raise ValueError(f"Selected hyperparameters missing {architecture}: {path}")
    return dict(value[architecture])
