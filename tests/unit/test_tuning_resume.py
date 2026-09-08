from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from grape_disease.training import tuning
from grape_disease.utils.io import write_json, write_text


class _ConditionalTrial:
    def __init__(self, optimizer: str) -> None:
        self.optimizer = optimizer
        self.names: list[str] = []

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        self.names.append(name)
        return self.optimizer if name == "optimizer" else choices[0]


def test_conditional_learning_rates_use_stable_distinct_parameter_names() -> None:
    search_space = {
        "optimizers": ["AdamW", "SGD"],
        "learning_rate": {"AdamW": [0.0001], "SGD": [0.003]},
        "weight_decay": [0.0001],
        "momentum": {"SGD": 0.9},
    }
    adamw = _ConditionalTrial("AdamW")
    sgd = _ConditionalTrial("SGD")
    assert tuning._suggest_parameters(adamw, search_space)["learning_rate"] == 0.0001
    assert tuning._suggest_parameters(sgd, search_space)["learning_rate"] == 0.003
    assert "learning_rate__adamw" in adamw.names
    assert "learning_rate__sgd" in sgd.names


def test_optuna_tuning_resumes_without_repeating_completed_runs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls: list[str] = []

    def fake_run_training(
        _repository_root: Path,
        _dataset_root: Path,
        _split_path: Path,
        _normalisation_path: Path,
        _protocol_path: Path,
        output_dir: Path,
        config: Any,
    ) -> dict[str, Any]:
        calls.append(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = config.__dict__
        write_text(output_dir / "run_config.yaml", yaml.safe_dump(payload))
        metadata = {
            "experiment_id": f"experiment-{len(calls)}",
            "status": "completed",
            "test_accessed": False,
            "validation_metrics": {"macro_f1": 0.5, "log_loss": 1.0},
        }
        write_json(output_dir / "run_metadata.json", metadata)
        return metadata

    monkeypatch.setattr(tuning, "run_training", fake_run_training)
    base_training = {
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "batch_size": 16,
        "effective_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "max_optimizer_updates": 10,
        "warmup_updates": 1,
        "early_stopping_patience_checks": 2,
        "minimum_improvement": 0.0001,
        "amp": False,
        "deterministic_level": "balanced",
        "device": "cpu",
        "num_workers": 0,
        "gradient_clip_norm": None,
        "code_commit": "tree-sha256:test",
        "architecture_config": None,
    }
    arguments = (
        tmp_path,
        tmp_path,
        tmp_path / "split.csv",
        tmp_path / "normalisation.json",
        tmp_path / "protocol.yaml",
        tmp_path / "output",
        "resnet50",
        {
            "optimizers": ["AdamW"],
            "learning_rate": [0.0001],
            "weight_decay": [0.0001],
        },
        base_training,
        [101, 202],
        2,
        9001,
    )
    first = tuning.run_architecture_tuning(*arguments)
    assert first["completed_trials"] == 2
    assert len(calls) == 4
    second = tuning.run_architecture_tuning(*arguments)
    assert second["completed_trials"] == 2
    assert len(calls) == 4
