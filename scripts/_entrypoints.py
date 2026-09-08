"""Shared implementation for the thin, gate-aware command-line scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from grape_disease.utils.config import load_structured_config
from grape_disease.utils.gates import require_gate_approval
from grape_disease.utils.logging import configure_logging


def _parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Execute project stage: {stage}")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--architecture")
    parser.add_argument("--regime")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--partition", choices=("validation", "test"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--attribution-registry", type=Path)
    parser.add_argument("--attribution-root", type=Path)
    parser.add_argument("--method-name")
    parser.add_argument("--results-root", type=Path)
    return parser


def _path(repository_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _paths(repository_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    return {
        name: _path(repository_root, value) for name, value in config["paths"].items()
    }


def _gate(paths: dict[str, Path], number: int) -> None:
    require_gate_approval(paths["gate_approvals"], number)


def _required(value: Any, name: str) -> Any:
    if value is None or value == "":
        raise ValueError(f"--{name.replace('_', '-')} is required for this stage")
    return value


def _guard_paths(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "test_lock_path": paths["test_lock"],
        "frozen_experiment_path": paths["frozen_experiment"],
        "manifest_path": paths["manifest"],
        "split_path": paths["split"],
        "protocol_path": paths["protocol"],
        "hypotheses_path": paths["hypotheses"],
        "final_config_path": paths["final_config"],
    }


def _frozen_input_paths(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "protocol": paths["protocol"],
        "manifest": paths["manifest"],
        "split": paths["split"],
        "normalisation": paths["normalisation"],
        "selected_hyperparameters": paths["selected_hyperparameters"],
        "focal_gamma": paths["focal_gamma"],
        "hypotheses": paths["hypotheses"],
        "xai_config": paths["xai_config"],
        "profiling_config": paths["profiling_config"],
        "statistics_config": paths["statistics_config"],
        "final_config": paths["final_config"],
    }


def _verify_frozen(paths: dict[str, Path]) -> None:
    from grape_disease.training.freeze import verify_frozen_inputs

    verify_frozen_inputs(paths["frozen_experiment"], _frozen_input_paths(paths))


def _consolidate(
    repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.data.group_review import (
        GroupReviewConfig,
        validate_and_consolidate_groups,
    )

    group_config = GroupReviewConfig.from_mapping(
        load_structured_config(_path(repository_root, config["group_review_config"]))
    )
    return validate_and_consolidate_groups(
        paths["manifest"],
        paths["similarity_candidates"],
        paths["review_decisions"],
        args.output_dir,
        group_config,
        args.dry_run,
    )


def _split(
    repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.data.split import SplitConfig, create_canonical_split

    _gate(paths, 3)
    split_config = SplitConfig.from_mapping(
        load_structured_config(_path(repository_root, config["split_config"]))
    )
    if split_config.approval_status not in {"approved_gate_3", "approved"}:
        raise PermissionError("Split implementation parameters remain pending Gate 3")
    return create_canonical_split(
        paths["manifest"],
        paths["confirmed_groups"],
        args.output_dir,
        split_config,
        args.dry_run,
    )


def _normalisation(
    _repository_root: Path,
    _config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.data.normalisation import calculate_train_normalisation

    _gate(paths, 4)
    return calculate_train_normalisation(
        paths["dataset_root"],
        paths["split"],
        paths["manifest"],
        args.output_dir,
        args.dry_run,
    )


def _smoke(
    repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.training.smoke import run_synthetic_smoke_test

    _gate(paths, 4)
    return run_synthetic_smoke_test(
        repository_root,
        args.output_dir,
        str(config.get("device", "cuda")),
        args.dry_run,
    )


def _tune(
    repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.training.tuning import run_architecture_tuning

    _gate(paths, 6)
    architecture = str(_required(args.architecture, "architecture"))
    protocol = load_structured_config(paths["protocol"])
    tuning = config["tuning"]
    base = dict(tuning["base_training"])
    if architecture == "yolov8n_cls":
        base["architecture_config"] = "configs/models/yolov8n_cls.yaml"
    return run_architecture_tuning(
        repository_root,
        paths["dataset_root"],
        paths["split"],
        paths["normalisation"],
        paths["protocol"],
        args.output_dir,
        architecture,
        protocol["tuning"]["search_spaces"][architecture],
        base,
        [int(seed) for seed in tuning["seeds"]],
        int(tuning["trials_per_architecture"]),
        int(tuning["sampler_seed"]),
        args.dry_run,
    )


def _focal(
    repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.training.tuning import (
        load_selected_hyperparameters,
        tune_focal_gamma,
    )

    _gate(paths, 6)
    architecture = str(_required(args.architecture, "architecture"))
    protocol = load_structured_config(paths["protocol"])
    base = dict(config["tuning"]["base_training"])
    if architecture == "yolov8n_cls":
        base["architecture_config"] = "configs/models/yolov8n_cls.yaml"
    selected = load_selected_hyperparameters(
        paths["selected_hyperparameters"], architecture
    )
    return tune_focal_gamma(
        repository_root,
        paths["dataset_root"],
        paths["split"],
        paths["normalisation"],
        paths["protocol"],
        args.output_dir,
        architecture,
        selected,
        base,
        [int(seed) for seed in config["tuning"]["seeds"]],
        [float(value) for value in protocol["focal_loss"]["gamma_candidates"]],
        args.dry_run,
    )


def _freeze(
    _repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.training.freeze import freeze_experiment

    _gate(paths, 7)
    code_commit = str(config["code_commit"])
    if code_commit.startswith("uncommitted"):
        raise ValueError("Record the immutable code commit before freezing")
    xai_config = load_structured_config(paths["xai_config"])
    if xai_config.get("sample", {}).get("selection_seed") is None:
        raise ValueError("Approve and record the XAI selection seed before freezing")
    return freeze_experiment(
        args.output_dir,
        code_commit,
        paths["protocol"],
        paths["manifest"],
        paths["split"],
        paths["normalisation"],
        paths["selected_hyperparameters"],
        paths["focal_gamma"],
        paths["hypotheses"],
        paths["xai_config"],
        paths["profiling_config"],
        paths["statistics_config"],
        [int(seed) for seed in config["confirmatory"]["seeds"]],
        args.dry_run,
    )


def _confirmatory_config(
    config: dict[str, Any],
    paths: dict[str, Path],
    architecture: str,
    regime: str,
    seed: int,
) -> Any:
    from grape_disease.training.runner import TrainingRunConfig
    from grape_disease.training.tuning import load_selected_hyperparameters

    selected = load_selected_hyperparameters(
        paths["selected_hyperparameters"], architecture
    )
    for key in ("trial", "mean_validation_macro_f1"):
        selected.pop(key, None)
    payload = {
        **config["confirmatory"],
        **selected,
        "model_name": architecture,
        "regime": regime,
        "seed": seed,
        "run_kind": "confirmatory",
        "code_commit": config["code_commit"],
        "focal_gamma": None,
    }
    payload.pop("seeds", None)
    if architecture == "yolov8n_cls":
        payload["architecture_config"] = "configs/models/yolov8n_cls.yaml"
    if regime == "focal_loss":
        focal = json.loads(paths["focal_gamma"].read_text(encoding="utf-8"))
        if "architectures" in focal:
            focal = focal["architectures"][architecture]
        elif focal.get("architecture") != architecture:
            raise ValueError(f"Focal gamma artefact lacks {architecture}")
        payload["focal_gamma"] = float(focal["selected_gamma"])
        if payload.get("gradient_clip_norm") is None:
            payload["gradient_clip_norm"] = 1.0
        payload["learning_rate"] = float(payload["learning_rate"]) * 0.01
    return TrainingRunConfig.from_mapping(payload)


def _train(
    repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.evaluation.guard import require_frozen_test_evaluation
    from grape_disease.training.runner import run_training

    _gate(paths, 9)
    _verify_frozen(paths)
    require_frozen_test_evaluation(**_guard_paths(paths))
    architecture = str(_required(args.architecture, "architecture"))
    regime = str(_required(args.regime, "regime"))
    seed = int(_required(args.seed, "seed"))
    if seed not in [int(value) for value in config["confirmatory"]["seeds"]]:
        raise ValueError("Seed is not in the frozen confirmatory seed list")
    run_config = _confirmatory_config(config, paths, architecture, regime, seed)
    return run_training(
        repository_root,
        paths["dataset_root"],
        paths["split"],
        paths["normalisation"],
        paths["protocol"],
        args.output_dir,
        run_config,
        args.dry_run,
    )


def _evaluate(
    repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
    fixed_partition: str | None = None,
) -> dict[str, Any]:
    from grape_disease.evaluation.runner import evaluate_checkpoint

    partition = fixed_partition or str(_required(args.partition, "partition"))
    _gate(paths, 9 if partition == "test" else 8)
    checkpoint = Path(_required(args.checkpoint, "checkpoint"))
    return evaluate_checkpoint(
        repository_root,
        paths["dataset_root"],
        paths["split"],
        paths["normalisation"],
        checkpoint,
        args.output_dir,
        partition,
        int(config["confirmatory"]["batch_size"]),
        str(config["device"]),
        _guard_paths(paths) if partition == "test" else None,
        args.dry_run,
    )


def _xai_sample(
    repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.xai.sample import create_xai_sample_manifest

    _gate(paths, 9)
    _verify_frozen(paths)
    sample_config = load_structured_config(paths["xai_config"])["sample"]
    return create_xai_sample_manifest(
        paths["split"],
        paths["manifest"],
        args.output_dir,
        int(sample_config["images_per_class"]),
        sample_config["selection_seed"],
        args.dry_run,
    )


def _ig(
    repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.xai.integrated_gradients import IntegratedGradientsConfig
    from grape_disease.xai.runner import run_integrated_gradients_sample

    _gate(paths, 9)
    _verify_frozen(paths)
    ig_config = IntegratedGradientsConfig.from_mapping(
        load_structured_config(
            _path(repository_root, config["integrated_gradients_config"])
        )
    )
    return run_integrated_gradients_sample(
        repository_root,
        paths["dataset_root"],
        Path(_required(args.checkpoint, "checkpoint")),
        paths["normalisation"],
        paths["xai_sample_manifest"],
        args.output_dir,
        ig_config,
        _guard_paths(paths),
        str(config["device"]),
        True,
        args.dry_run,
    )


def _specific_xai(
    repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.xai.runner import run_specific_xai_sample

    _gate(paths, 9)
    _verify_frozen(paths)
    return run_specific_xai_sample(
        repository_root,
        paths["dataset_root"],
        Path(_required(args.checkpoint, "checkpoint")),
        paths["normalisation"],
        paths["xai_sample_manifest"],
        args.output_dir,
        _guard_paths(paths),
        str(config["device"]),
        args.dry_run,
    )


def _faithfulness(
    repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.xai.faithfulness import PerturbationConfig
    from grape_disease.xai.runner import run_faithfulness_sample

    _gate(paths, 9)
    _verify_frozen(paths)
    perturbation = PerturbationConfig.from_mapping(
        load_structured_config(_path(repository_root, config["perturbation_config"]))
    )
    return run_faithfulness_sample(
        repository_root,
        paths["dataset_root"],
        Path(_required(args.checkpoint, "checkpoint")),
        paths["normalisation"],
        paths["xai_sample_manifest"],
        Path(_required(args.attribution_registry, "attribution_registry")),
        Path(_required(args.attribution_root, "attribution_root")),
        args.output_dir,
        perturbation,
        _guard_paths(paths),
        str(config["device"]),
        str(_required(args.method_name, "method_name")),
        args.dry_run,
    )


def _stability(
    _repository_root: Path,
    _config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from grape_disease.utils.io import write_csv, write_json
    from grape_disease.xai.stability import compare_maps

    _gate(paths, 9)
    _verify_frozen(paths)
    root = Path(_required(args.attribution_root, "attribution_root"))
    registry_paths = sorted(root.rglob("integrated_gradients_registry.csv"))
    registry_paths += sorted(root.rglob("specific_xai_registry.csv"))
    if args.dry_run:
        return {"dry_run": True, "registries": len(registry_paths), "maps_opened": 0}
    by_key: dict[tuple[str, str, str, str], list[tuple[int, Path, str]]] = {}
    for registry_path in registry_paths:
        registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
        for row in registry.to_dict("records"):
            method_target = row.get("target_type", row.get("method", "unknown"))
            key = (
                row["model_name"],
                row["regime"],
                row["image_id"],
                method_target,
            )
            by_key.setdefault(key, []).append(
                (
                    int(row["seed"]),
                    registry_path.parent / row["map_path"],
                    row.get("target_class", row.get("predicted_class", "unknown")),
                )
            )
    rows: list[dict[str, Any]] = []
    for (model_name, regime, image_id, method_target), entries in sorted(
        by_key.items()
    ):
        ordered = sorted(entries, key=lambda value: (value[0], str(value[1])))
        for first_index, (first_seed, first, first_target) in enumerate(ordered):
            for second_seed, second, second_target in ordered[first_index + 1 :]:
                if first_seed == second_seed:
                    continue
                comparison = compare_maps(
                    np.load(first)["ranking_map"], np.load(second)["ranking_map"]
                )
                rows.append(
                    {
                        "model_name": model_name,
                        "regime": regime,
                        "image_id": image_id,
                        "method_or_target": method_target,
                        "first_seed": first_seed,
                        "second_seed": second_seed,
                        "first_target_class": first_target,
                        "second_target_class": second_target,
                        "same_target_class": first_target == second_target,
                        "first_map": str(first),
                        "second_map": str(second),
                        **comparison,
                    }
                )
    if not rows:
        raise ValueError("Stability analysis requires maps from at least two seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "xai_stability.csv", rows, tuple(rows[0]))
    result = {"comparisons": len(rows), "top_k_fraction": 0.10}
    write_json(args.output_dir / "xai_stability_summary.json", result)
    return result


def _profile(
    repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.profiling.runner import ProfilingConfig, profile_checkpoint

    _gate(paths, 9)
    _verify_frozen(paths)
    profile_config = ProfilingConfig.from_mapping(
        load_structured_config(_path(repository_root, config["profiling_config"]))
    )
    return profile_checkpoint(
        repository_root,
        Path(_required(args.checkpoint, "checkpoint")),
        args.output_dir,
        profile_config,
        args.dry_run,
    )


def _statistics(
    repository_root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.statistics.paired import PairedBootstrapConfig
    from grape_disease.statistics.runner import run_predictive_statistics

    _gate(paths, 9)
    _verify_frozen(paths)
    statistics = load_structured_config(
        _path(repository_root, config["statistics_config"])
    )
    bootstrap = PairedBootstrapConfig(
        int(statistics["bootstrap_resamples"]),
        int(statistics["bootstrap_seed"]),
        float(statistics["confidence_level"]),
    )
    return run_predictive_statistics(
        Path(_required(args.results_root, "results_root")),
        args.output_dir,
        load_structured_config(paths["hypotheses"]),
        [str(name) for name in config["classes"]],
        bootstrap,
        args.dry_run,
    )


def _aggregate(
    _repository_root: Path,
    _config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from grape_disease.reporting.aggregate import aggregate_results

    _gate(paths, 10)
    _verify_frozen(paths)
    return aggregate_results(
        Path(_required(args.results_root, "results_root")), args.output_dir
    )


STAGES = {
    "consolidate": _consolidate,
    "split": _split,
    "normalisation": _normalisation,
    "smoke": _smoke,
    "tune": _tune,
    "focal": _focal,
    "freeze": _freeze,
    "train": _train,
    "evaluate_validation": lambda root, config, paths, args: _evaluate(
        root, config, paths, args, "validation"
    ),
    "evaluate_test": lambda root, config, paths, args: _evaluate(
        root, config, paths, args, "test"
    ),
    "xai_sample": _xai_sample,
    "integrated_gradients": _ig,
    "specific_xai": _specific_xai,
    "faithfulness": _faithfulness,
    "stability": _stability,
    "profile": _profile,
    "statistics": _statistics,
    "aggregate": _aggregate,
    "publication": _aggregate,
}


def main(stage: str, repository_root: Path) -> int:
    """Parse common flags, execute a stage, and print machine-readable output."""

    args = _parser(stage).parse_args()
    configure_logging(args.log_level)
    config = load_structured_config(args.config)
    paths = _paths(repository_root, config)
    result = STAGES[stage](repository_root, config, paths, args)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    passed = result.get("passed")
    return 0 if passed is not False else 2
