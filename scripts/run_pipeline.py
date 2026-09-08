"""Inspect or execute the resumable gated experimental pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from _bootstrap import add_source_tree_to_path

REPOSITORY_ROOT = add_source_tree_to_path()

from _entrypoints import (  # noqa: E402
    _confirmatory_config,
    _frozen_input_paths,
    _guard_paths,
    _paths,
)
from grape_disease.utils.config import load_structured_config  # noqa: E402
from grape_disease.utils.gates import (  # noqa: E402
    load_gate_approvals,
    require_gate_approval,
)
from grape_disease.utils.io import write_json, write_text  # noqa: E402
from grape_disease.utils.logging import configure_logging  # noqa: E402
from grape_disease.utils.provenance import resolve_code_version  # noqa: E402

ARCHITECTURES = ("resnet50", "vit_b_16", "swin_tiny", "yolov8n_cls")
REGIMES = ("standard_ce", "focal_loss", "balanced_ce")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("status", "tuning", "functional", "confirmatory"),
        default="status",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _status(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    gates = load_gate_approvals(paths["gate_approvals"])["gates"]
    artefacts = {
        "confirmed_groups": paths["confirmed_groups"].is_file(),
        "canonical_split": paths["split"].is_file(),
        "train_normalisation": paths["normalisation"].is_file(),
        "selected_hyperparameters": paths["selected_hyperparameters"].is_file(),
        "selected_focal_gamma": paths["focal_gamma"].is_file(),
        "frozen_experiment": paths["frozen_experiment"].is_file(),
        "final_config": paths["final_config"].is_file(),
        "xai_sample_manifest": paths["xai_sample_manifest"].is_file(),
    }
    pending_gates = [
        name for name, value in gates.items() if value.get("approved") is not True
    ]
    return {
        "schema_version": "1.0.0",
        "gates": gates,
        "artefacts": artefacts,
        "first_pending_gate": pending_gates[0] if pending_gates else None,
        "test_access": json.loads(paths["test_lock"].read_text(encoding="utf-8")).get(
            "test_access", False
        ),
        "planned_confirmatory_runs": len(ARCHITECTURES)
        * len(REGIMES)
        * len(config["confirmatory"]["seeds"]),
    }


def _run_tuning(
    config: dict[str, Any], paths: dict[str, Path], output_dir: Path, dry_run: bool
) -> dict[str, Any]:
    from grape_disease.training.tuning import (
        run_architecture_tuning,
        tune_focal_gamma,
    )

    require_gate_approval(paths["gate_approvals"], 6)
    protocol = load_structured_config(paths["protocol"])
    root = output_dir / "tuning"
    code_version = resolve_code_version(
        REPOSITORY_ROOT, str(config["tuning"]["base_training"]["code_commit"])
    )
    selected_all: dict[str, Any] = {}
    tuning_reports: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        base = dict(config["tuning"]["base_training"])
        base["code_commit"] = code_version
        if architecture == "yolov8n_cls":
            base["architecture_config"] = "configs/models/yolov8n_cls.yaml"
        architecture_output = root / architecture
        report = run_architecture_tuning(
            REPOSITORY_ROOT,
            paths["dataset_root"],
            paths["split"],
            paths["normalisation"],
            paths["protocol"],
            architecture_output,
            architecture,
            protocol["tuning"]["search_spaces"][architecture],
            base,
            [int(seed) for seed in config["tuning"]["seeds"]],
            int(config["tuning"]["trials_per_architecture"]),
            int(config["tuning"]["sampler_seed"]),
            dry_run,
        )
        tuning_reports[architecture] = report
        if not dry_run:
            selected = load_structured_config(
                architecture_output / "selected_hyperparameters.yaml"
            )
            selected_all[architecture] = selected[architecture]
    if dry_run:
        return {"dry_run": True, "architectures": tuning_reports}
    write_text(
        root / "selected_hyperparameters.yaml",
        __import__("yaml").safe_dump(selected_all, sort_keys=True),
    )
    focal_results: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        base = dict(config["tuning"]["base_training"])
        base["code_commit"] = code_version
        if architecture == "yolov8n_cls":
            base["architecture_config"] = "configs/models/yolov8n_cls.yaml"
        focal_results[architecture] = tune_focal_gamma(
            REPOSITORY_ROOT,
            paths["dataset_root"],
            paths["split"],
            paths["normalisation"],
            paths["protocol"],
            root / "focal" / architecture,
            architecture,
            selected_all[architecture],
            base,
            [int(seed) for seed in config["tuning"]["seeds"]],
            [float(value) for value in protocol["focal_loss"]["gamma_candidates"]],
        )
    write_json(root / "selected_focal_gamma.json", {"architectures": focal_results})
    result = {
        "tuning": tuning_reports,
        "focal_gamma": focal_results,
        "test_accessed": False,
    }
    write_json(root / "complete_tuning_summary.json", result)
    return result


def _run_confirmatory(
    config: dict[str, Any], paths: dict[str, Path], output_dir: Path, dry_run: bool
) -> dict[str, Any]:
    from grape_disease.evaluation.guard import require_frozen_test_evaluation
    from grape_disease.evaluation.runner import evaluate_checkpoint
    from grape_disease.training.freeze import verify_frozen_inputs
    from grape_disease.training.runner import run_training

    require_gate_approval(paths["gate_approvals"], 9)
    verify_frozen_inputs(paths["frozen_experiment"], _frozen_input_paths(paths))
    require_frozen_test_evaluation(**_guard_paths(paths))
    planned = [
        (architecture, regime, int(seed))
        for architecture in ARCHITECTURES
        for regime in REGIMES
        for seed in config["confirmatory"]["seeds"]
    ]
    if dry_run:
        return {
            "dry_run": True,
            "planned_runs": len(planned),
            "runs": [
                f"{model}/{regime}/seed_{seed}" for model, regime, seed in planned
            ],
        }
    completed: list[str] = []
    skipped: list[str] = []
    for architecture, regime, seed in planned:
        run_dir = output_dir / "confirmatory" / architecture / regime / f"seed_{seed}"
        run_id = f"{architecture}/{regime}/seed_{seed}"
        if not (run_dir / "run_metadata.json").is_file():
            run_training(
                REPOSITORY_ROOT,
                paths["dataset_root"],
                paths["split"],
                paths["normalisation"],
                paths["protocol"],
                run_dir,
                _confirmatory_config(config, paths, architecture, regime, seed),
            )
        if not (run_dir / "predictions_test.csv").is_file():
            checkpoints = list(run_dir.glob("*.pth"))
            if len(checkpoints) != 1:
                raise RuntimeError(f"Expected one checkpoint in {run_dir}")
            evaluate_checkpoint(
                REPOSITORY_ROOT,
                paths["dataset_root"],
                paths["split"],
                paths["normalisation"],
                checkpoints[0],
                run_dir,
                "test",
                int(config["confirmatory"]["batch_size"]),
                str(config["device"]),
                _guard_paths(paths),
            )
            completed.append(run_id)
        else:
            skipped.append(run_id)
    result = {
        "planned_runs": len(planned),
        "newly_completed": completed,
        "already_completed": skipped,
        "all_test_predictions_present": len(completed) + len(skipped) == len(planned),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "confirmatory_execution_summary.json", result)
    return result


def _run_functional_matrix(
    config: dict[str, Any], paths: dict[str, Path], output_dir: Path, dry_run: bool
) -> dict[str, Any]:
    """Exercise the full train/validation path for all 12 combinations."""

    from grape_disease.training.freeze import verify_frozen_inputs
    from grape_disease.training.runner import run_training

    require_gate_approval(paths["gate_approvals"], 7)
    if not paths["frozen_experiment"].is_file():
        raise FileNotFoundError("Freeze the experiment before functional validation")
    verify_frozen_inputs(paths["frozen_experiment"], _frozen_input_paths(paths))
    functional = config["functional_validation"]
    planned = [
        (architecture, regime) for architecture in ARCHITECTURES for regime in REGIMES
    ]
    if dry_run:
        return {
            "dry_run": True,
            "planned_combinations": len(planned),
            "test_accessed": False,
        }
    rows: list[dict[str, Any]] = []
    for architecture, regime in planned:
        run_dir = output_dir / "functional" / architecture / regime
        if (run_dir / "run_metadata.json").is_file():
            metadata = json.loads(
                (run_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
        else:
            base = _confirmatory_config(
                config,
                paths,
                architecture,
                regime,
                int(functional["seed"]),
            )
            run_config = replace(
                base,
                seed=int(functional["seed"]),
                batch_size=int(functional["batch_size"]),
                effective_batch_size=int(functional["effective_batch_size"]),
                gradient_accumulation_steps=int(
                    functional["gradient_accumulation_steps"]
                ),
                max_optimizer_updates=int(functional["max_optimizer_updates"]),
                warmup_updates=int(functional["warmup_updates"]),
                early_stopping_patience_checks=int(
                    functional["early_stopping_patience_checks"]
                ),
                run_kind=str(functional["run_kind"]),
            )
            metadata = run_training(
                REPOSITORY_ROOT,
                paths["dataset_root"],
                paths["split"],
                paths["normalisation"],
                paths["protocol"],
                run_dir,
                run_config,
            )
        rows.append(
            {
                "model_name": architecture,
                "regime": regime,
                "experiment_id": metadata["experiment_id"],
                "status": metadata["status"],
                "validation_macro_f1": metadata["validation_metrics"]["macro_f1"],
                "test_accessed": metadata["test_accessed"],
            }
        )
    result = {
        "schema_version": "1.0.0",
        "combinations": rows,
        "passed": len(rows) == 12
        and all(row["status"] == "completed" for row in rows)
        and not any(row["test_accessed"] for row in rows),
        "publication_eligible": False,
        "test_accessed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "functional_matrix_report.json", result)
    return result


def main() -> int:
    args = _parser().parse_args()
    configure_logging(args.log_level)
    config = load_structured_config(args.config)
    paths = _paths(REPOSITORY_ROOT, config)
    if args.stage == "status":
        result = _status(config, paths)
    elif args.stage == "tuning":
        result = _run_tuning(config, paths, args.output_dir, args.dry_run)
    elif args.stage == "functional":
        result = _run_functional_matrix(config, paths, args.output_dir, args.dry_run)
    else:
        result = _run_confirmatory(config, paths, args.output_dir, args.dry_run)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
