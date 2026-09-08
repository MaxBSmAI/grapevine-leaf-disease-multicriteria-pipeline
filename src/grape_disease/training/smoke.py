"""Synthetic forward/backward smoke checks for every confirmatory combination."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grape_disease.models.factory import CONFIRMATORY_MODELS, ModelSpec, build_model
from grape_disease.training.losses import build_loss
from grape_disease.training.reproducibility import set_reproducibility
from grape_disease.utils.dependencies import require_module
from grape_disease.utils.io import ensure_output_directory, write_json

torch = require_module("torch")


def run_synthetic_smoke_test(
    repository_root: Path,
    output_dir: Path,
    device_name: str = "cuda",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Verify finite logits/loss/gradients for all 12 architecture-regime pairs."""

    combinations = [
        (model, regime)
        for model in CONFIRMATORY_MODELS
        for regime in ("standard_ce", "focal_loss", "balanced_ce")
    ]
    if dry_run:
        return {"dry_run": True, "planned_combinations": len(combinations)}
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke test requested but unavailable")
    set_reproducibility(1729, "balanced")
    rows: list[dict[str, Any]] = []
    for model_name, regime in combinations:
        spec = ModelSpec(
            name=model_name,
            architecture_config=(
                "configs/models/yolov8n_cls.yaml"
                if model_name == "yolov8n_cls"
                else None
            ),
        )
        model = build_model(spec, repository_root).to(device).train()
        inputs = torch.randn(2, 3, 224, 224, device=device)
        targets = torch.tensor([0, 3], device=device)
        logits = model(inputs)
        loss_function = build_loss(regime, 2.0 if regime == "focal_loss" else None)
        loss = loss_function(logits, targets)
        loss.backward()
        finite_gradients = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        passed = (
            tuple(logits.shape) == (2, 4)
            and bool(torch.isfinite(logits).all().item())
            and bool(torch.isfinite(loss).item())
            and finite_gradients
        )
        rows.append(
            {
                "model_name": model_name,
                "regime": regime,
                "logits_shape": list(logits.shape),
                "loss": float(loss.detach().cpu().item()),
                "finite_gradients": finite_gradients,
                "passed": passed,
            }
        )
        del model, inputs, targets, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema_version": "1.0.0",
        "device": str(device),
        "combinations": rows,
        "passed": all(row["passed"] for row in rows),
    }
    output = ensure_output_directory(output_dir)
    write_json(output / "synthetic_smoke_report.json", report)
    return report


def run_batch_memory_preflight(
    repository_root: Path,
    output_dir: Path,
    batch_size: int = 16,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Exercise one FP32 AdamW update per architecture at the physical batch size."""

    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The batch-memory preflight requires CUDA")
    set_reproducibility(1729, "balanced")
    rows: list[dict[str, Any]] = []
    for model_name in CONFIRMATORY_MODELS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        row: dict[str, Any] = {
            "model_name": model_name,
            "batch_size": batch_size,
            "precision": "fp32",
            "optimizer": "AdamW",
            "passed": False,
        }
        try:
            spec = ModelSpec(
                name=model_name,
                architecture_config=(
                    "configs/models/yolov8n_cls.yaml"
                    if model_name == "yolov8n_cls"
                    else None
                ),
            )
            model = build_model(spec, repository_root).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            inputs = torch.randn(batch_size, 3, 224, 224, device=device)
            targets = torch.arange(batch_size, device=device) % 4
            logits = model(inputs)
            loss = build_loss("standard_ce", None)(logits, targets)
            loss.backward()
            optimizer.step()
            row.update(
                {
                    "loss": float(loss.detach().cpu().item()),
                    "finite_loss": bool(torch.isfinite(loss).item()),
                    "peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "passed": bool(torch.isfinite(loss).item()),
                }
            )
            del model, optimizer, inputs, targets, logits, loss
        except torch.OutOfMemoryError as exc:
            row["error"] = f"CUDA out of memory: {exc}"
        finally:
            torch.cuda.empty_cache()
        rows.append(row)
    report = {
        "schema_version": "1.0.0",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "precision": "fp32",
        "architectures": rows,
        "passed": all(row["passed"] for row in rows),
        "test_accessed": False,
        "dataset_accessed": False,
    }
    output = ensure_output_directory(output_dir)
    write_json(output / "batch16_memory_preflight.json", report)
    return report
