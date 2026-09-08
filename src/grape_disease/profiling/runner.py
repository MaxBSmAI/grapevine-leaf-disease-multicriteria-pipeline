"""CUDA-event latency and throughput profiling for frozen checkpoints."""

from __future__ import annotations

import math
import platform
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from grape_disease.models.factory import ModelSpec, build_model, load_model_state_strict
from grape_disease.utils.dependencies import collect_software_versions, require_module
from grape_disease.utils.hashing import sha256_file
from grape_disease.utils.io import ensure_output_directory, write_csv, write_json

torch = require_module("torch")


@dataclass(frozen=True)
class ProfilingConfig:
    """Frozen hardware-profiling design."""

    precisions: tuple[str, ...] = ("fp32", "fp16")
    batch_sizes: tuple[int, ...] = (1, 16)
    warmup_iterations: int = 50
    measured_iterations: int = 500
    independent_repetitions: int = 5
    execution_mode: str = "eager"
    device: str = "cuda"

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ProfilingConfig:
        return cls(
            precisions=tuple(str(item) for item in value["precisions"]),
            batch_sizes=tuple(int(item) for item in value["batch_sizes"]),
            warmup_iterations=int(value["warmup_iterations"]),
            measured_iterations=int(value["measured_iterations"]),
            independent_repetitions=int(value["independent_repetitions"]),
            execution_mode=str(value.get("execution_mode", "eager")),
            device=str(value.get("device", "cuda")),
        )

    def validate(self) -> None:
        if self.device != "cuda" or self.execution_mode != "eager":
            raise ValueError("The frozen protocol requires eager CUDA profiling")
        if set(self.precisions) != {"fp32", "fp16"}:
            raise ValueError("Profiling precisions must be fp32 and fp16")
        if any(value <= 0 for value in (*self.batch_sizes, self.measured_iterations)):
            raise ValueError("Profiling counts and batch sizes must be positive")


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def profile_checkpoint(
    repository_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    config: ProfilingConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Profile a checkpoint without reading train, validation, or test images."""

    config.validate()
    if dry_run:
        return {
            "dry_run": True,
            "checkpoint": str(checkpoint_path),
            "planned_measurements": len(config.precisions)
            * len(config.batch_sizes)
            * config.independent_repetitions
            * config.measured_iterations,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling was requested but CUDA is unavailable")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = ModelSpec(**checkpoint["model_config"])
    model = build_model(spec, repository_root).cuda().eval()
    load_model_state_strict(model, checkpoint["model_state_dict"])
    raw_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for precision in config.precisions:
        amp_enabled = precision == "fp16"
        for batch_size in config.batch_sizes:
            inputs = torch.randn(batch_size, 3, *spec.input_size, device="cuda")
            for _ in range(config.warmup_iterations):
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled),
                ):
                    model(inputs)
            torch.cuda.synchronize()
            repetition_means: list[float] = []
            all_observations: list[float] = []
            peak_memory: list[int] = []
            for repetition in range(config.independent_repetitions):
                torch.cuda.reset_peak_memory_stats()
                observations: list[float] = []
                for iteration in range(config.measured_iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    with (
                        torch.inference_mode(),
                        torch.autocast(
                            "cuda", dtype=torch.float16, enabled=amp_enabled
                        ),
                    ):
                        model(inputs)
                    end.record()
                    end.synchronize()
                    elapsed_ms = float(start.elapsed_time(end))
                    observations.append(elapsed_ms)
                    all_observations.append(elapsed_ms)
                    raw_rows.append(
                        {
                            "precision": precision,
                            "batch_size": batch_size,
                            "repetition": repetition,
                            "iteration": iteration,
                            "latency_ms": elapsed_ms,
                        }
                    )
                repetition_means.append(statistics.fmean(observations))
                peak_memory.append(int(torch.cuda.max_memory_allocated()))
            summaries.append(
                {
                    "precision": precision,
                    "batch_size": batch_size,
                    "mean_latency_ms": statistics.fmean(all_observations),
                    "std_latency_ms": statistics.stdev(all_observations),
                    "std_between_repetitions_ms": statistics.stdev(repetition_means),
                    "median_latency_ms": statistics.median(all_observations),
                    "p95_latency_ms": _percentile(all_observations, 0.95),
                    "p99_latency_ms": _percentile(all_observations, 0.99),
                    "throughput_images_per_second": batch_size
                    * 1000.0
                    / statistics.fmean(all_observations),
                    "peak_memory_bytes": max(peak_memory),
                }
            )
    output = ensure_output_directory(output_dir)
    write_csv(output / "profiling_raw.csv", raw_rows, tuple(raw_rows[0]))
    write_csv(output / "profiling_summary.csv", summaries, tuple(summaries[0]))
    metadata = {
        "schema_version": "1.0.0",
        "config": asdict(config),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "software_versions": collect_software_versions(),
    }
    write_json(output / "profiling_metadata.json", metadata)
    return metadata
