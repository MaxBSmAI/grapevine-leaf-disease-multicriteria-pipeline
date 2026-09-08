"""Verify imports, CUDA visibility and one finite forward per architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_source_tree_to_path

REPOSITORY_ROOT = add_source_tree_to_path()

from grape_disease.models.factory import (  # noqa: E402
    CONFIRMATORY_MODELS,
    ModelSpec,
    build_model,
    model_metadata,
)
from grape_disease.utils.config import load_structured_config  # noqa: E402
from grape_disease.utils.dependencies import (  # noqa: E402
    collect_software_versions,
    require_module,
)
from grape_disease.utils.io import ensure_output_directory, write_json  # noqa: E402
from grape_disease.utils.logging import configure_logging  # noqa: E402

torch = require_module("torch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = load_structured_config(args.config)
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the local pipeline configuration")
    if args.dry_run:
        report: dict[str, Any] = {
            "dry_run": True,
            "device": str(device),
            "architectures": list(CONFIRMATORY_MODELS),
            "forwards_executed": 0,
        }
    else:
        rows: list[dict[str, Any]] = []
        for model_name in CONFIRMATORY_MODELS:
            spec = ModelSpec(
                name=model_name,
                architecture_config=(
                    "configs/models/yolov8n_cls.yaml"
                    if model_name == "yolov8n_cls"
                    else None
                ),
            )
            model = build_model(spec, REPOSITORY_ROOT).to(device).eval()
            inputs = torch.randn(1, 3, 224, 224, device=device)
            with torch.inference_mode():
                logits = model(inputs)
            rows.append(
                {
                    "model_name": model_name,
                    "logits_shape": list(logits.shape),
                    "finite_logits": bool(torch.isfinite(logits).all().item()),
                    **model_metadata(model, spec),
                }
            )
            del model, inputs, logits
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report = {
            "schema_version": "1.0.0",
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
            "architectures": rows,
            "passed": all(
                row["logits_shape"] == [1, 4] and row["finite_logits"] for row in rows
            ),
            "software_versions": collect_software_versions(),
            "test_accessed": False,
        }
        output = ensure_output_directory(args.output_dir)
        write_json(output / "environment_verification.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
