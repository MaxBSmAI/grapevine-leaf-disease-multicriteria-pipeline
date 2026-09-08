from __future__ import annotations

import json
from pathlib import Path

from grape_disease.training.freeze import freeze_experiment, verify_frozen_inputs
from grape_disease.utils.hashing import sha256_file


def test_freeze_writes_a_hash_verifiable_final_config(tmp_path: Path) -> None:
    names = (
        "protocol",
        "manifest",
        "split",
        "normalisation",
        "selected",
        "focal",
        "hypotheses",
        "xai",
        "profiling",
        "statistics",
    )
    paths = {name: tmp_path / f"{name}.txt" for name in names}
    for name, path in paths.items():
        path.write_text(name, encoding="utf-8")
    output = tmp_path / "frozen"
    freeze_experiment(
        output,
        "abc123",
        paths["protocol"],
        paths["manifest"],
        paths["split"],
        paths["normalisation"],
        paths["selected"],
        paths["focal"],
        paths["hypotheses"],
        paths["xai"],
        paths["profiling"],
        paths["statistics"],
        [42, 1337, 2025, 31415, 27182],
    )
    frozen = json.loads((output / "FROZEN_EXPERIMENT.json").read_text())
    assert frozen["final_config_sha256"] == sha256_file(output / "FINAL_CONFIG.json")
    frozen_inputs = {
        "protocol": paths["protocol"],
        "manifest": paths["manifest"],
        "split": paths["split"],
        "normalisation": paths["normalisation"],
        "selected_hyperparameters": paths["selected"],
        "focal_gamma": paths["focal"],
        "hypotheses": paths["hypotheses"],
        "xai_config": paths["xai"],
        "profiling_config": paths["profiling"],
        "statistics_config": paths["statistics"],
        "final_config": output / "FINAL_CONFIG.json",
    }
    verified = verify_frozen_inputs(output / "FROZEN_EXPERIMENT.json", frozen_inputs)
    assert verified["code_commit"] == "abc123"
