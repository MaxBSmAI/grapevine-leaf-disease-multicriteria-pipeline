# Reproducibility policy

Every future run must record validated configuration, deterministic experiment
ID, software and hardware versions, seed, RNG state, dataset and split hashes,
code commit and dirty state, checkpoint hash, predictions, metrics, and logs.

The confirmatory level is `balanced`: deterministic algorithms are requested
with warnings enabled, cuDNN benchmarking is disabled, explicit generators are
used for loaders and samplers, and RNG states are checkpointed. Bitwise
identity across different hardware is not claimed.

`requirements-lock.txt` is currently an explicit Gate 1 placeholder and must
not be treated as a resolved environment. Dependency resolution and exact pins
require review before implementation or execution.

