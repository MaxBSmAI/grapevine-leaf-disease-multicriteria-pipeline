# Software architecture

## Design principles

The repository separates validated configuration, scientific logic,
command-line orchestration, generated artefacts, and provenance.

```text
validated YAML/JSON configs
          |
          v
thin scripts/CLI --> src/grape_disease --> versioned outputs
          |                   |
          v                   v
 protocol gates        structured provenance
```

All functional logic belongs in `src/grape_disease/`. The `scripts/` directory
will contain thin entry points. Generated artefacts must be derived from
manifests, predictions, and validated configurations rather than manual edits.

## Package boundaries

- `data`: manifest, validation, transforms, samplers, and split logic.
- `models`: a common logits-only interface for the four architectures.
- `training`: losses, loops, checkpoints, determinism, and tuning support.
- `evaluation`: individual predictions, metrics, and calibration.
- `xai`: common and architecture-specific explanation methods.
- `profiling`: model-only and end-to-end computational measurement.
- `statistics`: crossed paired resampling and multiplicity correction.
- `reporting`: publication tables and figures derived from source artefacts.
- `utils`: configuration schemas, hashing, identifiers, logging, and guards.

## Gate 1 limitation

Only the directory structure and governance documents exist. No functional
pipeline module has been implemented or migrated.

