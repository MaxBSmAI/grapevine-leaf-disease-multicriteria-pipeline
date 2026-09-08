# Command-line scripts

All experimental scripts accept `--config`, `--output-dir`, `--dry-run`, and
`--log-level`. Stage-specific scripts also require identifiers such as
`--architecture`, `--regime`, `--seed`, or `--checkpoint`. They fail closed if
the corresponding entry in `protocol/GATE_APPROVALS.json` is not approved.

## Orchestrator

- `run_pipeline.py --stage status`: read-only readiness and gate report.
- `verify_environment.py`: imports, CUDA visibility and one forward per model.
- `run_pipeline.py --stage tuning`: four validation-only architecture studies,
  focal-gamma selection, and merged frozen selections.
- `run_pipeline.py --stage functional`: twelve short train/validation runs,
  explicitly non-publication, as Gate-8 integration evidence.
- `run_pipeline.py --stage confirmatory`: resumable 4 × 3 × 5 execution and
  guarded per-seed test prediction generation.

## Data and gates

- `build_master_manifest.py`, `audit_originals.py`
- `identify_similarity_candidates.py`, `prioritise_similarity_review.py`
- `consolidate_confirmed_groups.py`
- `create_canonical_split.py`
- `calculate_train_normalisation.py`

## Models and evaluation

- `run_smoke_test.py`
- `tune_model.py`, `tune_focal_gamma.py`
- `freeze_experiment.py`, `train_model.py`
- `authorise_test_access.py` (requires explicit Gate 9 and confirmation flag)
- `evaluate_validation.py`, `evaluate_test.py`

## XAI, efficiency, statistics and publication

- `create_xai_manifest.py`
- `run_integrated_gradients.py`, `run_specific_xai.py`
- `run_xai_faithfulness.py`, `run_xai_stability.py`
- `profile_model.py`
- `run_paired_statistics.py`
- `aggregate_results.py`, `generate_publication_artifacts.py`

`setup_environment.ps1` creates the pinned Python environment. `run_unit_checks.py`
retains the lightweight checks, while `pytest`, Ruff, Mypy and the synthetic
12-combination smoke command provide the full implementation verification.
