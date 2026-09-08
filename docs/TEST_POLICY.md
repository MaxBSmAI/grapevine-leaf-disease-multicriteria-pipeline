# Test access policy

`protocol/TEST_LOCK.json` is the technical source of truth for test access. Its
initial state is `test_access=false`.

Development, smoke tests, tuning, debugging, checkpoint selection, XAI pilots,
and functional validation may use only train and validation. Static forensic
inspection of historical `legacy_non_publication` artefacts is allowed, but it
does not authorise new test predictions.

The implemented `evaluate_test.py`, confirmatory orchestrator, and test-based
XAI commands fail closed unless the lock, frozen experiment, manifest, split,
protocol, hypotheses, final configuration and code commitment agree by hash.
No `--force` bypass is implemented.
