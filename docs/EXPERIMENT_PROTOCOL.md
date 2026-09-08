# Experiment protocol

The authoritative machine-readable decisions are stored in
`protocol/FROZEN_PROTOCOL.yaml`. This document is an orientation layer and does
not override that file.

The confirmatory study consists of four architectures, three training regimes,
and five seeds. Development tuning uses separate seeds and only train and
validation. Model selection uses validation macro-F1 with predefined
tie-breaks. The primary test metric is the arithmetic mean of the five
seed-level macro-F1 values.

No test access is authorised at Gate 1. The canonical split, normalisation,
hyperparameters, XAI protocol, hypotheses, code commit, and configuration
hashes must be frozen before test access can be considered.

