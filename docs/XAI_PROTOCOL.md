# XAI protocol

Integrated Gradients is the common comparative method. Its principal target is
the predicted-class logit; the ground-truth-class logit is a separate secondary
analysis. The ranking map is the channel-wise L1 attribution and signed maps
are retained for complementary visualisation.

Insertion and deletion use non-overlapping 16 x 16 blocks and a Gaussian-blur
perturbation baseline. This baseline is distinct from the zero tensor in
normalised space used by Integrated Gradients. Curves cover 0 to 0.5 and must
be reported as partial AUC metrics.

The final XAI sample will contain ten test images per class selected by a
model-independent deterministic procedure before confirmatory predictions.
The 2,400 explanations are repeated measurements over seeds and groups, not
2,400 independent observations.

