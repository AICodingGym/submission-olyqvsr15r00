# Approved A, B, C, H experiments

User approved full-data training (A), lower-learning-rate continuation retaining
all three architectures (B), a separate masked-loss experiment (C), and leaderboard
submission (H).

## Controlled continuation protocol

All branches start independently from the SAME nine best 20k-breath checkpoints:
three distinct seeds each for BiGRU, BiLSTM and noncausal TCN. They preserve parent
optimizer and random-generator state, and retain the original 500 validation breaths.
Within a seed, B and C have identical starting weights and batch ordering.
Normalization is frozen from the parent training data to avoid changing the model's
input coordinate system. No validation labels are used for normalization.

- B: 20,000 total breaths, all-row MSE, LR 0.0003.
- C: same as B except inspiratory-only MAE. This changes both error norm and phase
  weighting together; it does not distinguish which of those two changes helps.
- A: same as B except all 67,905 available breaths, with the original 500 excluded
  from training (67,405 training breaths). A is warm-started, not trained from scratch.
- Each branch: at most 10 additional epochs, patience 5; retain parent model if
  continuation fails to improve validation. Epoch 0 denotes the parent checkpoint.
- Equal epoch limits do not mean equal optimizer updates or wall-clock budgets when
  data size changes. Interpret A as a practical expanded-data continuation comparison.
- Best/latest checkpoints, per-epoch histories, elapsed time, validation predictions,
  fixed mean/median ensembles and equal-weight cross-architecture ensembles saved.
- Repeated use of the holdout makes these exploratory results, not an unbiased test.

## Submission H

Current six-model LSTM+TCN mean, local MAE 0.709342. The CSV contains 603,600 finite
predictions with exact sample-submission IDs. Verified on both DataHub and locally.
Only the CSV is uploaded via AI Coding Gym, using --no-upload-logs and an isolated
workspace directory. No original Kaggle data or pretrained weights are used.

## Checks

Syntax checks passed. Masked-loss gradient check confirms expiratory outputs receive
zero direct gradient. Every checkpoint is reevaluated against its stored score;
ensemble evaluation asserts identical targets, IDs and masks across all runs.

Submission H succeeded: CLI returned score **0.72202**. Receipt saved in
`approved_abch_experiment/submission_receipt.json`.
