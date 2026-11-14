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

## Completed results

All branches used the exact original 500 held-out breaths and 15,177 scored rows.
All saved checkpoint scores reproduced; targets, IDs, masks and finite predictions
were verified locally after downloading the artifacts.

| Branch | GRU mean | LSTM mean | TCN mean | LSTM+TCN mean | Recorded training/validation minutes |
|---|---:|---:|---:|---:|---:|
| B | 0.694126 | 0.691858 | 0.661838 | 0.654759 | 4.56 |
| C | 0.682594 | 0.679270 | 0.646022 | 0.640911 | 5.11 |
| A | 0.644822 | 0.643859 | 0.620099 | 0.611418 | 15.62 |

C improved all nine paired best-checkpoint scores versus B. Its LSTM+TCN ensemble
improved 2.1% over B, supporting inspiratory MAE on this reused validation set.
A had the lowest MAE, 0.611418 (13.8% below the original 0.709342). All-model mean
was nearly tied at 0.612495. GRU remains retained in every experiment.
A and C were not combined; full-data masked-MAE training has not been tested.
A uses more optimizer updates than B for the same epoch count, so this does not
isolate dataset size from total training exposure.
All checkpoint artifacts are saved locally and in the DataHub home directory.

## Leaderboard outcomes

| Submission | Local MAE | AI Coding Gym MAE |
|---|---:|---:|
| Original 20k LSTM+TCN mean | 0.709342 | 0.72202 |
| A full-data LSTM+TCN mean | 0.611418 | 0.61766 |

Both uploads succeeded with session-log uploads disabled. Improved CSV and receipts
are in approved_abch_experiment/. No further training is running from this suite;
the GPU pod remains allocated.
