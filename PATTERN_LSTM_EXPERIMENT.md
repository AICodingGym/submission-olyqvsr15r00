# Control-pattern features and larger LSTM experiment

## Prespecified comparison

Implement the approved A–C concepts independently, without copying the public
notebook implementation. Preserve the original 500 validation breaths and train
on all other 67,405 breaths. Four variants start from scratch:

| Variant | Features | Model |
| --- | --- | --- |
| control | Existing 19 | Two-layer BiLSTM, 56 units per direction |
| A | 19 + 8 control-pattern features | Same small BiLSTM |
| AB | 27 + 34 within-breath features | Same small BiLSTM |
| ABC | Same 61 | BiLSTM stages 256/128/64, concatenating input and prior outputs |

Two fresh seeds are matched across variants to control initialization and shuffle
variability; seeds differ between replicates. All variants use batch size 128,
Adam at 0.001, gradient norm clipping at 1, and inspiratory row-weighted MAE.
Reduce learning rate by half after a plateau (patience 3); stop after ten epochs
without validation improvement, with a maximum of 40 epochs. Select each run's
best validation checkpoint and also report the two-seed mean prediction.

This is a finite-budget screen, not a reproduction of the public notebook's
15-fold, 300-epoch TPU training. Neither forward/reverse pairing nor extra folds
are included. Existing models and residual corrections remain unchanged. The
old residual estimator will not be applied to a differently trained model.

## Features and leakage controls

A groups training inputs by R, C, position in breath, and rounded inlet control.
It provides mean/min/max, deviations from them, log-count and an unseen flag.
Validation observations never contribute to these tables. Unseen groups fall
back to training R/C/position statistics, then training-global statistics.

B includes forward input/phase/time shifts, input differences, time-weighted
input integral, R/C interactions, whole-breath summaries, inspiratory length,
and nine R/C indicators. Every shift and summary remains within its breath.
Full breath inputs are available at prediction time. Pressure is removed before
feature generation. Normalization statistics use training breaths only.

The validation IDs, truth and inspiratory mask must exactly match the saved
reference arrays. Record training IDs, feature names, normalizers, aggregation
tables, configurations, histories, and best checkpoints for reproducibility.

## Interpretation

Compare incremental variants against the newly trained control at the same budget.
Compare absolute scores separately against the existing six-model ensemble
(0.57780540) and its OOF-trained residual correction (0.56740350).
Repeated use of the 500-breath holdout introduces model-selection optimism;
these scores are exploratory local validation, not independent leaderboard results.
No submission is part of this experiment.
