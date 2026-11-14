# Out-of-fold breath-offset and timestep residual correction

User requested executing the diagnostic recommendation: investigate whole-breath
pressure offsets, particularly R=50, and later-inspiration errors. No submission
is authorized by this request; the prior new-payload upload rejection still applies.

## Leakage controls and scope

- Current reference: selected CNN-BiLSTM + MAE-TCN ensemble, local MAE 0.577805.
- Randomly select 12,000 training breaths from AFTER the first 20,000, using seed
  20261113; stratify their three cross-fitting folds by R/C.
- A separate 1,000 breaths from that later region select teacher checkpoints.
- The original 500 validation breaths are excluded from every teacher and residual
  model training set. They only evaluate correction candidates afterward.
- Initialize teachers from older 20k sequence checkpoints, NOT the current full-data
  models. Both cross-fitting targets and internal validation breaths are absent
  from those old checkpoints' training data. This is a safe warm start.
- Each fold trains CNN-BiLSTM and TCN on all training breaths EXCEPT its held-out
  cross-fitting fold and the separate internal-validation set. Other folds' labels
  may be used, as in ordinary cross-fitting.
- One LSTM seed and one TCN seed per fold (three distinct original seeds across
  folds), 10 epochs maximum, inspiratory MAE, Adam .0003, batch 51.
- Teacher checkpoint selection uses ONLY the separate 1,000-breath inner set.
- The six-model final reference differs from each two-model cross-fitting teacher;
  residual-transfer mismatch is an explicit limitation to check on validation.
- This is a 12k-target cross-fitting screen, not exhaustive OOF coverage of all 67k.

## Prespecified residual models

Train two independent histogram gradient-boosting regressors on OOF errors:
1. Breath-offset target: median(true pressure minus teacher prediction) over
   inspiratory rows. Inputs are controls, lung settings, and teacher predictions.
2. Timestep-residual target: true pressure minus teacher prediction at each
   inspiratory timestep, with local/history features and whole-breath summaries.

Both use absolute-error loss, 150 iterations, 15 leaves, L2=10, no random row-based
early-stopping split. Minimum leaf sizes: 80 breaths / 100 scored rows respectively.
Four fixed-strength candidates are compared: half-strength offset on all breaths,
half-strength offset only for R=50, half-strength row correction on all inspiratory
rows, and half-strength row correction only at steps >=10. No weights or model
parameters are fit to primary validation labels. The unchanged model is a candidate.

Report OOF subgroup/offset diagnostics separately from correction validation.
OOF teacher predictions are held-out, but fitted residual-model training error is
not a validation score. No nested-CV generalization claim is made.


## Completed results — November 13, 2026

All six teacher runs completed ten epochs; checkpoint selection retained epochs
8–10 using only the separate internal set. Total teacher epoch time was about
12.3 minutes. Corrections fitted in about 12 seconds.

The original 500-breath / 15,177-inspiratory-row validation split is unchanged:

| Candidate (fixed 0.5 correction strength) | Local validation MAE |
| --- | ---: |
| Unchanged selected ensemble | 0.57780540 |
| Whole-breath offset | 0.56833772 |
| Whole-breath offset restricted to R=50 | 0.57361000 |
| Per-timestep residual, all inspiration | **0.56740350** |
| Per-timestep residual, steps 10 onward | 0.57175689 |

Retain the per-timestep residual correction as the leading experimental candidate:
absolute improvement 0.01040190, or 1.80%. The whole-breath correction also helps,
but combining the two has not been tested and is not implied by these results.
The underlying selected neural ensemble is preserved.

### What the out-of-fold predictions confirmed

The 12,000 OOF breaths contain 364,313 scored rows. Teacher OOF MAE is 0.62937;
this is a different evaluation population and a two-model teacher per fold, so it
is not directly comparable to the selected ensemble's 500-breath MAE.

- R=50 groups have OOF MAE 0.78815 (C=10), 0.81645 (C=20), and 0.89238 (C=50).
- OOF MAE rises from 0.38071 at steps 0–4 to 0.73972 at steps 20–29 and
  0.84195 at steps 30 onward. The last group has relatively few scored rows.
- A label-dependent perfect median-offset correction would reduce OOF MAE to
  0.36971. This is only a diagnostic oracle, not an achievable model result.
- The deployable row correction improves validation steps 20–29 from 0.67620
  to 0.66709. R=50,C=50 improves modestly from 0.92520 to 0.91822.
  The largest difficult-group gain is R=50,C=10: 0.69845 to 0.67759.

### Verification and limitations

Read back all six saved best checkpoints on DataHub and verified their actual
training IDs have no overlap with their held-out fold, the common inner set, or
the primary validation set. Counts account for all 67,905 breaths. Locally checked
OOF coverage and per-fold prediction alignment, exact primary validation IDs,
truth and masks, unchanged baseline predictions, and finite candidate outputs.

No correction parameters were fitted to primary validation labels. However, choosing
among these candidates and earlier experiments reuses that validation set, so the
reported gain is exploratory and needs independent confirmation. The OOF teacher
and selected final ensemble also differ, which can limit residual transfer.
This experiment does not establish MAE 0.2 as attainable.

Detailed metrics: `crossfit_experiment/correction/comparison.json`.
Saved correction estimators: `crossfit_experiment/correction/corrections.joblib`.
Saved predictions: `crossfit_experiment/correction/validation_predictions.npz`.
Teacher checkpoints, splits and OOF predictions: `crossfit_experiment/teachers/`.
No predictions were submitted. Last verified leaderboard score remains 0.61766;
0.56740 is a local validation result only.
