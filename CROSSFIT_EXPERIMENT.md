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
