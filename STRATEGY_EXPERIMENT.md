# Adaptations of the current 0.61766 leaderboard model

User requested applying the recommended strategies after reviewing:
- https://www.kaggle.com/competitions/ventilator-pressure-prediction/writeups/upstage-making-ai-beneficial-3rd-place-single-mode
- https://www.kaggle.com/competitions/ventilator-pressure-prediction/writeups/jun-koda-4th-place-solution-hacking-the-pid-contro

This experiment implements the four recommended next steps: full-data inspiratory
MAE, auxiliary pressure-change supervision, explicit categorical R/C conditioning,
and a convolutional front end to the bidirectional LSTM. These are independent
implementations inspired by the concepts, not reproductions of the winning solution.
No outside code, weights, original Kaggle data, controller constants or pressure
vocabulary are imported. PID inversion, mixup and recurrent reinitialization are
not part of this run.

## Protocol

Starting point: all three saved full-data LSTM and TCN seeds from experiment A,
whose six-model mean has local MAE 0.611418 and leaderboard MAE 0.61766.
GRU artifacts are retained; this screening focuses on the selected ensemble members.

All branches train on the same 67,405 breaths, hold out the exact original 500,
freeze training-only parent normalization, use Adam LR .0003 and batch size 51,
and run five additional epochs. Each branch has three seeds inherited from its
parents; same-seed branches share initial base weights and loader-generator state.
Adam is reset uniformly for all branches to accommodate new parameters. An MSE
control accounts for the extra training and optimizer reset.

Branches:
1. mse_control/lstm: unchanged model, all-row MSE.
2. mae/lstm and mae/tcn: unchanged models, inspiratory-only MAE.
3. delta/lstm: inspiratory MAE plus 0.2 * adjacent-pressure-difference MAE from
   a separate linear head on recurrent representations. Only pairs with both
   timesteps inspiratory are scored; no cross-breath differences.
4. categorical/lstm: add a zero-initialized projection of six R/C one-hot values
   to the normalized input. Original numeric R/C remains; this tests added
   categorical conditioning, not removal of numeric values.
5. cnn/lstm: add a residual two-convolution input front end (19 -> 32 -> 19,
   kernel 3, symmetric padding), with zero-initialized final convolution.

Delta, categorical and CNN are compared against mae/lstm, not only against the
older parent. Primary validation metric selects checkpoints; original parent
predictions are retained if no epoch improves them. No auxiliary labels are
required at inference. Equal-weight means are evaluated; no validation-fitted weights.

This is a short screening budget, not a definitive ranking of architectures.
The repeatedly used holdout remains exploratory. The current submission is preserved.

## Results

All 18 runs completed five epochs. Parent prediction identity and saved-checkpoint
MAE reproduction checks passed. Downloaded validation IDs, truth and masks exactly
match experiment A; all predictions are finite.

| LSTM branch | LSTM three-seed mean | Combined with unchanged TCN | Combined with MAE-trained TCN |
|---|---:|---:|---:|
| mse_control | 0.621467 | 0.603382 | 0.582987 |
| mae | 0.609330 | 0.595295 | 0.577942 |
| delta | 0.612764 | 0.596886 | 0.579178 |
| categorical | 0.610427 | 0.595367 | 0.577970 |
| cnn | 0.609411 | 0.595227 | 0.577805 |

Original selected ensemble: 0.611418. Best observed: CNN-LSTM + MAE-TCN,
0.577805 (5.5% lower local MAE). The unchanged MAE-LSTM + MAE-TCN ensemble
is nearly tied at 0.577942; the 0.000136 difference is not persuasive evidence
for preferring the CNN architecture. Delta and categorical variants did not beat
loss-only continuation in this short screening budget.
MSE LSTM control mean: 0.621467; MAE LSTM mean: 0.609330, so the paired control
supports MAE beyond additional training for LSTM. TCN had no new paired MSE
continuation control in this screen, so its gains cannot be attributed solely
to the loss change.
Recorded training/validation time: 1,013.95 seconds (16.9 minutes), excluding
preprocessing, checkpoint I/O, transfers and setup. The original model is preserved.

## Submission status

The selected ensemble CSV passed local and remote format checks: 603,600 rows,
exact sample-submission IDs, finite pressures, six checkpoint hashes and matching
CSV checksum. File: strategy_experiment/submission_strategies.csv.

The upload command did not run: automatic approval review rejected this new
prediction payload, requiring explicit payload-specific approval to upload it to
AI Coding Gym. No workaround or retry was attempted. Current verified leaderboard
MAE remains 0.61766; 0.577805 is local validation only.
