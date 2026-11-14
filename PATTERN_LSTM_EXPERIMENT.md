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


## Completed results — November 13, 2026

All eight runs completed 40 epochs on the allocated GTX 1080 Ti. Total training
time was 44.35 minutes, excluding preprocessing and verification. Seeds were
527193 and 1804369. The fixed validation set contains 15,177 scored rows.

| Variant | Seed 527193 best | Seed 1804369 best | Two-seed mean prediction MAE |
| --- | ---: | ---: | ---: |
| control | 0.54827 | 0.56314 | 0.52316 |
| A | 0.47783 | 0.50398 | 0.46156 |
| AB | 0.43111 | 0.41453 | 0.40017 |
| **ABC** | **0.35611** | **0.35134** | **0.32928** |

The ABC ensemble is the new leading local candidate, improving by 41.97% over
the previous corrected ensemble (0.56740350), and 37.06% over this experiment's
matched control. Each incremental change improves both individual seed results
and the corresponding two-seed ensemble. The current experiment's small-model
control is a plain BiLSTM, not the previous CNN-BiLSTM + TCN ensemble.

The larger network has 1,954,177 parameters (small control: 110,769). Its two best
checkpoints are at epochs 40 and 39, respectively. The first run's learning rate
remained 0.001; the second reached 0.00025. Both are near the epoch ceiling,
so a longer, lower-learning-rate continuation is a reasonable next experiment.
These observations do not prove that more training will improve validation.
The target MAE 0.2 has not yet been reached.

### Verification

- Feature checks passed: exact group means/deviation identities, unseen-group
  fallback, frozen aggregation tables, and no cross-breath shift contamination.
- A GPU smoke test exposed amplification of constant training features; constant
  feature scales now use 1 rather than 1e-6. The corrected smoke test passed before
  all full runs. The unseen-group flag was the relevant constant feature.
- All 67,405 training breath IDs match the fitted pattern-table ID set; none
  overlap the original 500 validation breaths. Compare sets because input breath
  IDs are not sorted in the dataset.
- Only nine validation rows out of 40,000 had unseen pattern keys.
- Reloaded all eight saved checkpoints, regenerated features using the saved
  training-only table and normalizers, and reproduced predictions exactly
  (maximum absolute difference 0.0 for each run).
- Verified validation ID order, pressure labels, inspiratory masks, finite outputs,
  and equality of each ensemble to the mean of its two saved member predictions.

The larger model is trained from scratch; no prior residual correction was reused.
The old models remain intact. These are local validation results with repeated
holdout use, not independent leaderboard scores. No test predictions were submitted.

### Artifacts and reproduction

Source: `pattern_features.py` and `train_pattern_lstm.py`.
Results: `pattern_lstm_experiment/results/{results,summary,audit,config}.json`.
Training-only statistics: `pattern_lstm_experiment/results/pattern_statistics.joblib`.
Winning checkpoints: `pattern_lstm_experiment/results/ABC/{527193,1804369}/best.pt`.
Winning validation ensemble: `pattern_lstm_experiment/results/ABC/ensemble.npz`.
Each run directory also contains `history.json` and `validation.npz`.

On the DataHub pod, from the experiment directory, reproduce into a NEW output
folder (the training script starts from scratch and is not a resume command):

```sh
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 python -u train_pattern_lstm.py \
  --data ../train.csv \
  --reference ../strategy_experiment/results/ensemble_predictions.npz \
  --output reproduction_results --seeds 527193 1804369 --epochs 40
```


## Coding Gym submission — November 13, 2026

After explicit user authorization, submitted the exact ABC two-seed ensemble.
Coding Gym returned **MAE 0.30326** (local validation: 0.32928264).
Result URL: https://aicodinggym.com/challenges/mle/ventilator-pressure-prediction

CSV: `pattern_lstm_experiment/submission_abc_032928.csv`, 603,600 rows.
SHA-256: `4cb3d0875643b0e9f348a072458277d73926ef3e58c51433ea154d2bfcdbbeb9`.
Verified original sample IDs/order, finite predictions, and checkpoint hashes.
Inference uses frozen training statistics and the two best ABC checkpoints;
there is no residual correction, retraining, or prediction clipping.

DataHub GPU inference sampled at 90–91% utilization on the GTX 1080 Ti, with
724 MiB total device memory in use. It was idle before inference because training
had completed. Historical sustained training utilization was not recorded and
cannot be asserted as 100%. Session-log upload was disabled for this submission.
The CLI also reported code pushed to branch `ventilator-pressure-prediction`.
