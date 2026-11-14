# Expanded-data architecture comparison

User approved keeping GRU while testing LSTM and TCN with bidirectional context
and more training breaths. LSTM and TCN are separate alternatives, not a combined
LSTM-TCN architecture. Fixed equal-weight combinations are evaluated afterward.

## Protocol

- First 20,000 breaths: 19,500 training and the same original 500 validation breaths.
- Validation IDs are selected by RandomState(42) from the FIRST 5,000 breath IDs,
  not by re-splitting all 20,000. All normalization is fit only on training rows.
- Features, Adam learning rate .001, all-row MSE, and batch size 51 unchanged.
- Up to 30 epochs, patience 8, three fresh recorded training seeds per model.
- Checkpoint selection uses inspiratory MAE. Best/latest checkpoints saved each epoch.
- Epoch timing excludes preprocessing and checkpoint writes.
- BiGRU: 2 layers, hidden size 64/direction, 107,265 parameters.
- BiLSTM: 2 layers, hidden size 56/direction, 110,769 parameters.
- TCN: 60 channels, five residual blocks, two kernel-3 convolutions per block,
  dilations 1/2/4/8/16, symmetric padding, 109,861 parameters. Noncausal convolutions
  use both temporal directions with a nominal receptive field of 125 steps.
- Architecture comparison approximately matches parameters and shares epoch limits;
  it does not match wall-clock budgets. Earlier 5k runs used up to 50 epochs, so
  cross-dataset comparisons change both data amount and training exposure.
- Fixed mean/median within-architecture ensembles; fixed equal-weight means across
  architectures. No validation-fitted ensemble weights.

## Verification

GPU smoke checks passed for all architectures: output shape, finite forward/backward
values, and nonzero dependence on future timesteps. Checkpoint predictions must
reproduce recorded MAE; evaluation asserts identical targets, masks and breath IDs.

This is exploratory validation reused for selection, not an unbiased final test.
No leaderboard upload or controller inversion is part of this experiment.

## Results

All nine runs completed 30 epochs. Original validation IDs, labels and masks
were verified against the previous experiment. All checkpoint reload checks passed.

| Model | Best single MAE | Mean ensemble MAE | Three-run training/validation seconds |
|---|---:|---:|---:|
| gru | 0.798127 | 0.745773 | 179.92 |
| lstm | 0.798024 | 0.746310 | 185.98 |
| tcn | 0.806925 | 0.730358 | 320.71 |

| Equal-weight combination | MAE |
|---|---:|
| gru_lstm_mean | 0.728296 |
| gru_tcn_mean | 0.717634 |
| lstm_tcn_mean | 0.709342 |
| all_mean | 0.710864 |

Best observed score: LSTM + TCN ensemble, 0.709342; all-model mean: 0.710864.
These are close; reused validation does not establish a statistically reliable winner.
GRU and LSTM ensembles are essentially tied. TCN individual runs did not beat the
best recurrent runs, but its ensemble and cross-architecture combinations helped.
Previous 5k BiGRU mean: 1.042093; new 20k BiGRU mean: 0.745773.
Best new combination improves the previous overall best by approximately 31.9%.
Total recorded training/validation time: 686.61 seconds, excluding preprocessing,
checkpoint I/O, transfers and setup. All files and checkpoints saved locally and remotely.
No predictions submitted. GPU pod remains allocated.
