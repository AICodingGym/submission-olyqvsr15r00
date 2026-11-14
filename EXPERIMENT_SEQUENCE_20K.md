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
