# Approved ensemble, bidirectional GRU, and diagnostic experiment

User approved options A, B, and C after discussing AmbrosM's writeup:
https://www.kaggle.com/competitions/ventilator-pressure-prediction/writeups/ambrosm-2-solution-the-inverse-of-a-pid-controller

Conceptual inspiration: ensemble predictions, and consider feedback as a reason
later control inputs may help infer earlier pressure. No external code, weights,
controller parameter lists, or original Kaggle data are used. The bidirectional
GRU is our experiment, not a reproduction of the writeup's architecture.

- Same first 5,000 local training breaths and seed-42 500-breath holdout.
- A: mean and median of the three existing best forward-GRU checkpoints.
- B: three fresh recorded seeds, two-layer bidirectional GRU, 64 units per
  direction, up to 50 epochs, patience 8, same features/Adam/MSE/batch size.
  This changes both available context and parameter count; it does not isolate
  bidirectionality from capacity. Mean/median bidirectional ensembles also evaluated.
- C: inspiratory MAE and row counts by R/C, exact valve saturation (0 or 100),
  and abrupt absolute u_in step change (above training inspiratory 95th percentile).
- Checkpoint reevaluation must reproduce stored best MAE within 1e-5.
- Same holdout is used for checkpoint selection and comparison, so improvements
  should later be checked on fresh held-out breaths. This is not a leaderboard score.

Scripts: gru_long.py --bidirectional and evaluate_gru_abc.py.
Runtime measurements include epoch training/validation but exclude checkpoint writes.

## Observed results

All models evaluated on the same 15,177 inspiratory rows.

| Model | MAE |
|---|---:|
| forward_1844954159 | 1.261023 |
| forward_568154637 | 1.247273 |
| forward_2119916341 | 1.279933 |
| forward_mean | 1.214631 |
| forward_median | 1.219959 |
| bidirectional_1321964005 | 1.106565 |
| bidirectional_1277287051 | 1.069425 |
| bidirectional_1592780024 | 1.138669 |
| bidirectional_mean | 1.042093 |
| bidirectional_median | 1.059553 |

Bidirectional runs each completed 50 epochs; training/validation durations: 26.63s, 24.92s, 26.89s.
Forward models have 41,345 parameters; bidirectional models have 107,265.

R=50,C=10 improved from 2.357875 to 1.551871 MAE with the mean ensemble,
but remains the highest-error lung setting. Two R/C groups worsened slightly.
Saturated rows and abrupt-change rows had lower MAE than their complements;
these diagnostics do not support assuming those conditions dominate errors.

All six checkpoint predictions reproduced recorded MAE within 1e-5.
Artifacts and checkpoints are in gru_abc_experiment/ locally and on DataHub.
No leaderboard submission was made.
