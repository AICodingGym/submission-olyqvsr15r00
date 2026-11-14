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
