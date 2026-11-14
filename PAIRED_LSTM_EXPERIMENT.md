# Forward/reverse feature pairing

The user approved testing mirror-paired features and requested efficient DataHub
GPU use. This experiment tests whether pairing adds value beyond continuing the
current ABC LSTM. No claim is made that it will reach MAE below 0.2.

## Design

- Preserve all 67,405 training breaths and the original 500 validation breaths.
- Reuse frozen training-only aggregation tables and normalizers from ABC.
- Each timestep receives `[features[t], features[79-t]]`, 122 channels. Prediction
  and loss remain in chronological order. No sequence crosses a breath boundary.
- This is an independently implemented mirror-pair adaptation, not a reproduction
  of reverse-order feature engineering from the reference notebook.
- Widen each input/skip projection and initialize added columns to zero. Verify
  that each paired model initially reproduces its original ABC checkpoint.
- Compare paired and unpaired continuations from both original best checkpoints.
  Fresh matched training seeds: 931741 and 2186703. Fresh Adam at 0.0003, gradient
  clipping at 1, plateau learning-rate reductions, patience 10, up to 40 epochs.
- Include the original checkpoint as epoch zero so a continuation cannot discard
  a better parent checkpoint. Save per-run best and last states and predictions.
- Compare two-seed means without fitting ensemble weights to validation.

## GPU throughput

Keep the training and validation tensors on GPU. Benchmark optimizer-step
throughput at batches 128, 256, 512, and 1024 using disposable paired models.
Select the smallest batch within 5% of maximum measured throughput; use the
same batch for paired and control runs. Record utilization and memory throughout
the run with nvidia-smi. High memory occupancy alone is not proof of efficiency.

The inference output from this experiment is not automatically authorized for
submission. The previously submitted model and its 0.30326 Coding Gym score
remain intact. The original ensemble's local validation MAE is 0.32928264.
