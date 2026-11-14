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


## Completed results — November 13, 2026

All four runs completed 40 additional epochs (43.44 training minutes total).

| Variant | First seed | Second seed | Two-seed ensemble MAE |
| --- | ---: | ---: | ---: |
| Original ABC parent ensemble | 0.35611 | 0.35134 | 0.32928 |
| Continued training, no pairing | 0.27825 | 0.30944 | 0.27608 |
| Continued training, mirror pairing | **0.27292** | **0.30330** | **0.27142** |

Pairing improves the matched continuation ensemble by 0.00465909 MAE (1.69%).
Both paired seeds improve over their matching controls. However, the larger
change from the original model comes from continued training. This experiment
supports a modest benefit from the mirror-paired representation; it does not
achieve the requested aspiration of MAE below 0.2. The paired ensemble is the
leading local candidate; no new predictions were submitted.

These comparisons reuse the original 500 validation breaths / 15,177 scored
rows, so they remain exploratory and have not received independent evaluation.
The last verified Coding Gym result is still 0.30326 from the original ABC model.
This is a warm-start adaptation with frozen original feature engineering, not
an exact reproduction of reverse-order feature generation in the public notebook.

## GPU verification

Batch benchmark (breaths per second): 128: 3227; 256: 3885; 512: 4224; 1024: 4312.
Selected 512 because it was within 5% of maximum throughput. Its benchmark peak
allocated memory was 3.78 GB versus 6.15 GB at batch 1024. Training inputs and
labels stayed on GPU, avoiding per-batch host transfers.

Two-second nvidia-smi samples across the entire invocation, including CPU
preparation, averaged 95.14% utilization (median 97%). From the first sample at
80% utilization onward, utilization averaged 96.27%; 98.54% of samples were at
least 90%. Peak observed device memory was 6,880 MiB, including the batch-1024
benchmark. These are utilization measurements, not a claim that every GPU
execution unit sustained its theoretical peak throughput.

## Verification and artifacts

Initial paired/parent equivalence passed (maximum difference 0.000003815).
Reloaded all four final checkpoints and reconstructed validation features using
frozen training-only statistics and normalizers. Each prediction array replayed
exactly, with maximum difference 0.0. Verified training IDs against the original
aggregation fit set, no validation overlap, exact validation IDs/truth/masks,
finite predictions, and two-seed mean ensemble equality.

- Results/configuration/summary: `paired_lstm_experiment/results/`.
- Best paired checkpoints: `results/paired/{931741,2186703}/best.pt`.
- Best unpaired checkpoints: `results/control/{931741,2186703}/best.pt`.
- Per-run histories, validation predictions, and last optimizer/model states
  reside beside the checkpoints.
- Paired ensemble: `paired_lstm_experiment/results/paired/ensemble.npz`.
- Full GPU samples: `paired_lstm_experiment/gpu_utilization.csv`.

Inference uses `PairedLSTM(..., paired=True)` and retains chronological output
order; do not reverse its output again. Preprocessing still uses the original
ABC `pattern_statistics.joblib` plus each checkpoint's mean and scale.
