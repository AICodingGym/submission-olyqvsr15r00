# Error analysis of the selected ensemble

Saved predictions only: no training or submission. Same 500 validation breaths.
Signed bias = prediction minus observed pressure. Groups are descriptive, not causal.
Pressure-based groups and peak locations use validation labels for diagnosis only.

## Time within breath

| Group | Rows | MAE | Signed bias | Share of total absolute error |
|---|---:|---:|---:|---:|
| 0 | 500 | 0.3218 | +0.0953 | 1.8% |
| 1-4 | 2000 | 0.3596 | +0.0236 | 8.2% |
| 5-9 | 2500 | 0.4857 | -0.0297 | 13.8% |
| 10-19 | 5000 | 0.6228 | -0.0408 | 35.5% |
| 20-29 | 4849 | 0.6762 | -0.0627 | 37.4% |
| 30-39 | 328 | 0.8607 | -0.0947 | 3.2% |

## Observed pressure

| Group | Rows | MAE | Signed bias | Share of total absolute error |
|---|---:|---:|---:|---:|
| 0-5 | 171 | 0.7757 | +0.5398 | 1.5% |
| 5-10 | 3626 | 0.3177 | +0.0901 | 13.1% |
| 10-20 | 5935 | 0.5788 | -0.0243 | 39.2% |
| 20-30 | 3825 | 0.7066 | -0.0717 | 30.8% |
| 30-40 | 1417 | 0.8247 | -0.2935 | 13.3% |
| >40 | 203 | 0.8780 | -0.5076 | 2.0% |

## Lung setting (R,C)

| Group | Rows | MAE | Signed bias | Share of total absolute error |
|---|---:|---:|---:|---:|
| 20,10 | 1209 | 0.5343 | -0.0146 | 7.4% |
| 20,20 | 1107 | 0.4803 | -0.0878 | 6.1% |
| 20,50 | 1728 | 0.4223 | -0.0146 | 8.3% |
| 5,10 | 1900 | 0.4966 | +0.0479 | 10.8% |
| 5,20 | 1698 | 0.3961 | +0.1813 | 7.7% |
| 5,50 | 1666 | 0.3669 | +0.0743 | 7.0% |
| 50,10 | 2127 | 0.6984 | -0.1246 | 16.9% |
| 50,20 | 2134 | 0.7785 | -0.2687 | 18.9% |
| 50,50 | 1608 | 0.9252 | -0.0389 | 17.0% |

## Error concentration

| Worst breaths by MAE | Their error share | Remaining-row MAE |
|---|---:|---:|
| 1% | 4.6% | 0.5572 |
| 5% | 16.8% | 0.5061 |
| 10% | 28.2% | 0.4613 |
| 20% | 45.2% | 0.3966 |

Full timestep statistics, worst-breath metadata, peak-row analysis, disagreement
groups and training histories are in diagnostics.json. All results use a repeatedly
tuned holdout; validate any resulting intervention independently.

## Interpretation

- R=50 accounts for 38.7% of scored rows and 52.8% of absolute error.
- Steps 10-29 contribute 72.9% of error; breath starts contribute relatively little.
- All nine lung-setting groups improved versus the previous model, but R=50,C=50 remains hardest.
- High pressures are underpredicted on average; however pressure >30 contributes only 15.4% of error.
- Peak rows alone account for 4.7% of error. A peak-only intervention cannot close the full gap.
- Even hypothetical perfect correction of the worst 10% of breaths leaves overall MAE 0.4147.
- A label-dependent per-breath median-offset oracle gives MAE 0.3337; this is a diagnostic, not a usable prediction or validation result.
- The five worst breaths have signed bias magnitude close to MAE, suggesting largely one-sided offsets for these cases.
- Five of the six chosen training runs have their lowest validation MAE at epoch five. Both training loss and validation MAE improved from first to last epoch in all six; this does not establish immunity to future overfitting.
- Recommended next diagnostic: confirm lung-setting and offset patterns using out-of-fold training predictions; fit any residual correction using those predictions, not these validation errors.
- Candidate follow-up: an input-conditioned residual/whole-breath context head targeting R=50 and later inspiration, with global MAE retained as the selection metric. It remains an untested hypothesis.
