# Model Performance Report — Bharat MSME Credit Radar

**Selected model:** `random_forest` (calibrated with isotonic regression)

## Held-out Test Set Metrics (Calibrated PD)

| Metric | Value |
|---|---|
| AUC-ROC | 0.946 |
| AUC-PR (Average Precision) | 0.715 |
| Gini Coefficient | 0.892 |
| KS Statistic | 0.764 |
| Brier Score | 0.0299 |
| Precision @ tuned threshold (0.34) | 0.712 |
| Recall @ tuned threshold | 0.662 |
| F1 Score | 0.686 |
| **Recall captured in top 10% riskiest accounts** | **77.9%** |
| **Recall captured in top 20% riskiest accounts** | **90.7%** |
| Top-decile lift | 7.79x |
| Test set size / stress rate | 8,400 rows / 6.52% |

## Confusion Matrix (at tuned F1 threshold)

| | Predicted Non-Stress | Predicted Stress |
|---|---|---|
| **Actual Non-Stress** | 7705 | 147 |
| **Actual Stress** | 185 | 363 |

## Calibration Curve (Predicted PD vs Observed Stress Rate, 10 bins)

| Bin | N | Mean Predicted PD | Observed Stress Rate |
|---|---|---|---|
| 0 | 7074 | 0.007 | 0.010 |
| 1 | 436 | 0.137 | 0.101 |
| 2 | 367 | 0.251 | 0.183 |
| 3 | 14 | 0.336 | 0.286 |
| 4 | 111 | 0.434 | 0.432 |
| 5 | 93 | 0.522 | 0.505 |
| 7 | 89 | 0.736 | 0.809 |
| 9 | 216 | 0.958 | 0.903 |

> Recall at the top 20% risk band is emphasised over plain accuracy because an early-warning system is judged by how many genuinely stressed accounts are surfaced within the review capacity a bank can actually action (a field-visit / stock-audit queue is realistically sized at the top 10-20% of the book), not by overall classification accuracy on a ~93% non-stress imbalanced target.

**Disclaimer:** These metrics are computed on synthetic data and validate the design logic of the pipeline, not real-world bank-grade model performance. See `reports/prototype_validation_note.md`.