# Model Performance Report — Bharat MSME Credit Radar

**Selected model:** `random_forest` (calibrated with isotonic regression)

## Held-out Test Set Metrics (Calibrated PD)

| Metric | Value |
|---|---|
| AUC-ROC | 0.954 |
| AUC-PR (Average Precision) | 0.738 |
| Gini Coefficient | 0.909 |
| KS Statistic | 0.780 |
| Brier Score | 0.0281 |
| Precision @ tuned threshold (0.38) | 0.782 |
| Recall @ tuned threshold | 0.621 |
| F1 Score | 0.692 |
| **Recall captured in top 10% riskiest accounts** | **79.9%** |
| **Recall captured in top 20% riskiest accounts** | **91.8%** |
| Top-decile lift | 7.99x |
| Test set size / stress rate | 8,400 rows / 6.40% |
| PSI (development/train vs holdout/test PD distribution) | 0.0025 (stable) |

## Confusion Matrix (at tuned F1 threshold)

| | Predicted Non-Stress | Predicted Stress |
|---|---|---|
| **Actual Non-Stress** | 7769 | 93 |
| **Actual Stress** | 204 | 334 |

## Calibration Curve (Predicted PD vs Observed Stress Rate, 10 bins)

| Bin | N | Mean Predicted PD | Observed Stress Rate |
|---|---|---|---|
| 0 | 7275 | 0.009 | 0.010 |
| 1 | 366 | 0.135 | 0.128 |
| 2 | 115 | 0.260 | 0.165 |
| 3 | 218 | 0.367 | 0.303 |
| 4 | 53 | 0.478 | 0.509 |
| 5 | 88 | 0.529 | 0.568 |
| 6 | 16 | 0.667 | 0.750 |
| 7 | 25 | 0.752 | 0.840 |
| 8 | 99 | 0.849 | 0.848 |
| 9 | 145 | 0.970 | 0.959 |

> Recall at the top 20% risk band is emphasised over plain accuracy because an early-warning system is judged by how many genuinely stressed accounts are surfaced within the review capacity a bank can actually action (a field-visit / stock-audit queue is realistically sized at the top 10-20% of the book), not by overall classification accuracy on a ~93% non-stress imbalanced target.

**Disclaimer:** These metrics are computed on synthetic data and validate the design logic of the pipeline, not real-world bank-grade model performance. See `reports/prototype_validation_note.md`.