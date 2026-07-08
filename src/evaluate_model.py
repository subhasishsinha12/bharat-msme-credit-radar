"""
Bharat MSME Credit Radar - Model Evaluation
==============================================
Reusable evaluation metrics for imbalanced 12-month default prediction, and
a standalone script that loads the trained + calibrated model, scores the
held-out test split, and writes `reports/model_performance_report.md`.

Metrics deliberately go beyond accuracy: AUC-ROC, AUC-PR, Gini, KS,
precision/recall/F1 at a tuned threshold, recall captured in the top 10%
and top 20% riskiest accounts (the metric a collections/EWS team actually
cares about), top-decile lift, Brier score (calibration), and the
confusion matrix.

Run:
    python src/evaluate_model.py
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

sys.path.insert(0, os.path.dirname(__file__))

MODELS_DIR = "models"
REPORTS_DIR = "reports"


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))


def gini_from_auc(auc: float) -> float:
    return 2 * auc - 1


def recall_at_top_pct(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float:
    n = len(y_true)
    k = max(int(np.ceil(n * pct)), 1)
    order = np.argsort(-y_prob)
    top_k_idx = order[:k]
    total_positives = y_true.sum()
    if total_positives == 0:
        return 0.0
    return float(y_true[top_k_idx].sum() / total_positives)


def top_decile_lift(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n = len(y_true)
    k = max(int(np.ceil(n * 0.10)), 1)
    order = np.argsort(-y_prob)
    top_decile_rate = y_true[order[:k]].mean()
    overall_rate = y_true.mean()
    if overall_rate == 0:
        return 0.0
    return float(top_decile_rate / overall_rate)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.linspace(0.01, 0.9, 90)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between a 'development' (expected) score
    distribution and a 'holdout/live' (actual) one. Bucket edges are the
    development set's own quantiles, per the standard PSI formulation.
    PSI < 0.10 = stable, 0.10-0.25 = moderate shift, > 0.25 = significant shift."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    if len(edges) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = np.clip(exp_counts / len(expected), 1e-4, None)
    act_pct = np.clip(act_counts / len(actual), 1e-4, None)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def compute_all_metrics(y_true, y_prob, threshold: float | None = None) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    auc_roc = float(roc_auc_score(y_true, y_prob))
    auc_pr = float(average_precision_score(y_true, y_prob))
    ks = ks_statistic(y_true, y_prob)
    gini = gini_from_auc(auc_roc)
    brier = float(brier_score_loss(y_true, y_prob))

    if threshold is None:
        threshold = best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "gini": gini,
        "ks_statistic": ks,
        "brier_score": brier,
        "threshold_used": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall_at_top10pct": recall_at_top_pct(y_true, y_prob, 0.10),
        "recall_at_top20pct": recall_at_top_pct(y_true, y_prob, 0.20),
        "top_decile_lift": top_decile_lift(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "positive_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }
    return metrics


def _load_test_split():
    from feature_engineering import build_features, get_model_feature_columns
    from train_model import encode_features, align_columns

    test_df = pd.read_csv(os.path.join(MODELS_DIR, "_test_split_raw.csv"))
    with open(os.path.join(MODELS_DIR, "feature_list.json")) as f:
        feature_list = json.load(f)

    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    text_model = joblib.load(os.path.join(MODELS_DIR, "text_risk_model.pkl"))

    engineered, _ = build_features(test_df, fit_text_model=False, text_artifacts=(vectorizer, text_model))
    X_test, _ = encode_features(engineered, feature_list["raw_feature_columns"], feature_list["categories"])
    X_test = align_columns(X_test, feature_list["encoded_columns"]).astype(float)
    y_test = engineered["stress_12m"].to_numpy()
    return X_test, y_test


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    X_test, y_test = _load_test_split()

    model = joblib.load(os.path.join(MODELS_DIR, "trained_model.pkl"))
    calibrator = joblib.load(os.path.join(MODELS_DIR, "calibrator.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_list.json")) as f:
        feature_list = json.load(f)

    prob_uncalibrated = model.predict_proba(X_test)[:, 1]
    prob_calibrated = calibrator.predict_proba(X_test)[:, 1]

    metrics_uncal = compute_all_metrics(y_test, prob_uncalibrated)
    metrics_cal = compute_all_metrics(y_test, prob_calibrated)

    # Calibration curve (10 bins by predicted probability)
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(prob_calibrated, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 9)
    calib_curve = []
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        calib_curve.append({
            "bin": b, "n": int(mask.sum()),
            "mean_predicted_pd": float(prob_calibrated[mask].mean()),
            "observed_stress_rate": float(y_test[mask].mean()),
        })

    psi = None
    training_report_path = os.path.join(MODELS_DIR, "training_report.json")
    if os.path.exists(training_report_path):
        with open(training_report_path) as f:
            psi = json.load(f).get("psi_dev_vs_holdout")

    report = {
        "best_model_name": feature_list["best_model_name"],
        "uncalibrated_metrics": {k: v for k, v in metrics_uncal.items() if k != "confusion_matrix"},
        "calibrated_metrics": {k: v for k, v in metrics_cal.items() if k != "confusion_matrix"},
        "confusion_matrix_calibrated": metrics_cal["confusion_matrix"].tolist(),
        "calibration_curve": calib_curve,
        "psi_dev_vs_holdout": psi,
    }
    with open(os.path.join(MODELS_DIR, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    _write_markdown_report(report)
    print("Evaluation complete. See models/evaluation_report.json and reports/model_performance_report.md")


def _write_markdown_report(report: dict):
    m = report["calibrated_metrics"]
    cm = report["confusion_matrix_calibrated"]
    lines = [
        "# Model Performance Report — Bharat MSME Credit Radar",
        "",
        f"**Selected model:** `{report['best_model_name']}` (calibrated with isotonic regression)",
        "",
        "## Held-out Test Set Metrics (Calibrated PD)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| AUC-ROC | {m['auc_roc']:.3f} |",
        f"| AUC-PR (Average Precision) | {m['auc_pr']:.3f} |",
        f"| Gini Coefficient | {m['gini']:.3f} |",
        f"| KS Statistic | {m['ks_statistic']:.3f} |",
        f"| Brier Score | {m['brier_score']:.4f} |",
        f"| Precision @ tuned threshold ({m['threshold_used']:.2f}) | {m['precision']:.3f} |",
        f"| Recall @ tuned threshold | {m['recall']:.3f} |",
        f"| F1 Score | {m['f1_score']:.3f} |",
        f"| **Recall captured in top 10% riskiest accounts** | **{m['recall_at_top10pct']:.1%}** |",
        f"| **Recall captured in top 20% riskiest accounts** | **{m['recall_at_top20pct']:.1%}** |",
        f"| Top-decile lift | {m['top_decile_lift']:.2f}x |",
        f"| Test set size / stress rate | {m['n']:,} rows / {m['positive_rate']:.2%} |",
    ]
    if report.get("psi_dev_vs_holdout") is not None:
        psi_val = report["psi_dev_vs_holdout"]
        psi_note = "stable" if psi_val < 0.10 else ("moderate shift" if psi_val < 0.25 else "significant shift")
        lines.append(f"| PSI (development/train vs holdout/test PD distribution) | {psi_val:.4f} ({psi_note}) |")
    lines += [
        "",
        "## Confusion Matrix (at tuned F1 threshold)",
        "",
        "| | Predicted Non-Stress | Predicted Stress |",
        "|---|---|---|",
        f"| **Actual Non-Stress** | {cm[0][0]} | {cm[0][1]} |",
        f"| **Actual Stress** | {cm[1][0]} | {cm[1][1]} |",
        "",
        "## Calibration Curve (Predicted PD vs Observed Stress Rate, 10 bins)",
        "",
        "| Bin | N | Mean Predicted PD | Observed Stress Rate |",
        "|---|---|---|---|",
    ]
    for row in report["calibration_curve"]:
        lines.append(f"| {row['bin']} | {row['n']} | {row['mean_predicted_pd']:.3f} | {row['observed_stress_rate']:.3f} |")

    lines += [
        "",
        "> Recall at the top 20% risk band is emphasised over plain accuracy because an early-warning "
        "system is judged by how many genuinely stressed accounts are surfaced within the review "
        "capacity a bank can actually action (a field-visit / stock-audit queue is realistically sized "
        "at the top 10-20% of the book), not by overall classification accuracy on a ~93% non-stress "
        "imbalanced target.",
        "",
        "**Disclaimer:** These metrics are computed on synthetic data and validate the design logic of "
        "the pipeline, not real-world bank-grade model performance. See `reports/prototype_validation_note.md`.",
    ]
    with open(os.path.join(REPORTS_DIR, "model_performance_report.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
