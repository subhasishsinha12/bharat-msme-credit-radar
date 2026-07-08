"""
Bharat MSME Credit Radar - Model Training
===========================================
Trains and compares Logistic Regression, Random Forest, XGBoost and
LightGBM on the engineered MSME stress dataset, selects the best model
based on recall at the top-20% risk band, AUC-PR, calibration quality and
interpretability, calibrates it (isotonic regression), and persists the
model artifacts used by the API / Streamlit app.

Run:
    python src/train_model.py
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
except ImportError:
    FrozenEstimator = None

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import build_features, get_model_feature_columns  # noqa: E402
from evaluate_model import compute_all_metrics, compute_psi  # noqa: E402
from model_utils import align_columns, encode_features, group_split  # noqa: E402,F401 (re-exported for callers)
import segment_models  # noqa: E402
import survival_model  # noqa: E402
import growth_propensity  # noqa: E402

DATA_PATH = os.path.join("data", "synthetic_msme_data.csv")
MODELS_DIR = "models"
REPORTS_DIR = "reports"
SEED = 42


def build_candidate_models(scale_pos_weight: float):
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=0.5, random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=20,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
            reg_lambda=2.0, scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr", random_state=SEED, n_jobs=-1,
        )
    except ImportError:
        print("xgboost not available, skipping.")
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, num_leaves=24,
            subsample=0.85, colsample_bytree=0.85, min_child_samples=30,
            reg_lambda=2.0, class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        print("lightgbm not available, skipping.")
    return models


def model_selection_score(metrics: dict) -> float:
    """Composite ranking score: recall@20% + AUC-PR weighted highest, with a
    calibration (lower Brier is better) and stability contribution."""
    return (
        0.40 * metrics["recall_at_top20pct"]
        + 0.35 * metrics["auc_pr"]
        + 0.15 * (1 - metrics["brier_score"])
        + 0.10 * metrics["ks_statistic"]
    )


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading data...")
    raw = pd.read_csv(DATA_PATH)

    print("Computing default value profile (for partial API payload imputation)...")
    id_and_target_cols = {"record_id", "borrower_id", "borrower_name", "obs_month", "stress_12m", "growth_need_12m"}
    text_cols = {"cam_remarks", "fi_remarks", "rcu_remarks", "collection_remarks", "stock_inspection_remarks"}
    default_profile = {}
    for col in raw.columns:
        if col in id_and_target_cols:
            continue
        if col in text_cols:
            default_profile[col] = ""
        elif pd.api.types.is_numeric_dtype(raw[col]):
            default_profile[col] = float(raw[col].median())
        else:
            default_profile[col] = raw[col].mode().iloc[0]
    with open(os.path.join(MODELS_DIR, "default_profile.json"), "w") as f:
        json.dump(default_profile, f, indent=2, default=str)

    print("Engineering features (incl. TF-IDF text risk model)...")
    engineered, text_artifacts = build_features(raw, fit_text_model=True)
    feature_cols = get_model_feature_columns(engineered)

    print("Splitting by borrower (train/calibration/test)...")
    train_df, calib_df, test_df = group_split(engineered)
    print(f"  train rows={len(train_df)}  calib rows={len(calib_df)}  test rows={len(test_df)}  "
          f"(stress rates: {train_df.stress_12m.mean():.2%} / {calib_df.stress_12m.mean():.2%} / {test_df.stress_12m.mean():.2%})")

    X_train_raw, categories = encode_features(train_df, feature_cols)
    encoded_columns = X_train_raw.columns.tolist()
    X_train = X_train_raw.astype(float)
    y_train = train_df["stress_12m"].to_numpy()

    X_calib, _ = encode_features(calib_df, feature_cols, categories)
    X_calib = align_columns(X_calib, encoded_columns).astype(float)
    y_calib = calib_df["stress_12m"].to_numpy()

    X_test, _ = encode_features(test_df, feature_cols, categories)
    X_test = align_columns(X_test, encoded_columns).astype(float)
    y_test = test_df["stress_12m"].to_numpy()

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"scale_pos_weight (imbalance ratio) = {scale_pos_weight:.2f}")

    models = build_candidate_models(scale_pos_weight)

    results = {}
    fitted = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        prob_test = model.predict_proba(X_test)[:, 1]
        metrics = compute_all_metrics(y_test, prob_test)
        metrics["selection_score"] = model_selection_score(metrics)
        results[name] = metrics
        fitted[name] = model
        print(f"  AUC-ROC={metrics['auc_roc']:.3f}  AUC-PR={metrics['auc_pr']:.3f}  "
              f"KS={metrics['ks_statistic']:.3f}  Recall@20%={metrics['recall_at_top20pct']:.3f}  "
              f"Brier={metrics['brier_score']:.4f}  score={metrics['selection_score']:.4f}")

    best_name = max(results, key=lambda n: results[n]["selection_score"])
    best_model = fitted[best_name]
    print(f"\nSelected best model: {best_name}")

    print("Calibrating best model (isotonic regression on held-out calibration set)...")
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(FrozenEstimator(best_model), method="isotonic")
    else:
        calibrator = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit")
    calibrator.fit(X_calib, y_calib)

    prob_test_calibrated = calibrator.predict_proba(X_test)[:, 1]
    calibrated_metrics = compute_all_metrics(y_test, prob_test_calibrated)
    print(f"Calibrated test metrics: AUC-ROC={calibrated_metrics['auc_roc']:.3f}  "
          f"AUC-PR={calibrated_metrics['auc_pr']:.3f}  Recall@20%={calibrated_metrics['recall_at_top20pct']:.3f}  "
          f"Brier={calibrated_metrics['brier_score']:.4f}")

    # Population Stability Index: development (train) vs holdout (test) score distribution.
    prob_train_calibrated = calibrator.predict_proba(X_train)[:, 1]
    psi_dev_vs_holdout = compute_psi(prob_train_calibrated, prob_test_calibrated)
    print(f"PSI (train/dev vs test/holdout PD distribution) = {psi_dev_vs_holdout:.4f}")

    # Persist artifacts
    joblib.dump(best_model, os.path.join(MODELS_DIR, "trained_model.pkl"))
    joblib.dump(calibrator, os.path.join(MODELS_DIR, "calibrator.pkl"))
    vectorizer, text_model = text_artifacts
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(text_model, os.path.join(MODELS_DIR, "text_risk_model.pkl"))

    feature_list = {
        "raw_feature_columns": feature_cols,
        "categorical_columns": list(categories.keys()),
        "categories": categories,
        "encoded_columns": encoded_columns,
        "best_model_name": best_name,
        "scale_pos_weight": scale_pos_weight,
    }
    with open(os.path.join(MODELS_DIR, "feature_list.json"), "w") as f:
        json.dump(feature_list, f, indent=2)

    # Save all-model comparison + selected model metrics for reporting
    all_results_serializable = {
        name: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items() if k != "confusion_matrix"}
        for name, m in results.items()
    }
    for name, m in results.items():
        all_results_serializable[name]["confusion_matrix"] = m["confusion_matrix"].tolist()

    report_payload = {
        "dataset_rows": len(engineered),
        "n_borrowers": engineered["borrower_id"].nunique(),
        "stress_rate_overall": float(engineered["stress_12m"].mean()),
        "train_rows": len(train_df),
        "calib_rows": len(calib_df),
        "test_rows": len(test_df),
        "models_compared": list(results.keys()),
        "uncalibrated_results": all_results_serializable,
        "best_model_name": best_name,
        "calibrated_test_metrics": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in calibrated_metrics.items()
        },
        "psi_dev_vs_holdout": psi_dev_vs_holdout,
    }
    with open(os.path.join(MODELS_DIR, "training_report.json"), "w") as f:
        json.dump(report_payload, f, indent=2)

    # Persist the exact test split (encoded) for downstream evaluation/explainability scripts
    test_df.to_csv(os.path.join(MODELS_DIR, "_test_split_raw.csv"), index=False)

    print("\n--- Segment-Wise MSME Model Design (Section 5) ---")
    segment_results = segment_models.train_all_segment_models(engineered, feature_cols, categories, encoded_columns)
    segment_models.save_segment_artifacts(segment_results, MODELS_DIR, feature_cols, categories, encoded_columns)

    print("\n--- Survival / Timing Model (Layer 4) ---")
    surv_model = survival_model.train_survival_model(engineered, feature_cols, categories, encoded_columns)
    survival_model.save_survival_artifacts(surv_model, MODELS_DIR)
    print("  Trained discrete-time (quarterly) hazard model for month-of-stress timing.")

    print("\n--- MSME Growth Propensity Engine (Section 12) ---")
    growth_model, growth_calibrator, growth_metrics = growth_propensity.train_growth_model(
        engineered, feature_cols, categories, encoded_columns
    )
    growth_propensity.save_growth_artifacts(
        growth_model, growth_calibrator, feature_cols, categories, encoded_columns, growth_metrics, MODELS_DIR
    )
    print(f"  AUC-ROC={growth_metrics['auc_roc']:.3f}  AUC-PR={growth_metrics['auc_pr']:.3f}  "
          f"Recall@20%={growth_metrics['recall_at_top20pct']:.3f}")

    print("\n--- Segment PD Benchmark Reference (percentile curves) ---")
    X_all, _ = encode_features(engineered, feature_cols, categories)
    X_all = align_columns(X_all, encoded_columns).astype(float)
    prob_all = calibrator.predict_proba(X_all)[:, 1]
    percentiles = np.arange(0, 101)
    segment_pd_reference = {"_overall": np.percentile(prob_all, percentiles).tolist()}
    for segment in engineered["segment"].unique():
        mask = (engineered["segment"] == segment).to_numpy()
        if mask.sum() >= 30:
            segment_pd_reference[segment] = np.percentile(prob_all[mask], percentiles).tolist()
    with open(os.path.join(MODELS_DIR, "segment_pd_reference.json"), "w") as f:
        json.dump(segment_pd_reference, f, indent=2)

    print("\nSaved: trained_model.pkl, calibrator.pkl, feature_list.json, tfidf_vectorizer.pkl, "
          "text_risk_model.pkl, training_report.json, segment_manifest.json + models/segments/*, "
          "survival_model.pkl, growth_model.pkl, growth_calibrator.pkl, growth_feature_list.json, "
          "segment_pd_reference.json")


if __name__ == "__main__":
    main()
