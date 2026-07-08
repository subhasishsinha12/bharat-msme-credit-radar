"""
Bharat MSME Credit Radar - MSME Growth Propensity Engine
============================================================
Section 12 of the design brief: "the data that reveals stress early also
reveals strength early." This module inverts the Radar's lens — for
accounts that are *not* under watch, it estimates the probability that the
borrower will need an enhanced working-capital limit or a new term loan
within 6-12 months, using the same signal spine (GST growth trajectory,
cash-flow headroom, capacity/EPFO signals, utilisation pattern, bureau
posture) that the risk engine already computes.

Guardrails are enforced in code, not left to policy discretion:
  - Only Green / Yellow grade accounts are eligible.
  - Any account with a fraud-keyword flag, a GST-bank authenticity mismatch,
    or an elevated near-term SMA migration probability is excluded, however
    high its raw growth score.
  - Every output is a lead for the banker, never an automated sanction.
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

try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
except ImportError:
    FrozenEstimator = None

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import align_columns, encode_features, group_split  # noqa: E402
from evaluate_model import compute_all_metrics  # noqa: E402

ELIGIBLE_GRADES = {"Green", "Yellow"}
SMA_MIGRATION_EXCLUSION_THRESHOLD = 0.05  # exclude if near-term (Q1) SMA migration probability >= 5%

MODEL_FILENAME = "growth_model.pkl"
CALIBRATOR_FILENAME = "growth_calibrator.pkl"
FEATURE_LIST_FILENAME = "growth_feature_list.json"


def train_growth_model(engineered: pd.DataFrame, feature_cols: list[str],
                        global_categories: dict, global_encoded_columns: list[str], seed: int = 42):
    """Trains and (where possible) calibrates a single portfolio-wide growth
    propensity model on `growth_need_12m`. Uses the same borrower-grouped
    split discipline as the PD models to avoid leakage."""
    train_df, calib_df, test_df = group_split(engineered, seed=seed)

    X_train, _ = encode_features(train_df, feature_cols, global_categories)
    X_train = align_columns(X_train, global_encoded_columns).astype(float)
    y_train = train_df["growth_need_12m"].to_numpy()

    X_calib, _ = encode_features(calib_df, feature_cols, global_categories)
    X_calib = align_columns(X_calib, global_encoded_columns).astype(float)
    y_calib = calib_df["growth_need_12m"].to_numpy()

    X_test, _ = encode_features(test_df, feature_cols, global_categories)
    X_test = align_columns(X_test, global_encoded_columns).astype(float)
    y_test = test_df["growth_need_12m"].to_numpy()

    model = RandomForestClassifier(
        n_estimators=350, max_depth=8, min_samples_leaf=15,
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    else:
        calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(X_calib, y_calib)

    prob_test = calibrator.predict_proba(X_test)[:, 1]
    metrics = compute_all_metrics(y_test, prob_test)

    return model, calibrator, {k: v for k, v in metrics.items() if k != "confusion_matrix"}


def save_growth_artifacts(model, calibrator, feature_cols, categories, encoded_columns,
                           metrics, models_dir: str):
    joblib.dump(model, os.path.join(models_dir, MODEL_FILENAME))
    joblib.dump(calibrator, os.path.join(models_dir, CALIBRATOR_FILENAME))
    payload = {
        "feature_cols": feature_cols,
        "categories": categories,
        "encoded_columns": encoded_columns,
        "metrics": metrics,
    }
    with open(os.path.join(models_dir, FEATURE_LIST_FILENAME), "w") as f:
        json.dump(payload, f, indent=2)


class GrowthPropensityEngine:
    """Loaded once by the scorer. Applies the eligibility guardrails and,
    for eligible accounts, produces a growth propensity score, suggested
    product, indicative quantum and outreach window."""

    def __init__(self, models_dir: str):
        self.available = os.path.exists(os.path.join(models_dir, MODEL_FILENAME))
        if not self.available:
            return
        self.model = joblib.load(os.path.join(models_dir, MODEL_FILENAME))
        self.calibrator = joblib.load(os.path.join(models_dir, CALIBRATOR_FILENAME))
        with open(os.path.join(models_dir, FEATURE_LIST_FILENAME)) as f:
            payload = json.load(f)
        self.feature_cols = payload["feature_cols"]
        self.categories = payload["categories"]
        self.encoded_columns = payload["encoded_columns"]

    def _raw_score(self, engineered_row: pd.DataFrame) -> float:
        X, _ = encode_features(engineered_row, self.feature_cols, self.categories)
        X = align_columns(X, self.encoded_columns).astype(float)
        return float(self.calibrator.predict_proba(X)[:, 1][0])

    def raw_scores_batch(self, engineered: pd.DataFrame) -> np.ndarray:
        """Vectorised scoring for a whole dataframe in a single model call —
        used by scoring.score_dataframe() so portfolio-level scoring doesn't
        pay the one-hot-encoding cost per row."""
        X, _ = encode_features(engineered, self.feature_cols, self.categories)
        X = align_columns(X, self.encoded_columns).astype(float)
        return self.calibrator.predict_proba(X)[:, 1]

    def is_eligible(self, risk_grade: str, fraud_flag: bool, gst_bank_mismatch_flag: bool,
                     sma_migration_probability: float | None) -> tuple[bool, str | None]:
        if risk_grade not in ELIGIBLE_GRADES:
            return False, f"Excluded: risk grade is {risk_grade}, not Green/Yellow."
        if fraud_flag:
            return False, "Excluded: fraud-risk keyword flag present."
        if gst_bank_mismatch_flag:
            return False, "Excluded: GST-bank authenticity mismatch present."
        if sma_migration_probability is not None and sma_migration_probability >= SMA_MIGRATION_EXCLUSION_THRESHOLD:
            return False, "Excluded: elevated near-term SMA migration probability."
        return True, None

    def _suggest_product(self, row: pd.Series) -> str:
        cc_util = float(row.get("cc_utilization_avg_3m", 0))
        gst_growth = float(row.get("gst_turnover_growth_yoy", 0))
        epfo_change = float(row.get("epfo_employee_count_change_6m", 0))
        is_cgtmse = str(row.get("CGTMSE_flag", "No")).lower() == "yes"

        if is_cgtmse and cc_util >= 80:
            return "CGTMSE-backed working-capital top-up"
        if epfo_change >= 8 and gst_growth >= 15:
            return "New machinery / capacity-expansion term loan"
        if cc_util >= 85:
            return "Working-capital (Cash Credit / OD) enhancement"
        return "Working-capital enhancement or new term loan (banker to assess product fit)"

    def _indicative_quantum(self, row: pd.Series) -> float:
        sanctioned_limit = float(row.get("sanctioned_limit", 0) or 0)
        monthly_credit = float(row.get("avg_monthly_bank_credit_6m", 0) or 0)
        surplus_ratio = max(float(row.get("monthly_surplus_ratio", 0) or 0), 0)
        headroom_based = monthly_credit * 3 * (0.5 + surplus_ratio)
        limit_based = sanctioned_limit * 0.25
        quantum = min(max(headroom_based, limit_based * 0.5), limit_based * 1.5) if sanctioned_limit else headroom_based
        return round(max(quantum, 0), -3)  # round to nearest '000

    def _outreach_window(self, growth_score: float) -> str:
        if growth_score >= 70:
            return "Next 30 days"
        if growth_score >= 45:
            return "Next quarter"
        return "Next 6 months (monitor for strengthening signal)"

    def score(self, engineered_row: pd.DataFrame, risk_grade: str, fraud_flag: bool,
              gst_bank_mismatch_flag: bool, sma_migration_probability: float | None) -> dict:
        if not self.available:
            return {"eligible": False, "reason": "Growth propensity model not trained."}

        eligible, reason = self.is_eligible(risk_grade, fraud_flag, gst_bank_mismatch_flag, sma_migration_probability)
        if not eligible:
            return {"eligible": False, "reason": reason}

        row = engineered_row.iloc[0]
        growth_score = round(self._raw_score(engineered_row) * 100, 1)
        return {
            "eligible": True,
            "growth_propensity_score": growth_score,
            "suggested_product": self._suggest_product(row),
            "indicative_quantum": self._indicative_quantum(row),
            "suggested_outreach_window": self._outreach_window(growth_score),
        }

    def score_batch(self, engineered: pd.DataFrame, risk_grades: pd.Series, fraud_flags: pd.Series,
                     gst_bank_mismatch_flags: pd.Series, sma_migration_probabilities: pd.Series) -> pd.DataFrame:
        """Vectorised counterpart of `score()` for portfolio-level scoring:
        one model call for the whole dataframe, eligibility applied as a
        boolean mask, and the (cheap, no-model) product/quantum/outreach
        helpers looped only over the rows that actually need them."""
        n = len(engineered)
        if not self.available:
            return pd.DataFrame({
                "growth_eligible": [False] * n,
                "growth_propensity_score": [None] * n,
                "growth_suggested_product": [None] * n,
            }, index=engineered.index)

        raw_scores = self.raw_scores_batch(engineered) * 100

        eligible_mask = (
            risk_grades.isin(ELIGIBLE_GRADES)
            & (~fraud_flags.astype(bool))
            & (~gst_bank_mismatch_flags.astype(bool))
            & (sma_migration_probabilities.fillna(0) < SMA_MIGRATION_EXCLUSION_THRESHOLD)
        ).to_numpy()

        scores = np.where(eligible_mask, np.round(raw_scores, 1), np.nan)
        products = [
            self._suggest_product(engineered.iloc[i]) if eligible_mask[i] else None
            for i in range(n)
        ]

        return pd.DataFrame({
            "growth_eligible": eligible_mask,
            "growth_propensity_score": scores,
            "growth_suggested_product": products,
        }, index=engineered.index)
