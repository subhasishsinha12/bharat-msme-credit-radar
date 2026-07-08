"""
Bharat MSME Credit Radar - End-to-End Scoring Pipeline
==========================================================
Single entry point used by both the FastAPI service and the Streamlit
dashboard: loads trained artifacts once, accepts either a partial
borrower payload (as a credit officer / API caller would submit) or a
full portfolio dataframe, and returns calibrated PD, risk grade, MSME
health score, data-quality score, SHAP-based reason codes and the
banker action recommendation.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import build_features, get_model_feature_columns, TEXT_COLUMNS  # noqa: E402
from train_model import encode_features, align_columns  # noqa: E402
from explainability import build_explainer, explain_row  # noqa: E402
from action_engine import compute_health_score, health_band, risk_grade, build_recommendation  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

ID_TARGET_COLS = {"record_id", "borrower_id", "borrower_name", "obs_month", "stress_12m"}

# Core fields used to compute the API "data_quality_score" — a curated
# checklist spanning every alternate-data rail (repayment, GST, cash-flow/AA,
# bureau, EPFO, text) that a banker would ideally supply for a trustworthy score.
CORE_FIELDS = [
    "segment", "loan_type", "current_dpd", "emi_bounce_count_6m", "cheque_return_count_6m",
    "cc_utilization_avg_3m", "drawing_power_decline_pct",
    "gst_turnover_growth_yoy", "gst_filing_delay_count_6m", "gstr1_vs_3b_mismatch_pct", "gst_status",
    "bank_credit_to_gst_sales_ratio", "cashflow_volatility_score", "debt_service_coverage_proxy",
    "bureau_score", "bureau_enquiry_count_3m", "bureau_dpd_last_12m",
    "epfo_employee_count_change_6m", "buyer_concentration_top2_pct",
    "cam_remarks",
]


class CreditRadarScorer:
    """Loads model artifacts once; reused across API requests / dashboard sessions."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.model = joblib.load(os.path.join(models_dir, "trained_model.pkl"))
        self.calibrator = joblib.load(os.path.join(models_dir, "calibrator.pkl"))
        with open(os.path.join(models_dir, "feature_list.json")) as f:
            self.feature_list = json.load(f)
        self.vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
        self.text_model = joblib.load(os.path.join(models_dir, "text_risk_model.pkl"))
        with open(os.path.join(models_dir, "default_profile.json")) as f:
            self.defaults = json.load(f)
        self.explainer = build_explainer(self.model)
        self.model_version = self.feature_list.get("best_model_name", "unknown")

    # ------------------------------------------------------------------ #
    # Single-borrower payload scoring (API `/score`, Streamlit manual entry)
    # ------------------------------------------------------------------ #
    def _payload_to_row(self, payload: dict) -> pd.DataFrame:
        row = dict(self.defaults)
        for k, v in payload.items():
            if v is None:
                continue
            row[k] = v
        row["borrower_id"] = payload.get("borrower_id", "MANUAL-ENTRY")
        row["borrower_name"] = payload.get("borrower_name", row["borrower_id"])
        row["obs_month"] = 0
        for tc in TEXT_COLUMNS:
            row.setdefault(tc, "")
        return pd.DataFrame([row])

    def _data_quality_score(self, payload: dict) -> int:
        provided = sum(1 for c in CORE_FIELDS if payload.get(c) not in (None, ""))
        pct = provided / len(CORE_FIELDS) * 100
        return int(round(max(20, min(100, pct))))

    def _encode(self, engineered: pd.DataFrame) -> pd.DataFrame:
        X, _ = encode_features(engineered, self.feature_list["raw_feature_columns"], self.feature_list["categories"])
        X = align_columns(X, self.feature_list["encoded_columns"]).astype(float)
        return X

    def score_payload(self, payload: dict) -> dict:
        raw_row = self._payload_to_row(payload)
        engineered, _ = build_features(raw_row, fit_text_model=False, text_artifacts=(self.vectorizer, self.text_model))
        X = self._encode(engineered)

        pd_12m = float(self.calibrator.predict_proba(X)[:, 1][0])
        grade = risk_grade(pd_12m)
        health_score, health_sub_scores = compute_health_score(engineered.iloc[0])
        dq_score = self._data_quality_score(payload)

        explanation = explain_row(self.explainer, X, top_n=5)
        top_risk_codes = [d["code"] for d in explanation["top_risk_drivers"]]
        recommendation = build_recommendation(engineered.iloc[0], pd_12m, health_score, top_risk_codes)

        return {
            "borrower_id": payload.get("borrower_id", "MANUAL-ENTRY"),
            "pd_12m": round(pd_12m, 4),
            "risk_grade": grade,
            "health_score": health_score,
            "health_band": health_band(health_score),
            "health_sub_scores": health_sub_scores,
            "data_quality_score": dq_score,
            "top_risk_drivers": explanation["top_risk_drivers"],
            "top_strength_drivers": explanation["top_strength_drivers"],
            "recommended_action": recommendation["recommended_action"],
            "action_checklist": recommendation["action_checklist"],
            "cgtmse_recommendation": recommendation["cgtmse_recommendation"],
            "model_version": self.model_version,
        }

    # ------------------------------------------------------------------ #
    # Batch / portfolio scoring
    # ------------------------------------------------------------------ #
    def score_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        engineered, _ = build_features(raw_df, fit_text_model=False, text_artifacts=(self.vectorizer, self.text_model))
        X = self._encode(engineered)

        pd_12m = self.calibrator.predict_proba(X)[:, 1]
        out = raw_df.copy().reset_index(drop=True)
        out["pd_12m"] = pd_12m
        out["risk_grade"] = out["pd_12m"].apply(risk_grade)

        health_results = engineered.apply(compute_health_score, axis=1)
        out["health_score"] = [h[0] for h in health_results]
        out["health_band"] = out["health_score"].apply(health_band)

        # carry a few engineered flags through for portfolio drill-downs
        for col in ["gst_bank_mismatch_flag", "cc_utilization_avg_3m", "business_stress_keyword_flag",
                    "fraud_keyword_flag", "gst_authenticity_score", "cashflow_strength_score",
                    "bureau_stress_score", "repayment_stress_index"]:
            out[col] = engineered[col]

        out["expected_stress_amount"] = out["pd_12m"] * out["outstanding_amount"]
        return out


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce a borrower-month panel to each borrower's most recent month —
    the 'current portfolio' view used for dashboards / portfolio summary."""
    idx = df.groupby("borrower_id")["obs_month"].idxmax()
    return df.loc[idx].reset_index(drop=True)
