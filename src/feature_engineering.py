"""
Bharat MSME Credit Radar - Feature Engineering
================================================
Transforms raw borrower-month fields (repayment, GST, bank/AA cash-flow,
bureau, EPFO, and free-text remarks) into risk-oriented model features
organised by the feature families required by the design spec.

Also contains a lightweight TF-IDF + Logistic Regression "first-stage" text
risk model as an optional enrichment on top of the keyword-based NLP
features (used for interpretability and speed in the prototype).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TEXT_COLUMNS = ["cam_remarks", "fi_remarks", "rcu_remarks", "collection_remarks", "stock_inspection_remarks"]

POSITIVE_KEYWORDS = [
    "satisfactorily", "satisfactory", "adequate", "genuine", "diversified", "regular",
    "healthy", "no adverse", "no discrepancy", "matches book records", "in line with projections",
    "operational with visible activity", "realised on due date", "cross-verified",
]
NEGATIVE_KEYWORDS = [
    "stress", "closed during visit", "not matching", "frequent promise to pay",
    "concentrat", "delayed", "suspected", "discrepancy", "inconclusive",
    "unreachable", "returned for insufficient funds", "ageing beyond", "obsolete",
    "scaled down", "declining", "could not be completed", "diverting funds",
    "common mobile number", "common email id", "access issue",
]
FRAUD_KEYWORDS = [
    "common mobile number", "common email id", "related party transactions suspected",
    "diverting funds to group entity suspected", "discrepancy noted in kyc documents",
    "address verification inconclusive",
]
BUSINESS_STRESS_KEYWORDS = [
    "cash flow stress", "declining turnover", "shop found closed", "frequent promise to pay",
    "unit operations appear scaled down", "borrower unreachable",
]
COLLATERAL_RISK_KEYWORDS = [
    "stock not matching", "stock ageing beyond", "obsolete stock",
    "stock verification could not be completed", "property access issue",
]
MANAGEMENT_QUALITY_KEYWORDS = [
    "related party transactions suspected", "promoter diverting funds", "discrepancy noted in kyc",
    "common mobile number", "common email id",
]


def _count_hits(text: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


def _any_hit(text: str, keywords: list[str]) -> int:
    return int(any(kw in text for kw in keywords))


def _minmax_scale(s: pd.Series, invert: bool = False) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        scaled = pd.Series(50.0, index=s.index)
    else:
        scaled = (s - lo) / (hi - lo) * 100
    return 100 - scaled if invert else scaled


def add_repayment_features(df: pd.DataFrame) -> pd.DataFrame:
    df["high_dpd_flag"] = (df["current_dpd"] > 30).astype(int)
    df["repeated_bounce_flag"] = (
        (df["emi_bounce_count_6m"] + df["cheque_return_count_6m"] + df["si_ecs_bounce_count_6m"]) >= 3
    ).astype(int)

    trend_raw = df["overdue_days_trend"].clip(lower=0) + 0.5 * (df["max_dpd_last_12m"] - df["current_dpd"]).clip(lower=0)
    df["dpd_trend_score"] = _minmax_scale(trend_raw).round(1)

    stress_raw = (
        0.35 * _minmax_scale(df["current_dpd"])
        + 0.20 * _minmax_scale(df["emi_bounce_count_6m"] + df["cheque_return_count_6m"] + df["si_ecs_bounce_count_6m"])
        + 0.20 * _minmax_scale(df["cc_utilization_avg_3m"])
        + 0.15 * df["dpd_trend_score"]
        + 0.10 * _minmax_scale(df["drawing_power_decline_pct"])
    )
    df["repayment_stress_index"] = stress_raw.clip(0, 100).round(1)
    return df


def add_gst_features(df: pd.DataFrame) -> pd.DataFrame:
    df["gst_bank_mismatch_flag"] = (
        (df["bank_credit_to_gst_sales_ratio"] < 0.6) | (df["bank_credit_to_gst_sales_ratio"] > 1.3)
    ).astype(int)
    df["high_itc_flag"] = (df["itc_to_sales_ratio"] > 0.75).astype(int)
    df["gst_delay_flag"] = (df["gst_filing_delay_count_6m"] >= 2).astype(int)
    df["turnover_spike_flag"] = (
        (df["sudden_turnover_spike_flag"] == 1) | (df["gst_turnover_growth_yoy"].abs() > 40)
    ).astype(int)
    df["buyer_concentration_flag"] = (df["buyer_concentration_top2_pct"] > 50).astype(int)
    df["supplier_concentration_flag"] = (df["supplier_concentration_top2_pct"] > 50).astype(int)

    penalty = (
        12 * df["gst_bank_mismatch_flag"] + 10 * df["high_itc_flag"] + 10 * df["gst_delay_flag"]
        + 8 * df["turnover_spike_flag"] + 8 * df["buyer_concentration_flag"] + 6 * df["supplier_concentration_flag"]
        + 15 * (df["gst_status"] != "Active").astype(int)
        + df["gstr1_vs_3b_mismatch_pct"].clip(0, 40) * 0.5
        + df["eway_bill_mismatch_flag"] * 8
    )
    df["gst_authenticity_score"] = (100 - penalty).clip(0, 100).round(1)
    return df


def add_cashflow_features(df: pd.DataFrame) -> pd.DataFrame:
    df["low_surplus_flag"] = (df["monthly_surplus_ratio"] < 0).astype(int)
    df["high_cash_deposit_flag"] = (df["cash_deposit_ratio"] > 0.4).astype(int)
    df["low_bank_credit_to_gst_flag"] = (df["bank_credit_to_gst_sales_ratio"] < 0.6).astype(int)
    df["cashflow_volatility_flag"] = (df["cashflow_volatility_score"] > 50).astype(int)
    df["dscr_proxy_flag"] = (df["debt_service_coverage_proxy"] < 1.0).astype(int)

    strength_raw = (
        30 * (1 - df["low_surplus_flag"])
        + 20 * (1 - df["high_cash_deposit_flag"])
        + 15 * (1 - df["low_bank_credit_to_gst_flag"])
        + 15 * (1 - df["cashflow_volatility_flag"])
        + 20 * (1 - df["dscr_proxy_flag"])
    )
    df["cashflow_strength_score"] = strength_raw.clip(0, 100).round(1)
    return df


def add_bureau_features(df: pd.DataFrame) -> pd.DataFrame:
    df["bureau_low_score_flag"] = (df["bureau_score"] < 650).astype(int)
    df["enquiry_spike_flag"] = (df["bureau_enquiry_count_3m"] >= 4).astype(int)
    unsecured_ratio = df["unsecured_loan_exposure"] / df["total_obligation"].replace(0, np.nan)
    df["unsecured_exposure_flag"] = (unsecured_ratio.fillna(0) > 0.4).astype(int)

    penalty = (
        35 * df["bureau_low_score_flag"] + 25 * df["enquiry_spike_flag"] + 20 * df["unsecured_exposure_flag"]
        + _minmax_scale(df["bureau_dpd_last_12m"]) * 0.20
    )
    df["bureau_stress_score"] = penalty.clip(0, 100).round(1)
    return df


def add_epfo_features(df: pd.DataFrame) -> pd.DataFrame:
    df["employee_decline_flag"] = (df["epfo_employee_count_change_6m"] < -10).astype(int)
    df["salary_irregularity_flag"] = (df["salary_payment_regularity_score"] < 70).astype(int)

    stability_raw = (
        50 * (1 - df["employee_decline_flag"])
        + 30 * (1 - df["salary_irregularity_flag"])
        + 0.20 * df["salary_payment_regularity_score"]
    )
    df["operating_stability_score"] = stability_raw.clip(0, 100).round(1)
    return df


RELATED_PARTY_KEYWORDS = [
    "related party transactions suspected", "promoter diverting funds to group entity suspected",
]
STOCK_STATEMENT_DELAY_KEYWORDS = ["stock verification could not be completed"]


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    combined = df[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1).str.lower()
    df["_combined_text"] = combined
    fi_text = df["fi_remarks"].fillna("").str.lower()
    cam_text = df["cam_remarks"].fillna("").str.lower()
    stock_text = df["stock_inspection_remarks"].fillna("").str.lower()

    df["risk_keyword_count"] = combined.apply(lambda t: _count_hits(t, NEGATIVE_KEYWORDS))
    df["positive_keyword_count"] = combined.apply(lambda t: _count_hits(t, POSITIVE_KEYWORDS))
    df["fraud_keyword_flag"] = combined.apply(lambda t: _any_hit(t, FRAUD_KEYWORDS))
    df["business_stress_keyword_flag"] = combined.apply(lambda t: _any_hit(t, BUSINESS_STRESS_KEYWORDS))
    df["collateral_risk_keyword_flag"] = combined.apply(lambda t: _any_hit(t, COLLATERAL_RISK_KEYWORDS))
    df["management_quality_keyword_flag"] = combined.apply(lambda t: _any_hit(t, MANAGEMENT_QUALITY_KEYWORDS))

    # Source-specific flags (rather than only the combined-text flags above) so
    # reason codes can point a banker to the exact remark type that moved the
    # score -- e.g. FI-NEG-RMK cites the field-visit report specifically,
    # not "some text somewhere."
    df["fi_negative_remark_flag"] = fi_text.apply(lambda t: _any_hit(t, NEGATIVE_KEYWORDS))
    df["related_party_keyword_flag"] = cam_text.apply(lambda t: _any_hit(t, RELATED_PARTY_KEYWORDS))
    df["stock_statement_delay_flag"] = stock_text.apply(lambda t: _any_hit(t, STOCK_STATEMENT_DELAY_KEYWORDS))

    raw_score = (
        df["risk_keyword_count"] * 12
        - df["positive_keyword_count"] * 5
        + df["fraud_keyword_flag"] * 20
        + df["business_stress_keyword_flag"] * 15
        + df["collateral_risk_keyword_flag"] * 12
        + df["management_quality_keyword_flag"] * 15
    )
    df["text_risk_score"] = raw_score.clip(0, 100).round(1)
    df = df.drop(columns=["_combined_text"])
    return df


def fit_text_tfidf_model(df: pd.DataFrame, target_col: str = "stress_12m"):
    """Optional first-stage TF-IDF + Logistic Regression text risk model.
    Returns (vectorizer, model). Used to enrich `text_model_risk_score`."""
    combined = df[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1).str.lower()
    combined = combined.apply(lambda t: re.sub(r"[^a-z\s]", " ", t))

    vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2), min_df=5)
    X_text = vectorizer.fit_transform(combined)

    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X_text, df[target_col])
    return vectorizer, model


def apply_text_tfidf_model(df: pd.DataFrame, vectorizer, model) -> pd.DataFrame:
    combined = df[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1).str.lower()
    combined = combined.apply(lambda t: re.sub(r"[^a-z\s]", " ", t))
    X_text = vectorizer.transform(combined)
    df["text_model_risk_score"] = (model.predict_proba(X_text)[:, 1] * 100).round(1)
    return df


def build_features(df: pd.DataFrame, fit_text_model: bool = True, text_artifacts: tuple | None = None) -> tuple[pd.DataFrame, tuple | None]:
    """Run the full feature engineering pipeline. Returns (engineered_df, text_artifacts)."""
    df = df.copy()
    df = add_repayment_features(df)
    df = add_gst_features(df)
    df = add_cashflow_features(df)
    df = add_bureau_features(df)
    df = add_epfo_features(df)
    df = add_text_features(df)

    if text_artifacts is not None:
        vectorizer, model = text_artifacts
        df = apply_text_tfidf_model(df, vectorizer, model)
    elif fit_text_model and "stress_12m" in df.columns:
        vectorizer, model = fit_text_tfidf_model(df)
        df = apply_text_tfidf_model(df, vectorizer, model)
        text_artifacts = (vectorizer, model)
    else:
        df["text_model_risk_score"] = 0.0

    return df, text_artifacts


NON_MODEL_COLUMNS = [
    "record_id", "borrower_id", "borrower_name", "obs_month",
    "cam_remarks", "fi_remarks", "rcu_remarks", "collection_remarks", "stock_inspection_remarks",
    "stress_12m", "growth_need_12m",
    # High-cardinality identifiers used by the graph contagion overlay
    # (src/graph_contagion.py), not fed into the PD models directly.
    "cluster_id", "anchor_buyer_id",
]
CATEGORICAL_COLUMNS = [
    "segment", "constitution", "sector", "geography", "loan_type",
    "collateral_available", "CGTMSE_flag", "SMA_status", "gst_status",
]


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_MODEL_COLUMNS]


if __name__ == "__main__":
    raw = pd.read_csv("data/synthetic_msme_data.csv")
    engineered, _ = build_features(raw)
    print(f"Engineered dataframe shape: {engineered.shape}")
    print(f"Model feature columns: {len(get_model_feature_columns(engineered))}")
