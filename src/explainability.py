"""
Bharat MSME Credit Radar - SHAP Explainability & Reason Codes
================================================================
Wraps SHAP around the trained tree model to produce, for any scored
borrower row, the top-5 risk drivers and top-5 strength drivers as
standardised, banker-readable reason codes (e.g. GST-FIL-DLY, CC-UTIL-HI).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import shap

# feature_name -> (risk_code, risk_description, strength_code, strength_description)
# risk_* is used when the feature pushes PD UP (positive SHAP contribution);
# strength_* is used when it pushes PD DOWN (negative SHAP contribution).
FEATURE_META: dict[str, tuple[str, str, str, str]] = {
    "current_dpd": ("DPD-HIGH", "Elevated current days-past-due", "DPD-CLEAN", "Account current with no overdue"),
    "max_dpd_last_12m": ("DPD-HIST-HIGH", "High DPD seen in last 12 months", "DPD-HIST-CLEAN", "Clean DPD history over last 12 months"),
    "high_dpd_flag": ("DPD-HIGH", "Account past 30 DPD", "DPD-CLEAN", "Account within standard DPD"),
    "repeated_bounce_flag": ("EMI-BOUNCE", "Repeated EMI / cheque / SI bounce pattern", "REPAY-CLEAN", "No repeated bounce pattern"),
    "emi_bounce_count_6m": ("EMI-BOUNCE", "EMI / ECS bounce pattern observed", "REPAY-CLEAN", "No EMI bounce in last 6 months"),
    "cheque_return_count_6m": ("CHQ-RETURN", "Cheque return instances observed", "REPAY-CLEAN", "No cheque returns in last 6 months"),
    "si_ecs_bounce_count_6m": ("SI-ECS-BOUNCE", "Standing instruction / ECS bounce observed", "REPAY-CLEAN", "SI/ECS instructions honoured regularly"),
    "repayment_regularity_score": ("REPAY-IRREG", "Weak repayment regularity score", "REPAY-CLEAN", "Strong repayment regularity"),
    "dpd_trend_score": ("DPD-TREND-UP", "Deteriorating DPD trend", "DPD-TREND-OK", "Stable / improving DPD trend"),
    "repayment_stress_index": ("REPAY-STRESS", "High composite repayment stress index", "REPAY-STRONG", "Low repayment stress index"),
    "cc_utilization_avg_3m": ("CC-UTIL-HI", "High cash credit utilization (3-month avg)", "CC-UTIL-OK", "Comfortable cash credit utilization"),
    "cc_utilization_avg_6m": ("CC-UTIL-HI", "High cash credit utilization (6-month avg)", "CC-UTIL-OK", "Comfortable cash credit utilization"),
    "cc_utilization_volatility": ("CC-UTIL-VOL", "Volatile cash credit utilization pattern", "CC-UTIL-STABLE", "Stable cash credit utilization pattern"),
    "drawing_power_decline_pct": ("DP-EROSION", "Drawing power erosion observed", "DP-STABLE", "Drawing power stable / improving"),
    "overdue_days_trend": ("OD-TREND-UP", "Rising overdue-days trend", "OD-TREND-OK", "Stable overdue-days trend"),
    "gst_bank_mismatch_flag": ("GST-BANK-MM", "GST turnover not matching bank credits", "GST-BANK-OK", "GST turnover reconciles with bank credits"),
    "bank_credit_to_gst_sales_ratio": ("GST-BANK-MM", "Bank credit to GST sales ratio outside expected range", "GST-BANK-OK", "Bank credit tracks GST sales appropriately"),
    "high_itc_flag": ("ITC-RISK", "High ITC-to-sales ratio", "ITC-OK", "ITC-to-sales ratio within normal range"),
    "itc_to_sales_ratio": ("ITC-RISK", "High ITC-to-sales ratio", "ITC-OK", "ITC-to-sales ratio within normal range"),
    "gst_delay_flag": ("GST-FIL-DLY", "GST filing delay observed", "GST-FIL-OK", "GST filings timely"),
    "gst_filing_delay_count_6m": ("GST-FIL-DLY", "GST filing delay observed", "GST-FIL-OK", "GST filings timely"),
    "gstr1_vs_3b_mismatch_pct": ("GSTR-MISMATCH", "GSTR-1 vs GSTR-3B mismatch", "GSTR-MATCH", "GSTR-1 / GSTR-3B reconcile well"),
    "turnover_spike_flag": ("GST-SPIKE", "Sudden / unexplained turnover spike", "GST-STABLE", "GST turnover stable"),
    "gst_turnover_growth_yoy": ("GST-DECLINE", "Declining GST turnover YoY", "GST-STABLE", "GST turnover stable or growing"),
    "sudden_turnover_spike_flag": ("GST-SPIKE", "Sudden turnover spike flagged", "GST-STABLE", "GST turnover stable"),
    "buyer_concentration_flag": ("BUY-CONC-HI", "High buyer concentration", "BUY-DIVERSE", "Diversified buyer base"),
    "buyer_concentration_top2_pct": ("BUY-CONC-HI", "High buyer concentration (top-2 buyers)", "BUY-DIVERSE", "Diversified buyer base"),
    "supplier_concentration_flag": ("SUP-CONC-HI", "High supplier concentration", "SUP-DIVERSE", "Diversified supplier base"),
    "supplier_concentration_top2_pct": ("SUP-CONC-HI", "High supplier concentration (top-2 suppliers)", "SUP-DIVERSE", "Diversified supplier base"),
    "eway_bill_mismatch_flag": ("EWAY-MM", "E-way bill mismatch observed", "EWAY-OK", "E-way bills consistent"),
    "gst_status": ("GST-STATUS-RISK", "GST registration not active", "GST-STATUS-OK", "GST registration active"),
    "nil_return_count_12m": ("GST-NIL-RETURNS", "Frequent nil GST returns filed", "GST-ACTIVE-FILING", "Regular non-nil GST filings"),
    "low_surplus_flag": ("CASHFLOW-LOW-SURPLUS", "Low / negative monthly cash surplus", "CASHFLOW-SURPLUS-OK", "Healthy monthly cash surplus"),
    "monthly_surplus_ratio": ("CASHFLOW-LOW-SURPLUS", "Low / negative monthly cash surplus ratio", "CASHFLOW-SURPLUS-OK", "Healthy monthly cash surplus ratio"),
    "high_cash_deposit_flag": ("CASH-DEPOSIT-HI", "High cash deposit ratio (informal collections)", "CASH-DEPOSIT-OK", "Healthy digital / cheque collection mix"),
    "cash_deposit_ratio": ("CASH-DEPOSIT-HI", "High cash deposit ratio (informal collections)", "CASH-DEPOSIT-OK", "Healthy digital / cheque collection mix"),
    "low_bank_credit_to_gst_flag": ("GST-BANK-MM", "Bank credits materially below GST sales", "GST-BANK-OK", "Bank credits track GST sales"),
    "cashflow_volatility_flag": ("CASH-VOL-HI", "High cash-flow volatility", "CASH-VOL-OK", "Stable cash-flow pattern"),
    "cashflow_volatility_score": ("CASH-VOL-HI", "High cash-flow volatility score", "CASH-VOL-OK", "Stable cash-flow pattern"),
    "dscr_proxy_flag": ("DSCR-WEAK", "Debt service coverage proxy below 1x", "DSCR-STRONG", "Comfortable debt service coverage"),
    "debt_service_coverage_proxy": ("DSCR-WEAK", "Weak debt service coverage proxy", "DSCR-STRONG", "Strong debt service coverage"),
    "cashflow_strength_score": ("CASHFLOW-WEAK", "Weak overall cash-flow strength score", "CASHFLOW-STRONG", "Strong overall cash-flow profile"),
    "upi_pos_collection_ratio": ("DIGITAL-COLLECTION-LOW", "Low digital (UPI/POS) collection ratio", "DIGITAL-COLLECTION-OK", "Healthy digital collection ratio"),
    "avg_monthly_balance": ("AMB-LOW", "Low average monthly bank balance", "AMB-OK", "Healthy average monthly bank balance"),
    "inward_return_count_6m": ("INWARD-RETURN-HI", "Frequent inward cheque/instrument returns", "INWARD-RETURN-OK", "No material inward returns"),
    "outward_return_count_6m": ("OUTWARD-RETURN-HI", "Frequent outward cheque/instrument returns", "OUTWARD-RETURN-OK", "No material outward returns"),
    "bureau_low_score_flag": ("BUR-SCORE-LOW", "Bureau score below policy threshold", "BUR-SCORE-OK", "Healthy bureau score"),
    "bureau_score": ("BUR-SCORE-LOW", "Low bureau score", "BUR-SCORE-OK", "Healthy bureau score"),
    "enquiry_spike_flag": ("BUR-ENQ-SPIKE", "Bureau enquiry spike observed", "BUR-ENQ-OK", "No unusual bureau enquiry activity"),
    "bureau_enquiry_count_3m": ("BUR-ENQ-SPIKE", "Bureau enquiry spike observed", "BUR-ENQ-OK", "No unusual bureau enquiry activity"),
    "unsecured_exposure_flag": ("UNSEC-EXPOSURE-HI", "High unsecured loan exposure share", "UNSEC-EXPOSURE-OK", "Contained unsecured exposure share"),
    "unsecured_loan_exposure": ("UNSEC-EXPOSURE-HI", "High unsecured loan exposure", "UNSEC-EXPOSURE-OK", "Contained unsecured exposure"),
    "bureau_stress_score": ("BUR-STRESS", "High composite bureau stress score", "BUR-STRONG", "Healthy composite bureau profile"),
    "bureau_dpd_last_12m": ("BUR-DPD-HIGH", "Adverse DPD reported at bureau", "BUR-DPD-CLEAN", "Clean bureau DPD history"),
    "employee_decline_flag": ("EPFO-DECLINE", "Employee headcount decline (EPFO)", "EPFO-STABLE", "Stable / growing employee headcount"),
    "epfo_employee_count_change_6m": ("EPFO-DECLINE", "Employee headcount decline (EPFO)", "EPFO-STABLE", "Stable / growing employee headcount"),
    "salary_irregularity_flag": ("EPFO-SALARY-IRREG", "Irregular salary payment pattern", "EPFO-SALARY-OK", "Regular salary payment pattern"),
    "salary_payment_regularity_score": ("EPFO-SALARY-IRREG", "Irregular salary payment pattern", "EPFO-SALARY-OK", "Regular salary payment pattern"),
    "operating_stability_score": ("OPS-UNSTABLE", "Weak operating stability score", "OPS-STABLE", "Strong operating stability score"),
    "risk_keyword_count": ("TXT-STRESS", "Negative business-stress remarks in credit notes", "TXT-CLEAN", "Predominantly positive credit officer remarks"),
    "text_risk_score": ("TXT-STRESS", "Elevated text-based risk score from remarks", "TXT-CLEAN", "Low text-based risk score from remarks"),
    "text_model_risk_score": ("TXT-STRESS", "NLP model flags adverse remarks", "TXT-CLEAN", "NLP model finds remarks reassuring"),
    "fraud_keyword_flag": ("RCU-RED-FLAG", "RCU / fraud-pattern adverse observation", "RCU-CLEAN", "No RCU / fraud-pattern observation"),
    "business_stress_keyword_flag": ("TXT-STRESS", "Negative business-stress remarks", "TXT-CLEAN", "No business-stress remarks"),
    "collateral_risk_keyword_flag": ("COLLATERAL-RISK", "Adverse stock / collateral inspection remarks", "COLLATERAL-OK", "Satisfactory stock / collateral remarks"),
    "management_quality_keyword_flag": ("MGMT-QUALITY-RISK", "Management / promoter quality concern noted", "MGMT-QUALITY-OK", "No management quality concern noted"),
    "business_vintage_years": ("VINTAGE-LOW", "Limited business vintage", "VINTAGE-OK", "Established business vintage"),
    "sanctioned_limit": ("LIMIT-SIZE", "Facility size contributes to exposure risk", "LIMIT-SIZE-OK", "Facility size within comfortable range"),
    "gst_authenticity_score": ("GST-AUTH-LOW", "Low GST authenticity / compliance score", "GST-AUTH-OK", "Strong GST authenticity / compliance score"),
}

RISK_TREE_MODELS = ("RandomForestClassifier", "XGBClassifier", "LGBMClassifier", "ExtraTreesClassifier",
                     "GradientBoostingClassifier", "DecisionTreeClassifier")


def _humanize(feature_name: str) -> str:
    name = re.sub(r"^(segment|constitution|sector|geography|loan_type|collateral_available|CGTMSE_flag|SMA_status|gst_status)_", r"\1: ", feature_name)
    name = name.replace("_", " ").strip()
    return name[0].upper() + name[1:] if name else name


def lookup_reason(feature_name: str, shap_value: float) -> tuple[str, str]:
    """Return (code, description) for a feature given the sign of its SHAP contribution."""
    base_name = feature_name
    # strip one-hot suffix to try to match the base categorical column meta, else fallback generic.
    if feature_name in FEATURE_META:
        risk_code, risk_desc, strength_code, strength_desc = FEATURE_META[feature_name]
        return (risk_code, risk_desc) if shap_value >= 0 else (strength_code, strength_desc)

    pretty = _humanize(base_name)
    if shap_value >= 0:
        return ("GEN-RISK", f"{pretty} contributes to elevated risk")
    return ("GEN-STRENGTH", f"{pretty} supports a healthier risk profile")


def build_explainer(model):
    model_type = type(model).__name__
    if model_type in RISK_TREE_MODELS:
        return shap.TreeExplainer(model)
    return shap.Explainer(model)


def explain_row(explainer, X_row: pd.DataFrame, top_n: int = 5) -> dict:
    """Compute SHAP values for a single-row dataframe and return top risk /
    strength drivers as banker-readable reason codes."""
    shap_out = explainer(X_row)
    values = shap_out.values
    if values.ndim == 3:  # (n_samples, n_features, n_classes) — take positive class
        values = values[:, :, -1]
    row_values = values[0]

    contributions = list(zip(X_row.columns.tolist(), row_values.tolist()))
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)

    risk_drivers, strength_drivers = [], []
    seen_codes = set()
    for feat, val in sorted(contributions, key=lambda t: t[1], reverse=True):
        if val <= 0:
            continue
        code, desc = lookup_reason(feat, val)
        if code in seen_codes:
            continue
        seen_codes.add(code)
        risk_drivers.append({"code": code, "description": desc, "impact": round(float(val), 4)})
        if len(risk_drivers) >= top_n:
            break

    seen_codes = set()
    for feat, val in sorted(contributions, key=lambda t: t[1]):
        if val >= 0:
            continue
        code, desc = lookup_reason(feat, val)
        if code in seen_codes:
            continue
        seen_codes.add(code)
        strength_drivers.append({"code": code, "description": desc, "impact": round(float(val), 4)})
        if len(strength_drivers) >= top_n:
            break

    base_value = shap_out.base_values[0]
    if hasattr(base_value, "__len__"):
        base_value = base_value[-1]

    return {
        "top_risk_drivers": risk_drivers,
        "top_strength_drivers": strength_drivers,
        "shap_base_value": float(base_value),
    }
