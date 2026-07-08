"""
Bharat MSME Credit Radar - CGTMSE Suitability Engine
========================================================
Section 11 of the design brief: a credit engine that only says approve or
reject wastes half its intelligence. The more valuable question for a
collateral-light MSME is structural — is this borrower viable but
under-collateralised, precisely the case CGTMSE exists for? This module
scores guarantee-fit alongside default risk and returns one of four
categories (not a binary), with the specific factors that drove it.

Applies to any borrower where a collateral-light structure is actually
relevant (no collateral, an existing CGTMSE flag, or the CGTMSE segment) —
not just accounts already tagged CGTMSE, since the point of the engine is
to *surface* good CGTMSE candidates who might otherwise be declined or
under-sanctioned for lack of collateral.
"""

from __future__ import annotations

import pandas as pd

CATEGORY_SUITABLE = "Suitable for CGTMSE"
CATEGORY_REDUCED_LIMIT = "Suitable with reduced limit"
CATEGORY_NEEDS_VERIFICATION = "Suitable after additional verification"
CATEGORY_NOT_SUITABLE = "Not suitable"


def is_cgtmse_relevant(row: pd.Series) -> bool:
    """A collateral-light structure is only a meaningful question for
    accounts that are actually collateral-short or already CGTMSE-tagged."""
    return (
        str(row.get("collateral_available", "Yes")).lower() == "no"
        or str(row.get("CGTMSE_flag", "No")).lower() == "yes"
        or str(row.get("segment", "")) == "CGTMSE"
    )


def _viability_score(row: pd.Series) -> float:
    """0-100 composite viability score from the suitability factors listed
    in the design brief: cash-flow sufficiency, GST continuity/authenticity,
    bureau discipline, operating (EPFO) stability, digital collection
    intensity and business vintage."""
    cashflow = float(row.get("cashflow_strength_score", 50))
    gst_auth = float(row.get("gst_authenticity_score", 50))
    bureau_discipline = 100 - float(row.get("bureau_stress_score", 50))
    ops_stability = float(row.get("operating_stability_score", 50))
    digital_intensity = float(row.get("upi_pos_collection_ratio", 0.3)) * 100
    vintage_component = min(float(row.get("business_vintage_years", 5)) / 10 * 100, 100)

    return round(
        0.28 * cashflow + 0.24 * gst_auth + 0.20 * bureau_discipline
        + 0.14 * ops_stability + 0.09 * digital_intensity + 0.05 * vintage_component,
        1,
    )


def _checklist_for_verification(row: pd.Series) -> list[str]:
    checklist = []
    if float(row.get("bank_credit_to_gst_sales_ratio", 1.0)) < 0.6 or float(row.get("bank_credit_to_gst_sales_ratio", 1.0)) > 1.3:
        checklist.append("Complete GST-bank reconciliation.")
    if float(row.get("buyer_concentration_top2_pct", 0)) > 50:
        checklist.append("Obtain buyer confirmation / purchase-order evidence for top-2 buyers.")
    if float(row.get("gstr1_vs_3b_mismatch_pct", 0)) > 10:
        checklist.append("Clarify GSTR-1 vs GSTR-3B mismatch with borrower.")
    if float(row.get("epfo_employee_count_change_6m", 0)) < -10:
        checklist.append("Verify reason for declining EPFO headcount via field visit.")
    if not checklist:
        checklist.append("Verify latest stock statement and bank conduct before sanction.")
    return checklist


def cgtmse_suitability(row: pd.Series, pd_12m: float, health_score: float, fraud_flag: bool) -> dict | None:
    """Returns the 4-category CGTMSE suitability assessment, or None if a
    collateral-light structure isn't a relevant question for this account."""
    if not is_cgtmse_relevant(row):
        return None

    viability = _viability_score(row)
    gst_auth = float(row.get("gst_authenticity_score", 50))

    if fraud_flag or gst_auth < 40:
        return {
            "category": CATEGORY_NOT_SUITABLE,
            "viability_score": viability,
            "rationale": "Fraud-risk or GST compliance/authenticity red flags present; "
                         "a guarantee cover must not become a channel for adverse selection.",
            "suggested_limit_cap_pct_of_request": 0,
            "checklist": [],
        }

    if viability >= 70 and pd_12m <= 0.05 and health_score >= 65:
        return {
            "category": CATEGORY_SUITABLE,
            "viability_score": viability,
            "rationale": "Viable cash flows, clean authenticity, collateral-light profile — "
                         "recommend a guarantee-backed structure at the requested tenor and limit.",
            "suggested_limit_cap_pct_of_request": 100,
            "checklist": [],
        }

    if viability >= 55 and pd_12m <= 0.20:
        return {
            "category": CATEGORY_REDUCED_LIMIT,
            "viability_score": viability,
            "rationale": "Viability confirmed, but cash-flow sufficiency supports a lower exposure "
                         "than requested under guarantee cover.",
            "suggested_limit_cap_pct_of_request": 70,
            "checklist": [],
        }

    if viability >= 40:
        return {
            "category": CATEGORY_NEEDS_VERIFICATION,
            "viability_score": viability,
            "rationale": "Specific gaps identified; suitability can be confirmed after the "
                         "checklist below is closed out.",
            "suggested_limit_cap_pct_of_request": 50,
            "checklist": _checklist_for_verification(row),
        }

    return {
        "category": CATEGORY_NOT_SUITABLE,
        "viability_score": viability,
        "rationale": "Cash-flow, compliance or bureau-discipline weaknesses do not support a "
                     "collateral-light structure at this time.",
        "suggested_limit_cap_pct_of_request": 0,
        "checklist": [],
    }
