"""
Bharat MSME Credit Radar - MSME Health Score & Banker Action Engine
======================================================================
Computes the 0-100 MSME Health Score (weighted sub-scores), maps PD to a
risk grade, and produces rule-based banker action recommendations
(including a dedicated CGTMSE suitability recommendation).
"""

from __future__ import annotations

import pandas as pd

RISK_GRADE_BANDS = [
    (0.00, 0.05, "Green"),
    (0.05, 0.10, "Yellow"),
    (0.10, 0.20, "Amber"),
    (0.20, 0.35, "Red"),
    (0.35, 1.01, "Black"),
]

HEALTH_BANDS = [
    (80, 101, "Strong"),
    (65, 80, "Good"),
    (50, 65, "Moderate"),
    (35, 50, "Weak"),
    (0, 35, "Critical"),
]

# Simple policy-style relative risk tiers for sector / geography (distinct from
# the hidden data-generation process) — used only for the small 5% weight in
# the health score, reflecting a credit-policy view of concentration risk.
SECTOR_RISK_TIER = {
    "Textile": 55, "Engineering": 70, "Chemicals": 65, "Food Processing": 75,
    "Gems & Jewellery": 45, "Services": 75, "Retail": 60, "Construction": 50,
}
GEOGRAPHY_RISK_TIER = {
    "Surat": 60, "Rajkot": 62, "Vadodara": 68, "Ahmedabad": 70, "Mumbai": 72,
    "Pune": 72, "Jaipur": 62, "Ludhiana": 55, "Coimbatore": 68,
}


def risk_grade(pd_12m: float) -> str:
    for lo, hi, label in RISK_GRADE_BANDS:
        if lo <= pd_12m < hi:
            return label
    return "Black"


def health_band(score: float) -> str:
    for lo, hi, label in HEALTH_BANDS:
        if lo <= score < hi:
            return label
    return "Critical"


def compute_health_score(row: pd.Series) -> tuple[float, dict]:
    """25% repayment conduct, 20% cash-flow strength, 15% GST authenticity,
    15% business stability, 10% bureau discipline, 10% fraud/integrity, 5% sector/geography."""
    repayment_conduct = 100 - float(row.get("repayment_stress_index", 50))
    cashflow_strength = float(row.get("cashflow_strength_score", 50))
    gst_authenticity = float(row.get("gst_authenticity_score", 50))

    vintage_component = min(float(row.get("business_vintage_years", 5)) / 15 * 100, 100)
    business_stability = 0.7 * float(row.get("operating_stability_score", 50)) + 0.3 * vintage_component

    bureau_discipline = 100 - float(row.get("bureau_stress_score", 50))

    integrity_flags = [
        float(row.get("fraud_keyword_flag", 0)),
        float(row.get("management_quality_keyword_flag", 0)),
        float(row.get("eway_bill_mismatch_flag", 0)),
        1.0 if str(row.get("gst_status", "Active")) != "Active" else 0.0,
    ]
    fraud_integrity = 100 - (sum(integrity_flags) / len(integrity_flags)) * 100

    sector_score = SECTOR_RISK_TIER.get(row.get("sector"), 60)
    geography_score = GEOGRAPHY_RISK_TIER.get(row.get("geography"), 60)
    sector_geo = (sector_score + geography_score) / 2

    sub_scores = {
        "repayment_conduct": round(repayment_conduct, 1),
        "cashflow_strength": round(cashflow_strength, 1),
        "gst_authenticity": round(gst_authenticity, 1),
        "business_stability": round(business_stability, 1),
        "bureau_discipline": round(bureau_discipline, 1),
        "fraud_integrity": round(fraud_integrity, 1),
        "sector_geography": round(sector_geo, 1),
    }

    health_score = (
        0.25 * repayment_conduct + 0.20 * cashflow_strength + 0.15 * gst_authenticity
        + 0.15 * business_stability + 0.10 * bureau_discipline + 0.10 * fraud_integrity
        + 0.05 * sector_geo
    )
    health_score = max(0.0, min(100.0, health_score))
    return round(health_score, 1), sub_scores


ACTION_BULLETS = {
    "Green": [
        "Continue normal monitoring.",
        "Eligible for faster renewal / enhancement if other policy norms are met.",
    ],
    "Yellow": [
        "Review next GST filing.",
        "Monitor bank credits and EMI conduct.",
        "Contact borrower for early engagement.",
    ],
    "Amber": [
        "Conduct field visit within 30 days.",
        "Review debtor ageing and stock statement.",
        "Hold enhancement until risk drivers improve.",
        "Verify GST-bank reconciliation.",
    ],
    "Red": [
        "Move to watchlist.",
        "Conduct stock audit.",
        "Freeze enhancement.",
        "Reduce exposure where applicable.",
        "Review collateral and guarantor strength.",
    ],
    "Black": [
        "Urgent recovery / restructuring review.",
        "Senior credit review.",
        "Legal / collection action where applicable.",
        "Stop additional exposure.",
    ],
}

_DRIVER_CLAUSES = {
    "GST-BANK-MM": "pending GST-bank reconciliation",
    "GST-FIL-DLY": "pending review of GST filing delays",
    "CC-UTIL-HI": "with review of cash credit utilization",
    "BUY-CONC-HI": "with review of buyer concentration",
    "EMI-BOUNCE": "given the EMI bounce pattern",
    "DP-EROSION": "given drawing power erosion",
    "BUR-ENQ-SPIKE": "given the bureau enquiry spike",
    "EPFO-DECLINE": "given declining employee headcount",
    "TXT-STRESS": "given adverse credit officer remarks",
    "RCU-RED-FLAG": "pending RCU / fraud-pattern verification",
    "CASH-VOL-HI": "given elevated cash-flow volatility",
    "ITC-RISK": "pending ITC-to-sales review",
    "DSCR-WEAK": "given weak debt service coverage",
}

_GRADE_HEADLINE = {
    "Green": "Green: continue normal monitoring",
    "Yellow": "Yellow Watch: early engagement recommended",
    "Amber": "Amber Watch: conduct field visit within 30 days",
    "Red": "Red Alert: move to watchlist and conduct stock audit",
    "Black": "Black - Urgent: senior credit review and recovery action",
}


def generate_action_narrative(grade: str, top_risk_codes: list[str]) -> str:
    headline = _GRADE_HEADLINE.get(grade, "Continue monitoring")
    clauses = [_DRIVER_CLAUSES[c] for c in top_risk_codes if c in _DRIVER_CLAUSES]
    if clauses:
        return f"{headline}, {' and '.join(clauses[:2])}."
    return f"{headline}."


def recommended_actions(grade: str) -> list[str]:
    return ACTION_BULLETS.get(grade, ACTION_BULLETS["Green"])


def cgtmse_recommendation(is_cgtmse: bool, pd_12m: float, health_score: float,
                           fraud_flag: bool, low_cashflow: bool, gst_compliance_poor: bool) -> str | None:
    """Special CGTMSE suitability recommendation. Returns None if not a CGTMSE case."""
    if not is_cgtmse:
        return None
    if fraud_flag or gst_compliance_poor:
        return "Not suitable due to cash-flow / fraud / compliance risk."
    if pd_12m <= 0.05 and health_score >= 65:
        return "Suitable for CGTMSE."
    if pd_12m <= 0.20 and not low_cashflow:
        return "Suitable with reduced limit."
    return "Suitable after verification."


def build_recommendation(row: pd.Series, pd_12m: float, health_score: float, top_risk_codes: list[str]) -> dict:
    grade = risk_grade(pd_12m)
    narrative = generate_action_narrative(grade, top_risk_codes)
    actions = recommended_actions(grade)

    is_cgtmse = str(row.get("CGTMSE_flag", "No")).lower() == "yes" or str(row.get("segment", "")) == "CGTMSE"
    cgtmse_note = cgtmse_recommendation(
        is_cgtmse=is_cgtmse,
        pd_12m=pd_12m,
        health_score=health_score,
        fraud_flag=bool(row.get("fraud_keyword_flag", 0)),
        low_cashflow=float(row.get("cashflow_strength_score", 50)) < 50,
        gst_compliance_poor=float(row.get("gst_authenticity_score", 50)) < 50,
    )

    return {
        "risk_grade": grade,
        "recommended_action": narrative,
        "action_checklist": actions,
        "cgtmse_recommendation": cgtmse_note,
    }
