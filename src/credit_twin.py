"""
Bharat MSME Credit Radar - The MSME Credit Twin
===================================================
Section 13 of the design brief. Where a traditional appraisal is a
photograph — accurate on sanction day and ageing every day thereafter —
the Credit Twin is a live feed: every available observation month for a
borrower is re-scored to build a Health Score / PD trajectory, from which
the direction of travel (improving / stable / deteriorating) is read.
That direction matters more than any single month's level.

This module operates on the full borrower-month panel (not just the
latest snapshot used by the portfolio dashboard), which is why it needs
the raw panel dataframe rather than just the scorer.
"""

from __future__ import annotations

import pandas as pd

TREND_IMPROVING = "Improving"
TREND_STABLE = "Stable"
TREND_DETERIORATING = "Deteriorating"

GRADE_ORDER = ["Green", "Yellow", "Amber", "Red", "Black"]
WATCHLIST_GRADES = {"Amber", "Red", "Black"}


def _classify_trend(health_scores: list[float]) -> str:
    if len(health_scores) < 2:
        return TREND_STABLE
    delta = health_scores[-1] - health_scores[0]
    if delta >= 3:
        return TREND_IMPROVING
    if delta <= -3:
        return TREND_DETERIORATING
    return TREND_STABLE


def build_credit_twin(borrower_id: str, raw_panel: pd.DataFrame, scorer) -> dict | None:
    """Re-scores every available month for `borrower_id` and assembles the
    Credit Twin view: trajectory, trend, refreshed PD/SMA migration
    probability, an escalating action recommendation, and a portfolio
    watchlist flag."""
    history = raw_panel[raw_panel["borrower_id"] == borrower_id].sort_values("obs_month")
    if history.empty:
        return None

    trajectory = []
    grade_sequence = []
    for _, row in history.iterrows():
        payload = row.drop(labels=[c for c in ["stress_12m", "growth_need_12m"] if c in row.index]).to_dict()
        result = scorer.score_payload(payload)
        trajectory.append({
            "obs_month": int(row["obs_month"]),
            "pd_12m": result["pd_12m"],
            "health_score": result["health_score"],
            "risk_grade": result["risk_grade"],
        })
        grade_sequence.append(result["risk_grade"])

    current = trajectory[-1]
    latest_full_result = scorer.score_payload(
        history.iloc[-1].drop(labels=[c for c in ["stress_12m", "growth_need_12m"] if c in history.iloc[-1].index]).to_dict()
    )

    trend = _classify_trend([t["health_score"] for t in trajectory])

    escalation_note = None
    if len(grade_sequence) >= 2:
        prev_rank = GRADE_ORDER.index(grade_sequence[-2]) if grade_sequence[-2] in GRADE_ORDER else 0
        curr_rank = GRADE_ORDER.index(grade_sequence[-1]) if grade_sequence[-1] in GRADE_ORDER else 0
        if curr_rank > prev_rank:
            escalation_note = (
                f"Grade migrated {grade_sequence[-2]} -> {grade_sequence[-1]} month-over-month — "
                "action recommendation escalated accordingly."
            )
        elif curr_rank < prev_rank:
            escalation_note = f"Grade improved {grade_sequence[-2]} -> {grade_sequence[-1]} month-over-month."

    return {
        "borrower_id": borrower_id,
        "borrower_name": history.iloc[-1].get("borrower_name", borrower_id),
        "segment": history.iloc[-1].get("segment"),
        "months_observed": len(trajectory),
        "trajectory": trajectory,
        "trend": trend,
        "current": {
            "pd_12m": current["pd_12m"],
            "health_score": current["health_score"],
            "risk_grade": current["risk_grade"],
            "sma_migration_probability": latest_full_result.get("sma_migration_probability"),
            "expected_months_to_stress": latest_full_result.get("expected_months_to_stress"),
        },
        "escalation_note": escalation_note,
        "recommended_action": latest_full_result["recommended_action"],
        "action_checklist": latest_full_result["action_checklist"],
        "on_watchlist": current["risk_grade"] in WATCHLIST_GRADES,
    }
