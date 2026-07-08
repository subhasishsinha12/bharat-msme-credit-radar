"""Pydantic request/response schemas for the Bharat MSME Credit Radar API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ScoreRequest(BaseModel):
    """Borrower scoring request. Only `borrower_id` is required — every other
    field is optional and, if omitted, is imputed from the training
    population's typical (median/mode) value. Supplying more fields raises
    the returned `data_quality_score` and the fidelity of the PD estimate."""

    model_config = ConfigDict(extra="allow", json_schema_extra={
        "example": {
            "borrower_id": "B000123",
            "segment": "Manufacturer",
            "loan_type": "Cash Credit",
            "gst_turnover_growth_yoy": -12.5,
            "gst_filing_delay_count_6m": 3,
            "gstr1_vs_3b_mismatch_pct": 18.2,
            "bank_credit_to_gst_sales_ratio": 0.62,
            "cc_utilization_avg_3m": 94.5,
            "emi_bounce_count_6m": 2,
            "bureau_score": 682,
            "bureau_enquiry_count_3m": 5,
            "buyer_concentration_top2_pct": 61,
            "epfo_employee_count_change_6m": -18,
            "cam_remarks": "cash flow stress visible and buyer concentration high",
        }
    })

    borrower_id: str = Field(..., description="Unique borrower / loan account identifier")

    # Borrower profile
    segment: Optional[str] = None
    constitution: Optional[str] = None
    sector: Optional[str] = None
    geography: Optional[str] = None
    business_vintage_years: Optional[float] = None
    loan_type: Optional[str] = None
    sanctioned_limit: Optional[float] = None
    outstanding_amount: Optional[float] = None
    collateral_available: Optional[str] = None
    CGTMSE_flag: Optional[str] = None

    # Repayment / conduct
    current_dpd: Optional[float] = None
    max_dpd_last_12m: Optional[float] = None
    emi_bounce_count_6m: Optional[float] = None
    cheque_return_count_6m: Optional[float] = None
    si_ecs_bounce_count_6m: Optional[float] = None
    cc_utilization_avg_3m: Optional[float] = None
    cc_utilization_avg_6m: Optional[float] = None
    drawing_power_decline_pct: Optional[float] = None
    overdue_days_trend: Optional[float] = None
    SMA_status: Optional[str] = None

    # GST
    gst_turnover_12m: Optional[float] = None
    gst_turnover_growth_yoy: Optional[float] = None
    gst_filing_delay_count_6m: Optional[float] = None
    gstr1_vs_3b_mismatch_pct: Optional[float] = None
    itc_to_sales_ratio: Optional[float] = None
    gst_status: Optional[str] = None
    buyer_concentration_top2_pct: Optional[float] = None
    supplier_concentration_top2_pct: Optional[float] = None

    # Bank / AA cash-flow
    bank_credit_to_gst_sales_ratio: Optional[float] = None
    cash_deposit_ratio: Optional[float] = None
    cashflow_volatility_score: Optional[float] = None
    debt_service_coverage_proxy: Optional[float] = None
    monthly_surplus_ratio: Optional[float] = None

    # Bureau
    bureau_score: Optional[float] = None
    bureau_enquiry_count_3m: Optional[float] = None
    bureau_dpd_last_12m: Optional[float] = None
    unsecured_loan_exposure: Optional[float] = None
    total_obligation: Optional[float] = None

    # EPFO
    epfo_employee_count: Optional[float] = None
    epfo_employee_count_change_6m: Optional[float] = None
    salary_payment_regularity_score: Optional[float] = None

    # Free-text remarks
    cam_remarks: Optional[str] = None
    fi_remarks: Optional[str] = None
    rcu_remarks: Optional[str] = None
    collection_remarks: Optional[str] = None
    stock_inspection_remarks: Optional[str] = None


class ReasonCode(BaseModel):
    code: str
    description: str
    impact: float


class CgtmseSuitability(BaseModel):
    category: str
    viability_score: float
    rationale: str
    suggested_limit_cap_pct_of_request: int
    checklist: list[str]


class GrowthPropensity(BaseModel):
    eligible: bool
    reason: Optional[str] = None
    growth_propensity_score: Optional[float] = None
    suggested_product: Optional[str] = None
    indicative_quantum: Optional[float] = None
    suggested_outreach_window: Optional[str] = None


class ScoreResponse(BaseModel):
    """The Common Interpretation Layer output contract (Section 8 / Layer 5
    of the design brief): identical shape regardless of which segment model
    (or the global fallback) produced the score."""

    borrower_id: str
    pd_12m: float
    risk_grade: str
    health_score: float
    health_band: str
    health_sub_scores: Optional[dict] = None
    data_quality_score: int
    model_confidence: str
    sma_migration_probability: Optional[float] = None
    expected_months_to_stress: Optional[float] = None
    segment_benchmark_percentile: Optional[float] = None
    cluster_stress_index: float
    in_elevated_cluster: bool
    top_risk_drivers: list[ReasonCode]
    top_strength_drivers: list[ReasonCode]
    recommended_action: str
    action_checklist: list[str]
    cgtmse_recommendation: Optional[str] = None
    cgtmse_suitability: Optional[CgtmseSuitability] = None
    growth_propensity: GrowthPropensity
    model_version: str


class HealthCheckResponse(BaseModel):
    status: str
    service: str


class PortfolioSummaryResponse(BaseModel):
    total_accounts: int
    total_exposure: float
    accounts_by_grade: dict
    exposure_by_grade: dict
    expected_stress_amount: float
    average_health_score: float
    average_pd: float
    top_10_high_risk_accounts: list[dict]
    sector_wise_summary: list[dict]
    geography_wise_summary: list[dict]
    cgtmse_portfolio_quality: dict
    growth_pipeline_summary: dict


class CreditTwinTrajectoryPoint(BaseModel):
    obs_month: int
    pd_12m: float
    health_score: float
    risk_grade: str


class CreditTwinCurrent(BaseModel):
    pd_12m: float
    health_score: float
    risk_grade: str
    sma_migration_probability: Optional[float] = None
    expected_months_to_stress: Optional[float] = None


class CreditTwinResponse(BaseModel):
    borrower_id: str
    borrower_name: str
    segment: Optional[str] = None
    months_observed: int
    trajectory: list[CreditTwinTrajectoryPoint]
    trend: str
    current: CreditTwinCurrent
    escalation_note: Optional[str] = None
    recommended_action: str
    action_checklist: list[str]
    on_watchlist: bool


class ClusterAlert(BaseModel):
    anchor_buyer_id: str
    cluster_id: str
    sector: str
    geography: str
    n_accounts: int
    avg_pd: float
    cluster_stress_index: float
    total_exposure: float
    message: str


class GrowthPipelineAccount(BaseModel):
    borrower_id: str
    borrower_name: str
    segment: str
    sector: str
    geography: str
    risk_grade: str
    growth_propensity_score: Optional[float] = None
    growth_suggested_product: Optional[str] = None
