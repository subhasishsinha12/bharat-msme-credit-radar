"""
Bharat MSME Credit Radar - FastAPI Scoring Service
=====================================================
Endpoints:
    GET  /health               - liveness check
    POST /score                 - score a single borrower (partial payload supported)
    GET  /portfolio-summary     - aggregate portfolio risk view for the loaded book
    GET  /credit-twin/{id}      - MSME Credit Twin: monthly trajectory + refreshed PD/action
    GET  /growth-pipeline       - eligible growth-propensity accounts (revenue intelligence)
    GET  /cluster-alerts        - graph contagion cluster alerts

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from scoring import CreditRadarScorer, latest_snapshot  # noqa: E402
from credit_twin import build_credit_twin  # noqa: E402
from graph_contagion import get_cluster_alerts  # noqa: E402
from api.schemas import (  # noqa: E402
    ClusterAlert, CreditTwinResponse, GrowthPipelineAccount, HealthCheckResponse,
    PortfolioSummaryResponse, ScoreRequest, ScoreResponse,
)

DATA_PATH = os.path.join(ROOT_DIR, "data", "synthetic_msme_data.csv")

app = FastAPI(
    title="Bharat MSME Credit Radar API",
    description="12-month predictive default intelligence and early-warning scoring service for Indian MSME loans. "
                 "Prototype built on synthetic data for the IDBI Innovate 2026 hackathon (Track 04).",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

_state: dict = {}


@app.on_event("startup")
def load_artifacts():
    _state["scorer"] = CreditRadarScorer()
    raw = pd.read_csv(DATA_PATH)
    _state["raw_panel"] = raw
    snapshot = latest_snapshot(raw)
    _state["portfolio_scored"] = _state["scorer"].score_dataframe(snapshot)
    _state["scorer"].refresh_cluster_context(_state["portfolio_scored"])


def get_scorer() -> CreditRadarScorer:
    if "scorer" not in _state:
        load_artifacts()
    return _state["scorer"]


def get_portfolio() -> pd.DataFrame:
    if "portfolio_scored" not in _state:
        load_artifacts()
    return _state["portfolio_scored"]


def get_raw_panel() -> pd.DataFrame:
    if "raw_panel" not in _state:
        load_artifacts()
    return _state["raw_panel"]


@app.get("/health", response_model=HealthCheckResponse)
def health():
    return {"status": "ok", "service": "Bharat MSME Credit Radar API"}


@app.post("/score", response_model=ScoreResponse)
def score_borrower(request: ScoreRequest):
    scorer = get_scorer()
    payload = request.model_dump(exclude_none=True)
    try:
        result = scorer.score_payload(payload)
    except Exception as exc:  # pragma: no cover - defensive guard for a hackathon prototype
        raise HTTPException(status_code=400, detail=f"Scoring failed: {exc}") from exc
    return result


@app.get("/portfolio-summary", response_model=PortfolioSummaryResponse)
def portfolio_summary():
    df = get_portfolio()

    accounts_by_grade = df["risk_grade"].value_counts().to_dict()
    exposure_by_grade = df.groupby("risk_grade")["outstanding_amount"].sum().to_dict()

    top10 = (
        df.sort_values("pd_12m", ascending=False)
        .head(10)[["borrower_id", "borrower_name", "segment", "sector", "geography",
                    "outstanding_amount", "pd_12m", "risk_grade", "health_score"]]
        .to_dict(orient="records")
    )

    sector_summary = (
        df.groupby("sector")
        .agg(accounts=("borrower_id", "count"), total_exposure=("outstanding_amount", "sum"),
             avg_pd=("pd_12m", "mean"), expected_stress_amount=("expected_stress_amount", "sum"))
        .reset_index().sort_values("expected_stress_amount", ascending=False)
        .to_dict(orient="records")
    )
    geography_summary = (
        df.groupby("geography")
        .agg(accounts=("borrower_id", "count"), total_exposure=("outstanding_amount", "sum"),
             avg_pd=("pd_12m", "mean"), expected_stress_amount=("expected_stress_amount", "sum"))
        .reset_index().sort_values("expected_stress_amount", ascending=False)
        .to_dict(orient="records")
    )

    cgtmse_book = df[df["cgtmse_suitability_category"].notna()]
    cgtmse_quality = {
        "accounts_assessed": int(len(cgtmse_book)),
        "by_category": cgtmse_book["cgtmse_suitability_category"].value_counts().to_dict(),
        "avg_pd_suitable": float(cgtmse_book.loc[cgtmse_book["cgtmse_suitability_category"] == "Suitable for CGTMSE", "pd_12m"].mean() or 0),
    }

    growth_eligible_df = df[df["growth_eligible"] == True]  # noqa: E712
    growth_pipeline_summary = {
        "eligible_accounts": int(len(growth_eligible_df)),
        "avg_growth_propensity_score": float(growth_eligible_df["growth_propensity_score"].mean() or 0),
        "total_exposure_of_eligible": float(growth_eligible_df["outstanding_amount"].sum()),
    }

    return {
        "total_accounts": int(len(df)),
        "total_exposure": float(df["outstanding_amount"].sum()),
        "accounts_by_grade": accounts_by_grade,
        "exposure_by_grade": {k: float(v) for k, v in exposure_by_grade.items()},
        "expected_stress_amount": float(df["expected_stress_amount"].sum()),
        "average_health_score": float(df["health_score"].mean()),
        "average_pd": float(df["pd_12m"].mean()),
        "top_10_high_risk_accounts": top10,
        "sector_wise_summary": sector_summary,
        "geography_wise_summary": geography_summary,
        "cgtmse_portfolio_quality": cgtmse_quality,
        "growth_pipeline_summary": growth_pipeline_summary,
    }


@app.get("/credit-twin/{borrower_id}", response_model=CreditTwinResponse)
def credit_twin(borrower_id: str):
    scorer = get_scorer()
    raw_panel = get_raw_panel()
    twin = build_credit_twin(borrower_id, raw_panel, scorer)
    if twin is None:
        raise HTTPException(status_code=404, detail=f"No records found for borrower_id={borrower_id}")
    return twin


@app.get("/growth-pipeline", response_model=list[GrowthPipelineAccount])
def growth_pipeline(limit: int = 50):
    df = get_portfolio()
    eligible = df[df["growth_eligible"] == True].sort_values(  # noqa: E712
        "growth_propensity_score", ascending=False
    ).head(limit)
    return eligible[["borrower_id", "borrower_name", "segment", "sector", "geography",
                      "risk_grade", "growth_propensity_score", "growth_suggested_product"]].to_dict(orient="records")


@app.get("/cluster-alerts", response_model=list[ClusterAlert])
def cluster_alerts(limit: int = 10):
    df = get_portfolio()
    return get_cluster_alerts(df, pd_col="pd_12m", top_n=limit)
