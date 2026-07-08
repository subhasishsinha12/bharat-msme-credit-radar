"""
Bharat MSME Credit Radar - FastAPI Scoring Service
=====================================================
Endpoints:
    GET  /health              - liveness check
    POST /score                - score a single borrower (partial payload supported)
    GET  /portfolio-summary    - aggregate portfolio risk view for the loaded book

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
from api.schemas import (  # noqa: E402
    HealthCheckResponse, PortfolioSummaryResponse, ScoreRequest, ScoreResponse,
)

DATA_PATH = os.path.join(ROOT_DIR, "data", "synthetic_msme_data.csv")

app = FastAPI(
    title="Bharat MSME Credit Radar API",
    description="12-month predictive default intelligence and early-warning scoring service for Indian MSME loans. "
                 "Prototype built on synthetic data for the IDBI Innovate 2026 hackathon (Track 04).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

_state: dict = {}


@app.on_event("startup")
def load_artifacts():
    _state["scorer"] = CreditRadarScorer()
    raw = pd.read_csv(DATA_PATH)
    snapshot = latest_snapshot(raw)
    _state["portfolio_scored"] = _state["scorer"].score_dataframe(snapshot)


def get_scorer() -> CreditRadarScorer:
    if "scorer" not in _state:
        load_artifacts()
    return _state["scorer"]


def get_portfolio() -> pd.DataFrame:
    if "portfolio_scored" not in _state:
        load_artifacts()
    return _state["portfolio_scored"]


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
    }
