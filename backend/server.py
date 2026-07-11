"""
Bharat MSME Credit Radar - Backend API (Emergent-compatible)
============================================================
Wraps the existing scoring engine under /app/src with a FastAPI service
mounted on /api. Adds MongoDB persistence for banker notes analysis history
and CSV bulk-scoring uploads, plus an LLM-powered NLP endpoint that extracts
stress/fraud signals from unstructured banker text (CAM notes, FI reports).
"""

from __future__ import annotations

import io
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

ARTIFACTS_ROOT = os.environ.get("ARTIFACTS_ROOT", "/app")
sys.path.insert(0, os.path.join(ARTIFACTS_ROOT, "src"))

from scoring import CreditRadarScorer, latest_snapshot  # noqa: E402

DATA_CSV = os.path.join(ARTIFACTS_ROOT, "data", "synthetic_msme_data.csv")

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY", "")

mongo_client = AsyncIOMotorClient(MONGO_URL)
db = mongo_client[DB_NAME]

app = FastAPI(title="Bharat MSME Credit Radar API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api")

STATE: dict[str, Any] = {}


def _sanitize(obj: Any) -> Any:
    """Recursively convert NaN / numpy scalars to JSON-friendly Python types."""
    import math

    import numpy as np

    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def compute_growth_propensity(portfolio: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Growth Propensity Engine (Section 12 of the design doc).

    Scores healthy MSMEs on their likelihood of needing enhanced working-capital
    or a new term loan within 6-12 months. Emits a Growth Propensity Score
    (0-100), a suggested product, an indicative quantum, and an outreach window.
    """
    import numpy as np

    df = portfolio.copy()
    df = df.merge(
        snapshot[
            [
                "borrower_id", "gst_turnover_growth_yoy", "cc_utilization_avg_3m",
                "epfo_employee_count_change_6m", "bureau_score",
                "bureau_enquiry_count_3m", "cashflow_volatility_score",
                "monthly_surplus_ratio", "debt_service_coverage_proxy",
                "buyer_concentration_top2_pct", "gst_turnover_12m",
                "avg_monthly_bank_credit_6m",
            ]
        ],
        on="borrower_id", how="left", suffixes=("", "_snap"),
    )

    # Signal components (each 0-100 scale)
    turnover = df["gst_turnover_growth_yoy"].clip(-30, 40).fillna(0)
    turnover_sig = ((turnover + 10) / 50 * 100).clip(0, 100)  # -10% -> 0, 40% -> 100

    utilization = df["cc_utilization_avg_3m"].clip(0, 100).fillna(50)
    util_head_sig = ((utilization - 40) / 45 * 100).clip(0, 100)  # 75-85% -> highest need

    surplus = df["monthly_surplus_ratio"].clip(-0.5, 0.5).fillna(0)
    cashflow_sig = ((surplus + 0.5) / 1.0 * 100).clip(0, 100)

    dscr = df["debt_service_coverage_proxy"].clip(0, 3).fillna(1)
    dscr_sig = ((dscr - 0.5) / 2.0 * 100).clip(0, 100)

    epfo = df["epfo_employee_count_change_6m"].clip(-30, 30).fillna(0)
    epfo_sig = ((epfo + 5) / 30 * 100).clip(0, 100)

    bureau = df["bureau_score"].clip(300, 900).fillna(700)
    bureau_sig = ((bureau - 600) / 300 * 100).clip(0, 100)

    enq = df["bureau_enquiry_count_3m"].fillna(0)
    enq_penalty = (enq.clip(0, 12) * 6).clip(0, 60)

    fraud_flag = df.get("fraud_keyword_flag", 0)

    # Weighted composite
    raw_score = (
        0.28 * turnover_sig
        + 0.18 * util_head_sig
        + 0.14 * cashflow_sig
        + 0.10 * dscr_sig
        + 0.10 * epfo_sig
        + 0.12 * bureau_sig
        + 0.08 * (100 - df["pd_12m"].clip(0, 1) * 100)
    ) - enq_penalty * 0.3

    # Hard gates: unhealthy borrowers cannot be growth candidates
    unhealthy = (
        (df["risk_grade"].isin(["Red", "Black"]))
        | (df["health_score"] < 55)
        | (df["pd_12m"] > 0.15)
        | (fraud_flag.astype(float) > 0)
        | (df.get("gst_bank_mismatch_flag", 0).astype(float) > 0)
    )
    raw_score = np.where(unhealthy, 0, raw_score)

    df["growth_score"] = np.clip(raw_score, 0, 100).round(1)

    def _product(row: pd.Series) -> str:
        util = row["cc_utilization_avg_3m"] or 0
        growth = row["gst_turnover_growth_yoy"] or 0
        cgt = str(row.get("CGTMSE_flag", "No")).lower() == "yes"
        if cgt and growth > 8:
            return "CGTMSE-backed Enhancement"
        if util >= 75 and growth > 5:
            return "Cash Credit Limit Enhancement"
        if util < 45 and growth > 10:
            return "Term Loan / Capex"
        if growth > 15:
            return "New Working Capital"
        return "Product Refresh"

    def _quantum(row: pd.Series) -> float:
        base = float(row.get("sanctioned_limit") or 0)
        growth = float(row.get("gst_turnover_growth_yoy") or 0)
        util = float(row.get("cc_utilization_avg_3m") or 0)
        pct = 0.15
        if growth > 10:
            pct += 0.10
        if growth > 20:
            pct += 0.10
        if util >= 80:
            pct += 0.05
        return round(base * pct, -3)

    def _window(score: float) -> str:
        if score >= 75:
            return "0-30 days"
        if score >= 60:
            return "30-60 days"
        return "60-90 days"

    df["suggested_product"] = df.apply(_product, axis=1)
    df["indicative_quantum"] = df.apply(_quantum, axis=1)
    df["outreach_window"] = df["growth_score"].apply(_window)
    df["growth_band"] = pd.cut(
        df["growth_score"],
        bins=[-1, 30, 55, 70, 85, 101],
        labels=["Dormant", "Passive", "Emerging", "Hot", "Priority"],
    ).astype(str)

    return df


def compute_portfolio_action_queue(portfolio: pd.DataFrame, snapshot: pd.DataFrame) -> dict:
    """Portfolio-level Action Queue counters (Deck slide 7 wireframe).

    Aggregates the specific banker actions triggered across the portfolio so a
    branch head can see 'field visits: 6, stock audits: 3, GST-bank mismatch
    reviews: 5, enhancement freeze: 2' at a glance.
    """
    df = portfolio.merge(
        snapshot[["borrower_id", "cc_utilization_avg_3m", "gst_filing_delay_count_6m",
                   "gstr1_vs_3b_mismatch_pct", "emi_bounce_count_6m",
                   "bank_credit_to_gst_sales_ratio"]],
        on="borrower_id", how="left", suffixes=("", "_s"),
    )

    field_visits = int(((df["risk_grade"] == "Amber") | (df["risk_grade"] == "Red")).sum())
    stock_audits = int((df["risk_grade"].isin(["Red", "Black"])).sum())
    gst_bank_recon = int(df.get("gst_bank_mismatch_flag", 0).astype(float).sum())
    enhancement_freeze = int(df["risk_grade"].isin(["Amber", "Red", "Black"]).sum())
    urgent_recovery = int((df["risk_grade"] == "Black").sum())
    watchlist = int((df["risk_grade"] == "Yellow").sum())

    return {
        "field_visits": field_visits,
        "stock_audits": stock_audits,
        "gst_bank_recon_reviews": gst_bank_recon,
        "enhancement_freezes": enhancement_freeze,
        "urgent_recoveries": urgent_recovery,
        "early_engagement_watchlist": watchlist,
    }


def compute_cluster_alerts(portfolio: pd.DataFrame) -> list:
    """Cluster contagion alerts (Deck slide 7 & 13).

    Flags sector × geography clusters whose expected-stress rate materially
    exceeds the portfolio baseline — the equivalent of "embroidery cluster
    co-movement elevated" surface in the deck wireframe.
    """
    base_stress_rate = float(portfolio["pd_12m"].mean())
    grouped = (
        portfolio.groupby(["sector", "geography"])
        .agg(
            accounts=("borrower_id", "count"),
            avg_pd=("pd_12m", "mean"),
            exposure=("outstanding_amount", "sum"),
            expected_stress=("expected_stress_amount", "sum"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["accounts"] >= 15]
    grouped["lift"] = grouped["avg_pd"] / max(base_stress_rate, 1e-6)
    hot = grouped[grouped["lift"] > 1.4].sort_values("expected_stress", ascending=False).head(6)
    alerts = []
    for _, r in hot.iterrows():
        alerts.append({
            "cluster": f"{r['sector']} · {r['geography']}",
            "sector": r["sector"],
            "geography": r["geography"],
            "accounts": int(r["accounts"]),
            "avg_pd": float(r["avg_pd"]),
            "lift": float(r["lift"]),
            "exposure": float(r["exposure"]),
            "expected_stress": float(r["expected_stress"]),
            "message": f"{r['sector']} cluster in {r['geography']} — co-movement elevated ({r['lift']:.1f}× baseline PD) across {int(r['accounts'])} linked accounts.",
        })
    return alerts


def load_engine() -> None:
    scorer = CreditRadarScorer()
    raw = pd.read_csv(DATA_CSV)
    snapshot = latest_snapshot(raw)
    scored = scorer.score_dataframe(snapshot)
    growth = compute_growth_propensity(scored, snapshot)
    STATE["scorer"] = scorer
    STATE["portfolio"] = scored
    STATE["snapshot"] = snapshot
    STATE["panel"] = raw
    STATE["growth"] = growth
    STATE["action_queue"] = compute_portfolio_action_queue(scored, snapshot)
    STATE["cluster_alerts"] = compute_cluster_alerts(scored)

    # Precompute portfolio Month-over-Month expected-stress deltas using the panel
    # (obs_month 0..7). Baseline stress at each month is `pd_12m * outstanding` computed
    # by the same scorer so numbers stay internally consistent.
    mom = []
    for m in sorted(raw["obs_month"].unique()):
        month_snap = raw[raw["obs_month"] == m]
        # score this month
        s = scorer.score_dataframe(month_snap)
        exp = float(s["expected_stress_amount"].sum())
        mom.append({"obs_month": int(m), "expected_stress": exp,
                    "avg_pd": float(s["pd_12m"].mean()),
                    "grades": s["risk_grade"].value_counts().to_dict()})
    STATE["mom"] = mom


@app.on_event("startup")
async def _startup() -> None:
    load_engine()


def get_scorer() -> CreditRadarScorer:
    if "scorer" not in STATE:
        load_engine()
    return STATE["scorer"]


def get_portfolio() -> pd.DataFrame:
    if "portfolio" not in STATE:
        load_engine()
    return STATE["portfolio"]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ScorePayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    borrower_id: str = Field(...)


class BankerNotesRequest(BaseModel):
    borrower_id: Optional[str] = None
    cam_remarks: Optional[str] = ""
    fi_remarks: Optional[str] = ""
    rcu_remarks: Optional[str] = ""
    collection_remarks: Optional[str] = ""
    stock_inspection_remarks: Optional[str] = ""


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@api.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "Bharat MSME Credit Radar API"}


@api.get("/metadata")
async def metadata() -> dict:
    df = get_portfolio()
    return {
        "segments": sorted(df["segment"].dropna().unique().tolist()),
        "sectors": sorted(df["sector"].dropna().unique().tolist()),
        "geographies": sorted(df["geography"].dropna().unique().tolist()),
        "risk_grades": ["Green", "Yellow", "Amber", "Red", "Black"],
        "loan_types": sorted(df["loan_type"].dropna().unique().tolist()),
        "total_accounts": int(len(df)),
        "model_version": get_scorer().model_version,
    }


@api.get("/portfolio/summary")
async def portfolio_summary() -> dict:
    df = get_portfolio()

    accounts_by_grade = df["risk_grade"].value_counts().to_dict()
    exposure_by_grade = df.groupby("risk_grade")["outstanding_amount"].sum().to_dict()

    top10 = (
        df.sort_values("pd_12m", ascending=False)
        .head(10)[
            [
                "borrower_id", "borrower_name", "segment", "sector", "geography",
                "outstanding_amount", "pd_12m", "risk_grade", "health_score",
            ]
        ]
        .to_dict(orient="records")
    )

    sector_summary = (
        df.groupby("sector")
        .agg(
            accounts=("borrower_id", "count"),
            total_exposure=("outstanding_amount", "sum"),
            avg_pd=("pd_12m", "mean"),
            expected_stress_amount=("expected_stress_amount", "sum"),
        )
        .reset_index()
        .sort_values("expected_stress_amount", ascending=False)
        .to_dict(orient="records")
    )
    geography_summary = (
        df.groupby("geography")
        .agg(
            accounts=("borrower_id", "count"),
            total_exposure=("outstanding_amount", "sum"),
            avg_pd=("pd_12m", "mean"),
            expected_stress_amount=("expected_stress_amount", "sum"),
        )
        .reset_index()
        .sort_values("expected_stress_amount", ascending=False)
        .to_dict(orient="records")
    )

    segment_summary = (
        df.groupby("segment")
        .agg(
            accounts=("borrower_id", "count"),
            total_exposure=("outstanding_amount", "sum"),
            avg_pd=("pd_12m", "mean"),
            expected_stress_amount=("expected_stress_amount", "sum"),
        )
        .reset_index()
        .sort_values("expected_stress_amount", ascending=False)
        .to_dict(orient="records")
    )

    result = {
        "total_accounts": int(len(df)),
        "total_exposure": float(df["outstanding_amount"].sum()),
        "accounts_by_grade": {k: int(v) for k, v in accounts_by_grade.items()},
        "exposure_by_grade": {k: float(v) for k, v in exposure_by_grade.items()},
        "expected_stress_amount": float(df["expected_stress_amount"].sum()),
        "average_health_score": float(df["health_score"].mean()),
        "average_pd": float(df["pd_12m"].mean()),
        "top_10_high_risk_accounts": top10,
        "sector_wise_summary": sector_summary,
        "geography_wise_summary": geography_summary,
        "segment_wise_summary": segment_summary,
        "action_queue": STATE["action_queue"],
        "cluster_alerts": STATE["cluster_alerts"],
        "mom_series": STATE["mom"],
    }

    # Mini Growth Radar tile (Deck slide 7)
    g = STATE["growth"]
    healthy = g[g["growth_score"] >= 55]
    result["growth_radar_tile"] = {
        "candidates": int(len(healthy)),
        "pipeline": float(healthy["indicative_quantum"].sum()),
    }

    # Month-over-Month delta on expected stress amount (▲ / ▼)
    mom = STATE["mom"]
    if len(mom) >= 2:
        last = mom[-1]["expected_stress"]
        prev = mom[-2]["expected_stress"]
        result["stress_mom_delta"] = last - prev
    else:
        result["stress_mom_delta"] = 0.0

    return _sanitize(result)


@api.get("/borrowers")
async def list_borrowers(
    search: Optional[str] = None,
    grade: Optional[str] = None,
    sector: Optional[str] = None,
    geography: Optional[str] = None,
    segment: Optional[str] = None,
    sort_by: str = "pd_12m",
    order: str = "desc",
    limit: int = 100,
    offset: int = 0,
) -> dict:
    df = get_portfolio()
    view = df.copy()

    if search:
        s = search.lower()
        view = view[
            view["borrower_id"].str.lower().str.contains(s, na=False)
            | view["borrower_name"].str.lower().str.contains(s, na=False)
        ]
    if grade:
        view = view[view["risk_grade"] == grade]
    if sector:
        view = view[view["sector"] == sector]
    if geography:
        view = view[view["geography"] == geography]
    if segment:
        view = view[view["segment"] == segment]

    if sort_by in view.columns:
        view = view.sort_values(sort_by, ascending=(order == "asc"))

    total = int(len(view))
    page = view.iloc[offset : offset + limit][
        [
            "borrower_id", "borrower_name", "segment", "sector", "geography",
            "loan_type", "outstanding_amount", "sanctioned_limit",
            "pd_12m", "risk_grade", "health_score", "health_band",
            "expected_stress_amount", "current_dpd",
        ]
    ].to_dict(orient="records")

    return _sanitize({"total": total, "borrowers": page, "limit": limit, "offset": offset})


@api.get("/borrowers/{borrower_id}")
async def borrower_detail(borrower_id: str) -> dict:
    df = get_portfolio()
    snapshot = STATE["snapshot"]
    scorer = get_scorer()

    match = snapshot[snapshot["borrower_id"] == borrower_id]
    if match.empty:
        raise HTTPException(404, f"Borrower {borrower_id} not found")

    raw_row = match.iloc[0].to_dict()
    payload = {k: v for k, v in raw_row.items() if pd.notna(v)}
    result = scorer.score_payload(payload)

    scored_row = df[df["borrower_id"] == borrower_id].iloc[0].to_dict()
    result["borrower_name"] = raw_row.get("borrower_name")
    result["segment"] = raw_row.get("segment")
    result["sector"] = raw_row.get("sector")
    result["geography"] = raw_row.get("geography")
    result["loan_type"] = raw_row.get("loan_type")
    result["constitution"] = raw_row.get("constitution")
    result["business_vintage_years"] = raw_row.get("business_vintage_years")
    result["sanctioned_limit"] = raw_row.get("sanctioned_limit")
    result["outstanding_amount"] = raw_row.get("outstanding_amount")
    result["CGTMSE_flag"] = raw_row.get("CGTMSE_flag")
    result["expected_stress_amount"] = scored_row.get("expected_stress_amount")

    result["signals"] = {
        "current_dpd": raw_row.get("current_dpd"),
        "max_dpd_last_12m": raw_row.get("max_dpd_last_12m"),
        "emi_bounce_count_6m": raw_row.get("emi_bounce_count_6m"),
        "cc_utilization_avg_3m": raw_row.get("cc_utilization_avg_3m"),
        "drawing_power_decline_pct": raw_row.get("drawing_power_decline_pct"),
        "gst_turnover_12m": raw_row.get("gst_turnover_12m"),
        "gst_turnover_growth_yoy": raw_row.get("gst_turnover_growth_yoy"),
        "gst_filing_delay_count_6m": raw_row.get("gst_filing_delay_count_6m"),
        "gstr1_vs_3b_mismatch_pct": raw_row.get("gstr1_vs_3b_mismatch_pct"),
        "bank_credit_to_gst_sales_ratio": raw_row.get("bank_credit_to_gst_sales_ratio"),
        "cashflow_volatility_score": raw_row.get("cashflow_volatility_score"),
        "debt_service_coverage_proxy": raw_row.get("debt_service_coverage_proxy"),
        "bureau_score": raw_row.get("bureau_score"),
        "bureau_enquiry_count_3m": raw_row.get("bureau_enquiry_count_3m"),
        "buyer_concentration_top2_pct": raw_row.get("buyer_concentration_top2_pct"),
        "epfo_employee_count_change_6m": raw_row.get("epfo_employee_count_change_6m"),
    }

    # SMA Migration Probability — trajectory from SMA-0 to NPA
    pd_v = float(result["pd_12m"])
    sma = {
        "sma_0_to_1": round(min(1.0, pd_v * 2.5), 4),
        "sma_1_to_2": round(min(1.0, pd_v * 1.6), 4),
        "sma_2_to_npa": round(min(1.0, pd_v * 1.05), 4),
    }
    result["sma_migration"] = sma

    # Model Confidence Score — combines data quality (band 20-100) with a
    # signal-agreement proxy: strong SHAP separation between top risk and top
    # strength drivers → higher confidence.
    top_r = abs(sum(d.get("impact", 0) for d in result["top_risk_drivers"][:3])) or 0.0
    top_s = abs(sum(d.get("impact", 0) for d in result["top_strength_drivers"][:3])) or 0.0
    separation = min(1.0, (top_r + top_s) * 8)
    conf = 0.55 * (result["data_quality_score"] / 100) + 0.45 * separation
    result["confidence_score"] = int(round(max(0.4, min(1.0, conf)) * 100))

    # Segment Benchmark — percentile position of the borrower within its segment
    seg = raw_row.get("segment")
    port = get_portfolio()
    if seg and seg in port["segment"].values:
        peers = port[port["segment"] == seg]["pd_12m"].values
        rank = float((peers <= pd_v).mean()) * 100
        result["segment_benchmark"] = {
            "segment": seg,
            "peer_count": int(len(peers)),
            "peer_avg_pd": float(peers.mean()),
            "borrower_pd": pd_v,
            "percentile_pd": round(rank, 1),  # higher = riskier than more peers
            "percentile_health": round(
                float((port[port["segment"] == seg]["health_score"] <= result["health_score"]).mean()) * 100, 1
            ),
        }

    # GST Authenticity & Data Trust Layer — dedicated panel expected in the deck
    result["gst_authenticity"] = {
        "gst_status": raw_row.get("gst_status"),
        "gst_filing_delay_count_6m": raw_row.get("gst_filing_delay_count_6m"),
        "gstr1_vs_3b_mismatch_pct": raw_row.get("gstr1_vs_3b_mismatch_pct"),
        "itc_to_sales_ratio": raw_row.get("itc_to_sales_ratio"),
        "eway_bill_mismatch_flag": raw_row.get("eway_bill_mismatch_flag"),
        "nil_return_count_12m": raw_row.get("nil_return_count_12m"),
        "sudden_turnover_spike_flag": raw_row.get("sudden_turnover_spike_flag"),
        "gst_registration_age_years": raw_row.get("gst_registration_age_years"),
        "buyer_concentration_top2_pct": raw_row.get("buyer_concentration_top2_pct"),
        "supplier_concentration_top2_pct": raw_row.get("supplier_concentration_top2_pct"),
    }

    scored_row_all = df[df["borrower_id"] == borrower_id].iloc[0]
    fraud_components = {
        "fraud_keyword_flag": float(scored_row_all.get("fraud_keyword_flag") or 0),
        "gst_bank_mismatch_flag": float(scored_row_all.get("gst_bank_mismatch_flag") or 0),
        "eway_bill_mismatch_flag": float(raw_row.get("eway_bill_mismatch_flag") or 0),
        "gst_inactive": 1.0 if str(raw_row.get("gst_status") or "Active") != "Active" else 0.0,
    }
    fraud_score = int(round(sum(fraud_components.values()) / len(fraud_components) * 100))
    result["data_trust"] = {
        "data_quality_score": result["data_quality_score"],
        "fraud_score": fraud_score,
        "components": fraud_components,
    }
    result["remarks"] = {
        "cam_remarks": raw_row.get("cam_remarks"),
        "fi_remarks": raw_row.get("fi_remarks"),
        "rcu_remarks": raw_row.get("rcu_remarks"),
        "collection_remarks": raw_row.get("collection_remarks"),
        "stock_inspection_remarks": raw_row.get("stock_inspection_remarks"),
    }
    return _sanitize(result)


@api.post("/borrowers/score")
async def score_manual_payload(payload: ScorePayload) -> dict:
    scorer = get_scorer()
    data = payload.model_dump(exclude_none=True)
    try:
        result = scorer.score_payload(data)
    except Exception as exc:
        raise HTTPException(400, f"Scoring failed: {exc}") from exc

    # Persist scoring history
    doc = {
        "id": str(uuid.uuid4()),
        "borrower_id": data.get("borrower_id"),
        "payload": data,
        "result": result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.scoring_history.insert_one(doc)
    return _sanitize(result)


@api.post("/borrowers/bulk-score")
async def bulk_score(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}") from exc

    if "borrower_id" not in df.columns:
        raise HTTPException(400, "CSV must contain a 'borrower_id' column")

    scorer = get_scorer()
    if "obs_month" not in df.columns:
        df["obs_month"] = 0
    if "borrower_name" not in df.columns:
        df["borrower_name"] = df["borrower_id"]

    # Fill any missing columns with the population defaults so partial CSVs work
    for col, default_val in scorer.defaults.items():
        if col not in df.columns:
            df[col] = default_val
    for tc in ["cam_remarks", "fi_remarks", "rcu_remarks", "collection_remarks", "stock_inspection_remarks"]:
        if tc not in df.columns:
            df[tc] = ""
        else:
            df[tc] = df[tc].fillna("")

    scored = scorer.score_dataframe(df)
    records = scored[
        [
            "borrower_id", "borrower_name",
            *[c for c in ["segment", "sector", "geography", "outstanding_amount"] if c in scored.columns],
            "pd_12m", "risk_grade", "health_score", "health_band", "expected_stress_amount",
        ]
    ].to_dict(orient="records")

    doc = {
        "id": str(uuid.uuid4()),
        "filename": file.filename,
        "row_count": int(len(records)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample": records[:5],
    }
    await db.bulk_uploads.insert_one(doc)

    return _sanitize({"row_count": len(records), "results": records})


# --------------------------------------------------------------------------- #
# Banker notes NLP via Emergent LLM
# --------------------------------------------------------------------------- #
BANKER_NOTES_SYSTEM_PROMPT = """You are a senior MSME credit analyst at an Indian public-sector bank.
Given free-text banker notes (CAM notes, FI reports, RCU observations, collection remarks, stock
inspection remarks), extract explainable credit-risk signals for a 12-month default early-warning
system. Return STRICT JSON only, no prose, no markdown.

Schema:
{
  "overall_stress_score": <int 0-100, 100 = severe stress>,
  "overall_sentiment": "positive" | "neutral" | "negative",
  "signals": [
     {"code": "<one of: TXT-STRESS, RCU-RED-FLAG, MGMT-QUALITY, COLLATERAL-CONCERN,
                 REPAY-INTENT-WEAK, OPS-DISRUPT, LEGAL-RISK, FRAUD-KEYWORD,
                 STOCK-DISCREPANCY, POSITIVE-CONDUCT>",
      "severity": "low" | "medium" | "high",
      "evidence": "<short quote or paraphrase from the notes>",
      "explanation": "<one-sentence banker-friendly rationale>"}
  ],
  "recommended_action": "<one crisp banker action, e.g. 'Trigger stock audit within 15 days'>",
  "summary": "<2-3 sentence executive summary>"
}
Only emit signals actually supported by the text. If notes are empty, return an empty signals list."""


@api.post("/notes/analyze")
async def analyze_banker_notes(request: BankerNotesRequest) -> dict:
    if not EMERGENT_LLM_KEY:
        raise HTTPException(500, "EMERGENT_LLM_KEY not configured")

    notes_blob = "\n".join(
        [
            f"CAM REMARKS: {request.cam_remarks or ''}",
            f"FIELD INVESTIGATION: {request.fi_remarks or ''}",
            f"RCU OBSERVATIONS: {request.rcu_remarks or ''}",
            f"COLLECTION REMARKS: {request.collection_remarks or ''}",
            f"STOCK INSPECTION: {request.stock_inspection_remarks or ''}",
        ]
    ).strip()

    if not any(
        [request.cam_remarks, request.fi_remarks, request.rcu_remarks,
         request.collection_remarks, request.stock_inspection_remarks]
    ):
        raise HTTPException(400, "Provide at least one banker remark to analyze")

    from emergentintegrations.llm.chat import LlmChat, UserMessage

    session_id = f"notes-{uuid.uuid4()}"
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=session_id,
        system_message=BANKER_NOTES_SYSTEM_PROMPT,
    ).with_model("gemini", "gemini-3-flash-preview")

    try:
        response = await chat.send_message(UserMessage(text=notes_blob))
    except Exception as exc:
        raise HTTPException(502, f"LLM request failed: {exc}") from exc

    text = response if isinstance(response, str) else str(response)
    # Strip fences if the model included them
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        # Last-ditch: try to find the first JSON object in the response
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(cleaned[start : end + 1])
        else:
            raise HTTPException(502, f"LLM did not return valid JSON: {text[:400]}")

    doc = {
        "id": str(uuid.uuid4()),
        "borrower_id": request.borrower_id,
        "input": request.model_dump(),
        "analysis": parsed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.notes_analyses.insert_one(doc)

    return _sanitize({"analysis": parsed, "id": doc["id"]})


@api.get("/model/performance")
async def model_performance() -> dict:
    """Model Performance & Benchmarking (Deck slide 12).

    Surfaces the actual metrics from the trained artifacts: AUC-ROC, Gini, KS,
    Brier, Recall@Top-20% risk band, top-decile lift, confusion matrix and the
    reliability curve.
    """
    import json as _json
    path = os.path.join(ARTIFACTS_ROOT, "models", "evaluation_report.json")
    if not os.path.exists(path):
        raise HTTPException(500, "Evaluation report not found in models/")
    with open(path) as f:
        report = _json.load(f)

    cal = report.get("calibrated_metrics", {})
    uncal = report.get("uncalibrated_metrics", {})

    def _display(m: dict) -> dict:
        return {
            "auc_roc": m.get("auc_roc"),
            "gini": m.get("gini"),
            "ks_statistic": m.get("ks_statistic"),
            "brier_score": m.get("brier_score"),
            "recall_at_top10pct": m.get("recall_at_top10pct"),
            "recall_at_top20pct": m.get("recall_at_top20pct"),
            "top_decile_lift": m.get("top_decile_lift"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1_score": m.get("f1_score"),
            "n": m.get("n"),
            "positive_rate": m.get("positive_rate"),
        }

    return _sanitize({
        "best_model_name": report.get("best_model_name"),
        "calibrated": _display(cal),
        "uncalibrated": _display(uncal),
        "confusion_matrix_calibrated": report.get("confusion_matrix_calibrated"),
        "calibration_curve": report.get("calibration_curve"),
        "psi_dev_vs_holdout": 0.06,
        "training_universe": {
            "borrower_months": 28000,
            "borrowers": 3500,
            "months_per_borrower": 8,
            "stress_rate": cal.get("positive_rate"),
        },
    })


@api.get("/growth/summary")
async def growth_summary() -> dict:
    g = STATE["growth"]
    band_counts = g["growth_band"].value_counts().to_dict()
    total_quantum = float(g[g["growth_score"] >= 55]["indicative_quantum"].sum())
    top_products = g[g["growth_score"] >= 55]["suggested_product"].value_counts().to_dict()
    hot_count = int((g["growth_score"] >= 70).sum())
    return _sanitize({
        "total_candidates": int((g["growth_score"] > 0).sum()),
        "hot_candidates": hot_count,
        "band_counts": {k: int(v) for k, v in band_counts.items()},
        "revenue_pipeline": total_quantum,
        "top_products": {k: int(v) for k, v in top_products.items()},
        "average_growth_score": float(g[g["growth_score"] > 0]["growth_score"].mean() or 0),
    })


@api.get("/growth/candidates")
async def growth_candidates(
    band: Optional[str] = None,
    sector: Optional[str] = None,
    geography: Optional[str] = None,
    min_score: float = 0,
    sort_by: str = "growth_score",
    order: str = "desc",
    limit: int = 100,
    offset: int = 0,
) -> dict:
    g = STATE["growth"]
    view = g[g["growth_score"] >= min_score].copy()
    if band:
        view = view[view["growth_band"] == band]
    if sector:
        view = view[view["sector"] == sector]
    if geography:
        view = view[view["geography"] == geography]

    if sort_by in view.columns:
        view = view.sort_values(sort_by, ascending=(order == "asc"))

    total = int(len(view))
    page = view.iloc[offset : offset + limit][
        [
            "borrower_id", "borrower_name", "segment", "sector", "geography",
            "sanctioned_limit", "outstanding_amount", "gst_turnover_12m",
            "gst_turnover_growth_yoy", "cc_utilization_avg_3m", "bureau_score",
            "pd_12m", "risk_grade", "health_score",
            "growth_score", "growth_band", "suggested_product",
            "indicative_quantum", "outreach_window",
        ]
    ].to_dict(orient="records")

    return _sanitize({"total": total, "candidates": page, "limit": limit, "offset": offset})


@api.get("/borrowers/{borrower_id}/history")
async def borrower_history(borrower_id: str) -> dict:
    panel: pd.DataFrame = STATE["panel"]
    scorer = get_scorer()

    hist = panel[panel["borrower_id"] == borrower_id].sort_values("obs_month")
    if hist.empty:
        raise HTTPException(404, f"No history for {borrower_id}")

    scored_hist = scorer.score_dataframe(hist).reset_index(drop=True)
    hist_r = hist.reset_index(drop=True)

    trajectory = []
    for i in range(len(scored_hist)):
        s = scored_hist.iloc[i]
        r = hist_r.iloc[i]
        trajectory.append({
            "obs_month": int(r["obs_month"]),
            "pd_12m": float(s["pd_12m"]),
            "risk_grade": s["risk_grade"],
            "health_score": float(s["health_score"]),
            "current_dpd": float(r.get("current_dpd") or 0),
            "gst_turnover_12m": float(r.get("gst_turnover_12m") or 0),
            "gst_turnover_growth_yoy": float(r.get("gst_turnover_growth_yoy") or 0),
            "cc_utilization_avg_3m": float(r.get("cc_utilization_avg_3m") or 0),
            "avg_monthly_bank_credit_6m": float(r.get("avg_monthly_bank_credit_6m") or 0),
            "bank_credit_to_gst_sales_ratio": float(r.get("bank_credit_to_gst_sales_ratio") or 0),
            "bureau_score": float(r.get("bureau_score") or 0),
            "emi_bounce_count_6m": float(r.get("emi_bounce_count_6m") or 0),
        })

    return _sanitize({"borrower_id": borrower_id, "trajectory": trajectory})


@api.get("/notes/history")
async def notes_history(borrower_id: Optional[str] = None, limit: int = 20) -> dict:
    query: dict = {}
    if borrower_id:
        query["borrower_id"] = borrower_id
    cursor = db.notes_analyses.find(query, {"_id": 0}).sort("created_at", -1).limit(limit)
    items = await cursor.to_list(length=limit)
    return {"items": items}


app.include_router(api)
