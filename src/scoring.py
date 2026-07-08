"""
Bharat MSME Credit Radar - End-to-End Scoring Pipeline
==========================================================
Single entry point used by both the FastAPI service and the Streamlit
dashboard. Implements the design brief's "Common Interpretation Layer"
(Layer 5): whichever segment model (or the global fallback model) produced
a borrower's PD, every score that leaves this module carries the same
output contract — 12-month PD, Health Score, Risk Grade, SMA migration
probability, top-5 SHAP reason codes (risk and strength), a segment
benchmark percentile, a model confidence label, a data quality score, a
model version, a CGTMSE suitability read, a growth propensity read (when
eligible) and a recommended banker action. A Branch Head in Surat and a
risk analyst at HO read the same card the same way, regardless of which
underlying model or segment produced it.
"""

from __future__ import annotations

import bisect
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import build_features, get_model_feature_columns, TEXT_COLUMNS  # noqa: E402
from model_utils import encode_features, align_columns  # noqa: E402
from explainability import build_explainer, explain_row  # noqa: E402
from action_engine import compute_health_score, health_band, risk_grade, build_recommendation  # noqa: E402
from segment_models import SegmentModelRegistry  # noqa: E402
from survival_model import (  # noqa: E402
    load_survival_artifacts, quarter_probabilities, sma_migration_probability, expected_months_to_stress,
)
from growth_propensity import GrowthPropensityEngine  # noqa: E402
from cgtmse_engine import cgtmse_suitability  # noqa: E402
from graph_contagion import apply_contagion_overlay, compute_cluster_stress  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

ID_TARGET_COLS = {"record_id", "borrower_id", "borrower_name", "obs_month", "stress_12m", "growth_need_12m"}

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


def model_confidence_label(data_quality_score: int) -> str:
    """Maps the data quality score to the ordinal confidence label used on
    the sample borrower card ('Med-High' etc.)."""
    if data_quality_score >= 90:
        return "High"
    if data_quality_score >= 70:
        return "Med-High"
    if data_quality_score >= 50:
        return "Medium"
    return "Low"


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

        # Common Interpretation Layer additions
        self.segment_registry = SegmentModelRegistry(models_dir)
        self.survival_model = load_survival_artifacts(models_dir)
        self.growth_engine = GrowthPropensityEngine(models_dir)

        ref_path = os.path.join(models_dir, "segment_pd_reference.json")
        self.segment_pd_reference = {}
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                self.segment_pd_reference = json.load(f)

        self._cluster_stats: dict | None = None  # populated via refresh_cluster_context()

    # ------------------------------------------------------------------ #
    # Portfolio / cluster context (set once at API/dashboard startup)
    # ------------------------------------------------------------------ #
    def refresh_cluster_context(self, scored_portfolio: pd.DataFrame) -> None:
        """Caches per-anchor-buyer cluster stress stats from the current
        portfolio snapshot so single-borrower `/score` calls can report
        cluster contagion context without re-scanning the whole book."""
        if "cluster_id" not in scored_portfolio.columns or "anchor_buyer_id" not in scored_portfolio.columns:
            self._cluster_stats = None
            return
        stats = compute_cluster_stress(scored_portfolio, pd_col="pd_12m")
        self._cluster_stats = stats.set_index("anchor_buyer_id")[["cluster_stress_index", "is_elevated"]].to_dict("index")

    def _segment_benchmark_percentile(self, segment: str, pd_12m: float) -> float | None:
        """`segment_pd_reference[segment]` is a 101-point percentile curve
        (index p == the PD value at the p-th percentile of that segment's
        training population), so bisecting into it directly returns an
        approximate percentile rank without needing the full population."""
        ref = self.segment_pd_reference.get(segment) or self.segment_pd_reference.get("_overall")
        if not ref:
            return None
        return float(min(max(bisect.bisect_left(ref, pd_12m), 0), 100))

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

    def _predict_pd(self, segment: str, engineered_row: pd.DataFrame, X: pd.DataFrame) -> tuple[float, str]:
        """Routes to the segment-specific model when available, else the
        global model — the model-routing half of the Common Interpretation
        Layer (Layer 4/5): different engines, one output contract.

        Segment predictions are shrinkage-blended toward the (larger-sample,
        better-calibrated) global model rather than trusted outright — see
        segment_models.SHRINKAGE_FULL_TRUST_EVENTS — so a segment model
        trained on only a few hundred synthetic borrowers cannot swing a
        borrower's PD to an extreme purely on small-sample noise."""
        global_pd = float(self.calibrator.predict_proba(X)[:, 1][0])
        if not self.segment_registry.has_segment_model(segment):
            return global_pd, f"GLB-{self.model_version}-v1.0"

        segment_pd, version = self.segment_registry.predict_pd(segment, engineered_row)
        weight = self.segment_registry.shrinkage_weight(segment)
        blended_pd = weight * segment_pd + (1 - weight) * global_pd
        return blended_pd, version

    def score_payload(self, payload: dict) -> dict:
        raw_row = self._payload_to_row(payload)
        engineered, _ = build_features(raw_row, fit_text_model=False, text_artifacts=(self.vectorizer, self.text_model))
        X = self._encode(engineered)
        row = engineered.iloc[0]
        segment = str(row.get("segment", ""))

        pd_12m, model_version = self._predict_pd(segment, engineered, X)
        grade = risk_grade(pd_12m)
        health_score, health_sub_scores = compute_health_score(row)
        dq_score = self._data_quality_score(payload)
        confidence = model_confidence_label(dq_score)

        explanation = explain_row(self.explainer, X, top_n=5)
        top_risk_codes = [d["code"] for d in explanation["top_risk_drivers"]]
        recommendation = build_recommendation(row, pd_12m, health_score, top_risk_codes)

        # Survival / timing layer -> SMA migration probability + expected months-to-stress
        sma_prob, emts = None, None
        if self.survival_model is not None:
            qp = quarter_probabilities(self.survival_model, X)
            sma_prob = float(sma_migration_probability(np.array([pd_12m]), qp)[0])
            emts_val = expected_months_to_stress(np.array([pd_12m]), qp)[0]
            emts = None if np.isnan(emts_val) else round(float(emts_val), 1)

        # Cluster contagion context (best-effort; requires refresh_cluster_context() to have been called)
        cluster_stress_index, in_elevated_cluster = 0.0, False
        anchor_id = row.get("anchor_buyer_id")
        if self._cluster_stats and anchor_id in self._cluster_stats:
            cluster_stress_index = float(self._cluster_stats[anchor_id]["cluster_stress_index"])
            in_elevated_cluster = bool(self._cluster_stats[anchor_id]["is_elevated"])

        # CGTMSE suitability engine
        suitability = cgtmse_suitability(row, pd_12m, health_score, fraud_flag=bool(row.get("fraud_keyword_flag", 0)))

        # Growth propensity engine (eligibility-gated)
        growth = self.growth_engine.score(
            engineered, risk_grade=grade,
            fraud_flag=bool(row.get("fraud_keyword_flag", 0)),
            gst_bank_mismatch_flag=bool(row.get("gst_bank_mismatch_flag", 0)),
            sma_migration_probability=sma_prob,
        )

        return {
            "borrower_id": payload.get("borrower_id", "MANUAL-ENTRY"),
            "pd_12m": round(pd_12m, 4),
            "risk_grade": grade,
            "health_score": health_score,
            "health_band": health_band(health_score),
            "health_sub_scores": health_sub_scores,
            "data_quality_score": dq_score,
            "model_confidence": confidence,
            "sma_migration_probability": round(sma_prob, 4) if sma_prob is not None else None,
            "expected_months_to_stress": emts,
            "segment_benchmark_percentile": self._segment_benchmark_percentile(segment, pd_12m),
            "cluster_stress_index": round(cluster_stress_index, 1),
            "in_elevated_cluster": in_elevated_cluster,
            "top_risk_drivers": explanation["top_risk_drivers"],
            "top_strength_drivers": explanation["top_strength_drivers"],
            "recommended_action": recommendation["recommended_action"],
            "action_checklist": recommendation["action_checklist"],
            "cgtmse_recommendation": suitability["category"] if suitability else None,
            "cgtmse_suitability": suitability,
            "growth_propensity": growth,
            "model_version": model_version,
        }

    # ------------------------------------------------------------------ #
    # Batch / portfolio scoring
    # ------------------------------------------------------------------ #
    def score_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        engineered, _ = build_features(raw_df, fit_text_model=False, text_artifacts=(self.vectorizer, self.text_model))
        X = self._encode(engineered)

        global_pd = self.calibrator.predict_proba(X)[:, 1]
        pd_12m = global_pd.copy()
        model_version = np.full(len(engineered), f"GLB-{self.model_version}-v1.0", dtype=object)

        # Route each segment with a dedicated model to that model's own predictions,
        # shrinkage-blended toward the global model (see _predict_pd docstring).
        for segment in engineered["segment"].unique():
            if not self.segment_registry.has_segment_model(segment):
                continue
            mask = (engineered["segment"] == segment).to_numpy()
            seg_rows = engineered.loc[mask]
            X_seg, _ = encode_features(seg_rows, self.segment_registry.feature_cols, self.segment_registry.categories)
            X_seg = align_columns(X_seg, self.segment_registry.encoded_columns).astype(float)
            estimator = self.segment_registry.calibrators.get(segment, self.segment_registry.models[segment])
            segment_pd = estimator.predict_proba(X_seg)[:, 1]
            weight = self.segment_registry.shrinkage_weight(segment)
            pd_12m[mask] = weight * segment_pd + (1 - weight) * global_pd[mask]
            model_version[mask] = self.segment_registry.meta[segment]["version"]

        out = raw_df.copy().reset_index(drop=True)
        out["pd_12m"] = pd_12m
        out["model_version"] = model_version
        out["risk_grade"] = out["pd_12m"].apply(risk_grade)

        health_results = engineered.apply(compute_health_score, axis=1)
        out["health_score"] = [h[0] for h in health_results]
        out["health_band"] = out["health_score"].apply(health_band)
        # Batch/portfolio rows come from the Bank's own systems (CBS/LOS/LMS), not a
        # partial API payload, so they are treated as fully populated for confidence purposes.
        out["model_confidence"] = model_confidence_label(100)

        # Segment benchmark percentile computed live within this scored population.
        out["segment_benchmark_percentile"] = (
            out.groupby(engineered["segment"])["pd_12m"].rank(pct=True) * 100
        ).round(1)

        # Survival / timing layer, vectorised across the whole portfolio.
        if self.survival_model is not None:
            qp = quarter_probabilities(self.survival_model, X)
            out["sma_migration_probability"] = sma_migration_probability(out["pd_12m"].to_numpy(), qp).round(4)
            out["expected_months_to_stress"] = expected_months_to_stress(out["pd_12m"].to_numpy(), qp).round(1)
        else:
            out["sma_migration_probability"] = np.nan
            out["expected_months_to_stress"] = np.nan

        # carry a few engineered flags through for portfolio drill-downs
        for col in ["gst_bank_mismatch_flag", "cc_utilization_avg_3m", "business_stress_keyword_flag",
                    "fraud_keyword_flag", "gst_authenticity_score", "cashflow_strength_score",
                    "bureau_stress_score", "repayment_stress_index"]:
            out[col] = engineered[col]

        out["expected_stress_amount"] = out["pd_12m"] * out["outstanding_amount"]

        # Graph contagion overlay (cluster stress index, elevated-cluster flag, adjusted PD).
        out = apply_contagion_overlay(out, pd_col="pd_12m")

        # CGTMSE suitability (no model call — cheap per-row rule evaluation).
        cgtmse_categories = []
        for i, row in engineered.iterrows():
            suitability = cgtmse_suitability(row, out.loc[i, "pd_12m"], out.loc[i, "health_score"],
                                              fraud_flag=bool(row.get("fraud_keyword_flag", 0)))
            cgtmse_categories.append(suitability["category"] if suitability else None)
        out["cgtmse_suitability_category"] = cgtmse_categories

        # Growth propensity: one vectorised model call for the whole portfolio.
        growth_df = self.growth_engine.score_batch(
            engineered, risk_grades=out["risk_grade"], fraud_flags=engineered["fraud_keyword_flag"],
            gst_bank_mismatch_flags=engineered["gst_bank_mismatch_flag"],
            sma_migration_probabilities=out["sma_migration_probability"],
        )
        out["growth_eligible"] = growth_df["growth_eligible"].to_numpy()
        out["growth_propensity_score"] = growth_df["growth_propensity_score"].to_numpy()
        out["growth_suggested_product"] = growth_df["growth_suggested_product"].to_numpy()

        return out


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce a borrower-month panel to each borrower's most recent month —
    the 'current portfolio' view used for dashboards / portfolio summary."""
    idx = df.groupby("borrower_id")["obs_month"].idxmax()
    return df.loc[idx].reset_index(drop=True)
