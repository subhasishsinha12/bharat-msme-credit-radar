"""
Bharat MSME Credit Radar - Streamlit Dashboard
=================================================
5 pages: Executive Dashboard, Borrower Scoring, Portfolio Heatmap,
Explainability, Model Performance.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from scoring import CreditRadarScorer, latest_snapshot  # noqa: E402
from action_engine import risk_grade, health_band  # noqa: E402

DATA_PATH = os.path.join(ROOT_DIR, "data", "synthetic_msme_data.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

GRADE_COLORS = {"Green": "#2e7d32", "Yellow": "#f9a825", "Amber": "#ef6c00", "Red": "#c62828", "Black": "#212121"}
GRADE_ORDER = ["Green", "Yellow", "Amber", "Red", "Black"]

st.set_page_config(page_title="Bharat MSME Credit Radar", page_icon="🇮🇳", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_scorer() -> CreditRadarScorer:
    return CreditRadarScorer()


@st.cache_data
def load_scored_portfolio() -> pd.DataFrame:
    scorer = load_scorer()
    raw = pd.read_csv(DATA_PATH)
    snapshot = latest_snapshot(raw)
    return scorer.score_dataframe(snapshot)


@st.cache_data
def load_evaluation_report() -> dict:
    path = os.path.join(MODELS_DIR, "evaluation_report.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def grade_badge(grade: str) -> str:
    color = GRADE_COLORS.get(grade, "#666")
    return f'<span style="background-color:{color};color:white;padding:4px 14px;border-radius:14px;font-weight:600;">{grade}</span>'


portfolio = load_scored_portfolio()
scorer = load_scorer()

st.sidebar.title("🇮🇳 Bharat MSME Credit Radar")
st.sidebar.caption("12-Month Predictive Default Intelligence & Early Warning Engine")
page = st.sidebar.radio("Navigate", [
    "1. Executive Dashboard", "2. Borrower Scoring", "3. Portfolio Heatmap",
    "4. Explainability", "5. Model Performance",
])
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Prototype built on **synthetic data** for IDBI Innovate 2026 (Track 04). "
    "Not for production lending decisions without full governance validation."
)

# =========================================================================== #
# PAGE 1 — EXECUTIVE DASHBOARD
# =========================================================================== #
if page.startswith("1"):
    st.title("Executive Dashboard")
    st.caption("Portfolio-wide 12-month predictive default intelligence")

    total_exposure = portfolio["outstanding_amount"].sum()
    total_accounts = len(portfolio)
    expected_stress = portfolio["expected_stress_amount"].sum()
    avg_health = portfolio["health_score"].mean()

    eval_report = load_evaluation_report()
    stress_capture = eval_report.get("calibrated_metrics", {}).get("recall_at_top20pct")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total MSME Exposure", f"₹{total_exposure/1e7:,.1f} Cr")
    c2.metric("Total Accounts", f"{total_accounts:,}")
    c3.metric("12M Expected Stress Amount", f"₹{expected_stress/1e7:,.1f} Cr")
    c4.metric("Average Health Score", f"{avg_health:.1f} / 100")
    c5.metric("Stress Capture (Top 20% band)", f"{stress_capture:.0%}" if stress_capture else "n/a")

    st.markdown("### Risk Grade Distribution")
    grade_counts = portfolio["risk_grade"].value_counts().reindex(GRADE_ORDER).fillna(0).astype(int)
    cols = st.columns(5)
    for col, g in zip(cols, GRADE_ORDER):
        col.markdown(
            f'<div style="text-align:center;padding:14px;border-radius:10px;background-color:{GRADE_COLORS[g]}22;">'
            f'<div style="font-size:26px;font-weight:700;color:{GRADE_COLORS[g]}">{grade_counts[g]}</div>'
            f'<div style="font-weight:600;">{g}</div></div>', unsafe_allow_html=True,
        )

    st.markdown("### ")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig = px.bar(
            x=GRADE_ORDER, y=[grade_counts[g] for g in GRADE_ORDER],
            color=GRADE_ORDER, color_discrete_map=GRADE_COLORS,
            labels={"x": "Risk Grade", "y": "Accounts"}, title="Accounts by Risk Grade",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with chart_col2:
        fig = px.histogram(
            portfolio, x="pd_12m", nbins=40, title="PD Distribution (12-Month)",
            labels={"pd_12m": "Predicted 12-Month PD"}, color_discrete_sequence=["#1565c0"],
        )
        st.plotly_chart(fig, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        sector_stress = portfolio.groupby("sector")["expected_stress_amount"].sum().sort_values(ascending=False)
        fig = px.bar(sector_stress, orientation="h", title="Sector-wise Expected Stress Amount",
                     labels={"value": "Expected Stress (₹)", "sector": "Sector"}, color_discrete_sequence=["#c62828"])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with chart_col4:
        geo_stress = portfolio.groupby("geography")["expected_stress_amount"].sum().sort_values(ascending=False)
        fig = px.bar(geo_stress, orientation="h", title="Geography-wise Expected Stress Amount",
                     labels={"value": "Expected Stress (₹)", "geography": "Geography"}, color_discrete_sequence=["#ef6c00"])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Stressed Sectors & Geographies")
    tcol1, tcol2 = st.columns(2)
    tcol1.dataframe(
        portfolio.groupby("sector").agg(accounts=("borrower_id", "count"), avg_pd=("pd_12m", "mean"),
                                         expected_stress=("expected_stress_amount", "sum"))
        .sort_values("avg_pd", ascending=False).head(5).style.format({"avg_pd": "{:.1%}", "expected_stress": "₹{:,.0f}"}),
        use_container_width=True,
    )
    tcol2.dataframe(
        portfolio.groupby("geography").agg(accounts=("borrower_id", "count"), avg_pd=("pd_12m", "mean"),
                                            expected_stress=("expected_stress_amount", "sum"))
        .sort_values("avg_pd", ascending=False).head(5).style.format({"avg_pd": "{:.1%}", "expected_stress": "₹{:,.0f}"}),
        use_container_width=True,
    )

# =========================================================================== #
# PAGE 2 — BORROWER SCORING
# =========================================================================== #
elif page.startswith("2"):
    st.title("Borrower Scoring")

    mode = st.radio("Select input mode", ["Pick existing borrower", "Manual entry"], horizontal=True)

    if mode == "Pick existing borrower":
        options = portfolio["borrower_id"] + " — " + portfolio["borrower_name"]
        choice = st.selectbox("Select borrower", options.sort_values())
        borrower_id = choice.split(" — ")[0]
        raw_row = portfolio[portfolio["borrower_id"] == borrower_id].iloc[0]
        payload = raw_row.drop(labels=[c for c in ["pd_12m", "risk_grade", "health_score", "health_band",
                                                     "expected_stress_amount"] if c in raw_row.index]).to_dict()
        result = scorer.score_payload(payload)
    else:
        with st.form("manual_entry"):
            c1, c2, c3 = st.columns(3)
            borrower_id = c1.text_input("Borrower ID", "B999999")
            segment = c2.selectbox("Segment", ["Trader", "Manufacturer", "Service", "NTC", "Existing", "CGTMSE", "Thin-file"])
            loan_type = c3.selectbox("Loan Type", ["Cash Credit", "Term Loan", "Machinery Loan", "LAP", "CGTMSE", "Working Capital"])
            c4, c5, c6 = st.columns(3)
            outstanding_amount = c4.number_input("Outstanding Amount (₹)", value=2000000.0)
            cc_utilization_avg_3m = c5.slider("CC Utilization Avg 3M (%)", 0.0, 130.0, 85.0)
            emi_bounce_count_6m = c6.number_input("EMI Bounce Count (6M)", value=1, min_value=0)
            c7, c8, c9 = st.columns(3)
            bureau_score = c7.number_input("Bureau Score", value=680, min_value=300, max_value=900)
            gst_filing_delay_count_6m = c8.number_input("GST Filing Delay Count (6M)", value=1, min_value=0)
            gstr1_vs_3b_mismatch_pct = c9.number_input("GSTR-1 vs 3B Mismatch (%)", value=8.0)
            c10, c11, c12 = st.columns(3)
            buyer_concentration_top2_pct = c10.slider("Buyer Concentration Top-2 (%)", 0.0, 100.0, 40.0)
            epfo_employee_count_change_6m = c11.number_input("EPFO Employee Count Change 6M (%)", value=0.0)
            bank_credit_to_gst_sales_ratio = c12.number_input("Bank Credit / GST Sales Ratio", value=0.85)
            cam_remarks = st.text_area("CAM Remarks", "business running satisfactorily")
            submitted = st.form_submit_button("Score Borrower")
        if not submitted:
            st.stop()
        payload = {
            "borrower_id": borrower_id, "segment": segment, "loan_type": loan_type,
            "outstanding_amount": outstanding_amount, "cc_utilization_avg_3m": cc_utilization_avg_3m,
            "emi_bounce_count_6m": emi_bounce_count_6m, "bureau_score": bureau_score,
            "gst_filing_delay_count_6m": gst_filing_delay_count_6m, "gstr1_vs_3b_mismatch_pct": gstr1_vs_3b_mismatch_pct,
            "buyer_concentration_top2_pct": buyer_concentration_top2_pct,
            "epfo_employee_count_change_6m": epfo_employee_count_change_6m,
            "bank_credit_to_gst_sales_ratio": bank_credit_to_gst_sales_ratio, "cam_remarks": cam_remarks,
        }
        raw_row = pd.Series(payload)
        result = scorer.score_payload(payload)

    st.markdown("---")
    st.subheader(f"Borrower Output Card — {raw_row.get('borrower_name', borrower_id)}")

    card1, card2, card3, card4 = st.columns(4)
    card1.metric("12-Month PD", f"{result['pd_12m']:.1%}")
    card2.markdown(f"**Risk Grade**<br>{grade_badge(result['risk_grade'])}", unsafe_allow_html=True)
    card3.metric("Health Score", f"{result['health_score']:.0f} / 100", result["health_band"])
    card4.metric("Data Quality Score", f"{result['data_quality_score']} / 100")

    info1, info2, info3, info4 = st.columns(4)
    info1.write(f"**Segment:** {raw_row.get('segment', '-')}")
    info2.write(f"**Facility:** {raw_row.get('loan_type', '-')}")
    info3.write(f"**Outstanding:** ₹{float(raw_row.get('outstanding_amount', 0)):,.0f}")
    info4.write(f"**Sector / Geography:** {raw_row.get('sector', '-')} / {raw_row.get('geography', '-')}")

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.markdown("#### 🔻 Top Risk Drivers")
        for d in result["top_risk_drivers"]:
            st.markdown(f"- **{d['code']}** — {d['description']} _(impact {d['impact']:+.3f})_")
    with dcol2:
        st.markdown("#### 🟢 Top Strength Drivers")
        for d in result["top_strength_drivers"]:
            st.markdown(f"- **{d['code']}** — {d['description']} _(impact {d['impact']:+.3f})_")

    st.markdown("#### 📋 Recommended Banker Action")
    st.info(result["recommended_action"])
    st.markdown("**Action checklist:**")
    for a in result["action_checklist"]:
        st.markdown(f"- {a}")
    if result.get("cgtmse_recommendation"):
        st.markdown(f"**CGTMSE Suitability:** {result['cgtmse_recommendation']}")

# =========================================================================== #
# PAGE 3 — PORTFOLIO HEATMAP
# =========================================================================== #
elif page.startswith("3"):
    st.title("Portfolio Heatmap & Action Queues")

    st.markdown("### Sector × Geography Risk Heatmap (Average PD)")
    pivot = portfolio.pivot_table(index="sector", columns="geography", values="pd_12m", aggfunc="mean")
    fig = px.imshow(pivot, color_continuous_scale="Reds", aspect="auto",
                     labels=dict(color="Avg PD"), text_auto=".1%")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### High-Risk Borrower Table (Red / Black)")
    high_risk = portfolio[portfolio["risk_grade"].isin(["Red", "Black"])].sort_values("pd_12m", ascending=False)
    st.dataframe(
        high_risk[["borrower_id", "borrower_name", "segment", "sector", "geography", "outstanding_amount",
                   "pd_12m", "risk_grade", "health_score"]].head(50),
        use_container_width=True,
    )

    st.markdown("### Action Queues")
    tabs = st.tabs(["Field Visit Required", "Stock Audit Required", "GST-Bank Mismatch",
                     "High CC Utilization", "Negative Text Remarks"])
    with tabs[0]:
        q = portfolio[portfolio["risk_grade"].isin(["Amber", "Red", "Black"])]
        st.caption(f"{len(q)} accounts require a field visit within policy timelines")
        st.dataframe(q[["borrower_id", "borrower_name", "sector", "geography", "pd_12m", "risk_grade"]].sort_values("pd_12m", ascending=False), use_container_width=True)
    with tabs[1]:
        q = portfolio[portfolio["risk_grade"].isin(["Red", "Black"])]
        st.caption(f"{len(q)} accounts require a stock audit")
        st.dataframe(q[["borrower_id", "borrower_name", "sector", "geography", "pd_12m", "risk_grade"]].sort_values("pd_12m", ascending=False), use_container_width=True)
    with tabs[2]:
        q = portfolio[portfolio["gst_bank_mismatch_flag"] == 1]
        st.caption(f"{len(q)} accounts show GST turnover vs bank credit mismatch")
        st.dataframe(q[["borrower_id", "borrower_name", "sector", "geography", "pd_12m", "risk_grade"]].sort_values("pd_12m", ascending=False), use_container_width=True)
    with tabs[3]:
        q = portfolio[portfolio["cc_utilization_avg_3m"] > 85]
        st.caption(f"{len(q)} accounts show cash-credit utilization above 85%")
        st.dataframe(q[["borrower_id", "borrower_name", "sector", "geography", "cc_utilization_avg_3m", "risk_grade"]].sort_values("cc_utilization_avg_3m", ascending=False), use_container_width=True)
    with tabs[4]:
        q = portfolio[(portfolio["business_stress_keyword_flag"] == 1) | (portfolio["fraud_keyword_flag"] == 1)]
        st.caption(f"{len(q)} accounts carry negative CAM / FI / RCU / collection / stock remarks")
        st.dataframe(q[["borrower_id", "borrower_name", "sector", "geography", "pd_12m", "risk_grade"]].sort_values("pd_12m", ascending=False), use_container_width=True)

# =========================================================================== #
# PAGE 4 — EXPLAINABILITY
# =========================================================================== #
elif page.startswith("4"):
    st.title("Explainability")
    st.caption("SHAP-based reason codes behind every score, for full banker transparency")

    options = portfolio["borrower_id"] + " — " + portfolio["borrower_name"]
    choice = st.selectbox("Select borrower to explain", options.sort_values())
    borrower_id = choice.split(" — ")[0]
    raw_row = portfolio[portfolio["borrower_id"] == borrower_id].iloc[0]
    payload = raw_row.drop(labels=[c for c in ["pd_12m", "risk_grade", "health_score", "health_band",
                                                 "expected_stress_amount"] if c in raw_row.index]).to_dict()
    result = scorer.score_payload(payload)

    st.metric("Predicted 12-Month PD", f"{result['pd_12m']:.1%}", result["risk_grade"])

    drivers = (
        [{"code": d["code"], "description": d["description"], "impact": d["impact"], "type": "Risk"} for d in result["top_risk_drivers"]]
        + [{"code": d["code"], "description": d["description"], "impact": d["impact"], "type": "Strength"} for d in result["top_strength_drivers"]]
    )
    ddf = pd.DataFrame(drivers).sort_values("impact")
    fig = px.bar(
        ddf, x="impact", y="code", color="type", orientation="h",
        color_discrete_map={"Risk": "#c62828", "Strength": "#2e7d32"},
        title="SHAP Contribution to Predicted PD (Reason Codes)",
        labels={"impact": "SHAP contribution to PD", "code": "Reason Code"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top Reason Codes")
    st.dataframe(ddf[["type", "code", "description", "impact"]], use_container_width=True)

    st.markdown("#### Model Confidence & Calibration Note")
    st.write(
        f"- Data quality score for this borrower: **{result['data_quality_score']}/100** "
        "(fraction of core alternate-data fields available).\n"
        f"- Model version: **{result['model_version']}**, probabilities calibrated with isotonic regression "
        "on a held-out calibration split so that a predicted 20% PD corresponds to an observed ~20% stress rate.\n"
        "- SHAP values are computed on the underlying tree model; the calibration layer adjusts the final "
        "probability but not the ranking of reason codes."
    )
    st.markdown(
        "> **Data quality note:** scores are only as reliable as the inputs supplied. Fields not provided by "
        "the caller are imputed from population-typical values, which pulls the estimate toward the average "
        "borrower — always check the data quality score before acting on a PD."
    )

# =========================================================================== #
# PAGE 5 — MODEL PERFORMANCE
# =========================================================================== #
elif page.startswith("5"):
    st.title("Model Performance")
    report = load_evaluation_report()
    if not report:
        st.warning("Run `python src/train_model.py` and `python src/evaluate_model.py` first.")
        st.stop()

    m = report["calibrated_metrics"]
    st.caption(f"Selected model: **{report['best_model_name']}** (isotonic-calibrated)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", f"{m['auc_roc']:.3f}")
    c2.metric("AUC-PR", f"{m['auc_pr']:.3f}")
    c3.metric("KS Statistic", f"{m['ks_statistic']:.3f}")
    c4.metric("Gini", f"{m['gini']:.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Recall @ Top 20%", f"{m['recall_at_top20pct']:.1%}")
    c6.metric("Recall @ Top 10%", f"{m['recall_at_top10pct']:.1%}")
    c7.metric("Top-Decile Lift", f"{m['top_decile_lift']:.2f}x")
    c8.metric("Brier Score", f"{m['brier_score']:.4f}")

    st.markdown("### Confusion Matrix (tuned F1 threshold)")
    cm = report["confusion_matrix_calibrated"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Predicted Non-Stress", "Predicted Stress"], y=["Actual Non-Stress", "Actual Stress"],
        colorscale="Blues", text=cm, texttemplate="%{text}",
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Calibration Curve")
    calib_df = pd.DataFrame(report["calibration_curve"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=calib_df["mean_predicted_pd"], y=calib_df["observed_stress_rate"],
                              mode="lines+markers", name="Model"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="Mean Predicted PD", yaxis_title="Observed Stress Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "> Metrics are computed on a held-out, borrower-level test split of the **synthetic** dataset. "
        "They validate the design logic of the pipeline (imbalance handling, calibration, explainability), "
        "not real-world bank-grade performance. See `reports/model_performance_report.md` and "
        "`reports/prototype_validation_note.md`."
    )
