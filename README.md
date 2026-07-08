# Bharat MSME Credit Radar

### A 12-Month Predictive Default Intelligence and Early Warning Engine for Indian MSME Loans

**Hackathon:** IDBI Innovate 2026 — **Track 04: MSME Credit | Predictive AI | Risk Management**

> "From document-based MSME lending to living credit intelligence."

> ⚠️ **Disclaimer:** This prototype uses synthetic MSME data for hackathon
> demonstration. No production lending decision should be made using this model without
> validation on historical bank data, governance approval, calibration, fairness
> testing, drift monitoring and human-in-the-loop controls.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Understanding — The Banker's View](#2-problem-understanding--the-bankers-view)
3. [Proposed Solution](#3-proposed-solution--bharat-msme-credit-radar)
4. [Solution Architecture](#4-solution-architecture)
5. [Segment-Specific MSME Model Design](#5-segment-specific-msme-model-design)
6. [12-Month Default Definition](#6-12-month-default-definition--the-target-variable)
7. [Accuracy & Model Evaluation Framework](#7-accuracy-and-model-evaluation-framework)
8. [Explainability Framework](#8-explainability-framework--the-common-interpretation-system)
9. [Unstructured Data Intelligence](#9-unstructured-data-intelligence--reading-the-banks-own-memory)
10. [GST and Cash-Flow Authenticity Engine](#10-gst-and-cash-flow-authenticity-engine)
11. [CGTMSE Suitability Engine](#11-cgtmse-suitability-engine)
12. [MSME Growth Propensity Engine](#12-msme-growth-propensity-engine)
13. [The MSME Credit Twin](#13-the-msme-credit-twin)
14. [Portfolio-Level Dashboard](#14-portfolio-level-dashboard--branch-ro-and-ho-views)
15. [Sample Borrower Output Card](#15-sample-borrower-output-card)
16. [Technology Stack](#16-technology-stack)
17. [Development Roadmap](#17-development-roadmap)
18. [Model Governance and Risk Controls](#18-model-governance-and-risk-controls)
19. [Expected Benefits](#19-expected-benefits)
20. [Differentiation from Existing Systems](#20-differentiation-from-existing-systems)
21. [How to Run Locally](#21-how-to-run-locally)
22. [API Usage](#22-api-usage)
23. [Streamlit Dashboard Usage](#23-streamlit-dashboard-usage)
24. [Repository Structure](#24-repository-structure)
25. [Limitations](#25-limitations)
26. [Final Pitch](#26-final-pitch)

---

## 1. Executive Summary

Indian MSME lending remains predominantly a document-anchored, sanction-time exercise.
Risk is assessed once — at appraisal — using audited or provisional financials, bureau
reports and collateral valuation, and is thereafter monitored through lagging indicators
such as overdue reports and SMA flags. Industry predictive capability for MSME stress
commonly sits around **16–22%**: fewer than one in four accounts that will slip into
stress over the next year are identified in advance. The consequence is surprise
slippage, reactive recovery, provisioning shocks and the rejection of viable MSMEs whose
strength is visible only in alternate data.

**Bharat MSME Credit Radar** is a bank-grade-in-design, India-stack-native, explainable
and continuously updating MSME credit intelligence platform. It fuses structured loan
and conduct data, alternate rails (GST, AA, UPI, bureau, EPFO, Udyam, MCA) and
NLP-processed banker intelligence into a single engine that produces, for every MSME
account: a calibrated 12-month Probability of Default, an MSME Financial Health Score,
an Early Warning Grade, SHAP-based reason codes, an SMA migration probability, a
CGTMSE suitability recommendation and a specific banker action. **Segment-wise models**
— trader, manufacturer, service, new-to-credit, existing, CGTMSE, thin-file — replace
the one-size-fits-all score, while a **Common Interpretation Layer** ensures every
output, regardless of segment or model, is read, compared and acted upon identically.

The same signal spine also powers an **MSME Growth Propensity Engine**: the data that
reveals stress early reveals strength early. Healthy borrowers whose GST trajectory,
cash-flow headroom and expansion signals indicate an approaching working-capital or
term-loan need are flagged for proactive, pre-qualified outreach — revenue intelligence
alongside risk intelligence, from one platform. A **Graph Contagion Overlay** further
flags borrowers sitting in a cluster (shared anchor buyer / trade pocket) that is
already showing broad-based stress, even before their own individual conduct has
deteriorated. And the **MSME Credit Twin** turns every borrower's monthly re-scoring
into a live trajectory — improving, stable or deteriorating — rather than a sanction-day
snapshot.

**What "prototype" means here, precisely:** every layer described in this README is
implemented in working code in this repository and can be run end-to-end on the
included synthetic dataset (see [How to Run Locally](#21-how-to-run-locally)). Nothing
below is slideware. What *is* still a design intent rather than a finished capability —
real GST/AA/bureau connectivity, a production-scale training population, a true graph
neural network, a fine-tuned language model, drift monitoring dashboards — is called out
explicitly in [Limitations](#25-limitations) and [Development Roadmap](#17-development-roadmap),
not glossed over.

## 2. Problem Understanding — The Banker's View

Ten structural weaknesses define the current state of MSME credit risk assessment:

1. **Fragmented risk assessment** — working capital, machinery term loans, CGTMSE
   facilities and OD limits are appraised through different templates and thresholds; the
   same borrower can look Amber in one framework and Green in another.
2. **Document dependency** — appraisal leans on audited/provisional financials and ITRs
   that are often delayed 9–18 months and compiled for tax rather than truth.
3. **Low availability of reliable financials** — a large share of Udyam-registered micro
   enterprises maintain no formal books at all.
4. **Static appraisal** — risk is measured at sanction and then effectively frozen.
5. **Delayed stress recognition** — stress is recognised only once it reaches the
   overdue report (SMA-1, SMA-2), by which time cash flow has already broken.
6. **Alternate data blindness** — GST returns are used as a turnover proxy at sanction,
   not as a monthly authenticity and continuity signal; AA, UPI and EPFO data are rarely
   used at all.
7. **Wasted unstructured intelligence** — the most prescient early warnings live in CAM
   notes, FI reports and RCU remarks, unread by any system.
8. **One-size-fits-all modelling** — a trader with a 15-day inventory cycle and a
   machinery-loan manufacturer with 120-day receivables fail differently.
9. **Non-comparable outputs** — a Branch Head cannot rank the riskiest 20 accounts on a
   common scale when methodologies differ by product and segment.
10. **Portfolio blind spots** — cluster contagion (e.g. a trade pocket hit by a buyer
    default), sector stress and geography-level deterioration are invisible until
    multiple accounts slip together.

**Why a 12-month early warning engine, not a sanction-time score:** a sanction-time
score answers "should we lend?" once. A 12-month early warning engine answers, every
month, "which standard accounts will be in stress within a year, and what should the
banker do about it now?" Twelve months is long enough to act (field visit, limit review,
restructuring dialogue) and short enough to be predictable from monthly GST, bank
conduct and behavioural data.

## 3. Proposed Solution — Bharat MSME Credit Radar

Bharat MSME Credit Radar is an end-to-end MSME credit intelligence platform operating
across the full lifecycle — at sanction, after disbursement and at portfolio level. It
is not a model; it is a decision system in which models are one layer. For every MSME
account, on every refresh cycle, the Radar generates:

| # | Output | Implemented in |
|---|---|---|
| 1 | 12-month Probability of Default (calibrated, segment-specific) | `src/scoring.py`, `src/segment_models.py` |
| 2 | MSME Financial Health Score (0–100) | `src/action_engine.py::compute_health_score` |
| 3 | Early Warning Grade (Green / Yellow / Amber / Red / Black) | `src/action_engine.py::risk_grade` |
| 4 | SMA migration probability (near-term, Q1 of the 12-month window) | `src/survival_model.py` |
| 5 | Growth Propensity Score for healthy accounts | `src/growth_propensity.py` |
| 6 | Top risk / strength drivers (SHAP reason codes, banker language) | `src/explainability.py` |
| 7 | Segment benchmark percentile | `src/scoring.py` |
| 8 | CGTMSE suitability recommendation with rationale | `src/cgtmse_engine.py` |
| 9 | Suggested banker action (specific, time-bound) | `src/action_engine.py` |
| 10 | Portfolio / cluster contagion contribution | `src/graph_contagion.py` |

### Solution Flow

```
DATA INGESTION            GST | AA | UPI/POS | Bureau | EPFO | Udyam | MCA
  (consent-first)          CBS | LOS | LMS | Bank statements | CAM/FI/RCU text
        |
>> DATA TRUST LAYER        Reconciliation, authenticity, fraud red flags
        |
FEATURE INTELLIGENCE       87 engineered features across 6 signal families
        |
AI/ML MODEL LAYER          Segment-wise models + survival timing + NLP + graph contagion
        |
>> EXPLAINABILITY          SHAP reason codes, confidence, data quality
        |
BANKER ACTION              Specific recommendation per account (+ CGTMSE + Growth read)
        |
PORTFOLIO MONITORING       Branch / RO / HO heatmaps, cluster alerts, growth pipeline
```

Two design principles govern the platform. First, **trust before intelligence**: no
signal enters the models until authenticity checks have run, because a model trained on
inflated GST turnover learns fraud as strength. Second, **every score must end in an
action**: a PD number that does not tell the branch what to do next is analytics, not
banking.

## 4. Solution Architecture

### Layer 1 — Consent and Data Ingestion Layer

*Design intent (prototype status: simulated via a synthetic generator, not live
connectors — see Limitations).* All borrower-side data is envisioned as acquired
consent-first, aligned to the DPDP framework and the Account Aggregator consent
architecture. `src/data_generator.py` produces borrower-month records that resemble
what these rails would deliver:

| Rail / Source | Data Modelled |
|---|---|
| Account Aggregator (Sahamati FIU) | Bank credits, debits, monthly balance, cash-flow granularity |
| GSTN (via GSP/consent) | GSTR-1/3B filing delay, mismatch %, ITC ratio, turnover growth, buyer/supplier concentration |
| UPI / POS | Digital collection ratio |
| Credit Bureau (CIBIL/CRIF-class) | Bureau score, enquiry count, DPD history, unsecured exposure |
| EPFO | Employee count, employee count change, salary payment regularity |
| Udyam / MCA | Business vintage, constitution |
| CBS / LOS / LMS | Facility structure, DPD, SMA status, utilisation, drawing power decline |
| Unstructured bank records | CAM notes, FI reports, RCU observations, collection & stock-inspection remarks |

### Layer 2 — Data Trust and Fraud Intelligence Layer

Before any feature is computed, incoming data is stress-tested for authenticity
(`src/feature_engineering.py::add_gst_features`, `add_text_features`):

- GST–bank reconciliation (`bank_credit_to_gst_sales_ratio`, `gst_bank_mismatch_flag`)
- GSTR-1 vs GSTR-3B mismatch (`gstr1_vs_3b_mismatch_pct`)
- ITC-to-sales ratio anomalies (`itc_to_sales_ratio`, `high_itc_flag`)
- GST filing delay / nil-return patterns, sudden turnover spikes
- E-way bill mismatch, GST registration status (cancelled/suspended)
- Cash deposit ratio vs declared digital/GST sales mix
- Related-party transaction keyword detection (`related_party_keyword_flag`)
- Cheque/ECS/SI bounce patterns

Every scored account carries a `gst_authenticity_score` (0–100) and a `data_quality_score`
— low-trust data caps confidence and is surfaced to the banker rather than silently
consumed (`model_confidence` field in every API response).

### Layer 3 — Feature Engineering Layer

87 model-ready features across 6 families (`src/feature_engineering.py`): repayment
behaviour, GST authenticity, cash-flow strength, bureau discipline, EPFO/operating
stability, and NLP/text signals — each combining level, flag and composite-score views.

### Layer 4 — AI/ML Model Layer

| Model | Role | Status |
|---|---|---|
| Logistic Regression | Regulatory-familiar benchmark / challenger baseline | ✅ trained (`src/train_model.py`) |
| Random Forest | Stability challenger, default global model | ✅ trained |
| XGBoost / LightGBM | Primary PD engines, selected per model-comparison metrics | ✅ trained |
| Segment-wise models (Trader/LightGBM, Manufacturer/XGBoost, Service/RF, NTC & Thin-file/Logistic, Existing & CGTMSE/RF) | Per-segment PD, shrinkage-blended with the global model | ✅ trained (`src/segment_models.py`) |
| Discrete-time survival (hazard) model | Timing of stress within the 12-month window (which quarter) | ✅ trained (`src/survival_model.py`) — synthetic proxy label, see Limitations |
| TF-IDF + Logistic Regression NLP | Reads CAM/FI/RCU/collection text; emits stress/fraud/continuity signals | ✅ trained (`src/feature_engineering.py`) — lightweight stand-in for a fine-tuned transformer |
| Graph contagion overlay (NetworkX) | Borrower–anchor-buyer cluster contagion | ✅ implemented (`src/graph_contagion.py`) — co-movement heuristic, not a trained GNN |
| Isotonic / sigmoid calibration | Converts raw scores into a trustworthy 12-month PD | ✅ implemented (`sklearn.CalibratedClassifierCV`) |

Routing logic (implemented in `src/scoring.py::CreditRadarScorer._predict_pd`): every
account is scored by its segment model when one exists and has adequate data; the
segment PD is **shrinkage-blended** toward the global model in proportion to how much
calibration evidence that segment has, so a segment trained on a few hundred synthetic
borrowers cannot swing a PD to an extreme on small-sample noise alone. Accounts in an
elevated-stress cluster additionally receive the graph contagion overlay. All accounts
receive the survival/timing layer.

### Layer 5 — Common Interpretation Layer

Whatever model produced the score, the output contract is identical (see
`ScoreResponse` in `api/schemas.py`): 12-month PD, Health Score, Risk Grade, SMA
migration probability, top-5 SHAP reason codes (risk and strength), segment benchmark
percentile, model confidence, data quality score, model version and recommended action.
A Branch Head in one city and a risk analyst at Head Office read the same card the same
way, regardless of which segment model or algorithm produced it.

### Layer 6 — Banker Action Layer

Every score resolves into a governed action, always subject to human decision
(`src/action_engine.py`, `src/cgtmse_engine.py`, `src/growth_propensity.py`):

| Context | Action Menu |
|---|---|
| Green / Yellow | Continue monitoring · review next GST filing · early engagement |
| Amber | Field visit within 30 days · debtor ageing / stock statement review · hold enhancement |
| Red / Black | Watchlist · stock audit · freeze enhancement · reduce exposure · recovery review |
| Under-collateralised & viable | CGTMSE suitability: Suitable / Suitable with reduced limit / Suitable after verification / Not suitable |
| Healthy & growing | Growth propensity: suggested product, indicative quantum, outreach window |

## 5. Segment-Specific MSME Model Design

One model cannot serve all MSME borrowers: the data that exists, the features that
matter and the way failure unfolds differ by segment. `src/segment_models.py` trains a
family of segment models beneath the shared Common Interpretation contract:

| Segment | Algorithm (this prototype) | Focus |
|---|---|---|
| Trader | LightGBM | Turnover velocity, buyer diversity, CC utilisation |
| Manufacturer | XGBoost | Capacity utilisation, receivable cycle, EPFO trend, DP erosion — paired with survival timing |
| Service | Random Forest | Collection regularity, client concentration, billing continuity |
| New-to-Credit (NTC) | Logistic Regression | Alternate-data scorecard, conservative calibration — inclusion lens |
| Existing | Random Forest | Full CBS/LMS history + all rails — richest model, feeds the Credit Twin |
| CGTMSE | Random Forest | All rails + viability features, paired with the CGTMSE suitability engine |
| Thin-file | Logistic Regression (or global fallback) | Alternate scorecard, NLP-weighted, human-in-loop mandatory |

Each segment model is trained on its own borrower-grouped train/calibration/test split.
A segment with fewer than 150 borrowers in the training data falls back entirely to the
global model at scoring time (`SegmentModelRegistry` in `src/segment_models.py`); a
segment that trains but has too few calibration-set stress events is **shrinkage-blended**
toward the global model rather than trusted at face value (see Layer 4 above and
Limitations). Model version strings follow the segment-prefixed convention used in the
design brief, e.g. `MFG-v1.0`, `TRD-v1.0`, or `GLB-<algorithm>-v1.0` for the fallback.

## 6. 12-Month Default Definition — The Target Variable

`stress_12m` is deliberately broader than the regulatory NPA event, because economic
loss begins well before Day-90. In `src/data_generator.py::compute_target`, a stress
event is a logistic combination of the strongest realized stress indicators — high
current DPD, EMI/cheque bounce counts, GSTR mismatch, drawing-power decline, low bureau
score, high buyer concentration, weak debt-service coverage, cash-flow volatility, GST
filing delay, falling EPFO headcount — plus a sizeable independent noise term, with the
intercept calibrated by bisection to a realistic 5–8% 12-month prevalence. This mirrors
the design brief's broader definition (90+ DPD / SMA-2→NPA / restructuring / settlement /
write-off / significant DP erosion / persistent bounce pattern / internal red-flag),
expressed through the observable proxies available in a synthetic panel rather than as
separately-modelled legal/recovery event flags.

Observation windows use only data available at the observation date (`obs_month`);
borrower-grouped splitting throughout the pipeline (`src/model_utils.py::group_split`)
ensures no borrower's months straddle train/calibration/test, eliminating the most
common form of leakage in a panel-data credit model.

## 7. Accuracy and Model Evaluation Framework

Default prediction is a heavily imbalanced problem (stress events are ~5–8% of
observations), so plain accuracy is not used for model selection. `src/evaluate_model.py`
and `src/train_model.py::model_selection_score` implement a discrimination +
calibration + stability framework:

| Metric | What It Measures | Indicative Target |
|---|---|---|
| AUC-ROC | Overall rank-ordering power | ≥ 0.85 |
| AUC-PR | Precision-recall trade-off under imbalance | Materially above base rate |
| KS statistic | Maximum separation of good/bad distributions | ≥ 45 |
| Gini coefficient | Discrimination (2×AUC−1) | ≥ 0.70 |
| Recall @ top 10% / 20% | Operational early-warning hit rate | ≥ 60% / ≥ 90% |
| Top-decile lift | Concentration of stress in the riskiest decile | ≥ 5× |
| Brier score | Probability accuracy after calibration | Minimised vs. baseline |
| Calibration curve | Does an 18% predicted PD mean an ~18% observed rate? | Slope ≈ 1, intercept ≈ 0 |
| PSI (population stability) | Input/score drift between development and holdout | < 0.10 green |

### This run's actual, live-computed figures (global model, held-out test split)

| Metric | Value |
|---|---|
| AUC-ROC | 0.954 |
| AUC-PR | 0.738 |
| Gini | 0.909 |
| KS Statistic | 0.780 |
| Recall @ top 20% risk band | 91.8% |
| Recall @ top 10% risk band | 79.9% |
| Top-decile lift | 7.99× |
| Brier Score (calibrated) | 0.0281 |
| PSI (train/dev vs test/holdout) | 0.0025 (stable) |

Selected model: **Random Forest**, isotonic-calibrated. Full breakdown, model comparison
table, confusion matrix and calibration curve: `reports/model_performance_report.md`
(regenerated by `python src/evaluate_model.py`). **These figures validate the pipeline's
design logic on synthetic data — they are not a claim of real-world, bank-grade
predictive performance.** See [Limitations](#25-limitations).

## 8. Explainability Framework — The Common Interpretation System

Every score carries SHAP (`TreeExplainer`) attributions mapped through a curated
banker-language reason-code dictionary (`src/explainability.py::FEATURE_META`, 70+
entries), including the design brief's standardised codes:

`GST-FIL-DLY` · `GST-BNK-MIS` · `CC-UTIL-HI` · `EMI-BNC-PTN` · `BUR-ENQ-SPK` ·
`BUY-CONC-HI` · `CF-VOL-HI` · `FI-NEG-RMK` · `STK-STMT-DLY` · `EPFO-DECL` ·
`SEC-STRESS` · `RPT-TXN-RISK` · `OPS-CONT-LO` · `DP-EROSION` · `ITC-RISK` ·
`RCU-RED-FLAG` — plus prototype-specific extensions (`DPD-HIST-HIGH`, `GSTR-MISMATCH`,
`BUR-DPD-CLEAN`, …) for finer-grained driver attribution.

The explainability contract per account: top-5 risk drivers, top-5 strength drivers,
segment benchmark percentile, model confidence, data quality score, model version — all
generated for **every** score, not just flagged ones, and reported together with the
calibration note in both the API response and the Explainability dashboard page.

## 9. Unstructured Data Intelligence — Reading the Bank's Own Memory

`src/feature_engineering.py` reads five text sources (CAM notes, FI reports, RCU
observations, collection remarks, stock-inspection remarks) and produces:

- Business-stress and fraud-risk keyword scores (`risk_keyword_count`, `fraud_keyword_flag`)
- Management-quality and collateral-concern flags
- **Source-specific** flags so a reason code can point to the exact remark type that
  moved the score: `fi_negative_remark_flag` (FI-NEG-RMK), `related_party_keyword_flag`
  (RPT-TXN-RISK), `stock_statement_delay_flag` (STK-STMT-DLY)
- A TF-IDF + Logistic Regression "first-stage" text risk model (`text_model_risk_score`)

**Honest scope note:** the design brief envisions an Indic-capable transformer
(IndicBERT/FinBERT-class) fine-tuned on banking text with Hinglish/vernacular tolerance.
This prototype implements the keyword + TF-IDF layer as an interpretable, fast,
dependency-light stand-in — swapping in a fine-tuned domain language model is a Phase 2
roadmap item (Section 17), not a change to the feature contract each text source feeds.

## 10. GST and Cash-Flow Authenticity Engine

GST is treated as a five-dimensional signal — turnover, compliance, authenticity,
continuity and counterparty quality — not a single turnover certificate
(`src/feature_engineering.py::add_gst_features`, `add_cashflow_features`):

| Signal | Feature(s) |
|---|---|
| GST turnover trend | `gst_turnover_growth_yoy`, `sudden_turnover_spike_flag` |
| Filing regularity | `gst_filing_delay_count_6m`, `gst_delay_flag`, `nil_return_count_12m` |
| GSTR-1 vs GSTR-3B mismatch | `gstr1_vs_3b_mismatch_pct` |
| ITC-to-sales ratio | `itc_to_sales_ratio`, `high_itc_flag` |
| E-way bill coherence | `eway_bill_mismatch_flag` |
| Buyer/supplier concentration | `buyer_concentration_top2_pct`, `supplier_concentration_top2_pct` |
| GSTIN status | `gst_status` (Active/Suspended/Cancelled) |
| GST sales vs bank credits | `bank_credit_to_gst_sales_ratio`, `gst_bank_mismatch_flag` — **the core authenticity ratio** |
| GST sales vs digital collections | `upi_pos_collection_ratio` |
| Cash intensity | `cash_deposit_ratio`, `high_cash_deposit_flag` |

These roll up into `gst_authenticity_score` (0–100), which feeds both the PD models and
the MSME Health Score.

## 11. CGTMSE Suitability Engine

`src/cgtmse_engine.py` scores guarantee-fit alongside default risk for any account where
a collateral-light structure is actually relevant (no collateral, existing CGTMSE flag,
or the CGTMSE segment). A composite 0–100 viability score (cash-flow strength, GST
authenticity, bureau discipline, operating/EPFO stability, digital collection intensity,
business vintage) resolves into one of four categories — never a binary approve/reject:

1. **Suitable for CGTMSE** — recommend the guarantee-backed structure as requested.
2. **Suitable with reduced limit** — viability confirmed, lower exposure recommended.
3. **Suitable after additional verification** — a generated checklist (GST-bank
   reconciliation, buyer confirmation, GSTR mismatch clarification, EPFO decline
   follow-up) closes the gap.
4. **Not suitable** — fraud-risk or compliance red flags present.

## 12. MSME Growth Propensity Engine

`src/growth_propensity.py` inverts the Radar's lens: for Green/Yellow accounts with
clean fraud and authenticity signals, it estimates the probability that the borrower
will need an enhanced working-capital limit or a new term loan within 6–12 months, using
GST growth trajectory, cash-flow headroom, EPFO/capacity signals, persistent high (but
serviced) CC utilisation, and digital collection intensity. Guardrails are enforced in
code, not policy discretion — automatically excluded if: risk grade is not Green/Yellow,
a fraud-keyword flag is present, a GST-bank mismatch is present, or the near-term SMA
migration probability is elevated. Every eligible account receives a growth propensity
score, a suggested product, an indicative quantum (derived from cash-flow headroom and
sanctioned-limit capacity), and a suggested outreach window — always a lead for the
banker, never an automated sanction.

The synthetic `growth_need_12m` target (`src/data_generator.py::compute_growth_target`)
is the deliberate mirror image of the stress target: built from expansion/headroom
signals rather than stress signals, at a ~12–16% prevalence.

## 13. The MSME Credit Twin

`src/credit_twin.py` re-scores every available observation month for a borrower to
build a Health Score / PD trajectory — where a traditional appraisal is a photograph,
the Credit Twin is a live feed. It emits, for any borrower:

1. Current Health Score and its trend (Improving / Stable / Deteriorating, from the
   slope across observed months)
2. Refreshed 12-month PD and near-term SMA migration probability
3. An expected months-to-stress estimate (from the survival/timing model)
4. An escalation note when the risk grade migrates month-over-month
5. A portfolio watchlist flag (Amber/Red/Black)

Exposed via `GET /credit-twin/{borrower_id}` and the dashboard's **MSME Credit Twin**
page, with a trajectory chart plotting Health Score and PD side by side across every
observed month.

## 14. Portfolio-Level Dashboard — Branch, RO and HO Views

Account-level intelligence aggregates into portfolio views (Streamlit `app/streamlit_app.py`,
API `GET /portfolio-summary`):

1. Total MSME exposure with Green/Yellow/Amber/Red/Black distribution (count and ₹ value)
2. **12-month expected stress amount = Σ (PD × EAD)** — the number that converts risk
   into provisioning foresight
3. Sector-wise and geography-wise stress heatmaps
4. **CGTMSE portfolio quality panel** — accounts by suitability category, avg. PD of the
   "Suitable" book
5. **Cluster contagion alerts** (`GET /cluster-alerts`, `src/graph_contagion.py`) —
   anchor-buyer clusters whose average PD is materially elevated relative to the
   portfolio, ranked by stress index
6. Action queues: field visit / stock audit / GST-bank mismatch / high CC utilisation /
   negative text remarks
7. **Growth pipeline** (`GET /growth-pipeline`) — pre-qualified healthy accounts flagged
   for working-capital enhancement or term-loan outreach, with suggested product and
   indicative quantum

## 15. Sample Borrower Output Card

A full worked example from this repo's actual trained model output — not a mock-up —
is in `reports/sample_borrower_output_card.md` (regenerated by `python src/train_model.py`
+ a live `/score` call). Card design intent: no black-box number stands alone — the PD,
the grade, the drivers with reason codes, the data-quality caveat and the specific
time-bound action arrive together, functioning simultaneously as a risk output, a
monitoring instruction and an audit record.

## 16. Technology Stack

| Layer | This Prototype | Production Roadmap |
|---|---|---|
| Data generation & features | Python, pandas, numpy | PostgreSQL, object storage lake, GST/AA/bureau connectors |
| Modelling | scikit-learn, XGBoost, LightGBM | + CatBoost, MLflow model registry |
| Calibration | scikit-learn `CalibratedClassifierCV` (isotonic / sigmoid) | Same, with periodic recalibration cadence |
| Survival / timing | Discrete-time (quarterly) hazard model, scikit-learn | Cox / gradient-boosted survival (`lifelines`, real event-time data) |
| Graph contagion | NetworkX co-movement overlay | Graph neural network / graph database |
| Explainability | SHAP | Same, + explainability store / audit log service |
| Text / NLP | scikit-learn TF-IDF + Logistic Regression + keyword rules | Fine-tuned Indic transformer (IndicBERT/FinBERT-class) |
| API | FastAPI, Pydantic, uvicorn | + auth, rate limiting, ULI/OCEN-compatible endpoints |
| Dashboard | Streamlit, Plotly | React + FastAPI, Power BI/Tableau for management MIS |
| Orchestration | Direct script execution | Airflow pipelines, Great Expectations data-quality gates |
| Governance | Model version tags, evaluation reports | Feature store, drift monitoring (Evidently-class), champion/challenger automation |

## 17. Development Roadmap

| Phase | Scope | Status |
|---|---|---|
| **Phase 1 — Prototype (this hackathon)** | Synthetic dataset, segment-wise + global PD models, survival timing, graph contagion overlay, CGTMSE + Growth Propensity engines, MSME Credit Twin, FastAPI + Streamlit | ✅ **Done — this repository** |
| **Phase 2 — MVP** | Real historical MSME data with genuine SMA/NPA-derived labels; CatBoost; fine-tuned NLP; validated performance report against Section 7 targets | Not started |
| **Phase 3 — Pilot** | Parallel run at selected branches; alert-vs-outcome comparison; reason-code fine-tuning; credit officer feedback loop | Not started |
| **Phase 4 — Production** | LOS/LMS/CBS + AA/GST/ULI/OCEN integration; drift monitoring; MRM governance sign-off; quarterly recalibration | Not started |

Phase 3's parallel run — model alerts vs. actual SMA migration over two quarters — is
the credibility event that would convert this prototype into a risk-management asset
defensible before Audit Committee and regulator.

## 18. Model Governance and Risk Controls

- **Human-in-the-loop decisioning** — the Radar recommends; the sanctioning authority
  decides. No automatic rejection is ever executed solely by AI.
- **Universal explainability** — every score ships with reason codes, confidence and
  data quality; unexplainable scores are ungoverned scores.
- **Segment-wise validation** — discrimination and calibration are (this prototype)
  evaluated and shrinkage-adjusted per segment, not merely in aggregate, precisely so
  thin-file borrowers are not silently mis-served — see `models/segment_manifest.json`.
- **Data privacy and consent** — designed for DPDP-aligned, purpose-bound consent (not
  a live consent-management implementation in this open prototype).
- **Model drift monitoring** — PSI computed at train time (`src/evaluate_model.py::compute_psi`);
  full amber/red drift dashboards are a Phase 4 item.
- **Challenger framework** — logistic regression is retained as a benchmark/challenger
  against every tree-based model in `src/train_model.py::build_candidate_models`.
- **Audit trail** — every score is tagged with a `model_version` string.
- **Data security** — this open hackathon prototype has **no authentication** on its
  API; access control is required before any real deployment (see Limitations).
- **Bias/fairness testing** — not implemented in this prototype; required before
  production use (Phase 2/3 item).

> "AI should assist credit decisioning, not replace credit officers. The Radar is
> designed as a force-multiplier for banker judgment — never a substitute for it."

## 19. Expected Benefits

The figures below are the **design brief's targets/aspirations for a production
deployment on a real bank book**, stated here for context — not a claim about this
synthetic prototype's real-world performance (see Section 10 / Limitations for what
this repo's own held-out test metrics actually show):

| Dimension | Target / Aspiration |
|---|---|
| Early warning capability | Stress capture in top risk bands lifted from 16–22% to 90%+ (4–5× improvement) |
| Lead time | Median alert lead of 2–3 quarters before first SMA-2 flag |
| Assessment speed | MSME credit assessment compressed from days to hours |
| Financial inclusion | Reduced rejection of viable thin-file MSMEs |
| CGTMSE quality | Better structure-fit at sanction, lower guarantee-claim incidence over time |
| Revenue intelligence | A pre-qualified enhancement/term-loan pipeline from the existing book |

## 20. Differentiation from Existing Systems

| Existing System | Bharat MSME Credit Radar |
|---|---|
| Uses mainly structured data | Structured, alternate and unstructured data in one engine |
| Static, sanction-time appraisal | Continuous MSME Credit Twin, re-scored every month |
| Generic score across all MSMEs | Segment-wise PD with a common interpretation contract |
| Little or no explainability | SHAP contributions + standardised reason codes on every score |
| Stress detected at SMA/overdue stage | 12-month early warning with survival-timed prioritisation |
| GST used as a turnover certificate | GST authenticity, compliance, continuity and counterparty intelligence |
| Manual remarks written and forgotten | NLP engine converts CAM/FI/RCU/collection text into live signals |
| Siloed, branch-level monitoring | Portfolio heatmaps with cluster contagion and sector overlays |
| Approve / reject mindset | Governed action menu: structure, verify, watch, protect, grow, recover |
| Risk-only decisioning | Dual engine: default prediction + growth propensity together |

## 21. How to Run Locally

Requires Python 3.10+. Works on Windows, macOS and Linux — no hardcoded paths.

```bash
git clone <this-repo>
cd bharat-msme-credit-radar
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 21.1 Regenerate the data and all model artifacts (optional — pre-built artifacts are already committed)

```bash
python src/data_generator.py --n-borrowers 3500 --months 8 --out data/synthetic_msme_data.csv
python src/train_model.py
python src/evaluate_model.py
```

`train_model.py` now trains the **entire model family** in one run: the global PD model
(with comparison across Logistic Regression / Random Forest / XGBoost / LightGBM),
isotonic calibration, all seven **segment-wise models**, the **survival/timing model**,
the **growth propensity model**, and the **segment PD benchmark reference** used for
percentile scoring. Artifacts land in `models/` — see [Repository Structure](#24-repository-structure).

### 21.2 Run the FastAPI scoring service

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs: `http://localhost:8000/docs`

### 21.3 Run the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501` with 7 pages (see [Streamlit Dashboard Usage](#23-streamlit-dashboard-usage)).

### 21.4 Explore the notebooks

```bash
jupyter notebook notebooks/
```

`01_generate_synthetic_data.ipynb` → `02_feature_engineering.ipynb` →
`03_model_training_evaluation.ipynb` → `04_shap_explainability.ipynb` cover the core
pipeline (data → features → global model → SHAP) and are pre-executed with saved
outputs. The newer layers added after this hackathon's second submission round —
segment models, survival timing, graph contagion, CGTMSE and Growth Propensity engines,
Credit Twin — are implemented as standalone modules/scripts (`src/*.py`) rather than
folded into the notebooks; see [Development Roadmap](#17-development-roadmap).

## 22. API Usage

### `GET /health`

```json
{"status": "ok", "service": "Bharat MSME Credit Radar API"}
```

### `POST /score`

Only `borrower_id` is required — every other field is optional and imputed from the
training population's typical (median/mode) values. The response's `data_quality_score`
and `model_confidence` tell you how much of the score rests on data you actually
supplied vs. population defaults. See `data/sample_score_payload.json` for a full
example request body.

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d @data/sample_score_payload.json
```

Response shape (the Common Interpretation Layer contract — identical regardless of
which segment model produced it):

```json
{
  "borrower_id": "B100282",
  "pd_12m": 0.0391,
  "risk_grade": "Green",
  "health_score": 76.1,
  "health_band": "Good",
  "health_sub_scores": { "repayment_conduct": 50.0, "cashflow_strength": 100.0, "...": "..." },
  "data_quality_score": 100,
  "model_confidence": "High",
  "sma_migration_probability": 0.0,
  "expected_months_to_stress": 10.4,
  "segment_benchmark_percentile": 92.0,
  "cluster_stress_index": 73.0,
  "in_elevated_cluster": true,
  "top_risk_drivers": [{"code": "REPAY-STRESS", "description": "...", "impact": 0.0418}],
  "top_strength_drivers": [{"code": "BUR-SCORE-OK", "description": "...", "impact": -0.0341}],
  "recommended_action": "Green: continue normal monitoring, ...",
  "action_checklist": ["Continue normal monitoring.", "..."],
  "cgtmse_recommendation": "Not suitable",
  "cgtmse_suitability": {"category": "Not suitable", "viability_score": 89.7, "rationale": "...", "checklist": []},
  "growth_propensity": {"eligible": false, "reason": "Excluded: fraud-risk keyword flag present."},
  "model_version": "EXG-v1.0"
}
```

### `GET /portfolio-summary`

Total accounts, total exposure, counts/exposure by risk grade, expected 12-month stress
amount (Σ PD × EAD), average PD/health score, top-10 highest-risk accounts,
sector-wise/geography-wise summaries, a **CGTMSE portfolio quality panel**, and a
**growth pipeline summary**.

### `GET /credit-twin/{borrower_id}`

The full MSME Credit Twin: monthly trajectory, trend classification, refreshed PD/SMA
migration probability, escalation note, recommended action, watchlist flag.

### `GET /growth-pipeline?limit=50`

The pre-qualified, risk-screened list of Green/Yellow accounts eligible for proactive
growth outreach, ranked by growth propensity score.

### `GET /cluster-alerts?limit=10`

Anchor-buyer clusters currently showing elevated, broad-based stress co-movement.

## 23. Streamlit Dashboard Usage

Launch with `streamlit run app/streamlit_app.py` and use the sidebar to navigate:

1. **Executive Dashboard** — portfolio KPIs, risk-grade distribution, PD histogram,
   sector/geography expected-stress charts.
2. **Borrower Scoring** — pick an existing borrower or enter details manually; the full
   output card including confidence, SMA migration probability, segment benchmark,
   cluster contagion warning, CGTMSE suitability and growth propensity read.
3. **Portfolio Heatmap** — sector × geography PD heatmap, high-risk borrower table,
   action queues, and **cluster contagion alerts**.
4. **Explainability** — SHAP driver bar chart and reason-code table for any borrower.
5. **Model Performance** — AUC-ROC, AUC-PR, KS, Gini, recall@20%, lift, confusion
   matrix, calibration curve.
6. **MSME Credit Twin** — pick a borrower, see their Health Score / PD trajectory across
   every observed month, trend classification, and refreshed action.
7. **Growth Propensity** — the pre-qualified growth pipeline, guardrails, score
   distribution and suggested-product breakdown.

## 24. Repository Structure

```text
bharat-msme-credit-radar/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── synthetic_msme_data.csv
│   └── sample_score_payload.json
├── notebooks/
│   ├── 01_generate_synthetic_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training_evaluation.ipynb
│   └── 04_shap_explainability.ipynb
├── src/
│   ├── data_generator.py        # synthetic data + cluster/anchor-buyer + growth target
│   ├── feature_engineering.py   # 87 features, GST authenticity, text/NLP signals
│   ├── model_utils.py           # shared borrower-grouped split + fixed-vocab encoding
│   ├── train_model.py           # global model + orchestrates all other training
│   ├── segment_models.py        # Segment-Specific MSME Model Design (Section 5)
│   ├── survival_model.py        # discrete-time hazard / timing model
│   ├── graph_contagion.py       # anchor-buyer cluster contagion overlay
│   ├── growth_propensity.py     # MSME Growth Propensity Engine
│   ├── cgtmse_engine.py         # CGTMSE Suitability Engine
│   ├── credit_twin.py           # MSME Credit Twin
│   ├── evaluate_model.py        # metrics, calibration curve, PSI
│   ├── explainability.py        # SHAP + reason-code library
│   └── action_engine.py         # Health Score + risk grade + action narrative
├── api/
│   ├── main.py                  # /score, /portfolio-summary, /credit-twin, /growth-pipeline, /cluster-alerts
│   └── schemas.py
├── app/
│   └── streamlit_app.py         # 7-page dashboard
├── models/
│   ├── trained_model.pkl, calibrator.pkl, feature_list.json   # global model
│   ├── segments/*.pkl, segment_manifest.json                  # segment-wise models
│   ├── survival_model.pkl                                     # timing model
│   ├── growth_model.pkl, growth_calibrator.pkl, growth_feature_list.json
│   ├── segment_pd_reference.json                              # percentile benchmark curves
│   └── tfidf_vectorizer.pkl, text_risk_model.pkl, default_profile.json,
│       training_report.json, evaluation_report.json
├── reports/
│   ├── model_performance_report.md
│   ├── prototype_validation_note.md
│   ├── sample_borrower_output_card.md
│   └── demo_script.md
└── assets/
    ├── architecture_diagram.png
    └── dashboard_screenshot.png
```

## 25. Limitations

This is a **hackathon design-logic prototype**, not a validated credit model:

- All data is synthetic; no real bureau, GST, AA, CBS or EPFO data was used.
- **Segment-wise models** are trained on a few hundred synthetic borrowers each;
  calibration on so few of a segment's stress events is inherently noisy, which is why
  every segment prediction is shrinkage-blended toward the larger-sample global model
  rather than trusted outright (`segment_models.py::SHRINKAGE_FULL_TRUST_EVENTS`) — a
  real deployment needs orders of magnitude more borrowers per segment before trusting
  segment models at full weight.
- The **survival/timing model**'s "which quarter will stress occur in" label is a
  synthetic severity-rank proxy, not real event-time data — a legitimate demonstration
  of discrete-time hazard modelling, not a validated timing prediction.
- The **graph contagion overlay** uses live PD co-movement within a cluster (not a
  trained graph neural network) — a fast, explainable heuristic standing in for the
  production roadmap's GNN/graph-database contagion model.
- The **growth propensity target** is synthetically constructed as the mirror image of
  the stress target; it demonstrates the guardrail/eligibility architecture, not a
  validated revenue-intelligence model.
- Text remarks are template-sampled, not genuine officer free text; the NLP layer is
  TF-IDF + Logistic Regression + keyword rules, not a fine-tuned language model.
- No out-of-time / multi-cycle validation, no fairness/bias testing, no drift-monitoring
  dashboards (PSI is computed once at train time, not tracked continuously).
- Health-score weights and risk-grade/PD bands follow the brief's illustrative design,
  not a bank's back-tested policy.
- The FastAPI service has no authentication — add access control before any real use.

See `reports/prototype_validation_note.md` for the full, honest breakdown.

## 26. Final Pitch

> "Bharat MSME Credit Radar combines GST, Account Aggregator, UPI, bureau, EPFO, bank
> conduct, structured loan data and unstructured banker intelligence to forecast MSME
> loan stress 12 months in advance. It gives every account a calibrated PD, a financial
> health score, explainable risk drivers and a recommended banker action. Its Growth
> Propensity Engine reads the same signals in reverse to surface healthy borrowers ready
> for enhanced limits and new term loans — risk intelligence and revenue intelligence in
> one platform. The solution improves portfolio quality, reduces surprise defaults,
> supports CGTMSE and MSME financial inclusion, grows quality credit from the existing
> book, and transforms MSME lending from static appraisal to living credit
> intelligence."
>
> The current state predicts one stress event in five. The Radar is built toward
> catching nine in ten — a year before they happen — and to tell the banker, in plain
> language, exactly what to do about each one.
