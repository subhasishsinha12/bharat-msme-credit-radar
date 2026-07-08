# 4-Minute Demo Script — Bharat MSME Credit Radar

**IDBI Innovate 2026 — Track 04: MSME Credit | Predictive AI | Risk Management**

---

## Minute 1 — Problem & Data (0:00–1:00)

**Say:** "MSME credit risk assessment in India today is fragmented and document-heavy —
a credit officer manually stitches together GST returns, bank statements, bureau
reports, EPFO filings and field-visit notes, usually only at renewal time. By the time
stress shows up as 90+ DPD, it is already too late to act. Bharat MSME Credit Radar
turns these same data rails into a living, continuously-updated 12-month early warning
signal."

**Show:**
- Open `notebooks/01_generate_synthetic_data.ipynb` or the repo `data/` folder.
- Point out the six data rails represented in the synthetic dataset: **repayment
  conduct (CBS)**, **GST returns**, **bank statement / Account Aggregator cash-flow**,
  **credit bureau**, **EPFO employment**, and **unstructured text** (CAM/FI/RCU/
  collection/stock-inspection remarks).
- Mention scale: 28,000 borrower-month records across 3,500 MSME borrowers, 8 sectors,
  9 geographies, with a realistic ~6.5% 12-month stress rate.

---

## Minute 2 — Score a Borrower (1:00–2:00)

**Say:** "Let's score one real borrower from the portfolio — a Textile machinery-loan
manufacturer in Coimbatore."

**Show:** Streamlit → **Borrower Scoring** page → select `B102733 — MSME Enterprise 02733`.

Walk through the output card live:
- **12-Month PD = 12.9%**
- **Risk Grade = Amber**
- **MSME Health Score = 39.9 (Weak)**
- **Model Confidence = High**, **Model Version = MFG-v1.0** (the segment-wise
  Manufacturer model, not a one-size-fits-all score)
- **Segment Benchmark Percentile = 92nd** — worse than 92% of Manufacturer peers

Switch to the **Explainability** page for the same borrower and show the SHAP bar chart:
- Top risk drivers: **EPFO-DECL** (falling employee headcount), **REPAY-STRESS**
  (repayment stress index), **OPS-CONT-LO** (weak operating continuity).
- Top strength drivers: strong GST authenticity, stable drawing power, clean DPD
  history.

**Say:** "The recommended action isn't a black box — it's derived directly from the
grade and the reason codes: *'Amber Watch: conduct field visit within 30 days, given
declining employee headcount and weak operating-continuity signals'* — with a concrete
checklist: field visit, debtor ageing review, hold on enhancement, GST-bank
reconciliation. And this account's actual outcome in the data is a real stress event —
this is a genuine early-warning catch, months before it would show up on an overdue
report."

---

## Minute 3 — The New Layers: Credit Twin, CGTMSE, Growth (2:00–3:00)

**Say:** "Three things happen underneath every score that a static scorecard can't do."

**Show:** Streamlit → **MSME Credit Twin** page → same borrower:
- The Health Score / PD trajectory chart across all 8 observed months — point out the
  direction of travel, not just the current level.
- Expected months-to-stress and near-term SMA migration probability.

**Show:** Back on **Borrower Scoring**, scroll to:
- **CGTMSE Suitability Engine** — a 4-category read (here: *Not suitable*, with a
  viability score and rationale), not an approve/reject binary.
- **Growth Propensity Engine** — pick a Green-grade borrower instead and show an
  eligible growth lead: suggested product, indicative quantum, outreach window.

**Say:** "The same signal spine that catches this Amber account also runs a Growth
Propensity Engine on the healthy 76% of the book — the data that reveals stress early
reveals strength early."

---

## Minute 4 — Portfolio Radar & Contagion (3:00–4:00)

**Say:** "Now zoom out to the full book."

**Show:** Streamlit → **Executive Dashboard**:
- Total exposure ₹674.5 Cr across 3,500 accounts.
- Grade split: ~2,643 Green / 186 Yellow / 129 Amber / 109 Red / 433 Black.
- 12-month expected stress amount (Σ PD × EAD) ≈ ₹78.4 Cr.
- Average health score ~71.4/100.

**Show:** **Portfolio Heatmap** page:
- Sector × geography PD heatmap.
- Action queues: field visit, stock audit, GST-bank mismatch, high CC utilization,
  negative text remarks.
- **Cluster Contagion Alerts** — anchor-buyer clusters whose average PD is materially
  elevated versus the portfolio; point out that an individual account in that cluster
  can still look fine on its own numbers.

**Show:** **Growth Propensity** page — the pre-qualified pipeline, guardrails, and
suggested-product breakdown.

**Close with:** "Bharat MSME Credit Radar transforms MSME lending from static
appraisal to living credit intelligence — one platform, one interpretation contract,
risk and revenue intelligence together."

---

## Fallback / Q&A talking points

- **Why not just optimize accuracy?** With a ~6.5% stress rate, a model that always
  predicts "no stress" is 93.5% accurate and useless. We optimize recall in the
  top 20% riskiest accounts — the size of queue a collections/EWS team can actually
  action — which the calibrated model captures at ~91% on the synthetic test set.
- **Is this production-ready?** No — see the Governance & Limitations section of
  `README.md`. This is a design-logic prototype on synthetic data, not a validated
  bank-grade model.
- **How is the PD calibrated?** Isotonic regression on a held-out calibration split,
  distinct from both the training and test borrowers (grouped by `borrower_id` to
  avoid leakage).
