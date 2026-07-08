# 3-Minute Demo Script — Bharat MSME Credit Radar

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

**Say:** "Let's score one real borrower from the portfolio — a Chemicals manufacturer
in Surat."

**Show:** Streamlit → **Borrower Scoring** page → select `B100282 — MSME Enterprise 00282`.

Walk through the output card live:
- **12-Month PD = 12.5%**
- **Risk Grade = Amber**
- **MSME Health Score = 44.8 (Weak)**
- **Data Quality Score = 100/100**

Switch to the **Explainability** page for the same borrower and show the SHAP bar chart:
- Top risk drivers: **REPAY-STRESS** (repayment stress index), **BUR-STRESS** (bureau
  stress), **EPFO-DECLINE** (falling employee headcount), **BUR-SCORE-LOW**.
- Top strength drivers: clean bureau DPD history, stable cash-flow, strong GST
  authenticity, diversified buyer base.

**Say:** "The recommended action isn't a black box — it's derived directly from the
grade and the reason codes: *'Amber Watch: conduct field visit within 30 days, given
declining employee headcount'* — with a concrete checklist: field visit, debtor ageing
review, hold on enhancement, GST-bank reconciliation."

---

## Minute 3 — Portfolio Radar (2:00–3:00)

**Say:** "Now zoom out to the full book."

**Show:** Streamlit → **Executive Dashboard**:
- Total exposure ₹674.5 Cr across 3,500 accounts.
- Grade split: 2,682 Green / 30 Yellow / 196 Amber / 206 Red / 386 Black.
- 12-month expected stress amount ≈ ₹78.3 Cr.
- Average health score 72.2/100.

**Show:** **Portfolio Heatmap** page:
- Sector × geography PD heatmap — call out Gems & Jewellery / Construction as the
  hottest cells.
- Action queues: accounts needing a field visit, a stock audit, showing GST-bank
  mismatch, high CC utilization, or negative text remarks — each one operational and
  exportable, not just a score.

**Close with:** "Bharat MSME Credit Radar transforms MSME lending from static
appraisal to living credit intelligence."

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
