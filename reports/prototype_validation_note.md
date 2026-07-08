# Prototype Validation Note — Bharat MSME Credit Radar

**IDBI Innovate 2026 — Track 04: MSME Credit | Predictive AI | Risk Management**

This note explains what this prototype does and does not demonstrate, and what would be
required before any of its logic could inform a real lending decision.

## 1. What was built and evaluated

| Item | Value |
|---|---|
| Dataset size | 28,000 borrower-month records, 3,500 unique borrowers, 8 months each |
| Stress rate | 6.52% (target band: 5–8%) |
| Train / Calibration / Test split | 14,696 / 4,904 / 8,400 rows — split **by borrower_id** (grouped), so no borrower's months appear in more than one split |
| Models tested | Logistic Regression, Random Forest, XGBoost, LightGBM |
| Best model selected | Random Forest (`models/feature_list.json → best_model_name`) |
| Selection criteria | Weighted composite of recall@top-20% risk band, AUC-PR, calibration (1 − Brier), and KS — not accuracy |
| Calibration method | Isotonic regression, fit on a held-out calibration split distinct from train/test |
| Held-out test AUC-ROC / AUC-PR | 0.946 / 0.715 |
| Recall captured in top 20% riskiest accounts | ~90.7% |
| Top-decile lift | ~7.8x |

Full metrics, confusion matrix and the calibration curve are in
`reports/model_performance_report.md` and `models/evaluation_report.json`.

## 2. Why recall at the top 20% risk band drives model selection

A 12-month early-warning system is judged by how many genuinely stressed accounts are
surfaced within the review capacity a bank can realistically action — a field-visit or
stock-audit queue is typically sized at the top 10–20% of a book, not 100% of it. With a
~6.5% base stress rate, a trivial "always predict non-stress" model scores >93% plain
accuracy while catching zero stressed accounts. Recall@top-20%, AUC-PR and calibration
are far more informative for this use case and were weighted accordingly during model
selection, per the design brief.

## 3. Honesty about synthetic data

All borrower records, remarks, and outcomes in this prototype are **synthetically
generated** (`src/data_generator.py`) to *resemble* Indian MSME credit behaviour — they
are not drawn from, or fitted to, any real bank's book. Specifically:

- The dataset is built around a hidden per-borrower "latent risk" factor that drives
  internally consistent repayment, GST, cash-flow, bureau, EPFO and text signals, plus a
  substantial independent noise term, so the target is learnable but not perfectly
  separable — deliberately avoiding an unrealistically "too good to be true" AUC.
- Relationships between features (e.g., GST filing delay → stress, buyer concentration →
  stress) reflect plausible domain priors encoded by the prototype's authors, not
  statistically estimated real-world elasticities.
- Text remarks are template-sampled from a fixed positive/negative phrase bank, not real
  officer language, and are consequently easier for a keyword/TF-IDF model to pick up on
  than genuine free text would be.
- Sector, geography and constitution risk tiers used in the MSME Health Score are
  illustrative policy priors, not empirically derived.

**Consequence:** the AUC-ROC, AUC-PR, KS, lift and recall figures reported here validate
that the *pipeline* (imbalance handling, calibration, explainability, action mapping)
works end-to-end and behaves sensibly — they are not evidence of real-world bank-grade
predictive performance, and must not be quoted as such outside this hackathon context.

## 4. Limitations

- No real bureau, GST, AA, or EPFO data was used or accessed.
- No fairness, disparate-impact, or protected-attribute bias testing has been performed.
- No temporal out-of-time validation (the panel spans 8 synthetic months, not multiple
  real economic cycles).
- Text NLP is a first-stage keyword + TF-IDF/LogReg model, not a validated language model.
- The health-score weighting (25/20/15/15/10/10/5%) is a reasonable starting design, not
  a regression-fitted or policy-approved weighting.
- Threshold and grade boundaries (Green/Yellow/Amber/Red/Black) follow the brief's
  illustrative PD bands, not a bank's back-tested risk appetite.

## 5. Next steps for real bank-data validation

1. Re-fit the feature engineering and model pipeline on a historical, de-identified loan
   book with a genuine 12-month forward-looking stress definition (90+ DPD / SMA-2→NPA
   migration / restructuring / write-off / legal recovery — as already encoded in
   `src/data_generator.py`'s target design).
2. Run out-of-time validation across at least 2–3 economic cycles/vintages.
3. Conduct segment-wise and protected-attribute fairness testing before any deployment.
4. Recalibrate probability bands and health-score weights against the bank's actual loss
   experience and credit policy.
5. Establish a champion/challenger framework and drift monitoring before any automated
   action is triggered from model output.
6. Route all model outputs through human credit officer review — see Governance below.

## 6. Governance & Safety Principles

- AI **assists** credit officers; it does not replace them.
- No automatic rejection, sanction, or restructuring decision solely on an AI score.
- Consent-based use of borrower data, aligned with the Digital Personal Data Protection
  (DPDP) Act's data-minimisation and purpose-limitation principles.
- Explainability (SHAP-based reason codes) is generated for every score, not just
  high-risk ones.
- Every score is tagged with a model version and reason-code set for audit trail purposes.
- Periodic model validation and recalibration on live portfolio performance.
- Segment-wise bias and fairness checks before and during production use.
- Ongoing drift monitoring on both input feature distributions and model calibration.
- A challenger-model framework to benchmark the production model on an ongoing basis.
- Secure API access control (authentication, rate limiting, audit logging) — not
  implemented in this open prototype, required before any real deployment.

---

**Disclaimer:** This prototype uses synthetic MSME data for hackathon demonstration. No
production lending decision should be made using this model without validation on
historical bank data, governance approval, calibration, fairness testing, drift
monitoring and human-in-the-loop controls.
