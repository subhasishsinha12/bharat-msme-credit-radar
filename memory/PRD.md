# Bharat MSME Credit Radar - PRD

## Original Problem Statement
Build a prototype of "Bharat MSME Credit Radar" — a bank-grade, India-stack-native,
explainable 12-Month Predictive Default Intelligence and Early Warning Engine for
Indian MSME loans (IDBI Innovate 2026, Track 04). Fuses structured loan data,
alternate rails (GST, AA, UPI, Bureau, EPFO) and NLP-processed banker notes into
per-borrower PD, Health Score, Risk Grade, SHAP reason codes and banker actions.

## Architecture
- **Existing engine (kept as-is)** `/app/src` — synthetic data generator, feature
  engineering (~300 features across 18 families), XGBoost/LightGBM/CatBoost
  ensemble with isotonic calibration, SHAP explainability, rule-based health
  score + action engine. Trained artifacts in `/app/models/`, synthetic 3500-row
  panel in `/app/data/`.
- **Backend** `/app/backend/server.py` — FastAPI on port 8001 under `/api`. Wraps
  the existing `CreditRadarScorer`, adds MongoDB persistence (`scoring_history`,
  `bulk_uploads`, `notes_analyses`) and an Emergent-LLM-powered banker-notes NLP
  endpoint (Gemini 3 Flash preview) that emits structured JSON stress signals.
- **Frontend** `/app/frontend` — React 18 + Tailwind, Terminal-Dark aesthetic
  (Bloomberg-terminal inspired, Chivo / IBM Plex Sans / JetBrains Mono, strict
  Green/Yellow/Amber/Red/Black semantic colors, no gradients or glassmorphism).
  Recharts for charts.

## Core Requirements (from doc)
- 12-Month PD (calibrated)
- MSME Health Score (0-100, weighted sub-scores)
- Early Warning Grade (Green/Yellow/Amber/Red/Black)
- SHAP-based top risk / strength drivers with banker-friendly explanations
- Recommended banker action + action checklist
- CGTMSE suitability recommendation
- Portfolio heatmaps (sector, geography, segment)
- CSV bulk scoring
- NLP on unstructured banker notes

## What's Been Implemented (Jan 2026)
### Backend
- `GET /api/health` — liveness
- `GET /api/metadata` — segments / sectors / geographies / risk grades
- `GET /api/portfolio/summary` — aggregate risk view (grades, sectors, geographies, segments, top-10 high risk, expected stress amount)
- `GET /api/borrowers` — paginated list with filters (search / grade / sector / geography / segment) and sort
- `GET /api/borrowers/{id}` — full scored detail: PD, health score, sub-scores, top risk/strength drivers, action narrative + checklist, CGTMSE recommendation, alternate & structured signals, banker remarks
- `GET /api/borrowers/{id}/history` — 8-month scored trajectory (PD, health, GST turnover, DPD, CC utilization, bureau score) — the "MSME Credit Twin"
- `POST /api/borrowers/score` — score a partial payload; persists to `scoring_history`
- `POST /api/borrowers/bulk-score` — multipart CSV upload; imputes missing columns from training-population defaults, returns scored rows
- `GET /api/growth/summary` — revenue pipeline: total candidates, hot count, band distribution, top suggested products
- `GET /api/growth/candidates` — Growth Propensity Engine: filterable candidate list with growth score, suggested product, indicative quantum, outreach window
- `POST /api/notes/analyze` — LLM (Gemini 3 Flash preview via emergentintegrations) extracts stress/fraud/sentiment signals from free-text banker remarks; persists to `notes_analyses`
- `GET /api/notes/history` — recent LLM analyses

### Frontend pages
- `/` **Portfolio Command Center** — KPI row, risk grade distribution bar+pie, sector & geography heatmaps (clickable → drill-down to filtered borrowers), top-10 high risk action queue
- `/borrowers` **Borrowers list** — searchable/filterable/paginated table with URL-param sync (shareable filtered views)
- `/borrowers/:id` **Borrower detail** — 12M PD hero, MSME Health Score radial gauge, exposure panel with CGTMSE note, banker action, top 5 risk & 5 strength drivers, **8-month MSME Credit Twin trajectory (PD/Health/GST/DPD/CC/Bureau charts)**, health sub-scores, alternate signals, banker remarks with one-click NLP handoff, "Officer Memo" button
- `/borrowers/:id/memo` **Officer Memo** — standalone print-friendly white letterhead, A4 CSS `@page`, PDF-ready via browser Save as PDF
- `/growth` **Growth Radar** — Growth Propensity Engine dashboard: candidate count, priority+hot, revenue pipeline, propensity band distribution (Priority/Hot/Emerging/Passive/Dormant), suggested products breakdown, filterable candidate table with growth score, indicative quantum, outreach window
- `/notes` **Banker Notes Analyzer** — LLM-powered structured signal extraction
- `/upload` **CSV Bulk Scoring** — drop-zone upload with template CSV

## Growth Propensity Engine Logic (v1)
Weighted composite (28% GST growth YoY, 18% CC utilization headroom, 14% cashflow surplus, 12% bureau, 10% DSCR, 10% EPFO delta, 8% inverse PD) with penalties for bureau enquiries, and hard gates that disqualify Red/Black grade, health<55, PD>15%, fraud flag or GST-bank mismatch.

Suggested product logic:
- CGTMSE eligible + turnover growth >8% → CGTMSE-backed Enhancement
- CC util ≥75% + growth >5% → CC Limit Enhancement
- CC util <45% + growth >10% → Term Loan / Capex
- Turnover growth >15% → New Working Capital
- Otherwise → Product Refresh

Indicative quantum: 15-40% of sanctioned limit based on growth & utilization. Outreach window: 0-30 / 30-60 / 60-90 days by score.

## Prioritized Backlog / Next Actions
- P1: Portfolio-page filters (drill into a specific sector / geography → filtered borrower list)
- P1: Trend charts on borrower detail (historical PD / GST / bank-credit trajectory using the panel data)
- P1: Growth Propensity Engine surface — score & suggested products for healthy borrowers
- P2: Auth (JWT or Emergent Google) — role tailored dashboards for Branch / RO / HO
- P2: Export scored CSV / PDF officer memo per borrower
- P2: Contagion Graph (GNN) view of counterparty risk
- P2: Model governance panel — feature drift charts (Evidently AI)
- P2: NLP model self-hosted (IndicBERT) fallback for on-prem deployment

## Notes
- All synthetic data — clearly disclaimed as prototype-only, not production lending
- SHAP explanations run for every /borrowers/{id} score in real time (~200-400ms)
- Grades: Green <5% PD, Yellow 5-10%, Amber 10-20%, Red 20-35%, Black ≥35%
