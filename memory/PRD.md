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
- `POST /api/borrowers/score` — score a partial payload; persists to `scoring_history`
- `POST /api/borrowers/bulk-score` — multipart CSV upload; imputes missing columns from training-population defaults, returns scored rows
- `POST /api/notes/analyze` — LLM (Gemini 3 Flash preview via emergentintegrations) extracts stress/fraud/sentiment signals from free-text banker remarks; persists to `notes_analyses`
- `GET /api/notes/history` — recent LLM analyses

### Frontend pages
- `/` **Portfolio Command Center** — KPI row (accounts, exposure, expected stress, avg health), risk grade distribution bar+pie, sector & geography heatmaps, top-10 high risk action queue
- `/borrowers` **Borrowers list** — searchable filtered paginated table, click through to detail
- `/borrowers/:id` **Borrower detail** — 12M PD hero, MSME Health Score radial gauge, exposure panel with CGTMSE note, banker action narrative + checklist, top 5 risk & 5 strength drivers (terminal-style reason codes), health sub-scores bars, alternate-signals grid, banker remarks with a one-click LLM-analysis handoff
- `/notes` **Banker Notes Analyzer** — free-text inputs for CAM/FI/RCU/collection/stock notes → LLM extracts structured signals with severity, evidence quote, explanation, sentiment, recommended action, summary; recent-analyses history
- `/upload` **CSV Bulk Scoring** — drop-zone upload, CSV template download, scored results table with per-row grade

## Tech Stack
- Backend: FastAPI, Motor/MongoDB, Pandas, XGBoost, LightGBM, SHAP, scikit-learn, joblib, emergentintegrations
- Frontend: React 18, react-router-dom, Recharts, Tailwind, sonner, lucide-react
- LLM: Gemini 3 Flash preview via Emergent Universal Key

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
