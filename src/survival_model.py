"""
Bharat MSME Credit Radar - Survival / Timing Model
=====================================================
Implements the "Layer 4" survival-timing model described in the design
brief: distinguishing month-3 stress from month-11 stress so the banker
knows not just *whether* an account is likely to slip, but roughly *when*
— which drives prioritisation of the action queue and feeds the
`sma_migration_probability` field of the Common Interpretation Layer.

Prototype scope: rather than a full continuous-time Cox / gradient-boosted
survival model (production roadmap item — needs true event-time data from
a real loan book), this is a discrete-time hazard model over four 3-month
buckets within the 12-month window:
    Q1 = months 1-3, Q2 = months 4-6, Q3 = months 7-9, Q4 = months 10-12

For rows that do go on to stress (`stress_12m == 1`), a synthetic "which
quarter" label is assigned from a severity rank of the same deterioration
signals used elsewhere in the platform (repayment stress index, DPD trend,
cash-flow volatility, GST filing delay, CC utilisation) -- borrowers who
already look severely stressed are assumed to be closer to the event than
those with only mild drift. This is an explicitly synthetic proxy for
demonstration (there is no real event-time data in a synthetic dataset);
see reports/prototype_validation_note.md.

A multinomial classifier is then trained to predict P(quarter | eventually
stresses), which is combined with the calibrated 12-month PD to produce a
per-quarter discrete hazard curve, an expected-months-to-stress estimate,
and a near-term (Q1, i.e. within 3 months) SMA migration probability.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import align_columns, encode_features  # noqa: E402

QUARTER_LABELS = {1: "Months 1-3", 2: "Months 4-6", 3: "Months 7-9", 4: "Months 10-12"}
QUARTER_MIDPOINT_MONTHS = {1: 1.5, 2: 4.5, 3: 7.5, 4: 10.5}

SEVERITY_COLUMNS = [
    "repayment_stress_index", "dpd_trend_score", "cashflow_volatility_score",
    "gst_filing_delay_count_6m", "cc_utilization_avg_3m", "current_dpd",
]


def assign_synthetic_quarter_labels(engineered: pd.DataFrame) -> pd.Series:
    """For stressed rows only: rank a severity composite and bucket into
    quarters 1 (most severe / soonest) .. 4 (least severe / latest)."""
    severity = sum(
        _minmax(engineered[c]) for c in SEVERITY_COLUMNS if c in engineered.columns
    )
    # Higher severity -> sooner (lower quarter number). qcut on descending severity.
    ranked_pct = severity.rank(pct=True, ascending=False)
    quarter = pd.cut(ranked_pct, bins=[-0.01, 0.25, 0.50, 0.75, 1.01], labels=[1, 2, 3, 4]).astype(float)
    return quarter


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def train_survival_model(engineered: pd.DataFrame, feature_cols: list[str], categories: dict,
                          encoded_columns: list[str], seed: int = 42):
    """Trains a 4-class multinomial LogisticRegression: P(quarter | will stress).
    Only fit on rows where stress_12m == 1 (event occurred within the window)."""
    stressed = engineered[engineered["stress_12m"] == 1].copy()
    stressed["_quarter"] = assign_synthetic_quarter_labels(stressed)
    stressed = stressed.dropna(subset=["_quarter"])

    X, _ = encode_features(stressed, feature_cols, categories)
    X = align_columns(X, encoded_columns).astype(float)
    y = stressed["_quarter"].astype(int).to_numpy()

    model = LogisticRegression(max_iter=1000, C=0.5, random_state=seed)
    model.fit(X, y)
    return model


def quarter_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    """Returns an (n, 4) array of P(quarter=1..4 | will stress), reindexed
    to always cover all 4 quarters even if a class was never observed."""
    raw = model.predict_proba(X)
    full = np.zeros((X.shape[0], 4))
    for i, cls in enumerate(model.classes_):
        full[:, int(cls) - 1] = raw[:, i]
    row_sums = full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return full / row_sums


def discrete_hazard_curve(pd_12m: np.ndarray, quarter_probs: np.ndarray) -> np.ndarray:
    """Per-quarter hazard = overall 12m PD * P(quarter | will stress)."""
    return quarter_probs * pd_12m.reshape(-1, 1)


def expected_months_to_stress(pd_12m: np.ndarray, quarter_probs: np.ndarray, low_risk_floor: float = 0.02) -> np.ndarray:
    """Weighted-average month-to-stress, using quarter midpoints. For very
    low-PD accounts this number is not operationally meaningful (there is
    unlikely to be a stress event at all) -- callers should gate display on
    a minimum PD threshold."""
    midpoints = np.array([QUARTER_MIDPOINT_MONTHS[q] for q in (1, 2, 3, 4)])
    expected = (quarter_probs * midpoints.reshape(1, -1)).sum(axis=1)
    expected = np.where(pd_12m < low_risk_floor, np.nan, expected)
    return expected


def sma_migration_probability(pd_12m: np.ndarray, quarter_probs: np.ndarray) -> np.ndarray:
    """Near-term (within 3 months, i.e. Q1) probability of migration toward
    SMA-1/SMA-2 -- the headline 'Common Interpretation Layer' field."""
    return pd_12m * quarter_probs[:, 0]


def save_survival_artifacts(model, models_dir: str):
    joblib.dump(model, os.path.join(models_dir, "survival_model.pkl"))


def load_survival_artifacts(models_dir: str):
    path = os.path.join(models_dir, "survival_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)
