"""
Bharat MSME Credit Radar - Shared Model Utilities
====================================================
Borrower-grouped splitting and fixed-vocabulary categorical encoding shared
by every model in the platform (global PD model, segment-wise PD models,
survival/timing model, growth propensity model) so that train/calibration/
test/inference always produce identical, leakage-free column sets.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from feature_engineering import CATEGORICAL_COLUMNS


def group_split(df: pd.DataFrame, label_col: str = "stress_12m", seed: int = 42):
    """Split by borrower_id (not row) so a borrower's months never straddle
    train/test, avoiding leakage."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=df["borrower_id"]))
    trainval_df, test_df = df.iloc[trainval_idx], df.iloc[test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)  # 0.25 * 0.70 = 0.175 -> ~60/17.5/22.5
    train_idx, calib_idx = next(gss2.split(trainval_df, groups=trainval_df["borrower_id"]))
    train_df, calib_df = trainval_df.iloc[train_idx], trainval_df.iloc[calib_idx]

    return train_df.reset_index(drop=True), calib_df.reset_index(drop=True), test_df.reset_index(drop=True)


def encode_features(df: pd.DataFrame, feature_cols: list[str], categories: dict | None = None):
    """One-hot encode categorical columns with a fixed category vocabulary
    so train/calib/test/inference always produce identical column sets."""
    work = df[feature_cols].copy()
    if categories is None:
        categories = {c: sorted(work[c].dropna().unique().tolist()) for c in CATEGORICAL_COLUMNS if c in work.columns}

    for col, cats in categories.items():
        work[col] = pd.Categorical(work[col], categories=cats)

    encoded = pd.get_dummies(work, columns=list(categories.keys()), dummy_na=False)
    return encoded, categories


def align_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.reindex(columns=columns, fill_value=0)
