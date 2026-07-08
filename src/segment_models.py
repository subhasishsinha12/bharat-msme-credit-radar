"""
Bharat MSME Credit Radar - Segment-Wise MSME Model Design
=============================================================
One model cannot serve all MSME borrowers (Section 5 of the design brief):
a trader with a 15-day inventory cycle and a machinery-loan manufacturer
with 120-day receivables fail differently. This module trains a family of
segment-specific PD models beneath the shared Common Interpretation
Layer contract (src/scoring.py always emits the same output fields no
matter which segment model produced them).

Segment -> algorithm mapping follows the design brief's Section 5 table,
substituting an available open-source equivalent where a named library
(CatBoost) isn't part of this prototype's dependency set:

    Trader        -> LightGBM         (fast, handles turnover-velocity features well)
    Manufacturer  -> XGBoost          (paired with the survival timing model)
    Service       -> Random Forest    (stand-in for CatBoost's categorical handling)
    NTC           -> Logistic Reg.    (alternate-data scorecard, conservative)
    Thin-file     -> Logistic Reg.    (alternate-data scorecard, NLP-weighted)
    Existing      -> Random Forest    (full-history ensemble)
    CGTMSE        -> Random Forest    (paired with the CGTMSE suitability engine)

Each segment model is independently trained, borrower-grouped
train/calibration/test split, and (where the segment has enough positive
examples) sigmoid-calibrated -- sigmoid rather than isotonic because
segment-level calibration sets are small (a few hundred borrowers each in
this synthetic prototype) and isotonic regression overfits readily at that
scale. Segments with too few borrowers or too few stress events fall back
to the global model at scoring time (see scoring.py) rather than shipping
an unstable segment-specific PD.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
except ImportError:
    FrozenEstimator = None

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import align_columns, encode_features, group_split  # noqa: E402
from evaluate_model import compute_all_metrics  # noqa: E402

SEGMENTS_DIR_NAME = "segments"
MIN_BORROWERS_FOR_SEGMENT_MODEL = 150
MIN_STRESS_EVENTS_FOR_CALIBRATION = 25
SHRINKAGE_FULL_TRUST_EVENTS = 100  # calibration-set stress events at which the segment model is fully trusted
UNCALIBRATED_TRUST_PENALTY = 0.6   # extra discount applied when a segment model has no calibration layer at all

SEGMENT_MODEL_SPEC = {
    "Trader":       {"prefix": "TRD", "algorithm": "lightgbm",
                      "note": "Turnover velocity, inventory-cycle proxy, buyer diversity, CC utilisation."},
    "Manufacturer": {"prefix": "MFG", "algorithm": "xgboost",
                      "note": "Capacity utilisation proxies, receivable cycle, EPFO trend, DP erosion; survival timing prominent."},
    "Service":      {"prefix": "SVC", "algorithm": "random_forest",
                      "note": "Collection regularity, client concentration, billing continuity."},
    "NTC":          {"prefix": "NTC", "algorithm": "logistic_regression",
                      "note": "New-to-credit: alternate-data scorecard, conservative calibration, inclusion lens."},
    "Existing":     {"prefix": "EXG", "algorithm": "random_forest",
                      "note": "Full CBS/LMS history + all rails; richest model, monthly Credit Twin refresh."},
    "CGTMSE":       {"prefix": "CGT", "algorithm": "random_forest",
                      "note": "All rails + viability features; paired with the CGTMSE suitability engine."},
    "Thin-file":    {"prefix": "THN", "algorithm": "logistic_regression",
                      "note": "Alternate scorecard with increased NLP/text weight; human-in-loop mandatory."},
}


def _build_segment_algorithm(algorithm: str, scale_pos_weight: float, seed: int = 42):
    if algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.07, subsample=0.85,
                colsample_bytree=0.85, min_child_weight=5, reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=seed, n_jobs=-1,
            )
        except ImportError:
            algorithm = "random_forest"
    if algorithm == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.06, num_leaves=20,
                subsample=0.85, colsample_bytree=0.85, min_child_samples=20,
                reg_lambda=2.0, class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1,
            )
        except ImportError:
            algorithm = "random_forest"
    if algorithm == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=seed)
    # default / fallback
    return RandomForestClassifier(
        n_estimators=300, max_depth=7, min_samples_leaf=15,
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
    )


def train_all_segment_models(engineered: pd.DataFrame, feature_cols: list[str],
                              global_categories: dict, global_encoded_columns: list[str],
                              seed: int = 42) -> dict:
    """Trains one PD model per MSME segment. Returns a dict keyed by segment
    name with model artifacts, calibrator (or None), and evaluation metrics.
    Segments with too little data are marked `insufficient_data=True` and
    are expected to fall back to the global model at scoring time."""
    results = {}

    for segment, spec in SEGMENT_MODEL_SPEC.items():
        subset = engineered[engineered["segment"] == segment].copy()
        n_borrowers = subset["borrower_id"].nunique()

        if n_borrowers < MIN_BORROWERS_FOR_SEGMENT_MODEL:
            results[segment] = {
                "insufficient_data": True,
                "n_borrowers": int(n_borrowers),
                "prefix": spec["prefix"],
                "algorithm": spec["algorithm"],
                "note": spec["note"],
            }
            print(f"  [{segment}] only {n_borrowers} borrowers (< {MIN_BORROWERS_FOR_SEGMENT_MODEL}); "
                  f"falls back to global model at scoring time.")
            continue

        train_df, calib_df, test_df = group_split(subset, seed=seed)

        X_train, categories = encode_features(train_df, feature_cols, global_categories)
        X_train = align_columns(X_train, global_encoded_columns).astype(float)
        y_train = train_df["stress_12m"].to_numpy()

        X_calib, _ = encode_features(calib_df, feature_cols, global_categories)
        X_calib = align_columns(X_calib, global_encoded_columns).astype(float)
        y_calib = calib_df["stress_12m"].to_numpy()

        X_test, _ = encode_features(test_df, feature_cols, global_categories)
        X_test = align_columns(X_test, global_encoded_columns).astype(float)
        y_test = test_df["stress_12m"].to_numpy()

        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = _build_segment_algorithm(spec["algorithm"], scale_pos_weight, seed=seed)
        model.fit(X_train, y_train)

        n_calib_events = int(y_calib.sum())
        calibrator = None
        if n_calib_events >= MIN_STRESS_EVENTS_FOR_CALIBRATION:
            try:
                if FrozenEstimator is not None:
                    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
                else:
                    calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
                calibrator.fit(X_calib, y_calib)
            except ValueError as exc:
                print(f"  [{segment}] calibration skipped ({exc}); using raw model probabilities.")
                calibrator = None

        prob_test = (calibrator or model).predict_proba(X_test)[:, 1]
        metrics = compute_all_metrics(y_test, prob_test) if y_test.sum() > 0 else {}

        # Shrinkage weight toward the global model: segment models trained on a
        # few hundred borrowers (typical of this synthetic prototype) are prone
        # to over-confident tails, especially when uncalibrated. Rather than
        # trust the segment model outright, its influence on the final PD is
        # blended with the (larger-sample, isotonic-calibrated) global model,
        # approaching full trust only once the segment has a substantial
        # calibration-set event count. This is a partial-pooling / shrinkage
        # estimator, standard practice for small-sample segment scorecards.
        shrinkage_weight = min(1.0, n_calib_events / SHRINKAGE_FULL_TRUST_EVENTS)
        if calibrator is None:
            shrinkage_weight *= UNCALIBRATED_TRUST_PENALTY

        results[segment] = {
            "insufficient_data": False,
            "prefix": spec["prefix"],
            "algorithm": type(model).__name__,
            "version": f"{spec['prefix']}-v1.0",
            "note": spec["note"],
            "n_borrowers": int(n_borrowers),
            "n_train_rows": len(train_df),
            "n_calib_events": n_calib_events,
            "calibrated": calibrator is not None,
            "shrinkage_weight": round(shrinkage_weight, 3),
            "model": model,
            "calibrator": calibrator,
            "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        }
        cal_note = "calibrated" if calibrator is not None else "uncalibrated (too few events)"
        auc = metrics.get("auc_roc")
        print(f"  [{segment}] {type(model).__name__} ({cal_note}) — "
              f"n_borrowers={n_borrowers}  AUC-ROC={auc:.3f}" if auc is not None else
              f"  [{segment}] {type(model).__name__} ({cal_note}) — n_borrowers={n_borrowers}")

    return results


def save_segment_artifacts(segment_results: dict, models_dir: str, feature_cols: list[str],
                            global_categories: dict, global_encoded_columns: list[str]):
    seg_dir = os.path.join(models_dir, SEGMENTS_DIR_NAME)
    os.makedirs(seg_dir, exist_ok=True)

    manifest = {}
    for segment, res in segment_results.items():
        entry = {k: v for k, v in res.items() if k not in ("model", "calibrator")}
        if not res.get("insufficient_data"):
            safe_name = segment.replace(" ", "_").replace("/", "_")
            joblib.dump(res["model"], os.path.join(seg_dir, f"{safe_name}_model.pkl"))
            if res["calibrator"] is not None:
                joblib.dump(res["calibrator"], os.path.join(seg_dir, f"{safe_name}_calibrator.pkl"))
            entry["_model_file"] = f"{safe_name}_model.pkl"
            entry["_calibrator_file"] = f"{safe_name}_calibrator.pkl" if res["calibrator"] is not None else None
        manifest[segment] = entry

    manifest["_feature_cols"] = feature_cols
    manifest["_categories"] = global_categories
    manifest["_encoded_columns"] = global_encoded_columns
    with open(os.path.join(models_dir, "segment_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


class SegmentModelRegistry:
    """Loaded once by the scorer; resolves a borrower's segment to its
    dedicated PD model, falling back to the global model when the segment
    has no dedicated model or the segment value is unrecognised."""

    def __init__(self, models_dir: str):
        manifest_path = os.path.join(models_dir, "segment_manifest.json")
        self.available = os.path.exists(manifest_path)
        self.models = {}
        self.calibrators = {}
        self.meta = {}
        if not self.available:
            return

        with open(manifest_path) as f:
            manifest = json.load(f)
        self.feature_cols = manifest.pop("_feature_cols")
        self.categories = manifest.pop("_categories")
        self.encoded_columns = manifest.pop("_encoded_columns")

        seg_dir = os.path.join(models_dir, SEGMENTS_DIR_NAME)
        for segment, entry in manifest.items():
            if entry.get("insufficient_data"):
                continue
            self.models[segment] = joblib.load(os.path.join(seg_dir, entry["_model_file"]))
            if entry.get("_calibrator_file"):
                self.calibrators[segment] = joblib.load(os.path.join(seg_dir, entry["_calibrator_file"]))
            self.meta[segment] = entry

    def has_segment_model(self, segment: str) -> bool:
        return segment in self.models

    def predict_pd(self, segment: str, engineered_row: pd.DataFrame) -> tuple[float, str]:
        """Returns (raw segment pd_12m, model_version) — NOT shrinkage-blended
        with the global model; see predict_pd_with_weight / scoring.py for
        the blended value actually surfaced to callers."""
        X, _ = encode_features(engineered_row, self.feature_cols, self.categories)
        X = align_columns(X, self.encoded_columns).astype(float)
        estimator = self.calibrators.get(segment, self.models[segment])
        pd_12m = float(estimator.predict_proba(X)[:, 1][0])
        return pd_12m, self.meta[segment]["version"]

    def shrinkage_weight(self, segment: str) -> float:
        return float(self.meta[segment].get("shrinkage_weight", 1.0))
