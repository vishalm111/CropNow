"""
Validation tests for prepared Pest Modelling Data (RBB primary, WSB separate).
Verifies chronological ordering, pure environmental feature enforcement,
WSB isolation, zero missing values, and leak-free chronological fold labels.
"""

import json
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.prepare_pest_dataset import (
    ENVIRONMENTAL_FEATURE_COLS,
    assign_pest_fold_labels,
)

PEST_OUT_DIR = ROOT_DIR / "data" / "processed" / "pest_risk"
RBB_CSV = PEST_OUT_DIR / "rbb_pest_risk_dataset.csv"
WSB_CSV = PEST_OUT_DIR / "wsb_pest_counts_separate.csv"
META_JSON = PEST_OUT_DIR / "rbb_chronological_fold_thresholds.json"


def test_pest_datasets_integrity_and_separation():
    assert RBB_CSV.exists(), f"Missing RBB dataset: {RBB_CSV}"
    assert WSB_CSV.exists(), f"Missing WSB separate file: {WSB_CSV}"
    assert META_JSON.exists(), f"Missing fold thresholds JSON: {META_JSON}"

    rbb = pd.read_csv(RBB_CSV)
    wsb = pd.read_csv(WSB_CSV)

    assert len(rbb) == 363
    assert len(wsb) == 363
    assert rbb.isna().sum().sum() == 0
    assert wsb.isna().sum().sum() == 0

    # Ensure WSB is strictly excluded from RBB modelling dataset
    assert "WSB" not in rbb.columns
    assert "wsb_count" not in rbb.columns
    assert "wsb_count" in wsb.columns
    assert "rbb_count" not in wsb.columns

    # Ensure non-environmental features are excluded from ENVIRONMENTAL_FEATURE_COLS
    forbidden = {"Date", "Year", "Month", "Day", "Week_Number", "Moon_Category", "RBB", "WSB", "Data_Split"}
    assert forbidden.isdisjoint(set(ENVIRONMENTAL_FEATURE_COLS))
    for col in ENVIRONMENTAL_FEATURE_COLS:
        assert col in rbb.columns


def test_chronological_ordering_and_causal_lags():
    rbb = pd.read_csv(RBB_CSV)

    # Strictly sorted by (sort_year, Week_Number) with no duplicates
    assert rbb.duplicated(subset=["sort_year", "Week_Number"]).sum() == 0
    sorted_check = rbb.sort_values(["sort_year", "Week_Number"]).reset_index(drop=True)
    assert rbb["sort_year"].equals(sorted_check["sort_year"])
    assert rbb["Week_Number"].equals(sorted_check["Week_Number"])

    # Exact 7-day cadence across all consecutive rows
    dates = pd.to_datetime(rbb["canonical_week_date"])
    diffs = dates.diff().dropna().dt.days
    assert (diffs == 7).all()

    # Verify causal 1-week weather lag exact equality
    assert np.allclose(rbb["t2m_lag_1w"].iloc[1:].to_numpy(), rbb["T2M"].iloc[:-1].to_numpy())
    assert np.allclose(rbb["rain_lag_1w"].iloc[1:].to_numpy(), rbb["PRECTOTCORR"].iloc[:-1].to_numpy())


def test_chronological_fold_quantiles_use_training_only():
    rbb = pd.read_csv(RBB_CSV)
    meta = json.loads(META_JSON.read_text(encoding="utf-8"))

    for yr_str, info in meta["chronological_walk_forward_folds"].items():
        val_yr = int(yr_str)
        if val_yr == 2024:
            train_df = rbb[rbb["sort_year"] < 2024]
            val_df = rbb[rbb["sort_year"] >= 2024]
        else:
            train_df = rbb[rbb["sort_year"] < val_yr]
            val_df = rbb[rbb["sort_year"] == val_yr]

        # Ensure zero future years leak into training fold
        assert (train_df["sort_year"] < val_yr).all()

        expected_lbls, t_low, t_high = assign_pest_fold_labels(train_df["RBB"] if "RBB" in train_df else train_df["rbb_count"], val_df["rbb_count"])
        assert round(t_low, 4) == info["rbb_q33_train_only"]
        assert round(t_high, 4) == info["rbb_q67_train_only"]
        assert val_df["rbb_risk_walk_forward_oof"].tolist() == expected_lbls


def test_saved_pest_risk_model_loads_and_predicts():
    import joblib

    model_path = BACKEND_DIR / "app" / "ml" / "models_pest_risk" / "best_pest_model.joblib"
    meta_path = BACKEND_DIR / "app" / "ml" / "models_pest_risk" / "model_metadata.json"
    report_path = BACKEND_DIR / "app" / "ml" / "evaluation" / "pest_risk" / "pest_risk_evaluation_report.md"

    assert model_path.exists(), f"Missing best pest model: {model_path}"
    assert meta_path.exists(), f"Missing pest model metadata: {meta_path}"
    assert report_path.exists(), f"Missing pest evaluation report: {report_path}"

    artifact = joblib.load(model_path)
    assert artifact["target_pest"] == "RBB"
    assert artifact["class_names"] == ["LOW", "MEDIUM", "HIGH"]
    feature_cols = artifact["feature_cols"]
    assert feature_cols == ENVIRONMENTAL_FEATURE_COLS
    assert len(feature_cols) == 26

    df = pd.read_csv(RBB_CSV)
    X = df[feature_cols].to_numpy(dtype=float)
    preds = artifact["estimator"].predict(X)
    probs = artifact["estimator"].predict_proba(X)

    assert len(preds) == len(df)
    assert set(np.unique(preds)).issubset({0, 1, 2})
    assert probs.shape == (len(df), 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
