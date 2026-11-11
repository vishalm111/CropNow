"""
Validation tests for the prepared Fungal-Risk Modelling Dataset (Flax Rust).
Verifies schema integrity, zero missing values, strict [d-7..d-1] temporal alignment,
and leak-free training-fold-only quantile labeling.
"""

import json
import pathlib
import sys
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.prepare_fungal_dataset import assign_fold_risk_labels

DATA_CSV = ROOT_DIR / "data" / "processed" / "fungal_risk" / "flax_rust_fungal_risk_dataset.csv"
THRESH_JSON = ROOT_DIR / "data" / "processed" / "fungal_risk" / "flax_rust_fold_thresholds.json"


def test_prepared_fungal_dataset_integrity():
    assert DATA_CSV.exists(), f"Missing prepared dataset: {DATA_CSV}"
    assert THRESH_JSON.exists(), f"Missing fold thresholds file: {THRESH_JSON}"

    df = pd.read_csv(DATA_CSV)
    assert len(df) == 70
    assert df.isna().sum().sum() == 0
    assert df.duplicated().sum() == 0
    assert df["observation_id"].is_unique

    # Verify strict causal lookback window [d-7 .. d-1]
    assert (df["window_end_day"] == df["days_since_june_1st"] - 1).all()
    assert (df["window_start_day"] == df["days_since_june_1st"] - 7).all()

    # Verify required feature groups exist
    required_cols = [
        "temp_mean_7d", "temp_min_7d", "temp_max_7d", "dp_depression_mean_7d",
        "rh_mean_7d", "rh_ge_80_hours_7d",
        "lw_mean_7d", "lw_wet_hours_7d", "favorable_infection_hours_7d",
        "rain_total_7d", "rain_days_count_7d", "soil_moisture_mean_7d",
        "temp_trend_slope_7d", "rh_trend_slope_7d", "lw_hours_trend_slope_7d",
        "rain_trend_slope_7d", "temp_delta_3d_vs_4d", "lw_hours_delta_3d_vs_4d",
        "prevalence", "risk_label_loyo_oof", "risk_label_loso_oof",
    ]
    for col in required_cols:
        assert col in df.columns


def test_loyo_and_loso_labels_match_training_fold_only_quantiles():
    df = pd.read_csv(DATA_CSV)
    meta = json.loads(THRESH_JSON.read_text(encoding="utf-8"))

    # Verify LOYO: thresholds computed strictly on rows where year != test_year
    for test_year_str, info in meta["loyo_fold_thresholds"].items():
        test_year = int(test_year_str)
        train_df = df[df["year"] != test_year]
        test_df = df[df["year"] == test_year]

        expected_labels, t_low, t_high = assign_fold_risk_labels(
            train_df["prevalence"], test_df["prevalence"]
        )
        assert round(t_low, 6) == info["q33_low_threshold"]
        assert round(t_high, 6) == info["q67_high_threshold"]
        assert test_df["risk_label_loyo_oof"].tolist() == expected_labels

    # Verify LOSO: thresholds computed strictly on rows where site != test_site
    for test_site, info in meta["loso_fold_thresholds"].items():
        train_df = df[df["site"] != test_site]
        test_df = df[df["site"] == test_site]

        expected_labels, t_low, t_high = assign_fold_risk_labels(
            train_df["prevalence"], test_df["prevalence"]
        )
        assert round(t_low, 6) == info["q33_low_threshold"]
        assert round(t_high, 6) == info["q67_high_threshold"]
        assert test_df["risk_label_loso_oof"].tolist() == expected_labels


def test_assign_fold_risk_labels_prevents_test_leakage():
    train_prev = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    # Extreme test values should NEVER shift training thresholds
    test_prev_a = np.array([0.02, 0.05, 0.08])
    test_prev_b = np.array([0.02, 0.99, 0.99])

    lbls_a, low_a, high_a = assign_fold_risk_labels(train_prev, test_prev_a)
    lbls_b, low_b, high_b = assign_fold_risk_labels(train_prev, test_prev_b)

    assert low_a == low_b
    assert high_a == high_b
    assert lbls_a == ["LOW", "MEDIUM", "HIGH"]
    assert lbls_b == ["LOW", "HIGH", "HIGH"]


def test_saved_fungal_risk_model_loads_and_predicts():
    import joblib

    model_path = BACKEND_DIR / "app" / "ml" / "models_fungal_risk" / "best_fungal_model.joblib"
    report_path = BACKEND_DIR / "app" / "ml" / "evaluation" / "fungal_risk" / "fungal_risk_evaluation_report.md"
    assert model_path.exists(), f"Missing best fungal model: {model_path}"
    assert report_path.exists(), f"Missing fungal evaluation report: {report_path}"

    artifact = joblib.load(model_path)
    assert artifact["class_names"] == ["LOW", "MEDIUM", "HIGH"]
    feature_cols = artifact["feature_cols"]
    assert len(feature_cols) == 38

    df = pd.read_csv(DATA_CSV)
    X = df[feature_cols].to_numpy(dtype=float)
    preds = artifact["estimator"].predict(X)
    probs = artifact["estimator"].predict_proba(X)

    assert len(preds) == len(df)
    assert set(np.unique(preds)).issubset({0, 1, 2})
    assert probs.shape == (len(df), 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
