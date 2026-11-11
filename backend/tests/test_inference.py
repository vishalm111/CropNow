"""
Unit and Integration Tests for NEST Prediction Inference Service.

Covers:
1. Strict requirement of 48 genuine, consecutive NEST packets (30-min cadence)
2. Rejection of incomplete sequences with an insufficient_history response
3. Prevention of duplicate readings / fabricated historical data
4. Detection of temporal gaps and duplicate timestamps
5. Successful prediction on 48 genuine consecutive packets across all 4 models
6. Pressure unit conversion (kPa -> hPa)
7. Relative moisture formula matching training
8. Removal of fabricated defaults (e.g. EC 6.5)
9. Validation of required telemetry features and physical bounds
10. FastAPI endpoint testing for quick predict and batch predict
"""

import pathlib
import sys
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

SCENARIO_PATH = ROOT_DIR / "data" / "synthetic_nest" / "scenario_01_normal.csv"

SYNTHETIC_DEVICE_ID = "SYNTH-NEST-001"
SYNTHETIC_FARM_ID = "synth-farm-01"

from app.main import app
from app.services.inference import (
    CROP_THRESHOLDS,
    REQUIRED_NEST_FIELDS,
    UNAVAILABLE_TRAINING_FEATURES,
    InsufficientHistoryError,
    ModelRegistry,
    PredictionBlockedError,
)


@pytest.fixture
def synthetic_sample() -> dict:
    """Project-owned synthetic NEST packet matching the 21-field ingestion schema."""
    return {
        "device_id": SYNTHETIC_DEVICE_ID,
        "farm_id": SYNTHETIC_FARM_ID,
        "crop_id": "chilli",
        "month": 7,
        "day": 12,
        "hour": 10,
        "temperature": 27.4,
        "humidity": 68.0,
        "pressure": 101.2,
        "solar_radiation": 510.0,
        "wind_speed": 1.8,
        "wind_direction": 185.0,
        "rainfall": 0.0,
        "soil_moisture_1": 24.0,
        "soil_moisture_2": 28.0,
        "soil_temperature": 23.5,
        "leaf_wetness": 20.0,
        "co": 0.02,
        "no2": 0.01,
        "o3": 0.03,
        "pm2_5": 18.0,
    }


@pytest.fixture
def genuine_48_packets() -> list[dict]:
    df = pd.read_csv(SCENARIO_PATH)
    # Ensure all required NEST fields and timestamp are present
    return df.to_dict(orient="records")[:48]


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.get()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Prediction on 48 genuine consecutive packets across all 4 models
# ---------------------------------------------------------------------------
def test_genuine_48_packets_prediction_all_models(registry, genuine_48_packets):
    assert len(genuine_48_packets) == 48
    for model_key in registry.available_models:
        result = registry.predict(genuine_48_packets, model_key=model_key)
        assert "predicted_soil_moisture" in result
        assert "current_soil_moisture" in result
        assert "change_predicted" in result
        assert 0.0 <= result["predicted_soil_moisture"] <= 100.0
        assert "unavailable_features" in result
        assert set(UNAVAILABLE_TRAINING_FEATURES).issubset(set(result["unavailable_features"]))


# ---------------------------------------------------------------------------
# 2. Rejection of incomplete sequences with InsufficientHistoryError
# ---------------------------------------------------------------------------
def test_reject_single_packet_with_insufficient_history(registry, synthetic_sample):
    with pytest.raises(InsufficientHistoryError) as exc_info:
        registry.predict([synthetic_sample])
    err = exc_info.value
    assert err.error_code == "insufficient_history"
    assert err.packets_provided == 1
    assert err.packets_required == 48
    assert "incomplete sequence" in str(err).lower()


@pytest.mark.parametrize("packet_count", [0, 5, 24, 47])
def test_reject_incomplete_sequence(registry, genuine_48_packets, packet_count):
    incomplete = genuine_48_packets[:packet_count]
    with pytest.raises(InsufficientHistoryError) as exc_info:
        registry.predict(incomplete)
    err = exc_info.value
    assert err.error_code == "insufficient_history"
    assert err.packets_provided == packet_count
    assert err.packets_required == 48


# ---------------------------------------------------------------------------
# 3. Never duplicate readings to fabricate historical data
# ---------------------------------------------------------------------------
def test_reject_duplicated_readings_fabrication(registry, synthetic_sample):
    # Fabricating a 48-packet history by duplicating a single packet 48 times
    fabricated = [dict(synthetic_sample) for _ in range(48)]
    with pytest.raises(InsufficientHistoryError) as exc_info:
        registry.predict(fabricated)
    err = exc_info.value
    assert err.error_code == "insufficient_history"
    assert "fabricated" in str(err).lower() or "duplicate" in str(err).lower()


# ---------------------------------------------------------------------------
# 4. Require consecutive 30-minute timestamps (detect gaps & duplicates)
# ---------------------------------------------------------------------------
def test_reject_timestamp_gap(registry, genuine_48_packets):
    gapped_packets = [dict(p) for p in genuine_48_packets]
    # Introduce a 2-hour gap between packet 10 and 11
    dt_gap = pd.to_datetime(gapped_packets[9]["timestamp"]) + pd.Timedelta(hours=2)
    gapped_packets[10]["timestamp"] = dt_gap.isoformat()

    with pytest.raises(InsufficientHistoryError) as exc_info:
        registry.predict(gapped_packets)
    err = exc_info.value
    assert err.error_code == "insufficient_history"
    assert "temporal gap" in str(err).lower()


def test_reject_duplicate_timestamps(registry, genuine_48_packets):
    dup_packets = [dict(p) for p in genuine_48_packets]
    # Make packet 5 share timestamp with packet 4
    dup_packets[5]["timestamp"] = dup_packets[4]["timestamp"]

    with pytest.raises(InsufficientHistoryError) as exc_info:
        registry.predict(dup_packets)
    err = exc_info.value
    assert err.error_code == "insufficient_history"
    assert "duplicate timestamp" in str(err).lower()


def test_reject_missing_timestamp_in_sequence(registry, genuine_48_packets):
    missing_ts_packets = [dict(p) for p in genuine_48_packets]
    del missing_ts_packets[12]["timestamp"]

    with pytest.raises(InsufficientHistoryError) as exc_info:
        registry.predict(missing_ts_packets)
    err = exc_info.value
    assert err.error_code == "insufficient_history"
    assert "missing timestamp" in str(err).lower()


# ---------------------------------------------------------------------------
# 5. Pressure conversion: kPa -> hPa
# ---------------------------------------------------------------------------
def test_pressure_conversion_kpa_to_hpa(registry, synthetic_sample):
    assert synthetic_sample["pressure"] == 101.2  # In kPa
    vec, unavail = registry._packet_to_features(synthetic_sample, residual=True)
    weather_pressure = vec[3]
    assert weather_pressure == pytest.approx(1012.0, rel=1e-4)

    scaler = registry._scalers["xgb_residual"]
    scaled_pressure = (weather_pressure - scaler.mean_[3]) / scaler.scale_[3]
    assert abs(scaled_pressure) < 2.0  # within normal z-score range


# ---------------------------------------------------------------------------
# 6. Relative moisture and distance calculations matching training
# ---------------------------------------------------------------------------
def test_relative_moisture_formula_matches_training(registry, synthetic_sample):
    vec, _ = registry._packet_to_features(synthetic_sample, residual=True)
    rel_sm = vec[13]
    dist_madp = vec[14]
    dist_fc = vec[15]

    chilli = CROP_THRESHOLDS["chilli"]
    sm_avg = (synthetic_sample["soil_moisture_1"] + synthetic_sample["soil_moisture_2"]) / 2.0
    expected_rel = (sm_avg - chilli["pwp"]) / (chilli["fc"] - chilli["pwp"])
    expected_dist_madp = sm_avg - chilli["madp"]
    expected_dist_fc = sm_avg - chilli["fc"]

    assert rel_sm == pytest.approx(expected_rel, rel=1e-4)
    assert dist_madp == pytest.approx(expected_dist_madp, rel=1e-4)
    assert dist_fc == pytest.approx(expected_dist_fc, rel=1e-4)
    assert dist_fc < 0


# ---------------------------------------------------------------------------
# 7. Removal of fabricated EC default (6.5) and baseline imputation
# ---------------------------------------------------------------------------
def test_removed_fabricated_ec_default(registry, synthetic_sample):
    vec, unavail = registry._packet_to_features(synthetic_sample, residual=True)
    ec_val = vec[8]
    assert ec_val != 6.5
    assert ec_val == pytest.approx(181.747, rel=1e-2)
    assert "ec" in unavail

    sample_with_ec = dict(synthetic_sample)
    sample_with_ec["ec"] = 250.0
    vec_custom, unavail_custom = registry._packet_to_features(sample_with_ec, residual=True)
    assert vec_custom[8] == 250.0
    assert "ec" not in unavail_custom


# ---------------------------------------------------------------------------
# 8. Block predictions when required sensor features are missing / corrupt
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("missing_field", [
    "temperature", "humidity", "pressure", "rainfall",
    "wind_speed", "solar_radiation", "soil_temperature",
    "soil_moisture_1", "soil_moisture_2", "crop_id",
])
def test_block_prediction_when_required_field_missing(registry, genuine_48_packets, missing_field):
    bad_packets = [dict(p) for p in genuine_48_packets]
    del bad_packets[0][missing_field]
    with pytest.raises(PredictionBlockedError) as exc_info:
        registry.predict(bad_packets)
    assert "missing required telemetry feature" in str(exc_info.value).lower()


@pytest.mark.parametrize("corrupt_field,corrupt_value", [
    ("humidity", -5.0),
    ("humidity", 105.0),
    ("pressure", -1.0),
    ("pressure", 0.0),
    ("soil_moisture_1", -10.0),
    ("soil_moisture_2", 110.0),
    ("temperature", 85.0),
    ("temperature", -50.0),
])
def test_block_prediction_on_physical_anomalies(registry, genuine_48_packets, corrupt_field, corrupt_value):
    bad_packets = [dict(p) for p in genuine_48_packets]
    bad_packets[5][corrupt_field] = corrupt_value
    with pytest.raises(PredictionBlockedError) as exc_info:
        registry.predict(bad_packets)
    assert "physical anomaly" in str(exc_info.value).lower() or "blocked" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 9. API endpoint verification for single-packet validation and batch prediction
# ---------------------------------------------------------------------------
def test_api_quick_predict_single_packet_validation_only(client, synthetic_sample):
    # Single packet sent to quick predict is strictly for validation, NOT temporal forecast
    resp = client.post("/predict/quick", json={"packet": synthetic_sample, "model": "xgb_residual"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "valid"
    assert data["device_id"] == SYNTHETIC_DEVICE_ID
    assert data["farm_id"] == SYNTHETIC_FARM_ID
    assert data["crop_id"] == "chilli"
    assert data["current_soil_moisture"] == 26.0  # (24.0 + 28.0) / 2
    assert data["pressure_hpa"] == 1012.0        # 101.2 kPa * 10
    assert data["temporal_prediction"] == "unavailable_requires_48_packets"
    assert "irrigation_advice" in data
    assert "unavailable_features" in data
    assert "Single packet input validated successfully" in data["message"]


def test_api_quick_predict_invalid_packet_blocked(client, synthetic_sample):
    # Corrupt a required physical range: humidity out of bounds
    invalid_sample = dict(synthetic_sample)
    invalid_sample["humidity"] = 150.0  # Invalid > 100%
    resp = client.post("/predict/quick", json={"packet": invalid_sample, "model": "xgb_residual"})
    assert resp.status_code in (400, 422)


def test_api_batch_predict_incomplete_rejected(client, genuine_48_packets):
    # 10 packets sent to batch predict must be rejected with insufficient_history
    resp = client.post("/predict/batch", json={"packets": genuine_48_packets[:10], "model": "xgb_residual"})
    assert resp.status_code == 400
    data = resp.json()
    assert data["status"] == "rejected"
    assert data["error"] == "insufficient_history"
    assert data["packets_provided"] == 10
    assert data["packets_required"] == 48


def test_api_batch_predict_genuine_48_packets_success(client, genuine_48_packets):
    # 48 genuine consecutive packets sent to batch predict must succeed
    resp = client.post("/predict/batch", json={"packets": genuine_48_packets, "model": "xgb_residual"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_soil_moisture"] > 0
    assert data["current_soil_moisture"] > 0
    assert data["confidence"] == "high"
    assert "unavailable_features" in data
    assert len(data["unavailable_features"]) > 0
