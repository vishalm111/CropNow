"""
Unit and Integration Tests for `backend/app/ml/nest_fungal_mapper.py`.

Validates:
1. Exact 38-feature order matching saved `best_fungal_model.joblib`.
2. Mathematical parity with training formulas (temperature, Magnus-Tetens dew point,
   humidity durations, leaf wetness durations, rainfall sums, soil moisture scaling,
   7-day OLS slopes, and 3d-vs-4d deltas).
3. Strict rejection of incomplete (<336), oversized (>336), gapped, duplicate,
   out-of-order, fabricated, or missing/NaN/out-of-range packets without inventing values.
"""

from __future__ import annotations

import math
import pathlib
import sys
from datetime import datetime, timedelta

import joblib
import numpy as np
import pytest

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.nest_fungal_mapper import (
    REQUIRED_PACKETS_FUNGAL,
    FungalFeatureVector,
    _compute_magnus_dew_point,
    map_nest_packets_to_fungal_features,
)
from app.services.inference import InsufficientHistoryError, PredictionBlockedError


def _make_valid_336_packets(start_ts: str = "2026-07-01T00:00:00") -> list[dict]:
    """Generate 336 deterministic varying 30-min NEST packets (7 days x 48 pkts/day)."""
    base_dt = datetime.fromisoformat(start_ts)
    packets: list[dict] = []
    for idx in range(REQUIRED_PACKETS_FUNGAL):
        day_idx = idx // 48
        step_in_day = idx % 48
        ts = base_dt + timedelta(minutes=30 * idx)

        # Deterministic varying weather across 7 days
        temp = 12.0 + day_idx * 1.0 + 4.0 * math.sin(math.pi * step_in_day / 48.0)
        rh = 65.0 + (20.0 if step_in_day < 16 else 0.0) + day_idx * 1.0  # 16 pkts (8h) >= 80% per day
        lw = 25.0 if step_in_day < 12 else 0.0                            # 12 pkts (6h) >= 20 & > 0 per day
        rain = 1.5 if (day_idx in (1, 3, 6) and step_in_day == 0) else 0.0
        solar = max(0.0, 500.0 * math.sin(math.pi * step_in_day / 48.0))
        sm1 = 20.0 + day_idx * 1.0
        sm2 = 20.0 + day_idx * 1.0

        packets.append({
            "timestamp": ts.isoformat(),
            "device_id": "NEST-001",
            "farm_id": "farm-01",
            "crop_id": "chilli",
            "temperature": round(temp, 4),
            "humidity": round(min(rh, 100.0), 4),
            "leaf_wetness": lw,
            "rainfall": rain,
            "solar_radiation": round(solar, 2),
            "soil_moisture_1": sm1,
            "soil_moisture_2": sm2,
            "soil_temperature": 18.0,
            "wind_speed": 1.2,
            "wind_direction": 180,
            "pressure": 100.8,
            "co": 0.04,
            "no2": 0.03,
            "o3": 0.04,
            "pm2_5": 45.0,
            "month": ts.month,
            "day": ts.day,
            "hour": ts.hour,
        })
    return packets


def test_mapper_exact_feature_order_and_model_compatibility():
    packets = _make_valid_336_packets()
    res = map_nest_packets_to_fungal_features(packets)

    assert isinstance(res, FungalFeatureVector)
    assert res.packets_processed == 336
    assert len(res.feature_names) == 38
    assert res.feature_array.shape == (1, 38)
    assert np.all(np.isfinite(res.feature_array))

    # Load saved fungal model (read-only) and verify exact feature column match
    model_path = BACKEND_DIR / "app" / "ml" / "models_fungal_risk" / "best_fungal_model.joblib"
    artifact = joblib.load(model_path)
    assert list(res.feature_names) == artifact["feature_cols"]

    pred = artifact["estimator"].predict(res.feature_array)
    probs = artifact["estimator"].predict_proba(res.feature_array)
    assert int(pred[0]) in (0, 1, 2)
    assert probs.shape == (1, 3)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_mapper_exact_training_formulas():
    """Verify exact mathematical match against analytical daily signals."""
    base_dt = datetime.fromisoformat("2026-07-01T00:00:00")
    packets: list[dict] = []
    for idx in range(336):
        day_idx = idx // 48
        step_in_day = idx % 48
        ts = base_dt + timedelta(minutes=30 * idx)

        # Daily mean temperature climbs strictly by +1.0 C/day: 10, 11, 12, 13, 14, 15, 16
        # Plus symmetric +/- 2 C swing so daily min = mean-2, max = mean+2, diurnal range = 4
        temp = (10.0 + day_idx) + (2.0 if step_in_day < 24 else -2.0)
        # First 10 packets of day -> RH = 85% (5.0h >= 80%), rest -> 60%
        rh = 85.0 if step_in_day < 10 else 60.0
        # First 8 packets of day -> LW = 25.0 (4.0h > 0 and >= 20), rest -> 0
        lw = 25.0 if step_in_day < 8 else 0.0
        # Rain = 2.0 mm on day 4 (step 0) and 4.0 mm on day 6 (step 0)
        rain = 2.0 if (day_idx == 4 and step_in_day == 0) else (4.0 if (day_idx == 6 and step_in_day == 0) else 0.0)
        # Soil moisture increases +1.0%/day from 20% -> 26% (Water_Content: 0.20 -> 0.26)
        sm = 20.0 + day_idx * 1.0

        packets.append({
            "timestamp": ts.isoformat(),
            "temperature": temp,
            "humidity": rh,
            "leaf_wetness": lw,
            "rainfall": rain,
            "solar_radiation": 200.0,
            "soil_moisture_1": sm,
            "soil_moisture_2": sm,
        })

    res = map_nest_packets_to_fungal_features(packets)
    f = res.feature_dict

    # Temperature checks
    assert f["temp_mean_7d"] == pytest.approx(13.0, abs=1e-4)
    assert f["temp_min_7d"] == pytest.approx(8.0, abs=1e-4)
    assert f["temp_max_7d"] == pytest.approx(18.0, abs=1e-4)
    assert f["temp_diurnal_range_mean_7d"] == pytest.approx(4.0, abs=1e-4)
    assert f["temp_mean_24h"] == pytest.approx(16.0, abs=1e-4)
    assert f["temp_trend_slope_7d"] == pytest.approx(1.0, abs=1e-4)
    # recent_3d means (14,15,16 -> 15.0) - prior_4d means (10,11,12,13 -> 11.5) = 3.5
    assert f["temp_delta_3d_vs_4d"] == pytest.approx(3.5, abs=1e-4)

    # Humidity duration checks (5h/day * 7 days = 35h >= 80%, 0h >= 90%)
    assert f["rh_ge_80_hours_7d"] == pytest.approx(35.0, abs=1e-2)
    assert f["rh_ge_90_hours_7d"] == pytest.approx(0.0, abs=1e-2)

    # Leaf wetness duration & favorable infection (LW > 0 & temp in [5, 25], all temps are 8..18 -> 4h/day * 7 = 28h)
    assert f["lw_wet_hours_7d"] == pytest.approx(28.0, abs=1e-2)
    assert f["lw_heavy_wet_hours_7d"] == pytest.approx(28.0, abs=1e-2)
    assert f["lw_wet_hours_24h"] == pytest.approx(4.0, abs=1e-2)
    assert f["favorable_infection_hours_7d"] == pytest.approx(28.0, abs=1e-2)

    # Rainfall checks
    assert f["rain_total_7d"] == pytest.approx(6.0, abs=1e-4)
    assert f["rain_max_daily_7d"] == pytest.approx(4.0, abs=1e-4)
    assert f["rain_days_count_7d"] == 2
    assert f["rain_total_24h"] == pytest.approx(4.0, abs=1e-4)

    # Soil moisture fraction scaling (/ 100.0) & trend (+0.01/day)
    assert f["soil_moisture_mean_7d"] == pytest.approx(0.23, abs=1e-4)
    assert f["soil_moisture_mean_24h"] == pytest.approx(0.26, abs=1e-4)
    assert f["soil_moisture_trend_slope_7d"] == pytest.approx(0.01, abs=1e-6)

    # Dew point check against direct Magnus-Tetens evaluation
    t_arr = np.array([p["temperature"] for p in packets])
    rh_arr = np.array([p["humidity"] for p in packets])
    dp_arr = _compute_magnus_dew_point(t_arr, rh_arr)
    assert f["dp_mean_7d"] == pytest.approx(float(np.mean(dp_arr)), abs=1e-3)
    assert f["dp_depression_mean_7d"] == pytest.approx(float(np.mean(t_arr - dp_arr)), abs=1e-3)


@pytest.mark.parametrize("count", [1, 48, 335, 337])
def test_mapper_rejects_wrong_packet_count(count):
    packets = _make_valid_336_packets()
    if count < 336:
        subset = packets[:count]
    else:
        extra = dict(packets[-1])
        extra["timestamp"] = (datetime.fromisoformat(packets[-1]["timestamp"]) + timedelta(minutes=30)).isoformat()
        subset = packets + [extra]

    with pytest.raises(InsufficientHistoryError):
        map_nest_packets_to_fungal_features(subset)


def test_mapper_rejects_temporal_gap_duplicate_and_out_of_order():
    # 1. Temporal gap (60 min jump)
    pkts_gap = _make_valid_336_packets()
    pkts_gap[150]["timestamp"] = (
        datetime.fromisoformat(pkts_gap[149]["timestamp"]) + timedelta(minutes=60)
    ).isoformat()
    with pytest.raises(InsufficientHistoryError):
        map_nest_packets_to_fungal_features(pkts_gap)

    # 2. Duplicate timestamp
    pkts_dup = _make_valid_336_packets()
    pkts_dup[200]["timestamp"] = pkts_dup[199]["timestamp"]
    with pytest.raises(InsufficientHistoryError):
        map_nest_packets_to_fungal_features(pkts_dup)

    # 3. Out-of-order timestamp
    pkts_rev = _make_valid_336_packets()
    pkts_rev[10]["timestamp"], pkts_rev[11]["timestamp"] = pkts_rev[11]["timestamp"], pkts_rev[10]["timestamp"]
    with pytest.raises(InsufficientHistoryError):
        map_nest_packets_to_fungal_features(pkts_rev)


def test_mapper_rejects_fabricated_identical_readings():
    packets = _make_valid_336_packets()
    base = packets[0]
    fabricated = []
    for p in packets:
        row = dict(base)
        row["timestamp"] = p["timestamp"]
        fabricated.append(row)

    with pytest.raises(InsufficientHistoryError):
        map_nest_packets_to_fungal_features(fabricated)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("leaf_wetness", None),
        ("temperature", float("nan")),
        ("rainfall", -0.5),
        ("humidity", 0.0),
        ("humidity", 105.0),
        ("soil_moisture_1", 120.0),
    ],
)
def test_mapper_rejects_missing_or_invalid_values_without_imputing(field, bad_value):
    packets = _make_valid_336_packets()
    packets[120][field] = bad_value
    with pytest.raises(PredictionBlockedError):
        map_nest_packets_to_fungal_features(packets)
