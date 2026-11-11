"""
Unit and Integration Tests for `backend/app/ml/nest_pest_mapper.py`.

Validates:
1. Exact 26-feature order parity against saved `best_pest_model.joblib`.
2. Mathematical parity with training formulas and unit conversions
   (`PRECTOTCORR = sum/7`, `GWETTOP = SM_avg / SP`, circular `WD2M` + `wd_sin/cos`,
   1-week lags, 4-week rolling means/sums, and 4-week OLS trend slopes).
3. Strict rejection of incomplete (<1344), oversized (>1344), gapped, duplicate,
   out-of-order, fabricated, missing/NaN, out-of-range, or uncalibrated crop inputs.
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

from app.ml.nest_pest_mapper import (
    REQUIRED_PACKETS_PEST,
    PestFeatureVector,
    map_nest_packets_to_pest_features,
)
from app.services.inference import InsufficientHistoryError, PredictionBlockedError


def _make_valid_1344_packets(start_ts: str = "2026-06-01T00:00:00") -> list[dict]:
    """Generate 1,344 deterministic varying 30-min NEST packets (4 weeks x 336 pkts/wk)."""
    base_dt = datetime.fromisoformat(start_ts)
    packets: list[dict] = []
    for idx in range(REQUIRED_PACKETS_PEST):
        wk_idx = idx // 336
        step_in_wk = idx % 336
        ts = base_dt + timedelta(minutes=30 * idx)

        temp = 25.0 + wk_idx * 0.5 + 3.0 * math.sin(math.pi * (step_in_wk % 48) / 48.0)
        rh = 80.0 + wk_idx * 1.0 - 4.0 * math.sin(math.pi * (step_in_wk % 48) / 48.0)
        rain = 1.4 if (step_in_wk % 48 == 0) else 0.0
        ws = 1.0 + wk_idx * 0.2 + (0.5 if step_in_wk % 2 == 0 else -0.2)
        wd = (170.0 + wk_idx * 10.0) % 360.0
        sm = 25.0 + wk_idx * 2.0

        packets.append({
            "timestamp": ts.isoformat(),
            "device_id": "NEST-001",
            "farm_id": "farm-01",
            "crop_id": "chilli",
            "temperature": round(temp, 4),
            "humidity": round(min(max(rh, 0.0), 100.0), 4),
            "rainfall": rain,
            "wind_speed": round(ws, 4),
            "wind_direction": wd,
            "soil_moisture_1": sm,
            "soil_moisture_2": sm,
            "leaf_wetness": 35.0,
            "solar_radiation": 400.0,
        })
    return packets


def test_pest_mapper_exact_feature_order_and_model_compatibility():
    packets = _make_valid_1344_packets()
    res = map_nest_packets_to_pest_features(packets)

    assert isinstance(res, PestFeatureVector)
    assert res.packets_processed == 1344
    assert res.crop_id == "chilli"
    assert res.saturation_point_used == 45.0
    assert len(res.feature_names) == 26
    assert res.feature_array.shape == (1, 26)
    assert np.all(np.isfinite(res.feature_array))

    model_path = BACKEND_DIR / "app" / "ml" / "models_pest_risk" / "best_pest_model.joblib"
    artifact = joblib.load(model_path)
    assert list(res.feature_names) == artifact["feature_cols"]

    pred = artifact["estimator"].predict(res.feature_array)
    probs = artifact["estimator"].predict_proba(res.feature_array)
    assert int(pred[0]) in (0, 1, 2)
    assert probs.shape == (1, 3)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_pest_mapper_exact_training_formulas_and_conversions():
    """Verify exact match for weekly means, mm/day rainfall, GWETTOP saturation, circular wind, and 4w slopes."""
    base_dt = datetime.fromisoformat("2026-06-01T00:00:00")
    packets: list[dict] = []
    for idx in range(1344):
        wk_idx = idx // 336
        step_in_wk = idx % 336
        ts = base_dt + timedelta(minutes=30 * idx)

        # Weekly mean temp climbs strictly by +1.0 C/wk: 24, 25, 26, 27 (+/- 2.0 diurnal swing)
        temp = (24.0 + wk_idx * 1.0) + (2.0 if step_in_wk % 2 == 0 else -2.0)
        # Weekly mean RH drops by -2.0%/wk: 86, 84, 82, 80
        rh = 86.0 - wk_idx * 2.0
        # Weekly rain totals: wk0=7mm (1mm/d), wk1=14mm (2mm/d), wk2=21mm (3mm/d), wk3=28mm (4mm/d)
        rain = ((wk_idx + 1) * 7.0) if step_in_wk == 0 else 0.0
        # Wind speed: wk2=1.0, wk3=2.0 +/- 0.5 -> WS2M=2.0, min=1.5, max=2.5, spread=1.0
        ws = (0.5 + wk_idx * 0.5) + (0.5 if step_in_wk % 2 == 0 else -0.5)
        # Circular wind in week 3: half 350 deg, half 10 deg -> circular mean MUST be 0.0 deg (NOT 180 deg)
        wd = (350.0 if step_in_wk % 2 == 0 else 10.0) if wk_idx == 3 else 90.0
        # Soil moisture % for chilli (SP=45%): wk0=22.5 (0.5), wk1=27.0 (0.6), wk2=31.5 (0.7), wk3=36.0 (0.8)
        sm = 22.5 + wk_idx * 4.5

        packets.append({
            "timestamp": ts.isoformat(),
            "crop_id": "chilli",
            "temperature": temp,
            "humidity": rh,
            "rainfall": rain,
            "wind_speed": ws,
            "wind_direction": wd,
            "soil_moisture_1": sm,
            "soil_moisture_2": sm,
        })

    res = map_nest_packets_to_pest_features(packets)
    f = res.feature_dict

    # Current week (week 3) base features
    assert f["T2M"] == pytest.approx(27.0, abs=1e-4)
    assert f["T2M_MIN"] == pytest.approx(25.0, abs=1e-4)
    assert f["T2M_MAX"] == pytest.approx(29.0, abs=1e-4)
    assert f["temp_diurnal_range"] == pytest.approx(4.0, abs=1e-4)
    assert f["RH2M"] == pytest.approx(80.0, abs=1e-4)
    assert f["PRECTOTCORR"] == pytest.approx(4.0, abs=1e-4)     # 28mm / 7 days = 4.0 mm/day
    assert f["WS2M"] == pytest.approx(2.0, abs=1e-4)
    assert f["WS2M_MIN"] == pytest.approx(1.5, abs=1e-4)
    assert f["WS2M_MAX"] == pytest.approx(2.5, abs=1e-4)
    assert f["wind_gust_spread"] == pytest.approx(1.0, abs=1e-4)

    # Circular wind direction across 350/10 deg boundary
    assert f["WD2M"] == pytest.approx(0.0, abs=1e-3) or f["WD2M"] == pytest.approx(360.0, abs=1e-3)
    assert f["wd_sin"] == pytest.approx(0.0, abs=1e-4)
    assert f["wd_cos"] == pytest.approx(1.0, abs=1e-4)

    # Soil moisture GWETTOP normalization (36.0 / 45.0 = 0.80)
    assert f["GWETTOP"] == pytest.approx(0.8, abs=1e-4)

    # 1-week lags (week 2)
    assert f["t2m_lag_1w"] == pytest.approx(26.0, abs=1e-4)
    assert f["rh2m_lag_1w"] == pytest.approx(82.0, abs=1e-4)
    assert f["rain_lag_1w"] == pytest.approx(3.0, abs=1e-4)
    assert f["gwettop_lag_1w"] == pytest.approx(0.7, abs=1e-4)
    assert f["ws2m_lag_1w"] == pytest.approx(1.5, abs=1e-4)

    # 4-week summaries & trends
    assert f["t2m_mean_4w"] == pytest.approx(25.5, abs=1e-4)      # mean(24,25,26,27)
    assert f["rh2m_mean_4w"] == pytest.approx(83.0, abs=1e-4)     # mean(86,84,82,80)
    assert f["rain_sum_4w"] == pytest.approx(10.0, abs=1e-4)      # 1+2+3+4 mm/day sums
    assert f["gwettop_mean_4w"] == pytest.approx(0.65, abs=1e-4)  # mean(0.5,0.6,0.7,0.8)
    assert f["t2m_trend_4w"] == pytest.approx(1.0, abs=1e-5)
    assert f["rh2m_trend_4w"] == pytest.approx(-2.0, abs=1e-5)
    assert f["gwettop_trend_4w"] == pytest.approx(0.1, abs=1e-5)


@pytest.mark.parametrize("count", [1, 48, 336, 1343, 1345])
def test_pest_mapper_rejects_wrong_packet_count(count):
    packets = _make_valid_1344_packets()
    if count < 1344:
        subset = packets[:count]
    else:
        extra = dict(packets[-1])
        extra["timestamp"] = (datetime.fromisoformat(packets[-1]["timestamp"]) + timedelta(minutes=30)).isoformat()
        subset = packets + [extra]

    with pytest.raises(InsufficientHistoryError):
        map_nest_packets_to_pest_features(subset)


def test_pest_mapper_rejects_gap_duplicate_out_of_order_and_fabricated():
    # 1. Temporal gap (60-min jump)
    pkts_gap = _make_valid_1344_packets()
    pkts_gap[600]["timestamp"] = (
        datetime.fromisoformat(pkts_gap[599]["timestamp"]) + timedelta(minutes=60)
    ).isoformat()
    with pytest.raises(InsufficientHistoryError):
        map_nest_packets_to_pest_features(pkts_gap)

    # 2. Duplicate timestamp
    pkts_dup = _make_valid_1344_packets()
    pkts_dup[900]["timestamp"] = pkts_dup[899]["timestamp"]
    with pytest.raises(InsufficientHistoryError):
        map_nest_packets_to_pest_features(pkts_dup)

    # 3. Out-of-order timestamp
    pkts_rev = _make_valid_1344_packets()
    pkts_rev[120]["timestamp"], pkts_rev[121]["timestamp"] = pkts_rev[121]["timestamp"], pkts_rev[120]["timestamp"]
    with pytest.raises(InsufficientHistoryError):
        map_nest_packets_to_pest_features(pkts_rev)

    # 4. Fabricated identical readings across all 1,344 packets
    pkts_fab = _make_valid_1344_packets()
    base = pkts_fab[0]
    fabricated = [{**base, "timestamp": p["timestamp"]} for p in pkts_fab]
    with pytest.raises(InsufficientHistoryError):
        map_nest_packets_to_pest_features(fabricated)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("wind_speed", None),
        ("wind_direction", float("nan")),
        ("wind_direction", 361.0),
        ("wind_speed", -0.5),
        ("rainfall", -1.0),
        ("humidity", 105.0),
        ("temperature", 85.0),
        ("soil_moisture_2", 110.0),
    ],
)
def test_pest_mapper_rejects_missing_or_invalid_units(field, bad_value):
    packets = _make_valid_1344_packets()
    packets[500][field] = bad_value
    with pytest.raises(PredictionBlockedError):
        map_nest_packets_to_pest_features(packets)


def test_pest_mapper_rejects_unknown_crop_without_saturation_point():
    packets = _make_valid_1344_packets()
    packets[-1]["crop_id"] = "unknown_crop_xyz"
    with pytest.raises(PredictionBlockedError):
        map_nest_packets_to_pest_features(packets)

    # Explicitly providing calibrated saturation_point succeeds
    res = map_nest_packets_to_pest_features(packets, saturation_point=50.0)
    assert res.saturation_point_used == 50.0
