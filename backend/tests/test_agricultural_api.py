"""
API integration tests for unified `POST /predict/agricultural` endpoint
and regression smoke checks on preserved endpoints.
"""

from __future__ import annotations

import math
import pathlib
import sys
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app

VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _make_nest_packets(
    count: int,
    *,
    crop_id: str = "chilli",
    start_ts: str = "2026-06-01T00:00:00",
) -> list[dict]:
    """
    Generate `count` chronological 30-min 21-field NEST telemetry packets
    satisfying irrigation/stability (48), fungal (336), and pest (1,344) mappers.
    """
    base_dt = datetime.fromisoformat(start_ts)
    packets: list[dict] = []
    for idx in range(count):
        wk_idx = idx // 336
        day_idx = idx // 48
        step_in_day = idx % 48
        ts = base_dt + timedelta(minutes=30 * idx)

        temp = 24.0 + (wk_idx * 0.6) + 4.0 * math.sin(math.pi * step_in_day / 48.0)
        rh = 68.0 + (18.0 if step_in_day < 16 else 0.0) + (wk_idx * 1.0)
        lw = 25.0 if step_in_day < 12 else 0.0
        rain = 1.4 if (step_in_day == 0 and day_idx % 2 == 1) else 0.0
        solar = max(0.0, 500.0 * math.sin(math.pi * step_in_day / 48.0))
        sm1 = 22.0 + 0.01 * (idx % 96) + wk_idx * 0.5
        sm2 = 22.5 + 0.01 * (idx % 96) + wk_idx * 0.5
        ws = 1.1 + wk_idx * 0.15 + (0.4 if idx % 2 == 0 else -0.15)
        wd = (170.0 + wk_idx * 12.0 + (idx % 5)) % 360.0

        packets.append({
            "timestamp": ts.isoformat(),
            "device_id": "NEST-001",
            "farm_id": "farm-01",
            "crop_id": crop_id,
            "temperature": round(temp, 4),
            "humidity": round(min(max(rh, 0.0), 100.0), 4),
            "leaf_wetness": lw,
            "rainfall": rain,
            "solar_radiation": round(solar, 2),
            "soil_moisture_1": round(sm1, 3),
            "soil_moisture_2": round(sm2, 3),
            "soil_temperature": round(19.0 + 0.5 * math.sin(math.pi * step_in_day / 48.0), 3),
            "wind_speed": round(ws, 4),
            "wind_direction": round(wd, 2),
            "pressure": 100.8,
            "co": 0.04,
            "no2": 0.03,
            "o3": 0.04,
            "pm2_5": 42.0,
            "month": ts.month,
            "day": ts.day,
            "hour": ts.hour,
        })
    return packets


def test_agricultural_24_packets_all_insufficient_history(client: TestClient):
    packets = _make_nest_packets(24, crop_id="chilli")
    resp = client.post("/predict/agricultural", json={"packets": packets})
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "success"
    assert body["total_packets_received"] == 24
    assert body["crop_id"] == "chilli"

    for key, req_pkts in [("irrigation", 48), ("fungal_risk", 336), ("pest_risk", 1344)]:
        section = body[key]
        assert section["status"] == "insufficient_history"
        assert section["ready"] is False
        assert section["packets_required"] == req_pkts
        assert section["packets_provided"] == 24
        assert section["confidence"] is None
        assert section["confidence_level"] == "unavailable"
        assert len(section["warnings"]) >= 1

    assert body["fungal_risk"]["risk_level"] is None
    assert body["pest_risk"]["risk_level"] is None
    assert body["irrigation"]["predicted_soil_moisture_24h"] is None


def test_agricultural_48_packets_only_irrigation_ready(client: TestClient):
    packets = _make_nest_packets(48, crop_id="chilli")
    resp = client.post(
        "/predict/agricultural",
        json={"packets": packets, "irrigation_model": "xgb_residual"},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_packets_received"] == 48

    irr = body["irrigation"]
    assert irr["status"] == "ready"
    assert irr["ready"] is True
    assert irr["packets_required"] == 48
    assert irr["packets_provided"] == 48
    assert irr["predicted_soil_moisture_24h"] is not None
    assert irr["stabilized_soil_moisture"] is not None
    assert isinstance(irr["irrigation_advisory"], bool)
    assert irr["stress_status"] in {"IRRIGATION_STRESS", "NO_STRESS"}
    assert isinstance(irr["actuator_decision"], bool)
    assert irr["irrigation_advice"] is not None
    assert 0.0 <= irr["confidence"] <= 1.0
    assert irr["confidence_level"] in {"low", "medium", "high"}

    assert body["fungal_risk"]["status"] == "insufficient_history"
    assert body["fungal_risk"]["ready"] is False
    assert body["fungal_risk"]["packets_required"] == 336
    assert body["pest_risk"]["status"] == "insufficient_history"
    assert body["pest_risk"]["ready"] is False
    assert body["pest_risk"]["packets_required"] == 1344


def test_agricultural_336_packets_irrigation_and_fungal_ready_with_chilli_domain_warning(
    client: TestClient,
):
    packets = _make_nest_packets(336, crop_id="chilli")
    resp = client.post("/predict/agricultural", json={"packets": packets})
    assert resp.status_code == 200
    body = resp.json()

    assert body["irrigation"]["status"] == "ready"
    assert body["irrigation"]["ready"] is True

    fungal = body["fungal_risk"]
    assert fungal["status"] == "ready"
    assert fungal["ready"] is True
    assert fungal["packets_required"] == 336
    assert fungal["packets_provided"] == 336
    assert fungal["risk_level"] in VALID_RISK_LEVELS
    assert set(fungal["probabilities"].keys()) == VALID_RISK_LEVELS
    assert sum(fungal["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert 0.0 <= fungal["confidence"] <= 1.0
    assert fungal["confidence_level"] in {"low", "medium", "high"}

    fungal_warning_text = " ".join(fungal["warnings"] + fungal["domain_limitations"]).lower()
    assert "domain_limitation" in fungal_warning_text
    assert "flax" in fungal_warning_text or "melampsora" in fungal_warning_text
    assert "chilli" in fungal_warning_text

    pest = body["pest_risk"]
    assert pest["status"] == "insufficient_history"
    assert pest["ready"] is False
    assert pest["packets_required"] == 1344


def test_agricultural_1344_packets_all_three_models_ready_and_domain_warnings(
    client: TestClient,
):
    packets = _make_nest_packets(1344, crop_id="chilli")
    resp = client.post("/predict/agricultural", json={"packets": packets})
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_packets_received"] == 1344
    assert body["irrigation"]["status"] == "ready"
    assert body["irrigation"]["ready"] is True

    for key, req_pkts, expected_domain_terms in [
        ("fungal_risk", 336, ("flax", "melampsora")),
        ("pest_risk", 1344, ("rice", "rbb")),
    ]:
        risk = body[key]
        assert risk["status"] == "ready"
        assert risk["ready"] is True
        assert risk["packets_required"] == req_pkts
        assert risk["packets_provided"] == req_pkts
        assert risk["risk_level"] in VALID_RISK_LEVELS
        assert set(risk["probabilities"].keys()) == VALID_RISK_LEVELS
        assert sum(risk["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
        assert risk["confidence"] == pytest.approx(max(risk["probabilities"].values()), abs=1e-4)
        assert risk["confidence_level"] in {"low", "medium", "high"}
        assert risk["window_start_timestamp"] is not None
        assert risk["window_end_timestamp"] is not None

        combined_text = " ".join(risk["warnings"] + risk["domain_limitations"]).lower()
        assert "domain_limitation" in combined_text
        assert "chilli" in combined_text
        assert any(term in combined_text for term in expected_domain_terms)


def test_agricultural_window_isolation_corruption_in_older_pest_history_does_not_block_recent_models(
    client: TestClient,
):
    """
    Corrupting packet index 100 (outside latest 336 and 48 windows, inside 1,344 pest window)
    must independently block pest_risk while irrigation and fungal_risk remain ready.
    """
    packets = _make_nest_packets(1344, crop_id="chilli")
    packets[100]["temperature"] = None  # NaN/missing in older week 1 only

    resp = client.post("/predict/agricultural", json={"packets": packets})
    assert resp.status_code == 200
    body = resp.json()

    assert body["irrigation"]["status"] == "ready"
    assert body["irrigation"]["ready"] is True
    assert body["fungal_risk"]["status"] == "ready"
    assert body["fungal_risk"]["ready"] is True

    pest = body["pest_risk"]
    assert pest["status"] in {"blocked", "insufficient_history"}
    assert pest["ready"] is False
    assert pest["risk_level"] is None
    assert len(pest["warnings"]) >= 1


def test_agricultural_rejects_mixed_device_ids_and_empty_packets(client: TestClient):
    resp_empty = client.post("/predict/agricultural", json={"packets": []})
    assert resp_empty.status_code == 422

    packets = _make_nest_packets(48, crop_id="chilli")
    packets[0]["device_id"] = "NEST-OTHER"
    resp_mixed = client.post("/predict/agricultural", json={"packets": packets})
    assert resp_mixed.status_code == 422


def test_preserved_existing_endpoints_smoke(client: TestClient):
    models_resp = client.get("/predict/models")
    assert models_resp.status_code == 200

    scenarios_resp = client.get("/stability/scenarios")
    assert scenarios_resp.status_code == 200

    packets_48 = _make_nest_packets(48, crop_id="chilli")
    batch_resp = client.post(
        "/predict/batch",
        json={"packets": packets_48, "model": "xgb_residual"},
    )
    assert batch_resp.status_code == 200
    assert "predicted_soil_moisture" in batch_resp.json()

    quick_resp = client.post(
        "/predict/quick",
        json={"packet": packets_48[-1]},
    )
    assert quick_resp.status_code == 200
    assert quick_resp.json()["status"] == "valid"
