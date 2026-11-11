"""
Integration and API Unit Tests for /stability/analyze Endpoint.

Covers:
1. POST /stability/analyze with genuine 48 packets (scenario_01_normal.csv):
   - Returns 200 OK.
   - Preserves both raw and filtered values for probe 1 and probe 2.
   - Provides accurate stability metrics and hysteresis decisions.
2. Rejection of invalid sequences:
   - Fewer than 48 packets (e.g. 1 packet, 24 packets) -> 400 Bad Request with insufficient_history.
   - Temporal gaps (missing 30-min cadence) -> 400 Bad Request with insufficient_history.
   - Duplicated timestamps -> 400 Bad Request with insufficient_history.
   - Fabricated duplicated readings -> 400 Bad Request with insufficient_history.
3. Sudden change detection on scenario_04_irrigation_event:
   - Confirms sudden event triggers are captured in timeline records and metrics.
4. Noise spike suppression on scenario_02_sensor_noise:
   - Confirms raw spikes (100% / 0%) are reported in raw readings while filtered readings remain stable.
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

from app.main import app

SCENARIOS_DIR = ROOT_DIR / "data" / "synthetic_nest"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def normal_48_packets() -> list[dict]:
    df = pd.read_csv(SCENARIOS_DIR / "scenario_01_normal.csv")
    return df.to_dict(orient="records")[:48]


@pytest.fixture
def noise_48_packets() -> list[dict]:
    df = pd.read_csv(SCENARIOS_DIR / "scenario_02_sensor_noise.csv")
    return df.to_dict(orient="records")[:48]


@pytest.fixture
def irrigation_48_packets() -> list[dict]:
    df = pd.read_csv(SCENARIOS_DIR / "scenario_04_irrigation_event.csv")
    return df.to_dict(orient="records")[:48]


# ---------------------------------------------------------------------------
# 1. Success on genuine 48-packet sequence
# ---------------------------------------------------------------------------
def test_stability_analyze_success(client, normal_48_packets):
    assert len(normal_48_packets) == 48
    resp = client.post("/stability/analyze", json={"packets": normal_48_packets})
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert data["device_id"] == "NEST-001"
    assert data["crop_id"] == "chilli"
    assert data["packets_analyzed"] == 48
    assert len(data["timeline"]) == 48

    # Verify first timeline record
    rec0 = data["timeline"][0]
    assert "raw_soil_moisture_1" in rec0
    assert "raw_soil_moisture_2" in rec0
    assert "filtered_soil_moisture_1" in rec0
    assert "filtered_soil_moisture_2" in rec0
    assert "irrigation_decision" in rec0
    assert "state_description" in rec0

    # Verify metrics summary
    metrics = data["metrics"]
    assert metrics["high_frequency_jitter_reduction_pct"] > 80.0
    assert metrics["dynamic_range_retention_pct"] > 80.0


# ---------------------------------------------------------------------------
# 2. Rejection of incomplete sequences
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("length", [1, 10, 24, 47])
def test_stability_analyze_rejects_incomplete_sequence(client, normal_48_packets, length):
    incomplete = normal_48_packets[:length]
    resp = client.post("/stability/analyze", json={"packets": incomplete})
    assert resp.status_code == 400
    data = resp.json()
    assert data["status"] == "rejected"
    assert data["error"] == "insufficient_history"
    assert data["packets_provided"] == length
    assert data["packets_required"] == 48


# ---------------------------------------------------------------------------
# 3. Rejection of fabricated identical duplicate readings
# ---------------------------------------------------------------------------
def test_stability_analyze_rejects_duplicate_readings(client, normal_48_packets):
    single = normal_48_packets[0]
    # Fabricate 48 identical readings
    fabricated = [dict(single) for _ in range(48)]
    # Add valid increments to timestamps to isolate reading duplication check
    for idx, p in enumerate(fabricated):
        p["timestamp"] = f"2026-06-15T{14 + (idx//2):02d}:{(idx%2)*30:02d}:00"

    resp = client.post("/stability/analyze", json={"packets": fabricated})
    assert resp.status_code == 400
    data = resp.json()
    assert data["status"] == "rejected"
    assert "Fabricated historical data detected" in data["detail"]


# ---------------------------------------------------------------------------
# 4. Rejection of temporal gap in sequence
# ---------------------------------------------------------------------------
def test_stability_analyze_rejects_temporal_gap(client, normal_48_packets):
    gapped = [dict(p) for p in normal_48_packets]
    # Inject 2-hour gap between step 10 and step 11
    gapped[11]["timestamp"] = "2026-06-15T21:30:00"

    resp = client.post("/stability/analyze", json={"packets": gapped})
    assert resp.status_code == 400
    data = resp.json()
    assert data["status"] == "rejected"
    assert "Temporal gap detected" in data["detail"]


# ---------------------------------------------------------------------------
# 5. Outlier noise suppression in Scenario 2
# ---------------------------------------------------------------------------
def test_stability_analyze_sensor_noise_suppression(client, noise_48_packets):
    resp = client.post("/stability/analyze", json={"packets": noise_48_packets})
    assert resp.status_code == 200
    data = resp.json()

    # Index 18 contains 100% sensor fault spike
    rec_spike = data["timeline"][18]
    assert rec_spike["raw_soil_moisture_1"] == 100.0
    assert rec_spike["raw_soil_moisture_2"] == 100.0
    # Filtered signals should filter out the 100% spike
    assert rec_spike["filtered_soil_moisture_1"] < 30.0
    assert rec_spike["filtered_soil_moisture_2"] < 32.0

    # Total variation reduction must be extreme (> 95%)
    assert data["metrics"]["total_variation_reduction_pct"] > 95.0
    # Genuine clean-signal range retention (excluding injected faults) must be high (> 85%)
    assert data["metrics"]["dynamic_range_retention_pct"] > 85.0
    # Outlier suppression reported separately from genuine signal retention
    assert data["metrics"]["outliers_detected_count"] == 2
    assert data["metrics"]["outlier_suppression_pct"] > 99.0
    # Actuator chatter eliminated (0 filtered switches)
    assert data["metrics"]["filtered_switch_count"] == 0


# ---------------------------------------------------------------------------
# 6. Sudden event preservation in Scenario 4
# ---------------------------------------------------------------------------
def test_stability_analyze_irrigation_sudden_event(client, irrigation_48_packets):
    resp = client.post("/stability/analyze", json={"packets": irrigation_48_packets})
    assert resp.status_code == 200
    data = resp.json()

    # Confirm sudden events were detected
    assert data["metrics"]["sudden_events_detected_count"] > 0

    # Verify irrigation starts ON (dry soil ~13%) and ends OFF (re-wet soil ~29%)
    assert data["timeline"][0]["irrigation_decision"] is True
    assert data["timeline"][-1]["irrigation_decision"] is False


# ---------------------------------------------------------------------------
# 7. GET /stability/scenarios (Catalog Listing)
# ---------------------------------------------------------------------------
def test_get_stability_scenarios_list(client):
    resp = client.get("/stability/scenarios")
    assert resp.status_code == 200
    data = resp.json()

    assert data["data_source"] == "SYNTHETIC"
    assert "SYNTHETIC" in data["note"]
    assert len(data["scenarios"]) == 5

    scenario_ids = [s["id"] for s in data["scenarios"]]
    assert scenario_ids == [
        "scenario_01_normal",
        "scenario_02_sensor_noise",
        "scenario_03_gradual_drying",
        "scenario_04_irrigation_event",
        "scenario_05_high_humidity",
    ]

    for s in data["scenarios"]:
        assert s["data_source"] == "SYNTHETIC"
        assert s["packet_count"] == 48
        assert len(s["name"]) > 0
        assert len(s["description"]) > 0


# ---------------------------------------------------------------------------
# 8. GET /stability/scenarios/{scenario_id} (Retrieve 48 Packets)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sc_id", [
    "scenario_01_normal",
    "scenario_02_sensor_noise",
    "scenario_03_gradual_drying",
    "scenario_04_irrigation_event",
    "scenario_05_high_humidity",
])
def test_get_scenario_packets_success(client, sc_id):
    resp = client.get(f"/stability/scenarios/{sc_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["id"] == sc_id
    assert data["data_source"] == "SYNTHETIC"
    assert data["packet_count"] == 48
    assert len(data["packets"]) == 48

    # Verify packets structure and SYNTHETIC mark
    p0 = data["packets"][0]
    assert p0["data_source"] == "SYNTHETIC"
    assert "soil_moisture_1" in p0
    assert "soil_moisture_2" in p0
    assert "timestamp" in p0


def test_get_scenario_packets_rejects_unapproved_id(client):
    # Non-whitelisted scenario IDs must return 404
    resp = client.get("/stability/scenarios/invalid_scenario")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()

    # Traversal attempts must also be blocked
    resp_trav = client.get("/stability/scenarios/..%2F..%2Fother")
    assert resp_trav.status_code in (404, 400)


# ---------------------------------------------------------------------------
# 9. Advisory vs Actuator Hysteresis Separation (Scenario 3 & Spikes)
# ---------------------------------------------------------------------------
def test_api_scenario_03_advisory_below_madp_independent_of_actuator(client):
    """
    In Scenario 3 (Gradual Drying), stabilized soil moisture drops below MADP (20.0%)
    at packet 47 (~19.75%), while hysteresis keeps actuator OFF (> 19.0%).
    API MUST report irrigation_advisory=True and stress_status='IRRIGATION_STRESS',
    and never 'NO_STRESS' when stabilized_average < madp_threshold.
    """
    sc_resp = client.get("/stability/scenarios/scenario_03_gradual_drying")
    assert sc_resp.status_code == 200
    packets = sc_resp.json()["packets"]

    resp = client.post("/stability/analyze", json={"packets": packets})
    assert resp.status_code == 200
    data = resp.json()

    madp = data["madp_threshold"]
    assert madp == 20.0
    timeline = data["timeline"]
    assert len(timeline) == 48

    # Verify strict invariant across all 48 packets
    for rec in timeline:
        assert "irrigation_advisory" in rec
        assert "stress_status" in rec
        if rec["stabilized_average"] < madp:
            assert rec["irrigation_advisory"] is True
            assert rec["stress_status"] == "IRRIGATION_STRESS"
        else:
            assert rec["irrigation_advisory"] is False
            assert rec["stress_status"] == "NO_STRESS"

    # Specifically validate packet 47 (final step)
    last_rec = timeline[-1]
    assert last_rec["stabilized_average"] < madp
    assert last_rec["irrigation_advisory"] is True
    assert last_rec["stress_status"] == "IRRIGATION_STRESS"
    assert last_rec["irrigation_decision"] is False  # Actuator remains OFF due to hysteresis


def test_api_sensor_spikes_do_not_trigger_false_advisory(client, noise_48_packets):
    """
    Verify on Scenario 2 (Sensor Noise & Spikes) that raw 0% dropout spikes
    do NOT trigger false irrigation_advisory or IRRIGATION_STRESS via API.
    """
    resp = client.post("/stability/analyze", json={"packets": noise_48_packets})
    assert resp.status_code == 200
    data = resp.json()

    # Packet 42 has raw 0.0% dropout spike
    rec_42 = data["timeline"][42]
    assert rec_42["raw_average"] == 0.0
    assert rec_42["stabilized_average"] > data["madp_threshold"]
    assert rec_42["irrigation_advisory"] is False
    assert rec_42["stress_status"] == "NO_STRESS"
    assert rec_42["irrigation_decision"] is False

    # Confirm zero false advisories across all 48 packets
    assert all(r["irrigation_advisory"] is False for r in data["timeline"])
    assert all(r["stress_status"] == "NO_STRESS" for r in data["timeline"])
