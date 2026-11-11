"""
Unit Tests for Soil Moisture Stability Module (stability.py).

Tests:
1. MedianFilter:
   - Suppresses isolated positive and negative spikes/outliers.
   - Preserves steady signals.
   - Odd window validation.

2. EWMASmoother:
   - Smooths high-frequency continuous jitter.
   - Detects sudden genuine physical changes (|Δ| >= sudden_threshold).
   - Fast-tracks / snaps without excessive lag latency during sudden events.

3. HysteresisSwitch:
   - Prevents actuator flapping near the threshold.
   - Activates only below (threshold - margin).
   - Deactivates only above (threshold + margin).
   - Retains state in the deadband margin.

4. SoilMoistureStabilizer (Dual Probe Integration):
   - Processes probe 1 and probe 2 separately.
   - Preserves raw readings.
   - Detects sudden events (e.g., scenario_04 irrigation event).
   - Filters out sensor spikes (e.g., scenario_02 sensor noise).
"""

import pytest
import pandas as pd
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

SCENARIOS_DIR = ROOT_DIR / "data" / "synthetic_nest"

from app.ml.stability import (
    MedianFilter,
    EWMASmoother,
    HysteresisSwitch,
    SoilMoistureStabilizer,
    StabilizedReading,
)


# ---------------------------------------------------------------------------
# 1. MedianFilter Tests
# ---------------------------------------------------------------------------
def test_median_filter_suppresses_isolated_spike():
    mf = MedianFilter(window_size=3)
    # Stream: steady at 25.0, single outlier spike to 100.0, back to 25.0
    inputs = [25.0, 25.0, 100.0, 25.0, 25.0]
    filtered = [mf.update(x) for x in inputs]

    # At index 2: window is [25.0, 25.0, 100.0] -> median is 25.0!
    assert filtered[2] == 25.0
    # At index 3: window is [25.0, 100.0, 25.0] -> median is 25.0
    assert filtered[3] == 25.0
    assert filtered[4] == 25.0


def test_median_filter_suppresses_drop_spike():
    mf = MedianFilter(window_size=5)
    # Stream with single 0.0 dropout
    inputs = [30.0, 30.0, 30.0, 0.0, 30.0, 30.0]
    filtered = [mf.update(x) for x in inputs]
    # Outlier at index 3 is rejected
    assert filtered[3] == 30.0
    assert filtered[4] == 30.0


def test_median_filter_invalid_window():
    with pytest.raises(ValueError):
        MedianFilter(window_size=0)
    with pytest.raises(ValueError):
        MedianFilter(window_size=4)  # Even window should raise ValueError


# ---------------------------------------------------------------------------
# 2. EWMASmoother Tests
# ---------------------------------------------------------------------------
def test_ewma_smooths_jitter():
    smoother = EWMASmoother(alpha=0.2, sudden_delta_threshold=3.0)
    # Constant signal with small ±0.2 jitter
    jittery_inputs = [20.0, 20.2, 19.8, 20.1, 19.9, 20.0]
    outputs = []
    for x in jittery_inputs:
        val, is_sudden = smoother.update(x)
        outputs.append(val)
        assert not is_sudden  # small jitter should not trigger sudden change flag

    # Output should stay tightly centered around 20.0
    assert abs(outputs[-1] - 20.0) < 0.1


def test_ewma_sudden_change_detection_and_bypass():
    smoother = EWMASmoother(alpha=0.1, sudden_delta_threshold=3.0, fast_alpha=0.95)
    # Baseline steady at 15.0
    for _ in range(5):
        smoother.update(15.0)

    # Sudden irrigation event: step jump from 15.0 to 25.0 (delta = 10.0 > 3.0)
    jump_val, is_sudden = smoother.update(25.0)
    assert is_sudden is True
    # Fast-alpha (0.95) ensures estimate snaps near 25.0 immediately instead of lagging at ~16.0
    assert jump_val > 24.0


# ---------------------------------------------------------------------------
# 3. HysteresisSwitch Tests
# ---------------------------------------------------------------------------
def test_hysteresis_switch_prevents_chatter():
    # Target threshold = 20.0, margin = 1.0 (Low = 19.0, High = 21.0)
    switch = HysteresisSwitch(threshold=20.0, margin=1.0, initial_state=False)

    # Signal starts high
    assert switch.update(25.0) is False
    assert switch.update(21.5) is False

    # Drops to 19.5 (below 20.0, but NOT below 19.0) -> Should NOT trigger yet (deadband)
    assert switch.update(19.5) is False

    # Drops to 18.9 (below 19.0) -> Triggers ON!
    assert switch.update(18.9) is True

    # Rises back into deadband (19.8, 20.5) -> MUST stay ON to prevent premature turnoff
    assert switch.update(19.8) is True
    assert switch.update(20.5) is True

    # Only when exceeding 21.0 does it turn OFF
    assert switch.update(21.2) is False

    # Dips slightly to 20.8 -> Stays OFF
    assert switch.update(20.8) is False


# ---------------------------------------------------------------------------
# 4. Integrated Dual-Probe Stabilizer Tests
# ---------------------------------------------------------------------------
def test_stabilizer_preserves_raw_readings():
    stabilizer = SoilMoistureStabilizer(
        median_window=3,
        ewma_alpha=0.3,
        sudden_change_threshold=3.0,
        madp_threshold=20.0,
        hysteresis_margin=1.0,
    )
    res = stabilizer.process_packet(soil_moisture_1=22.45, soil_moisture_2=28.12)
    assert isinstance(res, StabilizedReading)
    assert res.raw_1 == 22.45
    assert res.raw_2 == 28.12
    assert res.filtered_1 == 22.45  # first reading
    assert res.filtered_2 == 28.12
    assert res.stabilized_avg == round((22.45 + 28.12) / 2.0, 3)
    assert not res.irrigation_decision


def test_stabilizer_separate_probes():
    stabilizer = SoilMoistureStabilizer(median_window=3)
    # Probe 1 is steady at 15.0, Probe 2 has an isolated spike to 100.0
    _ = stabilizer.process_packet(15.0, 25.0)
    _ = stabilizer.process_packet(15.0, 25.0)
    res_spike = stabilizer.process_packet(15.0, 100.0)
    _ = stabilizer.process_packet(15.0, 25.0)

    # Probe 1 stays 15.0, Probe 2 suppresses 100.0 spike
    assert res_spike.raw_1 == 15.0
    assert res_spike.raw_2 == 100.0  # Raw preserved
    assert res_spike.filtered_1 == 15.0
    assert res_spike.filtered_2 == 25.0  # Filtered removes spike!


def test_stabilizer_on_scenario_sensor_noise():
    """Verify on real synthetic noise scenario that noise spikes do not cause false alarms."""
    csv_path = SCENARIOS_DIR / "scenario_02_sensor_noise.csv"
    if not csv_path.exists():
        pytest.skip(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    stabilizer = SoilMoistureStabilizer(
        median_window=5,
        ewma_alpha=0.3,
        sudden_change_threshold=3.0,
        madp_threshold=20.0,
        hysteresis_margin=1.0,
    )

    results = stabilizer.process_sequence(
        sm1_series=df["soil_moisture_1"].tolist(),
        sm2_series=df["soil_moisture_2"].tolist(),
    )
    assert len(results) == len(df)

    # In scenario 02, index 18 has soil_moisture_1=100.0, soil_moisture_2=100.0 (sensor fault)
    # and index 42 has 0.0, 0.0.
    res_18 = results[18]
    assert res_18.raw_1 == 100.0
    assert res_18.raw_2 == 100.0
    # Filtered value must be bounded near normal baseline (~24-28%), NOT 100%
    assert res_18.filtered_1 < 30.0
    assert res_18.filtered_2 < 32.0


def test_stabilizer_on_scenario_irrigation_event():
    """Verify on scenario 04 that genuine sudden change during irrigation is preserved."""
    csv_path = SCENARIOS_DIR / "scenario_04_irrigation_event.csv"
    if not csv_path.exists():
        pytest.skip(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    stabilizer = SoilMoistureStabilizer(
        median_window=3,
        ewma_alpha=0.3,
        sudden_change_threshold=2.0,  # Irrigation rise rate
        madp_threshold=20.0,
        hysteresis_margin=1.0,
    )

    results = stabilizer.process_sequence(
        sm1_series=df["soil_moisture_1"].tolist(),
        sm2_series=df["soil_moisture_2"].tolist(),
    )

    # Early steps: moisture ~13.5% < 19.0% -> Irrigation should be ON
    assert results[0].irrigation_decision is True

    # Check that sudden change was detected during active wetting (indices 8-12)
    sudden_detected = any(r.is_sudden_change_1 or r.is_sudden_change_2 for r in results[8:15])
    assert sudden_detected is True

    # Late steps: moisture rises to ~29% > 21.0% -> Irrigation should turn OFF
    assert results[-1].irrigation_decision is False


def test_advisory_separated_from_actuator_hysteresis():
    """
    When stabilized soil moisture falls below MADP (e.g. 19.5% < 20.0%) but remains
    above hysteresis turn-on (19.0%), irrigation advisory MUST report IRRIGATION_STRESS
    independently of the actuator staying OFF.
    """
    stabilizer = SoilMoistureStabilizer(
        median_window=3,
        ewma_alpha=1.0,  # immediate tracking for exact unit control
        madp_threshold=20.0,
        hysteresis_margin=1.0,
    )

    # 1. Above MADP -> NO_STRESS, actuator OFF
    res_ok = stabilizer.process_packet(22.0, 22.0)
    assert res_ok.irrigation_advisory is False
    assert res_ok.stress_status == "NO_STRESS"
    assert res_ok.irrigation_decision is False

    # 2. Below MADP (19.5% < 20.0%) but above hysteresis on (19.0%)
    # Feed enough identical readings so median window settles at 19.5
    _ = stabilizer.process_packet(19.5, 19.5)
    res_stress = stabilizer.process_packet(19.5, 19.5)
    assert res_stress.stabilized_avg == 19.5
    assert res_stress.irrigation_advisory is True
    assert res_stress.stress_status == "IRRIGATION_STRESS"
    # Hysteresis keeps actuator OFF because 19.5 > 19.0
    assert res_stress.irrigation_decision is False
    assert "NO_STRESS" not in res_stress.state_description


def test_scenario_03_gradual_drying_reports_stress_below_madp():
    """
    Verify on Scenario 3 (Gradual Soil Drying) that when stabilized average
    falls below MADP (20.0%), irrigation_advisory triggers True and stress_status
    is 'IRRIGATION_STRESS', even if actuator hysteresis keeps valve OFF.
    """
    csv_path = SCENARIOS_DIR / "scenario_03_gradual_drying.csv"
    if not csv_path.exists():
        pytest.skip(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    stabilizer = SoilMoistureStabilizer(
        median_window=5,
        ewma_alpha=0.3,
        sudden_change_threshold=3.0,
        madp_threshold=20.0,
        hysteresis_margin=1.0,
    )

    results = stabilizer.process_sequence(
        sm1_series=df["soil_moisture_1"].tolist(),
        sm2_series=df["soil_moisture_2"].tolist(),
    )
    assert len(results) == 48

    # Check invariant across all 48 packets: never report NO_STRESS below MADP
    for r in results:
        if r.stabilized_avg < 20.0:
            assert r.irrigation_advisory is True
            assert r.stress_status == "IRRIGATION_STRESS"
        else:
            assert r.irrigation_advisory is False
            assert r.stress_status == "NO_STRESS"

    # At packet 47 (last packet), stabilized_avg (~19.75%) < 20.0% MADP
    last = results[-1]
    assert last.stabilized_avg < 20.0
    assert last.irrigation_advisory is True
    assert last.stress_status == "IRRIGATION_STRESS"
    # Actuator remains OFF because 19.75% > 19.0% hysteresis lower bound
    assert last.irrigation_decision is False


def test_sensor_spikes_do_not_trigger_false_irrigation_advice():
    """
    Verify that isolated 0% ground dropout spikes in Scenario 2 and synthetic sequences
    do NOT trigger false irrigation_advisory or IRRIGATION_STRESS alerts.
    """
    csv_path = SCENARIOS_DIR / "scenario_02_sensor_noise.csv"
    if not csv_path.exists():
        pytest.skip(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    stabilizer = SoilMoistureStabilizer(
        median_window=5,
        ewma_alpha=0.3,
        sudden_change_threshold=3.0,
        madp_threshold=20.0,
        hysteresis_margin=1.0,
    )

    results = stabilizer.process_sequence(
        sm1_series=df["soil_moisture_1"].tolist(),
        sm2_series=df["soil_moisture_2"].tolist(),
    )

    # Index 42 is a 0.0% / 0.0% ground dropout spike
    res_dropout = results[42]
    assert res_dropout.raw_1 == 0.0
    assert res_dropout.raw_2 == 0.0
    assert res_dropout.stabilized_avg > 20.0
    assert res_dropout.irrigation_advisory is False
    assert res_dropout.stress_status == "NO_STRESS"
    assert res_dropout.irrigation_decision is False

    # Across the entire Scenario 2 baseline (~24-28%), zero false advisories should occur
    assert all(r.irrigation_advisory is False for r in results)
    assert all(r.stress_status == "NO_STRESS" for r in results)
