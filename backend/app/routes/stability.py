"""
FastAPI route for Soil Moisture Signal Conditioning and Stability Analysis.

Endpoints:
- POST /stability/analyze:
  Accepts 48 genuine, consecutive 30-minute NEST packets.
  Applies SoilMoistureStabilizer (Median filter + EWMA + Sudden-change bypass + Hysteresis switch)
  separately for both moisture sensors.
  Returns raw and filtered telemetry, sudden change flags, actuator decisions,
  and quantitative noise/chatter reduction metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.ml.stability import SoilMoistureStabilizer
from app.schemas.stability import (
    ScenarioDetailResponse,
    ScenarioListResponse,
    ScenarioSummary,
    StabilityAnalyzeRequest,
    StabilityAnalyzeResponse,
    StabilizedPacketRecord,
    StabilityMetricsSummary,
)
from app.services.inference import (
    CROP_THRESHOLDS,
    InsufficientHistoryError,
    PredictionBlockedError,
    validate_sequence_integrity,
)
from app.services.stability_metrics import compute_range_and_outlier_metrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stability", tags=["Stability"])

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
_SYNTHETIC_DIR = _ROOT_DIR / "data" / "synthetic_nest"

# Predefined whitelist of allowed synthetic scenarios
PREDEFINED_SCENARIOS = {
    "scenario_01_normal": {
        "file": "scenario_01_normal.csv",
        "name": "Scenario 1: Normal Diurnal Cycle",
        "description": "Baseline 48-packet diurnal cycle with typical soil drying, solar radiation curve, and normal sensor noise.",
    },
    "scenario_02_sensor_noise": {
        "file": "scenario_02_sensor_noise.csv",
        "name": "Scenario 2: Sensor Noise & Spikes",
        "description": "Simulated hardware faults with isolated 16-bit ADC overflow spikes (100% moisture) and ground dropouts (0% moisture).",
    },
    "scenario_03_gradual_drying": {
        "file": "scenario_03_gradual_drying.csv",
        "name": "Scenario 3: Gradual Soil Drying",
        "description": "Extended sunny period with high vapor pressure deficit causing soil moisture to steadily cross below the MADP threshold.",
    },
    "scenario_04_irrigation_event": {
        "file": "scenario_04_irrigation_event.csv",
        "name": "Scenario 4: Irrigation Infiltration",
        "description": "Active irrigation wetting front step response causing rapid moisture replenishment from 13.5% to 29.8%.",
    },
    "scenario_05_high_humidity": {
        "file": "scenario_05_high_humidity.csv",
        "name": "Scenario 5: High Humidity & Leaf Wetness",
        "description": "Extended monsoon/overcast conditions with >90% relative humidity, persistent canopy leaf wetness, and minimal soil drying.",
    },
}


def _total_variation(signal: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(signal))))


def _high_freq_energy(signal: np.ndarray) -> float:
    if len(signal) < 3:
        return 0.0
    return float(np.var(np.diff(signal, n=2)))


def _count_switches(binary_series: list[int] | np.ndarray) -> int:
    arr = np.array(binary_series, dtype=int)
    return int(np.sum(np.abs(np.diff(arr))))


@router.post(
    "/analyze",
    response_model=StabilityAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate raw vs filtered dual-probe soil moisture and actuator stability",
)
def analyze_stability(req: StabilityAnalyzeRequest) -> StabilityAnalyzeResponse | JSONResponse:
    """
    Accept 48 genuine consecutive NEST packets (30 min apart, 24-hour lookback).
    Stabilizes Probe 1 and Probe 2 independently through Median Filtering and EWMA,
    preserves genuine wetting front sudden changes, and evaluates actuator hysteresis.

    Rejects incomplete, gapped, or duplicated sequences with insufficient_history.
    """
    pkt_dicts = [p.model_dump() for p in req.packets]

    # 1. Enforce strict sequence integrity (48 genuine consecutive packets)
    try:
        validate_sequence_integrity(pkt_dicts, required_length=48)
    except InsufficientHistoryError as exc:
        logger.warning("Stability analysis rejected: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "rejected",
                "error": "insufficient_history",
                "detail": str(exc),
                "packets_provided": exc.packets_provided,
                "packets_required": exc.packets_required,
                "resolution": (
                    "Provide a complete sequence of 48 genuine, consecutive NEST packets "
                    "(30-minute cadence, 24 hours of telemetry) without duplicate readings or temporal gaps."
                ),
            },
        )
    except PredictionBlockedError as exc:
        logger.warning("Stability analysis blocked: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sequence validation blocked: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Sequence validation error in stability analysis")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation error: {exc}",
        ) from exc

    # 2. Extract telemetry and crop context
    last_pkt = req.packets[-1]
    crop_id = last_pkt.crop_id
    thresh = CROP_THRESHOLDS.get(crop_id, CROP_THRESHOLDS["chilli"])
    madp = thresh["madp"]

    raw_sm1 = np.array([float(p.soil_moisture_1) for p in req.packets], dtype=float)
    raw_sm2 = np.array([float(p.soil_moisture_2) for p in req.packets], dtype=float)
    raw_avg = (raw_sm1 + raw_sm2) / 2.0
    raw_irr = (raw_avg < madp).astype(int)

    # 3. Run SoilMoistureStabilizer
    stabilizer = SoilMoistureStabilizer(
        median_window=5,
        ewma_alpha=0.3,
        sudden_change_threshold=3.0,
        madp_threshold=madp,
        hysteresis_margin=1.0,
    )

    stabilized_readings = stabilizer.process_sequence(raw_sm1, raw_sm2)

    timeline: list[StabilizedPacketRecord] = []
    filt_sm1_list = []
    filt_sm2_list = []
    filt_avg_list = []
    filt_irr_list = []
    sudden_flags = []

    for idx, (p, r) in enumerate(zip(req.packets, stabilized_readings)):
        filt_sm1_list.append(r.filtered_1)
        filt_sm2_list.append(r.filtered_2)
        filt_avg_list.append(r.stabilized_avg)
        filt_irr_list.append(int(r.irrigation_decision))
        is_sudden = r.is_sudden_change_1 or r.is_sudden_change_2
        sudden_flags.append(is_sudden)

        timeline.append(
            StabilizedPacketRecord(
                timestamp=p.timestamp,
                packet_index=idx,
                raw_soil_moisture_1=r.raw_1,
                raw_soil_moisture_2=r.raw_2,
                raw_average=round((r.raw_1 + r.raw_2) / 2.0, 3),
                filtered_soil_moisture_1=r.filtered_1,
                filtered_soil_moisture_2=r.filtered_2,
                stabilized_average=r.stabilized_avg,
                is_sudden_change_probe_1=r.is_sudden_change_1,
                is_sudden_change_probe_2=r.is_sudden_change_2,
                irrigation_decision=r.irrigation_decision,
                irrigation_advisory=r.irrigation_advisory,
                stress_status=r.stress_status,
                state_description=r.state_description,
            )
        )

    filt_sm1 = np.array(filt_sm1_list)
    filt_sm2 = np.array(filt_sm2_list)
    filt_avg = np.array(filt_avg_list)

    # 4. Compute Stability Metrics
    tv_raw = _total_variation(raw_avg)
    tv_filt = _total_variation(filt_avg)
    tv_red_pct = ((tv_raw - tv_filt) / max(tv_raw, 1e-6)) * 100.0

    hf_raw = _high_freq_energy(raw_avg)
    hf_filt = _high_freq_energy(filt_avg)
    hf_red_pct = ((hf_raw - hf_filt) / max(hf_raw, 1e-6)) * 100.0

    p1_tv_red = ((_total_variation(raw_sm1) - _total_variation(filt_sm1)) / max(_total_variation(raw_sm1), 1e-6)) * 100.0
    p2_tv_red = ((_total_variation(raw_sm2) - _total_variation(filt_sm2)) / max(_total_variation(raw_sm2), 1e-6)) * 100.0

    fault_mask = np.array(
        [bool(p_dict.get("gt_sensor_fault", 0)) for p_dict in pkt_dicts],
        dtype=bool,
    )
    range_metrics = compute_range_and_outlier_metrics(
        raw_avg=raw_avg,
        filt_avg=filt_avg,
        fault_mask=fault_mask if np.any(fault_mask) else None,
    )

    raw_switches = _count_switches(raw_irr)
    filt_switches = _count_switches(filt_irr_list)
    chatter_red_pct = ((raw_switches - filt_switches) / max(raw_switches, 1)) * 100.0 if raw_switches > 0 else 0.0

    metrics = StabilityMetricsSummary(
        total_variation_reduction_pct=round(tv_red_pct, 2),
        high_frequency_jitter_reduction_pct=round(hf_red_pct, 2),
        probe1_noise_reduction_pct=round(p1_tv_red, 2),
        probe2_noise_reduction_pct=round(p2_tv_red, 2),
        dynamic_range_retention_pct=range_metrics["genuine_range_retention_pct"],
        clean_baseline_range=range_metrics["clean_baseline_range"],
        filtered_dynamic_range=range_metrics["filtered_dynamic_range"],
        outlier_suppression_pct=range_metrics["outlier_suppression_pct"],
        outliers_detected_count=range_metrics["outliers_detected_count"],
        sudden_events_detected_count=int(np.sum(sudden_flags)),
        raw_switch_count=raw_switches,
        filtered_switch_count=filt_switches,
        chatter_reduction_pct=round(chatter_red_pct, 2),
    )

    return StabilityAnalyzeResponse(
        status="success",
        device_id=last_pkt.device_id,
        farm_id=last_pkt.farm_id,
        crop_id=crop_id,
        madp_threshold=madp,
        packets_analyzed=len(req.packets),
        timeline=timeline,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# GET /stability/scenarios
# ---------------------------------------------------------------------------
@router.get(
    "/scenarios",
    response_model=ScenarioListResponse,
    status_code=status.HTTP_200_OK,
    summary="List available synthetic NEST telemetry scenarios",
)
def list_scenarios() -> ScenarioListResponse:
    """
    Return the 5 predefined synthetic NEST scenarios with their metadata.
    All data is strictly marked as SYNTHETIC.
    """
    scenarios_list = [
        ScenarioSummary(
            id=k,
            name=v["name"],
            description=v["description"],
            packet_count=48,
            data_source="SYNTHETIC",
        )
        for k, v in PREDEFINED_SCENARIOS.items()
    ]
    return ScenarioListResponse(
        scenarios=scenarios_list,
        data_source="SYNTHETIC",
        note="All scenario telemetry is SYNTHETIC generated data based on company NEST reference schema.",
    )


# ---------------------------------------------------------------------------
# GET /stability/scenarios/{scenario_id}
# ---------------------------------------------------------------------------
@router.get(
    "/scenarios/{scenario_id}",
    response_model=ScenarioDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve 48 synthetic packets for a predefined scenario",
)
def get_scenario_packets(scenario_id: str) -> ScenarioDetailResponse:
    """
    Retrieve exactly 48 genuine consecutive packets for one of the 5 predefined
    synthetic scenarios. Only predefined scenario IDs are permitted.
    All returned data is marked SYNTHETIC.
    """
    # Whitelist validation: only allow predefined scenarios
    if scenario_id not in PREDEFINED_SCENARIOS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Scenario '{scenario_id}' not found. "
                f"Permitted scenarios: {list(PREDEFINED_SCENARIOS.keys())}"
            ),
        )

    sc_meta = PREDEFINED_SCENARIOS[scenario_id]
    csv_file = _SYNTHETIC_DIR / sc_meta["file"]

    if not csv_file.exists():
        logger.error("Predefined scenario file missing: %s", csv_file)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scenario data file '{sc_meta['file']}' is missing from server storage.",
        )

    df = pd.read_csv(csv_file)
    packets = df.to_dict(orient="records")[:48]

    # Explicitly enforce SYNTHETIC attribution on each packet
    for p in packets:
        p["data_source"] = "SYNTHETIC"

    return ScenarioDetailResponse(
        id=scenario_id,
        name=sc_meta["name"],
        description=sc_meta["description"],
        data_source="SYNTHETIC",
        packet_count=len(packets),
        packets=packets,
    )
