"""
Unified Agricultural Inference Orchestrator
===========================================
Evaluates Irrigation/Stability (48 packets), Fungal Risk (336 packets), and
Pest Risk (1,344 packets) independently using existing feature mappers and
saved models without modifying models or fabricating missing history.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.ml.nest_fungal_mapper import (
    REQUIRED_PACKETS_FUNGAL,
    map_nest_packets_to_fungal_features,
)
from app.ml.nest_pest_mapper import (
    REQUIRED_PACKETS_PEST,
    map_nest_packets_to_pest_features,
)
from app.ml.stability import SoilMoistureStabilizer
from app.schemas.agricultural import (
    AgriculturalPredictResponse,
    ConfidenceLevel,
    IrrigationStabilityPrediction,
    RiskClassificationPrediction,
)
from app.schemas.prediction import IrrigationAdvice
from app.services.inference import (
    CROP_THRESHOLDS,
    InsufficientHistoryError,
    ModelRegistry,
    PredictionBlockedError,
)

_ML_DIR = Path(__file__).resolve().parent.parent / "ml"
_FUNGAL_MODEL_PATH = _ML_DIR / "models_fungal_risk" / "best_fungal_model.joblib"
_PEST_MODEL_PATH = _ML_DIR / "models_pest_risk" / "best_pest_model.joblib"

REQUIRED_PACKETS_IRRIGATION = 48


@lru_cache(maxsize=1)
def _load_fungal_artifact() -> dict[str, Any]:
    if not _FUNGAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved fungal model not found: {_FUNGAL_MODEL_PATH}")
    return joblib.load(_FUNGAL_MODEL_PATH)


@lru_cache(maxsize=1)
def _load_pest_artifact() -> dict[str, Any]:
    if not _PEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved pest model not found: {_PEST_MODEL_PATH}")
    return joblib.load(_PEST_MODEL_PATH)


def _prob_to_confidence_level(max_prob: float) -> ConfidenceLevel:
    if max_prob >= 0.70:
        return "high"
    if max_prob >= 0.50:
        return "medium"
    return "low"


def _evaluate_irrigation(
    packets: list[dict[str, Any]],
    crop_id: str,
    model_key: str,
) -> IrrigationStabilityPrediction:
    n_total = len(packets)
    if n_total < REQUIRED_PACKETS_IRRIGATION:
        return IrrigationStabilityPrediction(
            status="insufficient_history",
            ready=False,
            packets_required=REQUIRED_PACKETS_IRRIGATION,
            packets_provided=n_total,
            confidence=None,
            confidence_level="unavailable",
            warnings=[
                f"Insufficient history for irrigation/stability forecast: requires "
                f"{REQUIRED_PACKETS_IRRIGATION} consecutive 30-min packets (24h), received {n_total}. "
                f"Padding and fabrication are disabled."
            ],
        )

    window_48 = packets[-REQUIRED_PACKETS_IRRIGATION:]
    registry = ModelRegistry.get()
    if model_key not in registry.available_models:
        return IrrigationStabilityPrediction(
            status="blocked",
            ready=False,
            packets_required=REQUIRED_PACKETS_IRRIGATION,
            packets_provided=REQUIRED_PACKETS_IRRIGATION,
            confidence=None,
            confidence_level="unavailable",
            warnings=[f"Requested irrigation model '{model_key}' is not loaded."],
        )

    try:
        forecast = registry.predict(window_48, model_key=model_key)
    except InsufficientHistoryError as exc:
        return IrrigationStabilityPrediction(
            status="insufficient_history",
            ready=False,
            packets_required=REQUIRED_PACKETS_IRRIGATION,
            packets_provided=REQUIRED_PACKETS_IRRIGATION,
            confidence=None,
            confidence_level="unavailable",
            warnings=[str(exc)],
        )
    except PredictionBlockedError as exc:
        return IrrigationStabilityPrediction(
            status="blocked",
            ready=False,
            packets_required=REQUIRED_PACKETS_IRRIGATION,
            packets_provided=REQUIRED_PACKETS_IRRIGATION,
            confidence=None,
            confidence_level="unavailable",
            warnings=[str(exc)],
        )

    # Compute dual-probe stabilization + decoupled advisory/hysteresis on the same 48-packet window
    thresh = CROP_THRESHOLDS.get(crop_id, CROP_THRESHOLDS["chilli"])
    madp = float(thresh["madp"])
    fc = float(thresh["fc"])

    stabilizer = SoilMoistureStabilizer(
        median_window=5,
        ewma_alpha=0.3,
        sudden_change_threshold=3.0,
        madp_threshold=madp,
        hysteresis_margin=1.0,
    )
    raw_sm1 = [float(p["soil_moisture_1"]) for p in window_48]
    raw_sm2 = [float(p["soil_moisture_2"]) for p in window_48]
    stab_records = stabilizer.process_sequence(raw_sm1, raw_sm2)
    last_stab = stab_records[-1]

    pred_sm = float(forecast["predicted_soil_moisture"])
    soil_stress = pred_sm < madp
    deficit = max(0.0, madp - pred_sm)
    advice = IrrigationAdvice(
        irrigate=soil_stress,
        soil_stress=soil_stress,
        avg_soil_moisture=round(pred_sm, 3),
        madp_threshold=madp,
        fc_threshold=fc,
        moisture_deficit=round(deficit, 3),
    )

    reliability = forecast.get("reliability", "high")
    conf_level: ConfidenceLevel = "high" if reliability == "high" else "medium"
    conf_score = 0.90 if reliability == "high" else 0.70

    return IrrigationStabilityPrediction(
        status="ready",
        ready=True,
        packets_required=REQUIRED_PACKETS_IRRIGATION,
        packets_provided=REQUIRED_PACKETS_IRRIGATION,
        model_used=model_key,
        confidence=conf_score,
        confidence_level=conf_level,
        raw_soil_moisture=round((last_stab.raw_1 + last_stab.raw_2) / 2.0, 3),
        stabilized_soil_moisture=last_stab.stabilized_avg,
        predicted_soil_moisture_24h=pred_sm,
        change_predicted=float(forecast["change_predicted"]),
        irrigation_advisory=last_stab.irrigation_advisory,
        stress_status=last_stab.stress_status,  # type: ignore[arg-type]
        actuator_decision=last_stab.irrigation_decision,
        sudden_change_detected=last_stab.is_sudden_change_1 or last_stab.is_sudden_change_2,
        irrigation_advice=advice,
        warnings=list(forecast.get("warnings", [])),
        unavailable_features=list(forecast.get("unavailable_features", [])),
    )


def _evaluate_fungal(
    packets: list[dict[str, Any]],
    crop_id: str,
) -> RiskClassificationPrediction:
    n_total = len(packets)
    if n_total < REQUIRED_PACKETS_FUNGAL:
        return RiskClassificationPrediction(
            status="insufficient_history",
            ready=False,
            packets_required=REQUIRED_PACKETS_FUNGAL,
            packets_provided=n_total,
            confidence=None,
            confidence_level="unavailable",
            warnings=[
                f"Insufficient history for fungal-risk prediction: requires "
                f"{REQUIRED_PACKETS_FUNGAL} consecutive 30-min packets (7 days), received {n_total}. "
                f"History fabrication is disabled."
            ],
        )

    window_336 = packets[-REQUIRED_PACKETS_FUNGAL:]
    try:
        vec = map_nest_packets_to_fungal_features(window_336)
    except InsufficientHistoryError as exc:
        return RiskClassificationPrediction(
            status="insufficient_history",
            ready=False,
            packets_required=REQUIRED_PACKETS_FUNGAL,
            packets_provided=REQUIRED_PACKETS_FUNGAL,
            confidence=None,
            confidence_level="unavailable",
            warnings=[str(exc)],
        )
    except PredictionBlockedError as exc:
        return RiskClassificationPrediction(
            status="blocked",
            ready=False,
            packets_required=REQUIRED_PACKETS_FUNGAL,
            packets_provided=REQUIRED_PACKETS_FUNGAL,
            confidence=None,
            confidence_level="unavailable",
            warnings=[str(exc)],
        )

    artifact = _load_fungal_artifact()
    estimator = artifact["estimator"]
    class_names = artifact["class_names"]

    pred_idx = int(estimator.predict(vec.feature_array)[0])
    probs_arr = estimator.predict_proba(vec.feature_array)[0]
    prob_map = {
        str(cls_name): round(float(p), 4)
        for cls_name, p in zip(class_names, probs_arr)
    }
    max_prob = float(np.max(probs_arr))

    warnings: list[str] = []
    domain_limitations: list[str] = []

    if crop_id.lower() != "flax":
        limitation = (
            f"DOMAIN_LIMITATION (Flax Rust -> {crop_id}): Fungal-risk model ({artifact['model_name']}) "
            f"was trained on Colorado subalpine wild flax rust (Melampsora lini, 10.2–18.9°C), NOT "
            f"{crop_id} fungal pathogens. Cross-crop and climate thresholds are not calibrated for {crop_id}."
        )
        domain_limitations.append(limitation)
        warnings.append(limitation)

    temp_mean_7d = float(vec.feature_dict["temp_mean_7d"])
    if temp_mean_7d > 19.0 or float(vec.feature_dict["temp_max_7d"]) > 25.0:
        temp_warn = (
            f"OUT_OF_DISTRIBUTION_CLIMATE: 7-day mean temperature ({temp_mean_7d:.1f}°C) exceeds "
            f"flax-rust subalpine training envelope (10.2–18.9°C)."
        )
        warnings.append(temp_warn)

    return RiskClassificationPrediction(
        status="ready",
        ready=True,
        packets_required=REQUIRED_PACKETS_FUNGAL,
        packets_provided=REQUIRED_PACKETS_FUNGAL,
        model_used=f"fungal_{artifact['model_name']}",
        risk_level=class_names[pred_idx],
        probabilities=prob_map,  # type: ignore[arg-type]
        confidence=round(max_prob, 4),
        confidence_level=_prob_to_confidence_level(max_prob),
        warnings=warnings,
        domain_limitations=domain_limitations,
        window_start_timestamp=vec.window_start_timestamp,
        window_end_timestamp=vec.window_end_timestamp,
    )


def _evaluate_pest(
    packets: list[dict[str, Any]],
    crop_id: str,
) -> RiskClassificationPrediction:
    n_total = len(packets)
    if n_total < REQUIRED_PACKETS_PEST:
        return RiskClassificationPrediction(
            status="insufficient_history",
            ready=False,
            packets_required=REQUIRED_PACKETS_PEST,
            packets_provided=n_total,
            confidence=None,
            confidence_level="unavailable",
            warnings=[
                f"Insufficient history for pest-risk prediction: requires "
                f"{REQUIRED_PACKETS_PEST} consecutive 30-min packets (28 days / 4 weeks), received {n_total}. "
                f"History fabrication is disabled."
            ],
        )

    window_1344 = packets[-REQUIRED_PACKETS_PEST:]
    try:
        vec = map_nest_packets_to_pest_features(window_1344)
    except InsufficientHistoryError as exc:
        return RiskClassificationPrediction(
            status="insufficient_history",
            ready=False,
            packets_required=REQUIRED_PACKETS_PEST,
            packets_provided=REQUIRED_PACKETS_PEST,
            confidence=None,
            confidence_level="unavailable",
            warnings=[str(exc)],
        )
    except PredictionBlockedError as exc:
        return RiskClassificationPrediction(
            status="blocked",
            ready=False,
            packets_required=REQUIRED_PACKETS_PEST,
            packets_provided=REQUIRED_PACKETS_PEST,
            confidence=None,
            confidence_level="unavailable",
            warnings=[str(exc)],
        )

    artifact = _load_pest_artifact()
    estimator = artifact["estimator"]
    class_names = artifact["class_names"]

    pred_idx = int(estimator.predict(vec.feature_array)[0])
    probs_arr = estimator.predict_proba(vec.feature_array)[0]
    prob_map = {
        str(cls_name): round(float(p), 4)
        for cls_name, p in zip(class_names, probs_arr)
    }
    max_prob = float(np.max(probs_arr))

    warnings: list[str] = []
    domain_limitations: list[str] = []

    if crop_id.lower() != "rice":
        limitation = (
            f"DOMAIN_LIMITATION (Rice RBB -> {crop_id}): Pest-risk model ({artifact['model_name']}) "
            f"was trained on weekly Rice Black Bug (RBB) trap counts in wetland rice, NOT {crop_id} "
            f"pests. Soil wetness (GWETTOP) is approximated via SM/SP ({vec.saturation_point_used}%) calibration."
        )
        domain_limitations.append(limitation)
        warnings.append(limitation)

    return RiskClassificationPrediction(
        status="ready",
        ready=True,
        packets_required=REQUIRED_PACKETS_PEST,
        packets_provided=REQUIRED_PACKETS_PEST,
        model_used=f"pest_rbb_{artifact['model_name']}",
        risk_level=class_names[pred_idx],
        probabilities=prob_map,  # type: ignore[arg-type]
        confidence=round(max_prob, 4),
        confidence_level=_prob_to_confidence_level(max_prob),
        warnings=warnings,
        domain_limitations=domain_limitations,
        window_start_timestamp=vec.window_start_timestamp,
        window_end_timestamp=vec.window_end_timestamp,
    )


def evaluate_agricultural_predictions(
    packets: list[dict[str, Any]],
    irrigation_model: str = "xgb_residual",
) -> AgriculturalPredictResponse:
    """
    Evaluate Irrigation/Stability (48 pkts), Fungal Risk (336 pkts), and Pest Risk
    (1,344 pkts) independently on the provided packet history.
    """
    last_pkt = packets[-1]
    device_id = str(last_pkt.get("device_id") or "NEST-001")
    farm_id = str(last_pkt.get("farm_id") or "farm-01")
    crop_id = str(last_pkt.get("crop_id") or "chilli")

    irrigation_res = _evaluate_irrigation(packets, crop_id=crop_id, model_key=irrigation_model)
    fungal_res = _evaluate_fungal(packets, crop_id=crop_id)
    pest_res = _evaluate_pest(packets, crop_id=crop_id)

    return AgriculturalPredictResponse(
        status="success",
        device_id=device_id,
        farm_id=farm_id,
        crop_id=crop_id,
        total_packets_received=len(packets),
        irrigation=irrigation_res,
        fungal_risk=fungal_res,
        pest_risk=pest_res,
    )
