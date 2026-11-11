"""
Prediction endpoints for the NEST Agricultural Prediction System.

Routes
------
POST /predict/quick
    Single NEST packet → immediate soil-moisture forecast + irrigation advice.
    Useful for real-time device callbacks.

POST /predict/batch
    1–48 ordered NEST packets → sequence-based forecast.
    Uses the full 48-step GRU/XGBoost window.

GET  /predict/models
    Returns metadata about all loaded model variants.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas.agricultural import (
    AgriculturalPredictRequest,
    AgriculturalPredictResponse,
)
from app.schemas.prediction import (
    BatchPredictRequest,
    IrrigationAdvice,
    ModelInfoResponse,
    PredictionResponse,
    QuickPredictRequest,
    SinglePacketValidationResponse,
)
from app.services.agricultural import evaluate_agricultural_predictions
from app.services.inference import (
    CROP_THRESHOLDS,
    InsufficientHistoryError,
    ModelRegistry,
    PredictionBlockedError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])


# ---------------------------------------------------------------------------
# Shared helper: build IrrigationAdvice from moisture and crop thresholds
# ---------------------------------------------------------------------------
def _make_irrigation_advice(
    avg_sm: float,
    crop_id: str,
) -> IrrigationAdvice:
    thresh = CROP_THRESHOLDS.get(crop_id, CROP_THRESHOLDS["chilli"])
    madp = thresh["madp"]
    fc   = thresh["fc"]
    soil_stress = avg_sm < madp
    deficit     = max(0.0, madp - avg_sm)
    return IrrigationAdvice(
        irrigate=soil_stress,
        soil_stress=soil_stress,
        avg_soil_moisture=round(avg_sm, 3),
        madp_threshold=madp,
        fc_threshold=fc,
        moisture_deficit=round(deficit, 3),
    )


# Public export alias
make_irrigation_advice = _make_irrigation_advice


# ---------------------------------------------------------------------------
# Helper: assess prediction confidence based on packet count
# ---------------------------------------------------------------------------
def _confidence(n_packets: int) -> str:
    if n_packets >= 36:
        return "high"
    if n_packets >= 12:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# GET /predict/models
# ---------------------------------------------------------------------------
@router.get(
    "/models",
    response_model=ModelInfoResponse,
    summary="List loaded model variants and their metadata",
)
def get_model_info() -> ModelInfoResponse:
    """Return metadata for all loaded prediction models."""
    registry = ModelRegistry.get()
    info = registry.model_info()
    return ModelInfoResponse(**info)


# ---------------------------------------------------------------------------
# POST /predict/quick (Single Packet: Input Validation and Threshold Check ONLY)
# ---------------------------------------------------------------------------
@router.post(
    "/quick",
    response_model=SinglePacketValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate single NEST packet and check immediate agronomic state",
)
def predict_quick(req: QuickPredictRequest) -> SinglePacketValidationResponse | JSONResponse:
    """
    Validate a single inbound NEST IoT telemetry packet.

    **Note:**
    Single packets are validated for schema, sensor physical ranges, unit conversions,
    and immediate crop threshold status. Temporal 24-hour predictions are NOT performed
    on single packets to prevent artificial padding and flatlining; 48 genuine consecutive
    packets must be sent via `/predict/batch`.
    """
    registry = ModelRegistry.get()
    pkt_dict = req.packet.model_dump()

    try:
        val_result = registry.validate_single_packet(pkt_dict)
    except PredictionBlockedError as exc:
        logger.warning("Single packet validation blocked: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation failed: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Single packet validation encountered unexpected error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation error: {exc}",
        ) from exc

    crop_id = val_result["crop_id"]
    current_sm = val_result["current_soil_moisture"]
    irrigation_advice = _make_irrigation_advice(current_sm, crop_id)

    return SinglePacketValidationResponse(
        status="valid",
        device_id=val_result["device_id"],
        farm_id=val_result["farm_id"],
        crop_id=crop_id,
        current_soil_moisture=current_sm,
        pressure_hpa=val_result["pressure_hpa"],
        irrigation_advice=irrigation_advice,
        warnings=val_result["warnings"],
        unavailable_features=val_result["unavailable_features"],
        temporal_prediction="unavailable_requires_48_packets",
        message=val_result["message"],
    )


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------
@router.post(
    "/batch",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Sequence-based soil moisture forecast (48 consecutive 30-min packets)",
)
def predict_batch(req: BatchPredictRequest) -> PredictionResponse | JSONResponse:
    """
    Accept exactly 48 genuine chronologically ordered NEST packets (30 min apart,
    24h lookback) and return a 24-hour soil-moisture forecast.

    Automatic padding is disabled. Incomplete sequences or duplicate readings
    are rejected with an insufficient_history response.
    """
    registry = ModelRegistry.get()

    if req.model not in registry.available_models:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Model '{req.model}' is not available. "
                   f"Choose from: {registry.available_models}",
        )

    pkt_dicts = [p.model_dump() for p in req.packets]
    try:
        result = registry.predict(pkt_dicts, model_key=req.model)
    except InsufficientHistoryError as exc:
        logger.warning("Batch predict rejected due to insufficient history: %s", exc)
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
                    "(30-minute cadence, 24 hours of telemetry) without duplicate readings or temporal gaps via /predict/batch."
                ),
            },
        )
    except PredictionBlockedError as exc:
        logger.warning("Batch predict blocked: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction blocked: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Inference failed for batch predict")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc

    crop_id  = req.packets[-1].crop_id
    pred_sm  = result["predicted_soil_moisture"]
    irrigation_advice = _make_irrigation_advice(pred_sm, crop_id)

    return PredictionResponse(
        device_id=req.packets[-1].device_id,
        farm_id=req.packets[-1].farm_id,
        crop_id=crop_id,
        model_used=req.model,
        predicted_soil_moisture=pred_sm,
        current_soil_moisture=result["current_soil_moisture"],
        change_predicted=result["change_predicted"],
        irrigation_advice=irrigation_advice,
        confidence="high",
        data_source="LIVE",
        warnings=result["warnings"],
        unavailable_features=result.get("unavailable_features", []),
    )


@router.post(
    "/agricultural",
    response_model=AgriculturalPredictResponse,
    summary="Unified agricultural inference (irrigation/stability, fungal risk, pest risk)",
)
def predict_agricultural(req: AgriculturalPredictRequest) -> AgriculturalPredictResponse:
    """Evaluate irrigation/stability (48), fungal risk (336), and pest risk (1,344) independently."""
    return evaluate_agricultural_predictions(
        packets=req.packets,
        irrigation_model=req.irrigation_model,
    )
