"""
Pydantic schemas for the unified `POST /predict/agricultural` endpoint.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator

from app.schemas.prediction import IrrigationAdvice

ModelReadinessStatus = Literal["ready", "insufficient_history", "blocked"]
ConfidenceLevel = Literal["low", "medium", "high", "unavailable"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class AgriculturalPredictRequest(BaseModel):
    """
    Unified inbound NEST telemetry sequence request.
    Supports variable history lengths (1 to 2000 packets) and evaluates each
    agricultural model independently according to its minimum required window:
      - Irrigation / stability : 48 consecutive 30-min packets (24 hours)
      - Fungal risk            : 336 consecutive 30-min packets (7 days)
      - Pest risk              : 1,344 consecutive 30-min packets (4 weeks)
    """
    packets: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Chronologically ordered 30-minute NEST packets.",
    )
    irrigation_model: Literal["xgb", "gru", "xgb_residual", "gru_residual"] = Field(
        default="xgb_residual",
        description="Trained soil-moisture model variant to run when >= 48 packets are available.",
    )

    @model_validator(mode="after")
    def validate_single_device(self) -> "AgriculturalPredictRequest":
        device_ids = {str(p.get("device_id")) for p in self.packets if p.get("device_id")}
        if len(device_ids) > 1:
            raise ValueError(f"All packets must share the same device_id; got: {device_ids}")
        return self


class IrrigationStabilityPrediction(BaseModel):
    """Independent 24h irrigation forecast & stabilized soil-moisture status (requires 48 packets)."""
    status: ModelReadinessStatus
    ready: bool
    packets_required: int = 48
    packets_provided: int
    model_used: str | None = None
    confidence: float | None = Field(default=None, description="Numeric confidence score (0-1)")
    confidence_level: ConfidenceLevel = "unavailable"
    raw_soil_moisture: float | None = None
    stabilized_soil_moisture: float | None = None
    predicted_soil_moisture_24h: float | None = None
    change_predicted: float | None = None
    irrigation_advisory: bool | None = Field(default=None, description="True if stabilized soil moisture < MADP")
    stress_status: Literal["IRRIGATION_STRESS", "NO_STRESS"] | None = None
    actuator_decision: bool | None = Field(default=None, description="Hysteresis valve state (True = Valve ON)")
    sudden_change_detected: bool | None = None
    irrigation_advice: IrrigationAdvice | None = None
    warnings: list[str] = Field(default_factory=list)
    unavailable_features: list[str] = Field(default_factory=list)


class RiskClassificationPrediction(BaseModel):
    """Independent LOW / MEDIUM / HIGH biological risk prediction (fungal=336 pkts, pest=1344 pkts)."""
    status: ModelReadinessStatus
    ready: bool
    packets_required: int
    packets_provided: int
    model_used: str | None = None
    risk_level: RiskLevel | None = None
    probabilities: dict[RiskLevel, float] | None = None
    confidence: float | None = Field(default=None, description="Max predicted class probability (0-1)")
    confidence_level: ConfidenceLevel = "unavailable"
    warnings: list[str] = Field(default_factory=list)
    domain_limitations: list[str] = Field(default_factory=list)
    window_start_timestamp: str | None = None
    window_end_timestamp: str | None = None


class AgriculturalPredictResponse(BaseModel):
    """Unified multi-horizon response returning irrigation, fungal, and pest predictions independently."""
    status: Literal["success"] = "success"
    device_id: str
    farm_id: str
    crop_id: str
    total_packets_received: int
    irrigation: IrrigationStabilityPrediction
    fungal_risk: RiskClassificationPrediction
    pest_risk: RiskClassificationPrediction
