"""
Pydantic schemas for NEST IoT payload ingestion and prediction responses.

Field names and types are derived directly from:
  data/company_sample/nest_sample.json  (21 fields — the company reference contract)

No field may be renamed or its type changed without updating the reference contract.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Supported crops (extend as new NEST devices are onboarded)
# ---------------------------------------------------------------------------
CropId = Literal[
    "chilli", "tomato", "zucchini", "blueberry", "wheat", "rice", "maize"
]


# ---------------------------------------------------------------------------
# Inbound: single NEST IoT packet (matches nest_sample.json exactly)
# ---------------------------------------------------------------------------
class NESTPacket(BaseModel):
    """Single 30-minute NEST IoT telemetry packet received from a field device."""

    # -- Identity
    device_id: str = Field(..., description="Unique NEST device identifier (e.g. NEST-001)")
    farm_id:   str = Field(..., description="Farm identifier (e.g. farm-01)")
    crop_id:   CropId = Field(..., description="Crop type grown in this zone")

    # -- Temporal (extracted from device clock; UTC assumed)
    month: int = Field(..., ge=1, le=12,  description="Month of reading (1–12)")
    day:   int = Field(..., ge=1, le=31,  description="Day of month (1–31)")
    hour:  int = Field(..., ge=0, le=23,  description="Hour of reading (0–23, device local time)")

    # -- Microclimate
    temperature:    float = Field(..., ge=-20,  le=60,   description="Air temperature (°C)")
    humidity:       float = Field(..., ge=0,    le=100,  description="Relative humidity (%)")
    pressure:       float = Field(..., ge=80,   le=110,  description="Atmospheric pressure (kPa)")
    solar_radiation:float = Field(..., ge=0,    le=1500, description="Solar radiation (W/m²)")
    wind_speed:     float = Field(..., ge=0,    le=50,   description="Wind speed (m/s)")
    wind_direction: float = Field(..., ge=0,    le=360,  description="Wind direction (degrees)")
    rainfall:       float = Field(..., ge=0,    le=500,  description="Rainfall (mm)")

    # -- Soil sensors
    soil_moisture_1: float = Field(..., ge=0, le=100, description="Soil moisture probe 1 (%)")
    soil_moisture_2: float = Field(..., ge=0, le=100, description="Soil moisture probe 2 (%)")
    soil_temperature:float = Field(..., ge=-10, le=60, description="Soil temperature (°C)")

    # -- Canopy / surface
    leaf_wetness: float = Field(..., ge=0, le=100, description="Leaf wetness index (0–100)")

    # -- Air quality
    co:    float = Field(..., ge=0, le=10,   description="Carbon monoxide (ppm)")
    no2:   float = Field(..., ge=0, le=5,    description="Nitrogen dioxide (ppm)")
    o3:    float = Field(..., ge=0, le=1,    description="Ozone (ppm)")
    pm2_5: float = Field(..., ge=0, le=1000, description="PM2.5 particulate (µg/m³)")

    # -- Optional ISO timestamp (required for temporal sequence verification)
    timestamp: str | None = Field(default=None, description="ISO-8601 timestamp of reading (e.g. 2026-06-15T14:00:00)")

    @field_validator("humidity")
    @classmethod
    def humidity_range(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError(f"humidity must be 0–100, got {v}")
        return v

    @model_validator(mode="after")
    def soil_moisture_sanity(self) -> "NESTPacket":
        avg = (self.soil_moisture_1 + self.soil_moisture_2) / 2.0
        if avg < 0 or avg > 100:
            raise ValueError("Average soil moisture out of valid range 0–100 %")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "co": 0.04, "crop_id": "chilli", "day": 15,
                "device_id": "NEST-001", "farm_id": "farm-01", "hour": 14,
                "humidity": 72, "leaf_wetness": 35, "month": 6,
                "no2": 0.03, "o3": 0.04, "pm2_5": 48, "pressure": 100.8,
                "rainfall": 0, "soil_moisture_1": 22, "soil_moisture_2": 31,
                "soil_temperature": 24, "solar_radiation": 420,
                "wind_direction": 270, "wind_speed": 1.4, "temperature": 28,
            }
        }
    )


# ---------------------------------------------------------------------------
# Inbound: batch prediction request (up to 48 packets = one 24-hour window)
# ---------------------------------------------------------------------------
class BatchPredictRequest(BaseModel):
    """Ordered sequence of NEST packets for sequence-based inference (max 48)."""

    packets: list[NESTPacket] = Field(
        ..., min_length=1, max_length=48,
        description="Chronologically ordered packets (30-min apart)",
    )
    model: Literal["xgb", "gru", "xgb_residual", "gru_residual"] = Field(
        default="xgb_residual",
        description="Which trained model to use for inference",
    )

    @model_validator(mode="after")
    def all_same_device(self) -> "BatchPredictRequest":
        ids = {p.device_id for p in self.packets}
        if len(ids) > 1:
            raise ValueError(f"All packets must share the same device_id; got: {ids}")
        return self


# ---------------------------------------------------------------------------
# Inbound / Outbound: single-packet validation and advice (no temporal predictions)
# ---------------------------------------------------------------------------
class SinglePacketValidationResponse(BaseModel):
    """
    Validation response for a single NEST telemetry packet.
    Validates packet structure, ranges, sensor physical constraints,
    reports current soil moisture and real-time threshold status.
    Temporal predictions are strictly prohibited on single packets.
    """
    status:                Literal["valid"] = "valid"
    device_id:             str = Field(..., description="Source device ID")
    farm_id:               str = Field(..., description="Source farm ID")
    crop_id:               str = Field(..., description="Crop type")
    current_soil_moisture: float = Field(..., description="Average of probe 1 and probe 2 (%)")
    pressure_hpa:          float = Field(..., description="Atmospheric pressure normalized to hPa")
    irrigation_advice:     IrrigationAdvice = Field(..., description="Immediate agronomic state based on current moisture")
    warnings:              list[str] = Field(default_factory=list, description="Validation warnings")
    unavailable_features:  list[str] = Field(default_factory=list, description="Features not supplied by NEST hardware")
    temporal_prediction:   Literal["unavailable_requires_48_packets"] = "unavailable_requires_48_packets"
    message:               str = (
        "Single packet input validated successfully. Temporal 24-hour forecasts require 48 genuine, "
        "consecutive 30-minute packets via /predict/batch."
    )


class QuickPredictRequest(BaseModel):
    """Single NEST packet for validation and immediate agronomic threshold check."""

    packet: NESTPacket
    model: Literal["xgb", "gru", "xgb_residual", "gru_residual"] = "xgb_residual"


# ---------------------------------------------------------------------------
# Outbound: irrigation decision advice
# ---------------------------------------------------------------------------
class IrrigationAdvice(BaseModel):
    irrigate:         bool  = Field(..., description="True if irrigation is recommended")
    soil_stress:      bool  = Field(..., description="True if soil moisture is below MADP")
    avg_soil_moisture:float = Field(..., description="Average of both soil probes (%)")
    madp_threshold:   float = Field(..., description="MADP threshold for this crop (%)")
    fc_threshold:     float = Field(..., description="Field Capacity threshold (%)")
    moisture_deficit: float = Field(..., description="Distance below MADP (0 if not stressed)")


# ---------------------------------------------------------------------------
# Outbound: prediction response
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    """Soil moisture forecast and irrigation recommendation returned by /predict."""

    device_id:              str   = Field(..., description="Source device")
    farm_id:                str   = Field(..., description="Source farm")
    crop_id:                str   = Field(..., description="Crop type")
    model_used:             str   = Field(..., description="Model identifier")
    predicted_soil_moisture:float = Field(..., description="Predicted mean soil moisture in 24 h (%)")
    current_soil_moisture:  float = Field(..., description="Current averaged soil moisture (%)")
    change_predicted:       float = Field(..., description="Predicted change from current (%)")
    irrigation_advice:      IrrigationAdvice
    confidence:             str   = Field(..., description="low / medium / high — based on input completeness")
    data_source:            str   = Field(default="LIVE", description="LIVE or SYNTHETIC")
    warnings:               list[str] = Field(default_factory=list, description="Data quality warnings")
    unavailable_features:   list[str] = Field(default_factory=list, description="Training features not provided in telemetry")


# ---------------------------------------------------------------------------
# Outbound: insufficient history rejection response
# ---------------------------------------------------------------------------
class InsufficientHistoryResponse(BaseModel):
    """Returned when fewer than 48 genuine consecutive packets are provided."""
    status:           str = "rejected"
    error:            str = "insufficient_history"
    detail:           str
    packets_provided: int
    packets_required: int = 48
    resolution:       str = (
        "Provide a complete sequence of 48 genuine, consecutive NEST packets "
        "(30-minute cadence, 24 hours of telemetry) without duplicate readings or temporal gaps via /predict/batch."
    )


# ---------------------------------------------------------------------------
# Outbound: model info
# ---------------------------------------------------------------------------
class ModelInfoResponse(BaseModel):
    """Metadata about all loaded models."""
    models_available: list[str]
    feature_count:    dict[str, int]
    zones_trained_on: list[int]
    sequence_length:  int
    sampling_minutes: int
    target_column:    str
