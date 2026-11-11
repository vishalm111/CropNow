"""
Pydantic schemas for the Soil Moisture Stability and Signal Conditioning API.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class RawTelemetryPacket(BaseModel):
    """
    Inbound packet for stability and signal conditioning analysis.
    Preserves raw moisture signals (including probe faults / noise) without
    prematurely rejecting real-world electrical or sensor anomalies that need filtering.
    """
    device_id: str = Field(default="NEST-001", description="NEST device ID")
    farm_id: str = Field(default="farm-01", description="Farm ID")
    crop_id: str = Field(default="chilli", description="Crop type")
    soil_moisture_1: float = Field(..., description="Raw Probe 1 reading (%)")
    soil_moisture_2: float = Field(..., description="Raw Probe 2 reading (%)")
    timestamp: str | None = Field(default=None, description="ISO-8601 timestamp")
    temperature: float | None = Field(default=None)
    humidity: float | None = Field(default=None)
    pressure: float | None = Field(default=None)
    solar_radiation: float | None = Field(default=None)
    wind_speed: float | None = Field(default=None)
    rainfall: float | None = Field(default=None)

    model_config = {"extra": "allow"}


class StabilityAnalyzeRequest(BaseModel):
    """Sequence of 48 consecutive 30-minute NEST telemetry packets for stability analysis."""
    packets: list[RawTelemetryPacket] = Field(
        ...,
        min_length=1,
        max_length=48,
        description="Exactly 48 genuine, consecutive 30-minute packets.",
    )


class StabilizedPacketRecord(BaseModel):
    """Dual-probe observation holding both raw and filtered signals with decision state."""
    timestamp: str | None = Field(default=None, description="ISO-8601 timestamp")
    packet_index: int = Field(..., description="Index within the 48-packet window (0-47)")
    raw_soil_moisture_1: float = Field(..., description="Raw Probe 1 reading (%)")
    raw_soil_moisture_2: float = Field(..., description="Raw Probe 2 reading (%)")
    raw_average: float = Field(..., description="Raw arithmetic average of probe 1 and 2 (%)")
    filtered_soil_moisture_1: float = Field(..., description="Median + EWMA filtered Probe 1 (%)")
    filtered_soil_moisture_2: float = Field(..., description="Median + EWMA filtered Probe 2 (%)")
    stabilized_average: float = Field(..., description="Stabilized combined average (%)")
    is_sudden_change_probe_1: bool = Field(..., description="True if sudden physical transition was detected on Probe 1")
    is_sudden_change_probe_2: bool = Field(..., description="True if sudden physical transition was detected on Probe 2")
    irrigation_decision: bool = Field(..., description="Hysteresis-stabilized actuator decision (True = Valve ON)")
    irrigation_advisory: bool = Field(..., description="Threshold-based irrigation stress advisory (True if stabilized_average < MADP)")
    stress_status: Literal["IRRIGATION_STRESS", "NO_STRESS"] = Field(..., description="Explicit stress status independent of actuator hysteresis")
    state_description: str = Field(..., description="Human-readable state description")


class StabilityMetricsSummary(BaseModel):
    """Quantitative performance metrics comparing raw vs stabilized signals."""
    total_variation_reduction_pct: float = Field(..., description="Reduction in signal first-order roughness (%)")
    high_frequency_jitter_reduction_pct: float = Field(..., description="Suppression of high-frequency noise variance (%)")
    probe1_noise_reduction_pct: float = Field(..., description="Probe 1 total variation reduction (%)")
    probe2_noise_reduction_pct: float = Field(..., description="Probe 2 total variation reduction (%)")
    dynamic_range_retention_pct: float = Field(..., description="Retention of genuine clean-signal dynamic range excluding spikes (%)")
    clean_baseline_range: float = Field(..., description="Clean raw dynamic range excluding isolated fault spikes (%)")
    filtered_dynamic_range: float = Field(..., description="Filtered signal dynamic range (%)")
    outlier_suppression_pct: float = Field(..., description="Suppression of isolated sensor spike amplitude (%)")
    outliers_detected_count: int = Field(..., description="Count of isolated sensor spikes excluded from genuine range")
    sudden_events_detected_count: int = Field(..., description="Number of genuine sudden physical transitions fast-tracked")
    raw_switch_count: int = Field(..., description="Number of actuator toggles under raw thresholding")
    filtered_switch_count: int = Field(..., description="Number of actuator toggles under hysteresis control")
    chatter_reduction_pct: float = Field(..., description="Percentage reduction in actuator oscillation/chatter (%)")


class StabilityAnalyzeResponse(BaseModel):
    """Full analysis response returned by POST /stability/analyze."""
    status: Literal["success"] = "success"
    device_id: str = Field(..., description="NEST device ID")
    farm_id: str = Field(..., description="Farm ID")
    crop_id: str = Field(..., description="Crop type")
    madp_threshold: float = Field(..., description="MADP threshold (%)")
    packets_analyzed: int = Field(..., description="Count of packets processed")
    timeline: list[StabilizedPacketRecord] = Field(..., description="Chronological record of raw and filtered states")
    metrics: StabilityMetricsSummary = Field(..., description="Quantitative noise reduction and chatter suppression metrics")


# ---------------------------------------------------------------------------
# Synthetic Scenario Catalog Schemas
# ---------------------------------------------------------------------------
class ScenarioSummary(BaseModel):
    """Metadata describing an available synthetic scenario."""
    id: str = Field(..., description="Unique scenario ID (e.g. scenario_01_normal)")
    name: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="Description of scenario conditions")
    packet_count: int = Field(default=48, description="Number of packets")
    data_source: Literal["SYNTHETIC"] = "SYNTHETIC"


class ScenarioListResponse(BaseModel):
    """Response containing list of available synthetic scenarios."""
    scenarios: list[ScenarioSummary]
    data_source: Literal["SYNTHETIC"] = "SYNTHETIC"
    note: str = "All scenario telemetry is SYNTHETIC generated data based on company NEST reference schema."


class ScenarioDetailResponse(BaseModel):
    """Detailed response containing 48 packets for a specific scenario."""
    id: str = Field(..., description="Scenario ID")
    name: str = Field(..., description="Human-readable scenario title")
    description: str = Field(..., description="Description of the test conditions")
    data_source: Literal["SYNTHETIC"] = "SYNTHETIC"
    packet_count: int = Field(..., description="Number of packets returned")
    packets: list[dict[str, Any]] = Field(..., description="Chronological list of 48 telemetry packets")
