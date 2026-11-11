"""
NEST Telemetry -> Pest-Risk 26-Feature Mapper (RBB Model)
========================================================
Maps exactly 1,344 consecutive 30-minute NEST IoT telemetry packets (4 full weeks:
4 x 7 x 48 packets) into the exact 26 environmental features expected by the saved
pest-risk classifier (`backend/app/ml/models_pest_risk/best_pest_model.joblib`).

Strict Guarantees:
1. Reuses exact training feature formulas (`ENVIRONMENTAL_FEATURE_COLS`,
   `_rolling_ols_slope`, cyclic wind encoding, 1-week lags, and 4-week rolling
   means/sums/slopes).
2. Validates timestamps (1,344 consecutive 30-min intervals, no gaps/duplicates/
   out-of-order/fabricated readings), physical units, and missing/NaN values via
   `InsufficientHistoryError` and `PredictionBlockedError`.
3. Converts NEST units to NASA POWER scales transparently:
   - `rainfall` (mm/30-min) -> `PRECTOTCORR` (mm/day = weekly_sum / 7.0)
   - `soil_moisture_1/2` (% volumetric) -> `GWETTOP` (0-1 saturation index = SM_avg / SP)
   - `wind_direction` (deg) -> circular weekly mean + `wd_sin`/`wd_cos`
4. Never fabricates history, pads missing packets, or imputes unavailable values.
5. Does NOT modify or retrain any saved models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from app.ml.prepare_pest_dataset import (
    ENVIRONMENTAL_FEATURE_COLS,
    _rolling_ols_slope,
)
from app.services.inference import (
    CROP_THRESHOLDS,
    InsufficientHistoryError,
    PredictionBlockedError,
    validate_sequence_integrity,
)

REQUIRED_PACKETS_PEST = 1344           # 4 weeks * 7 days * 48 packets/day (30-min cadence)
PACKETS_PER_WEEK = 336                 # 7 days * 48 packets/day
DAYS_PER_WEEK = 7.0

REQUIRED_PEST_NEST_FIELDS = (
    "temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "soil_moisture_1",
    "soil_moisture_2",
)


@dataclass(frozen=True)
class PestFeatureVector:
    """Container holding ordered 26-feature array and named dictionary."""
    feature_names: tuple[str, ...]
    feature_dict: dict[str, float]
    feature_array: np.ndarray          # shape (1, 26), float64
    packets_processed: int
    crop_id: str
    saturation_point_used: float
    window_start_timestamp: str
    window_end_timestamp: str


def _resolve_saturation_point(
    packets: Sequence[dict[str, Any]],
    saturation_point: float | None,
) -> tuple[str, float]:
    """
    Resolve soil saturation point (%) for converting volumetric NEST soil moisture (%)
    into the 0-1 GWETTOP saturation index without inventing arbitrary defaults.
    """
    last_pkt = packets[-1]
    crop_id = str(last_pkt.get("crop_id", "")).strip().lower()

    if saturation_point is not None:
        sp = float(saturation_point)
        if not math.isfinite(sp) or not (0.0 < sp <= 100.0):
            raise PredictionBlockedError(
                f"Invalid saturation_point={saturation_point!r}; must be in (0, 100]%."
            )
        return (crop_id or "custom"), sp

    if not crop_id or crop_id not in CROP_THRESHOLDS:
        raise PredictionBlockedError(
            f"Missing or unsupported crop_id={last_pkt.get('crop_id')!r} and no explicit "
            f"saturation_point provided; cannot calibrate soil moisture to GWETTOP saturation index."
        )

    return crop_id, float(CROP_THRESHOLDS[crop_id]["sp"])


def _validate_nest_packets_for_pest(packets: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Strictly enforce 1,344 consecutive 30-min packets and validate all required
    environmental channels, units, and finite numeric bounds.
    """
    pkt_list = [dict(p) for p in packets]
    n = len(pkt_list)

    if n != REQUIRED_PACKETS_PEST:
        raise InsufficientHistoryError(
            f"Pest-risk mapping requires exactly {REQUIRED_PACKETS_PEST} consecutive "
            f"30-minute NEST packets (4 full weeks), but received {n}. "
            f"History fabrication and padding are prohibited.",
            packets_provided=n,
            packets_required=REQUIRED_PACKETS_PEST,
        )

    # Enforce consecutive 30-minute cadence and reject fabricated identical readings
    validate_sequence_integrity(pkt_list, required_length=REQUIRED_PACKETS_PEST)

    for idx, pkt in enumerate(pkt_list):
        for field in REQUIRED_PEST_NEST_FIELDS:
            if field not in pkt or pkt[field] is None:
                raise PredictionBlockedError(
                    f"Missing required field '{field}' at packet index {idx}. "
                    f"Imputation is prohibited."
                )
            try:
                val = float(pkt[field])
            except (TypeError, ValueError) as exc:
                raise PredictionBlockedError(
                    f"Non-numeric value for '{field}' at packet index {idx}: {pkt[field]!r}"
                ) from exc

            if not math.isfinite(val):
                raise PredictionBlockedError(
                    f"Non-finite (NaN/Inf) value for '{field}' at packet index {idx}."
                )

            if field == "temperature" and not (-50.0 <= val <= 70.0):
                raise PredictionBlockedError(
                    f"Out-of-range temperature={val} at packet index {idx}; must be in [-50, 70] C."
                )
            if field == "humidity" and not (0.0 <= val <= 100.0):
                raise PredictionBlockedError(
                    f"Out-of-range humidity={val} at packet index {idx}; must be in [0, 100]%."
                )
            if field == "rainfall" and val < 0.0:
                raise PredictionBlockedError(
                    f"Negative rainfall={val} at packet index {idx}."
                )
            if field == "wind_speed" and not (0.0 <= val <= 100.0):
                raise PredictionBlockedError(
                    f"Out-of-range wind_speed={val} at packet index {idx}; must be in [0, 100] m/s."
                )
            if field == "wind_direction" and not (0.0 <= val <= 360.0):
                raise PredictionBlockedError(
                    f"Out-of-range wind_direction={val} at packet index {idx}; must be in [0, 360] deg."
                )
            if field in ("soil_moisture_1", "soil_moisture_2") and not (0.0 <= val <= 100.0):
                raise PredictionBlockedError(
                    f"Out-of-range {field}={val} at packet index {idx}; must be in [0, 100]%."
                )

    return pkt_list


def _circular_mean_degrees(angles_deg: np.ndarray) -> float:
    """Compute circular mean angle in degrees wrapped to [0, 360)."""
    rad = np.deg2rad(angles_deg)
    mean_sin = float(np.mean(np.sin(rad)))
    mean_cos = float(np.mean(np.cos(rad)))
    if abs(mean_sin) < 1e-12 and abs(mean_cos) < 1e-12:
        return 0.0
    return float(np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360.0)


def map_nest_packets_to_pest_features(
    packets: Sequence[dict[str, Any]],
    saturation_point: float | None = None,
) -> PestFeatureVector:
    """
    Convert 1,344 consecutive 30-minute NEST packets (4 complete weeks)
    into the exact 26 environmental features expected by the RBB pest-risk model.
    """
    pkt_list = _validate_nest_packets_for_pest(packets)
    crop_id, sp = _resolve_saturation_point(pkt_list, saturation_point)

    temp = np.array([float(p["temperature"]) for p in pkt_list], dtype=float)
    rh = np.array([float(p["humidity"]) for p in pkt_list], dtype=float)
    rain = np.array([float(p["rainfall"]) for p in pkt_list], dtype=float)
    ws = np.array([float(p["wind_speed"]) for p in pkt_list], dtype=float)
    wd = np.array([float(p["wind_direction"]) for p in pkt_list], dtype=float)
    sm1 = np.array([float(p["soil_moisture_1"]) for p in pkt_list], dtype=float)
    sm2 = np.array([float(p["soil_moisture_2"]) for p in pkt_list], dtype=float)

    # Convert dual-probe soil moisture (%) to [0, 1] saturation index (GWETTOP)
    gwettop = np.clip(((sm1 + sm2) / 2.0) / sp, 0.0, 1.0)

    # Aggregate into 4 chronological weekly blocks (336 packets per week)
    weekly_rows: list[dict[str, float]] = []
    for wk_idx in range(4):
        s = wk_idx * PACKETS_PER_WEEK
        e = s + PACKETS_PER_WEEK

        t_wk = temp[s:e]
        ws_wk = ws[s:e]
        wd_mean_deg = _circular_mean_degrees(wd[s:e])

        weekly_rows.append({
            "T2M": float(np.mean(t_wk)),
            "T2M_MIN": float(np.min(t_wk)),
            "T2M_MAX": float(np.max(t_wk)),
            "RH2M": float(np.mean(rh[s:e])),
            # NASA POWER PRECTOTCORR is mean daily precipitation rate (mm/day)
            "PRECTOTCORR": float(np.sum(rain[s:e])) / DAYS_PER_WEEK,
            "WS2M": float(np.mean(ws_wk)),
            "WS2M_MAX": float(np.max(ws_wk)),
            "WS2M_MIN": float(np.min(ws_wk)),
            "WD2M": wd_mean_deg,
            "GWETTOP": float(np.mean(gwettop[s:e])),
        })

    win_df = pd.DataFrame(weekly_rows)
    curr = win_df.iloc[-1]       # Week t (packets 1008..1343)
    lag_1w = win_df.iloc[-2]     # Week t-1 (packets 672..1007)

    wd_rad = math.radians(float(curr["WD2M"]))

    # Exact training formulas from prepare_pest_dataset.py
    raw_features: dict[str, float] = {
        # Current-week base weather & soil (10)
        "T2M": round(float(curr["T2M"]), 4),
        "T2M_MIN": round(float(curr["T2M_MIN"]), 4),
        "T2M_MAX": round(float(curr["T2M_MAX"]), 4),
        "RH2M": round(float(curr["RH2M"]), 4),
        "PRECTOTCORR": round(float(curr["PRECTOTCORR"]), 4),
        "WS2M": round(float(curr["WS2M"]), 4),
        "WS2M_MAX": round(float(curr["WS2M_MAX"]), 4),
        "WS2M_MIN": round(float(curr["WS2M_MIN"]), 4),
        "WD2M": round(float(curr["WD2M"]), 4),
        "GWETTOP": round(float(curr["GWETTOP"]), 4),
        # Derived intra-week (4)
        "temp_diurnal_range": round(float(curr["T2M_MAX"] - curr["T2M_MIN"]), 4),
        "wind_gust_spread": round(float(curr["WS2M_MAX"] - curr["WS2M_MIN"]), 4),
        "wd_sin": round(math.sin(wd_rad), 6),
        "wd_cos": round(math.cos(wd_rad), 6),
        # 1-week lags (5)
        "t2m_lag_1w": round(float(lag_1w["T2M"]), 4),
        "rh2m_lag_1w": round(float(lag_1w["RH2M"]), 4),
        "rain_lag_1w": round(float(lag_1w["PRECTOTCORR"]), 4),
        "gwettop_lag_1w": round(float(lag_1w["GWETTOP"]), 4),
        "ws2m_lag_1w": round(float(lag_1w["WS2M"]), 4),
        # 4-week rolling summaries (4)
        "t2m_mean_4w": round(float(win_df["T2M"].mean()), 4),
        "rh2m_mean_4w": round(float(win_df["RH2M"].mean()), 4),
        "rain_sum_4w": round(float(win_df["PRECTOTCORR"].sum()), 4),
        "gwettop_mean_4w": round(float(win_df["GWETTOP"].mean()), 4),
        # 4-week rolling OLS trends (3)
        "t2m_trend_4w": round(_rolling_ols_slope(win_df["T2M"].to_numpy()), 6),
        "rh2m_trend_4w": round(_rolling_ols_slope(win_df["RH2M"].to_numpy()), 6),
        "gwettop_trend_4w": round(_rolling_ols_slope(win_df["GWETTOP"].to_numpy()), 6),
    }

    ordered_dict = {col: raw_features[col] for col in ENVIRONMENTAL_FEATURE_COLS}
    ordered_array = np.array(
        [[float(ordered_dict[col]) for col in ENVIRONMENTAL_FEATURE_COLS]],
        dtype=np.float64,
    )

    return PestFeatureVector(
        feature_names=tuple(ENVIRONMENTAL_FEATURE_COLS),
        feature_dict=ordered_dict,
        feature_array=ordered_array,
        packets_processed=len(pkt_list),
        crop_id=crop_id,
        saturation_point_used=sp,
        window_start_timestamp=str(pkt_list[0]["timestamp"]),
        window_end_timestamp=str(pkt_list[-1]["timestamp"]),
    )
