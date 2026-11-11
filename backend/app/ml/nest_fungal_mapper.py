"""
NEST Telemetry -> Fungal-Risk 38-Feature Mapper
=============================================
Maps exactly 336 consecutive 30-minute NEST IoT telemetry packets (7 full days)
into the exact 38 environmental features expected by the saved fungal-risk
classifier (`backend/app/ml/models_fungal_risk/best_fungal_model.joblib`).

Strict Guarantees:
1. Reuses exact training feature formulas (`_ols_slope`, Magnus-Tetens dew point,
   5-pt/30-min duration weighting, 3d-vs-4d deltas, and exact `FEATURE_COLS` order).
2. Rejects incomplete (<336 or !=336), gapped, out-of-order, duplicated, or
   missing/NaN/out-of-range telemetry via `InsufficientHistoryError` / `PredictionBlockedError`.
3. Never invents, pads, or imputes missing sensor values.
4. Does NOT modify or retrain any saved models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from app.ml.fungal_features import FEATURE_COLS
from app.ml.prepare_fungal_dataset import _ols_slope
from app.services.inference import (
    InsufficientHistoryError,
    PredictionBlockedError,
    validate_sequence_integrity,
)

REQUIRED_PACKETS_FUNGAL = 336          # 7 days * 48 packets/day (30-min cadence)
PACKETS_PER_DAY = 48
HOURS_PER_NEST_PACKET = 0.5            # 30 minutes = 0.5 hours

# Magnus-Tetens constants (Alduchov & Eskridge 1996)
_MAGNUS_A = 17.625
_MAGNUS_B = 243.04

REQUIRED_FUNGAL_NEST_FIELDS = (
    "temperature",
    "humidity",
    "leaf_wetness",
    "rainfall",
    "solar_radiation",
    "soil_moisture_1",
    "soil_moisture_2",
)


@dataclass(frozen=True)
class FungalFeatureVector:
    """Container holding ordered 38-feature array and named dictionary."""
    feature_names: tuple[str, ...]
    feature_dict: dict[str, float | int]
    feature_array: np.ndarray          # shape (1, 38), float64
    packets_processed: int
    window_start_timestamp: str
    window_end_timestamp: str


def _compute_magnus_dew_point(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Derive dew point (C) from temperature (C) and relative humidity (%)."""
    gamma = (_MAGNUS_A * temp_c) / (_MAGNUS_B + temp_c) + np.log(rh_pct / 100.0)
    return (_MAGNUS_B * gamma) / (_MAGNUS_A - gamma)


def _validate_nest_packets_for_fungal(packets: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Strictly validate sequence continuity (336 consecutive 30-min packets) and
    verify every required sensor channel is present, finite, and physically valid.
    Never invents or imputes missing values.
    """
    pkt_list = [dict(p) for p in packets]
    n = len(pkt_list)

    if n != REQUIRED_PACKETS_FUNGAL:
        raise InsufficientHistoryError(
            f"Fungal-risk mapping requires exactly {REQUIRED_PACKETS_FUNGAL} consecutive "
            f"30-minute NEST packets (7 full days), but received {n}. "
            f"Padding and partial windows are prohibited.",
            packets_provided=n,
            packets_required=REQUIRED_PACKETS_FUNGAL,
        )

    # Enforce consecutive 30-minute timestamp cadence & non-fabricated readings
    validate_sequence_integrity(pkt_list, required_length=REQUIRED_PACKETS_FUNGAL)

    # Validate every required physical channel in all 336 packets
    for idx, pkt in enumerate(pkt_list):
        for field in REQUIRED_FUNGAL_NEST_FIELDS:
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

            if field == "humidity" and not (0.0 < val <= 100.0):
                raise PredictionBlockedError(
                    f"Invalid humidity={val} at packet index {idx}; must be in (0, 100]."
                )
            if field == "temperature" and not (-50.0 <= val <= 70.0):
                raise PredictionBlockedError(
                    f"Out-of-range temperature={val} at packet index {idx}; must be in [-50, 70] C."
                )
            if field == "leaf_wetness" and not (0.0 <= val <= 100.0):
                raise PredictionBlockedError(
                    f"Out-of-range leaf_wetness={val} at packet index {idx}; must be in [0, 100]."
                )
            if field == "rainfall" and val < 0.0:
                raise PredictionBlockedError(
                    f"Negative rainfall={val} at packet index {idx}."
                )
            if field == "solar_radiation" and val < 0.0:
                raise PredictionBlockedError(
                    f"Negative solar_radiation={val} at packet index {idx}."
                )
            if field in ("soil_moisture_1", "soil_moisture_2") and not (0.0 <= val <= 100.0):
                raise PredictionBlockedError(
                    f"Out-of-range {field}={val} at packet index {idx}; must be in [0, 100]%."
                )

    return pkt_list


def map_nest_packets_to_fungal_features(
    packets: Sequence[dict[str, Any]],
) -> FungalFeatureVector:
    """
    Convert 336 consecutive 30-minute NEST packets (7 complete 24h blocks)
    into the exact 38 training features expected by the fungal-risk model.
    """
    pkt_list = _validate_nest_packets_for_fungal(packets)

    temp = np.array([float(p["temperature"]) for p in pkt_list], dtype=float)
    rh = np.array([float(p["humidity"]) for p in pkt_list], dtype=float)
    lw = np.array([float(p["leaf_wetness"]) for p in pkt_list], dtype=float)
    rain = np.array([float(p["rainfall"]) for p in pkt_list], dtype=float)
    solar = np.array([float(p["solar_radiation"]) for p in pkt_list], dtype=float)
    sm1 = np.array([float(p["soil_moisture_1"]) for p in pkt_list], dtype=float)
    sm2 = np.array([float(p["soil_moisture_2"]) for p in pkt_list], dtype=float)

    # Derived per-packet channels matching training dataset physics
    dew_point = _compute_magnus_dew_point(temp, rh)
    dp_depression = temp - dew_point
    water_content = ((sm1 + sm2) / 2.0) / 100.0  # Convert % moisture -> [0, 1] fraction

    is_rh_80 = (rh >= 80.0).astype(float)
    is_rh_90 = (rh >= 90.0).astype(float)
    is_lw_wet = (lw > 0.0).astype(float)
    is_lw_heavy = (lw >= 20.0).astype(float)
    is_favorable_inf = ((lw > 0.0) & (temp >= 5.0) & (temp <= 25.0)).astype(float)

    # Aggregate into 7 consecutive 24-hour blocks (48 packets per day)
    daily_rows: list[dict[str, float]] = []
    for day_idx in range(7):
        s = day_idx * PACKETS_PER_DAY
        e = s + PACKETS_PER_DAY

        t_day = temp[s:e]
        rh_day = rh[s:e]
        dp_day = dew_point[s:e]
        dpd_day = dp_depression[s:e]
        lw_day = lw[s:e]

        t_min = float(np.min(t_day))
        t_max = float(np.max(t_day))

        daily_rows.append({
            "temp_mean": float(np.mean(t_day)),
            "temp_min": t_min,
            "temp_max": t_max,
            "temp_diurnal_range": t_max - t_min,
            "dp_mean": float(np.mean(dp_day)),
            "dp_depression_mean": float(np.mean(dpd_day)),
            "rh_mean": float(np.mean(rh_day)),
            "rh_min": float(np.min(rh_day)),
            "rh_max": float(np.max(rh_day)),
            "rh_ge_80_hours": float(np.sum(is_rh_80[s:e])) * HOURS_PER_NEST_PACKET,
            "rh_ge_90_hours": float(np.sum(is_rh_90[s:e])) * HOURS_PER_NEST_PACKET,
            "lw_mean": float(np.mean(lw_day)),
            "lw_max": float(np.max(lw_day)),
            "lw_wet_hours": float(np.sum(is_lw_wet[s:e])) * HOURS_PER_NEST_PACKET,
            "lw_heavy_hours": float(np.sum(is_lw_heavy[s:e])) * HOURS_PER_NEST_PACKET,
            "favorable_inf_hours": float(np.sum(is_favorable_inf[s:e])) * HOURS_PER_NEST_PACKET,
            "rain_total": float(np.sum(rain[s:e])),
            "solar_mean": float(np.mean(solar[s:e])),
            "water_content_mean": float(np.mean(water_content[s:e])),
        })

    win_df = pd.DataFrame(daily_rows)
    prior_4d = win_df.iloc[:4]
    recent_3d = win_df.iloc[4:]
    day_prev = win_df.iloc[-1]

    # Exact training formulas from prepare_fungal_dataset.py
    raw_features: dict[str, float | int] = {
        # Temperature (8)
        "temp_mean_7d": round(float(win_df["temp_mean"].mean()), 4),
        "temp_min_7d": round(float(win_df["temp_min"].min()), 4),
        "temp_max_7d": round(float(win_df["temp_max"].max()), 4),
        "temp_std_7d": round(float(win_df["temp_mean"].std(ddof=1)), 4),
        "temp_diurnal_range_mean_7d": round(float(win_df["temp_diurnal_range"].mean()), 4),
        "temp_mean_24h": round(float(day_prev["temp_mean"]), 4),
        "dp_mean_7d": round(float(win_df["dp_mean"].mean()), 4),
        "dp_depression_mean_7d": round(float(win_df["dp_depression_mean"].mean()), 4),
        # Humidity (7)
        "rh_mean_7d": round(float(win_df["rh_mean"].mean()), 4),
        "rh_min_7d": round(float(win_df["rh_min"].min()), 4),
        "rh_max_7d": round(float(win_df["rh_max"].max()), 4),
        "rh_std_7d": round(float(win_df["rh_mean"].std(ddof=1)), 4),
        "rh_mean_24h": round(float(day_prev["rh_mean"]), 4),
        "rh_ge_80_hours_7d": round(float(win_df["rh_ge_80_hours"].sum()), 2),
        "rh_ge_90_hours_7d": round(float(win_df["rh_ge_90_hours"].sum()), 2),
        # Leaf Wetness (7)
        "lw_mean_7d": round(float(win_df["lw_mean"].mean()), 4),
        "lw_max_7d": round(float(win_df["lw_max"].max()), 4),
        "lw_mean_24h": round(float(day_prev["lw_mean"]), 4),
        "lw_wet_hours_7d": round(float(win_df["lw_wet_hours"].sum()), 2),
        "lw_heavy_wet_hours_7d": round(float(win_df["lw_heavy_hours"].sum()), 2),
        "lw_wet_hours_24h": round(float(day_prev["lw_wet_hours"]), 2),
        "favorable_infection_hours_7d": round(float(win_df["favorable_inf_hours"].sum()), 2),
        # Rainfall, Soil & Solar (7)
        "rain_total_7d": round(float(win_df["rain_total"].sum()), 4),
        "rain_max_daily_7d": round(float(win_df["rain_total"].max()), 4),
        "rain_days_count_7d": int((win_df["rain_total"] > 0.2).sum()),
        "rain_total_24h": round(float(day_prev["rain_total"]), 4),
        "soil_moisture_mean_7d": round(float(win_df["water_content_mean"].mean()), 4),
        "soil_moisture_mean_24h": round(float(day_prev["water_content_mean"]), 4),
        "solar_radiation_mean_7d": round(float(win_df["solar_mean"].mean()), 4),
        # 7-Day Trends (9)
        "temp_trend_slope_7d": round(_ols_slope(win_df["temp_mean"].to_numpy()), 4),
        "rh_trend_slope_7d": round(_ols_slope(win_df["rh_mean"].to_numpy()), 4),
        "lw_hours_trend_slope_7d": round(_ols_slope(win_df["lw_wet_hours"].to_numpy()), 4),
        "rain_trend_slope_7d": round(_ols_slope(win_df["rain_total"].to_numpy()), 4),
        "soil_moisture_trend_slope_7d": round(_ols_slope(win_df["water_content_mean"].to_numpy()), 6),
        "temp_delta_3d_vs_4d": round(float(recent_3d["temp_mean"].mean() - prior_4d["temp_mean"].mean()), 4),
        "rh_delta_3d_vs_4d": round(float(recent_3d["rh_mean"].mean() - prior_4d["rh_mean"].mean()), 4),
        "lw_hours_delta_3d_vs_4d": round(float(recent_3d["lw_wet_hours"].mean() - prior_4d["lw_wet_hours"].mean()), 4),
        "rain_delta_3d_vs_4d": round(float(recent_3d["rain_total"].mean() - prior_4d["rain_total"].mean()), 4),
    }

    # Preserve exact training feature order from FEATURE_COLS
    ordered_dict = {col: raw_features[col] for col in FEATURE_COLS}
    ordered_array = np.array([[float(ordered_dict[col]) for col in FEATURE_COLS]], dtype=np.float64)

    return FungalFeatureVector(
        feature_names=tuple(FEATURE_COLS),
        feature_dict=ordered_dict,
        feature_array=ordered_array,
        packets_processed=len(pkt_list),
        window_start_timestamp=str(pkt_list[0]["timestamp"]),
        window_end_timestamp=str(pkt_list[-1]["timestamp"]),
    )
