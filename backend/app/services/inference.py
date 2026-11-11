"""
NEST Irrigation Prediction — Inference Service
===============================================
Loads the trained residual and absolute models once at startup (singleton),
maps NEST IoT packet fields onto the internal feature space used during
training, and returns soil-moisture forecasts + irrigation recommendations.

Architectures (from pipeline.py / residual_pipeline.py):
  XGBoost: flat_seq (T*F) + window aggregation stats → scalar prediction
    - Absolute:  T=48, F=18 → 48*18 + 14 = 878 features
    - Residual:  T=48, F=21 → 48*21 + 17 = 1025 features

  GRU: 2-layer GRU → head = Sequential(Linear(64,32), ReLU, Dropout, Linear(32,1))
    - Absolute:  input_dim=18
    - Residual:  input_dim=21

NEST → model column mapping defined in NEST_TO_MODEL_MAP below.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).parent.parent.parent          # backend/
_ML_DIR      = _BACKEND_DIR / "app" / "ml"
_MODELS_ABS  = _ML_DIR / "models"
_MODELS_RES  = _ML_DIR / "models_residual"


class PredictionBlockedError(ValueError):
    """Raised when required features are missing or invalid, blocking unreliable predictions."""
    pass


class InsufficientHistoryError(PredictionBlockedError):
    """Raised when fewer than 48 genuine consecutive 30-minute packets are provided or data is fabricated."""
    def __init__(self, message: str, packets_provided: int, packets_required: int = 48):
        super().__init__(message)
        self.error_code = "insufficient_history"
        self.packets_provided = packets_provided
        self.packets_required = packets_required


# ── Required telemetry features from NEST field devices ───────────────────────
REQUIRED_NEST_FIELDS = (
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "solar_radiation",
    "soil_temperature",
    "soil_moisture_1",
    "soil_moisture_2",
    "rainfall",
    "crop_id",
)

# ── Training features that are unavailable in standard NEST telemetry ──────────
UNAVAILABLE_TRAINING_FEATURES = (
    "ec",
    "ph",
    "irrigation_duration_minutes",
    "water_vol_past_4h",
    "soil_temperature_7-18cm",
)

# ── Statistical neutral baselines (scaler training means, z = 0.0) ────────────
# Used when training features are not present in telemetry to avoid injecting
# arbitrary biased numbers (e.g. EC was mistakenly set to 6.5 instead of ~181.7 µS/cm).
TRAINING_FEATURE_MEANS: dict[str, float] = {
    "ec": 181.747,                  # µS/cm (training mean across all zones)
    "ph": 5.972,                    # pH (training mean across all zones)
    "irrigation_duration_minutes": 0.188,  # min (training mean)
    "water_vol_past_4h": 1.743,     # L (training mean)
    "soil_temperature_7-18cm": 28.683, # °C (training deep soil mean)
}

# ── Crop thresholds used at inference time for NEST crops ─────────────────────
CROP_THRESHOLDS: dict[str, dict[str, float]] = {
    "chilli":    {"pwp": 10.0, "madp": 20.0, "fc": 35.0, "sp": 45.0, "flow_rate": 1.0},
    "tomato":    {"pwp": 20.0, "madp": 40.0, "fc": 60.0, "sp": 80.0, "flow_rate": 1.1},
    "zucchini":  {"pwp": 40.0, "madp": 80.0, "fc": 90.0, "sp": 95.0, "flow_rate": 1.1},
    "blueberry": {"pwp": 35.0, "madp": 70.0, "fc": 90.0, "sp": 95.0, "flow_rate": 0.4},
    "wheat":     {"pwp": 15.0, "madp": 30.0, "fc": 55.0, "sp": 70.0, "flow_rate": 0.8},
    "rice":      {"pwp": 25.0, "madp": 60.0, "fc": 80.0, "sp": 90.0, "flow_rate": 2.0},
    "maize":     {"pwp": 20.0, "madp": 45.0, "fc": 65.0, "sp": 80.0, "flow_rate": 1.3},
}

# ── Training feature column order (matches pipeline.py) ───────────────────────
FEATURE_COLS = [
    "weather_temp", "weather_humidity", "weather_rain", "weather_pressure",
    "weather_wind_speed", "weather_radiation",
    "soil_temperature_0-7cm", "soil_temperature_7-18cm",
    "ec", "ph",
    "soil_moisture",
    "irrigation_duration_minutes", "water_vol_past_4h",
]
STATIC_COLS    = ["flow_rate", "pwp", "madp", "fc", "sp"]
CROP_NORM_COLS = ["relative_moisture", "dist_to_madp", "dist_to_fc"]

# Column orders (must match pipeline.py ALL_COLS / residual_pipeline.py RESIDUAL_ALL_COLS)
ABS_ALL_COLS = FEATURE_COLS + STATIC_COLS                    # 18 cols
RES_ALL_COLS = FEATURE_COLS + CROP_NORM_COLS + STATIC_COLS   # 21 cols

SEQ_LEN = 48    # training sequence length


# ── GRU architecture — matches SoilMoistureResidualGRU in both pipelines ──────
class _GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── Tabular feature extraction — mirrors pipeline.py extract_tabular_features ──
def _extract_abs_tabular(X_seq: np.ndarray) -> np.ndarray:
    """
    Replicates pipeline.py::extract_tabular_features.
    Input: (N, 48, 18); Output: (N, 48*18 + 14) = (N, 878)
    """
    N, T, F = X_seq.shape
    sm_idx    = ABS_ALL_COLS.index("soil_moisture")
    temp_idx  = ABS_ALL_COLS.index("weather_temp")
    rad_idx   = ABS_ALL_COLS.index("weather_radiation")
    rain_idx  = ABS_ALL_COLS.index("weather_rain")
    irr_idx   = ABS_ALL_COLS.index("irrigation_duration_minutes")
    water_idx = ABS_ALL_COLS.index("water_vol_past_4h")

    flat_seq    = X_seq.reshape(N, T * F)
    sm_win      = X_seq[:, :, sm_idx]
    temp_win    = X_seq[:, :, temp_idx]
    rad_win     = X_seq[:, :, rad_idx]
    rain_win    = X_seq[:, :, rain_idx]
    irr_win     = X_seq[:, :, irr_idx]

    stats = np.column_stack([
        sm_win[:, -1],
        sm_win[:, -1] - sm_win[:, 0],
        sm_win[:, -1] - sm_win[:, -9] if T >= 9 else np.zeros(N),
        np.mean(sm_win, axis=1),
        np.std(sm_win, axis=1),
        np.min(sm_win, axis=1),
        np.max(sm_win, axis=1),
        np.mean(temp_win, axis=1),
        np.max(temp_win, axis=1),
        np.mean(rad_win, axis=1),
        np.sum(rain_win, axis=1),
        np.sum(rain_win[:, -8:], axis=1) if T >= 8 else np.sum(rain_win, axis=1),
        np.sum(irr_win, axis=1),
        X_seq[:, -1, water_idx],
    ])
    return np.hstack([flat_seq, stats])


def _extract_res_tabular(X_seq: np.ndarray) -> np.ndarray:
    """
    Replicates residual_pipeline.py::extract_tabular_residual_features.
    Input: (N, 48, 21); Output: (N, 48*21 + 17) = (N, 1025)
    """
    N, T, F = X_seq.shape
    sm_idx     = RES_ALL_COLS.index("soil_moisture")
    rel_sm_idx = RES_ALL_COLS.index("relative_moisture")
    temp_idx   = RES_ALL_COLS.index("weather_temp")
    rad_idx    = RES_ALL_COLS.index("weather_radiation")
    rain_idx   = RES_ALL_COLS.index("weather_rain")
    irr_idx    = RES_ALL_COLS.index("irrigation_duration_minutes")
    water_idx  = RES_ALL_COLS.index("water_vol_past_4h")

    flat_seq   = X_seq.reshape(N, T * F)
    sm_win     = X_seq[:, :, sm_idx]
    rel_sm_win = X_seq[:, :, rel_sm_idx]
    temp_win   = X_seq[:, :, temp_idx]
    rad_win    = X_seq[:, :, rad_idx]
    rain_win   = X_seq[:, :, rain_idx]
    irr_win    = X_seq[:, :, irr_idx]

    stats = np.column_stack([
        sm_win[:, -1],
        rel_sm_win[:, -1],
        sm_win[:, -1] - sm_win[:, 0],
        sm_win[:, -1] - sm_win[:, -9] if T >= 9 else np.zeros(N),
        sm_win[:, -1] - sm_win[:, -3] if T >= 3 else np.zeros(N),
        np.mean(sm_win, axis=1),
        np.std(sm_win, axis=1),
        np.min(sm_win, axis=1),
        np.max(sm_win, axis=1),
        np.mean(temp_win, axis=1),
        np.max(temp_win, axis=1),
        np.mean(rad_win, axis=1),
        np.sum(rain_win, axis=1),
        np.sum(rain_win[:, -8:], axis=1) if T >= 8 else np.sum(rain_win, axis=1),
        np.sum(irr_win, axis=1),
        np.sum(irr_win[:, -8:], axis=1) if T >= 8 else np.sum(irr_win, axis=1),
        X_seq[:, -1, water_idx],
    ])
    return np.hstack([flat_seq, stats])


# ── Sequence Integrity Validation ─────────────────────────────────────────────
def validate_sequence_integrity(packets: list[dict], required_length: int = 48) -> None:
    """
    Validates that the provided sequence contains exactly `required_length`
    genuine, consecutive NEST packets spaced 30 minutes apart.

    Raises:
        InsufficientHistoryError: if sequence is incomplete, has temporal gaps,
                                  has duplicate timestamps, or contains fabricated/duplicated readings.
    """
    n = len(packets)
    if n < required_length:
        raise InsufficientHistoryError(
            f"Incomplete sequence: {required_length} genuine, consecutive 30-minute packets "
            f"(24-hour lookback) are required for temporal prediction, but only {n} were provided. "
            f"Automatic padding and reading fabrication are disabled.",
            packets_provided=n,
            packets_required=required_length,
        )

    # 1. Reject fabricated historical data (all packets identical duplicate readings)
    first = packets[0]
    all_readings_identical = all(
        p.get("soil_moisture_1") == first.get("soil_moisture_1")
        and p.get("soil_moisture_2") == first.get("soil_moisture_2")
        and p.get("temperature") == first.get("temperature")
        and p.get("solar_radiation") == first.get("solar_radiation")
        and p.get("humidity") == first.get("humidity")
        for p in packets
    )
    if all_readings_identical:
        raise InsufficientHistoryError(
            "Fabricated historical data detected: all 48 packets contain identical duplicate sensor readings. "
            "Duplicating a single reading to simulate temporal history is forbidden. Genuine time-varying telemetry is required.",
            packets_provided=n,
            packets_required=required_length,
        )

    # 2. Check timestamps: each packet must include a timestamp, and intervals must be 30 minutes
    timestamps = []
    for i, p in enumerate(packets):
        ts_val = p.get("timestamp")
        if not ts_val:
            raise InsufficientHistoryError(
                f"Missing timestamp at packet index {i}: each packet in the 48-step sequence must "
                f"include an ISO-8601 timestamp to verify consecutive 30-minute cadence.",
                packets_provided=n,
                packets_required=required_length,
            )
        try:
            dt = pd.to_datetime(ts_val).to_pydatetime()
            timestamps.append(dt)
        except Exception as err:
            raise InsufficientHistoryError(
                f"Invalid timestamp at packet index {i} ('{ts_val}'): {err}",
                packets_provided=n,
                packets_required=required_length,
            )

    for i in range(1, n):
        delta_min = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60.0
        if delta_min == 0.0:
            raise InsufficientHistoryError(
                f"Duplicate timestamp detected at packet index {i} ({timestamps[i].isoformat()}). "
                f"Consecutive packets must not share identical timestamps.",
                packets_provided=n,
                packets_required=required_length,
            )
        if abs(delta_min - 30.0) > 1.0:  # Allow max 1 min clock jitter
            raise InsufficientHistoryError(
                f"Temporal gap detected between packet {i-1} ({timestamps[i-1].isoformat()}) "
                f"and packet {i} ({timestamps[i].isoformat()}): interval is {delta_min:.1f} minutes, "
                f"expected exactly 30 minutes. Historical sequence is discontinuous.",
                packets_provided=n,
                packets_required=required_length,
            )


# ── Singleton model registry ───────────────────────────────────────────────────
class ModelRegistry:
    _instance: "ModelRegistry | None" = None

    def __init__(self) -> None:
        self._models:   dict[str, Any] = {}
        self._scalers:  dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self._load_all()

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        self._try_load_xgb(
            "xgb", _MODELS_ABS / "xgb_model.joblib",
            _MODELS_ABS / "scaler.joblib",
            n_features=len(ABS_ALL_COLS), is_residual=False,
        )
        self._try_load_gru(
            "gru", _MODELS_ABS / "gru_model.pt",
            _MODELS_ABS / "scaler.joblib",
            n_features=len(ABS_ALL_COLS), is_residual=False,
        )
        self._try_load_xgb(
            "xgb_residual", _MODELS_RES / "xgb_residual_model.joblib",
            _MODELS_RES / "scaler_residual.joblib",
            n_features=len(RES_ALL_COLS), is_residual=True,
        )
        self._try_load_gru(
            "gru_residual", _MODELS_RES / "gru_residual_model.pt",
            _MODELS_RES / "scaler_residual.joblib",
            n_features=len(RES_ALL_COLS), is_residual=True,
        )
        logger.info("ModelRegistry loaded: %s", list(self._models.keys()))

    def _try_load_xgb(self, key: str, model_path: Path, scaler_path: Path,
                      n_features: int, is_residual: bool) -> None:
        try:
            self._models[key]  = joblib.load(model_path)
            self._scalers[key] = joblib.load(scaler_path)
            self._metadata[key] = {
                "type": "xgb", "n_features": n_features, "residual": is_residual
            }
            logger.info("Loaded XGBoost model: %s", key)
        except Exception as exc:
            logger.warning("Could not load %s: %s", key, exc)

    def _try_load_gru(self, key: str, model_path: Path, scaler_path: Path,
                      n_features: int, is_residual: bool) -> None:
        try:
            self._scalers[key] = joblib.load(scaler_path)
            gru = _GRUModel(input_size=n_features)
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            gru.load_state_dict(state)
            gru.eval()
            self._models[key] = gru
            self._metadata[key] = {
                "type": "gru", "n_features": n_features, "residual": is_residual
            }
            logger.info("Loaded GRU model: %s", key)
        except Exception as exc:
            logger.warning("Could not load %s: %s", key, exc)

    # ------------------------------------------------------------------
    @property
    def available_models(self) -> list[str]:
        return list(self._models.keys())

    # ------------------------------------------------------------------
    def predict(
        self,
        packets: list[dict],
        model_key: str = "xgb_residual",
        strict: bool = False,
    ) -> dict:
        """
        Run inference on exactly 48 genuine, consecutive NEST packets (30 min apart).
        Validates sequence integrity, telemetry inputs, converts units, flags unavailable
        training features, and returns predicted_soil_moisture, current_soil_moisture,
        change_predicted, warnings.

        Rejects incomplete sequences (< 48 packets), gaps, duplicate timestamps,
        or duplicated readings with InsufficientHistoryError.
        """
        if model_key not in self._models:
            raise ValueError(
                f"Model '{model_key}' is not loaded. Available: {self.available_models}"
            )

        # 1. Enforce 48 genuine, consecutive NEST packets with 30-minute timestamps
        # Automatic padding and reading fabrication are disabled.
        validate_sequence_integrity(packets, required_length=SEQ_LEN)

        warnings: list[str] = []
        is_residual = self._metadata[model_key]["residual"]
        is_gru = self._metadata[model_key]["type"] == "gru"
        scaler = self._scalers[model_key]

        # Build feature matrix and collect unavailable features across packets
        rows = []
        all_unavailable = set()
        for p in packets:
            feat_vec, unavail = self._packet_to_features(p, is_residual)
            rows.append(feat_vec)
            all_unavailable.update(unavail)

        unavail_list = sorted(list(all_unavailable))

        # Check strict mode blocking
        if strict and unavail_list:
            raise PredictionBlockedError(
                f"Prediction blocked in strict mode: training features unavailable in telemetry: {unavail_list}."
            )

        # Flag unavailable features in warnings
        if unavail_list:
            warnings.append(
                f"Unavailable training features imputed at training baseline (z=0.0): {unavail_list}. "
                f"Hardware telemetry does not provide these channels."
            )

        # Check crop domain shift
        last_crop = str(packets[-1].get("crop_id", "chilli"))
        if last_crop not in ["tomato", "zucchini", "blueberry"]:
            thresh = CROP_THRESHOLDS.get(last_crop, CROP_THRESHOLDS["chilli"])
            warnings.append(
                f"Crop '{last_crop}' thresholds (PWP={thresh.get('pwp')}, MADP={thresh.get('madp')}, FC={thresh.get('fc')}) "
                f"are outside the model training set (Tomato, Zucchini, Blueberry). Uncertainty is elevated."
            )

        # Exactly 48 genuine consecutive packets (no padding)
        X = np.array(rows[:SEQ_LEN], dtype=np.float32)         # (48, n_feat)

        # Scale per-step (scaler was fit on (N*T, F) data)
        X_scaled = scaler.transform(X).astype(np.float32)      # (48, n_feat)

        # Current SM from last raw packet
        last_pkt = packets[-1]
        current_sm = (float(last_pkt["soil_moisture_1"]) + float(last_pkt["soil_moisture_2"])) / 2.0

        # ── XGBoost: build tabular features from full sequence ─────────
        if not is_gru:
            X_3d = X_scaled[np.newaxis, :, :]                  # (1, SEQ_LEN, n_feat)
            if is_residual:
                X_tab = _extract_res_tabular(X_3d)             # (1, 1025)
            else:
                X_tab = _extract_abs_tabular(X_3d)             # (1, 878)
            raw_pred = float(self._models[model_key].predict(X_tab)[0])

        # ── GRU: full scaled sequence (1, SEQ_LEN, n_feat) ────────────
        else:
            tensor = torch.tensor(
                X_scaled[np.newaxis, :, :], dtype=torch.float32
            )
            with torch.no_grad():
                raw_pred = float(self._models[model_key](tensor).item())

        # Reconstruct absolute prediction
        if is_residual:
            predicted_sm = float(np.clip(current_sm + raw_pred, 0, 100))
        else:
            predicted_sm = float(np.clip(raw_pred, 0, 100))

        reliability = "degraded" if (unavail_list or last_crop not in ["tomato", "zucchini", "blueberry"]) else "high"

        return {
            "predicted_soil_moisture": round(predicted_sm, 3),
            "current_soil_moisture":   round(current_sm, 3),
            "change_predicted":        round(predicted_sm - current_sm, 3),
            "warnings":                warnings,
            "unavailable_features":    unavail_list,
            "reliability":             reliability,
        }

    # ------------------------------------------------------------------
    def _packet_to_features(self, pkt: dict, residual: bool) -> tuple[list[float], list[str]]:
        """
        Map a NEST packet dict to the 18- or 21-element feature vector.
        Validates required features, converts units (pressure kPa -> hPa),
        calculates correct residual features, and flags unavailable training features.

        Returns:
            (feature_vector, unavailable_features_list)
        """
        # 1. Check required NEST sensor fields
        missing_required = [f for f in REQUIRED_NEST_FIELDS if f not in pkt or pkt[f] is None]
        if missing_required:
            dev = pkt.get("device_id", "unknown")
            raise PredictionBlockedError(
                f"Prediction blocked for device '{dev}': missing required telemetry feature(s): {missing_required}. "
                f"Cannot generate reliable predictions with missing sensor inputs."
            )

        # 2. Physical boundary sanity checks
        hum = float(pkt["humidity"])
        if hum < 0.0 or hum > 100.0:
            raise PredictionBlockedError(
                f"Prediction blocked: physical anomaly detected. Humidity {hum}% is outside valid range [0, 100]."
            )

        sm1 = float(pkt["soil_moisture_1"])
        sm2 = float(pkt["soil_moisture_2"])
        if sm1 < 0.0 or sm1 > 100.0 or sm2 < 0.0 or sm2 > 100.0:
            raise PredictionBlockedError(
                f"Prediction blocked: physical anomaly detected. Soil moisture probes ({sm1}%, {sm2}%) outside valid range [0, 100]."
            )

        raw_press = float(pkt["pressure"])
        if raw_press <= 0.0:
            raise PredictionBlockedError(
                f"Prediction blocked: atmospheric pressure {raw_press} must be positive."
            )

        temp = float(pkt["temperature"])
        if temp < -40.0 or temp > 70.0:
            raise PredictionBlockedError(
                f"Prediction blocked: air temperature {temp}°C outside valid terrestrial range [-40, 70]."
            )

        # 3. Unit conversion: NEST pressure in kPa -> hPa (ERA5 training unit)
        # 1 kPa = 10 hPa. Standard sea-level pressure is ~101.3 kPa = ~1013.25 hPa.
        # If value is already in hPa (> 500), keep as-is; otherwise convert kPa -> hPa.
        weather_pressure = raw_press * 10.0 if raw_press < 200.0 else raw_press

        crop = pkt["crop_id"]
        thresh = CROP_THRESHOLDS.get(crop, CROP_THRESHOLDS["chilli"])
        sm_avg = (sm1 + sm2) / 2.0

        # 4. Handle unavailable training features without fabricating fake numbers
        unavailable = []
        if "ec" in pkt and pkt["ec"] is not None:
            ec_val = float(pkt["ec"])
        else:
            ec_val = TRAINING_FEATURE_MEANS["ec"]
            unavailable.append("ec")

        if "ph" in pkt and pkt["ph"] is not None:
            ph_val = float(pkt["ph"])
        else:
            ph_val = TRAINING_FEATURE_MEANS["ph"]
            unavailable.append("ph")

        if "irrigation_duration_minutes" in pkt and pkt["irrigation_duration_minutes"] is not None:
            irr_val = float(pkt["irrigation_duration_minutes"])
        else:
            irr_val = TRAINING_FEATURE_MEANS["irrigation_duration_minutes"]
            unavailable.append("irrigation_duration_minutes")

        if "water_vol_past_4h" in pkt and pkt["water_vol_past_4h"] is not None:
            vol_val = float(pkt["water_vol_past_4h"])
        else:
            vol_val = TRAINING_FEATURE_MEANS["water_vol_past_4h"]
            unavailable.append("water_vol_past_4h")

        if "soil_temperature_7_18cm" in pkt and pkt["soil_temperature_7_18cm"] is not None:
            soil_temp_deep = float(pkt["soil_temperature_7_18cm"])
        else:
            # Single probe available: deep temperature is unmeasured
            soil_temp_deep = TRAINING_FEATURE_MEANS["soil_temperature_7-18cm"]
            unavailable.append("soil_temperature_7-18cm")

        soil_temp_surf = float(pkt["soil_temperature"])
        rain_val = float(pkt["rainfall"])
        wind_val = float(pkt["wind_speed"])
        rad_val = float(pkt["solar_radiation"])

        # ── Dynamic (13 cols) ────────────────────────────────────────────
        vec = [
            temp,              # weather_temp
            hum,               # weather_humidity
            rain_val,          # weather_rain
            weather_pressure,  # weather_pressure (in hPa)
            wind_val,          # weather_wind_speed
            rad_val,           # weather_radiation
            soil_temp_surf,    # soil_temperature_0-7cm
            soil_temp_deep,    # soil_temperature_7-18cm (unmeasured deep probe)
            ec_val,            # ec (µS/cm)
            ph_val,            # ph
            sm_avg,            # soil_moisture
            irr_val,           # irrigation_duration_minutes
            vol_val,           # water_vol_past_4h
        ]

        # ── Crop-normalised (3 cols) — residual models only ─────────────
        # Training formula: (sm - PWP) / (FC - PWP), dist_to_madp = sm - madp, dist_to_fc = sm - fc
        if residual:
            pwp = thresh["pwp"]
            madp = thresh["madp"]
            fc = thresh["fc"]
            denom = max(fc - pwp, 1.0)
            relative_moisture = (sm_avg - pwp) / denom
            dist_to_madp = sm_avg - madp
            dist_to_fc = sm_avg - fc
            vec += [relative_moisture, dist_to_madp, dist_to_fc]

        # ── Static / crop-spec (5 cols) ───────────────────────────────────
        vec += [
            thresh["flow_rate"],
            thresh["pwp"],
            thresh["madp"],
            thresh["fc"],
            thresh["sp"],
        ]
        return vec, unavailable

    # ------------------------------------------------------------------
    def validate_single_packet(self, pkt: dict) -> dict:
        """
        Validate a single inbound NEST packet without generating temporal predictions.
        Checks required fields, verifies physical sanity constraints, calculates
        current soil moisture, pressure conversion, and returns immediate agronomic advice.
        """
        # Run feature conversion and sanity checks (raises PredictionBlockedError on invalid input)
        _, unavailable = self._packet_to_features(pkt, residual=True)

        sm1 = float(pkt["soil_moisture_1"])
        sm2 = float(pkt["soil_moisture_2"])
        current_sm = (sm1 + sm2) / 2.0

        raw_press = float(pkt["pressure"])
        pressure_hpa = raw_press * 10.0 if raw_press < 200.0 else raw_press

        crop_id = str(pkt.get("crop_id", "chilli"))
        thresh = CROP_THRESHOLDS.get(crop_id, CROP_THRESHOLDS["chilli"])

        warnings = []
        if crop_id not in ["tomato", "zucchini", "blueberry"]:
            warnings.append(
                f"Crop '{crop_id}' thresholds (PWP={thresh.get('pwp')}, MADP={thresh.get('madp')}, FC={thresh.get('fc')}) "
                f"are outside the model training set (Tomato, Zucchini, Blueberry)."
            )

        return {
            "status": "valid",
            "device_id": pkt.get("device_id", "unknown"),
            "farm_id": pkt.get("farm_id", "unknown"),
            "crop_id": crop_id,
            "current_soil_moisture": round(current_sm, 3),
            "pressure_hpa": round(pressure_hpa, 2),
            "warnings": warnings,
            "unavailable_features": unavailable,
            "temporal_prediction": "unavailable_requires_48_packets",
            "message": (
                "Single packet input validated successfully. Temporal 24-hour forecasts require 48 genuine, "
                "consecutive 30-minute packets via /predict/batch."
            ),
        }

    # ------------------------------------------------------------------
    def model_info(self) -> dict:
        return {
            "models_available": self.available_models,
            "feature_count":    {k: v["n_features"] for k, v in self._metadata.items()},
            "zones_trained_on": [1, 2, 4, 5],
            "sequence_length":  SEQ_LEN,
            "sampling_minutes": 30,
            "target_column":    "target_mean_24h",
        }
