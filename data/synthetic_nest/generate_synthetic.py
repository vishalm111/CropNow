"""
NEST Synthetic Dataset Generator
=================================
Generates 5 synthetic scenarios × 48 packets (30-min apart) from the
company reference sample `data/company_sample/nest_sample.json`.

All 21 original NEST fields are preserved.
Additional columns added per packet:
  - timestamp          : ISO-8601 datetime (30-min cadence)
  - packet_index       : 0–47
  - scenario           : scenario slug
  - data_source        : always "SYNTHETIC"
  - gt_label           : ground-truth human-readable label
  - gt_irrigate        : 1 = should irrigate, 0 = should not
  - gt_soil_stress     : 1 = soil below crop MADP threshold, 0 = ok
  - gt_sensor_fault    : 1 = packet contains injected spike, 0 = clean

Chilli crop thresholds (from dataset DATA_DICTIONARY.md / pipeline.py):
  - FC  (Field Capacity)   = 35 %
  - MADP (Management Allowable Depletion Point) = 20 %

Output CSVs saved to data/synthetic_nest/:
  scenario_01_normal.csv
  scenario_02_sensor_noise.csv
  scenario_03_gradual_drying.csv
  scenario_04_irrigation_event.csv
  scenario_05_high_humidity.csv
"""

import json
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent.parent.parent          # CropNow/
SAMPLE_JSON = WORKSPACE / "data" / "company_sample" / "nest_sample.json"
OUT_DIR = WORKSPACE / "data" / "synthetic_nest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Seed for reproducibility ───────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)
random.seed(42)

# ── Load reference sample ─────────────────────────────────────────────────────
with open(SAMPLE_JSON) as f:
    BASE = json.load(f)

# ── Chilli crop thresholds ────────────────────────────────────────────────────
CHILLI_FC   = 35.0   # Field Capacity  (%)
CHILLI_MADP = 20.0   # Management Allowable Depletion Point (%)
CROP_ID     = BASE["crop_id"]   # "chilli"

# ── Timeline: start at sample hour on 2026-06-15, 48 packets × 30 min ─────────
START_TS = datetime(2026, int(BASE["month"]), int(BASE["day"]),
                    int(BASE["hour"]), 0, 0)
TIMESTAMPS = [START_TS + timedelta(minutes=30 * i) for i in range(48)]

N = 48   # packets per scenario

# ── Helper: build a base row dict from the reference sample ───────────────────
def base_row(ts: datetime, packet_idx: int, overrides: dict) -> dict:
    """Clone BASE, apply overrides, then inject metadata columns."""
    row = dict(BASE)
    # Update time fields from timestamp
    row["hour"]  = ts.hour
    row["day"]   = ts.day
    row["month"] = ts.month
    # Apply scenario-specific field overrides
    row.update(overrides)
    # Clip moisture to valid range
    row["soil_moisture_1"] = float(np.clip(row["soil_moisture_1"], 0, 100))
    row["soil_moisture_2"] = float(np.clip(row["soil_moisture_2"], 0, 100))
    # Metadata
    row["timestamp"]    = ts.isoformat()
    row["packet_index"] = packet_idx
    return row


def add_gt_labels(rows: list[dict], scenario: str,
                  fault_indices: set | None = None) -> list[dict]:
    """
    Attach ground-truth columns to every row.
    - gt_irrigate  : 1 if average moisture < MADP
    - gt_soil_stress: same as gt_irrigate (alias for ML target)
    - gt_sensor_fault: 1 if this packet index is a known injected spike
    - gt_label     : human-readable string
    - scenario     : slug
    - data_source  : "SYNTHETIC"
    """
    fault_indices = fault_indices or set()
    for row in rows:
        avg_sm = (row["soil_moisture_1"] + row["soil_moisture_2"]) / 2.0
        irrigate = int(avg_sm < CHILLI_MADP)
        fault    = int(row["packet_index"] in fault_indices)

        if fault:
            label = "SENSOR_FAULT"
        elif irrigate:
            label = "IRRIGATE"
        elif avg_sm >= CHILLI_FC:
            label = "FIELD_CAPACITY"
        else:
            label = "NORMAL"

        row["scenario"]        = scenario
        row["data_source"]     = "SYNTHETIC"
        row["gt_irrigate"]     = irrigate
        row["gt_soil_stress"]  = irrigate
        row["gt_sensor_fault"] = fault
        row["gt_label"]        = label
    return rows


# ── Column order (matches original JSON fields + new cols) ───────────────────
ORIGINAL_FIELDS = [
    "co", "crop_id", "day", "device_id", "farm_id", "hour",
    "humidity", "leaf_wetness", "month", "no2", "o3", "pm2_5",
    "pressure", "rainfall", "soil_moisture_1", "soil_moisture_2",
    "soil_temperature", "solar_radiation", "wind_direction",
    "wind_speed", "temperature",
]
META_FIELDS = [
    "timestamp", "packet_index", "scenario", "data_source",
    "gt_label", "gt_irrigate", "gt_soil_stress", "gt_sensor_fault",
]
COLUMN_ORDER = ORIGINAL_FIELDS + META_FIELDS


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — Normal readings
# Stable soil moisture between MADP and FC with realistic diurnal variation.
# Temperature rises midday; humidity drops; solar radiation peaks at noon.
# ══════════════════════════════════════════════════════════════════════════════
def scenario_normal() -> pd.DataFrame:
    rows = []
    sm1 = BASE["soil_moisture_1"]   # 22 %
    sm2 = BASE["soil_moisture_2"]   # 31 %
    for i, ts in enumerate(TIMESTAMPS):
        # Diurnal hour (wraps around midnight)
        hour_of_day = ts.hour + ts.minute / 60.0
        # Temperature: 24°C night, +6°C midday peak
        temp = 25.0 + 6.0 * math.sin(math.pi * (hour_of_day - 6) / 12) \
               + rng.normal(0, 0.3)
        # Humidity inverse with temperature
        hum  = 72.0 - 0.8 * (temp - 25.0) + rng.normal(0, 1.0)
        # Solar radiation: daytime trapezoid
        solar = max(0, 480 * math.sin(math.pi * (hour_of_day - 6) / 12)) \
                + rng.normal(0, 10)
        solar = max(0, solar)
        # Soil moisture: gentle evapotranspiration -0.05%/packet + small noise
        sm1 = sm1 - 0.05 + rng.normal(0, 0.05)
        sm2 = sm2 - 0.05 + rng.normal(0, 0.05)
        sm1 = float(np.clip(sm1, CHILLI_MADP + 1, CHILLI_FC - 1))
        sm2 = float(np.clip(sm2, CHILLI_MADP + 1, CHILLI_FC - 1))

        rows.append(base_row(ts, i, {
            "temperature":      round(float(temp), 2),
            "humidity":         round(float(np.clip(hum, 40, 95)), 1),
            "solar_radiation":  round(float(solar), 1),
            "soil_moisture_1":  round(sm1, 2),
            "soil_moisture_2":  round(sm2, 2),
            "soil_temperature": round(23.0 + 0.05 * i + rng.normal(0, 0.1), 2),
            "wind_speed":       round(float(np.abs(1.4 + rng.normal(0, 0.2))), 2),
            "leaf_wetness":     round(float(np.clip(35 + rng.normal(0, 2), 0, 100)), 1),
            "rainfall":         0,
            "co":               round(float(np.clip(0.04 + rng.normal(0, 0.003), 0.01, 0.2)), 4),
            "no2":              round(float(np.clip(0.03 + rng.normal(0, 0.002), 0.005, 0.1)), 4),
            "o3":               round(float(np.clip(0.04 + rng.normal(0, 0.003), 0.01, 0.15)), 4),
            "pm2_5":            round(float(np.clip(48 + rng.normal(0, 3), 10, 150)), 1),
            "pressure":         round(float(np.clip(100.8 + rng.normal(0, 0.1), 98, 104)), 2),
        }))
    rows = add_gt_labels(rows, "normal")
    return pd.DataFrame(rows)[COLUMN_ORDER]


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — Sensor noise / spikes
# Clean baseline with 6 injected outlier spikes (random packets).
# Spikes: extreme moisture readings, impossible temperature or negative humidity.
# ══════════════════════════════════════════════════════════════════════════════
def scenario_sensor_noise() -> pd.DataFrame:
    rows = []
    spike_indices = set(rng.choice(48, size=6, replace=False).tolist())
    sm1 = 25.0
    sm2 = 29.0
    for i, ts in enumerate(TIMESTAMPS):
        hour_of_day = ts.hour + ts.minute / 60.0
        temp  = 26.0 + 4.0 * math.sin(math.pi * (hour_of_day - 6) / 12) + rng.normal(0, 0.3)
        solar = max(0, 400 * math.sin(math.pi * (hour_of_day - 6) / 12)) + rng.normal(0, 8)

        sm1 += rng.normal(-0.04, 0.06)
        sm2 += rng.normal(-0.04, 0.06)
        sm1 = float(np.clip(sm1, 20, 34))
        sm2 = float(np.clip(sm2, 20, 34))

        if i in spike_indices:
            spike_type = int(rng.integers(0, 4))
            if spike_type == 0:
                # Moisture dropout → 0
                s1_out, s2_out = 0.0, 0.0
                temp_out = round(float(temp), 2)
                hum_out  = round(float(np.clip(72 + rng.normal(0, 1), 40, 95)), 1)
            elif spike_type == 1:
                # Moisture stuck at max (sensor saturation)
                s1_out, s2_out = 100.0, 100.0
                temp_out = round(float(temp), 2)
                hum_out  = round(float(np.clip(72 + rng.normal(0, 1), 40, 95)), 1)
            elif spike_type == 2:
                # Temperature spike
                s1_out, s2_out = round(sm1, 2), round(sm2, 2)
                temp_out = round(float(rng.uniform(55, 80)), 2)   # impossible temp
                hum_out  = round(float(np.clip(72 + rng.normal(0, 1), 40, 95)), 1)
            else:
                # Humidity spike (> 100 or negative — clamped to realistic later)
                s1_out, s2_out = round(sm1, 2), round(sm2, 2)
                temp_out = round(float(temp), 2)
                hum_out  = -5.0   # impossible value
            rows.append(base_row(ts, i, {
                "temperature":      temp_out,
                "humidity":         hum_out,
                "solar_radiation":  round(float(solar), 1),
                "soil_moisture_1":  s1_out,
                "soil_moisture_2":  s2_out,
                "soil_temperature": round(23.5 + rng.normal(0, 0.2), 2),
                "wind_speed":       round(float(np.abs(1.4 + rng.normal(0, 0.3))), 2),
                "leaf_wetness":     round(float(np.clip(35 + rng.normal(0, 2), 0, 100)), 1),
                "rainfall": 0,
                "co":    round(float(np.clip(0.04 + rng.normal(0, 0.003), 0.01, 0.2)), 4),
                "no2":   round(float(np.clip(0.03 + rng.normal(0, 0.002), 0.005, 0.1)), 4),
                "o3":    round(float(np.clip(0.04 + rng.normal(0, 0.003), 0.01, 0.15)), 4),
                "pm2_5": round(float(np.clip(48 + rng.normal(0, 3), 10, 150)), 1),
                "pressure": round(float(np.clip(100.8 + rng.normal(0, 0.1), 98, 104)), 2),
            }))
        else:
            rows.append(base_row(ts, i, {
                "temperature":      round(float(temp), 2),
                "humidity":         round(float(np.clip(72 + rng.normal(0, 1), 40, 95)), 1),
                "solar_radiation":  round(float(solar), 1),
                "soil_moisture_1":  round(sm1, 2),
                "soil_moisture_2":  round(sm2, 2),
                "soil_temperature": round(23.5 + rng.normal(0, 0.2), 2),
                "wind_speed":       round(float(np.abs(1.4 + rng.normal(0, 0.3))), 2),
                "leaf_wetness":     round(float(np.clip(35 + rng.normal(0, 2), 0, 100)), 1),
                "rainfall": 0,
                "co":    round(float(np.clip(0.04 + rng.normal(0, 0.003), 0.01, 0.2)), 4),
                "no2":   round(float(np.clip(0.03 + rng.normal(0, 0.002), 0.005, 0.1)), 4),
                "o3":    round(float(np.clip(0.04 + rng.normal(0, 0.003), 0.01, 0.15)), 4),
                "pm2_5": round(float(np.clip(48 + rng.normal(0, 3), 10, 150)), 1),
                "pressure": round(float(np.clip(100.8 + rng.normal(0, 0.1), 98, 104)), 2),
            }))
    rows = add_gt_labels(rows, "sensor_noise", fault_indices=spike_indices)
    return pd.DataFrame(rows)[COLUMN_ORDER]


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — Gradual soil drying
# Soil moisture starts near FC (35%) and dries linearly toward MADP over 48h.
# High temperature and low humidity accelerate evapotranspiration.
# Rainfall remains 0 throughout.  gt_irrigate flips to 1 once below MADP.
# ══════════════════════════════════════════════════════════════════════════════
def scenario_gradual_drying() -> pd.DataFrame:
    rows = []
    # Start near Field Capacity; drop ~0.33%/packet → reaches MADP at packet ~45
    sm1_start, sm2_start = 34.5, 33.8
    total_drop = sm1_start - (CHILLI_MADP - 1.0)      # ~15.5 %
    drop_per_packet = total_drop / 46.0

    for i, ts in enumerate(TIMESTAMPS):
        hour_of_day = ts.hour + ts.minute / 60.0
        # Hot, dry day — accelerates drying
        temp  = 30.0 + 5.0 * math.sin(math.pi * (hour_of_day - 6) / 12) + rng.normal(0, 0.4)
        hum   = 55.0 - 0.9 * (temp - 30.0) + rng.normal(0, 1.5)
        solar = max(0, 650 * math.sin(math.pi * (hour_of_day - 6) / 12)) + rng.normal(0, 15)
        solar = max(0, solar)

        sm1 = sm1_start - i * drop_per_packet + rng.normal(0, 0.05)
        sm2 = sm2_start - i * drop_per_packet + rng.normal(0, 0.05)

        rows.append(base_row(ts, i, {
            "temperature":      round(float(temp), 2),
            "humidity":         round(float(np.clip(hum, 25, 90)), 1),
            "solar_radiation":  round(float(solar), 1),
            "soil_moisture_1":  float(np.clip(round(sm1, 2), 0, 100)),
            "soil_moisture_2":  float(np.clip(round(sm2, 2), 0, 100)),
            "soil_temperature": round(25.0 + 0.05 * i + rng.normal(0, 0.15), 2),
            "wind_speed":       round(float(np.abs(2.0 + rng.normal(0, 0.4))), 2),
            "leaf_wetness":     round(float(np.clip(20 - 0.3 * i + rng.normal(0, 1.5), 0, 100)), 1),
            "rainfall":         0,
            "co":    round(float(np.clip(0.04 + rng.normal(0, 0.003), 0.01, 0.2)), 4),
            "no2":   round(float(np.clip(0.03 + rng.normal(0, 0.002), 0.005, 0.1)), 4),
            "o3":    round(float(np.clip(0.045 + rng.normal(0, 0.003), 0.01, 0.15)), 4),
            "pm2_5": round(float(np.clip(52 + rng.normal(0, 4), 10, 150)), 1),
            "pressure": round(float(np.clip(100.5 + rng.normal(0, 0.15), 98, 104)), 2),
        }))
    rows = add_gt_labels(rows, "gradual_drying")
    return pd.DataFrame(rows)[COLUMN_ORDER]


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 4 — Irrigation event
# Soil starts dry (below MADP); irrigation fires at packet 8 and runs for 6
# packets (3 hours). Moisture rises rapidly from ~15% to ~32%.
# Rainfall stays 0 (drip/sprinkler, not rain).
# After irrigation, soil slowly returns toward stable equilibrium.
# ══════════════════════════════════════════════════════════════════════════════
def scenario_irrigation_event() -> pd.DataFrame:
    rows = []
    IRRIGATE_START = 8      # packet index when irrigation begins
    IRRIGATE_END   = 14     # packet index when irrigation ends (exclusive)
    RISE_PER_PACKET = 2.8   # % soil moisture gained per irrigation packet

    sm1 = 14.0   # starts below MADP
    sm2 = 16.0

    for i, ts in enumerate(TIMESTAMPS):
        hour_of_day = ts.hour + ts.minute / 60.0
        temp  = 27.0 + 4.0 * math.sin(math.pi * (hour_of_day - 6) / 12) + rng.normal(0, 0.3)
        hum   = 68.0 - 0.6 * (temp - 27.0) + rng.normal(0, 1.2)
        solar = max(0, 500 * math.sin(math.pi * (hour_of_day - 6) / 12)) + rng.normal(0, 10)
        solar = max(0, solar)

        # Rainfall sensor stays 0 (drip irrigation, not rain)
        rain = 0

        if IRRIGATE_START <= i < IRRIGATE_END:
            # Irrigation active: moisture rises steeply
            sm1 += RISE_PER_PACKET + rng.normal(0, 0.15)
            sm2 += RISE_PER_PACKET + rng.normal(0, 0.15)
            # Humidity rises near canopy during irrigation
            hum = min(hum + 8, 95)
            # Leaf wetness increases (sprinkler/overhead)
            lw = 70.0 + rng.normal(0, 3)
        elif i >= IRRIGATE_END:
            # Post-irrigation: gentle drying again
            sm1 -= 0.08 + rng.normal(0, 0.04)
            sm2 -= 0.08 + rng.normal(0, 0.04)
            lw = max(35.0, 70.0 - 3.0 * (i - IRRIGATE_END) + rng.normal(0, 2))
        else:
            # Pre-irrigation: continuing to dry
            sm1 -= 0.12 + rng.normal(0, 0.04)
            sm2 -= 0.12 + rng.normal(0, 0.04)
            lw = 30.0 + rng.normal(0, 2)

        sm1 = float(np.clip(sm1, 0, 100))
        sm2 = float(np.clip(sm2, 0, 100))

        rows.append(base_row(ts, i, {
            "temperature":      round(float(temp), 2),
            "humidity":         round(float(np.clip(hum, 40, 98)), 1),
            "solar_radiation":  round(float(solar), 1),
            "soil_moisture_1":  round(sm1, 2),
            "soil_moisture_2":  round(sm2, 2),
            "soil_temperature": round(23.0 + rng.normal(0, 0.2), 2),
            "wind_speed":       round(float(np.abs(1.2 + rng.normal(0, 0.3))), 2),
            "leaf_wetness":     round(float(np.clip(lw, 0, 100)), 1),
            "rainfall":         rain,
            "co":    round(float(np.clip(0.04 + rng.normal(0, 0.003), 0.01, 0.2)), 4),
            "no2":   round(float(np.clip(0.03 + rng.normal(0, 0.002), 0.005, 0.1)), 4),
            "o3":    round(float(np.clip(0.04 + rng.normal(0, 0.003), 0.01, 0.15)), 4),
            "pm2_5": round(float(np.clip(45 + rng.normal(0, 3), 10, 150)), 1),
            "pressure": round(float(np.clip(100.9 + rng.normal(0, 0.1), 98, 104)), 2),
        }))
    rows = add_gt_labels(rows, "irrigation_event")
    return pd.DataFrame(rows)[COLUMN_ORDER]


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 5 — High humidity and leaf wetness
# Simulates pre-monsoon / foggy conditions.
# Humidity > 85%, leaf wetness > 70, cool temperatures, low solar radiation.
# Soil moisture stable (absorbed atmospheric moisture slows evapotranspiration).
# No active rain, but high condensation environment.
# ══════════════════════════════════════════════════════════════════════════════
def scenario_high_humidity() -> pd.DataFrame:
    rows = []
    sm1 = 27.0
    sm2 = 29.5

    for i, ts in enumerate(TIMESTAMPS):
        hour_of_day = ts.hour + ts.minute / 60.0
        # Cool, overcast: smaller diurnal swing
        temp  = 23.0 + 2.5 * math.sin(math.pi * (hour_of_day - 6) / 12) + rng.normal(0, 0.4)
        # Humidity stays very high
        hum   = 88.0 + 5.0 * math.sin(math.pi * (hour_of_day + 6) / 12) + rng.normal(0, 1.5)
        hum   = float(np.clip(hum, 80, 100))
        # Low solar radiation (overcast)
        solar = max(0, 120 * math.sin(math.pi * (hour_of_day - 6) / 12)) + rng.normal(0, 15)
        solar = max(0, solar)
        # Leaf wetness very high (dew / fog)
        lw    = 75.0 + 10.0 * math.sin(math.pi * i / 24) + rng.normal(0, 2)
        lw    = float(np.clip(lw, 60, 100))
        # Soil moisture barely changes — high humidity suppresses ET
        sm1  += rng.normal(0, 0.04)
        sm2  += rng.normal(0, 0.04)
        sm1   = float(np.clip(sm1, 20, 40))
        sm2   = float(np.clip(sm2, 20, 40))

        rows.append(base_row(ts, i, {
            "temperature":      round(float(temp), 2),
            "humidity":         round(hum, 1),
            "solar_radiation":  round(float(solar), 1),
            "soil_moisture_1":  round(sm1, 2),
            "soil_moisture_2":  round(sm2, 2),
            "soil_temperature": round(21.5 + rng.normal(0, 0.15), 2),
            "wind_speed":       round(float(np.abs(0.5 + rng.normal(0, 0.2))), 2),
            "leaf_wetness":     round(lw, 1),
            "rainfall":         0,
            "co":    round(float(np.clip(0.038 + rng.normal(0, 0.003), 0.01, 0.2)), 4),
            "no2":   round(float(np.clip(0.028 + rng.normal(0, 0.002), 0.005, 0.1)), 4),
            "o3":    round(float(np.clip(0.032 + rng.normal(0, 0.003), 0.01, 0.15)), 4),
            "pm2_5": round(float(np.clip(60 + rng.normal(0, 5), 10, 150)), 1),   # higher in humid/foggy
            "pressure": round(float(np.clip(101.2 + rng.normal(0, 0.1), 98, 104)), 2),
        }))
    rows = add_gt_labels(rows, "high_humidity")
    return pd.DataFrame(rows)[COLUMN_ORDER]


# ══════════════════════════════════════════════════════════════════════════════
# Generate and save all scenarios
# ══════════════════════════════════════════════════════════════════════════════
scenarios = [
    ("scenario_01_normal",          scenario_normal),
    ("scenario_02_sensor_noise",    scenario_sensor_noise),
    ("scenario_03_gradual_drying",  scenario_gradual_drying),
    ("scenario_04_irrigation_event",scenario_irrigation_event),
    ("scenario_05_high_humidity",   scenario_high_humidity),
]

summary_rows = []
for slug, fn in scenarios:
    df = fn()
    out_path = OUT_DIR / f"{slug}.csv"
    df.to_csv(out_path, index=False)

    label_counts = df["gt_label"].value_counts().to_dict()
    summary_rows.append({
        "scenario":   slug,
        "rows":       len(df),
        "columns":    len(df.columns),
        "gt_irrigate_pct": f"{df['gt_irrigate'].mean()*100:.1f}%",
        "gt_fault_pct":    f"{df['gt_sensor_fault'].mean()*100:.1f}%",
        "label_dist": str(label_counts),
        "file":       str(out_path.name),
    })
    print(f"✓ {slug}.csv  |  rows={len(df)}  |  labels={label_counts}")

# ── Save summary CSV ──────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "generation_summary.csv", index=False)
print(f"\n✓ generation_summary.csv saved.")
print(f"\nAll files in {OUT_DIR}:")
for f in sorted(OUT_DIR.glob("*.csv")):
    print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
