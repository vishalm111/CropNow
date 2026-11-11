# NEST Irrigation Dataset Audit Report

**Audit Target:** `02_processed_data/preprocessed`
**Dataset Reference:** Multi-Sector Precision-Irrigation Field Trial (Arnesano, Lecce, Italy, 2025)
**Location on Disk:** `data/Soil Moisture, Irrigation Actuator and Weather Dat/Soil Moisture, Irrigation Actuator and Weather Dat/02_processed_data/preprocessed/`
**Audit Date:** November 2026
**Status:** Completed (No modifications made to source data; no models trained)

---

## Executive Summary

The preprocessed layer consists of five per-sector CSV files derived from IoT sensors and Open-Meteo ERA5 reanalysis weather data. The files are indexed at a nominal 10-minute cadence and are designed for 24-hour ahead soil moisture forecasting.

| Sector | Crop / Setting | Total Rows | Temporal Coverage | 10-min Adherence | Target Nulls | Max Gap |
|---|---|---|---|---|---|---|
| **Zone 1** | Tomato, open field | 9,531 | 2025-02-28 15:00 to 2025-09-04 18:50 | 91.53% | 0 | 3d 05h 20m |
| **Zone 2** | Tomato, open field | 11,929 | 2025-02-28 14:20 to 2025-09-23 10:10 | 91.23% | 0 | 8d 08h 00m |
| **Zone 3** | Tomato, pots | 1,118 | 2025-02-14 13:10 to 2025-06-25 08:10 | 72.18% | 0 | 12d 19h 50m |
| **Zone 4** | Zucchini, soil | 12,676 | 2025-03-01 07:10 to 2025-09-23 14:40 | 92.21% | 0 | 2d 21h 40m |
| **Zone 5** | Blueberry, soil | 11,814 | 2025-02-11 09:20 to 2025-09-22 20:00 | 86.50% | 0 | 14d 02h 00m |

---

## 1. Actual Timestamp Intervals and Missing Time Periods

Although the data originates on a regular 10-minute sampling grid, the preprocessing pipeline (`03_code/02_preprocessing.py`) drops rows where soil probe data is missing or corrupted beyond a 2-step (20-minute) interpolation limit. As a result, **none of the datasets are continuous**.

### Interval Distribution (Delta Breakdown)

- **Nominal Cadence (10 min):**
  - Zone 1: 8,724 steps (91.53%)
  - Zone 2: 10,883 steps (91.23%)
  - Zone 3: 807 steps (72.18%)
  - Zone 4: 11,689 steps (92.21%)
  - Zone 5: 10,219 steps (86.50%)
- **20-minute intervals (1 interpolated gap step):**
  - Zone 1: 1.63% | Zone 2: 1.36% | Zone 3: 3.22% | Zone 4: 1.80% | Zone 5: 9.55%
- **Gaps > 10 minutes:**
  - **Zone 1:** 806 gaps (Max: 3 days 05:20:00)
    - `2025-04-25 14:10:00` to `2025-04-28 19:30:00` (77.3 hours)
    - `2025-05-04 16:00:00` to `2025-05-07 11:50:00` (67.8 hours)
    - `2025-07-18 05:30:00` to `2025-07-20 16:20:00` (58.8 hours)
  - **Zone 2:** 1,045 gaps (Max: 8 days 08:00:00)
    - `2025-04-23 14:20:00` to `2025-05-01 22:20:00` (200.0 hours)
    - `2025-07-06 18:10:00` to `2025-07-13 05:50:00` (155.7 hours)
    - `2025-07-15 19:00:00` to `2025-07-19 04:30:00` (81.5 hours)
  - **Zone 3:** 310 gaps (Max: 12 days 19:50:00)
    - `2025-04-24 10:20:00` to `2025-05-07 06:10:00` (307.8 hours)
    - Probe completely ceased functioning on `2025-06-25 08:10:00`; zero data exists after June.
  - **Zone 4:** 986 gaps (Max: 2 days 21:40:00)
    - `2025-04-28 09:10:00` to `2025-05-01 06:50:00` (69.7 hours)
    - `2025-05-03 00:00:00` to `2025-05-05 10:10:00` (58.2 hours)
  - **Zone 5:** 1,594 gaps (Max: 14 days 02:00:00)
    - `2025-02-14 11:40:00` to `2025-02-28 13:40:00` (338.0 hours)
    - `2025-05-02 03:00:00` to `2025-05-05 21:30:00` (90.5 hours)

---

## 2. Duplicate Timestamps

- **Duplicate Check:** Evaluated `ts.duplicated().sum()` on all 5 files.
- **Result:** **0 duplicate timestamps** in all five preprocessed CSV files. Every timestamp is strictly unique and sorted chronologically.

---

## 3. Missing Values in Both 24-Hour Target Columns

The dataset specifies two 24-hour targets computed across a forward horizon of 144 steps (144 × 10 min = 24 hours):
1. `target_point_24h`: Single-point soil moisture reading at $t + 24\text{h}$.
2. `target_mean_24h`: Average soil moisture across the interval $[t, t + 24\text{h}]$.

### Audit Findings

- `target_point_24h`: **0 missing values** across all five files.
- `target_mean_24h`: **0 missing values** across all five files.

### Why Target Columns Have Zero Nulls
In `03_code/02_preprocessing.py`, the preprocessing pipeline executes:
```python
df = df.dropna(subset=[f"target_point_{horizon}h", f"target_mean_{horizon}h"])
```
Any row at the end of the season or immediately preceding a major data dropout that lacked a valid 24-hour target window was explicitly dropped before writing the preprocessed CSVs.

*(Note: Diagnostic column `real_moisture_delta` contains exactly 144 null values at the tail end of each file, but this is a diagnostic outcome measure and not a model target).*

---

## 4. Feasibility of Creating 48 Consecutive 30-Minute Packets

A window of **48 consecutive 30-minute packets** corresponds to $48 \times 30 = 1,440\text{ minutes} = 24\text{ continuous hours}$.

Because the original data is at 10-minute intervals, a 30-minute packet corresponds to 3 aggregated 10-minute intervals. For 48 continuous packets, an uninterrupted contiguous span of 24 hours (144 continuous 10-minute intervals) without missing periods is required.

### Zone-by-Zone Feasibility Analysis

| Zone | 30-min Bins with Data | Contiguous 24h Blocks ($\ge 48$ packets) | Valid 48-Packet Sliding Windows | Longest Continuous Streak | Reliable? |
|---|---|---|---|---|---|
| **Zone 1** | 3,663 / 9,032 (40.6%) | 7 blocks | 1,569 windows | 1,106 packets (23.0 days) | **Conditionally** |
| **Zone 2** | 4,625 / 9,929 (46.6%) | 8 blocks | 2,421 windows | 901 packets (18.8 days) | **Conditionally** |
| **Zone 3** | 573 / 6,279 (9.1%) | **0 blocks** | **0 windows** | 16 packets (8.0 hours) | **NO (Impossible)** |
| **Zone 4** | 4,815 / 9,904 (48.6%) | 6 blocks | 2,581 windows | 1,413 packets (29.4 days) | **Conditionally** |
| **Zone 5** | 4,642 / 10,727 (43.3%) | 9 blocks | 2,277 windows | 488 packets (10.2 days) | **Conditionally** |

### Assessment
- **Zone 3:** **Completely Unusable** for 48 consecutive 30-minute packets. The probe failed frequently before dying in late June; its longest uninterrupted run was only 16 packets (8 hours).
- **Zones 1, 2, 4, 5:** **Cannot be created unconditionally at arbitrary timestamps**. Over 50% of the overall seasonal timeline consists of dropped gaps. However, they **can be reliably constructed within verified continuous segments** (which provide 1,500 to 2,500 sliding 24-hour windows per zone).

---

## 5. Feature Classification & Data Leakage Prevention

To prevent catastrophic data leakage during model training and production inference, columns are classified into strict usage roles:

### A. Prohibited Features (Severe Data Leakage)

1. **`real_moisture_delta` (DO NOT USE)**
   - *Reason:* Defined as `soil_moisture(t + 24h) - soil_moisture(t)`. It mathematically embeds the future target value into the input space.
2. **`water_vol_to_24h` (DO NOT USE for predictive models)**
   - *Reason:* Represents the total irrigation volume applied over the **future** 24 hours. In production inference, future irrigation is unknown. Using this feature constitutes forward-looking lookahead leakage unless modeling a strictly scheduled scenario.

### B. Valid Target Columns

1. **`target_mean_24h` (Recommended Primary Target)**
   - Continuous regression target: Mean volumetric water content over the next 24 hours. Smoothes out transient sensor noise.
2. **`target_point_24h` (Alternative Target)**
   - Exact soil moisture reading 24 hours ahead.

### C. Safe Input Features

1. **Environmental / ERA5 Weather Telemetry (Instantaneous & Lagged)**
   - `weather_temp` (°C)
   - `weather_humidity` (% RH)
   - `weather_rain` (mm)
   - `weather_pressure` (hPa)
   - `weather_wind_speed` (km/h)
   - `weather_radiation` (W/m²)
   - `soil_temperature_0-7cm` (°C)
   - `soil_temperature_7-18cm` (°C)
2. **Past Irrigation Actuator Telemetry**
   - `irrigation_duration_minutes`: Current 10-minute bin actuator runtime.
   - `water_vol_past_4h`: Accumulated water volume applied over the preceding 4 hours (backward-looking; zero leakage).
3. **Current Soil State**
   - `soil_moisture`: Cleaned volumetric water content (0–100%) at time $t$.
   - `ec` (Soil Electrical Conductivity): *Requires filtering* — 16-bit sensor error overflow values exist in Zone 3 (1.70%) and Zone 5 (0.22%). Filter to $[0, 1000]\,\mu\text{S/cm}$.
   - `ph` (Soil pH): *Requires filtering* — out-of-band error codes exist in Zone 3 (1.07%) and Zone 5 (0.28%). Filter to $[3.0, 9.0]$.

### D. Features Requiring Caution

1. **Calendar Features (`month`, `day`, `hour`, `day_period`, `quarter`)**
   - *Risk:* The dataset covers only 7 months of a single growing season (2025). High-capacity models (e.g., Random Forests, Gradient Boosters, MLPs) risk memorizing specific calendar dates rather than learning physical soil hydrology.
   - *Recommendation:* If utilized, encode `hour` and `month` with cyclical trigonometric transforms ($\sin/\cos$) and validate rigorously against temporal holdout splits.
2. **`ts` (Timestamp)**
   - Use strictly as a sequence index; never supply as a raw numeric feature.

---

## 6. Recommendations for Architecture & Downstream Pipeline

1. **Exclude Zone 3** from sequence models requiring multi-hour historical lookbacks due to probe failure and absence of continuous 24h spans.
2. **Segment Filtering:** Form rolling sequences or packets strictly within contiguous sub-intervals where `ts.diff() == 10 min` (or 30 min after aggregation) rather than naively shifting across gaps.
3. **Sensor Range Guards:** Implement sanity assertions on incoming inferences to discard or flag hardware error states on EC ($[0, 1000]$) and pH ($[3, 9]$).
4. **Target Alignment:** Train models to predict `target_mean_24h` using historical weather, past irrigation (`water_vol_past_4h`), and current soil moisture.
