# Fungal & Pest Model Compatibility Audit Against 21-Field NEST Schema

**Date of Audit:** November 10, 2026
**Reference Schema Audited:** [`data/company_sample/nest_sample.json`](file:///c:/Users/Vishal/Downloads/CropNow/data/company_sample/nest_sample.json) (`21` NEST telemetry fields, `30-minute` packet cadence)
**Models Audited:**
1. **Fungal-Risk Classifier (`XGBoost`, 38 features):** [`backend/app/ml/models_fungal_risk/best_fungal_model.joblib`](file:///c:/Users/Vishal/Downloads/CropNow/backend/app/ml/models_fungal_risk/best_fungal_model.joblib)
2. **Pest-Risk Classifier (`LogisticRegression`, 26 features):** [`backend/app/ml/models_pest_risk/best_pest_model.joblib`](file:///c:/Users/Vishal/Downloads/CropNow/backend/app/ml/models_pest_risk/best_pest_model.joblib)

**Constraint Adherence:** Zero models modified; zero API endpoints implemented.

---

## 1. Executive Summary

Both new classifiers were built on **multi-day / multi-week temporal aggregations** rather than instantaneous single-packet snapshots. Comparing their feature specifications against the **21-field NEST telemetry contract** (`30-minute` cadence) reveals:

| Audit Dimension | Fungal-Risk Model (`XGBoost`, 38 features) | Pest-Risk Model (`LogisticRegression`, 26 features) | Contrast with Existing Soil-Moisture Models |
| :--- | :--- | :--- | :--- |
| **Directly usable from 1 single NEST packet** | **0 / 38** *(requires temporal windowing)* | **0 / 26** *(requires weekly windowing)* | 0 / 18 (requires 48 packets = 24h) |
| **Base NEST sensors matching 1:1 (no unit conversion)** | `temperature`, `humidity`, `leaf_wetness`, `solar_radiation`, `rainfall` *(with cadence normalization)* | `temperature`, `humidity`, `wind_speed`, `wind_direction`, `rainfall` *(after weekly aggregation)* | Same |
| **Features requiring unit conversion / mathematical derivation** | **5 features:** `soil_moisture_*` (`% -> 0–1` fraction), `dp_mean_7d` & `dp_depression_mean_7d` (Magnus-Tetens from `temperature` + `humidity`) | **8+ features:** `GWETTOP` (`0–1` saturation index vs `0–100%` volumetric SM), `wd_sin`/`wd_cos`, `PRECTOTCORR` (`mm/day` vs incremental mm), intra-week spreads | `soil_moisture` avg (`(SM1+SM2)/2`) |
| **Completely unavailable sensors in NEST** | **0** *(if `dew_point` is derived via Magnus-Tetens; **2** if only raw sensors counted)* | **0** *(all 6 base signals map to NEST sensors, pending `GWETTOP` calibration)* | `ec`, `ph`, `irrigation_duration`, `water_vol_past_4h`, `soil_temp_7-18cm` |
| **Minimum Historical Packets Required (30-min NEST)** | **336 packets** (**7 days** $\times$ 48 pkts/day) | **1,344 packets** (**28 days / 4 weeks** $\times$ 48 pkts/day) | **48 packets** (**24 hours**) |
| **Compatibility with Current 48-Packet NEST Window** | **Incompatible** (short by `288` packets / 6 days) | **Incompatible** (short by `1,296` packets / 27 days) | Compatible (`48` packets) |

---

## 2. Reference: The 21-Field NEST Schema (`30-Minute` Cadence)

From [`nest_sample.json`](file:///c:/Users/Vishal/Downloads/CropNow/data/company_sample/nest_sample.json):
- **Meteorological & Canopy (`7`):** `temperature` (°C), `humidity` (% RH), `leaf_wetness` (0–100), `rainfall` (mm), `solar_radiation` (W/m²), `pressure` (kPa), `wind_speed` (m/s), `wind_direction` (deg 0–360) *(8 fields)*.
- **Soil (`3`):** `soil_moisture_1` (%), `soil_moisture_2` (%), `soil_temperature` (°C).
- **Air Quality (`4`):** `co` (ppm), `no2` (ppm), `o3` (ppm), `pm2_5` (µg/m³).
- **Calendar & Device Metadata (`6`):** `crop_id`, `device_id`, `farm_id`, `month`, `day`, `hour`.

*(Note: Air quality fields `co`, `no2`, `o3`, `pm2_5`, `pressure`, and `soil_temperature` are unused by both the fungal and pest classifiers.)*

---

## 3. Model 1 Audit: Fungal-Risk Classifier (`XGBoost`, 38 Features)

Trained on 7-day preceding microclimate summaries (`[d-7 .. d-1]`) from the subalpine flax rust (*Melampsora lini*) dataset (`38` pure environmental features).

### 3.1 Feature Compatibility Matrix (`38` Features)

#### A. Directly Available Base Signals (Requires 7-Day Aggregation Only — `31` features)
No single-packet feature is directly usable, as all 38 features summarize 24h or 7-day history. However, **31 features** map directly from NEST sensors without unit conversion once buffered over 7 days (`336` packets at 30-min cadence):

| NEST Source Sensor | Fungal Model Features (`31`) | Required Computation Over `[d-7 .. d-1]` |
| :--- | :--- | :--- |
| `temperature` (°C) | `temp_mean_7d`, `temp_min_7d`, `temp_max_7d`, `temp_std_7d`, `temp_diurnal_range_mean_7d`, `temp_mean_24h`, `temp_trend_slope_7d`, `temp_delta_3d_vs_4d` *(8)* | Daily/7d min, mean, max, std, OLS slope over 7 daily means, recent 3d minus prior 4d |
| `humidity` (% RH) | `rh_mean_7d`, `rh_min_7d`, `rh_max_7d`, `rh_std_7d`, `rh_mean_24h`, `rh_ge_80_hours_7d`, `rh_ge_90_hours_7d`, `rh_trend_slope_7d`, `rh_delta_3d_vs_4d` *(9)* | Daily/7d stats; duration hours counted as `(packets meeting threshold) × 0.5h` |
| `leaf_wetness` (`0–100`) | `lw_mean_7d`, `lw_max_7d`, `lw_mean_24h`, `lw_wet_hours_7d`, `lw_heavy_wet_hours_7d`, `lw_wet_hours_24h`, `lw_hours_trend_slope_7d`, `lw_hours_delta_3d_vs_4d`, `favorable_infection_hours_7d` *(9)* | Wetness hours = `sum(leaf_wetness > 0) × 0.5h`; heavy = `>= 20`; favorable = `leaf_wetness > 0 & 5 <= temp <= 25` |
| `rainfall` (mm) | `rain_total_7d`, `rain_max_daily_7d`, `rain_days_count_7d`, `rain_total_24h`, `rain_trend_slope_7d`, `rain_delta_3d_vs_4d` *(6)* | Daily/7d sums, rainy day count (`> 0.2 mm`), slope & 3d-vs-4d delta *(confirm NEST `rainfall` is per-packet incremental mm)* |
| `solar_radiation` (W/m²) | `solar_radiation_mean_7d` *(1)* | 7-day arithmetic mean |

*(Total above = 8 + 9 + 9 + 6 + 1 = **33** features directly from 5 sensors without unit scaling.)*

#### B. Features Requiring Conversion or Formula Derivation (`5` features)
| Fungal Model Feature(s) | Underlying NEST Sensor(s) | Required Conversion / Formula | Risk / Calibration Caveat |
| :--- | :--- | :--- | :--- |
| `soil_moisture_mean_7d`, `soil_moisture_mean_24h`, `soil_moisture_trend_slope_7d` *(3)* | `soil_moisture_1`, `soil_moisture_2` (`0–100%`) | `((soil_moisture_1 + soil_moisture_2) / 2.0) / 100.0` (percentage $\to$ fractional `0–1` `Water_Content`) | Training range is `0.009–0.331`; raw percentages (`22–31`) would be 100$\times$ out-of-scale without `/ 100.0`. Note probe dielectric calibration differences. |
| `dp_mean_7d`, `dp_depression_mean_7d` *(2)* | `temperature` (°C), `humidity` (% RH) | **Derived via Magnus-Tetens formula:** compute instantaneous `dew_point` per packet, then `dp_depression = temperature - dew_point`, then aggregate 7-day mean. | Not directly emitted by NEST hardware, so must be synthesized mathematically in preprocessing. |

#### C. Unavailable Features & Domain Gaps
- **Missing Hardware Sensors:** **None** if Magnus-Tetens dew point derivation is permitted (**2 features unavailable** if strictly restricted to direct hardware fields without thermodynamic equations).
- **Cadence Discrepancy:** Training features used `5-minute` packets (`5/60 h` granularity) whereas NEST emits `30-minute` packets (`0.5 h` granularity).
- **Severe Climate Domain Shift:** Trained in Colorado subalpine meadows (`temp_mean_7d`: `10.2–18.9°C`, `temp_max_7d` is the #1 XGBoost split feature). Lowland tropical NEST deployments (`25–35°C`) fall completely outside the training temperature domain, and `favorable_infection_hours_7d` hardcodes `[5, 25]°C`.

### 3.2 Minimum Historical Packets Required (Fungal Model)
- **Minimum Required:** **`336` consecutive 30-minute packets** (**7 full days** = `7 × 48` packets).
- **Justification:** Features like `temp_trend_slope_7d`, `lw_hours_delta_3d_vs_4d` (comparing recent 3 days vs prior 4 days), and `rain_days_count_7d` require 7 distinct 24-hour blocks. A single 24-hour (`48`-packet) NEST buffer cannot populate this feature vector.

---

## 4. Model 2 Audit: Pest-Risk Classifier (`LogisticRegression`, 26 Features)

Trained on weekly NASA POWER weather summaries and multi-week lags/trends (`[t-3 .. t]` weeks) for Rice Black Bug (`RBB`) abundance (`26` features).

### 4.1 Feature Compatibility Matrix (`26` Features)

#### A. Directly Available Base Signals Requiring Weekly & Multi-Week Aggregation (`18` features)
No feature is directly available from a single NEST packet. **18 features** require temporal aggregation across 7-day (weekly) windows over 4 consecutive weeks (`1,344` packets at 30-min cadence), with matching physical units:

| NEST Source Sensor | Pest Model Features (`18`) | Required Aggregation Across 4 Weekly Bins (`w-3 .. w`) |
| :--- | :--- | :--- |
| `temperature` (°C) | `T2M`, `T2M_MIN`, `T2M_MAX`, `temp_diurnal_range`, `t2m_lag_1w`, `t2m_mean_4w`, `t2m_trend_4w` *(7)* | Weekly mean/min/max, `T2M_MAX - T2M_MIN`, `1-week` lag, `4-week` rolling mean & OLS slope |
| `humidity` (% RH) | `RH2M`, `rh2m_lag_1w`, `rh2m_mean_4w`, `rh2m_trend_4w` *(4)* | Weekly mean, `1-week` lag, `4-week` rolling mean & OLS slope |
| `wind_speed` (m/s) | `WS2M`, `WS2M_MIN`, `WS2M_MAX`, `wind_gust_spread`, `ws2m_lag_1w` *(5)* | Weekly mean/min/max, `WS2M_MAX - WS2M_MIN`, `1-week` lag |
| `wind_direction` (deg) | `WD2M` *(1)* | Weekly circular vector mean `atan2(mean(sin), mean(cos))` wrapped to `[0, 360)` |
| `rainfall` (mm) | *(See conversion note below for `PRECTOTCORR`, `rain_lag_1w`, `rain_sum_4w`)* | — |

#### B. Features Requiring Conversion + Multi-Week Aggregation (`9` features)
| Pest Model Feature(s) | Underlying NEST Sensor(s) | Required Conversion / Calibration | Risk / Calibration Caveat |
| :--- | :--- | :--- | :--- |
| `GWETTOP`, `gwettop_lag_1w`, `gwettop_mean_4w`, `gwettop_trend_4w` *(4)* | `soil_moisture_1`, `soil_moisture_2` (`0–100%`) | Normalize volumetric `%` to **saturation fraction (`0–1`)**: e.g., `((SM1 + SM2)/2) / SP` (where `SP` = crop/soil saturation point, e.g. `45%` for Chilli $\to$ `26.5/45 = 0.59`), then aggregate weekly & compute 4w mean/slope. | **Critical Scale Trap:** Simply dividing by `100` (`0.22–0.31`) places inputs far below NASA POWER `GWETTOP` (`0.56–0.96`). Because `LogisticRegression` uses `StandardScaler` and `gwettop_trend_4w` (`0.307`) & `gwettop_lag_1w` (`0.205`) are top drivers, uncalibrated units will severely bias predictions. |
| `PRECTOTCORR`, `rain_lag_1w`, `rain_sum_4w` *(3)* | `rainfall` (mm) | Aggregate weekly NEST rainfall and convert to **daily precipitation rate (`mm/day`)**: `weekly_rain_sum / 7.0` (matches NASA POWER `PRECTOTCORR` mean `7.35 mm/day`, max `112.08`), then compute 1w lag and 4w sum. | Summing 30-min packets over a week without `/ 7.0` inflates `PRECTOTCORR` and `rain_sum_4w` by ~`7×` relative to the training scaler! |
| `wd_sin`, `wd_cos` *(2)* | `wind_direction` (deg `0–360`) | Compute weekly circular mean `WD2M`, then `sin(deg2rad(WD2M))` and `cos(deg2rad(WD2M))`. | Standard trigonometric conversion; safe once weekly mean direction is computed circularly. |

#### C. Unavailable Features
- **Missing Hardware Sensors:** **None** — every underlying physical quantity (`temperature`, `humidity`, `rainfall`, `wind_speed`, `wind_direction`, `soil_moisture`) exists in the 21-field NEST payload, **but** `GWETTOP` requires crop/soil saturation point normalization and `PRECTOTCORR` requires daily-rate normalization.
- **Note on High-Value Unused NEST Sensors:** The pest model does not utilize `leaf_wetness` or `solar_radiation`.

### 4.2 Minimum Historical Packets Required (Pest Model)
- **Minimum Required:** **`1,344` consecutive 30-minute packets** (**4 full weeks = 28 days** = `4 × 7 × 48` packets).
- **Justification:** `t2m_mean_4w`, `rh2m_mean_4w`, `rain_sum_4w`, `gwettop_mean_4w`, `t2m_trend_4w`, `rh2m_trend_4w`, and `gwettop_trend_4w` require 4 complete weekly averages (`[t-3, t-2, t-1, t]`). A 48-packet (`24h`) or even 336-packet (`7d`) buffer cannot compute 4-week rolling trends.

---

## 5. Architectural Gap Summary for Future API Integration

If these models are later integrated into the FastAPI backend alongside NEST telemetry:
1. **Packet History Window Mismatch:** Existing NEST endpoints (`/predict`, `/stability/analyze`) validate **48 consecutive packets (24 hours)**.
   - Fungal-risk inference requires **7× more history (`336` packets / 7 days)**.
   - Pest-risk inference requires **28× more history (`1,344` packets / 28 days)**.
2. **Mandatory Telemetry Adapters:**
   - **Fungal Adapter:** Resample/aggregate `336` packets into 7 daily blocks, derive `dew_point` via Magnus-Tetens, scale soil moisture `/ 100.0`, and scale duration counters by `0.5 h/packet`. *(Must address subalpine-to-tropical temperature domain shift before production use.)*
   - **Pest Adapter:** Aggregate `1,344` packets into 4 weekly blocks, convert rainfall to `mm/day` (`weekly_sum / 7.0`), normalize soil moisture to `GWETTOP` saturation ratio (`SM_avg / SP`), compute circular wind mean + `sin`/`cos`, and compute 4-week slopes.
