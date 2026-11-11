# Fungal-Risk Modelling Dataset Preparation & Assumptions

**Date Prepared:** November 10, 2026
**Preparation Script:** [`backend/app/ml/prepare_fungal_dataset.py`](file:///c:/Users/Vishal/Downloads/CropNow/backend/app/ml/prepare_fungal_dataset.py)
**Output Table:** [`data/processed/fungal_risk/flax_rust_fungal_risk_dataset.csv`](file:///c:/Users/Vishal/Downloads/CropNow/data/processed/fungal_risk/flax_rust_fungal_risk_dataset.csv) (`70 rows` $\times$ `52 columns`)
**Fold Threshold Metadata:** [`data/processed/fungal_risk/flax_rust_fold_thresholds.json`](file:///c:/Users/Vishal/Downloads/CropNow/data/processed/fungal_risk/flax_rust_fold_thresholds.json)
**Model Training Status:** **NOT TRAINED** (dataset preparation and verification only).

---

## 1. Source Data & Temporal Join Design

- **Disease Observations Source:** `Cleaned_Epi.rds` (`126` total surveys over 2019–2023 across 4 sites: `BT`, `CC`, `GM`, `HM`).
- **Environmental Telemetry Sources:**
  - `cleaned_temp.rds` (2019–2023): 5-min air temperature (`Temp`), relative humidity (`RH`), and dew point (`Dew_point`).
  - `cleaned_weather.rds` (2020–2023): 5-min leaf wetness (`Leaf_Wetness`), rainfall (`Rain`), solar radiation (`Solar_Radiation`), and volumetric soil moisture (`Water_Content`).
- **Lookback Window Join (`[d-7 .. d-1]`):**
  - Each disease observation at `(Site, Year, d = days_since_june_1st)` is joined strictly to its **7 complete preceding calendar days (`d - 7` through `d - 1`)** rather than including survey day `d`.
  - **2019 Excluded (`23` rows):** `cleaned_weather.rds` begins in 2020; 2019 surveys lack leaf wetness, rainfall, solar radiation, and soil moisture.
  - **Complete 7-Day Window Gate (`70 / 103` rows retained in 2020–2023):** Every retained observation has all `7` preceding days with $\ge 270 / 288$ uniform 5-minute bins ($\ge 93.75\%$ daily coverage). `33` early-season rows lacking full 7-day preceding logger deployment were excluded to avoid partial-window bias.
  - **Zero Missing Values:** Across all `70` retained rows and `52` columns, missing values = **`0`** and duplicate rows = **`0`**.

---

## 2. Engineered Features (`38` Predictor Features)

All 5-minute readings were first rounded and aggregated onto a uniform 5-minute grid (normalizing `HM 2020`'s 1-minute temperature logging rate to 5-minute bins) and inner-joined per `(Site, Year, days_since_june_1st, ts5)`:

| Feature Group | Features Included | Ecological / Epidemiological Rationale |
| :--- | :--- | :--- |
| **Temperature (`8` features)** | `temp_mean_7d`, `temp_min_7d`, `temp_max_7d`, `temp_std_7d`, `temp_diurnal_range_mean_7d`, `temp_mean_24h`, `dp_mean_7d`, `dp_depression_mean_7d` | Governs rust spore germination velocity, incubation latency, and night-time condensation risk (`Temp - Dew_point`). |
| **Humidity (`7` features)** | `rh_mean_7d`, `rh_min_7d`, `rh_max_7d`, `rh_std_7d`, `rh_mean_24h`, `rh_ge_80_hours_7d`, `rh_ge_90_hours_7d` | High RH duration ($\ge 80\%$, $\ge 90\%$) enables urediniospore germination and prevents desiccation. |
| **Leaf Wetness (`7` features)** | `lw_mean_7d`, `lw_max_7d`, `lw_mean_24h`, `lw_wet_hours_7d`, `lw_heavy_wet_hours_7d`, `lw_wet_hours_24h`, `favorable_infection_hours_7d` | Free leaf moisture is required for appressorium formation; `favorable_infection_hours_7d` counts hours where `Leaf_Wetness > 0` AND `5.0 <= Temp <= 25.0°C`. |
| **Rainfall, Soil & Solar (`7` features)** | `rain_total_7d`, `rain_max_daily_7d`, `rain_days_count_7d`, `rain_total_24h`, `soil_moisture_mean_7d`, `soil_moisture_mean_24h`, `solar_radiation_mean_7d` | Rain splash drives spore dispersal; soil water content reflects microclimate moisture; high UV/solar reduces spore viability. |
| **7-Day Environmental Trends (`9` features)** | `temp_trend_slope_7d`, `rh_trend_slope_7d`, `lw_hours_trend_slope_7d`, `rain_trend_slope_7d`, `soil_moisture_trend_slope_7d` *(OLS slope over days `-7..-1`)*; plus `temp_delta_3d_vs_4d`, `rh_delta_3d_vs_4d`, `lw_hours_delta_3d_vs_4d`, `rain_delta_3d_vs_4d` *(mean of recent 3d minus prior 4d)* | Captures whether conditions are actively warming/cooling or wetting/drying leading into the survey date. |

---

## 3. Leak-Free `LOW / MEDIUM / HIGH` Label Derivation

To eliminate test-set threshold leakage, **no global dataset quantile column is used**. Continuous targets (`prevalence`, `prevalence_no_recov`, `new_infections`, `incidence_rate`, `host_count`) are preserved in the table, and categorical risk labels are computed **exclusively inside each CV training fold** via [`assign_fold_risk_labels`](file:///c:/Users/Vishal/Downloads/CropNow/backend/app/ml/prepare_fungal_dataset.py#L84-L110):
- **Rule:** Using training-fold `prevalence` tertiles (`q33.33` and `q66.67`):
  - **`LOW`:** $\text{prevalence} \le q_{33.3,\text{train}}$
  - **`MEDIUM`:** $q_{33.3,\text{train}} < \text{prevalence} \le q_{66.7,\text{train}}$
  - **`HIGH`:** $\text{prevalence} > q_{66.7,\text{train}}$

### Documented Fold Thresholds (Calculated Within Training Folds Only)

#### Leave-One-Year-Out (`risk_label_loyo_oof` — `LOW`: `22`, `MEDIUM`: `23`, `HIGH`: `25`)
| Held-Out Test Year | Train `n` | Test `n` | `q33` Threshold (Train Only) | `q67` Threshold (Train Only) | Held-Out Test Distribution |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **2020** | 57 | 13 | `0.0482` | `0.1523` | `LOW`: 2, `MEDIUM`: 9, `HIGH`: 2 |
| **2021** | 54 | 16 | `0.0642` | `0.1694` | `LOW`: 10, `MEDIUM`: 5, `HIGH`: 1 |
| **2022** | 54 | 16 | `0.0475` | `0.1043` | `LOW`: 3, `MEDIUM`: 1, `HIGH`: 12 |
| **2023** | 45 | 25 | `0.0493` | `0.1275` | `LOW`: 7, `MEDIUM`: 8, `HIGH`: 10 |

#### Leave-One-Site-Out (`risk_label_loso_oof` — `LOW`: `23`, `MEDIUM`: `19`, `HIGH`: `28`)
| Held-Out Test Site | Train `n` | Test `n` | `q33` Threshold (Train Only) | `q67` Threshold (Train Only) | Held-Out Test Distribution |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **BT** | 48 | 22 | `0.0485` | `0.1295` | `LOW`: 5, `MEDIUM`: 9, `HIGH`: 8 |
| **CC** | 51 | 19 | `0.0525` | `0.1179` | `LOW`: 7, `MEDIUM`: 3, `HIGH`: 9 |
| **GM** | 51 | 19 | `0.0431` | `0.1140` | `LOW`: 2, `MEDIUM`: 6, `HIGH`: 11 |
| **HM** | 60 | 10 | `0.0688` | `0.1639` | `LOW`: 9, `MEDIUM`: 1, `HIGH`: 0 |

---

## 4. Documented Modelling Assumptions & Constraints

1. **Causal Strictness (`[d-7 .. d-1]`):** All environmental features stop at day `d - 1` (`23:55`) prior to `Disease_date` `d` so survey-day afternoon weather cannot leak into the prediction.
2. **Wind/Gust Omission:** `ws`, `wd`, and `Gust_Speed` are omitted because the High Meadow (`HM`) anemometer was offline for all `46` days of `2023` (`13,498` missing 5-min observations). Keeping wind features would either drop `HM 2023` or force cross-site imputation.
3. **Single-Point Sensor NA Recovery:** `cleaned_temp.rds` contained `3` isolated single-reading NAs (`CC` 2021 day 16, `CC` 2022 days 19 & 22) that caused R's original `cleaned_daily_temp.rds` to mark those entire days `NA`. Resampling directly from 5-minute data (`>= 270 / 288` valid bins) cleanly recovers those complete days.
4. **Favorable Temperature Window (`[5°C, 25°C]`):** `favorable_infection_hours_7d` uses subalpine *Melampsora lini* germination temperatures (`5–25°C`) alongside `Leaf_Wetness > 0`. When adapting features to warmer NEST crops (e.g. chilli), temperature interaction thresholds must be recalibrated.
5. **Prevalence Shift Across Years & Elevation:** Notice `2021` had cooler/drier test prevalence (`LOW: 10`), `2022` had high outbreak prevalence (`HIGH: 12`), and high-elevation `HM` (`3,410 m`) exhibits markedly lower rust prevalence (`LOW: 9`) than `GM`/`CC`/`BT`. Evaluating models with both **LOYO** and **LOSO** CV prevents optimistic accuracy estimates.
