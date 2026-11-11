# Fungal Flax Rust Dataset Audit Report

**Date of Audit:** November 10, 2026
**Target Directory Requested:** `data/external/fungal_flax_rust/`
**Actual Path on Disk:** `data/external/pest/fungal_flax_rust/` *(nested inside `pest/` rather than directly under `data/external/`)*
**Source Dataset:** Multi-year Flax Rust (*Melampsora lini*) Epidemiological & 5-Minute Microclimate Study (Dryad DOI: `10.5061/dryad.98sf7m0tc`, RMBL Colorado subalpine sites: `BT`, `CC`, `GM`, `HM`)
**Constraint Adherence:** Datasets untouched; zero models trained or modified.

---

## 1. Executive Summary

Unlike the unlabeled `Grape_Disease_Dataset.csv` and the binary rice pest datasets (`RBB`/`WSB`), the **`fungal_flax_rust/`** dataset contains **both genuine fungal disease observations AND continuous 5-minute microclimate + leaf wetness + rainfall + soil moisture telemetry** across multiple years (2019–2023) and 4 elevation-gradient sites (`BT`, `CC`, `GM`, `HM`).

### Verdict: Supports Supervised Fungal-Risk Modeling (with Sample-Size & Domain Caveats)
- **Supervised Target Availability:** Yes — provides continuous site-level epidemic metrics (`new_infections`, `prevalence`, `prevalence_noRecov` in `Cleaned_Epi.rds`), plant-level health status (`H`/`D` across 2020–2023 in `Plant_locations.xlsx`), and 6-tier fungal tissue severity (`Disease Class (0-5)` in 2023 `Epidemiology.xlsx` and `2023_Epidemiology_Final_Total_Surveys.xlsx`).
- **Environmental Feature Alignment with NEST:** High — includes `Temp`, `RH`, `Dew_point`, `Leaf_Wetness`, `Rain`, `Solar_Radiation`, `ws`, `wd`, `Gust_Speed`, and `Water_Content` (soil moisture) at **5-minute resolution** (easily resampled to 30-minute NEST cadences).
- **Primary Constraints:**
  1. **Small Site-Survey Sample Size (`n = 75` full-feature rows; `n = 73` with full 7-day weather lookback):** Because `cleaned_weather.rds` (Leaf Wetness, Rain, Soil Moisture) starts in **2020**, the 2019 epidemiology surveys (`n = 23`) lack leaf wetness and rainfall. The `n = 73–75` complete site-survey samples (2020–2023) favor **low-variance regularized tabular/tree models over engineered hydrological features with Leave-One-Year-Out / Leave-One-Site-Out CV**, rather than deep neural sequence models.
  2. **Crop / Climate Domain Shift:** Collected in high-elevation subalpine Colorado meadows (`2,440–3,410 m`, mean air temp `14.0°C`) on wild flax (*Linum lewisii*), whereas NEST targets lowland/tropical crops like chilli (`~25–31°C`).

---

## 2. File Inventory

| File Name | Size | Format | Dimensions (Rows $\times$ Cols) | Role in Study |
| :--- | :---: | :---: | :---: | :--- |
| [`Cleaned_Epi.rds`](file:///c:/Users/Vishal/Downloads/CropNow/data/external/pest/fungal_flax_rust/Cleaned_Epi.rds) | 3,179 B | R Data | `126` $\times$ `9` | Summarized site-date epidemic prevalence & new infections (2019–2023) |
| [`cleaned_temp.rds`](file:///c:/Users/Vishal/Downloads/CropNow/data/external/pest/fungal_flax_rust/cleaned_temp.rds) | 3.15 MB | R Data | `291,264` $\times$ `7` | 5-min (and 1-min at HM 2020) air temp, RH, dew point (2019–2023) |
| [`cleaned_weather.rds`](file:///c:/Users/Vishal/Downloads/CropNow/data/external/pest/fungal_flax_rust/cleaned_weather.rds) | 1.40 MB | R Data | `191,477` $\times$ `11` | 5-min leaf wetness, rain, solar, wind, gust, soil water content (2020–2023) |
| [`cleaned_daily_temp.rds`](file:///c:/Users/Vishal/Downloads/CropNow/data/external/pest/fungal_flax_rust/cleaned_daily_temp.rds) | 37,075 B | R Data | `902` $\times$ `12` | Daily min/mean/max of Temp, RH, Dew Point (2019–2023) |
| [`cleaned_daily_weather.rds`](file:///c:/Users/Vishal/Downloads/CropNow/data/external/pest/fungal_flax_rust/cleaned_daily_weather.rds) | 32,396 B | R Data | `648` $\times$ `24` | Daily min/mean/max of Leaf Wetness, Rain, Solar, Wind, Soil Moisture (2020–2023) |
| [`Epidemiology.xlsx`](file:///c:/Users/Vishal/Downloads/CropNow/data/external/pest/fungal_flax_rust/Epidemiology.xlsx) | 460 KB | Excel | `6,293` $\times$ `14` *(+17 blank cols)* | Raw plant-level epidemiological surveys (2019–2023) |
| [`2023_Epidemiology_Final_Total_Surveys.xlsx`](file:///c:/Users/Vishal/Downloads/CropNow/data/external/pest/fungal_flax_rust/2023_Epidemiology_Final_Total_Surveys.xlsx) | 17.9 KB | Excel | `376` $\times$ `9` | 2023 end-of-year severity & newly diseased surveys |
| [`Plant_locations.xlsx`](file:///c:/Users/Vishal/Downloads/CropNow/data/external/pest/fungal_flax_rust/Plant_locations.xlsx) | 845 KB | Excel | 3 sheets (`10,485` + `87` + `385`) | Spatial `(X,Y,x,y)` plant status (`H`, `D`, `X`) & height (2020–2023) |
| [`Population_density.xlsx`](file:///c:/Users/Vishal/Downloads/CropNow/data/external/pest/fungal_flax_rust/Population_density.xlsx) | 109 KB | Excel | `3,708` $\times$ `10` | 1m $\times$ 1m grid counts `#H`, `#D`, `# Seedlings` (2018–2020) |
| [`README.md`](file:///c:/Users/Vishal/Downloads/CropNow/data/external/pest/fungal_flax_rust/README.md) | 9.9 KB | Markdown | 176 lines | Study metadata, site coordinates, variable definitions |

---

## 3. Disease Labels & Class Distributions

### 3.1 Site-Level Epidemic Time Series (`Cleaned_Epi.rds` — `126 rows`)
Cleaned summary of fungal epidemic progression across the 4 primary sites (`BT`: 37, `CC`: 36, `GM`: 34, `HM`: 19) over 2019–2023 (`2019`: 23, `2020`: 26, `2021`: 22, `2022`: 25, `2023`: 30; note `HM` has 0 rows in 2019):
- **`new_infections` (New fungal infections per survey):**
  - Range: `0` to `274` (Mean: `29.64`, Median: `12.0`, `Q25`: `5.25`, `Q75`: `27.0`, `Q90`: `68.5`).
  - Only `4 / 126` rows have `0` new infections.
- **`prevalence` (Recovery-adjusted infected host fraction):**
  - Range: `0.00178` (`0.18%`) to `0.59242` (`59.24%`).
  - Quantiles: `Q25 = 2.04%`, `Q33.3 = 3.23%`, `Median = 5.59%`, `Q66.7 = 9.93%`, `Q75 = 14.31%`, `Q90 = 30.71%`.
- **`prevalence_noRecov` (Cumulative prevalence without recovery):**
  - Range: `0.18%` to `59.44%` (Median: `7.19%`, Mean: `12.23%`).
- **`count` (Total host plants at site):** `415` to `1,287` (`Mean = 794.2`).

### 3.2 Plant-Level Severity Labels (`Disease Class (0-5)` & `status`)
Per `README.md`, `Disease Class (0-5)` measures visible infected tissue percentage (`0`: `0%`, `1`: `0–20%`, `2`: `20–40%`, `3`: `40–60%`, `4`: `60–80%`, `5`: `80–100%`) and was systematically introduced in **2023**:

| Dataset | Target Column | Distribution / Non-Null Counts | Notes / Caveats |
| :--- | :--- | :--- | :--- |
| **`Epidemiology.xlsx`** (`6,293` rows) | `Disease Class (0-5)` | `0`: `311`, `1`: `855`, `2`: `539`, `3`: `56`, `4`: `11`, `5`: `7`, `'looks bad'`: `1`, `Null`: **`4,513 (71.7%)`** | `1,776` of `1,780` scored rows are from **2023** (`4` in 2022). Includes `1,562` `# Diseased Stems` records and 15 sub-site names (`CC3`, `RG1`, `MB1`, etc.). |
| **`2023_Epidemiology_Final_Total_Surveys.xlsx`** (`376` rows) | `Disease Class (0-5)` & `New?` | `0`: `47`, `1`: `154`, `2`: `148`, `3`: `16`, `4`: `1`, `Null`: `10` (`2.7%`); `New? = 1`: `20` | End-of-year 2023 survey (`2023-08-07`, `08-08`, `08-10`). Strongly skewed toward classes `1–2`. |
| **`Plant_locations.xlsx`** (`10,957` rows across 3 sheets) | `status` & `Class(0-5)` | Combined `status`: **`H`: `9,631`**, **`D`: `634`**, `dead`/`X`/`lost`: `29`, `Null`: `661`; `Class(0-5)` populated on `562` rows (2023). | Covers `2020–2023` (`Sheet1`), `2023` (`Juliana added`, `Peyton added`). Severe healthy/diseased class imbalance (`~93.6% H` vs `~6.2% D`). |
| **`Population_density.xlsx`** (`3,708` rows) | `#H`, `#D`, `# Seedlings` | `2,798` non-null grid cells (`909` trailing blank rows), `2018–2020`. | Spatial host density in 1m $\times$ 1m grid cells; `date` is missing on `56.9%` of rows. |

---

## 4. Environmental Features & Data Quality (Missing Values & Outliers)

### 4.1 High-Frequency & Daily Microclimate Files

| File | Span | Variables & Ranges | Missing Values (NAs) | Duplicates |
| :--- | :---: | :--- | :--- | :---: |
| **`cleaned_temp.rds`** (`291,264` rows) | 2019–2023 | • `Temp`: `-5.79` to `38.23°C` (Mean `13.95°C`)<br>• `RH`: `7.37` to `100.00%` (Mean `56.37%`)<br>• `Dew_point`: `-16.73` to `20.42°C` (Mean `3.65°C`) | **`5` rows (`0.002%`)** in `Temp`, `RH`, `Dew_point` (all at site `CC`) | `0` exact row dups; `0` `(Site, times)` dups |
| **`cleaned_weather.rds`** (`191,477` rows) | **2020–2023** *(no 2019)* | • `Leaf_Wetness`: `0.0` to `100.0` (Mean `18.72`, Median `0.0`, P75 `10.0`)<br>• `Rain`: `0.0` to `4.20` mm/step (Mean `0.00542`)<br>• `Solar_Radiation`: `0.6` to `1,276.9 W/m²`<br>• `Water_Content` (soil): `-0.0086` to `0.3909` (Mean `0.1838`)<br>• `ws`: `0.0` to `8.02` | **`13,498` rows (`7.05%`)** in `ws`, `wd`, `Gust_Speed` (entire `HM 2023` wind sensor offline). **0 NAs** in `Leaf_Wetness`, `Rain`, `Solar_Radiation`, `Water_Content`. | `0` exact row dups; `0` `(Site, times)` dups |
| **`cleaned_daily_temp.rds`** (`902` rows) | 2019–2023 | Daily `min`/`mean`/`max` of `temp`, `rh`, `dp` | **`5` rows (`0.55%`)** (propagated from the 5 single-point `CC` NAs in `cleaned_temp.rds`) | `0` dups on `(Site, Year, days_since_june_1st)` |
| **`cleaned_daily_weather.rds`** (`648` rows) | **2020–2023** *(no 2019)* | Daily `min`/`mean`/`max` of `leaf_wetness`, `rain`, `solar_radiation`, `ws`, `wd`, `gust_speed`, `water_content` | **`46` rows (`7.10%`)** in wind/gust (`HM 2023`). **0 NAs** in leaf wetness, rain, solar, soil water content. | `0` dups on `(Site, Year, days_since_june_1st)` |

### 4.2 Excel Data Hygiene Issues
All 4 Excel workbooks contain blank trailing/formatting rows (`Epidemiology.xlsx`: `886` blank rows [`14.1%`]; `Plant_locations.xlsx`: `567` blank rows [`5.4%`]; `Population_density.xlsx`: `909` blank rows [`24.5%`]; `2023_Epidemiology`: `3` blank rows [`0.8%`]), Excel serial date numbers (e.g. `45145` = `2023-08-07`), and minor text anomalies (e.g. `'looks bad'` in `Disease Class (0-5)` in `Epidemiology.xlsx`, `',18'` in `x` coordinate of `Peyton added`).

---

## 5. Dates, Temporal Alignment & Join Keys

### 5.1 Temporal Integrity of Weather Telemetry
- **`cleaned_weather.rds` (2020–2023):** Strictly monotonic per `(Site, Year)`, **0 negative/zero deltas**, **`99.999%` exact 5-minute cadence** (`191,459 / 191,461` intervals = `5 min`, with only `2` intervals = `10 min`). Supports **clean resampling to 30-minute NEST intervals**!
- **`cleaned_temp.rds` (2019–2023):** Strictly monotonic per `(Site, Year)`, **0 negative/zero deltas**, **0 gaps > 5.5 minutes** (`259,564` transitions at `5 min`, plus `31,672` transitions at `1 min` during `HM 2020`).
- **Seasonal Window:** Telemetry is collected during the summer growing/epidemic season (`days_since_june_1st` spanning days `9–90`, i.e., mid-June through late August each year).

### 5.2 Join Keys Between Disease Observations & Weather Data

| Source Table | Target Table | Join Key(s) | Exact Match Rate / Notes |
| :--- | :--- | :--- | :--- |
| **`Cleaned_Epi.rds`** (`126` rows) | **`cleaned_daily_temp.rds`** (`902` rows) | `(Site, Year, days_since_june1st)` $\leftrightarrow$ `(Site, Year, days_since_june_1st)` | **`101 / 126` (`80.2%`)** same-day match; **`93 / 126` (`73.8%`)** have complete **7-day Temp/RH lookback** (`d-6..d`). *(Note column spelling difference: `days_since_june1st` vs `days_since_june_1st`.)* |
| **`Cleaned_Epi.rds`** (`126` rows) | **`cleaned_daily_weather.rds`** (`648` rows) | `(Site, Year, days_since_june1st)` $\leftrightarrow$ `(Site, Year, days_since_june_1st)` | **`77 / 126` (`61.1%` overall; `77 / 103 = 74.8%` in 2020–2023)** same-day match. |
| **`Cleaned_Epi.rds`** (`126` rows) | **Both `daily_temp` + `daily_weather`** | `(Site, Year, days_since_june1st)` | **`75 / 126` (`59.5%` overall; `75 / 103 = 72.8%` in 2020–2023)** match both same-day; **`73 / 103` (`70.9%`)** have complete **7-day multi-sensor weather history** (`2020–2023`: `BT`=24, `CC`=22, `GM`=19, `HM`=10). |
| **`Cleaned_Epi.rds`** | **5-min `cleaned_temp` & `cleaned_weather`** | `(Site, Year, as.Date(Disease_date))` $\leftrightarrow$ `(Site, Year, as.Date(times))` | Same-day and multi-day 5-min/30-min lag windows directly extractable by filtering `times <= Disease_date`. |
| **`Epidemiology.xlsx` / `Plant_locations.xlsx`** | **Daily or 5-min weather RDS** | `(Site, Year, Date)` $\leftrightarrow$ `(Site, Year, as.Date(times))` | Joins cleanly for the 4 core sites (`BT`, `CC`, `GM`, `HM`); `Epidemiology.xlsx` sub-sites (`CC3`, `RG1`, `MB1`, etc.) have no matching weather stations. |

---

## 6. Supervised Fungal-Risk Modeling Feasibility

**Overall Finding: YES — Supports Supervised Fungal-Risk Modeling (Best Structured as Tabular Multi-Day Hydrological Feature Modeling)**

| Modeling Formulation | Feasibility | Target Definition (`LOW / MEDIUM / HIGH`) | Key Strengths & Constraints |
| :--- | :---: | :--- | :--- |
| **1. Site-Level Fungal Outbreak / Infection Rate Risk (`Cleaned_Epi.rds` + Weather)** | **Recommended** | Discretize weekly/survey `new_infections / count` (incidence rate) or `prevalence` into `LOW / MEDIUM / HIGH` tiers (e.g., prevalence tertiles `<3.2%`, `3.2–9.9%`, `>9.9%`, or `new_infections` quartiles). | • Direct empirical pairing between real fungal infection surges and exact preceding `Leaf_Wetness`, `Rain`, `Temp`, `RH`, `Dew_point`, and soil `Water_Content`.<br>• **Constraint:** Only `n = 73–75` complete 2020–2023 multi-sensor rows (`n = 93–101` if temp/RH-only from 2019). Requires regularized shallow models (Logistic Regression, Random Forest, shallow XGBoost) with **Leave-One-Year-Out** or **Leave-One-Site-Out CV**. |
| **2. Plant-Level Disease Severity (`2023 Epidemiology` `Class 0-5`)** | **Limited** | Group `Disease Class (0-5)`: `0` (Low/Healthy), `1–2` (Medium), `3–5` (High). | • Over `2,100` scored plants in 2023 (`Epidemiology.xlsx` + `2023_Final_Surveys`), **but** all plants surveyed at the same `(Site, Date)` share identical macro-weather features, and `Class 3–5` is rare (`~4%`). |
| **3. Plant-Level Binary Infection Status (`Plant_locations.xlsx` `H` vs `D`)** | **Secondary** | Binary (`H` vs `D`) across `2020–2023` (`n = 10,265` H/D rows), not natively 3-class risk. | • Heavily imbalanced (`6.2% D`) and site-date weather is shared across hundreds of plants per survey date. |

### Alignment Notes for NEST Platform Integration
1. **Compatible Sensors:** Every core NEST fungal-risk driver (`temperature`, `humidity`, `leaf_wetness`, `rainfall`, `solar_radiation`, `wind_speed`, `soil_moisture_1`) exists at **5-minute resolution** in `cleaned_temp.rds` + `cleaned_weather.rds` and can be aggregated to NEST's 30-minute cadence.
2. **2019 Sensor Gap:** Do not join `2019` surveys to leaf wetness/rainfall (absent in `cleaned_weather.rds`), and drop/impute the offline wind/gust columns at `HM 2023` (or exclude wind features when using `HM 2023`).
3. **Domain Calibration Required:** Because flax rust occurs in `~3–38°C` (mean `14°C`) Colorado alpine meadows whereas NEST monitors warmer agricultural zones (`chilli`, `tomato`), temperature-interaction rules trained directly on this dataset would require climate/crop re-calibration before production deployment on tropical crops.
