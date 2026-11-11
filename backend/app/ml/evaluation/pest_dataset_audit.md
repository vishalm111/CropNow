# Pest Outbreak Datasets Audit Report

**Date of Audit:** November 10, 2026
**Location Audited:** `data/external/pest/`
**Files Covered:**
1. `data/external/pest/weekly_lot_a_with_moon_dataset.csv` (366 rows $\times$ 18 columns)
2. `data/external/pest/RBB_P75.1_binary_labels.csv` (366 rows $\times$ 14 columns)
3. `data/external/pest/WSB_P94.3_binary_labels.csv` (366 rows $\times$ 14 columns)

*(Note: `data/external/pest/` also contains a nested subdirectory `fungal/Grape Disease Dataset/Grape_Disease_Dataset.csv`, which is audited separately in [`fungal_dataset_audit.md`](fungal_dataset_audit.md).)*
**Constraint Adherence:** Datasets untouched; zero models trained or modified.

---

## 1. Dataset Inventory & Relationships

All three pest CSVs share an identical **366-row weekly observational backbone** covering 2018–2024:
- **`weekly_lot_a_with_moon_dataset.csv`** is the master feature + continuous trap-count table containing weekly weather/soil telemetry (NASA POWER meteorological schema), lunar phase category (`Moon_Category`), and continuous weekly trap counts for **RBB** (Rice Black Bug) and **WSB** (White Stem Borer).
- **`RBB_P75.1_binary_labels.csv`** and **`WSB_P94.3_binary_labels.csv`** are derived binary-label tables (`Source_Row_ID` 1..366 maps 1:1 row-for-row to `weekly_lot_a_with_moon_dataset.csv`, `Date` and `Pest_Count` match 100%). They contain **no environmental features** themselves and must be joined to `weekly_lot_a_with_moon_dataset.csv` for feature-based modeling.

---

## 2. File-by-File Inspection

### 2.1 `weekly_lot_a_with_moon_dataset.csv`
- **Dimensions:** `366 rows` $\times$ `18 columns` (`31,716` bytes)
- **Exact Columns (`18`):** `Date`, `Year`, `Month`, `Day`, `RBB`, `WSB`, `T2M`, `T2M_MIN`, `T2M_MAX`, `RH2M`, `PRECTOTCORR`, `WS2M`, `WS2M_MAX`, `WS2M_MIN`, `WD2M`, `GWETTOP`, `Moon_Category`, `Week_Number`
- **Missing Values:** **0 nulls (0.0%)** across all 18 columns.
- **Duplicates:**
  - Exact full-row duplicates: **0**
  - Duplicate `Date` strings: **7 duplicate dates** (`359` unique dates across `366` rows).
  - Duplicate `(Year, Month, Day)` tuples: **2**.

#### Environmental Features Available (`11` features + `5` calendar/time fields)

| Feature | Meaning | Min | Max | Mean $\pm$ Std | Mapping to NEST Schema |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `T2M` | Air Temp at 2m (°C) | 24.12 | 28.49 | $26.09 \pm 0.73$ | Maps to `temperature` (weekly mean) |
| `T2M_MIN` | Min Temp at 2m (°C) | 18.57 | 24.41 | $22.56 \pm 0.87$ | Diurnal min temp |
| `T2M_MAX` | Max Temp at 2m (°C) | 27.48 | 34.28 | $30.43 \pm 1.13$ | Diurnal max temp |
| `RH2M` | Relative Humidity at 2m (%) | 72.17 | 89.71 | $84.43 \pm 2.82$ | Maps to `humidity` |
| `PRECTOTCORR` | Corrected Precipitation / Rainfall (mm/day or weekly) | 0.00 | 112.08 | $7.35 \pm 8.87$ | Maps to `rainfall` |
| `WS2M` | Wind Speed at 2m (m/s) | 0.64 | 2.28 | $1.04 \pm 0.23$ | Maps to `wind_speed` |
| `WS2M_MAX` | Max Wind Speed at 2m (m/s) | 1.21 | 4.11 | $2.04 \pm 0.46$ | Peak wind |
| `WS2M_MIN` | Min Wind Speed at 2m (m/s) | 0.16 | 0.91 | $0.41 \pm 0.12$ | Min wind |
| `WD2M` | Wind Direction at 2m (deg) | 59.93 | 276.19 | $174.39 \pm 56.31$ | Maps to `wind_direction` |
| `GWETTOP` | Surface Soil Wetness (0–1 fraction) | 0.56 | 0.96 | $0.84 \pm 0.07$ | Related to `soil_moisture_1` (fraction scale) |
| `Moon_Category`| Lunar cycle phase category (`0–7`) | 0 | 7 | $3.60 \pm 1.87$ | Nocturnal light-trap catch bias feature |
| *(Missing)* | `leaf_wetness`, `solar_radiation`, `pressure`, air quality (`co`, `no2`, `o3`, `pm2_5`) | — | — | — | Absent from this dataset |

#### Target Variables (`RBB` and `WSB` continuous trap counts)
- **`RBB` (Rice Black Bug weekly count):**
  - Range: `0` to `108,355` (extreme right skew: Mean = `2,271.44`, Std = `9,203.80`).
  - Quantiles: `Q0 = 0`, `Q25 = 2.0`, `Q50 (Median) = 44.0`, `Q75 = 563.75`, `Q75.1 = 579.14`, `Q90 = 3,772.0`, `Q95 = 12,068.5`, `Q100 = 108,355`.
  - Zero count (`RBB == 0`): **65 / 366 rows (17.8%)**.
- **`WSB` (White Stem Borer weekly count):**
  - Range: `0` to `613` (right skewed: Mean = `23.39`, Std = `59.15`).
  - Quantiles: `Q0 = 0`, `Q25 = 1.0`, `Q50 (Median) = 3.0`, `Q75 = 13.0`, `Q90 = 58.5`, `Q94.3 = 134.195`, `Q95 = 139.75`, `Q100 = 613`.
  - Zero count (`WSB == 0`): **81 / 366 rows (22.1%)**.

---

### 2.2 `RBB_P75.1_binary_labels.csv`
- **Dimensions:** `366 rows` $\times$ `14 columns` (`50,919` bytes)
- **Exact Columns (`14`):** `Observation_ID`, `Source_Row_ID`, `Pest`, `Date`, `Year`, `Week_Number`, `Pest_Count`, `Percentile_Label`, `Percentile`, `Threshold_Count`, `Comparison_Rule`, `Binary_Label`, `Binary_Label_Definition`, `Data_Split`
- **Missing Values:** **0 nulls (0.0%)**
- **Duplicates:** Exact full-row duplicates = **0**; duplicate `Date` values = **7** (`359` unique dates).
- **Environmental Features:** **NONE** (metadata & labels only; joins 1:1 on `Source_Row_ID` with `weekly_lot_a_with_moon_dataset.csv`).
- **Target Label & Class Distribution:**
  - Rule: `Binary_Label = 1` if `Pest_Count > 579.14` (`P75.1`, 75.1st percentile), else `0`.
  - **`0` (does not exceed threshold):** **275 rows (75.1%)**
  - **`1` (count exceeds threshold):** **91 rows (24.9%)**
- **`Data_Split` Distribution & Leakage Risk:**
  - `train`: `292` rows (`0`: `219`, `1`: `73`) — date span `2018-01-04` to `2024-12-26`.
  - `test`: `74` rows (`0`: `56`, `1`: `18`) — date span `2018-04-19` to `2024-12-30`.
  - **Warning:** `Data_Split` is randomly/interleaved across 2018–2024 (row 0 = train, row 1 = test, etc.), creating **temporal leakage** if used directly in sequence models.

---

### 2.3 `WSB_P94.3_binary_labels.csv`
- **Dimensions:** `366 rows` $\times$ `14 columns` (`47,477` bytes)
- **Exact Columns (`14`):** Same 14 columns as `RBB_P75.1_binary_labels.csv` (`Pest = "WSB"`, `Percentile_Label = "P94.3"`, `Threshold_Count = 134.195`).
- **Missing Values:** **0 nulls (0.0%)**
- **Duplicates:** Exact full-row duplicates = **0**; duplicate `Date` values = **7** (`359` unique dates).
- **Environmental Features:** **NONE** (must join 1:1 to `weekly_lot_a_with_moon_dataset.csv`).
- **Target Label & Class Distribution:**
  - Rule: `Binary_Label = 1` if `Pest_Count > 134.195` (`P94.3`, 94.3rd percentile), else `0`.
  - **`0` (does not exceed threshold):** **345 rows (94.3%)**
  - **`1` (count exceeds threshold):** **21 rows (5.7%)** *(severely imbalanced positive class)*
- **`Data_Split` Distribution & Leakage Risk:**
  - `train`: `292` rows (`0`: `275`, `1`: `17`) — interleaved across 2018–2024.
  - `test`: `74` rows (`0`: `70`, `1`: `4`) — only **4 positive examples** in the test split.

---

## 3. Temporal Alignment Audit

**Assessment: Partially Suitable for Macro-Weekly Modeling ONLY After Date Repair — Incompatible with 30-Minute NEST Sequences**

1. **Resolution & Scale:**
   - The datasets cover ~7 years (`2018-01-04` to `2024-12-30`) at a **weekly cadence** (`Week_Number` 1–52/53 per year, 366 total weeks).
   - Cannot align with **30-minute NEST telemetry packets** (`48 packets = 24h`) without downsampling NEST data to weekly aggregates or treating weekly weather summaries separately.
2. **Severe Date String Corruption (`Date` vs `Year/Month/Day`):**
   - In `weekly_lot_a_with_moon_dataset.csv`, parsing `Date` yields **non-monotonic timestamps** (`is_monotonic_increasing = False`), with **7 duplicate dates** and negative backward jumps (e.g., `-11` days, `-2` days, `-16` days).
   - Cross-checking `Date` against the separate `Year`, `Month`, and `Day` columns reveals **57 out of 366 rows (15.6%) disagree** (for example: row 4 has `Date = 1/14/2018` while `Year=2018, Month=2, Day=14`; row 52 has `Date = 2/7/2019` while `Year=2019, Month=3, Day=7`).
   - Even `(Year, Month, Day)` jumps backwards around lunar-aligned rows (e.g. row 20 = `2018-05-24`, row 21 = `2018-05-18`, row 22 = `2018-06-07`).
   - Note that `Week_Number` increases cleanly (`1, 2, ..., 52/53`) per year, meaning `Year + Week_Number` is the most reliable temporal ordering key.
3. **Random/Interleaved `Data_Split` Leakage:**
   - Both binary label files assign `train` and `test` across interleaved weeks throughout 2018–2024 instead of a forward chronological holdout (e.g. 2018–2022 train, 2023–2024 test).

---

## 4. LOW / MEDIUM / HIGH Risk Prediction Assessment

| Dataset | Native `LOW/MEDIUM/HIGH` Labels? | Can Support 3-Tier Risk Modeling? | Key Conditions & Limitations |
| :--- | :---: | :---: | :--- |
| **`weekly_lot_a_with_moon_dataset.csv`** | **No** (Continuous counts `RBB`, `WSB`) | **YES (Indirectly — Weekly Rice Pest Risk)** | Can derive `LOW / MEDIUM / HIGH` classes by discretizing continuous `RBB` or `WSB` trap counts (e.g., quantile or agronomic thresholds) paired with the 11 weekly weather/soil-wetness features. <br>• **`RBB`:** Well-spread across quartiles (`Q25=2`, `Q50=44`, `Q75=564`, `Q100=108,355`), supporting distinct 3-tier binning.<br>• **`WSB`:** Highly zero-inflated (`22.1%` zeros, `Q75=13`), making 3-tier separation tighter (`0`, `1–13`, `>13` or `>134`).<br>• **Scope caveat:** Predicts **rice insect trap counts (`RBB`, `WSB`) at weekly resolution**, NOT fungal disease and NOT 30-min NEST telemetry. |
| **`RBB_P75.1_binary_labels.csv`** | **No** (Binary `0`/`1` only) | **Only by re-binning `Pest_Count` + Joining** | Binary label alone (`75.1%` vs `24.9%`) only supports 2-class alert prediction. Contains no weather features; must join to `weekly_lot_a_with_moon_dataset.csv` and re-discretize raw `Pest_Count` into 3 tiers (using a chronological split rather than the leaky `Data_Split`). |
| **`WSB_P94.3_binary_labels.csv`** | **No** (Binary `0`/`1` only) | **Only by re-binning `Pest_Count` + Joining** | Binary label is severely imbalanced (`94.3%` vs `5.7%`, only 21 positives total and 4 in test). Contains no weather features; requires joining to `weekly_lot_a_with_moon_dataset.csv` and re-binning `Pest_Count`. |

---

## 5. Summary Conclusions Across Both `fungal/` and `pest/` Directories

1. **No dataset currently provides native 3-class `LOW / MEDIUM / HIGH` disease or pest labels.**
2. **Fungal Disease (`Grape_Disease_Dataset.csv`):** Unlabeled environmental burst log (temperature, humidity, raw leaf wetness with `1024` spikes), no rainfall, 105-day gap, non-monotonic timestamps. **Cannot support supervised fungal disease risk training (`LOW / MEDIUM / HIGH`) without ground-truth disease labels.**
3. **Pest Outbreak (`weekly_lot_a_with_moon_dataset.csv` + label tables):** Complete zero-null weekly meteorological + pest count data over 7 years (2018–2024). **Can support weekly pest risk prediction** (either binary outbreak classification as defined by `P75.1`/`P94.3`, or 3-tier `LOW / MEDIUM / HIGH` derived from continuous `RBB`/`WSB` counts) once `(Year, Week_Number)` ordering is enforced and a chronological train/test split replaces the interleaved `Data_Split`.
