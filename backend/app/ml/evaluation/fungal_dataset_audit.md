# Fungal Disease Dataset Audit Report

**Date of Audit:** November 10, 2026
**Location Audited:** `data/external/pest/fungal/Grape Disease Dataset/Grape_Disease_Dataset.csv`
**Path Discrepancy Note:** The expected directory `data/external/fungal/` does **not** exist at the top level of `data/external/`. Instead, the `fungal/` directory was placed inside `data/external/pest/fungal/`.
**Constraint Adherence:** Datasets untouched; zero models trained or modified.

---

## 1. Dataset Inventory & Schema

| Attribute | Value |
| :--- | :--- |
| **File Name** | `Grape_Disease_Dataset.csv` |
| **Full Path** | `data/external/pest/fungal/Grape Disease Dataset/Grape_Disease_Dataset.csv` |
| **File Size** | 287,093 bytes (~280 KB) |
| **Row Count** | 10,001 rows |
| **Column Count** | 5 columns |
| **Exact Column Headers** | `'Date '`, `'Time'`, `'Temperature'`, `'Humidity'`, `'LW '` |
| **Schema Quirks** | Leading/trailing whitespace in column headers (`'Date '` and `'LW '` have trailing spaces). |

---

## 2. Data Quality: Missing Values & Duplicates

| Column | Data Type | Non-Null Count | Missing Values | Unique Values | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `'Date '` | `object` (string) | 10,001 | **0 (0.0%)** | 25 | Format `M/D/YYYY` |
| `'Time'` | `object` (string) | 10,001 | **0 (0.0%)** | 9,118 | Mixed 24-hour and 12-hour strings |
| `'Temperature'` | `float64` | 10,001 | **0 (0.0%)** | 69 | Air temperature (°C) |
| `'Humidity'` | `int64` | 10,001 | **0 (0.0%)** | 47 | Relative humidity (% RH) |
| `'LW '` | `int64` | 10,001 | **0 (0.0%)** | 22 | Leaf wetness sensor raw reading |

- **Exact Duplicate Rows:** **0**
- **Duplicate Combined Timestamps (`Date + Time`):** **0**
- **Sensor Outliers / Faults Detected in `'LW '`:**
  - `8,933` rows (`89.3%`) equal `0` (dry leaf).
  - `2` rows equal `1024` (10-bit ADC saturation fault on `7/9/2023` at `12:18:12` and `12:18:15`).
  - `3` additional rows exceed `50`: `928`, `690`, `708` (all occurring in a ~5-minute burst on `7/9/2023` between `12:17:48` and `12:23:04`).
  - Remaining non-zero values sit in the `1–41` range.

---

## 3. Available Environmental Features

| Feature | Present? | Statistics (Min / Max / Mean $\pm$ Std / Median) | Alignment with NEST Telemetry |
| :--- | :---: | :--- | :--- |
| **Temperature** (`Temperature`) | **Yes** | `25.20°C` / `32.10°C` / `26.68°C` $\pm$ `1.08°C` (Median: `26.40°C`) | Maps to NEST `temperature` |
| **Humidity** (`Humidity`) | **Yes** | `17%` / `76%` / `67.79%` $\pm$ `9.21%` (Median: `70.0%`) | Maps to NEST `humidity` |
| **Leaf Wetness** (`LW `) | **Yes** | `0` / `1024` / `0.90` $\pm$ `19.89` (Median: `0`, P99: `7`) | Maps to NEST `leaf_wetness` (requires spike filtering and scale normalization) |
| **Rainfall** | **No** | — | Missing (present in NEST schema) |
| **Solar Radiation / Wind / Pressure** | **No** | — | Missing |
| **Soil Moisture / Soil Temp** | **No** | — | Missing |

---

## 4. Target Labels & Class Distribution

| Attribute | Finding |
| :--- | :--- |
| **Target Label Column Present?** | **NONE** |
| **Disease Classes / Labels** | **0 label columns** — The CSV contains strictly unlabeled microclimate & leaf wetness sensor telemetry (`Date `, `Time`, `Temperature`, `Humidity`, `LW `). |
| **Class Distribution** | Not applicable (unlabeled dataset). |

---

## 5. Temporal Alignment & Suitability for Temporal Modeling

**Assessment: DOES NOT SUPPORT Temporal Modeling**

1. **Extreme Irregularity & Fragmented Burst Sessions:**
   - Spans `2023-03-20 01:04:53` to `2023-08-05 13:51:13`, but covers only **25 active dates** across that ~4.5-month period.
   - Logging occurs at **3–5 second intervals** (median step = `4.0s`, `95.8%` of deltas are `3–5s`) in short bursts: per-day continuous span ranges from **5.75 minutes** minimum (`7/4/2023`: 98 rows) to **30.95 minutes** median, up to 818 minutes max.
2. **Massive Temporal Gaps:**
   - **41 gaps exceed 60 seconds**, including a **105-day uninterrupted gap** between `2023-03-20 01:23:14` (row 375) and `2023-07-03 02:30:44` (row 376), plus multi-day and multi-hour gaps on active days.
3. **Non-Monotonic Timestamps (Clock Format Corruption):**
   - `ts.is_monotonic_increasing` is **`False`**.
   - **2 backward time jumps** caused by ambiguous 12-hour vs 24-hour clock strings without AM/PM markers:
     - Row 62 (`3/20/2023 13:04:16`) $\to$ Row 63 (`3/20/2023 1:04:53`) [jumps backward 12 hours].
     - Row 893 (`7/6/2023 6:44:19`) $\to$ Row 894 (`7/6/2023 5:21:00`) [jumps backward ~1.4 hours].
4. **Incompatibility with NEST Cadence & Disease Incubation Windows:**
   - Fungal disease progression operates on multi-hour to multi-day wetness/temperature/humidity accumulation windows (e.g. 24–72h continuous leaf wetness duration at 30-min or hourly steps). Short 6–30 minute logger bursts separated by days of missing data cannot establish continuous 24-hour lookback sequences.

---

## 6. LOW / MEDIUM / HIGH Risk Prediction Assessment

**Can this dataset support `LOW / MEDIUM / HIGH` fungal disease risk prediction?**

**NO — Not as a supervised ML dataset.**

| Requirement | Status in `Grape_Disease_Dataset.csv` | Verdict |
| :--- | :--- | :---: |
| **Ground-Truth Disease Labels** | Completely absent (0 label columns). | **Blocker** |
| **3-Class Target (`LOW`/`MEDIUM`/`HIGH`)** | No severity score, incidence count, or disease status column exists to bin into 3 tiers. | **Blocker** |
| **Key Fungal Drivers (`Rainfall`)** | Absent (`Temperature`, `Humidity`, `LW` only; `LW` is `0` for `89.3%` of rows and humidity maxes at `76%`, rarely reaching typical fungal sporulation thresholds of $>85\%$). | **Gap** |
| **Continuous Temporal Sequences** | Fragmented 4-second bursts over 25 days with a 105-day gap and non-monotonic timestamps. | **Blocker** |

### Practical Implication
Supervised training of a fungal disease classifier (`LOW / MEDIUM / HIGH`) cannot be performed on `Grape_Disease_Dataset.csv` without external ground-truth disease annotations. Generating heuristic rules on `Temperature`/`Humidity`/`LW` alone would only reproduce synthetic thresholds rather than learned biological risk.
