# Pest-Risk Modelling Data Preparation & Audit Report

**Date Prepared:** November 10, 2026
**Source File:** [`data/external/pest/weekly_lot_a_with_moon_dataset.csv`](file:///c:/Users/Vishal/Downloads/CropNow/data/external/pest/weekly_lot_a_with_moon_dataset.csv) (`366` weekly records, 2018–2024)
**Preprocessing Script:** [`backend/app/ml/prepare_pest_dataset.py`](file:///c:/Users/Vishal/Downloads/CropNow/backend/app/ml/prepare_pest_dataset.py)
**Primary RBB Output Table:** [`data/processed/pest_risk/rbb_pest_risk_dataset.csv`](file:///c:/Users/Vishal/Downloads/CropNow/data/processed/pest_risk/rbb_pest_risk_dataset.csv) (`363 rows` $\times$ `33 columns`)
**Isolated WSB Target File:** [`data/processed/pest_risk/wsb_pest_counts_separate.csv`](file:///c:/Users/Vishal/Downloads/CropNow/data/processed/pest_risk/wsb_pest_counts_separate.csv) (`363 rows` $\times$ `5 columns`)
**Fold Threshold Metadata:** [`data/processed/pest_risk/rbb_chronological_fold_thresholds.json`](file:///c:/Users/Vishal/Downloads/CropNow/data/processed/pest_risk/rbb_chronological_fold_thresholds.json)
**Model Training Status:** **NOT TRAINED** (data preparation and audit only).

---

## 1. Chronological Sorting & Raw Timestamp Audit Findings

Inspecting `weekly_lot_a_with_moon_dataset.csv` prior to sorting revealed critical ordering anomalies in the raw `Date` and `Year` fields:
1. **Raw File Not Sorted by `(Year, Week_Number)`:** Naive comparison showed `df` did not match `df.sort_values(['Year', 'Week_Number'])` and had **2 duplicate `(Year, Week_Number)` pairs**.
2. **Exact 7-Day Physical Cadence Confirmed:** Aligning rows `0..365` to the weekly Thursday timeline starting `2018-01-04` (`2018-01-04 + 7*i days`) matched **`366 / 366` (`100.0%`)** of the CSV's `Week_Number` column via ISO-8601 week numbering (`isocalendar().week`).
3. **3 ISO-Week Year-Boundary Anomalies Detected & Repaired:**
   - **Row 156** (`2020-12-31`, ISO Week 53 of **2020**): Corrupted in CSV as `Year=2021`, `Date='7/17/2021'`, `Week_Number=53`. Naive sort by `(Year, Week_Number)` would push this 2020 week to the end of 2021. Repaired `sort_year = 2020`.
   - **Row 260** (`2022-12-29`, ISO Week 52 of **2022**): Corrupted in CSV as `Year=2023`, `Date='10/24/2023'`, `Week_Number=52`. Naive sort would push this 2022 week to the end of 2023. Repaired `sort_year = 2022`.
   - **Row 365** (`2025-01-02` / `12/30/2024`, ISO Week 1 of **2025**): Marked `Year=2024`, `Week_Number=1`. Naive sort would shift the final week of the dataset to the start of 2024. Repaired `sort_year = 2025`.
4. **Result:** Sorting chronologically by `(sort_year, Week_Number)` yields **0 duplicate week keys** and an exact, unbroken **7-day cadence across all records**.
5. **Warm-up Trimming:** Dropping the initial 3 warm-up weeks (`2018` weeks 1–3) required for complete 4-week rolling meteorological history yields **`363` clean, zero-null rows**.

---

## 2. Target Separation & Exclusion of Non-Environmental Features

- **Primary Target (`rbb_count`):** Weekly Rice Black Bug (`RBB`) trap counts (`0` to `108,355`, median `44.0`).
- **WSB Strictly Isolated:** White Stem Borer (`WSB`) counts are removed entirely from `rbb_pest_risk_dataset.csv` and stored separately in `wsb_pest_counts_separate.csv` to guarantee zero target leakage.
- **Non-Environmental Columns Excluded from Predictors:**
  - `Moon_Category` (non-meteorological lunar phase indicator)
  - `Date`, `Year`, `Month`, `Day`, `Week_Number` (corrupted calendar strings / calendar proxies)
  - Historical pest trap counts (`RBB` lags, `WSB`)
  - External `Data_Split` from `RBB_P75.1_binary_labels.csv` (rejected because it interleaves `train` and `test` randomly across 2018–2024).

### Engineered Environmental-Only Feature Matrix (`26` Features)
| Feature Group | Features Included | Count |
| :--- | :--- | :---: |
| **Current-Week Base Weather & Soil** | `T2M`, `T2M_MIN`, `T2M_MAX`, `RH2M`, `PRECTOTCORR`, `WS2M`, `WS2M_MAX`, `WS2M_MIN`, `WD2M`, `GWETTOP` | 10 |
| **Derived Intra-Week Meteorological** | `temp_diurnal_range` (`T2M_MAX - T2M_MIN`), `wind_gust_spread` (`WS2M_MAX - WS2M_MIN`), `wd_sin`, `wd_cos` (cyclic wind direction) | 4 |
| **1-Week Causal Environmental Lags (`t-1`)** | `t2m_lag_1w`, `rh2m_lag_1w`, `rain_lag_1w`, `gwettop_lag_1w`, `ws2m_lag_1w` | 5 |
| **4-Week Rolling Weather History (`[t-3..t]`)** | `t2m_mean_4w`, `rh2m_mean_4w`, `rain_sum_4w`, `gwettop_mean_4w` | 4 |
| **4-Week Environmental Trend Slopes** | `t2m_trend_4w`, `rh2m_trend_4w`, `gwettop_trend_4w` (OLS slope over 4 weeks) | 3 |

All `363` rows have **0 missing values** and **0 duplicate rows**.

---

## 3. Chronological Validation & Training-Fold-Only `LOW / MEDIUM / HIGH` Labels

Instead of the supplied random/interleaved split, the pipeline establishes an **expanding-window yearly walk-forward chronological validation scheme** across `2021–2024`, plus a chronological `train_early (2018–2022)` / `val (2023)` / `holdout_test (2024)` partition.

Within each fold, RBB tertile thresholds (`q33.33` and `q66.67`) are computed **strictly from that fold's prior-year training rows only** (`LOW: rbb <= q33`, `MEDIUM: q33 < rbb <= q67`, `HIGH: rbb > q67`):

| Walk-Forward Fold | Training Years (`sort_year`) | Train `n` | Eval `n` | `q33` Threshold (Train Only) | `q67` Threshold (Train Only) | Eval Class Distribution |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Fold 1 (2021)** | `2018–2020` | 154 | 52 | `2.00` | `143.00` | `LOW`: 24, `MEDIUM`: 13, `HIGH`: 15 |
| **Fold 2 (2022)** | `2018–2021` | 206 | 52 | `1.00` | `117.00` | `LOW`: 11, `MEDIUM`: 18, `HIGH`: 23 |
| **Fold 3 (2023)** | `2018–2022` | 258 | 52 | `2.00` | `144.67` | `LOW`: 7, `MEDIUM`: 27, `HIGH`: 18 |
| **Fold 4 (2024 Holdout)** | `2018–2023` | 310 | 53 | `4.00` | `145.00` | `LOW`: 6, `MEDIUM`: 20, `HIGH`: 27 |

### Key Analytical Observations
1. **Zero-Skewed Lower Tertile (`q33` = `1–4` RBB insects):** Because ~18% of weeks record `0` RBB catches and ~33% record `<= 4` catches, `LOW` risk cleanly captures near-zero dormant weeks while `HIGH` (`> 117–145` RBB) captures outbreak surges.
2. **Temporal Drift Between Years:** Outbreak frequency shifted upward in `2022` and `2024` (`23` and `27` `HIGH` weeks respectively) compared to `2021` (`15` `HIGH`, `24` `LOW`), reinforcing why **chronological walk-forward validation** is essential instead of interleaved splits.
