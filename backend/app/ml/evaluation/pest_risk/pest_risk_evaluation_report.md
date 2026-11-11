# Pest-Risk Classifier Evaluation Report (RBB LOW / MEDIUM / HIGH)

**Dataset:** `data/processed/pest_risk/rbb_pest_risk_dataset.csv` (`363 rows` $\times$ `26` environmental-only features; `WSB` excluded)
**Validation Scheme:** 4 Expanding-Window Chronological Walk-Forward Folds (`2021`, `2022`, `2023`, `2024`; `209` pooled out-of-fold predictions)
**Leakage Guards:** RBB count tertiles (`q33`, `q67`) and `StandardScaler` computed **strictly inside each training fold only**. Random/interleaved split rejected.
**Models Evaluated:** XGBoost, Logistic Regression, Shallow Random Forest — **No GRU used**.
**Selected Best Model:** **`LogisticRegression`** (saved separately to `backend/app/ml/models_pest_risk/best_pest_model.joblib`)

---

## 1. Chronological Cross-Validation Model Comparison

| Model | 4-Fold Pooled Macro F1 | 4-Fold Balanced Acc | 4-Fold Accuracy | 2024 Holdout Macro F1 | 2024 Holdout Bal. Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.3451** | **0.3454** | 0.3589 | 0.4789 | 0.5500 |
| **LogisticRegression** *(Best)* | **0.4164** | **0.4372** | 0.4163 | 0.3796 | 0.4364 |
| **ShallowRandomForest** | **0.3454** | **0.3451** | 0.3589 | 0.5100 | 0.5302 |

---

## 2. Per-Class Performance, Confusion Matrices & Per-Fold Breakdown

### 2.1 XGBoost (Pooled Macro F1: `0.3451` | Balanced Accuracy: `0.3454`)

**Per-Class Metrics (2021–2024 Pooled OOF, `n=209`):**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | 0.1905 | 0.2500 | 0.2162 | 48 |
| **MEDIUM** | 0.3933 | 0.4487 | 0.4192 | 78 |
| **HIGH** | 0.4912 | 0.3373 | 0.4000 | 83 |

**Confusion Matrix (Pooled OOF `n=209`):**

| Actual \ Predicted | LOW | MEDIUM | HIGH |
| :--- | :---: | :---: | :---: |
| **LOW** | 12 | 22 | 14 |
| **MEDIUM** | 28 | 35 | 15 |
| **HIGH** | 23 | 32 | 28 |

**Per-Fold Chronological Breakdown:**

| Held-Out Eval Year | Train n | Eval n | Train `q33` | Train `q67` | Macro F1 | Bal. Accuracy | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2021` | 154 | 52 | 2.00 | 143.00 | 0.2314 | 0.2553 | 0.2308 |
| `2022` | 206 | 52 | 1.00 | 117.00 | 0.3246 | 0.3297 | 0.3269 |
| `2023` | 258 | 52 | 2.00 | 144.67 | 0.3337 | 0.3192 | 0.3846 |
| `2024 (+ 2024-12-30 wk)` | 310 | 53 | 4.00 | 145.00 | 0.4789 | 0.5500 | 0.4906 |

### 2.2 LogisticRegression (Pooled Macro F1: `0.4164` | Balanced Accuracy: `0.4372`)

**Per-Class Metrics (2021–2024 Pooled OOF, `n=209`):**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | 0.3418 | 0.5625 | 0.4252 | 48 |
| **MEDIUM** | 0.3953 | 0.4359 | 0.4146 | 78 |
| **HIGH** | 0.5909 | 0.3133 | 0.4094 | 83 |

**Confusion Matrix (Pooled OOF `n=209`):**

| Actual \ Predicted | LOW | MEDIUM | HIGH |
| :--- | :---: | :---: | :---: |
| **LOW** | 27 | 15 | 6 |
| **MEDIUM** | 32 | 34 | 12 |
| **HIGH** | 20 | 37 | 26 |

**Per-Fold Chronological Breakdown:**

| Held-Out Eval Year | Train n | Eval n | Train `q33` | Train `q67` | Macro F1 | Bal. Accuracy | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2021` | 154 | 52 | 2.00 | 143.00 | 0.4410 | 0.4455 | 0.4615 |
| `2022` | 206 | 52 | 1.00 | 117.00 | 0.2600 | 0.3176 | 0.2692 |
| `2023` | 258 | 52 | 2.00 | 144.67 | 0.5097 | 0.5362 | 0.5385 |
| `2024 (+ 2024-12-30 wk)` | 310 | 53 | 4.00 | 145.00 | 0.3796 | 0.4364 | 0.3962 |

### 2.3 ShallowRandomForest (Pooled Macro F1: `0.3454` | Balanced Accuracy: `0.3451`)

**Per-Class Metrics (2021–2024 Pooled OOF, `n=209`):**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | 0.1905 | 0.2500 | 0.2162 | 48 |
| **MEDIUM** | 0.4000 | 0.4359 | 0.4172 | 78 |
| **HIGH** | 0.4754 | 0.3494 | 0.4028 | 83 |

**Confusion Matrix (Pooled OOF `n=209`):**

| Actual \ Predicted | LOW | MEDIUM | HIGH |
| :--- | :---: | :---: | :---: |
| **LOW** | 12 | 20 | 16 |
| **MEDIUM** | 28 | 34 | 16 |
| **HIGH** | 23 | 31 | 29 |

**Per-Fold Chronological Breakdown:**

| Held-Out Eval Year | Train n | Eval n | Train `q33` | Train `q67` | Macro F1 | Bal. Accuracy | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2021` | 154 | 52 | 2.00 | 143.00 | 0.2319 | 0.2470 | 0.2308 |
| `2022` | 206 | 52 | 1.00 | 117.00 | 0.2659 | 0.3058 | 0.2692 |
| `2023` | 258 | 52 | 2.00 | 144.67 | 0.2931 | 0.2840 | 0.3654 |
| `2024 (+ 2024-12-30 wk)` | 310 | 53 | 4.00 | 145.00 | 0.5100 | 0.5302 | 0.5660 |

---

## 3. Top Environmental Drivers (`LogisticRegression`, Full-Dataset Fit)

Reference full-dataset RBB count thresholds: `LOW <= 5.67`, `MEDIUM <= 188.67`, `HIGH > 188.67`.

| Rank | Environmental Feature | Importance / Mean Abs Coef |
| :---: | :--- | :---: |
| 1 | `T2M_MAX` | 0.3833 |
| 2 | `gwettop_trend_4w` | 0.3070 |
| 3 | `WS2M` | 0.3017 |
| 4 | `t2m_lag_1w` | 0.2331 |
| 5 | `T2M` | 0.2156 |
| 6 | `rain_sum_4w` | 0.2147 |
| 7 | `gwettop_lag_1w` | 0.2053 |
| 8 | `T2M_MIN` | 0.2021 |
| 9 | `rh2m_lag_1w` | 0.1960 |
| 10 | `t2m_mean_4w` | 0.1960 |

---

## 4. Saved Artifacts (Separate from Soil Moisture & Fungal Models)

1. **Best Model Bundle:** `backend/app/ml/models_pest_risk/best_pest_model.joblib`
2. **Model Metadata:** `backend/app/ml/models_pest_risk/model_metadata.json`
3. **Structured Metrics JSON:** `backend/app/ml/evaluation/pest_risk/pest_risk_metrics.json`
4. **Confusion Matrices Visualization:** `backend/app/ml/evaluation/pest_risk/pest_risk_confusion_matrices.png`
