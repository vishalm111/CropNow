# Fungal-Risk Classifier Evaluation Report (LOW / MEDIUM / HIGH)

**Dataset:** `data/processed/fungal_risk/flax_rust_fungal_risk_dataset.csv` (`70 rows` $\times$ `38` pure environmental features)
**Validation Schemes:** Leave-One-Year-Out (LOYO, 4 folds: `2020–2023`) & Leave-One-Site-Out (LOSO, 4 folds: `BT, CC, GM, HM`)
**Leakage Guard:** Prevalence tertile cutoffs (`q33`, `q67`) computed **strictly inside each training fold only**; `StandardScaler` fit inside training folds via `Pipeline`.
**Models Evaluated:** Logistic Regression, Shallow Random Forest (`max_depth=3`), Regularized XGBoost (`max_depth=2`) — **No GRU used**.
**Selected Best Model:** **`XGBoost`** (saved separately to `backend/app/ml/models_fungal_risk/best_fungal_model.joblib`)

---

## 1. Cross-Validation Model Comparison

| Model | LOYO Macro F1 | LOYO Bal. Acc | LOSO Macro F1 | LOSO Bal. Acc | Mean Macro F1 | Mean Bal. Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LogisticRegression** | 0.4904 | 0.4843 | 0.3688 | 0.3731 | **0.4296** | **0.4287** |
| **ShallowRandomForest** | 0.4193 | 0.4135 | 0.3279 | 0.3259 | **0.3736** | **0.3697** |
| **XGBoost** *(Best)* | 0.5121 | 0.5170 | 0.3734 | 0.3787 | **0.4427** | **0.4478** |

---

## 2. Per-Class Performance & Confusion Matrices

### 2.1 LogisticRegression

#### Leave-One-Year-Out (LOYO) — Macro F1: `0.4904` | Balanced Accuracy: `0.4843` | Accuracy: `0.4857`

**Per-Class Metrics:**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | 0.3704 | 0.4545 | 0.4082 | 22 |
| **MEDIUM** | 0.4400 | 0.4783 | 0.4583 | 23 |
| **HIGH** | 0.7222 | 0.5200 | 0.6047 | 25 |

**Confusion Matrix:**

| Actual \ Predicted | LOW | MEDIUM | HIGH |
| :--- | :---: | :---: | :---: |
| **LOW** | 10 | 8 | 4 |
| **MEDIUM** | 11 | 11 | 1 |
| **HIGH** | 6 | 6 | 13 |

**Per-Fold Breakdown:**

| Held-Out Fold | Train n | Test n | Train `q33` | Train `q67` | Macro F1 | Bal. Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2020` | 57 | 13 | 0.0482 | 0.1523 | 0.6071 | 0.6481 |
| `2021` | 54 | 16 | 0.0642 | 0.1694 | 0.3657 | 0.5667 |
| `2022` | 54 | 16 | 0.0475 | 0.1043 | 0.3838 | 0.3611 |
| `2023` | 45 | 25 | 0.0493 | 0.1275 | 0.4027 | 0.4798 |

#### Leave-One-Site-Out (LOSO) — Macro F1: `0.3688` | Balanced Accuracy: `0.3731` | Accuracy: `0.4000`

**Per-Class Metrics:**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | 0.3667 | 0.4783 | 0.4151 | 23 |
| **MEDIUM** | 0.1250 | 0.1053 | 0.1143 | 19 |
| **HIGH** | 0.6250 | 0.5357 | 0.5769 | 28 |

**Confusion Matrix:**

| Actual \ Predicted | LOW | MEDIUM | HIGH |
| :--- | :---: | :---: | :---: |
| **LOW** | 11 | 10 | 2 |
| **MEDIUM** | 10 | 2 | 7 |
| **HIGH** | 9 | 4 | 15 |

**Per-Fold Breakdown:**

| Held-Out Fold | Train n | Test n | Train `q33` | Train `q67` | Macro F1 | Bal. Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BT` | 48 | 22 | 0.0485 | 0.1295 | 0.4516 | 0.4824 |
| `CC` | 51 | 19 | 0.0525 | 0.1179 | 0.3818 | 0.3651 |
| `GM` | 51 | 19 | 0.0431 | 0.1140 | 0.2562 | 0.4545 |
| `HM` | 60 | 10 | 0.0688 | 0.1639 | 0.2308 | 0.1667 |

### 2.2 ShallowRandomForest

#### Leave-One-Year-Out (LOYO) — Macro F1: `0.4193` | Balanced Accuracy: `0.4135` | Accuracy: `0.4143`

**Per-Class Metrics:**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | 0.3000 | 0.4091 | 0.3462 | 22 |
| **MEDIUM** | 0.4091 | 0.3913 | 0.4000 | 23 |
| **HIGH** | 0.6111 | 0.4400 | 0.5116 | 25 |

**Confusion Matrix:**

| Actual \ Predicted | LOW | MEDIUM | HIGH |
| :--- | :---: | :---: | :---: |
| **LOW** | 9 | 9 | 4 |
| **MEDIUM** | 11 | 9 | 3 |
| **HIGH** | 10 | 4 | 11 |

**Per-Fold Breakdown:**

| Held-Out Fold | Train n | Test n | Train `q33` | Train `q67` | Macro F1 | Bal. Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2020` | 57 | 13 | 0.0482 | 0.1523 | 0.2571 | 0.3148 |
| `2021` | 54 | 16 | 0.0642 | 0.1694 | 0.3352 | 0.5333 |
| `2022` | 54 | 16 | 0.0475 | 0.1043 | 0.3851 | 0.3889 |
| `2023` | 45 | 25 | 0.0493 | 0.1275 | 0.2333 | 0.3095 |

#### Leave-One-Site-Out (LOSO) — Macro F1: `0.3279` | Balanced Accuracy: `0.3259` | Accuracy: `0.3429`

**Per-Class Metrics:**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | 0.2812 | 0.3913 | 0.3273 | 23 |
| **MEDIUM** | 0.1765 | 0.1579 | 0.1667 | 19 |
| **HIGH** | 0.5714 | 0.4286 | 0.4898 | 28 |

**Confusion Matrix:**

| Actual \ Predicted | LOW | MEDIUM | HIGH |
| :--- | :---: | :---: | :---: |
| **LOW** | 9 | 11 | 3 |
| **MEDIUM** | 10 | 3 | 6 |
| **HIGH** | 13 | 3 | 12 |

**Per-Fold Breakdown:**

| Held-Out Fold | Train n | Test n | Train `q33` | Train `q67` | Macro F1 | Bal. Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BT` | 48 | 22 | 0.0485 | 0.1295 | 0.3902 | 0.4454 |
| `CC` | 51 | 19 | 0.0525 | 0.1179 | 0.3512 | 0.3651 |
| `GM` | 51 | 19 | 0.0431 | 0.1140 | 0.2562 | 0.4545 |
| `HM` | 60 | 10 | 0.0688 | 0.1639 | 0.1407 | 0.5556 |

### 2.3 XGBoost

#### Leave-One-Year-Out (LOYO) — Macro F1: `0.5121` | Balanced Accuracy: `0.5170` | Accuracy: `0.5143`

**Per-Class Metrics:**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | 0.4667 | 0.6364 | 0.5385 | 22 |
| **MEDIUM** | 0.5263 | 0.4348 | 0.4762 | 23 |
| **HIGH** | 0.5714 | 0.4800 | 0.5217 | 25 |

**Confusion Matrix:**

| Actual \ Predicted | LOW | MEDIUM | HIGH |
| :--- | :---: | :---: | :---: |
| **LOW** | 14 | 4 | 4 |
| **MEDIUM** | 8 | 10 | 5 |
| **HIGH** | 8 | 5 | 12 |

**Per-Fold Breakdown:**

| Held-Out Fold | Train n | Test n | Train `q33` | Train `q67` | Macro F1 | Bal. Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2020` | 57 | 13 | 0.0482 | 0.1523 | 0.1953 | 0.2407 |
| `2021` | 54 | 16 | 0.0642 | 0.1694 | 0.5823 | 0.7333 |
| `2022` | 54 | 16 | 0.0475 | 0.1043 | 0.4960 | 0.5278 |
| `2023` | 45 | 25 | 0.0493 | 0.1275 | 0.2916 | 0.3929 |

#### Leave-One-Site-Out (LOSO) — Macro F1: `0.3734` | Balanced Accuracy: `0.3787` | Accuracy: `0.4000`

**Per-Class Metrics:**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | 0.3438 | 0.4783 | 0.4000 | 23 |
| **MEDIUM** | 0.2143 | 0.1579 | 0.1818 | 19 |
| **HIGH** | 0.5833 | 0.5000 | 0.5385 | 28 |

**Confusion Matrix:**

| Actual \ Predicted | LOW | MEDIUM | HIGH |
| :--- | :---: | :---: | :---: |
| **LOW** | 11 | 9 | 3 |
| **MEDIUM** | 9 | 3 | 7 |
| **HIGH** | 12 | 2 | 14 |

**Per-Fold Breakdown:**

| Held-Out Fold | Train n | Test n | Train `q33` | Train `q67` | Macro F1 | Bal. Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BT` | 48 | 22 | 0.0485 | 0.1295 | 0.4994 | 0.6204 |
| `CC` | 51 | 19 | 0.0525 | 0.1179 | 0.4343 | 0.4392 |
| `GM` | 51 | 19 | 0.0431 | 0.1140 | 0.2213 | 0.4242 |
| `HM` | 60 | 10 | 0.0688 | 0.1639 | 0.1407 | 0.5556 |

---

## 3. Top Predictive Features (`XGBoost`, Full-Dataset Fit)

Reference full-dataset prevalence thresholds: `LOW <= 0.0520`, `MEDIUM <= 0.1313`, `HIGH > 0.1313`.

| Rank | Feature | Importance / Mean Abs Coef |
| :---: | :--- | :---: |
| 1 | `temp_max_7d` | 0.0528 |
| 2 | `soil_moisture_mean_7d` | 0.0485 |
| 3 | `temp_mean_7d` | 0.0484 |
| 4 | `lw_max_7d` | 0.0470 |
| 5 | `temp_mean_24h` | 0.0459 |
| 6 | `dp_mean_7d` | 0.0420 |
| 7 | `favorable_infection_hours_7d` | 0.0383 |
| 8 | `soil_moisture_mean_24h` | 0.0369 |
| 9 | `temp_diurnal_range_mean_7d` | 0.0361 |
| 10 | `temp_min_7d` | 0.0304 |

---

## 4. Saved Artifacts

1. **Best Model Artifact:** `backend/app/ml/models_fungal_risk/best_fungal_model.joblib`
2. **Model Metadata:** `backend/app/ml/models_fungal_risk/model_metadata.json`
3. **Structured Metrics JSON:** `backend/app/ml/evaluation/fungal_risk/fungal_risk_metrics.json`
4. **Confusion Matrices Visualization:** `backend/app/ml/evaluation/fungal_risk/fungal_risk_confusion_matrices.png`
