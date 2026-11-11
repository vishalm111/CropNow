# Diagnostic Report: Root-Cause Analysis of Persistence vs. ML Models

**Target File:** `backend/app/ml/evaluation/diagnostic_report.md`
**Investigation Scope:** Deep empirical analysis into why the Persistence Baseline ($\hat{y}_{t+24\text{h}} = y_t$) achieved lower overall MAE/RMSE than pretrained XGBoost and GRU, while XGBoost outperformed Persistence on Zone 1.
**Diagnostic Areas Evaluated:** Feature scaling/order, target alignment, per-zone error breakdown (Bias-Variance decomposition), chronological data splits (covariate/concept shift), and irrigation class imbalance.
**Constraint Adherence:** Zero retraining executed; no dataset or model files modified.

---

## Executive Summary

The overall metric gap (Persistence: 0.383% MAE vs. GRU: 5.784% vs. XGBoost: 6.440%) is **not caused by model misconfiguration, feature ordering bugs, or broken target alignment**. Instead, it is the mathematical result of three intersecting domain factors:

1. **Ultra-High Autoregressive Inertia:** Soil moisture is a slow-moving hydraulic state variable with a 24-hour autocorrelation of **$r = 0.840$ to $0.954$**. During the September test period, the true 24-hour mean moisture differed from current moisture by an absolute average of only **$0.383\%$**.
2. **Absolute Level vs. Residual Delta Formulation:** The models were tasked with predicting absolute moisture ($0–100\%$) rather than the residual delta ($\Delta y = y_{t+24\text{h}} - y_t$). For absolute predictors, a minor calibration error of $5–10\%$ across a $100\%$ dynamic range produces an MAE $15\times$ larger than the near-zero change of persistence.
3. **Severe End-of-Season Distribution Shift:** Chronological splitting placed training in July–August (active irrigation, high moisture) and testing in late September (post-harvest, zero irrigation, unprecedented dry-down). In Zones 4 and 5, **the test target values were entirely outside the range of values seen during training**, causing tree extrapolation ceilings and $90–99\%$ of the MSE to manifest as pure static bias.
4. **Zone 1 Proves the Hypothesis:** In Zone 1 (Tomato), where training and test distributions overlapped, **XGBoost decisively beat Persistence (MAE: 0.165% vs. 0.424%; MSE: 0.037 vs. 0.423 — an 11.4× error reduction)**.

---

## 1. Feature Scaling and Order Analysis

### Feature Pipeline Verification
- **Feature Ordering:** Verified strictly identical between `backend/app/ml/pipeline.py` and `backend/app/ml/evaluate_saved_models.py`. Both enforce `all_cols = FEATURE_COLS + STATIC_COLS` (18 columns total: 13 dynamic, 5 static crop thresholds).
- **Tabular Features:** 878 features in total (48 steps $\times$ 18 features = 864 sequence features + 14 rolling statistical aggregates).

### Diagnostic Findings

1. **Multi-Crop Pooling Distortion:**
   `StandardScaler` was fitted across the pooled multi-zone training set. Because soil moisture levels differ fundamentally across crops (Tomato $\approx 18–22\%$, Zucchini $\approx 78–80\%$, Blueberry $\approx 65\%$), the pooled scaler calculated:
   - `mean(soil_moisture)` = **49.185%**
   - `scale(soil_moisture)` = **27.437%**

   Standardizing this multi-modal distribution compressed regional variations into arbitrary Z-scores ($Z \approx -1.1$ for Tomato, $Z \approx +1.1$ for Zucchini).

2. **Target Unscaled vs. Normalized Inputs:**
   In `pipeline.py`, inputs were standardized to $\sim \mathcal{N}(0, 1)$, but the regression target $y$ was left unscaled in raw percentages ($[0, 100]$).
   - **Impact on GRU:** The GRU loss was set to `nn.HuberLoss(delta=1.0)`. With target values in $[0, 100]$, any prediction error $> 1.0$ saturates the loss into the linear regime with a constant gradient of $\pm 1.0$. This heavily penalized outlier exploration and caused the network to revert to predicting the historical sector mean.
   - **Impact on XGBoost:** While decision trees are invariant to monotonic transformations of individual columns, pooling across zones meant split thresholds on normalized features had to represent multiple crops simultaneously.

3. **Feature Importance Lag Latency:**
   Analyzing the feature importance of `xgb_model.joblib`:
   - **Rank 1 (45.09%):** Step 17 ($t - 15.0\text{ h}$) `soil_moisture`
   - **Rank 2 (23.94%):** Step 15 ($t - 16.0\text{ h}$) `soil_moisture`
   - **Rank 3 (14.16%):** Step 16 ($t - 15.5\text{ h}$) `soil_moisture`
   - **Rank 4 (12.83%):** Step 18 ($t - 14.5\text{ h}$) `soil_moisture`
   - **Sum of Steps 15–18:** **96.02% of total model importance!**
   - **Current Step 47 ($t - 0\text{ h}$) Importance:** **0.028%**

   *Root Cause:* The trees split almost exclusively on soil moisture from 15 hours prior (a proxy for the sector's daily baseline). When a sudden seasonal dry-down occurred in September, relying on 15-hour-old readings introduced severe lag latency, whereas Persistence relied 100% on the immediate observation at $t=0$.

---

## 2. Target Alignment & Autoregressive Signal Inertia

### Target Formulation Check
The alignment of `target_mean_24h` was re-verified against the forward 48-step rolling mean of 30-minute resampled moisture:
- Zone 1: Pearson $r = 0.99958$, MAE diff = $0.044\%$
- Zone 2: Pearson $r = 0.99991$, MAE diff = $0.008\%$
- Zone 4: Pearson $r = 0.99953$, MAE diff = $0.100\%$
- Zone 5: Pearson $r = 0.99959$, MAE diff = $0.600\%$
The target is mathematically and physically aligned with the true 24-hour forward mean.

### Autocorrelation & The "Persistence Advantage"
Soil moisture in undisturbed soil without heavy rainfall exhibits massive temporal inertia:

| Lag Distance | Zone 1 (Tomato) | Zone 2 (Tomato) | Zone 4 (Zucchini) | Zone 5 (Blueberry) |
|---|---|---|---|---|
| **Lag 1 (30 min)** | 0.9419 | 0.9736 | 0.9470 | 0.9865 |
| **Lag 8 (4 hours)** | 0.8771 | 0.9270 | 0.8435 | 0.9623 |
| **Lag 24 (12 hours)** | 0.8558 | 0.9155 | 0.7797 | 0.9285 |
| **Lag 48 (24 hours)** | **0.8401** | **0.9040** | **0.7955** | **0.9540** |

Across the entire test set, the true average absolute difference between the current reading $y_t$ and the 24-hour target $y_{t+24\text{h}}$ was only:
- Zone 1: $|\Delta y| = 0.424\%$
- Zone 2: $|\Delta y| = 0.442\%$
- Zone 4: $|\Delta y| = 0.319\%$
- Zone 5: $|\Delta y| = 0.361\%$
- **Overall Test Set:** **$0.383\%$**

**Why Persistence Dominates Absolute Regression:**
Persistence implicitly predicts $\widehat{\Delta y} = 0$. Since the true $\Delta y$ in September averaged only $0.383\%$, Persistence achieved an MAE of $0.383\%$ by definition. For an absolute regression model predicting $\hat{y} \in [0, 100]$, even a $2\%$ calibration bias produces an error $5\times$ higher than Persistence.

---

## 3. Per-Zone Error & Bias-Variance Decomposition

To quantify whether model error stemmed from random noise (variance) or systematic level shift (bias), an exact Bias-Variance decomposition ($\text{MSE} = \text{Bias}^2 + \text{Variance}$) was performed on the test set:

| Zone & Crop | Model | Test MSE | $\text{Bias}^2$ | Variance | $\text{Bias}^2$ % of MSE | Mean Error (Bias) |
|---|---|---|---|---|---|---|
| **Zone 1: Tomato**<br>*(N=211)* | Persistence<br>**XGBoost**<br>GRU | 0.423<br>**0.037**<br>0.378 | 0.001<br>0.027<br>0.119 | 0.422<br>0.009<br>0.259 | 0.3%<br>74.1%<br>31.5% | +0.04%<br>**+0.16%**<br>+0.35% |
| **Zone 2: Tomato**<br>*(N=345)* | **Persistence**<br>XGBoost<br>GRU | **0.270**<br>0.664<br>2.906 | 0.001<br>0.644<br>1.462 | 0.270<br>0.020<br>1.444 | 0.3%<br>**97.0%**<br>50.3% | **+0.03%**<br>+0.80%<br>+1.21% |
| **Zone 4: Zucchini**<br>*(N=347)* | **Persistence**<br>XGBoost<br>GRU | **0.296**<br>58.544<br>64.025 | 0.034<br>53.144<br>25.764 | 0.262<br>5.400<br>38.261 | 11.6%<br>**90.8%**<br>40.2% | **+0.19%**<br>+7.29%<br>+5.08% |
| **Zone 5: Blueberry**<br>*(N=330)* | **Persistence**<br>XGBoost<br>GRU | **0.253**<br>230.342<br>151.852 | 0.036<br>229.847<br>151.005 | 0.218<br>0.495<br>0.847 | 14.1%<br>**99.8%**<br>99.4% | **+0.19%**<br>+15.16%<br>+12.29% |

### Critical Takeaways:
1. **In Zone 1, XGBoost Beat Persistence:** XGBoost achieved an MSE of **0.037** vs. Persistence's **0.423** (**11.4× lower mean squared error**). Because Zone 1's moisture remained stable across summer and early September ($19–22\%$), XGBoost learned genuine diurnal evapotranspiration dynamics and outperformed the naive baseline.
2. **In Zones 4 & 5, Errors are 90–99% Pure Static Bias:**
   - Zone 4: Mean error = **+7.29%** (Variance is only 5.40).
   - Zone 5: Mean error = **+15.16%** (Variance is only 0.495).
   The models are tracking the dynamic shape of the series correctly, but are shifted upward by a static offset because the test set dried down far below the training range.

---

## 4. Chronological Data Splits & Covariate Shift

The 70% / 15% / 15% chronological split created an extreme temporal covariate and concept shift:

| Zone | Split | Calendar Window | Moisture Mean (Min – Max) | 24h Target Mean (Min – Max) | Out-of-Distribution Status |
|---|---|---|---|---|---|
| **Zone 1** | Train<br>Test | Jul 24 – Aug 23<br>Aug 31 – Sep 04 | 21.66% (16.0 – 25.0)<br>19.24% (15.0 – 20.0) | 21.59% (19.9 – 24.6)<br>19.21% (18.9 – 19.3) | **In-Distribution** (XGBoost beats Persistence) |
| **Zone 2** | Train<br>Test | Jul 24 – Sep 05<br>Sep 16 – Sep 23 | 18.01% (12.3 – 21.0)<br>15.37% (12.3 – 16.0) | 17.95% (15.9 – 20.6)<br>15.35% (15.1 – 15.6) | **Slight Shift** (0.8% offset) |
| **Zone 4** | Train<br>Test | Jul 24 – Sep 05<br>Sep 16 – Sep 23 | 78.78% (59.7 – 99.0)<br>**56.03%** (52.3 – 58.0) | 78.23% (**63.4** – 86.0)<br>**55.84%** (55.0 – 57.5) | **Severe OOD:** Test target max ($57.5\%$) is well below Train target min ($63.4\%$) |
| **Zone 5** | Train<br>Test | Jul 24 – Sep 03<br>Sep 14 – Sep 21 | 65.12% (23.0 – 90.5)<br>**1.12%** (0.0 – 3.0) | 63.57% (**22.8** – 68.3)<br>**0.93%** (0.0 – 2.4) | **Extreme OOD:** Test targets $\sim 1\%$ vs. Train target min of $22.8\%$ |

### The Tree Extrapolation Barrier
Decision trees (XGBoost) partition feature space with orthogonal hyperplanes and assign constant leaf values computed from training instances. **A decision tree cannot extrapolate beyond the extreme values observed in training.**
- In Zone 5, the lowest target observed during training was **22.80%**. The lowest leaf node value XGBoost could possibly output was $\sim 16.03\%$.
- When presented with a test sample whose ground truth was **0.93%**, XGBoost output its lowest possible leaf ($16.03\%$), mathematically guaranteeing an error of $16.03 - 0.93 = +15.10\%$.
- Persistence, by contrast, reads $y_t = 1.12\%$ directly from the current sensor, incurring an error of only $0.19\%$.

---

## 5. Irrigation Policy Shift & Class Imbalance

### Irrigation Policy Inversion Between Train and Test
In precision farming, soil moisture is driven by applied water. During training, the crops were actively cultivated and irrigated. In late September (test set), the growing season concluded, and farm valves were closed permanently:

| Sector / Crop | Sequences with Active Irrigation (Train) | Mean Irrigation Minutes / 24h (Train) | Sequences with Active Irrigation (Test) | Mean Irrigation Minutes / 24h (Test) |
|---|---|---|---|---|
| **Zone 1: Tomato** | 0.0% | 0.00 min | 0.0% | 0.00 min |
| **Zone 2: Tomato** | 0.0% | 0.00 min | 0.0% | 0.00 min |
| **Zone 4: Zucchini** | **89.2%** | **24.77 min** | **0.0%** | **0.00 min** |
| **Zone 5: Blueberry** | **61.1%** | **5.90 min** | **0.0%** | **0.00 min** |

- **Physics of the Error:** In Zone 4, whenever soil moisture reached $\sim 60\%$ during training, the farmer irrigated ($89.2\%$ of sequences), rebounding moisture toward $80\%$. The models learned that a soil state around $60\%$ with high radiation leads to stabilization or recovery.
- In the test period, irrigation was permanently discontinued ($0.0\%$). Soil at $60\%$ kept drying down to $55\%$. Because future irrigation was excluded (to satisfy the zero-leakage requirement), the models expected the historical irrigation policy to support the soil level, overpredicting moisture by $+7.29\%$.

### Severe Irrigation Class Imbalance
The distribution of agronomic classes in the test set is completely monolithic:

| Split | Zone 1 (Tomato) | Zone 2 (Tomato) | Zone 4 (Zucchini) | Zone 5 (Blueberry) |
|---|---|---|---|---|
| **Train Set** | 99.1% Stress, 0.9% Danger | 81.1% Danger, 18.9% Stress | 71.6% Stress, **28.4% Optimal** | 95.2% Stress, 4.8% Danger |
| **Test Set** | **100.0% Danger** | **100.0% Danger** | **100.0% Stress** | **100.0% Danger** |

- In the entire test set, there are **zero samples** in the Optimal, Excess, or Saturation classes.
- $100\%$ of test samples fall below the Management Allowed Depletion Point (MADP).
- This extreme imbalance explains why all models achieved $100\%$ precision/recall on the binary irrigation gate: every crop had completely dried into the deficit irrigation zone by late September.

---

## 6. Summary Comparison: Why Each Model Behaved As It Did

```
+-----------------------------------------------------------------------------------------------------+
| PERSISTENCE BASELINE (MAE: 0.383%)                                                                  |
| • Exploits ultra-high physical inertia (autocorrelation r = 0.84 to 0.95).                          |
| • Assumes Delta_y = 0. In an un-irrigated dry-down, true Delta_y is only 0.38%, so error is tiny.   |
| • Fully immune to seasonal covariate shift, mean level shifts, and tree extrapolation limits.       |
+-----------------------------------------------------------------------------------------------------+
                                                  vs
+-----------------------------------------------------------------------------------------------------+
| PRETRAINED XGBOOST (MAE: 6.440%)                                                                    |
| • Decisively BEATS Persistence on Zone 1 (0.165% vs 0.424% MAE, 11.4x lower MSE).                   |
| • Fails on Zones 4 and 5 due to tree extrapolation boundary (cannot predict below training min).     |
| • Concentrated 96% of split importance on 15-hour lags, causing lag latency during steep dry-down.  |
| • 90-99% of MSE on Zones 4 & 5 is pure static level bias (+7.29% and +15.16%).                      |
+-----------------------------------------------------------------------------------------------------+
                                                  vs
+-----------------------------------------------------------------------------------------------------+
| PRETRAINED GRU (MAE: 5.784%)                                                                        |
| • Better than XGBoost on non-linear extrapolation (Zone 5 MAE: 12.29% vs 15.16%).                    |
| • Suffered from Huber loss gradient saturation (delta=1.0 on unscaled target range 0-100).         |
| • Overpredicted moisture due to unobserved end-of-season irrigation abandonment.                    |
+-----------------------------------------------------------------------------------------------------+
```

---

## 7. Recommendations for Future Model Refactoring

*(Note: These are architectural recommendations for future iterations; no retraining or code modification was performed).*

1. **Reformulate as Residual Delta Prediction:**
   Train models to predict the 24-hour moisture change $\Delta y = y_{t+24\text{h}} - y_t$ instead of absolute moisture $y$.
   - A model predicting $\widehat{\Delta y} = 0$ will match Persistence at worst.
   - The ML model will then only need to learn the deviations caused by weather forcing, evapotranspiration, and past irrigation.
2. **Zone-Specific Models or Relative Moisture Scaling:**
   Instead of pooling raw moisture percentages across heterogeneous crops, normalize soil moisture relative to crop-specific water retention:
   $$\text{Relative Moisture} = \frac{\text{Moisture} - \text{PWP}}{\text{FC} - \text{PWP}}$$
   This maps all crops to an identical $[0, 1]$ operational scale, eliminating multi-modal pooling distortion.
3. **Target Scaling for Neural Networks:**
   Normalize targets using training set statistics so `HuberLoss` or `MSELoss` gradients operate on $\mathcal{N}(0, 1)$ errors rather than raw percentages.
4. **Shift Feature Focus to Instantaneous Lags:**
   Regularize tree depth or drop intermediate historical lags to force decision trees to anchor predictions on $t=0$ rather than 15-hour-old lags.
