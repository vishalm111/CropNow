# NEST → Model Feature Mapping Audit

**Audit scope:** `backend/app/services/inference.py`
**Reference sources:** `pipeline.py`, `residual_pipeline.py`, `DATA_DICTIONARY.md`, scaler fitted statistics
**Audit date:** 2026-11-08
**Auditor:** Automated — `data/synthetic_nest/mapping_audit.py`

---

## Executive Summary

| Severity | Count | Category |
|---|---|---|
| 🔴 **CRITICAL** | 2 | Wrong unit · Wrong formula — produce severely distorted scaled inputs |
| 🟠 **HIGH** | 2 | Structurally absent features filled with biased constants |
| 🟡 **MEDIUM** | 3 | Assumption gaps that reduce accuracy but don't corrupt the pipeline |
| 🟢 **LOW** | 2 | Minor design decisions with negligible numerical effect |

---

## CRITICAL Issues

### C1 — `weather_pressure`: unit mismatch (kPa vs hPa)

| Item | Value |
|---|---|
| **NEST field** | `pressure` = 100.8 **kPa** |
| **Training column** | `weather_pressure` — ERA5 data in **hPa (millibars)** |
| **Scaler fitted mean** | 1008.97 hPa  ·  std = 3.04 |
| **Training data range** | 997.9 – 1029.4 hPa (Zone 1) |
| **What inference sends** | 100.8 — which after StandardScaler becomes (100.8 − 1009.0) / 3.04 = **−299 σ** |
| **Effect** | A z-score of −299 σ is catastrophically out of distribution. Both XGBoost (tree extrapolation) and GRU (activation saturation) will produce undefined behaviour on every packet. |

**Root cause:** NEST device reports absolute atmospheric pressure in kPa (SI unit); ERA5 reanalysis weather data is in hPa. The two are equivalent at scale factor × 10 (100.8 kPa = 1008 hPa). The pipeline never normalises units; the difference is baked into the scaler.

**Required fix:** Multiply `pressure` by 10 before passing to the feature vector:
```python
pkt.get("pressure", 101.3) * 10.0,  # kPa → hPa
```

---

### C2 — `relative_moisture`: wrong denominator formula

| Item | Value |
|---|---|
| **Training formula** (`residual_pipeline.py` line 94) | `(soil_moisture − PWP) / (FC − PWP)` |
| **Inference formula** (`inference.py` line 346) | `(sm_avg − MADP) / (FC − MADP)` |
| **Scaler fitted mean** | 0.3705  ·  std = 0.3669 |
| **Example (chilli SM=26.5 %)** | Training → 0.6600;  Inference → 0.4333 (34% relative error) |

**Root cause:** The inference formula substitutes `MADP` for `PWP` in both the numerator offset and the denominator. This shifts and rescales the agronomic coordinate, producing a systematically incorrect z-score for every prediction.

**Required fix:**
```python
# Use PWP, not MADP, to match training formula
relative_moisture = (sm_avg - thresh["pwp"]) / max(thresh["fc"] - thresh["pwp"], 1.0)
```

---

## HIGH Issues

### H1 — `ec`: wrong default value (unit confusion)

| Item | Value |
|---|---|
| **NEST payload** | `ec` field absent — not measured by NEST devices |
| **Training column** | `ec` — soil electrical conductivity in **µS/cm** |
| **Scaler fitted mean** | 181.7 µS/cm  ·  std = 117.1 |
| **Training data range** | 97 – 679 µS/cm (Zone 1) |
| **Inference default** | `6.5` — which after scaling becomes (6.5 − 181.7) / 117.1 = **−1.496 σ** |
| **Effect** | The default sits 1.5 σ below training mean on every packet, biasing the model as if soil EC were always extremely low. This is not a neutral imputation — it corresponds to the 7th percentile of training EC values. |

**Root cause:** The comment in inference.py says `# ec (neutral agronomic default)` and assigns 6.5, apparently confusing this with pH scale. EC is measured in µS/cm (hundreds), not 0–14 pH units.

**Required fix:** Use the training mean as the neutral imputation:
```python
181.7,  # ec — training-set mean (µS/cm); NEST device does not measure soil EC
```

---

### H2 — `water_vol_past_4h`: zero default is structurally biased

| Item | Value |
|---|---|
| **NEST payload** | Absent — NEST tracks soil moisture, not valve volumes |
| **Training column** | `water_vol_past_4h` — litres applied in past 4 h |
| **Scaler fitted mean** | 1.743 L  ·  std = 6.175 |
| **Inference default** | `0.0` → scaled to (0 − 1.74) / 6.18 = **−0.282 σ** |
| **Effect** | A zero is in-distribution (most packets have no irrigation), so the scaled value (−0.28 σ) is acceptable. However, it means the model always "sees" no recent irrigation history, which may underestimate post-irrigation drying velocity in sequences that follow a real event. This is an inherent NEST schema limitation, not a code bug. |

**Recommendation:** Document as a known structural gap. The default 0.0 is the best feasible approximation given NEST's schema, but callers should be warned that irrigation-event predictions may be biased.

---

## MEDIUM Issues

### M1 — `soil_temperature_7-18cm`: mirrored from single NEST probe

| Item | Value |
|---|---|
| **NEST payload** | Single `soil_temperature` field (depth not specified) |
| **Training columns** | Two distinct probes: `soil_temperature_0-7cm` and `soil_temperature_7-18cm` |
| **Scaler means** | 0–7 cm: 29.07 °C  ·  std 4.77 |  7–18 cm: 28.68 °C  ·  std **1.53** |
| **Inference behaviour** | Sends identical value to both columns |
| **Effect** | The 7–18 cm probe has much lower std (1.53 vs 4.77) because it tracks diurnal cycles with a lag and muted amplitude. By mirroring the surface reading, the deep probe channel sees artificially high variance. After scaling, every packet with temperature away from mean produces a z-score ~3× larger than expected in the deep-probe column. |

**Recommendation:** Use a smoothed/lagged estimate for `soil_temperature_7-18cm` if NEST provides timestamped history. With a single point, mirroring is unavoidable — but callers should be aware of the inflated z-score.

---

### M2 — `irrigation_duration_minutes`: zero default is plausible but biased toward no-irrigation

| Item | Value |
|---|---|
| **NEST payload** | Absent |
| **Training mean** | 0.188 min/packet  ·  std 1.682 |
| **Training range** | 0 – 10 min/packet |
| **Inference default** | 0.0 → scaled to (0 − 0.188) / 1.682 = −0.112 σ |
| **Effect** | Mild. The zero is the mode of training data (most bins have no irrigation), so the imputation is statistically reasonable. The model will never infer an ongoing irrigation event from NEST data alone. |

**Recommendation:** Accept the 0.0 default; document the limitation. If NEST devices are co-located with smart valves, integrate valve-open duration into the packet schema.

---

### M3 — Chilli crop thresholds lie completely outside training distribution

| Item | Chilli inference | Training range (all zones) |
|---|---|---|
| `pwp` | 10.0 | 20 – 40 |
| `madp` | 20.0 | 40 – 80 |
| `fc`  | 35.0 | 60 – 90 |
| `sp`  | 45.0 | 80 – 95 |
| `flow_rate` | 1.0 | 0.4 – 1.1 |

**Scaler fitted mean for MADP** = 59.79 ± 17.98. Chilli MADP = 20.0 → z-score = (20 − 59.8) / 18.0 = **−2.21 σ**.
**Scaler fitted mean for FC** = 76.76 ± 14.90. Chilli FC = 35.0 → z-score = (35 − 76.8) / 14.9 = **−2.81 σ**.

**Effect:** Every chilli inference packet carries static columns that are 2–3 σ out of the training distribution. The models have never seen crop parameters this low. XGBoost's leaf values are anchored to training data ranges; GRU hidden states may saturate.

**Recommendation:** Treat chilli (and any non-tomato/zucchini/blueberry crop) predictions as **unreliable until retrained on data from that crop**. Surface a warning in the API response when `crop_id` is not in the training set.

---

## LOW Issues

### L1 — `wind_speed`: unit assumption unverified

| Item | Value |
|---|---|
| **NEST field** | `wind_speed` — unit not documented in `nest_sample.json` |
| **Training column** | `weather_wind_speed` — ERA5 data in **m/s** |
| **Scaler fitted mean** | 10.99 m/s  ·  std 5.44 |
| **NEST sample value** | 1.4 |
| **If NEST is in km/h** | 1.4 km/h = 0.39 m/s — scaled to (0.39 − 11.0) / 5.44 = −1.95 σ |
| **If NEST is in m/s** | 1.4 m/s — scaled to (1.4 − 11.0) / 5.44 = −1.76 σ |

**Effect:** Either way the z-score is similar (−1.76 to −1.95 σ) because the sample value is a light breeze in any unit. However if NEST reports in km/h and a typical windy packet is 30 km/h (8.3 m/s), the scaled value would be −0.49 σ (correct) vs (30 − 11) / 5.44 = +3.49 σ (severely wrong if treated as m/s). The unit of `wind_speed` must be confirmed with the NEST hardware team.

---

### L2 — `ph` default is acceptable but slightly high

| Item | Value |
|---|---|
| **Inference default** | 7.0 |
| **Training mean** | 5.97  ·  std 0.82 |
| **Scaled default** | (7.0 − 5.97) / 0.82 = **+1.26 σ** |

**Effect:** A pH of 7.0 is 1.26 σ above the training mean (5.97). Using the training mean (6.0) instead of 7.0 would produce a neutral z-score of zero. This is a minor inaccuracy with small practical effect given pH's low predictive importance.

---

## Feature Coverage Matrix

| Training feature | NEST field | Status | z-score at NEST sample |
|---|---|---|---|
| `weather_temp` | `temperature` | ✅ Direct | (28 − 26.5) / 3.96 = +0.38 |
| `weather_humidity` | `humidity` | ✅ Direct | (72 − 61.0) / 18.9 = +0.58 |
| `weather_rain` | `rainfall` | ✅ Direct (sum semantics match) | (0 − 0.12) / 1.46 = −0.08 |
| `weather_pressure` | `pressure` | 🔴 **C1: kPa vs hPa** | (100.8 − 1009.0) / 3.04 = **−299** |
| `weather_wind_speed` | `wind_speed` | 🟡 L1: unit unverified | ≈ −1.8 |
| `weather_radiation` | `solar_radiation` | ✅ Direct (W/m² matches) | (420 − 200) / 244.8 = +0.90 |
| `soil_temperature_0-7cm` | `soil_temperature` | ✅ Acceptable | (24 − 29.1) / 4.77 = −1.07 |
| `soil_temperature_7-18cm` | `soil_temperature` (mirrored) | 🟡 M1: incorrect std | Inflated by 3× |
| `ec` | *(absent)* | 🟠 **H1: 6.5 wrong unit** | (6.5 − 181.7) / 117.1 = **−1.50** |
| `ph` | *(absent)* | 🟢 L2: slightly high default | (7.0 − 5.97) / 0.82 = +1.26 |
| `soil_moisture` | avg(`soil_moisture_1`, `soil_moisture_2`) | ✅ Correct averaging | — |
| `irrigation_duration_minutes` | *(absent)* | 🟡 M2: zero acceptable | −0.11 |
| `water_vol_past_4h` | *(absent)* | 🟠 **H2: structural gap** | −0.28 |
| `relative_moisture` | derived | 🔴 **C2: wrong formula** | 34% relative error |
| `dist_to_madp` | derived | ✅ Formula correct | — |
| `dist_to_fc` | derived | ✅ Formula correct | — |
| `flow_rate` | crop thresholds | 🟡 M3: OOD crop | — |
| `pwp` | crop thresholds | 🟡 M3: OOD — z=−2.1 | — |
| `madp` | crop thresholds | 🟡 M3: OOD — z=−2.2 | — |
| `fc` | crop thresholds | 🟡 M3: OOD — z=−2.8 | — |
| `sp` | crop thresholds | 🟡 M3: OOD — z=−2.9 | — |

---

## Column Order Verification

```
Residual column order (inference.py vs residual_pipeline.py): ✅ MATCH
Absolute column order (inference.py vs pipeline.py):          ✅ MATCH
```

---

## Absent NEST Fields (not used by models — no issue)

| NEST field | Disposition |
|---|---|
| `device_id`, `farm_id` | Identity only — not a model feature |
| `crop_id` | Used to look up thresholds — not a direct feature |
| `month`, `day`, `hour` | Calendar features excluded from training (`pipeline.py` FEATURE_COLS does not include them) |
| `leaf_wetness` | Not in training features — NEST-specific, no model column |
| `co`, `no2`, `o3`, `pm2_5` | Air quality — not in training features |
| `wind_direction` | Not in training features |

---

## Recommended Fixes (Priority Order)

| Priority | Issue | Fix |
|---|---|---|
| 1 | 🔴 C1 — pressure unit | Multiply `pressure` × 10 before feature vector |
| 2 | 🔴 C2 — relative_moisture formula | Change denominator to `FC − PWP`; numerator offset to `PWP` |
| 3 | 🟠 H1 — ec default | Change default from `6.5` to `181.7` (training µS/cm mean) |
| 4 | 🟠 H2 — water_vol_past_4h | Acceptable as 0.0; add API warning for irrigation-event packets |
| 5 | 🟡 M3 — OOD crops | Add API warning when `crop_id` ∉ {`tomato`, `zucchini`, `blueberry`} |
| 6 | 🟡 L2 — ph default | Change from `7.0` to `5.97` (training mean) |
| 7 | 🟡 L1 — wind_speed unit | Confirm NEST hardware unit with vendor; convert if km/h |

> [!CAUTION]
> Issues C1 and C2 are active in production inference right now. Every call to `/predict/quick` or `/predict/batch` with a NEST packet is producing severely distorted inputs for `weather_pressure` (−299 σ) and `relative_moisture`. Fixing these two issues should be the first priority before any real-device integration.

> [!NOTE]
> The column ordering between `inference.py` and both training pipelines is verified correct. No array-index transposition errors exist.

> [!IMPORTANT]
> Do **not** retrain models to fix these issues. All six problems are purely in the inference mapping layer (`_packet_to_features`). The models themselves, their weights, and their scalers are correct and must remain unchanged.
