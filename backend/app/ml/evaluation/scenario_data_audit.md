# Synthetic NEST Scenario Data Audit Report

**Date of Audit:** November 10, 2026
**Audit Scope:** All 5 synthetic scenario datasets in `data/synthetic_nest/`
**API Endpoints Audited:** `GET /stability/scenarios` and `GET /stability/scenarios/{scenario_id}`
**Pipeline Evaluated:** Dual-Probe Signal Conditioning (`SoilMoistureStabilizer`: Median Filter + EWMA + Sudden Change Bypass + Hysteresis Switch)
**Target Crop:** Chilli (`crop_id: chilli`, Field Capacity = 35.0%, MADP = 20.0%, Hysteresis deadband = ±1.0%)

---

## 1. Executive Summary

This audit evaluates the integrity, statistical properties, API loading behavior, and cross-scenario characteristics of all five synthetic NEST telemetry datasets. Each scenario comprises **48 consecutive packets at a 30-minute cadence (24-hour total observation window)**, preserving all 21 NEST hardware telemetry fields plus 8 ground-truth and provenance metadata fields.

### Key Audit Findings:
1. **API Parity Confirmed (100% Match):** Each of the 5 REST API endpoints (`/stability/scenarios/{scenario_id}`) strictly loads and serves the exact corresponding CSV file from disk. All 48 rows, timestamps, and sensor values match without alteration or packet truncation.
2. **Signal Conditioning Correctness:** The dual-probe stabilization pipeline behaves as designed. Isolated hardware faults in Scenario 2 are eradicated (99.2% total variation reduction), wetting front dynamics in Scenario 4 are fast-tracked (5 sudden change bypass events), and sensor noise in steady-state scenarios is smoothed.
3. **Root Cause of Scenarios 1, 2, and 3 Similarity:** Scenarios 1, 2, and 3 exhibit pronounced visual and operational similarity on dashboard views due to:
   - **Identical Actuator Response:** All three scenarios result in **0 filtered actuator switches** (the valve remains continuously OFF / Standby across all 48 packets).
   - **Overlapping Baseline Ranges:** Scenarios 1 and 2 operate in nearly identical soil moisture bands (~21–25% for Probe 1, ~28–30% for Probe 2). In Scenario 2, 42 of 48 packets are clean baseline packets, and the median filter removes the 2 soil moisture spikes, rendering the stabilized trajectory indistinguishable from Scenario 1.
   - **Smooth Non-Infiltrating Dynamics:** None of Scenarios 1, 2, or 3 experience sudden infiltration steps ($|\Delta| \ge 3\%$), resulting in **0 sudden events detected** across all three scenarios.
   - **Shared Synthetic Seeds & Metadata:** All three share identical fixed environmental variables (0.0 mm rainfall, identical trace gas concentrations, identical device/farm IDs, and similar noon-peaking solar radiation curves).

---

## 2. Quantitative Comparison of Soil Moisture Sequences

The table below summarizes the quantitative properties of `soil_moisture_1` (shallow root zone, 0–7 cm), `soil_moisture_2` (deep root zone), the combined dual-probe average $((SM1 + SM2) / 2)$, and sudden step jumps ($|\Delta| = |SM_t - SM_{t-1}| > 3.0\%$) across all 48 packets.

### Cross-Scenario Statistical Matrix

| Metric | Scenario 1: Normal Diurnal | Scenario 2: Sensor Noise | Scenario 3: Gradual Drying | Scenario 4: Irrigation Event | Scenario 5: High Humidity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **File Name** | `scenario_01_normal.csv` | `scenario_02_sensor_noise.csv` | `scenario_03_gradual_drying.csv` | `scenario_04_irrigation_event.csv` | `scenario_05_high_humidity.csv` |
| **Packet Count** | 48 | 48 | 48 | 48 | 48 |
| **Time Window** | 24 h (30-min steps) | 24 h (30-min steps) | 24 h (30-min steps) | 24 h (30-min steps) | 24 h (30-min steps) |
| **SM1: Start $\to$ End** | 22.00% $\to$ 21.00% | 24.94% $\to$ 23.06% | 34.59% $\to$ 18.67% | 13.90% $\to$ 26.63% | 27.01% $\to$ 27.40% |
| **SM1: Min / Max** | 21.00% / 22.00% | 0.00% / 100.00% | 18.67% / 34.59% | 13.08% / 29.78% | 26.97% / 27.44% |
| **SM1: Mean $\pm$ Std** | 21.17% $\pm$ 0.31% | 25.05% $\pm$ 11.47% | 26.58% $\pm$ 4.67% | 25.08% $\pm$ 5.73% | 27.21% $\pm$ 0.14% |
| **SM1: Max Step $|\Delta|$** | 0.13% | 76.03% | 0.46% | 3.14% | 0.08% |
| **SM1: Jumps $> 3.0\%$** | **0** | **4** (indices 17, 18, 41, 42) | **0** | **1** (index 8) | **0** |
| **SM2: Start $\to$ End** | 30.85% $\to$ 28.47% | 28.84% $\to$ 27.25% | 33.78% $\to$ 18.01% | 15.88% $\to$ 29.59% | 29.54% $\to$ 29.29% |
| **SM2: Min / Max** | 28.47% / 30.85% | 0.00% / 100.00% | 18.01% / 33.78% | 15.21% / 32.13% | 29.29% / 29.75% |
| **SM2: Mean $\pm$ Std** | 29.54% $\pm$ 0.74% | 28.84% $\pm$ 11.13% | 25.87% $\pm$ 4.67% | 27.60% $\pm$ 6.02% | 29.50% $\pm$ 0.16% |
| **SM2: Max Step $|\Delta|$** | 0.15% | 72.04% | 0.48% | 2.98% | 0.08% |
| **SM2: Jumps $> 3.0\%$** | **0** | **4** (indices 17, 18, 41, 42) | **0** | **0** ($|\Delta| = 2.98\%$) | **0** |
| **Avg SM: Min / Max** | 24.73% / 26.43% | 0.00% / 100.00% | 18.34% / 34.19% | 14.14% / 30.96% | 28.27% / 28.44% |
| **Avg SM: Mean $\pm$ Std**| 25.36% $\pm$ 0.51% | 26.94% $\pm$ 11.29% | 26.23% $\pm$ 4.67% | 26.34% $\pm$ 5.87% | 28.35% $\pm$ 0.04% |
| **Pkts Below MADP (20%)**| 0 | 1 (packet 42) | 6 (packets 42–47) | 10 (packets 0–9) | 0 |
| **Pkts Below 19% (On-Thresh)**| 0 | 1 (packet 42) | 3 (packets 45–47) | 9 (packets 0–8) | 0 |
| **Raw Switches** | 0 | 2 (0% dropout) | 1 (crosses MADP) | 1 (irrigation cut-off)| 0 |
| **Filtered Switches** | **0** | **0** (spike removed) | **0** (lag prevents <19%)| **1** (clean shut-off)| **0** |
| **Sudden Events Detected**| **0** | **0** (cleaned before EWMA)| **0** (smooth drying) | **5** (active infiltration)| **0** |

---

## 3. Scenario-by-Scenario Audit Details

### 3.1 Scenario 1: Normal Diurnal Cycle (`scenario_01_normal.csv`)
- **Physical Dynamics:** Baseline soil moisture under normal evapotranspiration. Soil dries slowly at $-0.05\%$ per packet.
- **Generator Constraint Artifact:** Probe 1 started at 22.0% and was clipped by `np.clip(sm1, CHILLI_MADP + 1, CHILLI_FC - 1)` (clamped between $21.0\%$ and $34.0\%$). As a result, Probe 1 reaches $21.00\%$ around packet 20 and remains flat at exactly $21.00\%$ through packet 47.
- **Noise & Jumps:** Zero sudden jumps ($|\Delta| \le 0.13\%$). Standard deviation is minimal ($\pm 0.31\%$ on SM1, $\pm 0.74\%$ on SM2).
- **Actuator Behavior:** Average moisture stays safely above MADP ($24.73\% \ge 20.0\%$). Zero actuator switches (Actuator = OFF throughout).

### 3.2 Scenario 2: Sensor Noise & Spikes (`scenario_02_sensor_noise.csv`)
- **Physical Dynamics:** Clean baseline soil drying with 6 injected hardware faults at randomly selected packet indices (1, 18, 22, 32, 34, 42).
- **Fault Allocation Breakdown:**
  - Packet 1, 32, 34: Negative humidity faults ($\text{hum} = -5.0\%$). Soil moisture is unaffected.
  - Packet 22: High temperature fault ($\text{temp} = 59.03^\circ\text{C}$). Soil moisture is unaffected.
  - **Packet 18:** Sensor saturation spike ($\text{SM1} = 100.0\%$, $\text{SM2} = 100.0\%$). Step jumps: $+75.99\%$ into spike, $-76.03\%$ returning.
  - **Packet 42:** Sensor ground dropout ($\text{SM1} = 0.0\%$, $\text{SM2} = 0.0\%$). Step jumps: $-23.41\%$ into dropout, $+23.34\%$ returning.
- **Clean Baseline vs Outliers:** For 46 out of 48 packets, SM1 drifts cleanly between $23.06\%$ and $24.94\%$ (clean mean $23.95\% \pm 0.54\%$), while SM2 drifts between $27.25\%$ and $28.87\%$ (clean mean $28.01\% \pm 0.49\%$).
- **Stabilizer Impact:** Because the spikes are isolated single-packet anomalies, the 5-point median filter removes them completely. The resulting filtered trajectory is virtually indistinguishable from a clean, flat baseline at $24.0\%$ and $28.0\%$.

### 3.3 Scenario 3: Gradual Soil Drying (`scenario_03_gradual_drying.csv`)
- **Physical Dynamics:** High vapor pressure deficit under continuous sunshine ($30^\circ\text{C}$ mean temperature, $54.5\%$ mean humidity, peak solar radiation $657.6\,\text{W/m}^2$). Moisture drops steadily by $\approx -0.337\%$ per 30-minute packet from Field Capacity ($34.5\%$) down to below MADP ($18.34\%$).
- **Noise & Jumps:** Absolutely zero sudden jumps ($|\Delta| \le 0.48\%$). Smooth linear drift across all 48 packets.
- **Threshold Interaction:** Raw average moisture drops below MADP ($20.0\%$) at packet 42 ($19.95\%$) and below the hysteresis turn-on threshold ($19.0\%$) at packet 45 ($18.91\%$).
- **Filter Inertia & Zero Filtered Switches:** The EWMA filter ($\alpha = 0.25$) introduces natural phase lag during continuous monotonic descent. By packet 47, while raw average has reached $18.34\%$, the stabilized average has only declined to $19.75\%$. Because $19.75\% > 19.0\%$, the Schmitt trigger hysteresis switch **never triggers ON**. Filtered switches remain $0$.

### 3.4 Scenario 4: Irrigation Infiltration (`scenario_04_irrigation_event.csv`)
- **Physical Dynamics:** Active irrigation event. Soil begins depleted ($13.9\%$ SM1, $15.9\%$ SM2, well below MADP). Irrigation valves open at packet 8 and run through packet 13 (3 hours), injecting $+2.8\%$ moisture per packet before gradually settling.
- **Sudden Step Jumps:**
  - Packet 8 onset: SM1 jumps from $13.47\%$ to $16.61\%$ ($|\Delta| = 3.14\% > 3.0\%$).
  - Subsequent irrigation packets (8–13) register continuous rapid steps of $+2.7\%$ to $+2.9\%$ per 30 min.
- **Fast-Track Preservation:** The stabilizer's sudden-change detector recognizes the infiltration step and switches to $\alpha_{\text{fast}} = 0.95$ for 5 packets, preserving the true physical wetting front with zero delay.
- **Actuator Operation:** Latches ON at packet 0 ($14.89\% < 19.0\%$), remains ON through irrigation, and cleanly turns OFF at packet 11 after stabilized moisture safely exceeds the Field Capacity release threshold ($21.0\%$).

### 3.5 Scenario 5: High Humidity & Leaf Wetness (`scenario_05_high_humidity.csv`)
- **Physical Dynamics:** Overcast, pre-monsoon conditions with high relative humidity ($82.4\%$ to $96.5\%$, mean $88.4\%$), high leaf wetness ($60$ to $100$), cool temperatures ($23.1^\circ\text{C}$), and low solar radiation ($41.5\,\text{W/m}^2$).
- **Soil Moisture Stability:** Condensation and atmospheric saturation completely suppress evapotranspiration. Soil moisture remains rock-solid between $27.0\%$ and $27.4\%$ on SM1, and $29.3\%$ and $29.8\%$ on SM2.
- **Noise & Jumps:** Zero sudden jumps ($|\Delta| \le 0.08\%$, lowest variance across all scenarios). Total variation reduction is $63.3\%$.

---

## 4. API Endpoint Verification

All API endpoints were audited against live HTTP requests and compared byte-by-byte with disk CSV files:

```
[Endpoint Check]
GET /stability/scenarios
Status: 200 OK
Payload: 5 ScenarioSummary records, all marked "SYNTHETIC"

[Per-Scenario Verification Matrix]
1. scenario_01_normal:
   - Route: GET /stability/scenarios/scenario_01_normal
   - Target File: data/synthetic_nest/scenario_01_normal.csv
   - Result: 48/48 packets identical match (timestamps, SM1, SM2, temperature, metadata).
2. scenario_02_sensor_noise:
   - Route: GET /stability/scenarios/scenario_02_sensor_noise
   - Target File: data/synthetic_nest/scenario_02_sensor_noise.csv
   - Result: 48/48 packets identical match. Spikes at packets 18 and 42 faithfully preserved.
3. scenario_03_gradual_drying:
   - Route: GET /stability/scenarios/scenario_03_gradual_drying
   - Target File: data/synthetic_nest/scenario_03_gradual_drying.csv
   - Result: 48/48 packets identical match. Steady drying slope confirmed.
4. scenario_04_irrigation_event:
   - Route: GET /stability/scenarios/scenario_04_irrigation_event
   - Target File: data/synthetic_nest/scenario_04_irrigation_event.csv
   - Result: 48/48 packets identical match. Infiltration step response confirmed.
5. scenario_05_high_humidity:
   - Route: GET /stability/scenarios/scenario_05_high_humidity
   - Target File: data/synthetic_nest/scenario_05_high_humidity.csv
   - Result: 48/48 packets identical match. High-humidity plateau confirmed.
```

**Conclusion:** Zero discrepancies between the CSV storage layer and the REST API responses.

---

## 5. Investigation: Why Scenarios 1, 2, and 3 Appear Similar

Users and observers inspecting dashboard views frequently note that Scenarios 1, 2, and 3 look surprisingly alike. The root causes are analyzed below across five distinct dimensions:

### 1. Identical Actuator State Timeline (Completely Flat OFF / 0 Line)
On the dashboard's **Irrigation Actuator Timeline** graph:
- **Scenario 1:** Actuator remains `0` (OFF) for all 48 packets because moisture never approaches MADP.
- **Scenario 2:** Raw data has 2 momentary transitions due to the $0\%$ dropout spike at packet 42, but the stabilizer's median filter eliminates the spike entirely. Thus, the filtered actuator decision is `0` (OFF) for all 48 packets.
- **Scenario 3:** While raw average drops below the $19.0\%$ activation threshold during the final 3 packets ($18.91\% \to 18.34\%$), EWMA filter phase lag ($\alpha = 0.25$) holds the stabilized average at $19.75\%$. Because $19.75\% > 19.0\%$, the actuator **never activates**.
- **Result:** In all three scenarios, the Actuator Timeline is an **identical horizontal line at 0 (STANDBY)**.

### 2. Overlapping Moisture Bands and Baseline Trajectories
- In **Scenario 1**, Probe 1 is flat at $21.0\%$ (clamped by generator artifact) and Probe 2 is around $29.5\%$.
- In **Scenario 2**, the underlying baseline generator initialized Probe 1 at $25.0\%$ and Probe 2 at $29.0\%$. Except for packets 18 ($100\%$) and 42 ($0\%$), all 46 packets drift gently around $24.0\%$ (SM1) and $28.0\%$ (SM2).
- In **Scenario 3**, Probe 1 starts at $34.5\%$ and dries to $18.7\%$. Between hours 6 and 18 (packets 12 to 36), Probe 1 traverses from $28\%$ down to $22\%$—**directly through the exact moisture band of Scenario 1 and Scenario 2**.
- When visualized on a $0–100\%$ scale or autoscaled chart with filtered overlays, the central portion of Scenario 3 closely mimics the stationary trajectories of Scenarios 1 and 2.

### 3. Suppression of Outliers Leaves Scenario 2 Looking Identical to a Flat Baseline
- Scenario 2 only injected **two** soil moisture faults (one saturation at packet 18, one dropout at packet 42). The other four faults were injected into temperature ($59^\circ\text{C}$) and relative humidity ($-5.0\%$).
- The 5-point median filter in `SoilMoistureStabilizer` completely excises both single-sample spikes.
- Consequently, the **stabilized curve** for Scenario 2 is an ultra-smooth, gentle horizontal line ($24.0\%$ for Probe 1, $28.0\%$ for Probe 2), virtually indistinguishable in curvature from Scenario 1.

### 4. Zero Sudden Infiltration Events in All Three Scenarios
- The `SoilMoistureStabilizer` classifies any $|\Delta| \ge 3.0\%$ step as a sudden wetting event.
- In Scenario 1, the max step is $0.13\%$. Sudden events = $0$.
- In Scenario 2, the median filter eliminates outliers before the EWMA sudden-change detector evaluates steps. Sudden events = $0$.
- In Scenario 3, the drying rate is smooth and continuous ($-0.33\%$ per packet, max step $0.48\%$). Sudden events = $0$.
- In contrast, Scenario 4 records **5 sudden change events**. On the metrics summary cards, Scenarios 1, 2, and 3 all report **`Sudden Events Detected: 0`**.

### 5. Static Metadata and Uniform Environmental Forcing
All three scenarios were constructed by mutating a single base sample (`data/company_sample/nest_sample.json`):
- Identical `device_id` (`SN-001`), `farm_id` (`FARM-01`), and `crop_id` (`chilli`).
- Identical start timestamp (`2026-06-15T14:00:00`) and 30-minute progression.
- Identical trace gas concentrations ($CO \approx 0.04$, $NO_2 \approx 0.03$, $O_3 \approx 0.04\,\text{ppm}$, $PM2.5 \approx 48\,\mu\text{g/m}^3$).
- Zero rainfall ($0.0\,\text{mm}$) across all 48 packets for all three datasets.
- Similar sinusoidal diurnal solar radiation profiles peaking at noon.

---

## 6. Summary Comparison Table: Scenarios 1, 2, and 3

| Feature | Scenario 1 (Normal) | Scenario 2 (Sensor Noise) | Scenario 3 (Gradual Drying) | Contrast with Scenario 4 (Irrigation) |
| :--- | :---: | :---: | :---: | :---: |
| **Raw SM1 Trajectory** | $22.0\% \to 21.0\%$ (clamped flat) | $24.9\% \to 23.1\%$ (+ 2 spikes) | $34.6\% \to 18.7\%$ (continuous) | $13.9\% \to 26.6\%$ (steep step) |
| **Stabilized SM1 Trajectory** | Smooth line at $21.1\%$ | Smooth line at $24.0\%$ | Smooth line $34.5\% \to 19.8\%$ | Fast-track jump $+13\%$ |
| **Raw SM2 Trajectory** | $30.8\% \to 28.5\%$ | $28.8\% \to 27.2\%$ (+ 2 spikes) | $33.8\% \to 18.0\%$ (continuous) | $15.9\% \to 29.6\%$ (steep step) |
| **Stabilized SM2 Trajectory** | Smooth line at $29.5\%$ | Smooth line at $28.0\%$ | Smooth line $33.8\% \to 19.7\%$ | Fast-track jump $+14\%$ |
| **Sudden Steps Detected** | **0** | **0** | **0** | **5** (active infiltration) |
| **Filtered Actuator Switches** | **0** (Always OFF) | **0** (Always OFF) | **0** (Always OFF) | **1** (ON $\to$ OFF) |
| **Filtered Actuator State** | Standby (0) | Standby (0) | Standby (0) | Active Irrigation $\to$ Standby |
| **High-Freq Jitter Reduction**| 96.22% | 100.00% | 94.02% | -45.02% (preserves front) |
| **Rainfall** | 0.0 mm | 0.0 mm | 0.0 mm | 0.0 mm (drip irrigation) |
| **Visual Dashboard Profile** | Flat / Steady | Flat / Steady (post-filter)| Gently sloped / Steady | Dynamic step rise & valve shut-off |

---

## 7. Recommendations for Future Dataset Iterations (Informational Only)

Per instructions, no files or datasets have been altered. If synthetic dataset generators are enhanced in future revisions, consider the following refinements:
1. **Extend Scenario 3 Drying Window or Steepness:** Allow Scenario 3 to drop to $\approx 15\%$ or extend to 60 packets so that stabilized moisture drops decisively below the $19.0\%$ hysteresis turn-on threshold, allowing users to observe the filtered actuator turning ON.
2. **Increase Scenario 2 Soil Moisture Fault Density:** Add 2–3 additional multi-packet moisture bursts or sensor dropouts so that raw data has more visible soil moisture divergence from the clean baseline.
3. **Unclamp Scenario 1:** Widen the lower clip threshold in `generate_synthetic.py` so Probe 1 does not flatten artificially at $21.00\%$ after packet 20.
