# NEST Agricultural Prediction System — Demo Guide

This guide walks through starting the backend and frontend locally, demonstrating all three agricultural predictions (**Pest Risk**, **Irrigation**, **Fungal Risk**), showcasing why missing history is never fabricated, and inspecting the preserved soil moisture stability charts.

---

## 1. Prerequisites

- **Python:** `3.11+` (`python3.11` / `venv`)
- **Node.js & npm:** Required for Vite React frontend (`Node 18+` recommended; verified on Node `v24.19.0`)
- **Workspace root:** `CropNow/`

---

## 2. Environment Setup & Running Locally

Open **two terminals** from the repository root (`CropNow/`):

### Terminal 1 — Start the FastAPI Backend (`127.0.0.1:8000`)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Quick checks once started:
- Root metadata: `http://127.0.0.1:8000/`
- Health check: `http://127.0.0.1:8000/health`
- Swagger API UI: `http://127.0.0.1:8000/docs`

### Terminal 2 — Start the React Dashboard (`http://localhost:5173/`)

```powershell
cd frontend
npm install
# Optional: copy .env.example to .env if using a non-default backend URL
npm run dev
```

Open **`http://localhost:5173/`** in your browser. Confirm the top card displays a green **Connected** badge alongside response latency and service metadata (`v0.1.0`).

---

## 3. Live Dashboard Walkthrough

### Part A — Initial State & Limitations Transparency
Before clicking any analysis button:
- All three cards inside **Unified Agricultural Intelligence (`POST /predict/agricultural`)** display **`INSUFFICIENT HISTORY`** with `0 / 1,344 pkts` (**Pest Risk**), `0 / 48 pkts` (**Irrigation**), and `0 / 336 pkts` (**Fungal Risk**).
- The **Synthetic Telemetry & Cross-Crop Domain Limitations** banner upfront explains:
  - Built-in scenarios contain `48` synthetic `30-minute` NEST telemetry packets (`24h`), and missing multi-week history is never invented or padded.
  - Biological risk models were trained on Colorado wild flax rust (*Melampsora lini*) and Rice Black Bug (`RBB`), not field chilli crops.

---

### Part B — Demo the 5 Built-In Synthetic Scenarios (`48` Packets = `24h`)
Inside the **Soil Moisture Stability & Signal Conditioning** card:
1. Choose a scenario from **Select Synthetic NEST Scenario** and keep `xgb_residual (Recommended)` selected under **Irrigation Model**.
2. Click **`Run Stability Analysis`**.
   The frontend calls `GET /stability/scenarios/{id}` for the 48 synthetic packets and concurrently posts them to `POST /stability/analyze` and `POST /predict/agricultural`.

#### Recommended Scenarios to Show During Demo (`SYNTHETIC` 48-Packet Scenarios)
| Scenario | What to Point Out on Prediction Cards (`POST /predict/agricultural`) | What to Point Out on Stability Graphs (`POST /stability/analyze`) |
| :--- | :--- | :--- |
| **`Scenario 1: Normal Diurnal Cycle`** | **Irrigation:** `READY` (`48/48 pkts`), `NOT REQUIRED` (`24.8%` stabilized, `22.7%` 24h forecast), confidence `70.0% (MEDIUM)`, warning noting 5 unavailable training channels imputed at `z=0.0`.<br>**Fungal Risk:** `INSUFFICIENT HISTORY` (`48/336 pkts — 14%`).<br>**Pest Risk:** `INSUFFICIENT HISTORY` (`48/1,344 pkts — 4%`). | Baseline steady signal (~`21%` Probe 1, ~`29.5%` Probe 2), `96.2%` high-freq jitter reduction, actuator stays `VALVE CLOSED (OFF)`. |
| **`Scenario 2: Sensor Noise & Spikes`** | **Irrigation:** **`BLOCKED`** (`confidence: Unavailable`) because physical-range anomaly guards reject synthetic fault packets (`-5.0%` humidity / extreme faults) instead of issuing an unsafe ML forecast.<br>**Fungal / Pest Risk:** `INSUFFICIENT HISTORY` (`48/336`, `48/1,344`). | **Sensor 1 & 2 Charts:** Even when ML forecasting blocks on sensor faults, signal conditioning smooths the `100%` (packet 18) & `0%` (packet 42) spikes (`99.8%` outlier suppression, `2 -> 0` actuator switches). |
| **`Scenario 3: Gradual Soil Drying`** | **Irrigation:** `READY` (`48/48 pkts`), **`REQUIRED`** (`19.8%` stabilized < `20.0%` MADP; `18.3%` 24h forecast), confidence `70.0% (MEDIUM)`, while **Actuator State** stays `VALVE CLOSED (OFF)` (`19.8% > 19.0%` turn-on threshold). | Demonstrates decoupled advisory (`STRESS`) vs. Schmitt hysteresis preventing premature relay chatter (`0` filtered switches). |
| **`Scenario 4: Irrigation Infiltration`** | **Irrigation:** `READY` (`48/48 pkts`), `NOT REQUIRED` at window end (`28.5%` stabilized, `26.0%` 24h forecast after active wetting replenishment). | EWMA fast-track (`|\Delta| >= 3.0%`) detects `5` sudden wetting events (`95.5%` genuine range retention, `60.0 min` overall trajectory lag) and **Actuator Timeline** steps cleanly `ON (1) -> OFF (0)`. |

*(Note: Because Scenarios 1, 2, and 3 all keep the filtered valve `OFF (0)` and have zero infiltration steps, their Actuator Timeline charts look similarly flat — **Scenario 4** is best for showing dynamic valve transitions and fast-tracking.)*

---

### Part C — Demo Multi-Day (`336` Packets) & Multi-Week (`1,344` Packets) Predictions (Synthetic Test JSON)
Because built-in scenarios hold only `48` packets, use **`Upload Packet JSON (48 / 336 / 1,344)`** to demonstrate full 3-model readiness without fabricating history.

> **Important Provenance & Anti-Fabrication Notice:**
> 1. The generator snippet below creates **SYNTHETIC multi-week test telemetry** strictly to verify `336`- and `1,344`-packet API/UI execution — it is **NOT real NEST hardware telemetry** and must never be presented as real field validation.
> 2. Do **not** clone or repeat a single 48-packet CSV to reach 336 or 1,344 packets. The backend detects duplicated consecutive sensor blocks and rejects fabricated history. Supply chronologically ordered 30-minute packets with varying weather/soil values.

You can generate sample test JSON payloads locally in a temporary scratch directory (without modifying repo files):

```powershell
python -c "
import json, math
from datetime import datetime, timedelta

def gen(n):
    t0 = datetime.fromisoformat('2026-06-01T00:00:00')
    out = []
    for i in range(n):
        wk, day, s = i // 336, i // 48, i % 48
        ts = t0 + timedelta(minutes=30*i)
        out.append({
            'timestamp': ts.isoformat(), 'device_id': 'NEST-DEMO', 'farm_id': 'farm-01', 'crop_id': 'chilli',
            'temperature': round(24.0 + wk*0.6 + 4.0*math.sin(math.pi*s/48.0), 4),
            'humidity': round(min(max(68.0 + (18.0 if s < 16 else 0.0) + wk, 0.0), 100.0), 4),
            'leaf_wetness': 25.0 if s < 12 else 0.0,
            'rainfall': 1.4 if (s == 0 and day % 2 == 1) else 0.0,
            'solar_radiation': round(max(0.0, 500.0*math.sin(math.pi*s/48.0)), 2),
            'soil_moisture_1': round(22.0 + 0.01*(i % 96) + wk*0.5, 3),
            'soil_moisture_2': round(22.5 + 0.01*(i % 96) + wk*0.5, 3),
            'soil_temperature': round(19.0 + 0.5*math.sin(math.pi*s/48.0), 3),
            'wind_speed': round(1.1 + wk*0.15 + (0.4 if i % 2 == 0 else -0.15), 4),
            'wind_direction': round((170.0 + wk*12.0 + (i % 5)) % 360.0, 2),
            'pressure': 100.8, 'co': 0.04, 'no2': 0.03, 'o3': 0.04, 'pm2_5': 42.0,
            'month': ts.month, 'day': ts.day, 'hour': ts.hour
        })
    return out

open('demo_336.json', 'w').write(json.dumps(gen(336)))
open('demo_1344.json', 'w').write(json.dumps(gen(1344)))
print('Generated demo_336.json and demo_1344.json')
"
```

1. **Upload `demo_336.json` (`336` packets = 7 days):**
   - Header badge updates to `UPLOADED JSON` (`336 Packets Evaluated • Crop: chilli`).
   - **Irrigation:** `READY` (`48 / 48 pkts`).
   - **Fungal Risk:** Transitions to **`READY`** (`336 / 336 pkts`), showing predicted tier (`LOW` / `MEDIUM` / `HIGH`), confidence %, class probability bars, `DOMAIN_LIMITATION (Flax Rust -> chilli)`, and `OUT_OF_DISTRIBUTION_CLIMATE` warnings.
   - **Pest Risk:** Stays **`INSUFFICIENT HISTORY`** (`336 / 1,344 pkts — 25%`).
   - **Stability Graphs:** Automatically update using the trailing `48` packets (`packets.slice(-48)`).
2. **Upload `demo_1344.json` (`1,344` packets = 28 days):**
   - **All three cards** transition to **`READY`**:
     - **Pest Risk:** `LOW` / `MEDIUM` / `HIGH` (`1,344 / 1,344 pkts`), confidence %, probability bars, and `DOMAIN_LIMITATION (Rice RBB -> chilli)`.
     - **Irrigation:** `REQUIRED` / `NOT REQUIRED` (`48 / 48 pkts`) + actuator valve state.
     - **Fungal Risk:** `LOW` / `MEDIUM` / `HIGH` (`336 / 336 pkts`) + flax-rust/climate warnings.

*(To return to synthetic scenarios afterwards, simply select a scenario from the dropdown and click `Run Stability Analysis`.)*

---

## 4. Direct API & Automated Test Commands

### Query `POST /predict/agricultural` via PowerShell
```powershell
$payload = @{
  irrigation_model = "xgb_residual"
  packets = (Get-Content .\demo_1344.json | ConvertFrom-Json)
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/agricultural" `
  -Method Post -ContentType "application/json" -Body $payload | ConvertTo-Json -Depth 4
```

### Run Full Verification Suites
```powershell
# Backend full test suite (112 tests covering mappers, stability, datasets, and /predict/agricultural)
python -m pytest backend/tests -v

# Frontend lint & production bundle verification
cd frontend
npm run lint
npm run build
```

---

## 5. Key Demo Talking Points (Honesty & Safety Highlights)

- **Independent Multi-Horizon Design:** A farmer with 24 hours (`48` packets) of telemetry immediately receives actionable soil moisture stability + 24h irrigation guidance without waiting 28 days, while multi-week biological models fail closed as `INSUFFICIENT HISTORY`.
- **Transparent Domain Boundaries:** UI banners and API `domain_limitations` prevent overclaiming cross-crop accuracy until real regional chilli telemetry and labeled chilli disease/pest observations are collected for re-calibration.
- **No Sensor Fault Relay Chatter:** Median + EWMA + Schmitt hysteresis ensures transient ADC spikes never trigger unnecessary pump cycling, while step-change bypass ensures real irrigation events appear with zero phase lag.
