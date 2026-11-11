# Data Architecture & Dataset Registry

This directory contains data assets for the **NEST Agricultural Prediction System**. To maintain data integrity, clean API contracts, and reproducible modeling pipelines, the data assets are strictly segregated into distinct logical layers.

---

## Directory Structure

```text
data/
├── company_sample/
│   └── nest_sample.json                   # Reference NEST IoT packet schema (local contract, excluded from public Git)
├── synthetic_nest/                        # Synthetic 5-scenario NEST telemetry CSVs & generator scripts (SYNTHETIC)
├── Soil Moisture, Irrigation Actuator and Weather Dat/   # External precision-irrigation field dataset (Mendeley Data, CC BY 4.0)
│   └── Soil Moisture, Irrigation Actuator and Weather Dat/
│       ├── 01_raw_data/                   # Untouched IoT sensor measurements & valve telemetry
│       ├── 02_processed_data/              # Validated events, merged 10-min grid, preprocessed tables
│       ├── 03_code/                       # Preprocessing pipeline scripts and agronomic configs
│       ├── README.md                      # Mendeley trial documentation
│       └── DATA_DICTIONARY.md             # Column-by-column descriptions
├── external/pest/fungal_flax_rust/        # External wild flax rust epidemiology & weather dataset (Dryad, CC0 1.0)
├── processed/fungal_risk/                 # Derived 7-day lookback fungal-risk feature table & fold thresholds (CC0 1.0)
└── README.md                              # This registry document
```

---

## 1. Company-Provided NEST Sample

- **Path:** [`data/company_sample/nest_sample.json`](file:///c:/Users/Vishal/Downloads/CropNow/data/company_sample/nest_sample.json)
- **Data Classification:** **Reference Schema / Static Contract** (Not historical time-series data).
- **Purpose:**
  - Serves as the single source of truth for the company's proprietary NEST IoT telemetry payload.
  - Defines the field names, structural keys, and expected data types (e.g. multi-depth soil moisture `soil_moisture_1` & `soil_moisture_2`, leaf wetness, particulate matter `pm2_5`, air quality metrics `co`, `no2`, `o3`, solar radiation, and microclimate parameters).
  - Used by API gateways, Pydantic ingestion models, and validation layers to verify payload format compliance prior to model inference.
  - **Note:** This single packet represents an instantaneous payload snapshot for a single crop (`chilli`) and device (`NEST-001`). It is not a historical time series and is not used for model training.

---

## 2. External Precision-Irrigation Dataset (Mendeley Data — CC BY 4.0)

- **Path:** [`data/Soil Moisture, Irrigation Actuator and Weather Dat/`](file:///c:/Users/Vishal/Downloads/CropNow/data/Soil%20Moisture,%20Irrigation%20Actuator%20and%20Weather%20Dat/)
- **Source & DOI:** Mendeley Data — [https://doi.org/10.17632/c837v6p8ph](https://doi.org/10.17632/c837v6p8ph) (V2 DOI: [10.17632/c837v6p8ph.2](https://doi.org/10.17632/c837v6p8ph.2))
- **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**) — sharing and adaptation permitted with attribution. Meteorological features are derived from Open-Meteo historical weather API (CC BY 4.0) / Copernicus ERA5 reanalysis.
- **Citation:**
  > Adamo, Tommaso; Colizzi, Lucio; Dimauro, Giovanni; Guerriero, Emanuela; Lomonte, Nunzia (2026). *Soil Moisture, Irrigation Actuator and Weather Dataset from a Multi-Sector Precision-Irrigation Field Trial*. Mendeley Data, DOI: `10.17632/c837v6p8ph` (`10.17632/c837v6p8ph.2`).
- **Data Classification:**
  - `01_raw_data/`: **Real Primary Data** (Untouched physical sensor logs and valve transitions from an experimental field trial in Arnesano, Province of Lecce, Apulia, Italy, February–September 2025; note `raw_actuators.csv` and `raw_measurements.csv` exceed 100 MiB and are omitted from Git).
  - `02_processed_data/`: **Derived Data** (Cleaned, quality-controlled, 10-minute resampled, ERA5 weather-merged, and feature-engineered for machine learning benchmarks).
- **Purpose:**
  - Provides ground-truth agricultural time-series data documenting crop soil response to measured water volumes applied via smart valves across 5 distinct sectors (Tomato, Zucchini, Blueberry).
  - Serves as the primary training and evaluation substrate for hydraulic forecasting models (predicting 24-hour forward soil moisture and agronomic water stress).
  - Maintained in its original directory structure and naming conventions to preserve full provenance and code compatibility.

---

## 3. External Wild Flax Rust Dataset (Dryad — CC0 1.0)

- **Path:** [`data/external/pest/fungal_flax_rust/`](file:///c:/Users/Vishal/Downloads/CropNow/data/external/pest/fungal_flax_rust/) *(derived feature table in [`data/processed/fungal_risk/`](file:///c:/Users/Vishal/Downloads/CropNow/data/processed/fungal_risk/))*
- **Source & DOI:** Dryad Digital Repository — [https://doi.org/10.5061/dryad.98sf7m0tc](https://doi.org/10.5061/dryad.98sf7m0tc)
- **License / Attribution:** **CC0 1.0 Universal (Public Domain Dedication)**. Zero copyright restrictions apply; please attribute the original authors and associated studies when referencing this dataset.
- **Citation:**
  > Duggal, Keenan; Jiranek, Juliana; Machado, Madison; Smith, Peyton; Miller, Ian F.; Metcalf, Jessica C. E. *Data from: A multi-year case study highlighting the influence of hydrological conditions on epidemic dynamics in a natural plant pathosystem*. Dryad Digital Repository, DOI: `10.5061/dryad.98sf7m0tc`.
  > Foundational field-site paper: Miller, I. F., Jiranek, J., Brownell, M. et al. (2022), *Predicting the effects of climate change on the cross-scale epidemiological dynamics of a fungal plant pathogen*, *Scientific Reports* 12, 14823 (`https://doi.org/10.1038/s41598-022-18851-z`); associated *Oikos* article DOI: `10.1111/oik.10845`.
- **Purpose:** Observational epidemiological surveys (`Cleaned_Epi.rds`) and 5-minute meteorological/leaf-wetness telemetry (`cleaned_temp.rds`, `cleaned_weather.rds`) across four Colorado subalpine sites (`BT`, `CC`, `GM`, `HM`), used to build the leak-free 7-day fungal-risk classifier dataset (`flax_rust_fungal_risk_dataset.csv`).

---

## 4. Synthetic NEST Data

- **Path:** [`data/synthetic_nest/`](file:///c:/Users/Vishal/Downloads/CropNow/data/synthetic_nest/)
- **Data Classification:** **Synthetic / Simulated Data** — generated programmatically from the company reference schema.
- **Generator script:** [`data/synthetic_nest/generate_synthetic.py`](file:///c:/Users/Vishal/Downloads/CropNow/data/synthetic_nest/generate_synthetic.py) (seed=42, fully reproducible)
- **Purpose:**
  - End-to-end integration testing, edge-case validation, and API mocking without touching production or empirical field data.
  - Each CSV contains 48 packets (30 min apart, 2026-06-15 14:00 → 2026-06-16 13:30) across 5 agricultural scenarios.

### Generated Scenarios

| File | Scenario | Packets | Key Characteristic |
|---|---|---|---|
| `scenario_01_normal.csv` | Normal readings | 48 | Stable diurnal cycles, SM between MADP and FC |
| `scenario_02_sensor_noise.csv` | Sensor noise/spikes | 48 | 6 injected outliers (dropout, saturation, temperature/humidity spikes) |
| `scenario_03_gradual_drying.csv` | Gradual soil drying | 48 | Starts at FC (34.5%), dries to below MADP by packet 42 |
| `scenario_04_irrigation_event.csv` | Irrigation event | 48 | Starts dry (<MADP), irrigation fires packets 8–13, moisture recovers |
| `scenario_05_high_humidity.csv` | High humidity & leaf wetness | 48 | Humidity 80–100%, leaf wetness 60–100, overcast / pre-monsoon |
| `generation_summary.csv` | Metadata summary | 5 rows | Per-scenario label counts and irrigation percentages |

### Ground-Truth Columns (added by generator, not in live telemetry)

| Column | Type | Description |
|---|---|---|
| `timestamp` | ISO-8601 | 30-min cadence start time |
| `packet_index` | int 0–47 | Packet sequence number |
| `scenario` | string | Scenario slug |
| `data_source` | string | Always `"SYNTHETIC"` |
| `gt_label` | string | `NORMAL`, `IRRIGATE`, `SENSOR_FAULT`, `FIELD_CAPACITY` |
| `gt_irrigate` | 0/1 | 1 when avg soil moisture < MADP (20 % for chilli) |
| `gt_soil_stress` | 0/1 | Same as `gt_irrigate` (ML target alias) |
| `gt_sensor_fault` | 0/1 | 1 for injected spike packets (Scenario 2 only) |

---

## 5. Data Provenance & Classification Summary

| Data Asset | Path | Classification / License | Modifiable? | Role in System |
|---|---|---|---|---|
| **NEST Sample Payload** | `data/company_sample/nest_sample.json` | **Proprietary Reference Contract** *(excluded from public Git)* | Read-Only | Local contract schema for live IoT ingestion & validation |
| **Field Sensor Raw & Merged Logs** | `.../Soil Moisture.../{01_raw,02_processed,03_code}` | **Real Public Data (`CC BY 4.0`, DOI `10.17632/c837v6p8ph`)** | Read-Only | Untouched logs (`raw_actuators`/`raw_measurements` >100 MiB excluded) & benchmark tables |
| **Wild Flax Rust Dataset & Derived CSV** | `data/external/pest/fungal_flax_rust/` & `data/processed/fungal_risk/` | **Real Public Data (`CC0 1.0`, DOI `10.5061/dryad.98sf7m0tc`)** | Read-Only | Epidemiological surveys & 7-day lookback fungal-risk training data |
| **Synthetic NEST Telemetry** | `data/synthetic_nest/` | **Synthetic (`SYNTHETIC`)** | Read-Only (Regenerate via script) | 5-scenario × 48-packet CSVs for integration testing and API mocking |

---

## 6. Architectural Isolation Rules

1. **Logical Separation:** Company IoT contracts (`company_sample/`) and external academic/field benchmarks (`Soil Moisture.../`) must remain separate.
2. **Schema Integrity:** The keys and data types declared in `nest_sample.json` are immutable reference specifications.
3. **No Cross-Contamination:** Synthetic generation routines must write exclusively to `synthetic_nest/` and never overwrite or mingle with empirical field measurements.
