# CropNow Repository Cleanup Audit

**Audit Date:** November 11, 2026
**Scope:** Full static audit of Python imports, FastAPI routes, React/Vite frontend references, model artifacts, datasets (`465.26 MB`), tests, dependencies, and build/test caches.
**Policy:** Read-only audit — **no files, code, models, datasets, or dependencies were removed or modified.**

---

## Executive Summary

| Finding Area | Key Observation |
| :--- | :--- |
| **Safe-to-Remove Files** | 5 unreferenced Vite/React starter assets (`hero.png`, `react.svg`, `vite.svg`, `icons.svg`, unlinked `favicon.svg`), regenerable build/test caches (`.pytest_cache/`, `__pycache__/`, `frontend/dist/`), and 3 dead exports/stubs (`getActiveBaseUrl`, `API_BASE_URL`, unused `api_router` in `routes/__init__.py`). |
| **Must Keep (Runtime & Models)** | All 4 saved model directories (`models/`, `models_residual/`, `models_fungal_risk/`, `models_pest_risk/`), FastAPI routes, mappers, offline training modules still imported at runtime/tests (`prepare_fungal_dataset.py`, `prepare_pest_dataset.py`, `train_fungal_risk.py`), all 8 pytest suites (`112` tests), React components, and the 5 scenario CSVs. |
| **Critical Dependency Gap (`backend/requirements.txt`)** | `backend/requirements.txt` lists only `fastapi`, `uvicorn[standard]`, and `pydantic` — missing `numpy`, `pandas`, `joblib`, `torch`, `xgboost`, `scikit-learn`, and `pytest` required by the running backend and test suite. |
| **Frontend Dependencies (`frontend/package.json`)** | Clean — `react`, `react-dom`, `recharts` + Vite/React types/`oxlint` are all actively used; **zero unused npm packages**. |
| **Largest Storage Footprint** | `data/` totals `465.26 MB` (`64` files), dominated by raw Arnesano trial logs (`377.49 MB` `raw_irrigation_actuators` + `49.90 MB` `raw_sensor`) not read at API runtime. |

---

## 1. Safe to Remove

These items were verified to have **zero active runtime, test, route, or frontend build dependencies** (or are 100% regenerable caches/artifacts):

### 1.1 Unreferenced Frontend Starter Assets
Static scans across all `.js`, `.jsx`, `.css`, and `.html` files in `frontend/` (excluding `dist/` and `node_modules/`) found zero references:

| Path | Size | Verification Evidence |
| :--- | :---: | :--- |
| `frontend/src/assets/hero.png` | ~18 KB *(part of `0.02 MB` `assets/`)* | `0` imports/references in `frontend/src/` or `index.html`; leftover Vite scaffold asset. |
| `frontend/src/assets/react.svg` | *in `assets/`* | `0` references anywhere in `frontend/`. |
| `frontend/src/assets/vite.svg` | *in `assets/`* | `0` references anywhere in `frontend/`. |
| `frontend/public/icons.svg` | in `public/` | `0` references in `index.html`, CSS, or JSX. |
| `frontend/public/favicon.svg` | in `public/` | `0` references in `frontend/index.html` *(either remove or link via `<link rel="icon" href="/favicon.svg">`)*. |

### 1.2 Dead Code / Unused Exports (Verified Zero Callers)
| Location & Symbol | Verification Evidence |
| :--- | :--- |
| `frontend/src/services/api.js` → `getActiveBaseUrl` & `API_BASE_URL` | `0` imports/uses anywhere in `frontend/src/` outside `api.js` following removal of the Backend Connection Status details card. |
| `backend/app/routes/__init__.py` → `api_router` | Creates an `APIRouter` registering only `health_router`; `backend/app/main.py` directly imports `health.router`, `prediction.router`, and `stability.router`. `0` external references to `api_router`. |
| `backend/app/ml/__init__.py` docstring | Outdated placeholder text ("will be implemented in subsequent phases"); safe to replace with an accurate package docstring or leave empty. |

### 1.3 Regenerable Caches & Build Artifacts
| Path | Status in `.gitignore` | Notes |
| :--- | :---: | :--- |
| `.pytest_cache/` (`0.01 MB`) | Tracked on disk | Regenerated automatically by `pytest`. |
| `**/__pycache__/` (`backend/app/`, `backend/tests/`, `data/synthetic_nest/`) | Ignored (`__pycache__/`) | Python bytecode cache; safely recreatable. |
| `frontend/dist/` (`5` files, `0.67 MB`) | Ignored (`dist/`) | Production bundle rebuilt anytime via `npm run build`. |

---

## 2. Must Keep

Removing or altering any item in this section will break FastAPI endpoints, model loading, feature mapping, UI rendering, or the `112`-test pytest suite.

### 2.1 Backend Runtime API, Schemas, Services & Mappers
| Path | Runtime / Test Dependency Role |
| :--- | :--- |
| `backend/app/main.py`, `backend/app/utils/config.py`, `backend/app/utils/__init__.py` | FastAPI app entrypoint, CORS (`5173`/`3000`), startup model preload (`ModelRegistry`), settings loader. |
| `backend/app/routes/health.py` | Serves `GET /health` (polled every 10s by React dashboard). |
| `backend/app/routes/prediction.py` | Serves `POST /predict/agricultural` (used by React UI), `GET /predict/models`, `POST /predict/quick`, `POST /predict/batch`. |
| `backend/app/routes/stability.py` | Serves `GET /stability/scenarios`, `GET /stability/scenarios/{scenario_id}`, `POST /stability/analyze` (all used by React UI). |
| `backend/app/schemas/{agricultural,prediction,stability}.py` | Request/response Pydantic v2 validation contracts. |
| `backend/app/services/inference.py` | `ModelRegistry`, `_GRUModel`, `CROP_THRESHOLDS`, `TRAINING_FEATURE_MEANS`, strict 48-pkt validator (`InsufficientHistoryError`, `PredictionBlockedError`). |
| `backend/app/services/agricultural.py` | Orchestrates independent 48 / 336 / 1,344 inference (`evaluate_agricultural_predictions`). |
| `backend/app/services/stability_metrics.py` | Signal noise, outlier suppression, and chatter metrics for `POST /stability/analyze`. |
| `backend/app/ml/stability.py` | `MedianFilter`, `EWMASmoother`, `HysteresisSwitch`, `SoilMoistureStabilizer` (imported by `routes/stability.py` and `services/agricultural.py`). |
| `backend/app/ml/nest_fungal_mapper.py` | Converts 336 consecutive NEST packets into 38 fungal features (`services/agricultural.py`). |
| `backend/app/ml/nest_pest_mapper.py` | Converts 1,344 consecutive NEST packets into 26 pest features (`services/agricultural.py`). |
| `backend/app/ml/prepare_fungal_dataset.py`, `prepare_pest_dataset.py`, `train_fungal_risk.py` | **Important hidden runtime imports:** `nest_fungal_mapper.py` imports from `prepare_fungal_dataset` and `train_fungal_risk`; `nest_pest_mapper.py` imports from `prepare_pest_dataset`; tests also import from both prep scripts. **Do not delete without extracting shared constants/helpers first.** |

### 2.2 Saved ML Models & Metadata (`13` Files across `4` Directories)
| Model Directory | Size | Runtime / Test Consumers |
| :--- | :---: | :--- |
| `backend/app/ml/models_residual/` (`4` files) | `0.27 MB` | `ModelRegistry` (`xgb_residual` default in `/predict/agricultural`, plus `gru_residual`). |
| `backend/app/ml/models/` (`5` files) | `1.40 MB` | `ModelRegistry` (`xgb`, `gru` variants in `/predict/agricultural` & `/predict/batch`, `test_inference.py`, `smoke_test.py`). |
| `backend/app/ml/models_fungal_risk/` (`2` files) | `0.22 MB` | `services/agricultural.py` (`best_fungal_model.joblib`), `test_nest_fungal_mapper.py`, `test_fungal_dataset.py`. |
| `backend/app/ml/models_pest_risk/` (`2` files) | `0.01 MB` | `services/agricultural.py` (`best_pest_model.joblib`), `test_nest_pest_mapper.py`, `test_pest_dataset.py`. |

### 2.3 Active Datasets, Contracts & Scenarios
| Path | Role |
| :--- | :--- |
| `data/company_sample/nest_sample.json` | 21-field reference schema contract used across tests and synthetic generators. |
| `data/synthetic_nest/scenario_01_normal.csv` .. `scenario_05_high_humidity.csv` | Served live by `GET /stability/scenarios/{scenario_id}` for UI demos and tested by `test_stability_api.py` & `test_inference.py`. |
| `data/processed/fungal_risk/` (`2` files) & `data/processed/pest_risk/` (`3` files) | Processed feature tables and training-fold quantile thresholds verified by `test_fungal_dataset.py` and `test_pest_dataset.py`. |

### 2.4 React Frontend Source & Packages
- **Active Source:** `frontend/index.html`, `frontend/vite.config.js`, `frontend/.oxlintrc.json`, `frontend/.env.example`, `frontend/package.json`, `frontend/package-lock.json`, `frontend/src/{main.jsx,index.css,App.jsx,App.css,services/api.js}`, `frontend/src/components/{AgriculturalPredictions,InfoTip,SensorCards,TechnicalDetails}.jsx`.
- **NPM Packages:** `react`, `react-dom`, `recharts` (UI & graphs); `@vitejs/plugin-react`, `vite`, `oxlint`, `@types/react`, `@types/react-dom` (build, lint & JSX typing).

### 2.5 Test Suite & Final Documentation
- All 8 pytest modules under `backend/tests/` (`112` tests passing).
- Final documentation: `docs/TECHNICAL_REPORT.md`, `docs/DEMO_GUIDE.md`, `docs/CLEANUP_AUDIT.md`, plus model evaluation reports & plots under `backend/app/ml/evaluation/` (`31` files, `3.97 MB`).

---

## 3. Needs Manual Verification

These items are **not broken or strictly unused**, or they hold significant disk space / maintenance risk and require an explicit team decision before deleting or refactoring:

### 3.1 Large Raw External Datasets (`~459 MB` Not Loaded at Runtime)
| Path | Size | Trade-Off |
| :--- | :---: | :--- |
| `data/Soil Moisture, Irrigation Actuator and Weather Dat/.../01_raw_data/` | `427.39 MB` (`377.49` + `49.90 MB`) | Not read by FastAPI or pytest. Only needed to reproduce the 2025 Italian field preprocessing (`03_code/`). Archiving externally saves >90% of repo size. |
| `data/Soil Moisture.../02_processed_data/{merged,preprocessed}/` & `03_code/` | `31.34 MB` | Needed only if re-running `backend/app/ml/pipeline.py` or `residual_pipeline.py` to retrain irrigation models. (`03_code/01_merge_datasets.py` is also the sole script in the entire repo using `requests`.) |
| `data/external/pest/fungal_flax_rust/` & `data/external/pest/*.csv` | `~5.8+ MB` | Source public datasets for `prepare_fungal_dataset.py` & `prepare_pest_dataset.py`. Not needed at runtime, required for audit/reproducibility. |
| `data/external/pest/fungal/Grape Disease Dataset/Grape_Disease_Dataset.csv` | Small CSV | Audited as **unlabeled and unusable** for supervised modeling; never loaded by any script. Keep only as historical audit artifact or delete to clean `data/external/`. |

### 3.2 Offline Training, Audit & Ad-Hoc Scripts
| File | Current Status & Recommendation |
| :--- | :--- |
| `backend/app/ml/evaluate_saved_models.py` (`534` lines) | `0` imports anywhere; standalone script that generated `evaluation_summary.md` and plots. Keep for audit reproducibility or move into `scripts/` outside `app/`. |
| `backend/app/ml/pipeline.py` (`564` lines) & `backend/app/ml/residual_pipeline.py` (`636` lines) | Offline training scripts (never imported at FastAPI runtime — only referenced by audit/eval scripts). Verify before archiving/moving out of runtime `backend/app/`. |
| `backend/app/ml/evaluation/run_stability_evaluation.py` & `backend/app/ml/dataset_audit.md` | Standalone evaluation generator (`0` callers) and earlier dataset audit doc. Verify whether to move scripts out of `app/ml/evaluation/`. |
| `data/synthetic_nest/{generate_synthetic,inspect_models,mapping_audit,smoke_test,verify}.py` + `generation_summary.csv` | Manual developer utilities. `smoke_test.py` currently still passes (`exit=0`) via the compatibility alias `make_irrigation_advice` in `routes/prediction.py`; its coverage overlaps `backend/tests/test_inference.py` & `test_agricultural_api.py`. |

### 3.3 Legacy API Compatibility Routes (`backend/app/routes/prediction.py`)
- `POST /predict/quick`, `POST /predict/batch`, and `GET /predict/models` are **not called by the current React dashboard** (which calls `/predict/agricultural`, `/stability/*`, and `/health`), though they are actively tested in `backend/tests/test_inference.py` & `test_agricultural_api.py`. Confirm if external hardware/clients depend on them before ever deprecating.

### 3.4 Code Duplication & Refactoring Candidates
| Duplicated Logic | Locations | Recommended Manual Check |
| :--- | :--- | :--- |
| **GRU Network Architecture & Tabular Extraction** | `services/inference.py` (`_GRUModel`, `_extract_abs_tabular`, `_extract_res_tabular`) vs. `ml/pipeline.py` (`SoilMoistureGRU`, `extract_tabular_features`) & `ml/residual_pipeline.py` (`SoilMoistureResidualGRU`, `extract_tabular_residual_features`) | Keeping duplicate runtime vs. offline copies avoids coupling API startup to training scripts, but any architectural change must stay synchronized. |
| **Crop Threshold Tables** | `CROP_THRESHOLDS` (`services/inference.py`) vs. `CROP_SPECS` (`ml/pipeline.py`) vs. `models_residual/model_metadata.json` | Ensure single source of truth if crop parameters are updated. |
| **Sequence Integrity Validation** | `validate_sequence_integrity` (`services/inference.py`), `_validate_nest_packets_for_fungal` (`nest_fungal_mapper.py`), `_validate_nest_packets_for_pest` (`nest_pest_mapper.py`) | Similar timestamp spacing & NaN checks repeated in 3 places with different required field rules; candidate for a shared helper once regression tests cover any consolidation. |
| **OLS Slope / Feature Helpers** | `prepare_fungal_dataset.py`, `prepare_pest_dataset.py`, `nest_pest_mapper.py`, `nest_fungal_mapper.py` | Confirm import boundaries before relocating offline training files out of `backend/app/ml/`. |

### 3.5 Environment, Requirements & Legacy Doc Mismatches
| Item | Issue Discovered |
| :--- | :--- |
| `backend/requirements.txt` vs. Actual Runtime Imports | Lists only `fastapi>=0.110.0`, `uvicorn[standard]>=0.28.0`, `pydantic>=2.0.0`. Missing runtime dependencies: **`numpy`, `pandas`, `joblib`, `torch`, `xgboost`, `scikit-learn`** (and dev/test/prep: `pytest`, `httpx`, `matplotlib`, `pyreadr`, `openpyxl`). |
| `backend/venv/` vs. Global Python 3.11 | Tests and server currently execute against system Python 3.11 rather than `backend/venv/`; verify venv packages before clean-machine deployment. |
| `frontend/.env` vs. `frontend/.env.example` | `frontend/.env` exists locally (git-ignored) with the default `VITE_API_BASE_URL=http://127.0.0.1:8000`; `api.js` already defaults to the same URL. |
| `README.md` & `data/README.md` | Contain legacy starter statements ("No ML models... are implemented", "synthetic_nest/: Empty staging area") superseded by `docs/TECHNICAL_REPORT.md`. |
| `backend/app/ml/evaluation/diagnostic_report.md` & `models/evaluation_results.json` | Historical pre-residual evaluation artifacts (`evaluation_results.json` sits inside `models/` alongside weights). Verify if team retains all historical artifacts for traceability. |
| `pyrightconfig.json` & `.vscode/settings.json` | IDE/type-checker config (`pyrightconfig.json` disables strict optional checks; `.vscode/` is git-ignored). Confirm team IDE preferences. |
