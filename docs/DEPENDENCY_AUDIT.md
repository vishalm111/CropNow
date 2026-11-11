# CropNow Backend Dependency Audit

**Audit Date:** November 11, 2026
**Scope:** Static AST import scan of all Python files in `backend/` (runtime API, schemas, services, ML mappers, offline training/evaluation scripts, and `backend/tests/`), model unpickling/load checks across all saved `.joblib` and `.pt` artifacts, and comparison against `backend/requirements.txt`, `backend/venv`, and system `python3.11`.
**Status:** **Resolved in code & dependency manifests** (`backend/requirements.txt`, `backend/requirements-dev.txt`, `backend/app/ml/fungal_features.py`, and lazy `matplotlib` imports). **No Python packages were installed and no saved model artifacts were modified.**

---

## Executive Summary

| Finding Area | Status | Summary |
| :--- | :---: | :--- |
| **`backend/requirements.txt` Completeness** | **Fixed** | Initially listed only `fastapi>=0.110.0`, `uvicorn[standard]>=0.28.0`, and `pydantic>=2.0.0`. Now declares all `9` confirmed runtime/inference dependencies bounded to verified artifact compatibility. |
| **API Startup Coupling (`matplotlib`)** | **Fixed** | `FEATURE_COLS` moved to `backend/app/ml/fungal_features.py` and `matplotlib` made lazy in training/evaluation plotting functions. Importing `app.main` loads `0` `matplotlib` modules. |
| **`backend/requirements-dev.txt`** | **Created** | Added `-r requirements.txt` plus `pytest`, `httpx`, and `matplotlib` for testing and offline training/evaluation scripts. |
| **`backend/venv` Environment** | **Pending Install** | `backend/venv` (`21` packages, created from the initial 3-line `requirements.txt`) still fails on `import app.main` (`No module named 'joblib'`) because package installation was explicitly deferred. |
| **System `python3.11` Verification** | **Passing (`112/112`)** | Executes `app.main`, loads all 6 `.joblib` and 2 `.pt` artifacts with `0` warnings, and passes all `112` tests (`pytest`). Unrelated global `streamlit`/`protobuf` conflict remains outside CropNow. |
| **Unused Entries in Baseline `requirements.txt`** | **None (`0`)** | All 3 baseline entries (`fastapi`, `uvicorn[standard]`, `pydantic`) are actively required. |

*(Sections 1–3 below preserve the baseline pre-fix audit evidence; Section 4 documents the applied fixes and post-edit verification.)*

---

## 1. Current `backend/requirements.txt` vs. Installed Environments

### 1.1 Declared in `backend/requirements.txt` (Lines 1–3)

```text
fastapi>=0.110.0
uvicorn[standard]>=0.28.0
pydantic>=2.0.0
```

### 1.2 Environment Comparison (`backend/venv` vs. System `python3.11`)

Both Python executables run **Python `3.11.9`**:
- **`backend/venv`**: `c:\Users\Vishal\Downloads\CropNow\backend\venv\Scripts\python.exe`
- **System `python3.11`**: `C:\Users\Vishal\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe`

| Package | Role in CropNow | In `requirements.txt`? | `backend/venv` Version | System `python3.11` Version | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `fastapi` | Web framework, CORS, `TestClient` | Yes (`>=0.110.0`) | `0.162.2` | `0.143.0` | Both require `starlette>=0.46.0` and `pydantic>=2.9.0` |
| `uvicorn` | ASGI server CLI (`uvicorn app.main:app`) | Yes (`[standard]>=0.28.0`) | `0.54.0` | `0.54.0` | `[standard]` extras `httptools` (`0.9.0`) & `watchfiles` (`1.3.0`) exist in `venv`, **missing** in `python3.11` |
| `pydantic` | Schema validation (`BaseModel`, v2 validators) | Yes (`>=2.0.0`) | `2.14.3` | `2.14.0` | Lower bound `>=2.0.0` is looser than FastAPI's `>=2.9.0` requirement |
| `starlette` | FastAPI ASGI/HTTP core *(transitive)* | — | `1.10.0` | `1.7.0` | Installed automatically via `fastapi` |
| `numpy` | Numerical arrays & signal/feature math | **NO** | **Not installed** | `2.2.4` | Required by runtime services, mappers, `scipy`, `pandas`, `torch` |
| `pandas` | DataFrames, timestamps (`to_datetime`), CSV loading | **NO** | **Not installed** | `2.2.3` | Required by `routes/stability.py`, `services/inference.py`, mappers |
| `joblib` | Loading `.joblib` scalers & classifiers | **NO** | **Not installed** | `1.6.0` | First missing import on startup (`services/agricultural.py:15`) |
| `torch` | `nn.Module` GRU definition & `torch.load` | **NO** | **Not installed** | `2.14.1+cpu` | Used directly in `services/inference.py` (`weights_only=True`) |
| `xgboost` | Unpickling `XGBRegressor`/`XGBClassifier` + top-level import | **NO** | **Not installed** | `3.2.0` | Needed at runtime & startup (via `train_fungal_risk.py` import) |
| `scikit-learn` | Unpickling `StandardScaler` / `Pipeline` / `LogisticRegression` | **NO** | **Not installed** | `1.9.0` | Needed at runtime & startup; brings `scipy`, `threadpoolctl`, `joblib` |
| `scipy` | Required backend for `scikit-learn` & `xgboost` *(transitive)* | **NO** | **Not installed** | `1.17.1` | Transitive requirement of `scikit-learn` & `xgboost` (`numpy>=1.26.4,<2.7`) |
| `matplotlib` | Training/eval plots *(also pulled into API startup!)* | **NO** | **Not installed** | `3.11.1` | Imported at top level by `train_fungal_risk.py` (see Section 2.2) |
| `pytest` | Backend test runner (`112` tests) | **NO** | **Not installed** | `9.1.1` | Dev/test dependency (`pytest-asyncio==1.4.0` is installed but unused) |
| `httpx` | Required runtime backend for `fastapi.testclient.TestClient` | **NO** | **Not installed** | `0.28.1` | Dev/test dependency used by 3 API test modules |

---

## 2. Detailed Dependency Analysis by Usage Scope

### 2.1 Unused Dependencies in `backend/requirements.txt`

**None.** All 3 declared requirements are actively needed:
- `fastapi` — Imported in `backend/app/main.py`, `backend/app/routes/{health,prediction,stability}.py`, and `backend/tests/test_{agricultural_api,inference,stability_api}.py`.
- `uvicorn[standard]` — Not imported inside Python source files (standard for an ASGI CLI entrypoint), but invoked via `uvicorn app.main:app` in `README.md`, `backend/README.md`, and `docs/DEMO_GUIDE.md`.
- `pydantic` — Imported directly in `backend/app/routes/health.py`, `backend/app/schemas/{agricultural,prediction,stability}.py`, and `backend/app/utils/config.py` using Pydantic v2 features (`ConfigDict`, `field_validator`, `model_validator`).

*(Side note for dev tooling cleanup: `pytest-asyncio==1.4.0` is installed globally, but zero backend tests contain `asyncio` or `async def`.)*

---

### 2.2 FastAPI Startup & API Runtime Dependencies (Missing from `requirements.txt`)

Every package below is required simply to execute `from app.main import app` and serve live endpoints (`/health`, `/predict/*`, `/stability/*`):

| Missing Package | Runtime Files & Artifacts Requiring It |
| :--- | :--- |
| **`joblib`** | `backend/app/services/inference.py` (loads `scaler.joblib`, `xgb_model.joblib`, `scaler_residual.joblib`, `xgb_residual_model.joblib`), `backend/app/services/agricultural.py` (loads `best_fungal_model.joblib`, `best_pest_model.joblib`). |
| **`numpy`** | `backend/app/routes/stability.py`, `backend/app/services/inference.py`, `backend/app/services/agricultural.py`, `backend/app/services/stability_metrics.py`, `backend/app/ml/nest_fungal_mapper.py`, `backend/app/ml/nest_pest_mapper.py`, `backend/app/ml/prepare_fungal_dataset.py`, `backend/app/ml/prepare_pest_dataset.py`. |
| **`pandas`** | `backend/app/routes/stability.py` (`pd.read_csv` on synthetic scenario CSVs), `backend/app/services/inference.py` (`pd.to_datetime`), `backend/app/services/stability_metrics.py`, `nest_fungal_mapper.py`, `nest_pest_mapper.py`, `prepare_fungal_dataset.py`, `prepare_pest_dataset.py`. |
| **`torch`** | `backend/app/services/inference.py` (`_GRUModel(nn.Module)`, `torch.load(..., weights_only=True)`, `torch.tensor`, `torch.no_grad` for `gru_model.pt` and `gru_residual_model.pt`). Note: `weights_only=True` requires PyTorch `>=2.0.0`. |
| **`xgboost`** | 1) **Artifact unpickling:** `xgb_model.joblib` (`XGBRegressor`), `xgb_residual_model.joblib` (`XGBRegressor`), and `best_fungal_model.joblib` (`XGBClassifier`).<br>2) **Module import at startup:** `train_fungal_risk.py:54` runs `from xgboost import XGBClassifier`. |
| **`scikit-learn`** | 1) **Artifact unpickling:** `scaler.joblib` & `scaler_residual.joblib` (`StandardScaler`), `best_pest_model.joblib` (`sklearn.pipeline.Pipeline` with `StandardScaler` + `LogisticRegression`).<br>2) **Module import at startup:** `train_fungal_risk.py:42-53` imports `sklearn.*` at module top level. |
| **`matplotlib`** *(coupled startup import)* | **Unexpected startup dependency:** `backend/app/routes/prediction.py` → `services/agricultural.py` → `ml/nest_fungal_mapper.py:27` runs `from app.ml.train_fungal_risk import FEATURE_COLS`. Importing `train_fungal_risk.py` executes `import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt` at module top level (lines 37–39). **FastAPI cannot currently start without `matplotlib` installed.** |

---

### 2.3 Testing & Verification Dependencies (Missing)

| Package | Usage in `backend/tests/` |
| :--- | :--- |
| **`pytest`** | Imported directly in `7` of `8` test files under `backend/tests/` (`test_agricultural_api.py`, `test_fungal_dataset.py`, `test_inference.py`, `test_nest_fungal_mapper.py`, `test_nest_pest_mapper.py`, `test_stability.py`, `test_stability_api.py`). |
| **`httpx`** | Not imported directly in project code, but **required at runtime by `fastapi.testclient.TestClient`** in `test_agricultural_api.py`, `test_inference.py`, and `test_stability_api.py`. |

---

### 2.4 Offline Training, Evaluation & External Data Scripts (Reference Only)

All Python dependencies of `backend/app/ml/{pipeline,residual_pipeline,train_fungal_risk,train_pest_risk,evaluate_saved_models,prepare_fungal_dataset,prepare_pest_dataset}.py`, `backend/app/ml/evaluation/run_stability_evaluation.py`, and `data/synthetic_nest/*.py` are already covered by the packages above (`numpy`, `pandas`, `joblib`, `torch`, `xgboost`, `scikit-learn`, `matplotlib`).

Two external/non-backend script requirements were also observed:
- **`requests`** (`2.32.3` in global `python3.11`): Only imported by the offline 2025 field preprocessing script `data/Soil Moisture, Irrigation Actuator and Weather Dat/.../03_code/01_merge_datasets.py` — **not imported anywhere in `backend/`**.
- **`Rscript` binary** (external R installation, non-PyPI): Used via `subprocess` in `backend/app/ml/prepare_fungal_dataset.py` only when regenerating `data/processed/fungal_risk/` from raw `.rds` files.
- **No dynamic `openpyxl` or `pyreadr` usage exists** in any `.py` file across `backend/` or `data/`.

---

## 3. Version Conflicts, Compatibility & Environment Risks

1. **`backend/venv` vs. System `python3.11` Drift & Broken `venv`**
   - `backend/venv` cannot start the FastAPI server (`ModuleNotFoundError: No module named 'joblib'`) or run `pytest`.
   - `fastapi` (`0.162.2` vs `0.143.0`), `pydantic` (`2.14.3` vs `2.14.0`), `starlette` (`1.10.0` vs `1.7.0`), and `websockets` (`17.4` vs `16.1.1`) differ between `backend/venv` and global `python3.11`.
   - `uvicorn[standard]` extras `httptools` and `watchfiles` are missing in system `python3.11` (Uvicorn falls back to pure-Python HTTP/stat reloading).

2. **`pydantic>=2.0.0` Floor Is Outdated for Current FastAPI**
   - Installed `fastapi` (`0.143.0` global / `0.162.2` venv) enforces `pydantic>=2.9.0` and `starlette>=0.46.0`. Updating the lower bound to `pydantic>=2.9.0` avoids misleading resolution constraints.

3. **Artifact Compatibility (`scikit-learn` / `xgboost` / `numpy` / `torch`)**
   - Loading all 6 `.joblib` files and 2 `.pt` files in system `python3.11` (`numpy==2.2.4`, `pandas==2.2.3`, `joblib==1.6.0`, `scikit-learn==1.9.0`, `xgboost==3.2.0`, `torch==2.14.1+cpu`) succeeds with **`0` captured warnings** (`112/112` tests passing).
   - **Caution on unpinned `scikit-learn` & `xgboost`:** `.joblib` serialized `StandardScaler`, `Pipeline`, `LogisticRegression`, `XGBRegressor`, and `XGBClassifier` objects can break or emit `InconsistentVersionWarning` across major/minor upgrades. Bounding upper minor/major ranges (or pinning tested versions) is strongly recommended.
   - **Cross-constraint:** `scipy==1.17.1` requires `numpy>=1.26.4,<2.7.0` (satisfied by `2.2.4`).
   - **PyTorch installer prerequisite in `backend/venv`:** `torch==2.14.1` declares `setuptools>=77.0.3`, whereas `backend/venv` currently has `setuptools==65.5.0` (and `pip==24.0`). Installing PyTorch inside `backend/venv` will require upgrading `pip`/`setuptools` first or alongside it, and specifying CPU wheels (`--index-url https://download.pytorch.org/whl/cpu` on Linux/CI) prevents downloading unnecessary multi-GB CUDA wheels if GPU is not used.

4. **Pre-existing Conflict in Global `python3.11` (`pip check` Exit Code `1`)**
   - System `python3.11` reports:
     `streamlit 1.43.2 has requirement protobuf<6,>=3.20, but you have protobuf 7.36.1.`
   - Although unrelated to CropNow packages, this confirms why CropNow should run from an isolated `backend/venv` rather than the shared global Python installation.

---

## 4. Changes Applied & Compatibility Basis

### 4.1 Code Decoupling & Lazy `matplotlib` Imports

1. **Created standalone fungal schema module:** `backend/app/ml/fungal_features.py` exporting the `38`-feature `FEATURE_COLS` list.
2. **Updated imports:**
   - `backend/app/ml/nest_fungal_mapper.py` now imports `FEATURE_COLS` from `app.ml.fungal_features` (breaking the startup dependency on `train_fungal_risk.py`).
   - `backend/app/ml/train_fungal_risk.py` now imports `FEATURE_COLS` from `app.ml.fungal_features` instead of redefining it inline.
3. **Made `matplotlib` lazy across all training and evaluation modules:** Removed top-level `matplotlib` / `matplotlib.pyplot` imports and moved `import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt` inside plotting functions/blocks in:
   - `backend/app/ml/train_fungal_risk.py` (`_plot_confusion_matrices`)
   - `backend/app/ml/train_pest_risk.py` (`_plot_confusion_matrices`)
   - `backend/app/ml/residual_pipeline.py` (visualization generation block)
   - `backend/app/ml/evaluate_saved_models.py` (visualization generation block)
   - `backend/app/ml/evaluation/run_stability_evaluation.py` (plot generation block)

### 4.2 Artifact Compatibility Audit Behind Version Bounds

Before updating `requirements.txt`, all saved `.joblib` and `.pt` model artifacts were inspected for serialization versions (without modifying any artifacts):
- **Scikit-learn pickles (`scaler.joblib`, `scaler_residual.joblib`, `best_pest_model.joblib`):** `__getstate__()` records `_sklearn_version: '1.9.0'` and NumPy `2.x` (`numpy._core`) arrays. Because scikit-learn pickles frequently warn or fail across minor/major releases and NumPy `1.x` cannot unpickle NumPy `2.x` arrays, `scikit-learn>=1.9.0,<1.10.0` and `numpy>=2.0.0,<2.7.0` (matching `scipy==1.17.1`'s `<2.7` upper bound) are used.
- **XGBoost boosters (`xgb_model.joblib`, `xgb_residual_model.joblib`, `best_fungal_model.joblib`):** Booster `save_config()` JSON records `"version": [3, 2, 0]`, giving `xgboost>=3.2.0,<4.0.0`.
- **PyTorch state dicts (`gru_model.pt`, `gru_residual_model.pt`):** Loaded via `torch.load(..., map_location="cpu", weights_only=True)` in `services/inference.py`, which requires `torch>=2.0.0,<3.0.0` (verified on `2.14.1+cpu`).
- **FastAPI / Pydantic / Pandas / Joblib:** `fastapi>=0.110.0,<1.0.0`, `uvicorn[standard]>=0.28.0,<1.0.0`, `pydantic>=2.9.0,<3.0.0` (matching FastAPI's `>=2.9.0` lower bound), `pandas>=2.2.2,<3.0.0` (NumPy 2.x ABI support), and `joblib>=1.4.0,<2.0.0` (matching scikit-learn's `>=1.4.0` floor).

### 4.3 Updated Dependency Manifests

**`backend/requirements.txt`** *(runtime API, model loading, and inference)*:
```txt
# API & ASGI runtime
fastapi>=0.110.0,<1.0.0
uvicorn[standard]>=0.28.0,<1.0.0
pydantic>=2.9.0,<3.0.0

# Data processing & feature engineering
numpy>=2.0.0,<2.7.0
pandas>=2.2.2,<3.0.0

# Model loading & ML inference (bounded to saved artifact compatibility)
joblib>=1.4.0,<2.0.0
scikit-learn>=1.9.0,<1.10.0
xgboost>=3.2.0,<4.0.0
torch>=2.0.0,<3.0.0
```

**`backend/requirements-dev.txt`** *(tests + offline training/evaluation visualizations)*:
```txt
-r requirements.txt

# Automated API and ML test suite
pytest>=8.0.0,<10.0.0
httpx>=0.27.0,<1.0.0

# Offline training & evaluation visualization (lazy-imported only in plot functions)
matplotlib>=3.8.0,<4.0.0
```

---

## 5. Post-Change Verification & Next Environment Step

- **API Startup Isolation Check (`python3.11`):** Importing `app.main` loads `[]` `matplotlib` modules (`0` imported), and `nest_fungal_mapper.FEATURE_COLS == train_fungal_risk.FEATURE_COLS` (`38` features).
- **Backend Pytest (`python3.11 -m pytest backend/tests -v`):** `112 passed in 11.48s` (`exit code 0`).
- **Frontend Lint (`npm run lint` / `oxlint`):** `Found 0 warnings and 0 errors.` (`102ms`, `8` files, `104` rules, `exit code 0`).
- **Frontend Production Build (`npm run build` / `vite build`):** `395 modules transformed`, `built in 390ms` (`exit code 0`).

### 5.1 `pip install --dry-run` Resolution Validation *(No Packages Installed)*

| Environment | Manifest | Outcome | Details & Resolver Output |
| :--- | :--- | :---: | :--- |
| **System `python3.11`** (`3.11.9`, `pip 26.2.2`) | `backend/requirements.txt` | **Passed (`exit code 0`)** | Satisfied `fastapi==0.143.0`, `uvicorn==0.54.0`, `pydantic==2.14.0`, `numpy==2.2.4`, `pandas==2.2.3`, `joblib==1.6.0`, `scikit-learn==1.9.0`, `xgboost==3.2.0`, `torch==2.14.1` (`scipy==1.17.1`, `starlette==1.7.0`, etc.). Missing optional `uvicorn[standard]` extras resolved cleanly: `Would install httptools-0.9.0 watchfiles-1.3.0`. Zero conflicts or unavailable versions. |
| **System `python3.11`** (`3.11.9`, `pip 26.2.2`) | `backend/requirements-dev.txt` | **Passed (`exit code 0`)** | Satisfied all `requirements.txt` packages plus `pytest==9.1.1`, `httpx==0.28.1`, `matplotlib==3.11.1` and their dependencies (`httpcore==1.0.9`, `pluggy==1.6.0`, `pillow==11.1.0`, etc.). Output: `Would install httptools-0.9.0 watchfiles-1.3.0`. Zero conflicts or unavailable versions. |
| **`backend/venv`** (`3.11.9`, `pip 24.0`) | `backend/requirements.txt` | **Passed (`exit code 0`)** | Satisfied existing `fastapi==0.162.2`, `uvicorn==0.54.0`, `pydantic==2.14.3` (plus `httptools==0.9.0`, `watchfiles==1.3.0`, etc.) and resolved from PyPI with zero conflicts: `Would install Jinja2-3.1.6 MarkupSafe-3.0.4 cloudpickle-3.1.2 filelock-4.2.10 fsspec-2026.9.0 joblib-1.6.0 mpmath-1.3.0 narwhals-2.28.0 networkx-3.6.1 numpy-2.4.6 pandas-2.3.3 python-dateutil-4.0.0 pytz-2026.6 scikit-learn-1.9.1 scipy-1.17.1 setuptools-84.2.1 sympy-1.14.0 threadpoolctl-3.8.0 torch-2.15.0 tzdata-2026.6 xgboost-3.2.0`. *(Note: `pip` upgraded `setuptools` to `84.2.1` in resolution to satisfy `torch>=2.0.0` / `2.15.0`.)* |
| **`backend/venv`** (`3.11.9`, `pip 24.0`) | `backend/requirements-dev.txt` | **Canceled before completion (`task-2338`)** | Metadata resolution completed for all runtime + dev entries (`pytest-9.2.0`, `httpx-0.28.1`, `matplotlib-3.11.2`, `certifi-2026.10.23`, `httpcore-1.0.9`, etc.) with zero reported conflicts or unavailable versions; execution was canceled (`Tool execution was canceled`) mid-download at `Downloading fonttools-4.68.0-cp311-cp311-win_amd64.whl (1.7 MB)` before emitting the final `Would install` summary / exit code. |

**Pending manual step (not executed per policy):** When ready to synchronize `backend/venv`:
1. Upgrade `pip` and `setuptools>=77.0.3` inside `backend/venv` (required by recent PyTorch `2.x` wheels, as confirmed by `backend/venv`'s dry-run selecting `setuptools-84.2.1`).
2. Install `-r backend/requirements-dev.txt` (or `-r backend/requirements.txt` for runtime-only deployment).
3. Run `backend\venv\Scripts\python.exe -m pip check` and `backend\venv\Scripts\python.exe -m pytest backend/tests -v`.
