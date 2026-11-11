# CropNow Final Repository Integrity & Cleanup Report

**Audit Date:** November 11, 2026
**Workspace Audited:** `c:\Users\Vishal\Downloads\CropNow`
**Policy:** Read-only verification — **no files deleted, no source code modified, no changes staged, and no commits created.**

---

## Executive Summary

| Check Area | Status | Summary |
| :--- | :---: | :--- |
| **Git Repository Scope (`git status` / `diff`)** | **Warning (No Local `.git`)** | `CropNow` has no `.git/` folder of its own; Git resolves upward to `C:/Users/Vishal/.git`. `0` files inside `CropNow` are tracked (`git status --short -- .` → `?? ./`). `git diff --check` exited `0`; `git diff --stat` is empty. |
| **Temporary & Unexpected Files** | **Clean (`0` Found)** | Checked `212` non-`venv`/non-`node_modules` files: **zero** instances of `run_audit.py`, `output.txt`, `.tmp`, `.bak`, `.swp`, `.swo`, `.log`, `.orig`, `.rej`, `.DS_Store`, or `Thumbs.db`. |
| **`.gitignore` Coverage** | **1 Missing Entry** | Covers `.env`, `.env.local`, `.env.*.local`, `venv/`/`.venv/`, `node_modules/`, `dist/`, and `__pycache__/`, but **does not ignore `.pytest_cache/`**. |
| **Credentials, Keys & Local Env Files** | **Clean (`0` Leaks)** | `0` secret patterns matched across project text/code/config files; only `frontend/.env` and `frontend/.env.example` exist (`VITE_API_BASE_URL` variable only, no secrets). |
| **Models, Datasets, Tests & Reports** | **All Intact** | All `13` model/metadata files across `4` model directories, `5` scenario CSVs, `5` processed ML dataset files, `nest_sample.json`, `8` pytest modules, `31` evaluation report/plot files, and all `docs/` reports remain intact. |

---

## 1. Git Status & Diff Checks

- **`git rev-parse --show-toplevel --git-dir`:** Resolved to `C:/Users/Vishal` (`C:/Users/Vishal/.git`), NOT `c:\Users\Vishal\Downloads\CropNow`.
- **`git ls-files -- .`:** `0` tracked files under `CropNow`.
- **`git status --short -- .`:** `?? ./` (the entire `CropNow` directory is untracked relative to `C:/Users/Vishal/.git`; unscoped `git status --short` enumerates unrelated folders in `C:/Users/Vishal/`).
- **`git diff --check`:** Exited `0` with no output (no whitespace errors or conflict markers in any tracked diff).
- **`git diff --stat`:** Empty (`0` tracked modifications under `C:/Users/Vishal/.git`).

---

## 2. File Inventory & Unexpected Temporary File Check

Scanning all `212` files inside `CropNow` (excluding `backend/venv/` and `frontend/node_modules/` internals):

- **Unexpected temporary artifacts:** **None (`0`)** — confirmed absence of `run_audit.py`, `output.txt`, `.tmp`, `.bak`, `.swp`, `.swo`, `.log`, `.orig`, `.rej`, `.DS_Store`, and `Thumbs.db`.
- **Regenerable build/test caches currently on disk** *(created by verification test/build runs after earlier cache cleanup)*:
  - `.pytest_cache/` *(exists from running `pytest backend/tests -v`)*
  - `7` `__pycache__/` directories (`backend/app/`, `backend/app/ml/`, `backend/app/routes/`, `backend/app/schemas/`, `backend/app/services/`, `backend/app/utils/`, `backend/tests/`)
  - `frontend/dist/` *(exists from running `npm run build`)*
- **Local environment & IDE directories currently on disk:** `backend/venv/`, `frontend/node_modules/`, `frontend/.env`, `.vscode/`.

---

## 3. `.gitignore` Coverage Assessment

Checked [`/.gitignore`](../.gitignore) (51 lines) and [`/frontend/.gitignore`](../frontend/.gitignore) (24 lines):

| Pattern / Asset | Covered? | Rule Location | Notes |
| :--- | :---: | :--- | :--- |
| `.env` / local env files | **Yes** *(partial for non-`.local` variants)* | `.gitignore:31-33` | `.env`, `.env.local`, `.env.*.local` (note: `.env.production` without `.local` would not match; `frontend/.env.example` remains intentionally unignored). |
| Virtual environments | **Yes** | `.gitignore:7-9, 27-28` | `env/`, `venv/`, `ENV/`, `.venv/`. |
| `node_modules` | **Yes** | `.gitignore:36`, `frontend/.gitignore:10` | `node_modules/` and `node_modules`. |
| `dist` | **Yes** | `.gitignore:12, 37-38`, `frontend/.gitignore:11-12` | `dist/`, `.dist/`, `dist`, `dist-ssr`. |
| `__pycache__` | **Yes** | `.gitignore:2-4` | `__pycache__/`, `*.py[cod]`, `*$py.class`. |
| `.pytest_cache` | **NO** | *Missing* | **Gap:** `.pytest_cache/` is absent from `.gitignore` and currently exists on disk after pytest runs. |

---

## 4. Credentials & Secret Scan

- **Tracked secrets/env files:** `0` (`0` files under `CropNow` are tracked in Git).
- **`.env*` files on disk (names & variable keys only):**
  - `frontend/.env` → defines only `VITE_API_BASE_URL`
  - `frontend/.env.example` → defines only `VITE_API_BASE_URL`
- **Automated pattern scan** (API tokens, cloud keys, private key blocks, hardcoded secret/password literals across all non-binary source, config, and doc files): **`0` matches.**

---

## 5. Integrity of Required Models, Datasets, Tests & Reports

All core deliverables and runtime dependencies are verified present on disk:

- **Saved Models & Metadata (`13` files across `4` directories):**
  - `backend/app/ml/models/` (`5`): `gru_model.pt`, `xgb_model.joblib`, `scaler.joblib`, `model_metadata.json`, `evaluation_results.json`
  - `backend/app/ml/models_residual/` (`4`): `gru_residual_model.pt`, `xgb_residual_model.joblib`, `scaler_residual.joblib`, `model_metadata.json`
  - `backend/app/ml/models_fungal_risk/` (`2`): `best_fungal_model.joblib`, `model_metadata.json`
  - `backend/app/ml/models_pest_risk/` (`2`): `best_pest_model.joblib`, `model_metadata.json`
- **Datasets & Reference Contracts:**
  - `data/company_sample/nest_sample.json`
  - `data/synthetic_nest/scenario_01_normal.csv` .. `scenario_05_high_humidity.csv` (`5` CSVs)
  - `data/processed/fungal_risk/` (`2` files) and `data/processed/pest_risk/` (`3` files)
- **Automated Test Suite (`8` modules):**
  - `test_agricultural_api.py`, `test_fungal_dataset.py`, `test_inference.py`, `test_nest_fungal_mapper.py`, `test_nest_pest_mapper.py`, `test_pest_dataset.py`, `test_stability.py`, `test_stability_api.py`
- **Documentation & Evaluation Artifacts:**
  - `docs/TECHNICAL_REPORT.md`, `docs/DEMO_GUIDE.md`, `docs/CLEANUP_AUDIT.md`, `docs/DEPENDENCY_AUDIT.md`, `docs/FINAL_CLEANUP_REPORT.md`
  - `backend/app/ml/evaluation/`: all `31` evaluation reports, metrics JSONs, confusion matrices, and plots intact
- **Backend & Frontend Core Files:** `backend/app/main.py`, `backend/app/ml/fungal_features.py`, `backend/requirements.txt`, `backend/requirements-dev.txt`, `frontend/package.json`, `frontend/src/components/{AgriculturalPredictions,InfoTip,SensorCards,TechnicalDetails}.jsx`.

---

## 6. Remaining Issues Requiring Attention Before Submission / Commit

1. **Initialize a Dedicated Git Repository inside `CropNow` (or Commit from Isolated Repo Root):** Currently `git` binds to `C:/Users/Vishal/.git`, leaving all `CropNow` files outside version control (`?? ./`) and risking accidental home-directory Git operations.
2. **Add `.pytest_cache/` to `.gitignore`:** Prevents accidental commits of `.pytest_cache/` once `CropNow` is tracked under Git.
3. **Optional `backend/venv` Package Synchronization:** `backend/venv` still holds only the initial `3` web packages (`21` total). Once synced with `pip install --upgrade pip setuptools>=77.0.3` and `pip install -r backend/requirements-dev.txt`, both backend runtime and tests can execute directly inside `backend/venv` rather than system Python `3.11`.
4. **Optional Cache Hygiene Before Archival/Submission:** Optionally delete regenerable `.pytest_cache/`, `__pycache__/`, and `frontend/dist/` right before creating a clean archive/zip if desired (all other runtime/submission checks already verified).
