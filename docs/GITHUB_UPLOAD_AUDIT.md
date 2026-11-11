# CropNow GitHub Upload & Dataset Redistribution Audit

**Audit Date:** November 11, 2026\
**Repository Root:** `c:\Users\Vishal\Downloads\CropNow\.git`\
**Status:** **Prepared (Uncommitted)** — attribution added to `data/README.md`, Category 3 datasets unstaged (`git rm --cached -r`) and added to `.gitignore` without deleting local files or modifying models/training code. No commit or push performed.

---

## 1. Revised Staged Repository Size Summary

After unstaging the `7` excluded files under `data/external/pest/` and `data/processed/pest_risk/` (`507,328` bytes / `0.48 MiB` removed from index, preserved on disk):

- **Revised Total Staged Paths:** **`162`** (down from `169` paths).
- **Revised Total Staged Size (Working-Tree Size of All Staged Paths):** **`98,310,166` bytes (`93.76 MiB` / `98.31 MB`)**.
- **Revised Staged Size Under `data/`:** **`91,525,948` bytes (`87.29 MiB` / `91.53 MB`)** across **`53`** files (`93.10%` of total staged size).
- **Per-File GitHub Limit Check:** Largest single staged file remains `MEASUREMENT_all.csv` (`26,603,330` bytes / `25.37 MiB`), below GitHub's `50 MiB` warning and `100 MiB` hard block limits. *(Note: `.gitignore`, `data/README.md`, and `docs/GITHUB_UPLOAD_AUDIT.md` have pending working-tree updates [`AM` / `??`] ready to stage before committing.)*

### Revised Staged Breakdown by Top-Level Path

| Path | Staged Files | Exact Size (Bytes) | Size (MiB) | Share of Total |
| :--- | :---: | :---: | :---: | :---: |
| `data/` | `53` | `91,525,948` | `87.29 MiB` | `93.10%` |
| `backend/` | `84` | `6,537,410` | `6.23 MiB` | `6.65%` |
| `frontend/` | `17` | `167,041` | `0.16 MiB` | `0.17%` |
| `docs/` | `5` *(+ updated audit)* | `73,901` | `0.07 MiB` | `0.08%` |
| `README.md`, `.gitignore`, `pyrightconfig.json` | `3` | `5,866` | `0.01 MiB` | `<0.01%` |
| **Total Staged Paths** | **`162`** | **`98,310,166`** | **`93.76 MiB`** | **`100.00%`** |

### Revised Staged Breakdown Inside `data/` (`53` Files)

| Dataset / Subset | Staged Files | Exact Size (Bytes) | Size (MiB) | Disposition |
| :--- | :---: | :---: | :---: | :--- |
| `data/Soil Moisture, Irrigation Actuator and Weather Dat/.../` | `29` | `85,331,893` | `81.38 MiB` | **Kept** — `CC BY 4.0`, DOI added to `data/README.md` |
| `data/external/pest/fungal_flax_rust/` | `10` | `6,073,441` | `5.79 MiB` | **Kept** — `CC0 1.0`, Dryad DOI & attribution added to `data/README.md` |
| `data/synthetic_nest/` | `11` | `81,346` | `0.08 MiB` | **Kept** — project-authored `SYNTHETIC` scenarios |
| `data/processed/fungal_risk/` | `2` | `29,242` | `0.03 MiB` | **Kept** — derived from CC0 Dryad flax rust dataset |
| `data/README.md` | `1` | `10,026` | `0.01 MiB` | **Kept** — updated with CC BY 4.0 & CC0 1.0 citations |

---

## 2. Final Classification & Decisions Under `data/`

### Category 1: Safe to Redistribute *(Kept Staged)*

| Dataset / Path | Staged Files & Size | Verified License, Source & Citation Added to `data/README.md` |
| :--- | :---: | :--- |
| **`data/synthetic_nest/`** & **`data/README.md`** | `12` files, `91,372` B (`0.09 MiB`) | Project-authored synthetic telemetry (`generate_synthetic.py`, seed=42, `data_source="SYNTHETIC"`) and project registry docs. Needed for `GET /stability/scenarios/*` and tests. |
| **Italian Multi-Sector Precision-Irrigation Trial**<br>`data/Soil Moisture, Irrigation Actuator and Weather Dat/.../` | `29` files, `85,331,893` B (`81.38 MiB`) | Verified Mendeley Data record under **Creative Commons Attribution 4.0 International (`CC BY 4.0`)** (weather via Open-Meteo / Copernicus ERA5, `CC BY 4.0`).<br>**Citation added to `data/README.md`:** Adamo, Tommaso; Colizzi, Lucio; Dimauro, Giovanni; Guerriero, Emanuela; Lomonte, Nunzia (2026), DOI `10.17632/c837v6p8ph` (`10.17632/c837v6p8ph.2`). |
| **Colorado Wild Flax Rust Dataset & Derived Table**<br>`data/external/pest/fungal_flax_rust/` & `data/processed/fungal_risk/` | `12` files, `6,102,683` B (`5.82 MiB`) | Verified Dryad Digital Repository record under **CC0 1.0 Universal (Public Domain Dedication)**.<br>**Attribution & DOI added to `data/README.md`:** Duggal, Keenan; Jiranek, Juliana; Machado, Madison; Smith, Peyton; Miller, Ian F.; Metcalf, Jessica C. E., Dryad DOI `10.5061/dryad.98sf7m0tc` (also citing Miller et al., *Sci Rep* 12, 14823, 2022, DOI `10.1038/s41598-022-18851-z`, and *Oikos* DOI `10.1111/oik.10845`). |

---

### Category 2: Redistribution Permission Unclear

**None remaining staged.** *(The earlier provisional Category 2 holds on the Italian Mendeley and Dryad Flax Rust datasets were resolved by verifying both DOIs and official CC BY 4.0 / CC0 1.0 license terms and documenting them in `data/README.md`.)*

---

### Category 3: Excluded *(Unstaged via `git rm --cached -r` & Added to `.gitignore`; Preserved Locally)*

| Path Unstaged & Gitignored | Files | Exact Local Size | Reason for Exclusion |
| :--- | :---: | :---: | :--- |
| `data/external/pest/weekly_lot_a_with_moon_dataset.csv` | `1` | `31,716` B | Zero README, zero identifiable trap-count source/author, zero open license. |
| `data/external/pest/RBB_P75.1_binary_labels.csv` | `1` | `50,919` B | Zero README, source, or license notice (derived binary labels from above). |
| `data/external/pest/WSB_P94.3_binary_labels.csv` | `1` | `47,477` B | Zero README, source, or license notice (derived binary labels from above). |
| `data/processed/pest_risk/` (`3` derived files) | `3` | `90,123` B | Derived from undocumented-license rice pest CSVs above; excluded pending upstream clearance. |
| `data/external/pest/fungal/Grape Disease Dataset/` | `1` | `287,093` B | Zero README, zero source or license, and audited as unlabeled/completely unused. |
| **Total Unstaged & Ignored** | **`7`** | **`507,328` B (`0.48 MiB`)** | All `7` files verified intact on local disk (`git rm --cached -r` + `.gitignore` lines `52–57`). |

---

## 3. Unstaged Files Already Excluded from Index

- **`data/company_sample/nest_sample.json`** (`424` bytes) — Remains unstaged (company-provided reference schema).
- **`raw_actuators.csv`** (`140,413,377` bytes / `133.91 MiB`) and **`raw_measurements.csv`** (`255,415,996` bytes / `243.58 MiB`) — Remain unstaged (`>100 MiB` GitHub file-size limit).

---

## 4. Pre-Commit Operational Notes

1. **Stage updated documentation/config files before committing:** Running `git status --short` shows `AM .gitignore`, `AM data/README.md`, and `?? docs/GITHUB_UPLOAD_AUDIT.md` in the working tree because only the unstaging action was applied to the Git index. Stage these files when ready to commit.
2. **Fresh-clone test expectations:** Locally, all `112` tests pass with the datasets present on disk. On a clean GitHub clone without the excluded/unstaged `data/processed/pest_risk/` (`test_pest_dataset.py`) and `data/company_sample/nest_sample.json` (`test_inference.py`), those specific local-file tests expect local copies of those datasets unless skipped or mocked for CI.
