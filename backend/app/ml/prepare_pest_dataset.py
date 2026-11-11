"""
Pest-Risk Modelling Data Preparation (RBB Primary Target, WSB Separate)
=======================================================================
Prepares a clean, chronologically ordered, leak-free modelling dataset from
`data/external/pest/weekly_lot_a_with_moon_dataset.csv`:

Key Engineering & Integrity Guarantees:
1. Target Separation: Uses `RBB` (Rice Black Bug) weekly pest counts as the
   primary target. Keeps `WSB` (White Stem Borer) strictly isolated in a
   separate output file (`wsb_pest_counts_separate.csv`) so it cannot leak
   into RBB predictors.
2. Chronological Sorting & Boundary Repair: Orders observations chronologically
   by `(sort_year, Week_Number)` where `sort_year` repairs the 3 ISO-week
   year-boundary labels (`2020-12-31` wk 53 labeled 2021, `2022-12-29` wk 52
   labeled 2023, and `2024-12-30` wk 1 of ISO-2025) so all 366 weekly records
   follow exact 7-day chronological continuity.
3. Environmental Features Only: Excludes calendar fields (`Year`, `Month`, `Day`,
   `Week_Number`), corrupted `Date` strings, lunar phase (`Moon_Category`),
   and historical target lags (`RBB`, `WSB`). Engineers strictly causal 1-week
   and 4-week meteorological/soil lag & trend features from the 10 weather/soil
   columns (`T2M`, `T2M_MIN`, `T2M_MAX`, `RH2M`, `PRECTOTCORR`, `WS2M`,
   `WS2M_MAX`, `WS2M_MIN`, `WD2M`, `GWETTOP`).
4. Chronological Validation & Fold-Only Quantile Labels: Replaces the random/
   interleaved `Data_Split` in the external binary CSVs with 4 expanding-window
   yearly walk-forward folds (`2018-2020 -> 2021`, ..., `2018-2023 -> 2024`).
   Computes `LOW / MEDIUM / HIGH` tertile thresholds (`q33.3`, `q66.7`) strictly
   within each training fold only.
5. Zero Model Training: Saves preprocessed datasets and fold metadata only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
RAW_PEST_CSV = ROOT_DIR / "data" / "external" / "pest" / "weekly_lot_a_with_moon_dataset.csv"
OUT_DIR = ROOT_DIR / "data" / "processed" / "pest_risk"

BASE_ENV_COLS = [
    "T2M",
    "T2M_MIN",
    "T2M_MAX",
    "RH2M",
    "PRECTOTCORR",
    "WS2M",
    "WS2M_MAX",
    "WS2M_MIN",
    "WD2M",
    "GWETTOP",
]

ENVIRONMENTAL_FEATURE_COLS = [
    # Current week meteorological & soil features (10)
    "T2M", "T2M_MIN", "T2M_MAX", "RH2M", "PRECTOTCORR",
    "WS2M", "WS2M_MAX", "WS2M_MIN", "WD2M", "GWETTOP",
    # Derived intra-week meteorological features (4)
    "temp_diurnal_range", "wind_gust_spread", "wd_sin", "wd_cos",
    # Causal 1-week lag environmental features (5)
    "t2m_lag_1w", "rh2m_lag_1w", "rain_lag_1w", "gwettop_lag_1w", "ws2m_lag_1w",
    # Causal 4-week rolling environmental summaries [t-3..t] (4)
    "t2m_mean_4w", "rh2m_mean_4w", "rain_sum_4w", "gwettop_mean_4w",
    # Causal 4-week environmental trend slopes (3)
    "t2m_trend_4w", "rh2m_trend_4w", "gwettop_trend_4w",
]

WALK_FORWARD_VALIDATION_YEARS = [2021, 2022, 2023, 2024]


def _rolling_ols_slope(arr: np.ndarray) -> float:
    """Compute OLS linear slope across an ordered window of weekly readings."""
    x = np.arange(len(arr), dtype=float)
    if len(x) < 2 or np.std(arr) < 1e-12:
        return 0.0
    return float(np.polyfit(x, arr, 1)[0])


def assign_pest_fold_labels(
    train_counts: pd.Series | np.ndarray,
    eval_counts: pd.Series | np.ndarray,
    q_low: float = 1.0 / 3.0,
    q_high: float = 2.0 / 3.0,
) -> tuple[list[str], float, float]:
    """
    Calculate LOW / MEDIUM / HIGH RBB count thresholds STRICTLY within the
    training fold and assign labels to `eval_counts`.
    """
    train_arr = np.asarray(train_counts, dtype=float)
    eval_arr = np.asarray(eval_counts, dtype=float)

    t_low = float(np.quantile(train_arr, q_low))
    t_high = float(np.quantile(train_arr, q_high))

    labels: list[str] = []
    for val in eval_arr:
        if val <= t_low:
            labels.append("LOW")
        elif val <= t_high:
            labels.append("MEDIUM")
        else:
            labels.append("HIGH")

    return labels, t_low, t_high


def _repair_iso_year_for_chronological_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure chronological ordering by `(sort_year, Week_Number)`.

    In the raw CSV, `Week_Number` follows ISO-8601 weeks on a strict 7-day cadence
    starting 2018-01-04 (366/366 exact match), while 3 year-boundary rows carry
    an adjacent calendar/corrupted Year:
      - Row 156: Week 53 of ISO-2020 (marked Year=2021, Date='7/17/2021') -> sort_year=2020
      - Row 260: Week 52 of ISO-2022 (marked Year=2023, Date='10/24/2023') -> sort_year=2022
      - Row 365: Week 1  of ISO-2025 (marked Year=2024, Date='12/30/2024') -> sort_year=2025

    Using the authoritative 7-day ISO timeline (`2018-01-04 + 7*i`) resolves the 3
    year-boundary mismatches so `(sort_year, Week_Number)` has 0 duplicates and
    exact chronological order.
    """
    df = df.copy()
    expected_dates = pd.date_range("2018-01-04", periods=len(df), freq="7D")
    iso_cal = expected_dates.isocalendar()

    # Verify Week_Number matches ISO week 100% across all 366 records
    assert (iso_cal.week.to_numpy() == df["Week_Number"].to_numpy()).all(), (
        "Week_Number does not match expected weekly ISO sequence"
    )

    df["raw_row_index"] = np.arange(len(df), dtype=int)
    df["sort_year"] = iso_cal.year.to_numpy(dtype=int)
    df["canonical_week_date"] = expected_dates.strftime("%Y-%m-%d")

    df = df.sort_values(["sort_year", "Week_Number"]).reset_index(drop=True)
    assert df.duplicated(subset=["sort_year", "Week_Number"]).sum() == 0
    return df


def prepare_pest_datasets(save_outputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Prepare chronologically sorted, pure-environmental RBB modelling dataset
    and separate WSB target archive with training-fold-only chronological labels.
    """
    if not RAW_PEST_CSV.exists():
        raise FileNotFoundError(f"Source pest dataset not found: {RAW_PEST_CSV}")

    raw_df = pd.read_csv(RAW_PEST_CSV)
    df = _repair_iso_year_for_chronological_sort(raw_df)

    # Engineer environmental-only predictors (zero calendar, lunar, or pest-lag features)
    df["temp_diurnal_range"] = (df["T2M_MAX"] - df["T2M_MIN"]).round(4)
    df["wind_gust_spread"] = (df["WS2M_MAX"] - df["WS2M_MIN"]).round(4)
    wd_rad = np.deg2rad(df["WD2M"].to_numpy(dtype=float))
    df["wd_sin"] = np.round(np.sin(wd_rad), 6)
    df["wd_cos"] = np.round(np.cos(wd_rad), 6)

    # 1-week environmental lags (strictly causal t-1)
    df["t2m_lag_1w"] = df["T2M"].shift(1)
    df["rh2m_lag_1w"] = df["RH2M"].shift(1)
    df["rain_lag_1w"] = df["PRECTOTCORR"].shift(1)
    df["gwettop_lag_1w"] = df["GWETTOP"].shift(1)
    df["ws2m_lag_1w"] = df["WS2M"].shift(1)

    # 4-week rolling environmental history [t-3..t]
    df["t2m_mean_4w"] = df["T2M"].rolling(4).mean().round(4)
    df["rh2m_mean_4w"] = df["RH2M"].rolling(4).mean().round(4)
    df["rain_sum_4w"] = df["PRECTOTCORR"].rolling(4).sum().round(4)
    df["gwettop_mean_4w"] = df["GWETTOP"].rolling(4).mean().round(4)

    # 4-week rolling OLS trends
    df["t2m_trend_4w"] = df["T2M"].rolling(4).apply(_rolling_ols_slope, raw=True).round(6)
    df["rh2m_trend_4w"] = df["RH2M"].rolling(4).apply(_rolling_ols_slope, raw=True).round(6)
    df["gwettop_trend_4w"] = df["GWETTOP"].rolling(4).apply(_rolling_ols_slope, raw=True).round(6)

    # Drop the initial 3 warm-up weeks (2018 wks 1-3) lacking complete 4-week history
    clean_df = df.iloc[3:].copy().reset_index(drop=True)

    # Ensure strictly zero nulls in environmental features and targets
    assert clean_df[ENVIRONMENTAL_FEATURE_COLS].isna().sum().sum() == 0

    # Expanding-window chronological validation folds (training-fold-only quantiles)
    # Also assign fold_group_id and walk-forward out-of-fold labels for 2021-2024
    walk_forward_folds: dict[str, Any] = {}
    oof_labels = pd.Series(index=clean_df.index, data=pd.NA, dtype="object")

    for val_year in WALK_FORWARD_VALIDATION_YEARS:
        # Train strictly on prior years; evaluate on val_year
        # Note: final 2024 ISO-week 2025 wk 1 (2024-12-30) belongs chronologically to the 2024 holdout tail
        if val_year == 2024:
            train_mask = clean_df["sort_year"] < 2024
            val_mask = clean_df["sort_year"] >= 2024
        else:
            train_mask = clean_df["sort_year"] < val_year
            val_mask = clean_df["sort_year"] == val_year

        train_rbb = clean_df.loc[train_mask, "RBB"]
        val_rbb = clean_df.loc[val_mask, "RBB"]

        train_lbls, t_low, t_high = assign_pest_fold_labels(train_rbb, train_rbb)
        val_lbls, _, _ = assign_pest_fold_labels(train_rbb, val_rbb)
        oof_labels.loc[val_mask] = val_lbls

        walk_forward_folds[str(val_year)] = {
            "train_years": [int(y) for y in sorted(clean_df.loc[train_mask, "sort_year"].unique())],
            "eval_year_label": "2024 (+ 2024-12-30 wk)" if val_year == 2024 else str(val_year),
            "train_n": int(train_mask.sum()),
            "eval_n": int(val_mask.sum()),
            "rbb_q33_train_only": round(t_low, 4),
            "rbb_q67_train_only": round(t_high, 4),
            "train_class_counts": pd.Series(train_lbls).value_counts().to_dict(),
            "eval_class_counts": pd.Series(val_lbls).value_counts().to_dict(),
        }

    # Build primary RBB dataset (STRICTLY excluding WSB, Moon_Category, and non-env features)
    rbb_id_cols = ["raw_row_index", "sort_year", "Week_Number", "canonical_week_date"]
    rbb_dataset = clean_df[rbb_id_cols + ["RBB"] + ENVIRONMENTAL_FEATURE_COLS].copy()
    rbb_dataset = rbb_dataset.rename(columns={"RBB": "rbb_count"})
    rbb_dataset["rbb_risk_walk_forward_oof"] = oof_labels.fillna("TRAIN_ONLY_BASE_2018_2020")

    # Assign final split tag strictly chronologically (2018-2022 train/dev, 2023 val, 2024 final holdout test)
    split_map = np.where(
        rbb_dataset["sort_year"] <= 2022,
        "train_early",
        np.where(rbb_dataset["sort_year"] == 2023, "chronological_val_2023", "chronological_test_2024"),
    )
    rbb_dataset["chronological_split"] = split_map

    # Build separate WSB archive (isolated from RBB modelling dataset)
    wsb_separate = clean_df[rbb_id_cols + ["WSB"]].copy().rename(columns={"WSB": "wsb_count"})

    metadata: dict[str, Any] = {
        "source_file": str(RAW_PEST_CSV.relative_to(ROOT_DIR)),
        "raw_rows": int(len(raw_df)),
        "warmup_dropped_rows": 3,
        "prepared_rows": int(len(rbb_dataset)),
        "primary_target": "rbb_count",
        "excluded_columns": [
            "WSB (saved separately to wsb_pest_counts_separate.csv)",
            "Moon_Category (non-meteorological lunar phase)",
            "Date / Year / Month / Day (corrupted date strings & calendar proxies excluded from features)",
            "Data_Split from binary CSVs (rejected due to random/interleaved temporal leakage)",
        ],
        "environmental_feature_count": len(ENVIRONMENTAL_FEATURE_COLS),
        "environmental_features": ENVIRONMENTAL_FEATURE_COLS,
        "chronological_walk_forward_folds": walk_forward_folds,
    }

    if save_outputs:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        rbb_dataset.to_csv(OUT_DIR / "rbb_pest_risk_dataset.csv", index=False)
        wsb_separate.to_csv(OUT_DIR / "wsb_pest_counts_separate.csv", index=False)
        (OUT_DIR / "rbb_chronological_fold_thresholds.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    return rbb_dataset, wsb_separate, metadata


if __name__ == "__main__":
    rbb_df, wsb_df, meta = prepare_pest_datasets(save_outputs=True)
    print(f"RBB Dataset shape: {rbb_df.shape[0]} rows x {rbb_df.shape[1]} columns")
    print(f"WSB Separate shape: {wsb_df.shape[0]} rows x {wsb_df.shape[1]} columns")
    print(f"Null count in RBB dataset: {int(rbb_df.isna().sum().sum())}")
    print(f"Null count in WSB separate: {int(wsb_df.isna().sum().sum())}")
    print(f"Environmental features count: {meta['environmental_feature_count']}")
    print("\nChronological Walk-Forward Fold Thresholds (Train-Fold-Only):")
    for yr, info in meta["chronological_walk_forward_folds"].items():
        print(
            f"  Eval {info['eval_year_label']:22s} | train_n={info['train_n']:3d}, eval_n={info['eval_n']:2d} | "
            f"q33={info['rbb_q33_train_only']:6.2f}, q67={info['rbb_q67_train_only']:6.2f} | "
            f"eval_dist={info['eval_class_counts']}"
        )
