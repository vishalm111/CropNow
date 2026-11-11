"""
Fungal-Risk Modelling Dataset Preparation (Flax Rust Pathosystem)
=================================================================
Prepares a clean, leak-free tabular dataset joining site-level fungal disease
observations (`Cleaned_Epi.rds`) with 5-minute / daily environmental telemetry
(`cleaned_temp.rds`, `cleaned_weather.rds`) from `data/external/pest/fungal_flax_rust/`.

Key Engineering Principles:
1. Zero Same-Day Leakage: Uses strictly preceding 7-day window `[d-7 .. d-1]`
   before each survey date `d` (`days_since_june_1st`).
2. Complete-Day Quality Gate: Resamples 5-minute (and 1-minute at HM 2020) telemetry
   onto a uniform 5-minute grid and retains complete 24-hour days (>= 270 / 288 bins)
   so partial deployment/retrieval boundary days do not skew averages or sums.
3. Leak-Free Labels: Stores raw continuous targets (`prevalence`, `new_infections`,
   `incidence_rate`) and computes `LOW / MEDIUM / HIGH` labels strictly inside
   training folds only (Leave-One-Year-Out and Leave-One-Site-Out) using
   prevalence tertiles (33.3rd and 66.7th percentiles).
4. Zero Model Training: Prepares and validates the dataset only; does not train models.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
RAW_FLAX_DIR = ROOT_DIR / "data" / "external" / "pest" / "fungal_flax_rust"
OUT_DIR = ROOT_DIR / "data" / "processed" / "fungal_risk"

DEFAULT_RSCRIPT_PATHS = [
    shutil.which("Rscript"),
    r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.4.0\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.3.3\bin\Rscript.exe",
]


def _find_rscript() -> str:
    """Locate Rscript executable for reading untouched .rds source files."""
    for candidate in DEFAULT_RSCRIPT_PATHS:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError(
        "Rscript executable not found. Install R or add Rscript to PATH to read .rds source files."
    )


def _export_rds_to_csv(temp_dir: Path) -> tuple[Path, Path, Path]:
    """Export Cleaned_Epi.rds, cleaned_temp.rds, cleaned_weather.rds to temp CSVs without modifying sources."""
    rscript = _find_rscript()
    epi_csv = temp_dir / "Cleaned_Epi.csv"
    temp_csv = temp_dir / "cleaned_temp.csv"
    weath_csv = temp_dir / "cleaned_weather.csv"

    r_code = f"""
    epi <- readRDS("{ (RAW_FLAX_DIR / 'Cleaned_Epi.rds').as_posix() }")
    mt  <- readRDS("{ (RAW_FLAX_DIR / 'cleaned_temp.rds').as_posix() }")
    mw  <- readRDS("{ (RAW_FLAX_DIR / 'cleaned_weather.rds').as_posix() }")

    epi$Disease_date <- as.character(epi$Disease_date)
    mt$times <- format(mt$times, "%Y-%m-%d %H:%M:%S", tz="UTC")
    mw$times <- format(mw$times, "%Y-%m-%d %H:%M:%S", tz="UTC")

    write.csv(epi, "{ epi_csv.as_posix() }", row.names=FALSE)
    write.csv(mt,  "{ temp_csv.as_posix() }", row.names=FALSE)
    write.csv(mw,  "{ weath_csv.as_posix() }", row.names=FALSE)
    """
    r_script_file = temp_dir / "export_rds.R"
    r_script_file.write_text(r_code, encoding="utf-8")

    proc = subprocess.run(
        [rscript, str(r_script_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Rscript RDS export failed:\n{proc.stderr}")

    return epi_csv, temp_csv, weath_csv


def assign_fold_risk_labels(
    train_prevalence: pd.Series | np.ndarray,
    eval_prevalence: pd.Series | np.ndarray,
    q_low: float = 1.0 / 3.0,
    q_high: float = 2.0 / 3.0,
) -> tuple[list[str], float, float]:
    """
    Compute LOW / MEDIUM / HIGH prevalence thresholds STRICTLY on `train_prevalence`
    and apply them to `eval_prevalence` without target leakage.

    Thresholds:
      - LOW:    prevalence <= q33_train
      - MEDIUM: q33_train < prevalence <= q67_train
      - HIGH:   prevalence > q67_train
    """
    train_arr = np.asarray(train_prevalence, dtype=float)
    eval_arr = np.asarray(eval_prevalence, dtype=float)

    t_low = float(np.quantile(train_arr, q_low))
    t_high = float(np.quantile(train_arr, q_high))

    labels = []
    for val in eval_arr:
        if val <= t_low:
            labels.append("LOW")
        elif val <= t_high:
            labels.append("MEDIUM")
        else:
            labels.append("HIGH")

    return labels, t_low, t_high


def _ols_slope(values: np.ndarray) -> float:
    """Compute OLS linear trend slope over ordered daily values (unit/day)."""
    x = np.arange(len(values), dtype=float)
    if len(x) < 2 or np.std(values) < 1e-12:
        return 0.0
    slope = float(np.polyfit(x, values, 1)[0])
    return slope


def _build_daily_environmental_table(temp_csv: Path, weath_csv: Path) -> pd.DataFrame:
    """
    Resample 5-min (and 1-min HM 2020) telemetry to uniform 5-min bins,
    join temperature/RH/dew-point with leaf wetness/rain/solar/soil water content,
    and aggregate to complete daily summaries (`>= 270 / 288` 5-min bins).
    """
    df_t = pd.read_csv(temp_csv)
    df_w = pd.read_csv(weath_csv)

    # Filter to 2020-2023 where both temperature/RH and weather/leaf wetness sensors exist
    df_t = df_t[df_t["Year"] >= 2020].copy()
    df_w = df_w[df_w["Year"] >= 2020].copy()

    # Round timestamps to 5-minute grid (normalizes HM 2020 1-min logging to 5-min cadence)
    df_t["ts5"] = pd.to_datetime(df_t["times"]).dt.round("5min")
    df_w["ts5"] = pd.to_datetime(df_w["times"]).dt.round("5min")

    # Group to uniform 5-min grid (handling any sub-5-min rows or single NA readings gracefully)
    t_5m = (
        df_t.groupby(["Site", "Year", "days_since_june_1st", "ts5"], as_index=False)
        .agg(
            Temp=("Temp", "mean"),
            RH=("RH", "mean"),
            Dew_point=("Dew_point", "mean"),
        )
    )
    w_5m = (
        df_w.groupby(["Site", "Year", "days_since_june_1st", "ts5"], as_index=False)
        .agg(
            Leaf_Wetness=("Leaf_Wetness", "mean"),
            Rain=("Rain", "sum"),
            Solar_Radiation=("Solar_Radiation", "mean"),
            Water_Content=("Water_Content", "mean"),
        )
    )

    # Inner-join 5-min grids per site/year/day
    merged_5m = pd.merge(
        t_5m,
        w_5m,
        on=["Site", "Year", "days_since_june_1st", "ts5"],
        how="inner",
    )

    # Precompute indicator flags per 5-minute bin (each valid bin = 5/60 hours)
    merged_5m["dp_depression"] = merged_5m["Temp"] - merged_5m["Dew_point"]
    merged_5m["is_rh_80"] = (merged_5m["RH"] >= 80.0).astype(float)
    merged_5m["is_rh_90"] = (merged_5m["RH"] >= 90.0).astype(float)
    merged_5m["is_lw_wet"] = (merged_5m["Leaf_Wetness"] > 0.0).astype(float)
    merged_5m["is_lw_heavy"] = (merged_5m["Leaf_Wetness"] >= 20.0).astype(float)
    # Biological infection window: free surface moisture + moderate subalpine temp [5, 25] C
    merged_5m["is_favorable_inf"] = (
        (merged_5m["Leaf_Wetness"] > 0.0)
        & (merged_5m["Temp"] >= 5.0)
        & (merged_5m["Temp"] <= 25.0)
    ).astype(float)

    hrs_per_bin = 5.0 / 60.0

    daily = (
        merged_5m.groupby(["Site", "Year", "days_since_june_1st"], as_index=False)
        .agg(
            n_bins=("ts5", "count"),
            temp_mean=("Temp", "mean"),
            temp_min=("Temp", "min"),
            temp_max=("Temp", "max"),
            temp_std=("Temp", "std"),
            dp_mean=("Dew_point", "mean"),
            dp_depression_mean=("dp_depression", "mean"),
            rh_mean=("RH", "mean"),
            rh_min=("RH", "min"),
            rh_max=("RH", "max"),
            rh_std=("RH", "std"),
            rh_ge_80_bins=("is_rh_80", "sum"),
            rh_ge_90_bins=("is_rh_90", "sum"),
            lw_mean=("Leaf_Wetness", "mean"),
            lw_max=("Leaf_Wetness", "max"),
            lw_wet_bins=("is_lw_wet", "sum"),
            lw_heavy_bins=("is_lw_heavy", "sum"),
            favorable_inf_bins=("is_favorable_inf", "sum"),
            rain_total=("Rain", "sum"),
            solar_mean=("Solar_Radiation", "mean"),
            water_content_mean=("Water_Content", "mean"),
        )
    )

    # Convert bin counts to hours
    daily["rh_ge_80_hours"] = daily["rh_ge_80_bins"] * hrs_per_bin
    daily["rh_ge_90_hours"] = daily["rh_ge_90_bins"] * hrs_per_bin
    daily["lw_wet_hours"] = daily["lw_wet_bins"] * hrs_per_bin
    daily["lw_heavy_hours"] = daily["lw_heavy_bins"] * hrs_per_bin
    daily["favorable_inf_hours"] = daily["favorable_inf_bins"] * hrs_per_bin
    daily["temp_diurnal_range"] = daily["temp_max"] - daily["temp_min"]

    # Retain only complete 24-hour days (>= 270 / 288 5-min bins = >= 22.5h coverage)
    # so partial logger deployment/retrieval days cannot distort daily means or sums.
    daily = daily[daily["n_bins"] >= 270].copy()
    daily = daily.dropna().reset_index(drop=True)
    return daily


def prepare_dataset(save_outputs: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Execute end-to-end join, feature extraction, and fold-safe label threshold audit.
    """
    if not RAW_FLAX_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {RAW_FLAX_DIR}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        epi_csv, temp_csv, weath_csv = _export_rds_to_csv(tmp_path)

        df_epi = pd.read_csv(epi_csv)
        daily_env = _build_daily_environmental_table(temp_csv, weath_csv)

    # Filter disease observations to 2020-2023 where leaf wetness & rain sensors exist
    df_epi = df_epi[df_epi["Year"] >= 2020].copy()
    df_epi = df_epi.sort_values(["Year", "Site", "days_since_june1st"]).reset_index(drop=True)

    # Index daily environmental lookup by (Site, Year, days_since_june_1st)
    env_lookup = daily_env.set_index(["Site", "Year", "days_since_june_1st"])

    rows: list[dict[str, Any]] = []
    skipped_incomplete: list[str] = []

    for _, epi_row in df_epi.iterrows():
        site = str(epi_row["Site"])
        year = int(epi_row["Year"])
        d_survey = int(epi_row["days_since_june1st"])
        window_days = list(range(d_survey - 7, d_survey))  # strictly [d-7 .. d-1]

        keys = [(site, year, d) for d in window_days]
        if not all(k in env_lookup.index for k in keys):
            skipped_incomplete.append(f"{site}_{year}_d{d_survey}")
            continue

        win_df = env_lookup.loc[keys].reset_index()
        assert len(win_df) == 7

        # Recent 3-day [d-3..d-1] vs Prior 4-day [d-7..d-4] split for trend deltas
        prior_4d = win_df.iloc[:4]
        recent_3d = win_df.iloc[4:]
        day_prev = win_df.iloc[-1]  # day d-1 (24h prior to survey)

        host_count = int(epi_row["count"])
        new_inf = int(epi_row["new_infections"])
        prevalence = float(epi_row["prevalence"])
        prev_no_recov = float(epi_row["prevalence_noRecov"])

        record: dict[str, Any] = {
            # Identifiers & temporal anchors
            "observation_id": f"{site}_{year}_d{d_survey}",
            "site": site,
            "year": year,
            "disease_date": str(epi_row["Disease_date"]),
            "days_since_june_1st": d_survey,
            "window_start_day": d_survey - 7,
            "window_end_day": d_survey - 1,
            # Continuous disease targets (labels derived strictly within training folds)
            "host_count": host_count,
            "new_infections": new_inf,
            "incidence_rate": round(new_inf / max(host_count, 1), 6),
            "prevalence": round(prevalence, 6),
            "prevalence_no_recov": round(prev_no_recov, 6),
            # Temperature features (C)
            "temp_mean_7d": round(float(win_df["temp_mean"].mean()), 4),
            "temp_min_7d": round(float(win_df["temp_min"].min()), 4),
            "temp_max_7d": round(float(win_df["temp_max"].max()), 4),
            "temp_std_7d": round(float(win_df["temp_mean"].std(ddof=1)), 4),
            "temp_diurnal_range_mean_7d": round(float(win_df["temp_diurnal_range"].mean()), 4),
            "temp_mean_24h": round(float(day_prev["temp_mean"]), 4),
            "dp_mean_7d": round(float(win_df["dp_mean"].mean()), 4),
            "dp_depression_mean_7d": round(float(win_df["dp_depression_mean"].mean()), 4),
            # Humidity features (% RH)
            "rh_mean_7d": round(float(win_df["rh_mean"].mean()), 4),
            "rh_min_7d": round(float(win_df["rh_min"].min()), 4),
            "rh_max_7d": round(float(win_df["rh_max"].max()), 4),
            "rh_std_7d": round(float(win_df["rh_mean"].std(ddof=1)), 4),
            "rh_mean_24h": round(float(day_prev["rh_mean"]), 4),
            "rh_ge_80_hours_7d": round(float(win_df["rh_ge_80_hours"].sum()), 2),
            "rh_ge_90_hours_7d": round(float(win_df["rh_ge_90_hours"].sum()), 2),
            # Leaf wetness features
            "lw_mean_7d": round(float(win_df["lw_mean"].mean()), 4),
            "lw_max_7d": round(float(win_df["lw_max"].max()), 4),
            "lw_mean_24h": round(float(day_prev["lw_mean"]), 4),
            "lw_wet_hours_7d": round(float(win_df["lw_wet_hours"].sum()), 2),
            "lw_heavy_wet_hours_7d": round(float(win_df["lw_heavy_hours"].sum()), 2),
            "lw_wet_hours_24h": round(float(day_prev["lw_wet_hours"]), 2),
            "favorable_infection_hours_7d": round(float(win_df["favorable_inf_hours"].sum()), 2),
            # Rainfall, solar & soil water content
            "rain_total_7d": round(float(win_df["rain_total"].sum()), 4),
            "rain_max_daily_7d": round(float(win_df["rain_total"].max()), 4),
            "rain_days_count_7d": int((win_df["rain_total"] > 0.2).sum()),
            "rain_total_24h": round(float(day_prev["rain_total"]), 4),
            "soil_moisture_mean_7d": round(float(win_df["water_content_mean"].mean()), 4),
            "soil_moisture_mean_24h": round(float(day_prev["water_content_mean"]), 4),
            "solar_radiation_mean_7d": round(float(win_df["solar_mean"].mean()), 4),
            # 7-day environmental trends (OLS slopes & 3d-vs-4d deltas)
            "temp_trend_slope_7d": round(_ols_slope(win_df["temp_mean"].to_numpy()), 4),
            "rh_trend_slope_7d": round(_ols_slope(win_df["rh_mean"].to_numpy()), 4),
            "lw_hours_trend_slope_7d": round(_ols_slope(win_df["lw_wet_hours"].to_numpy()), 4),
            "rain_trend_slope_7d": round(_ols_slope(win_df["rain_total"].to_numpy()), 4),
            "soil_moisture_trend_slope_7d": round(_ols_slope(win_df["water_content_mean"].to_numpy()), 6),
            "temp_delta_3d_vs_4d": round(float(recent_3d["temp_mean"].mean() - prior_4d["temp_mean"].mean()), 4),
            "rh_delta_3d_vs_4d": round(float(recent_3d["rh_mean"].mean() - prior_4d["rh_mean"].mean()), 4),
            "lw_hours_delta_3d_vs_4d": round(float(recent_3d["lw_wet_hours"].mean() - prior_4d["lw_wet_hours"].mean()), 4),
            "rain_delta_3d_vs_4d": round(float(recent_3d["rain_total"].mean() - prior_4d["rain_total"].mean()), 4),
        }
        rows.append(record)

    dataset = pd.DataFrame(rows)

    # Compute out-of-fold LOW / MEDIUM / HIGH labels strictly from training folds only!
    # 1. Leave-One-Year-Out (LOYO) CV folds
    loyo_labels = pd.Series(index=dataset.index, dtype="object")
    loyo_thresholds: dict[str, Any] = {}
    for test_year in sorted(dataset["year"].unique()):
        train_mask = dataset["year"] != test_year
        test_mask = dataset["year"] == test_year
        fold_lbls, t_low, t_high = assign_fold_risk_labels(
            dataset.loc[train_mask, "prevalence"],
            dataset.loc[test_mask, "prevalence"],
        )
        loyo_labels.loc[test_mask] = fold_lbls
        train_lbls, _, _ = assign_fold_risk_labels(
            dataset.loc[train_mask, "prevalence"],
            dataset.loc[train_mask, "prevalence"],
        )
        loyo_thresholds[str(test_year)] = {
            "train_years": [int(y) for y in sorted(dataset.loc[train_mask, "year"].unique())],
            "test_year": int(test_year),
            "train_n": int(train_mask.sum()),
            "test_n": int(test_mask.sum()),
            "q33_low_threshold": round(t_low, 6),
            "q67_high_threshold": round(t_high, 6),
            "train_class_counts": pd.Series(train_lbls).value_counts().to_dict(),
            "test_class_counts": pd.Series(fold_lbls).value_counts().to_dict(),
        }
    dataset["risk_label_loyo_oof"] = loyo_labels

    # 2. Leave-One-Site-Out (LOSO) CV folds
    loso_labels = pd.Series(index=dataset.index, dtype="object")
    loso_thresholds: dict[str, Any] = {}
    for test_site in sorted(dataset["site"].unique()):
        train_mask = dataset["site"] != test_site
        test_mask = dataset["site"] == test_site
        fold_lbls, t_low, t_high = assign_fold_risk_labels(
            dataset.loc[train_mask, "prevalence"],
            dataset.loc[test_mask, "prevalence"],
        )
        loso_labels.loc[test_mask] = fold_lbls
        train_lbls, _, _ = assign_fold_risk_labels(
            dataset.loc[train_mask, "prevalence"],
            dataset.loc[train_mask, "prevalence"],
        )
        loso_thresholds[str(test_site)] = {
            "train_sites": sorted(dataset.loc[train_mask, "site"].unique().tolist()),
            "test_site": str(test_site),
            "train_n": int(train_mask.sum()),
            "test_n": int(test_mask.sum()),
            "q33_low_threshold": round(t_low, 6),
            "q67_high_threshold": round(t_high, 6),
            "train_class_counts": pd.Series(train_lbls).value_counts().to_dict(),
            "test_class_counts": pd.Series(fold_lbls).value_counts().to_dict(),
        }
    dataset["risk_label_loso_oof"] = loso_labels

    metadata: dict[str, Any] = {
        "source_directory": str(RAW_FLAX_DIR.relative_to(ROOT_DIR)),
        "rows_total_2020_2023_epi": int(len(df_epi)),
        "rows_retained_complete_7d": int(len(dataset)),
        "rows_skipped_incomplete_window": len(skipped_incomplete),
        "skipped_observation_ids": skipped_incomplete,
        "lookback_window": "[d-7 .. d-1] (strictly prior 7 complete calendar days)",
        "quantile_definition": {
            "method": "tertiles computed strictly on training fold prevalence",
            "q_low": 1.0 / 3.0,
            "q_high": 2.0 / 3.0,
            "LOW": "prevalence <= q33_train",
            "MEDIUM": "q33_train < prevalence <= q67_train",
            "HIGH": "prevalence > q67_train",
        },
        "loyo_fold_thresholds": loyo_thresholds,
        "loso_fold_thresholds": loso_thresholds,
    }

    if save_outputs:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = OUT_DIR / "flax_rust_fungal_risk_dataset.csv"
        out_meta = OUT_DIR / "flax_rust_fold_thresholds.json"
        dataset.to_csv(out_csv, index=False)
        out_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return dataset, metadata


if __name__ == "__main__":
    df_out, meta_out = prepare_dataset(save_outputs=True)
    print(f"Prepared dataset shape: {df_out.shape[0]} rows x {df_out.shape[1]} columns")
    print(f"Null count across entire dataset: {int(df_out.isna().sum().sum())}")
    print(f"Duplicate rows: {int(df_out.duplicated().sum())}")
    print(f"Skipped incomplete windows ({meta_out['rows_skipped_incomplete_window']}): {meta_out['skipped_observation_ids'][:10]}...")
    print("\nLOYO Out-of-Fold Class Counts:")
    print(df_out["risk_label_loyo_oof"].value_counts())
    print("\nLOSO Out-of-Fold Class Counts:")
    print(df_out["risk_label_loso_oof"].value_counts())
    print("\nLOYO Fold Thresholds (computed within training folds only):")
    for y, info in meta_out["loyo_fold_thresholds"].items():
        print(f"  Test Year {y}: train_n={info['train_n']}, test_n={info['test_n']}, q33={info['q33_low_threshold']:.4f}, q67={info['q67_high_threshold']:.4f}, test_dist={info['test_class_counts']}")
    print("\nLOSO Fold Thresholds (computed within training folds only):")
    for s, info in meta_out["loso_fold_thresholds"].items():
        print(f"  Test Site {s}: train_n={info['train_n']}, test_n={info['test_n']}, q33={info['q33_low_threshold']:.4f}, q67={info['q67_high_threshold']:.4f}, test_dist={info['test_class_counts']}")
