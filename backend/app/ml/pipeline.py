"""
Irrigation Prediction Pipeline for NEST Dataset.

Requirements addressed:
1. Zones 1, 2, 4, 5 included; Zone 3 excluded.
2. 10-minute sensor telemetry resampled into 30-minute packets.
3. 48-step sequences (24h lookback) constructed strictly without crossing timestamp gaps.
4. `target_mean_24h` alignment verified before setting as regression target.
5. Future-looking columns (real_moisture_delta, water_vol_to_24h) excluded to prevent leakage.
6. Chronological train/val/test splits (70% / 15% / 15%) with a 24-hour boundary gap.
7. Evaluated with MAE and RMSE across overall test set and per-zone.
8. Models, scalers, and evaluation results saved to disk.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# ---------------------------------------------------------------------------
# Configuration & Documented Agronomic Constants
# ---------------------------------------------------------------------------
CROP_SPECS = {
    1: {"name": "Tomato (Open Field)", "soil_model": "TOMATO_SOIL", "flow_rate": 1.1, "pwp": 20.0, "madp": 40.0, "fc": 60.0, "sp": 80.0},
    2: {"name": "Tomato (Open Field)", "soil_model": "TOMATO_SOIL", "flow_rate": 1.1, "pwp": 20.0, "madp": 40.0, "fc": 60.0, "sp": 80.0},
    4: {"name": "Zucchini (Soil)",     "soil_model": "ZUCCHINI_SOIL", "flow_rate": 1.1, "pwp": 40.0, "madp": 80.0, "fc": 90.0, "sp": 95.0},
    5: {"name": "Blueberry (Soil)",    "soil_model": "BLUEBERRY_SOIL", "flow_rate": 0.4, "pwp": 35.0, "madp": 70.0, "fc": 90.0, "sp": 95.0},
}

AGRONOMIC_CLASSES = {0: "Danger", 1: "Stress", 2: "Optimal", 3: "Excess", 4: "Saturation"}

def classify_agronomic_state(moisture: float, pwp: float, madp: float, fc: float, sp: float) -> int:
    """Classify soil moisture according to documented crop-specific thresholds."""
    if moisture < pwp:
        return 0  # Danger
    if moisture < madp:
        return 1  # Stress
    if moisture < fc:
        return 2  # Optimal
    if moisture < sp:
        return 3  # Excess
    return 4      # Saturation

FEATURE_COLS = [
    "weather_temp",
    "weather_humidity",
    "weather_rain",
    "weather_pressure",
    "weather_wind_speed",
    "weather_radiation",
    "soil_temperature_0-7cm",
    "soil_temperature_7-18cm",
    "ec",
    "ph",
    "soil_moisture",
    "irrigation_duration_minutes",
    "water_vol_past_4h",
]

STATIC_COLS = ["flow_rate", "pwp", "madp", "fc", "sp"]

SEQUENCE_LENGTH = 48  # 48 steps * 30 min = 24 hours lookback
BOUNDARY_GAP_HOURS = 24

# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
def load_and_resample_zone(csv_path: str, zone_id: int):
    """Load preprocessed 10-minute CSV, filter sensor error codes, and resample to 30-min packets."""
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").set_index("ts")

    # Range filter EC and pH sensor error codes (16-bit overflows)
    df["ec"] = df["ec"].clip(lower=0.0, upper=1000.0)
    df["ph"] = df["ph"].clip(lower=3.0, upper=9.0)

    # 30-minute resampling aggregation
    res = df.resample("30min").agg({
        "weather_temp": "mean",
        "weather_humidity": "mean",
        "weather_rain": "sum",
        "weather_pressure": "mean",
        "weather_wind_speed": "mean",
        "weather_radiation": "mean",
        "soil_temperature_0-7cm": "mean",
        "soil_temperature_7-18cm": "mean",
        "ec": "mean",
        "ph": "mean",
        "soil_moisture": "mean",
        "irrigation_duration_minutes": "sum",
        "water_vol_past_4h": "last",
        "target_mean_24h": "last",
    })

    # A packet is valid if primary sensors and targets are not null
    valid_mask = (
        res["soil_moisture"].notna()
        & res["weather_temp"].notna()
        & res["target_mean_24h"].notna()
    )

    # Attach static crop specifications
    spec = CROP_SPECS[zone_id]
    for col in STATIC_COLS:
        res[col] = spec[col]

    return res, valid_mask


def build_contiguous_sequences(res: pd.DataFrame, valid_mask: pd.Series, zone_id: int):
    """Construct 48-step sequences without crossing timestamp gaps."""
    # Contiguous groups: an interruption occurs whenever valid_mask is False or ts.diff() != 30 min
    is_break = (~valid_mask)
    block_id = is_break.cumsum()
    valid_blocks = res[valid_mask].groupby(block_id[valid_mask])

    sequences = []
    targets = []
    metadata = []

    all_cols = FEATURE_COLS + STATIC_COLS

    for _, block in valid_blocks:
        # Check strict 30-min cadence within block
        diffs = block.index.to_series().diff().dropna()
        if not (diffs == pd.Timedelta(minutes=30)).all():
            continue

        if len(block) >= SEQUENCE_LENGTH:
            block_features = block[all_cols].values
            block_targets = block["target_mean_24h"].values
            block_timestamps = block.index

            for i in range(len(block) - SEQUENCE_LENGTH + 1):
                seq_x = block_features[i : i + SEQUENCE_LENGTH]
                seq_y = block_targets[i + SEQUENCE_LENGTH - 1]
                start_t = block_timestamps[i]
                end_t = block_timestamps[i + SEQUENCE_LENGTH - 1]

                sequences.append(seq_x)
                targets.append(seq_y)
                metadata.append({
                    "zone": zone_id,
                    "crop": CROP_SPECS[zone_id]["name"],
                    "start_ts": start_t,
                    "end_ts": end_t,
                    "target": seq_y,
                    "last_moisture": seq_x[-1, FEATURE_COLS.index("soil_moisture")],
                    "pwp": CROP_SPECS[zone_id]["pwp"],
                    "madp": CROP_SPECS[zone_id]["madp"],
                    "fc": CROP_SPECS[zone_id]["fc"],
                    "sp": CROP_SPECS[zone_id]["sp"],
                })

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32), pd.DataFrame(metadata)


def split_zone_chronological(meta_df: pd.DataFrame):
    """Assign train/val/test splits chronologically with a 24-hour boundary gap."""
    min_t = meta_df["end_ts"].min()
    max_t = meta_df["end_ts"].max()
    span = max_t - min_t

    t_train_end = min_t + span * 0.70
    t_val_start = t_train_end + pd.Timedelta(hours=BOUNDARY_GAP_HOURS)
    t_val_end = min_t + span * 0.85
    t_test_start = t_val_end + pd.Timedelta(hours=BOUNDARY_GAP_HOURS)

    splits = pd.Series("buffer", index=meta_df.index)
    splits[meta_df["end_ts"] <= t_train_end] = "train"
    splits[(meta_df["start_ts"] >= t_val_start) & (meta_df["end_ts"] <= t_val_end)] = "val"
    splits[meta_df["start_ts"] >= t_test_start] = "test"

    return splits


# ---------------------------------------------------------------------------
# Tabular Feature Extraction for XGBoost
# ---------------------------------------------------------------------------
def extract_tabular_features(X_seq: np.ndarray, all_cols: list) -> np.ndarray:
    """Extract flattened features and window aggregations for tree models."""
    N, T, F = X_seq.shape
    sm_idx = all_cols.index("soil_moisture")
    temp_idx = all_cols.index("weather_temp")
    rad_idx = all_cols.index("weather_radiation")
    rain_idx = all_cols.index("weather_rain")
    irr_idx = all_cols.index("irrigation_duration_minutes")
    water_idx = all_cols.index("water_vol_past_4h")

    # 1. Full flattened sequence: N x (T * F)
    flat_seq = X_seq.reshape(N, T * F)

    # 2. Key domain aggregations over the 48-step window
    sm_window = X_seq[:, :, sm_idx]
    temp_window = X_seq[:, :, temp_idx]
    rad_window = X_seq[:, :, rad_idx]
    rain_window = X_seq[:, :, rain_idx]
    irr_window = X_seq[:, :, irr_idx]

    stats = np.column_stack([
        sm_window[:, -1],                         # Current moisture
        sm_window[:, -1] - sm_window[:, 0],       # 24h moisture delta
        sm_window[:, -1] - sm_window[:, -9],      # 4h moisture delta (8 steps)
        np.mean(sm_window, axis=1),               # Window mean
        np.std(sm_window, axis=1),                # Window std
        np.min(sm_window, axis=1),                # Window min
        np.max(sm_window, axis=1),                # Window max
        np.mean(temp_window, axis=1),             # Mean temp
        np.max(temp_window, axis=1),              # Max temp
        np.mean(rad_window, axis=1),              # Mean solar radiation
        np.sum(rain_window, axis=1),              # Total rain 24h
        np.sum(rain_window[:, -8:], axis=1),      # Rain last 4h
        np.sum(irr_window, axis=1),               # Total irrigation minutes 24h
        X_seq[:, -1, water_idx],                  # Water volume past 4h
    ])

    return np.hstack([flat_seq, stats])


# ---------------------------------------------------------------------------
# GRU Neural Network Architecture
# ---------------------------------------------------------------------------
class SoilMoistureGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last_step = out[:, -1, :]  # State at t=48
        pred = self.head(last_step)
        return pred.squeeze(-1)


# ---------------------------------------------------------------------------
# Main Training & Evaluation Routine
# ---------------------------------------------------------------------------
def run_pipeline():
    print("=" * 70)
    print("STARTING NEST IRRIGATION PREDICTION PIPELINE")
    print("=" * 70)

    base_dir = Path("data/Soil Moisture, Irrigation Actuator and Weather Dat/Soil Moisture, Irrigation Actuator and Weather Dat/02_processed_data/preprocessed")
    artifacts_dir = Path("backend/app/ml/models")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    zones = [1, 2, 4, 5]  # Zone 3 strictly excluded
    all_cols = FEATURE_COLS + STATIC_COLS

    zone_sequences = {}
    zone_targets = {}
    zone_metas = {}

    print("\n--- 1. Data Resampling & Sequence Generation ---")
    for z in zones:
        csv_file = base_dir / f"dataset_zone_{z}_preprocessed.csv"
        res_30m, valid_mask = load_and_resample_zone(str(csv_file), z)
        X_z, y_z, meta_z = build_contiguous_sequences(res_30m, valid_mask, z)
        meta_z["split"] = split_zone_chronological(meta_z)

        # Verification of target_mean_24h alignment
        forward_48 = pd.api.indexers.FixedForwardWindowIndexer(window_size=48)
        calc_30m_mean = res_30m["soil_moisture"].rolling(window=forward_48, min_periods=35).mean()
        corr = res_30m["target_mean_24h"].corr(calc_30m_mean)
        mae_align = (res_30m["target_mean_24h"] - calc_30m_mean).dropna().abs().mean()
        print(f"Zone {z} ({CROP_SPECS[z]['name']}):")
        print(f"  Target alignment correlation: {corr:.5f}, MAE diff: {mae_align:.4f}%")
        print(f"  Total valid 48-step sequences: {len(X_z):,}")
        counts = meta_z["split"].value_counts()
        print(f"  Split counts: Train={counts.get('train', 0)}, Val={counts.get('val', 0)}, Test={counts.get('test', 0)}, Buffer={counts.get('buffer', 0)}")

        zone_sequences[z] = X_z
        zone_targets[z] = y_z
        zone_metas[z] = meta_z

    # Combine into pooled splits
    X_train_list, y_train_list, meta_train_list = [], [], []
    X_val_list, y_val_list, meta_val_list = [], [], []
    X_test_list, y_test_list, meta_test_list = [], [], []

    for z in zones:
        meta_z = zone_metas[z]
        X_z = zone_sequences[z]
        y_z = zone_targets[z]

        train_idx = np.where(meta_z["split"] == "train")[0]
        val_idx = np.where(meta_z["split"] == "val")[0]
        test_idx = np.where(meta_z["split"] == "test")[0]

        X_train_list.append(X_z[train_idx])
        y_train_list.append(y_z[train_idx])
        meta_train_list.append(meta_z.iloc[train_idx])

        X_val_list.append(X_z[val_idx])
        y_val_list.append(y_z[val_idx])
        meta_val_list.append(meta_z.iloc[val_idx])

        X_test_list.append(X_z[test_idx])
        y_test_list.append(y_z[test_idx])
        meta_test_list.append(meta_z.iloc[test_idx])

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    meta_train = pd.concat(meta_train_list, ignore_index=True)

    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    meta_val = pd.concat(meta_val_list, ignore_index=True)

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    meta_test = pd.concat(meta_test_list, ignore_index=True)

    print("\n--- 2. Dataset Split Summary ---")
    print(f"Total Training Sequences:   {len(X_train):,}")
    print(f"Total Validation Sequences: {len(X_val):,}")
    print(f"Total Test Sequences:       {len(X_test):,}")

    # Standard Scaler (fitted ONLY on Training Set)
    print("\n--- 3. Feature Normalization ---")
    N_tr, T, F = X_train.shape
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, F)
    scaler.fit(X_train_2d)

    X_train_norm = scaler.transform(X_train_2d).reshape(N_tr, T, F).astype(np.float32)
    X_val_norm = scaler.transform(X_val.reshape(-1, F)).reshape(len(X_val), T, F).astype(np.float32)
    X_test_norm = scaler.transform(X_test.reshape(-1, F)).reshape(len(X_test), T, F).astype(np.float32)

    # Save Scaler
    scaler_path = artifacts_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved feature scaler to: {scaler_path}")

    # -----------------------------------------------------------------------
    # 4. XGBoost Training
    # -----------------------------------------------------------------------
    print("\n--- 4. Training XGBoost Model ---")
    X_train_tab = extract_tabular_features(X_train_norm, all_cols)
    X_val_tab = extract_tabular_features(X_val_norm, all_cols)
    X_test_tab = extract_tabular_features(X_test_norm, all_cols)

    xgb_model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        early_stopping_rounds=25,
        random_state=42,
        n_jobs=-1,
    )

    xgb_model.fit(
        X_train_tab,
        y_train,
        eval_set=[(X_val_tab, y_val)],
        verbose=50,
    )

    xgb_test_preds = xgb_model.predict(X_test_tab)
    xgb_val_preds = xgb_model.predict(X_val_tab)

    xgb_path = artifacts_dir / "xgb_model.joblib"
    joblib.dump(xgb_model, xgb_path)
    print(f"Saved XGBoost model to: {xgb_path}")

    # -----------------------------------------------------------------------
    # 5. GRU Neural Network Training
    # -----------------------------------------------------------------------
    print("\n--- 5. Training GRU Model ---")
    device = torch.device("cpu")
    train_dataset = TensorDataset(torch.from_numpy(X_train_norm), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val_norm), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test_norm), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    gru_model = SoilMoistureGRU(input_dim=F, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(gru_model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_weights = None
    patience = 8
    patience_counter = 0

    for epoch in range(1, 36):
        gru_model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = gru_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gru_model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
        train_loss /= len(train_dataset)

        gru_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = gru_model(batch_x)
                val_loss += criterion(preds, batch_y).item() * len(batch_y)
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = gru_model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    gru_model.load_state_dict(best_weights)
    gru_path = artifacts_dir / "gru_model.pt"
    torch.save(gru_model.state_dict(), gru_path)
    print(f"Saved GRU model weights to: {gru_path}")

    gru_model.eval()
    with torch.no_grad():
        gru_test_preds = gru_model(torch.from_numpy(X_test_norm)).numpy()
        gru_val_preds = gru_model(torch.from_numpy(X_val_norm)).numpy()

    # -----------------------------------------------------------------------
    # 6. Evaluation & Agronomic Class Agreement
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (TEST SET)")
    print("=" * 70)

    results = {
        "overall": {
            "xgboost": {
                "val_mae": float(mean_absolute_error(y_val, xgb_val_preds)),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, xgb_val_preds))),
                "test_mae": float(mean_absolute_error(y_test, xgb_test_preds)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, xgb_test_preds))),
            },
            "gru": {
                "val_mae": float(mean_absolute_error(y_val, gru_val_preds)),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, gru_val_preds))),
                "test_mae": float(mean_absolute_error(y_test, gru_test_preds)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, gru_test_preds))),
            },
        },
        "per_zone": {},
    }

    print("\nOVERALL TEST METRICS:")
    print(f"  XGBoost: MAE = {results['overall']['xgboost']['test_mae']:.3f}%, RMSE = {results['overall']['xgboost']['test_rmse']:.3f}%")
    print(f"  GRU:     MAE = {results['overall']['gru']['test_mae']:.3f}%, RMSE = {results['overall']['gru']['test_rmse']:.3f}%")

    print("\nPER-ZONE BREAKDOWN (TEST SET):")
    print("-" * 70)
    for z in zones:
        z_mask = (meta_test["zone"] == z).values
        y_true_z = y_test[z_mask]
        xgb_z = xgb_test_preds[z_mask]
        gru_z = gru_test_preds[z_mask]
        meta_sub = meta_test.iloc[z_mask]

        # Calculate Agronomic Class Agreement
        true_classes = [
            classify_agronomic_state(yt, row["pwp"], row["madp"], row["fc"], row["sp"])
            for yt, (_, row) in zip(y_true_z, meta_sub.iterrows())
        ]
        xgb_classes = [
            classify_agronomic_state(yp, row["pwp"], row["madp"], row["fc"], row["sp"])
            for yp, (_, row) in zip(xgb_z, meta_sub.iterrows())
        ]
        gru_classes = [
            classify_agronomic_state(yp, row["pwp"], row["madp"], row["fc"], row["sp"])
            for yp, (_, row) in zip(gru_z, meta_sub.iterrows())
        ]

        xgb_acc = float(np.mean(np.array(true_classes) == np.array(xgb_classes))) * 100
        gru_acc = float(np.mean(np.array(true_classes) == np.array(gru_classes))) * 100

        z_res = {
            "name": CROP_SPECS[z]["name"],
            "test_samples": int(np.sum(z_mask)),
            "xgboost": {
                "mae": float(mean_absolute_error(y_true_z, xgb_z)),
                "rmse": float(np.sqrt(mean_squared_error(y_true_z, xgb_z))),
                "agronomic_class_acc": xgb_acc,
            },
            "gru": {
                "mae": float(mean_absolute_error(y_true_z, gru_z)),
                "rmse": float(np.sqrt(mean_squared_error(y_true_z, gru_z))),
                "agronomic_class_acc": gru_acc,
            },
        }
        results["per_zone"][z] = z_res

        print(f"Zone {z} - {CROP_SPECS[z]['name']} (N={z_res['test_samples']}):")
        print(f"  XGBoost -> MAE: {z_res['xgboost']['mae']:.3f}%, RMSE: {z_res['xgboost']['rmse']:.3f}%, Agronomic State Acc: {xgb_acc:.1f}%")
        print(f"  GRU     -> MAE: {z_res['gru']['mae']:.3f}%, RMSE: {z_res['gru']['rmse']:.3f}%, Agronomic State Acc: {gru_acc:.1f}%")

    # Save results and metadata
    results_path = artifacts_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation metrics to: {results_path}")

    metadata = {
        "feature_cols": FEATURE_COLS,
        "static_cols": STATIC_COLS,
        "sequence_length": SEQUENCE_LENGTH,
        "sampling_minutes": 30,
        "boundary_gap_hours": BOUNDARY_GAP_HOURS,
        "target_col": "target_mean_24h",
        "zones_included": zones,
        "zones_excluded": [3],
        "crop_specs": CROP_SPECS,
    }
    meta_path = artifacts_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {meta_path}")

    print("\nPipeline execution completed successfully.")
    return results


if __name__ == "__main__":
    run_pipeline()
