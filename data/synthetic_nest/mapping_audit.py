"""
Audit helper: inspect scaler fitted statistics and actual training ranges
for all features used in both model families.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path("backend")))

import joblib
import numpy as np
import pandas as pd

ABS_ALL_COLS = [
    "weather_temp","weather_humidity","weather_rain","weather_pressure",
    "weather_wind_speed","weather_radiation",
    "soil_temperature_0-7cm","soil_temperature_7-18cm",
    "ec","ph",
    "soil_moisture",
    "irrigation_duration_minutes","water_vol_past_4h",
    "flow_rate","pwp","madp","fc","sp",
]
RES_ALL_COLS = [
    "weather_temp","weather_humidity","weather_rain","weather_pressure",
    "weather_wind_speed","weather_radiation",
    "soil_temperature_0-7cm","soil_temperature_7-18cm",
    "ec","ph",
    "soil_moisture",
    "irrigation_duration_minutes","water_vol_past_4h",
    "relative_moisture","dist_to_madp","dist_to_fc",
    "flow_rate","pwp","madp","fc","sp",
]

print("="*70)
print("ABSOLUTE MODEL SCALER — fitted mean / std per feature")
print("="*70)
sc_abs = joblib.load("backend/app/ml/models/scaler.joblib")
for col, mean, std in zip(ABS_ALL_COLS, sc_abs.mean_, sc_abs.scale_):
    print(f"  {col:<35s}  mean={mean:>10.4f}  std={std:>10.4f}")

print()
print("="*70)
print("RESIDUAL MODEL SCALER — fitted mean / std per feature")
print("="*70)
sc_res = joblib.load("backend/app/ml/models_residual/scaler_residual.joblib")
for col, mean, std in zip(RES_ALL_COLS, sc_res.mean_, sc_res.scale_):
    print(f"  {col:<35s}  mean={mean:>10.4f}  std={std:>10.4f}")

print()
print("="*70)
print("ACTUAL PREPROCESSED DATA RANGES (Zone 1 sample)")
print("="*70)
zone1 = pd.read_csv(
    "data/Soil Moisture, Irrigation Actuator and Weather Dat/"
    "Soil Moisture, Irrigation Actuator and Weather Dat/"
    "02_processed_data/preprocessed/dataset_zone_1_preprocessed.csv"
)
check_cols = [
    "weather_temp","weather_humidity","weather_rain","weather_pressure",
    "weather_wind_speed","weather_radiation",
    "soil_temperature_0-7cm","soil_temperature_7-18cm",
    "ec","ph","soil_moisture",
    "irrigation_duration_minutes","water_vol_past_4h",
]
stats = zone1[check_cols].agg(["min","max","mean","std"]).T
print(stats.to_string())

print()
print("="*70)
print("NEST SAMPLE FIELD VALUES vs TRAINING MEANS")
print("="*70)
nest = {
    "temperature": 28, "humidity": 72, "rainfall": 0,
    "pressure": 100.8, "wind_speed": 1.4, "solar_radiation": 420,
    "soil_temperature": 24,
    "soil_moisture_1": 22, "soil_moisture_2": 31,
    "leaf_wetness": 35,
    "co": 0.04, "no2": 0.03, "o3": 0.04, "pm2_5": 48,
}
nest_sm_avg = (nest["soil_moisture_1"] + nest["soil_moisture_2"]) / 2.0
print(f"  NEST pressure      = {nest['pressure']} kPa")
print(f"  Training pressure  = {sc_abs.mean_[3]:.4f} ± {sc_abs.scale_[3]:.4f}")
print(f"  Ratio              = {nest['pressure'] / sc_abs.mean_[3]:.4f}  (expect ~1.0 if same units)")
print()
print(f"  NEST soil_moisture_avg = {nest_sm_avg:.1f} %")
print(f"  Training soil_moisture mean = {sc_abs.mean_[10]:.4f} ± {sc_abs.scale_[10]:.4f}")
print()
print(f"  NEST weather_rain  = {nest['rainfall']} (mm per packet)")
print(f"  Training rain mean = {sc_abs.mean_[2]:.6f} ± {sc_abs.scale_[2]:.6f}")
print(f"  Rain resampling    = SUM over 30-min window (3 x 10-min bins)")
print()
print(f"  NEST irrigation_duration_minutes = NOT IN NEST PAYLOAD")
print(f"  Training irr_duration mean = {sc_abs.mean_[11]:.6f} ± {sc_abs.scale_[11]:.6f}")
print(f"  Inference default          = 0.0")
print()
print(f"  NEST water_vol_past_4h = NOT IN NEST PAYLOAD")
print(f"  Training water_vol mean = {sc_abs.mean_[12]:.6f} ± {sc_abs.scale_[12]:.6f}")
print(f"  Inference default       = 0.0")
print()
print(f"  NEST ec (soil EC) = NOT IN NEST PAYLOAD")
print(f"  Training ec mean  = {sc_abs.mean_[8]:.4f} ± {sc_abs.scale_[8]:.4f}")
print(f"  Inference default = 6.5  (WRONG — ec is µS/cm, not pH units)")
print()
print(f"  NEST ph = NOT IN NEST PAYLOAD")
print(f"  Training ph mean  = {sc_abs.mean_[9]:.4f} ± {sc_abs.scale_[9]:.4f}")
print(f"  Inference default = 7.0  (plausible pH range)")
print()
print(f"  NEST soil_temperature (single probe)")
print(f"  Training 0-7cm mean  = {sc_abs.mean_[6]:.4f} ± {sc_abs.scale_[6]:.4f}")
print(f"  Training 7-18cm mean = {sc_abs.mean_[7]:.4f} ± {sc_abs.scale_[7]:.4f}")
print(f"  Inference mirrors 0-7cm → 7-18cm")

print()
print("="*70)
print("CROP THRESHOLD COMPARISON: NEST chilli vs Training zones")
print("="*70)
chilli = {"pwp": 10.0, "madp": 20.0, "fc": 35.0, "sp": 45.0, "flow_rate": 1.0}
training = {
    1: {"name":"Tomato","pwp":20,"madp":40,"fc":60,"sp":80,"flow_rate":1.1},
    2: {"name":"Tomato","pwp":20,"madp":40,"fc":60,"sp":80,"flow_rate":1.1},
    4: {"name":"Zucchini","pwp":40,"madp":80,"fc":90,"sp":95,"flow_rate":1.1},
    5: {"name":"Blueberry","pwp":35,"madp":70,"fc":90,"sp":95,"flow_rate":0.4},
}
print(f"  Chilli (inference): {chilli}")
for z,v in training.items():
    print(f"  Zone {z} {v['name']:10s} (training): pwp={v['pwp']}, madp={v['madp']}, fc={v['fc']}, sp={v['sp']}, flow={v['flow_rate']}")

print()
print(f"  Training static cols: mean PWP  = {sc_abs.mean_[14]:.2f} ± {sc_abs.scale_[14]:.2f}")
print(f"  Training static cols: mean MADP = {sc_abs.mean_[15]:.2f} ± {sc_abs.scale_[15]:.2f}")
print(f"  Training static cols: mean FC   = {sc_abs.mean_[16]:.2f} ± {sc_abs.scale_[16]:.2f}")
print(f"  Chilli MADP = 20.0 (vs training range 40–80, mean={sc_abs.mean_[15]:.1f})")

print()
print("="*70)
print("COLUMN ORDER CHECK — residual pipeline vs inference.py")
print("="*70)
# From residual_pipeline.py line 104-105
res_pipeline = [
    "weather_temp","weather_humidity","weather_rain","weather_pressure",
    "weather_wind_speed","weather_radiation",
    "soil_temperature_0-7cm","soil_temperature_7-18cm",
    "ec","ph","soil_moisture",
    "irrigation_duration_minutes","water_vol_past_4h",
    "relative_moisture","dist_to_madp","dist_to_fc",
    "flow_rate","pwp","madp","fc","sp",
]
match = res_pipeline == RES_ALL_COLS
print(f"  Residual col order matches inference.py: {match}")
if not match:
    for i,(a,b) in enumerate(zip(res_pipeline, RES_ALL_COLS)):
        if a != b:
            print(f"  MISMATCH at index {i}: pipeline='{a}' vs inference='{b}'")

# From pipeline.py lines 53-70
abs_pipeline = [
    "weather_temp","weather_humidity","weather_rain","weather_pressure",
    "weather_wind_speed","weather_radiation",
    "soil_temperature_0-7cm","soil_temperature_7-18cm",
    "ec","ph","soil_moisture",
    "irrigation_duration_minutes","water_vol_past_4h",
    "flow_rate","pwp","madp","fc","sp",
]
match2 = abs_pipeline == ABS_ALL_COLS
print(f"  Absolute col order matches inference.py:  {match2}")

print()
print("="*70)
print("RELATIVE MOISTURE FORMULA CHECK")
print("="*70)
# Training: (sm - PWP) / (FC - PWP)  [residual_pipeline.py line 94]
# Inference: (sm - MADP) / (FC - MADP)  [inference.py line 346]
print("  Training formula:  relative_moisture = (soil_moisture - PWP) / (FC - PWP)")
print("  Inference formula: relative_moisture = (sm_avg - MADP) / (FC - MADP)")
print("  MISMATCH: denominator uses MADP in inference, PWP in training!")
chilli_sm = 26.5
chilli_pwp = 10.0; chilli_madp = 20.0; chilli_fc = 35.0
train_rm = (chilli_sm - chilli_pwp) / (chilli_fc - chilli_pwp)
infer_rm = (chilli_sm - chilli_madp) / (chilli_fc - chilli_madp)
print(f"  Example SM=26.5, chilli: training_rm={train_rm:.4f}, inference_rm={infer_rm:.4f}")
