import pandas as pd
import json
from pathlib import Path

out = Path("data/synthetic_nest")
ref = json.load(open("data/company_sample/nest_sample.json"))
ref_keys = set(ref.keys())

print("=== COLUMN CHECK ===")
for f in sorted(out.glob("scenario_*.csv")):
    df = pd.read_csv(f)
    orig_cols = set(df.columns) & ref_keys
    missing   = ref_keys - orig_cols
    print(f"{f.name}: {len(df.columns)} cols | orig fields preserved={len(orig_cols)}/21 | missing={missing or 'none'}")

print()
print("=== TIMESTAMP RANGE CHECK ===")
for f in sorted(out.glob("scenario_*.csv")):
    df = pd.read_csv(f)
    ts = pd.to_datetime(df["timestamp"])
    intervals = ts.diff().dropna().dt.total_seconds() / 60
    uniq = intervals.unique().tolist()
    print(f"{f.name}: {ts.iloc[0]} -> {ts.iloc[-1]} | 30-min intervals only={all(v==30.0 for v in uniq)}")

print()
print("=== GT LABEL DISTRIBUTION ===")
for f in sorted(out.glob("scenario_*.csv")):
    df = pd.read_csv(f)
    print(f"--- {f.name} ---")
    grp = df.groupby(["gt_label","gt_irrigate","gt_soil_stress","gt_sensor_fault","data_source"]).size()
    print(grp.to_string())
    print()

print("=== IRRIGATION EVENT: moisture trace (every 4 packets) ===")
df = pd.read_csv(out / "scenario_04_irrigation_event.csv")
sub = df[["packet_index","timestamp","soil_moisture_1","soil_moisture_2","gt_label"]].iloc[::4]
print(sub.to_string(index=False))

print()
print("=== SENSOR NOISE: spike packets ===")
df2 = pd.read_csv(out / "scenario_02_sensor_noise.csv")
spikes = df2[df2["gt_sensor_fault"] == 1][["packet_index","soil_moisture_1","soil_moisture_2","temperature","humidity","gt_label"]]
print(spikes.to_string(index=False))

print()
print("=== DRYING SCENARIO: last 10 packets ===")
df3 = pd.read_csv(out / "scenario_03_gradual_drying.csv")
sub3 = df3[["packet_index","soil_moisture_1","soil_moisture_2","gt_label","gt_irrigate"]].tail(10)
print(sub3.to_string(index=False))

print()
print("=== ALL data_source values (must be SYNTHETIC) ===")
for f in sorted(out.glob("scenario_*.csv")):
    df = pd.read_csv(f)
    vals = df["data_source"].unique().tolist()
    print(f"  {f.name}: {vals}")
