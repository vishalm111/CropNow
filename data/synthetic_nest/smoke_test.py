"""
Smoke test: validate imports, model loading, and strict 48-packet sequence validation.
Run from CropNow root: python data/synthetic_nest/smoke_test.py
"""
import sys
import json
import pathlib
import os
import pandas as pd

ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent  # CropNow/
BACKEND = ROOT / "backend"
os.chdir(str(BACKEND))                   # uvicorn expects cwd = backend/
sys.path.insert(0, str(BACKEND))

print("=== Import check ===")
try:
    from app.schemas.prediction import NESTPacket
    from app.services.inference import ModelRegistry, CROP_THRESHOLDS, InsufficientHistoryError
    from app.routes.prediction import make_irrigation_advice
except ImportError:
    from backend.app.schemas.prediction import NESTPacket
    from backend.app.services.inference import ModelRegistry, CROP_THRESHOLDS, InsufficientHistoryError
    from backend.app.routes.prediction import make_irrigation_advice

print("  schemas.prediction  : OK")
print("  services.inference  : OK")

print()
print("=== Model loading ===")
reg = ModelRegistry.get()
print(f"  Available models    : {reg.available_models}")

print()
print("=== 1. Rejection of single packet (no automatic padding) ===")
sample = json.load(open(ROOT / "data" / "company_sample" / "nest_sample.json"))
try:
    reg.predict([sample])
    assert False, "Should have raised InsufficientHistoryError"
except InsufficientHistoryError as e:
    print(f"  Single packet correctly rejected: '{e.error_code}' (packets={e.packets_provided}/48) — PASS")

print()
print("=== 2. Rejection of fabricated duplicate readings ===")
try:
    reg.predict([sample] * 48)
    assert False, "Should have raised InsufficientHistoryError"
except InsufficientHistoryError as e:
    print(f"  Duplicated readings correctly rejected: '{e}' — PASS")

print()
print("=== 3. 48 genuine consecutive packets (scenario_01_normal.csv) ===")
scenario_df = pd.read_csv(ROOT / "data" / "synthetic_nest" / "scenario_01_normal.csv")
genuine_48 = scenario_df.to_dict(orient="records")[:48]
assert len(genuine_48) == 48

for model_key in reg.available_models:
    result = reg.predict(genuine_48, model_key=model_key)
    print(f"  [{model_key:20s}]  current={result['current_soil_moisture']:.2f}%  "
          f"predicted={result['predicted_soil_moisture']:.2f}%  "
          f"delta={result['change_predicted']:+.3f}%  "
          f"warnings={len(result['warnings'])}")

print()
print("=== 4. Irrigation advice check ===")
for crop, thresh in CROP_THRESHOLDS.items():
    test_sm = thresh["madp"] - 2
    adv = make_irrigation_advice(test_sm, crop)
    assert adv.irrigate, f"Expected irrigate=True for {crop} at SM={test_sm}"
print("  All crops: irrigate=True when SM < MADP — PASS")

adv = make_irrigation_advice(50.0, "chilli")
assert not adv.irrigate
print("  Chilli at SM=50%: irrigate=False — PASS")

print()
print("All smoke tests PASSED.")
