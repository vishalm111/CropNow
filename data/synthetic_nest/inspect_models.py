import sys, pathlib
sys.path.insert(0, str(pathlib.Path("backend")))

import joblib, torch

print("=== XGB model n_features ===")
xgb = joblib.load("backend/app/ml/models/xgb_model.joblib")
print(f"  xgb n_features_in_ : {xgb.n_features_in_}")
xgb_res = joblib.load("backend/app/ml/models_residual/xgb_residual_model.joblib")
print(f"  xgb_res n_features  : {xgb_res.n_features_in_}")

print()
print("=== GRU state dict keys ===")
gru_state = torch.load("backend/app/ml/models/gru_model.pt", map_location="cpu", weights_only=True)
print("  gru keys:", list(gru_state.keys()))
gru_res_state = torch.load("backend/app/ml/models_residual/gru_residual_model.pt", map_location="cpu", weights_only=True)
print("  gru_res keys:", list(gru_res_state.keys()))

print()
print("=== GRU layer shapes ===")
for k, v in gru_state.items():
    print(f"  {k}: {tuple(v.shape)}")
