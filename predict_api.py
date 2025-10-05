import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------
app = FastAPI(title="🌍 Climate Prediction API (Physical Model)",
              description="Predicts near-surface air temperature based on physical climate drivers.",
              version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Paths and model loading
# ---------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "climate_model_physical12.json")
SCALER_PATH = os.getenv("SCALER_PATH", "driver_scaler.pkl")
DATA_PATH = os.getenv("DATA_PATH", "climate_merged.csv")
BASELINE_PATH = os.getenv("BASELINE_PATH", "baseline.txt")
BASELINE_MODE = os.getenv("BASELINE_MODE", "file")  # file | first120 | fullmean

# Load model and scaler
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load dataset for lag/rolling context
df = pd.read_csv(DATA_PATH).dropna().reset_index(drop=True)

# ---------------------------------------------------------
# Baseline handling
# ---------------------------------------------------------
with open(BASELINE_PATH) as f:
    baseline_file = float(f.read().strip())

def compute_baseline():
    """Compute baseline dynamically or from file."""
    mode = BASELINE_MODE.lower()
    if mode == "first120" and len(df) >= 120:
        return float(df["tas"].iloc[:120].mean())
    elif mode == "fullmean":
        return float(df["tas"].mean())
    return baseline_file

# ---------------------------------------------------------
# Lag and rolling context
# ---------------------------------------------------------
def latest_context():
    """Retrieve recent TAS context with same lag/rolling and dampening as training."""
    last = df["tas"].iloc[-1]
    lag3 = df["tas"].iloc[-3]
    lag6 = df["tas"].iloc[-6]
    lag12 = df["tas"].iloc[-12]
    roll3 = float(df["tas"].iloc[-3:].mean())
    roll12 = float(df["tas"].iloc[-12:].mean())

    # Apply 0.25 dampening from training
    lag3 *= 0.25
    lag6 *= 0.25
    lag12 *= 0.25
    roll3 *= 0.25
    roll12 *= 0.25

    return lag3, lag6, lag12, roll3, roll12

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"message": "🌍 Climate Prediction API is live!", "usage": "POST /predict with sst, co2, land_temp, precip"}

@app.post("/predict")
def predict(data: dict, debug: int = Query(0, description="Set 1 for full diagnostics")):
    """
    Predict global near-surface air temperature based on drivers.
    Expects a JSON body with sst, co2, land_temp, precip.
    """
    try:
        # 1️⃣ Extract inputs
        sst = float(data["sst"])
        co2 = float(data["co2"])
        land_temp = float(data["land_temp"])
        precip = float(data["precip"])

        # 2️⃣ Scale the 4 main drivers
        scaled = scaler.transform([[sst, co2, land_temp, precip]])[0]
        sst_s, co2_s, land_s, precip_s = scaled.tolist()

        # 3️⃣ Get recent TAS context (lags & rolling averages)
        lag3, lag6, lag12, roll3, roll12 = latest_context()

        # 4️⃣ Combine features (must match training order)
        X = np.array([[sst_s, co2_s, land_s, precip_s, lag3, lag6, lag12, roll3, roll12]])

        # 5️⃣ Predict anomaly & add baseline
        anomaly_pred = float(model.predict(X)[0])
        baseline = compute_baseline()
        SENSITIVITY = 3.0  # try 2–4 range
        predicted_tas = baseline + anomaly_pred * SENSITIVITY


        # 6️⃣ Output
        if debug:
            return {
                "predicted_tas": round(predicted_tas, 3),
                "baseline_used": baseline,
                "raw_inputs": {"sst": sst, "co2": co2, "land_temp": land_temp, "precip": precip},
                "scaled_inputs": {"sst_s": sst_s, "co2_s": co2_s, "land_temp_s": land_s, "precip_s": precip_s},
                "lags_rolls": {
                    "tas_lag3": lag3, "tas_lag6": lag6, "tas_lag12": lag12,
                    "tas_roll3": roll3, "tas_roll12": roll12
                },
                "anomaly_pred": anomaly_pred,
                "baseline_mode": BASELINE_MODE,
            }

        return {"predicted_tas": round(predicted_tas, 3)}

    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# Run directly (for local testing)
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predict_api:app", host="0.0.0.0", port=8000, reload=True)

