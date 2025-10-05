import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import xgboost as xgb

# ======================
# FastAPI Setup
# ======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# Load model & baseline
# ======================
model = xgb.XGBRegressor()
model.load_model("climate_model_advanced.json")

with open("baseline.txt") as f:
    baseline = float(f.read().strip())

# ======================
# Load dataset once
# ======================
df = pd.read_csv("climate_merged.csv")
df = df.reset_index(drop=True)

# ======================
# Routes
# ======================
@app.get("/")
def root():
    return {"message": "🌍 Advanced Climate Prediction API is live! Use POST /predict"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Inputs from frontend
        sst = float(data["sst"])
        co2 = float(data["co2"])
        land_temp = float(data["land_temp"])
        precip = float(data["precip"])

        # --- Dynamically compute lag/rolling features ---
        last_tas = df["tas"].iloc[-1]
        tas_lag3 = df["tas"].iloc[-3]
        tas_lag6 = df["tas"].iloc[-6]
        tas_lag12 = df["tas"].iloc[-12]
        tas_roll3 = df["tas"].iloc[-3:].mean()
        tas_roll12 = df["tas"].iloc[-12:].mean()

        # Combine all features
        X = np.array([[
            sst, co2, land_temp, precip,
            last_tas, tas_lag3, tas_lag6, tas_lag12,
            tas_roll3, tas_roll12
        ]])

        # Predict anomaly
        anomaly_pred = model.predict(X)[0]

        # Convert back to absolute TAS
        predicted_tas = baseline + anomaly_pred

        # ✅ Simplify response for frontend
        return {"predicted_tas": float(predicted_tas)}

    except Exception as e:
        return {"error": str(e)}
