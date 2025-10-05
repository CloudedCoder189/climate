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
# Load dataset for lags
# ======================
df = pd.read_csv(r"C:\Users\nirmi\Downloads\climate_change\backend\climate_merged.csv")
df = df.reset_index(drop=True)

# Compute lag & rolling features from the last known data
last_tas = df["tas"].iloc[-1]          # last month
tas_lag3 = df["tas"].iloc[-3]          # 3 months ago
tas_lag6 = df["tas"].iloc[-6]          # 6 months ago
tas_lag12 = df["tas"].iloc[-12]        # 12 months ago
tas_roll3 = df["tas"].iloc[-3:].mean() # last 3-month avg
tas_roll12 = df["tas"].iloc[-12:].mean() # last 12-month avg

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

        # Backend adds lag + rolling features automatically
        X = np.array([[
            sst, co2, land_temp, precip,
            last_tas, tas_lag3, tas_lag6, tas_lag12,
            tas_roll3, tas_roll12
        ]])

        # Predict anomaly
        anomaly_pred = model.predict(X)[0]

        # Convert back to absolute TAS
        predicted_tas = baseline + anomaly_pred

        return {
            "baseline": baseline,
            "last_tas": float(last_tas),
            "tas_lag3": float(tas_lag3),
            "tas_lag6": float(tas_lag6),
            "tas_lag12": float(tas_lag12),
            "tas_roll3": float(tas_roll3),
            "tas_roll12": float(tas_roll12),
            "predicted_anomaly": float(anomaly_pred),
            "predicted_tas": float(predicted_tas)
        }

    except Exception as e:
        return {"error": str(e)}
