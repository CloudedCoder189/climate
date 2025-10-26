from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from itertools import combinations
from datetime import datetime
import uvicorn
from xgboost import XGBRegressor

# === Paths ===
MODEL_PATH  = r"C:\Users\nirmi\cliamtenw\output\climate_model_advanced.json"
SCALER_PATH = r"C:\Users\nirmi\cliamtenw\output\driver_scaler_clean.pkl"
DATA_PATH   = r"C:\Users\nirmi\cliamtenw\output\climate_cleaned.csv"

print("📦 Loading model and scaler...")
model = XGBRegressor()
model.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Model and scaler loaded successfully.")

# === Feature helpers ===
def create_lags(df, col, lags):
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def create_rolls(df, col, windows):
    for w in windows:
        df[f"{col}_roll{w}"] = df[col].rolling(window=w).mean()
    return df

def add_interactions(df, cols):
    for c1, c2 in combinations(cols, 2):
        df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
    return df


# === FastAPI App ===
app = FastAPI(
    title="🌎 Climate Prediction API",
    description="Predicts global temperature anomaly (°C) from CO₂, SST, Precipitation, and TAS inputs.",
    version="3.1"
)

class ClimateInput(BaseModel):
    co2: float
    sst: float
    precip: float
    tas: float


@app.get("/")
def root():
    return {
        "message": "🌎 Climate Prediction API is live!",
        "usage": "POST /predict with co2, sst, precip, tas"
    }


@app.post("/predict")
def predict(inputs: ClimateInput):
    try:
        # --- Load historical data for lag/roll context ---
        hist = pd.read_csv(DATA_PATH, parse_dates=["date"])
        hist = hist.dropna()  # clean history
        new = pd.DataFrame([{
            "date": datetime.now(),
            "co2": inputs.co2,
            "sst": inputs.sst,
            "precip": inputs.precip,
            "tas": inputs.tas
        }])

        # Merge last 36 months + new sample
        df = pd.concat([hist.tail(36), new], ignore_index=True)

        # --- Rebuild features (same as training) ---
        lags = [1, 3, 6, 12, 24, 36]
        windows = [3, 6, 12, 24]
        for col in ["co2", "temperature_anomaly", "precip", "sst", "tas"]:
            df = create_lags(df, col, lags)
            df = create_rolls(df, col, windows)
        df = add_interactions(df, ["co2", "temperature_anomaly", "precip", "sst", "tas"])

        # Fill missing lag/roll values with last known ones (keep new row)
        df = df.fillna(method="ffill")

        # --- Prepare final row for prediction ---
        X_new = df.drop(columns=["temperature_anomaly", "date"]).iloc[-1:]
        X_scaled = pd.DataFrame(scaler.transform(X_new), columns=scaler.feature_names_in_)

        # --- Predict ---
        pred = model.predict(X_scaled)[0]

        return {
            "predicted_temperature_anomaly": round(float(pred), 4),
            "units": "°C",
            "model_version": "5-dataset (CO₂, SST, Precip, TAS)",
            "model_rmse": "0.072 °C",
            "r2": "0.85"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
