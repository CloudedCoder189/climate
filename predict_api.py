from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os

# Initialize FastAPI app
app = FastAPI(title="Climate Prediction API")

# -------------------------
# Load Model
# -------------------------
MODEL_PATH = "climate_model.json"

# Load XGBoost model from JSON
import xgboost as xgb
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

# -------------------------
# Request Schema
# -------------------------
class ClimateInput(BaseModel):
    sst: float        # Sea Surface Temperature anomaly
    land_temp: float  # Land temperature anomaly
    co2: float        # CO2 levels (ppm)
    precip: float     # Precipitation
    tas: float        # Last known near-surface air temperature

# -------------------------
# Root Route
# -------------------------
@app.get("/")
def home():
    return {"message": "🌎 Climate Prediction API is live! Use POST /predict to get predictions."}

# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict")
def predict(input_data: ClimateInput):
    try:
        # Convert input into model-ready format
        features = np.array([[input_data.sst, input_data.land_temp, input_data.co2, input_data.precip, input_data.tas]])

        # Predict raw temperature anomaly
        raw_pred = model.predict(features)[0]

        # Optionally apply calibration
        calibrated_pred = (raw_pred + input_data.tas) / 2

        return {
            "last_tas": input_data.tas,
            "raw_predicted_tas": float(raw_pred),
            "calibrated_predicted_tas": float(calibrated_pred)
        }
    except Exception as e:
        return {"error": str(e)}
