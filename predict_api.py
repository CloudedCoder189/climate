from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import xgboost as xgb
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS so frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # replace "*" with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load XGBoost model from JSON
model = xgb.XGBRegressor()
model.load_model("climate_model.json")

@app.get("/")
def read_root():
    return {"message": "🌍 Climate Prediction API is live! Use POST /predict to get predictions."}

@app.post("/predict")
def predict(data: dict):
    try:
        # Extract input features
        sst = float(data["sst"])
        land_temp = float(data["land_temp"])
        co2 = float(data["co2"])
        precip = float(data["precip"])
        tas = float(data.get("tas", 14.0))  # last month TAS (default ~14.0°C)

        # Create feature vector for prediction
        X = np.array([[sst, land_temp, co2, precip, tas]])

        # Run prediction
        raw_pred = model.predict(X)[0]

        # Simple calibration (example)
        calibrated = (raw_pred + tas) / 2  

        return {
            "last_tas": tas,
            "raw_predicted_tas": float(raw_pred),
            "calibrated_predicted_tas": float(calibrated)
        }
    except Exception as e:
        return {"error": str(e)}
