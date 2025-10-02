import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load model + features
model = joblib.load("climate_model_regularized.pkl")
features = joblib.load("features_regularized.pkl")

df = pd.read_csv("climate_merged.csv", parse_dates=["date"])
last_tas = df["tas"].iloc[-1]

app = FastAPI()

class Inputs(BaseModel):
    sst: float
    co2: float
    land_temp: float
    precip: float

@app.post("/predict")
def predict(data: Inputs):
    X = [[
        data.sst,
        data.co2,
        data.land_temp,
        data.precip,
        last_tas
    ]]
    pred = model.predict(X)[0]

    alpha = 0.7
    calibrated = alpha * pred + (1 - alpha) * last_tas

    return {
        "last_tas": float(last_tas),
        "raw_predicted_tas": float(pred),
        "calibrated_predicted_tas": float(calibrated)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # 👈 Use Render’s PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
