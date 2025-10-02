import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model and features
model = joblib.load("climate_model_regularized.pkl")
features = joblib.load("features_regularized.pkl")

# Load dataset for last_tas
df = pd.read_csv("climate_merged.csv")
last_tas = df["tas"].iloc[-1]

app = FastAPI()

class Inputs(BaseModel):
    sst: float
    co2: float
    land_temp: float
    precip: float

@app.get("/")
def home():
    return {"status": "ok", "message": "Climate API is running 🚀"}

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
