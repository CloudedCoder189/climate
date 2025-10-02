import os
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

model = xgb.XGBRegressor()
model.load_model("climate_model.json")  

df = pd.read_csv(r"climate_merged.csv", parse_dates=["date"])
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
    pred = model.predict(pd.DataFrame(X, columns=["sst","co2","land_temp","precip","tas"]))[0]

    alpha = 0.7
    calibrated = alpha * pred + (1 - alpha) * last_tas

    return {
        "last_tas": float(last_tas),
        "raw_predicted_tas": float(pred),
        "calibrated_predicted_tas": float(calibrated)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
