from itertools import combinations
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "climate_model_advanced.json"
SCALER_PATH = BASE_DIR / "driver_scaler.pkl"
DATA_PATH = BASE_DIR / "climate_cleaned.csv"

LAGS = (1, 3, 6, 12, 24, 36)
ROLLING_WINDOWS = (3, 6, 12, 24)
DRIVER_COLUMNS = ("co2", "temperature_anomaly", "precip", "sst", "tas")
REQUIRED_HISTORY_COLUMNS = {"date", *DRIVER_COLUMNS}

MODEL_VERSION = "5-dataset (CO₂, SST, Precip, TAS)"
MODEL_RMSE = "0.072 °C"
MODEL_R2 = "0.85"


def create_lags(dataframe: pd.DataFrame, column: str) -> None:
    for lag in LAGS:
        dataframe[f"{column}_lag{lag}"] = dataframe[column].shift(lag)


def create_rolling_means(dataframe: pd.DataFrame, column: str) -> None:
    for window in ROLLING_WINDOWS:
        dataframe[f"{column}_roll{window}"] = dataframe[column].rolling(window=window).mean()


def add_interactions(dataframe: pd.DataFrame) -> None:
    for first, second in combinations(DRIVER_COLUMNS, 2):
        dataframe[f"{first}_x_{second}"] = dataframe[first] * dataframe[second]


def load_model() -> XGBRegressor:
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model file not found: {MODEL_PATH.name}")

    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def load_scaler():
    if not SCALER_PATH.is_file():
        raise RuntimeError(f"Scaler file not found: {SCALER_PATH.name}")
    return joblib.load(SCALER_PATH)


def load_history() -> tuple[pd.DataFrame | None, str | None]:
    """Load historical context without making the whole API crash if it is absent."""
    if not DATA_PATH.is_file():
        return None, f"Required historical context file is missing: {DATA_PATH.name}"

    try:
        history = pd.read_csv(DATA_PATH, parse_dates=["date"])
    except Exception as exc:
        return None, f"Could not read {DATA_PATH.name}: {exc}"

    missing_columns = REQUIRED_HISTORY_COLUMNS.difference(history.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        return None, f"Historical data is missing required columns: {missing}"

    history = history.dropna(subset=list(REQUIRED_HISTORY_COLUMNS)).copy()
    if len(history) < max(LAGS):
        return None, f"Historical data needs at least {max(LAGS)} complete rows."

    return history, None


model = load_model()
scaler = load_scaler()
history, history_error = load_history()

app = FastAPI(
    title="Climate Prediction API",
    description=(
        "Predicts global temperature anomaly from CO₂, sea-surface temperature, "
        "precipitation, and near-surface air temperature inputs."
    ),
    version="3.2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ClimateInput(BaseModel):
    co2: float
    sst: float
    precip: float
    tas: float


@app.get("/")
def root():
    return {
        "message": "Climate Prediction API",
        "ready": history is not None,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "ready": history is not None,
        "model_loaded": True,
        "scaler_loaded": True,
        "historical_data_loaded": history is not None,
        "historical_data_error": history_error,
    }


@app.post("/predict")
def predict(inputs: ClimateInput):
    if history is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=history_error or "Historical data is unavailable.",
        )

    try:
        new_row = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp.now(),
                    "co2": inputs.co2,
                    "temperature_anomaly": float("nan"),
                    "precip": inputs.precip,
                    "sst": inputs.sst,
                    "tas": inputs.tas,
                }
            ]
        )

        features_frame = pd.concat([history.tail(max(LAGS)), new_row], ignore_index=True)

        for column in DRIVER_COLUMNS:
            create_lags(features_frame, column)
            create_rolling_means(features_frame, column)
        add_interactions(features_frame)

        features_frame = features_frame.ffill()
        prediction_row = features_frame.drop(columns=["temperature_anomaly", "date"]).iloc[-1:]

        expected_features = list(scaler.feature_names_in_)
        missing_features = set(expected_features).difference(prediction_row.columns)
        if missing_features:
            missing = ", ".join(sorted(missing_features))
            raise RuntimeError(f"Prediction features do not match the scaler. Missing: {missing}")

        prediction_row = prediction_row.reindex(columns=expected_features)
        scaled_features = scaler.transform(prediction_row)
        prediction = model.predict(scaled_features)[0]

        return {
            "predicted_temperature_anomaly": round(float(prediction), 4),
            "units": "°C",
            "model_version": MODEL_VERSION,
            "model_rmse": MODEL_RMSE,
            "r2": MODEL_R2,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed because the model inputs could not be prepared.",
        ) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
