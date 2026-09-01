# Climate Prediction API

A machine-learning API for estimating global temperature anomaly from climate indicators including atmospheric CO₂, sea-surface temperature, precipitation, and near-surface air temperature.

## Overview

This project uses an XGBoost regression model exposed through a FastAPI backend. The API reconstructs the same feature pipeline used during training, including lagged values, rolling averages, and interaction features, before generating a temperature-anomaly prediction.

### Inputs

- CO₂ concentration
- Sea-surface temperature (SST)
- Precipitation
- Near-surface air temperature (TAS)

### Output

The `/predict` endpoint returns the estimated global temperature anomaly in °C along with model metadata.

## Model

The prediction pipeline uses:

- XGBoost regression
- Standardized numerical features
- Lag features at 1, 3, 6, 12, 24, and 36 months
- Rolling averages over 3, 6, 12, and 24 months
- Pairwise interaction features between climate variables

Current model metadata reported by the API:

- RMSE: 0.072 °C
- R²: 0.85

## Tech Stack

- Python
- FastAPI
- XGBoost
- pandas
- scikit-learn / joblib
- Uvicorn

## API Usage

Run the API locally:

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

Then send a POST request to `/predict` with JSON in this format:

```json
{
  "co2": 420.0,
  "sst": 0.6,
  "precip": 2.8,
  "tas": 1.1
}
```

Example response:

```json
{
  "predicted_temperature_anomaly": 1.2345,
  "units": "°C",
  "model_version": "5-dataset (CO₂, SST, Precip, TAS)",
  "model_rmse": "0.072 °C",
  "r2": "0.85"
}
```

## Project Structure

```text
climate/
├── api.py                       # FastAPI application and feature pipeline
├── climate_model_advanced.json  # Trained XGBoost model
├── driver_scaler.pkl            # Saved feature scaler
├── requirements.txt             # Python dependencies
└── Procfile                     # Deployment process definition
```

## Notes

The API requires the historical climate dataset used to reconstruct lag and rolling features at prediction time. The model and preprocessing artifacts are stored separately from the API code.

## Future Improvements

- Package the feature-engineering pipeline separately from the API layer
- Add automated tests for the prediction endpoint
- Add model evaluation plots and a reproducible training notebook
- Improve deployment and dataset handling
