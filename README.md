# Climate Prediction API

A FastAPI service built around an XGBoost regression model for predicting global temperature anomaly from climate-driver inputs.

## Inputs

The API accepts four current climate variables:

- `co2` — atmospheric CO₂
- `sst` — sea-surface temperature
- `precip` — precipitation
- `tas` — near-surface air temperature

The trained model also depends on lagged, rolling, and interaction features built from historical climate data.

## Important data requirement

The trained model and scaler are included in this repository, but the historical context file `climate_cleaned.csv` is **not currently included**. That file is required to reconstruct the same lag and rolling features used by the model.

The API now handles this explicitly:

- `GET /health` reports whether the service is ready for predictions.
- `POST /predict` returns HTTP `503` when the historical context file is unavailable.
- `climate_cleaned.example.csv` documents the expected column schema.

Required columns:

```text
date, co2, temperature_anomaly, precip, sst, tas
```

At least 36 complete historical rows are required because the longest model lag is 36 periods.

## Model

- Algorithm: XGBoost regression
- Reported RMSE: `0.072 °C`
- Reported R²: `0.85`
- Feature engineering: lag features, rolling means, and pairwise interactions

The reported metrics are values associated with the existing trained artifact; this repository does not currently include the original training/evaluation pipeline needed to reproduce them from scratch.

## Project structure

```text
.
├── api.py
├── climate_model_advanced.json
├── driver_scaler.pkl
├── climate_cleaned.example.csv
├── requirements.txt
├── Procfile
└── README.md
```

When available, place the real historical file at:

```text
climate_cleaned.csv
```

in the repository root.

## Run locally

```bash
git clone https://github.com/CloudedCoder189/climate.git
cd climate
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn api:app --reload
```

Interactive API documentation is available at `/docs`.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"co2": 420.0, "sst": 20.1, "precip": 2.7, "tas": 14.9}'
```

A prediction is returned only when the required historical context has loaded successfully.

## Tech stack

- Python
- FastAPI
- pandas
- XGBoost
- scikit-learn
- joblib
- Uvicorn
