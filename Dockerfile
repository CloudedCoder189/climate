FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict_api.py .
COPY climate_model_regularized.pkl .
COPY features_regularized.pkl .
COPY climate_merged.csv .

EXPOSE 8080

CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "8080"]
