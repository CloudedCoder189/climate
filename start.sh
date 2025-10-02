#!/bin/bash
# Start the FastAPI app on port 8080
exec uvicorn predict_api:app --host 0.0.0.0 --port 8080
