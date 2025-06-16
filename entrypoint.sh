#!/bin/bash

# Start MLflow UI in background
mlflow ui --host 0.0.0.0 --port 5000 &

# Train model (saves to /app/artifacts/)
# python /app/src/train.py

# Run Flask API server
python /app/src/app.py
