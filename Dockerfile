# Use a lightweight Python base image
FROM python:3.10-slim

# Set workdir inside container
WORKDIR /app

# Install system dependencies (needed for numpy/pandas/scikit-learn/xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Make start.sh executable
RUN chmod +x start.sh

# Expose port
EXPOSE 8080

# Run the start script
CMD ["./start.sh"]
